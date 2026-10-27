"""Sandbox worker: ``python -m dainamo.engine.sandbox`` (one GPU, chosen by CUDA_VISIBLE_DEVICES).

Keeps a profile's models loaded between generations so trying prompts, LoRAs and settings costs
sampling time only. Commands arrive as JSON lines on stdin; events go out as JSON lines on stdout
(library output is redirected to stderr, the worker log):

  in:  {"cmd": "generate", "id", "out_dir", "profile": {id, family, data}, "params": {...}}
       {"cmd": "refmod", "id", "out_dir", "profile", "paths", "sheet", "name", "settings": {refmod_*}}
       {"cmd": "upscale", "id", "out_dir", "model", "inputs": [{path, kind}], "factor", "options"}
       {"cmd": "upscale_dataset", "id", "model", "paths", "resolution", "max_factor"}
       {"cmd": "cancel", "id"}   {"cmd": "unload"}   {"cmd": "shutdown"}
  out: ready | status {message} | started {id}
       | progress {id, done, total, message, image, images, step, steps, sec_per_step, eta, started}
         (an upscale's: done and total in frames, an image counting one, with input, inputs, eta and started;
         a dataset upscale's: done and total in images, with made so far)
       | output {id, item} | done {id, elapsed, loras; a RefMod's or a dataset upscale's made}
       | error {id, message} | cancelled {id} | idle | exit

Memory: the diffusion model and VAE stay on the GPU, apart from the share of the diffusion model's
blocks that the request's ``offload`` keeps in system RAM (changing it reloads the model, as does a request's
``model``: a finetuned diffusion model in place of the profile's). The text
encoder is loaded only to encode new prompts (encodings are cached), then parked in system RAM when
there's room for it, else unloaded. When it doesn't fit next to the diffusion model, the diffusion
model and VAE are parked in RAM while it runs. A request with ``low_vram`` streams the text encoder's
layers from system RAM instead, so only the rest of it comes to the GPU, and keeps the VAE in system RAM
while sampling (unless input images or reference audio need it), moving the diffusion model there while a render decodes if
the GPU hasn't room for both. Closing stdin (the server going away) shuts the worker down.

Upscaling (``engine.upscale``) keeps its upscaler loaded too, until another model is asked for. The profile's
models are parked in system RAM while it runs if the GPU hasn't room for both, and an upscaler that takes much
VRAM (SeedVR2) is unloaded when a generation or RefMod comes next.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict

EVENTS = None
# prompt encodings kept, the least recently used dropped beyond it. They wait in system RAM in bf16: LTX 2.5's are
# 12 MB each, Qwen Image's with reference images tens of MB. A checkpoint grid or strength sweep reuses one prompt
# (and its negative), so this is for going back and forth between prompts, at up to a GB or so.
TEXTS_KEPT = 32


def emit(event: str, **kw) -> None:
    EVENTS.write(json.dumps({"event": event, **kw}) + "\n")
    EVENTS.flush()


class Session:
    def __init__(self):
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profile_key = None
        self.driver = self.model = self.adapters = None
        self.vae_loaded = False
        self.vae_parked: list | None = None  # what ``unpark`` puts back of a VAE waiting in system RAM
        self.texts: OrderedDict = OrderedDict()  # condition key -> its encoding, most recently used last
        self.te_where: str | None = None  # "gpu", "cpu" (parked) or None (not loaded)
        self.te_parked: list | None = None  # what ``unpark`` puts back of a parked, streamed text encoder
        self.lora_files: OrderedDict = OrderedDict()  # path -> (mtime, state dict), most recent last
        self.lora_key = None
        self.lora_report: list = []
        self.upscaler = None  # the loaded upscaler (``engine.upscale.Upscaler``), the file it came from, its VRAM
        self.upscaler_path: str | None = None
        self.upscaler_bytes = 0

    # --- lifecycle -----------------------------------------------------------
    def unload_all(self):
        self.driver = self.model = self.adapters = None
        self.vae_loaded, self.te_where, self.te_parked, self.lora_key = False, None, None, None
        self.vae_parked, self.texts = None, OrderedDict()
        self.profile_key = None
        self.drop_upscaler(clean=False)
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch._C._host_emptyCache()  # streamed blocks' and layers' pinned RAM

    def ensure_driver(self, profile: dict):
        key = (profile["id"], json.dumps(profile["data"], sort_keys=True))
        if key == self.profile_key:
            return
        from .families import get_family

        self.unload_all()
        # its own copy: a generation may point it at a finetuned diffusion model (``use_model``)
        self.driver = get_family(profile["family"])(dict(profile["data"]), self.device,
                                                    log=lambda m: emit("status", message=str(m).strip()))
        self.profile_key = key

    def use_model(self, path: str):
        """Generate with the diffusion model at ``path`` (the profile's, or a finetuned one in its place), reloading
        only the diffusion model when it changes, and what came from its file with it (YuE2's composer and VAE)."""
        if path == self.driver.profile["diffusion_model"]:
            return
        if self.model is not None:
            emit("status", message="Switching diffusion model")
            self.drop_model()
        self.driver.profile["diffusion_model"] = path
        bundled = self.driver.bundled  # parts loaded from the old file: from the new one when next needed
        if "text_encoder" in bundled:  # (and what it encoded)
            if self.te_where is not None:
                self.driver.unload_text_encoder()
            self.te_where, self.te_parked, self.texts = None, None, OrderedDict()
        if "vae" in bundled:
            self.vae_loaded, self.vae_parked = False, None

    def drop_model(self):
        """Unload the diffusion model only (e.g. to reload it with another offload setting)."""
        if self.driver is not None and self.driver.swapper is not None:
            self.driver.swapper.remove()
            self.driver.swapper = None
        self.model = self.adapters = self.lora_key = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch._C._host_emptyCache()  # its streamed blocks' pinned RAM

    def ensure_models(self, vae: bool = True):
        """The diffusion model, and the VAE on the GPU (``vae``) or else out of it (in system RAM, if loaded)."""
        from .lora import AdapterStack
        from .offload import park

        if self.model is None:
            emit("status", message="Loading the diffusion model")
            self.model = self.driver.load_transformer().requires_grad_(False).eval()
            self.adapters = AdapterStack(self.model, self.device)
            self.lora_key = None
        if vae:
            self.vae_to_gpu()
        elif self.vae_loaded and self.vae_parked is None:
            self.vae_parked = park(self._vae_modules())

    def vae_to_gpu(self):
        from .offload import unpark

        if self.vae_parked is not None:
            unpark(self.vae_parked)
            self.vae_parked = None
        elif not self.vae_loaded:
            self.driver.load_vae()
            self.vae_loaded = True

    def decode(self, lat, low_vram: bool, w: int, h: int, frames: int = 1, keep_vae: bool = False):
        """The render's media (``with_vae``)."""
        return self.with_vae(lambda: self.driver.to_media(lat)[0], low_vram, w, h, frames, keep_vae)

    def with_vae(self, fn, low_vram: bool, w: int, h: int, frames: int = 1, keep_vae: bool = False):
        """``fn()`` (decoding or encoding a w x h render) with the VAE on the GPU. In low VRAM mode the VAE comes to
        the GPU just for this (the diffusion model moving to system RAM meanwhile, if there isn't room for both),
        then goes back to system RAM unless ``keep_vae`` (the next render's input images need it)."""
        if not low_vram:
            return fn()
        from .memory import decode_bytes
        from .offload import make_room, park, unpark

        on_gpu = self.vae_loaded and self.vae_parked is None
        moved = make_room(decode_bytes(self.driver, w, h, frames, loaded=on_gpu), [self.model], self.driver.swapper)
        try:
            self.vae_to_gpu()
            return fn()
        finally:
            if not keep_vae:
                self.vae_parked = park(self._vae_modules())
            unpark(moved)

    # --- text --------------------------------------------------------------
    def _te_module(self):
        te = getattr(self.driver, "text_encoder", None)
        return te if isinstance(te, self.torch.nn.Module) else getattr(te, "model", None)

    def _vae_modules(self):
        return [m for m in (getattr(self.driver, a, None) for a in ("vae", "video_vae", "audio_vae"))
                if isinstance(m, self.torch.nn.Module)]

    def _gpu_modules(self):
        return ([self.model] if self.model is not None else []) + self._vae_modules()

    def _te_estimate(self) -> float:
        """VRAM the text encoder will take, before it's loaded (streamed: the part that isn't streamed)."""
        bundled = self.driver.bundled.get("text_encoder")
        if bundled:  # part of the diffusion-model file (YuE2's composer)
            from .memory import bundled_bytes
            return bundled_bytes(self.driver.profile["diffusion_model"], *bundled) + 1.0e9
        if self.driver.text_encoder_device != self.driver.device:
            from .memory import streamed_text_encoder
            resident, _, inflight = streamed_text_encoder(self.driver, self.driver.profile["text_encoder"])
            return resident + inflight + 1.5e9
        from .comfy_io import shard_files  # a single file, or a Hugging Face folder of shards
        return sum(os.path.getsize(p) for p in shard_files(self.driver.profile["text_encoder"])) * 1.25 + 1.5e9

    @staticmethod
    def _bytes(module) -> int:
        return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))

    def encode(self, conds: dict):
        """Encode the conditions not cached yet: {key: (prompt, images)}. The cache keeps the ``TEXTS_KEPT`` most
        recently used."""
        from .offload import park, unpark

        for k in conds:
            if k in self.texts:
                self.texts.move_to_end(k)
        missing = {k: v for k, v in conds.items() if k not in self.texts}
        if not missing:
            return
        torch = self.torch
        parked = []
        if self.te_where != "gpu" and torch.cuda.is_available():
            te = self._te_module()
            need = self._bytes(te) if te is not None and not self.driver.text_swappers else self._te_estimate()
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0] < need + 1.5e9 and self._gpu_modules():
                emit("status", message="Moving the diffusion model to system RAM while the text encoder runs")
                parked = park(self._gpu_modules(), getattr(self.driver, "swapper", None))
        try:
            if self.te_parked is not None:
                unpark(self.te_parked)
                self.te_parked = None
            elif self.te_where == "cpu":
                emit("status", message="Moving the text encoder to the GPU")
                self._te_module().to(self.device)
            elif self.te_where is None:
                emit("status", message="Loading the text encoder")
                self.driver.load_text_encoder()
            self.te_where = "gpu"
            emit("status", message=f"Encoding {len(missing)} prompt(s)")
            try:
                for k, (prompt, images) in missing.items():
                    self.texts[k] = self.driver.encode_condition(prompt, images)
            except BaseException:
                # off the GPU as after an encode that worked: left there (and marked so), later requests would skip
                # parking and run it beside the diffusion model
                try:
                    self._park_text_encoder()
                except Exception:
                    self.driver.unload_text_encoder()
                    self.te_where, self.te_parked = None, None
                raise
            while len(self.texts) > TEXTS_KEPT:  # (never this request's: they're the most recent)
                self.texts.popitem(last=False)
            self._park_text_encoder()
        finally:
            if parked:
                unpark(parked)

    def _park_text_encoder(self):
        """Keep the text encoder in system RAM when there's room (re-encoding then takes seconds), else unload.
        A streamed one's layers are in RAM already, so parking it moves only the rest."""
        import psutil

        from .offload import park

        te, swappers = self._te_module(), self.driver.text_swappers
        on_gpu = sum(t.numel() * t.element_size() for t in list(te.parameters()) + list(te.buffers())
                     if t.is_cuda) if te is not None else 0
        if te is not None and swappers and psutil.virtual_memory().available > on_gpu + 8e9:
            self.te_parked = park([te], swappers)
            self.te_where = "cpu"
        elif te is not None and not swappers and psutil.virtual_memory().available > self._bytes(te) + 8e9:
            te.to("cpu")
            self.te_where = "cpu"
        else:
            self.driver.unload_text_encoder()
            self.te_where = None
        gc.collect()
        self.torch.cuda.empty_cache()

    # --- LoRAs -----------------------------------------------------------------
    def _lora_file(self, path: str) -> dict:
        from safetensors import safe_open
        from safetensors.torch import load_file

        mtime = os.stat(path).st_mtime_ns
        hit = self.lora_files.get(path)
        if hit and hit[0] == mtime:
            self.lora_files.move_to_end(path)
            return hit[1]
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        family = meta.get("dainamo.family")
        if family and family != self.driver.info.id:
            raise ValueError(f"{os.path.basename(path)} was trained for {family}, not {self.driver.info.id}")
        sd = load_file(path)
        self.lora_files[path] = (mtime, sd)
        while len(self.lora_files) > 16:  # a checkpoint grid goes through up to 12, plus the LoRAs it adds
            self.lora_files.popitem(last=False)
        return sd

    def apply_loras(self, loras: list[dict]) -> list[dict]:
        key = tuple((lr["path"], os.stat(lr["path"]).st_mtime_ns, float(lr["strength"])) for lr in loras)
        if key != self.lora_key:
            if loras:
                emit("status", message=f"Applying {len(loras)} LoRA(s)")
            files = [(os.path.basename(lr["path"]), self._lora_file(lr["path"]), float(lr["strength"])) for lr in loras]
            # ``set`` clears what's applied before it can fail: if it does, nothing is applied, so a later request
            # for the LoRAs applied before mustn't pass for done
            self.lora_key, self.lora_report = None, []
            self.lora_report = self.adapters.set(files)
            self.lora_key = key
        return self.lora_report

    # --- generation ------------------------------------------------------------
    def generate(self, msg: dict, is_cancelled):
        """Renders a request's images (or videos, or songs): each seed with the LoRAs, and without them to compare,
        or a checkpoint grid's cells. Image-to-Image starts each from its input image (``params.strength``). The
        hi-res fix (``params.hires``) makes each render bigger after it's decoded (``_hires_upscale``) and renders it
        again from there at the bigger size; each first pass is an output too (variant "pass1", ``of`` the variant
        it's the first pass of), and the second pass is the render's own output."""
        from .media import save_media
        from .sampling import Cancelled, img2img_sigmas, model_calls
        from .trainer import cond_key

        rid, p, out_dir = msg["id"], msg["params"], msg["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        emit("started", id=rid)
        self.drop_upscaler(big_only=True)
        self.ensure_driver(msg["profile"])
        self.use_model(p.get("model") or msg["profile"]["data"]["diffusion_model"])
        d = self.driver
        w, h = int(p["width"]), int(p["height"])
        frames = int(p.get("frames") or 1)
        if frames > 1 and d.snap_frames:
            frames = d.snap_frames(frames)
        images = list(p.get("images") or [])
        # Image-to-Image's image is where sampling starts, not a condition or a reference
        init_path = images.pop(0) if p.get("workflow") == "i2i" and images else None
        hires = p.get("hires") or None
        audio = list(p.get("audio") or []) if frames > 1 else []
        refs = {**({"refmods": p["refmods"]} if p.get("refmods") else {}), **({"audio": audio} if audio else {}),
                # Reference-to-Video's images are references whatever the model was trained for (MiniMax H3's fl2va
                # would take one as its first frame)
                **({"references": True} if p.get("workflow") == "r2v" else {})}

        def condition(size: tuple[int, int]):
            """The prompt's condition for a render ``size`` (width, height) -> (its cache key, (prompt, images))."""
            prompt, cond_images = d.sample_condition({"prompt": p["prompt"], "width": size[0], "height": size[1],
                                                      "images": images, "frames": frames,
                                                      "lyrics": p.get("lyrics") or "", "seconds": p.get("seconds"),
                                                      **refs})
            return cond_key(prompt, cond_images), (prompt, cond_images)

        # the condition for each pass: an edit model's references are sized for the render, so the hi-res fix's
        # bigger second pass has its own (the same key, encoded once, where the size doesn't change it)
        key, cond = condition((w, h))
        conds = {key: cond}
        keys = [key]
        if hires:
            key2, cond2 = condition((int(hires["width"]), int(hires["height"])))
            conds[key2] = cond2
            keys.append(key2)
        cfg = float(p.get("cfg") or 1.0)
        negative = str(p.get("negative") or "")
        use_cfg = d.supports_samplers and cfg != 1
        if use_cfg:
            conds[negative] = (negative, None)
        if is_cancelled():
            raise Cancelled()
        low_vram = bool(p.get("low_vram")) and d.info.streams_text_encoder
        if low_vram != d.low_vram:
            if self.te_where is not None:  # loaded the other way
                d.unload_text_encoder()
                self.te_where, self.te_parked = None, None
            d.low_vram = low_vram
        offload = float(p.get("offload") or 0) / 100
        if self.model is not None and offload != d.offload:
            emit("status", message="Reloading the diffusion model with the new offload setting")
            self.drop_model()
        d.offload = offload
        self.encode(conds)
        keep_vae = bool(images or audio)  # the family's sampling encodes its input images with the VAE
        self.ensure_models(vae=not low_vram or keep_vae)
        sampler, scheduler, shift = p.get("sampler") or "euler", p.get("scheduler") or "default", p.get("shift")
        steps = int(p["steps"])
        init, strength = None, float(p.get("strength") or 1.0)
        if init_path:
            from .dataset import load_and_crop

            emit("status", message="Encoding the input image")
            px = load_and_crop(init_path, (w, h))  # cover-cropped to the size, as training crops
            init = self.with_vae(lambda: d.encode_image(px), low_vram, w, h, keep_vae=keep_vae)
        fixed = p.get("loras") or []
        grid = p.get("grid")
        # a checkpoint grid: the base model (if asked), then each strength with each checkpoint, per seed;
        # else the LoRAs, and the base model to compare with
        if grid:
            cells = ([("base", None, None)] if grid["base"] else []) + \
                [("grid", c, s) for s in range(len(grid["strengths"])) for c in range(len(grid["checkpoints"]))]
            report = self.apply_loras(fixed + [{"path": grid["checkpoints"][0]["path"], "strength": 1.0}])
        else:
            cells = [("lora", None, None)] + ([("base", None, None)] if p.get("compare_base") and fixed else [])
            report = self.apply_loras(fixed)
        seeds = [int(p["seed"]) + i for i in range(int(p.get("count") or 1))]
        renders = [(i, seed, *cell) for i, seed in enumerate(seeds) for cell in cells]

        def calls(n: int) -> int:
            return model_calls(sampler, n) if d.supports_samplers else n

        # each pass's size, steps and model calls per render: the render, then the hi-res fix's second pass
        passes = [{"size": (w, h), "steps": steps, "calls": calls(steps), "area": 1.0}]
        if init is not None:  # the steps from the strength down (``img2img_sigmas``: as asked, or the nearest)
            passes[0]["steps"] = len(img2img_sigmas(d, scheduler, steps, w, h, shift, strength)) - 1
            passes[0]["calls"] = calls(passes[0]["steps"])
        if hires:
            size2 = (int(hires["width"]), int(hires["height"]))
            asked2 = int(hires.get("steps") or steps)
            steps2 = len(img2img_sigmas(d, scheduler, asked2, *size2, shift, float(hires["strength"]))) - 1
            passes.append({"size": size2, "steps": steps2, "calls": calls(steps2), "asked": asked2,
                           "area": size2[0] * size2[1] / (w * h)})
        total = sum(ps["calls"] for ps in passes) * len(renders)
        images_made = len(renders) * len(passes)
        # sampling and decoding times so far per pass, for seconds per step and the time left
        clock = {"started": time.time(), "last": 0.0, "sampling": [0.0] * len(passes), "calls": [0] * len(passes),
                 "decodes": [[] for _ in passes]}
        at = {"pass": 0, "base": 0, "image": 0}  # the render in progress: its pass, the calls before it, its number

        def per_call(k: int) -> float | None:
            """Seconds a model call of pass ``k`` takes: as timed, else the first pass's scaled by the area."""
            if clock["calls"][k]:
                return clock["sampling"][k] / clock["calls"][k]
            if k and clock["calls"][0]:
                return clock["sampling"][0] / clock["calls"][0] * passes[k]["area"] ** 1.25
            return None

        def per_decode(k: int) -> float:
            done = clock["decodes"][k] or [x * passes[k]["area"] for x in clock["decodes"][0]]
            return sum(done) / len(done) if done else 0.0

        def progress(done: int, message: str | None = None):
            """``done`` model calls of ``total`` (all renders and passes) -> a progress event with the step counter."""
            k, ps = at["pass"], passes[at["pass"]]
            eta = None
            if per_call(0) is not None:
                left = [len(renders) * q["calls"] for q in passes]  # calls still to make, per pass
                for j in range(len(passes)):
                    made = min(left[j], max(0, done - sum(len(renders) * q["calls"] for q in passes[:j])))
                    left[j] -= made
                decodes = [len(renders) - len(clock["decodes"][j]) for j in range(len(passes))]
                eta = sum(left[j] * (per_call(j) or 0) + decodes[j] * per_decode(j) for j in range(len(passes)))
            pc = per_call(k)
            emit("progress", id=rid, done=done, total=total, message=message, image=at["image"] + 1,
                 images=images_made, step=min(ps["steps"], round((done - at["base"]) * ps["steps"] / ps["calls"])),
                 steps=ps["steps"], sec_per_step=pc * ps["calls"] / ps["steps"] if pc else None, eta=eta,
                 started=clock["started"])

        def note(text):
            if is_cancelled():
                raise Cancelled()
            progress(at["base"], text)

        def callback(done, _total):
            if is_cancelled():
                raise Cancelled()
            now = time.time()
            clock["sampling"][at["pass"]] += now - clock["last"]
            clock["calls"][at["pass"]] += 1
            clock["last"] = now
            progress(at["base"] + done)

        def use_cell(variant, c, s) -> str:
            """Apply a render's LoRAs (a grid cell's checkpoint at its strength), or turn them off for the base
            model -> its label."""
            nonlocal report
            if variant == "grid":
                ckpt, strength_ = grid["checkpoints"][c], grid["strengths"][s]
                report = self.apply_loras(fixed + [{"path": ckpt["path"], "strength": strength_}])
                label = f"{ckpt['label']} at {strength_:g}"
            else:
                label = "Base model" if variant == "base" else "Sampling"
            self.adapters.set_enabled(variant != "base")
            return label

        def render(k: int, n: int, seed: int, label: str, init_, strength_, steps_):
            """One pass of one render, sampled and decoded -> its media."""
            ps = passes[k]
            rw, rh = ps["size"]
            at.update({"pass": k, "base": sum(len(renders) * q["calls"] for q in passes[:k]) + n * ps["calls"],
                       "image": k * len(renders) + n})
            progress(at["base"], f"{label} · seed {seed}")
            clock["last"] = time.time()
            extra = {"init": init_, "strength": strength_} if init_ is not None else {}
            lat = d.sample(self.model, self.texts[keys[k]], self.texts.get(negative) if use_cfg else None,
                           rw, rh, steps_, cfg, seed, frames=frames, images=images or None, sampler=sampler,
                           scheduler=scheduler, shift=shift, callback=callback, **refs, **extra)
            progress(at["base"] + ps["calls"], "Decoding")
            if is_cancelled():
                raise Cancelled()
            t = time.time()
            media = self.decode(lat, low_vram, rw, rh, frames, keep_vae=keep_vae)
            clock["decodes"][k].append(time.time() - t)
            return media

        def item(seed, variant, c, s, size) -> dict:
            """An output's details: its seed and variant (a grid cell's checkpoint and strength), and its size
            (none for a song)."""
            out = {"seed": seed, "variant": variant, **({"checkpoint": c, "strength": s} if variant == "grid" else {})}
            return {**out, "width": size[0], "height": size[1]} if size[0] else out

        firsts = []  # the hi-res fix's first passes: (render, its media)
        d.progress_note = note
        try:
            for n, (i, seed, variant, c, s) in enumerate(renders):
                label = use_cell(variant, c, s)
                media = render(0, n, seed, label, init, strength, steps)
                stem = f"{i}_c{c}_s{s}" if variant == "grid" else f"{i}_{variant}"
                if hires:
                    saved = save_media(media, out_dir, f"{stem}_pass1")
                    emit("output", id=rid, item={**saved, **item(seed, variant, c, s, (w, h)), "variant": "pass1",
                                                  "of": variant})
                    firsts.append(((n, i, seed, variant, c, s, stem), media))
                else:
                    saved = save_media(media, out_dir, stem)
                    emit("output", id=rid, item={**saved, **item(seed, variant, c, s, (w, h))})
            if hires:
                at.update({"pass": 1, "base": len(renders) * passes[0]["calls"], "image": len(renders)})
                bigger = self._hires_upscale([m for _, m in firsts], hires, passes[1]["size"], note,
                                             _checker(is_cancelled))
                for ((n, i, seed, variant, c, s, stem), _), big in zip(firsts, bigger):
                    label = use_cell(variant, c, s)
                    at.update({"pass": 1, "base": len(renders) * passes[0]["calls"] + n * passes[1]["calls"],
                               "image": len(renders) + n})
                    note(f"Hi-res fix · encoding seed {seed}")
                    px = _pixels(big)
                    init2 = self.with_vae(lambda: d.encode_image(px), low_vram, *passes[1]["size"], keep_vae=keep_vae)
                    media = render(1, n, seed, f"Hi-res fix · {label}", init2, float(hires["strength"]),
                                   passes[1]["asked"])
                    saved = save_media(media, out_dir, stem)
                    emit("output", id=rid, item={**saved, **item(seed, variant, c, s, passes[1]["size"])})
        finally:
            self.adapters.set_enabled(True)
            d.progress_note = None
        emit("done", id=rid, elapsed=round(time.time() - t0, 1), loras=report)

    def _hires_upscale(self, images: list, hires: dict, size: tuple[int, int], note, check) -> list:
        """The hi-res fix's first passes (PIL images) made ``size`` (width, height): resized (``hires["model"]``
        "resize": Lanczos), or upscaled with the upscaler file in it, the Session's cached one as an upscale run
        uses (``ensure_upscaler``), with the profile's models parked in system RAM meanwhile when it hasn't room
        beside them (``_upscaler_room``). A big upscaler (SeedVR2) is unloaded after, before the second pass.
        ``note(text)`` shows what it's doing; ``check()`` raises to cancel."""
        from PIL import Image

        if hires["model"] == "resize":
            note(f"Hi-res fix · resizing to {size[0]}×{size[1]}")
            return [im.convert("RGB").resize(size, Image.Resampling.LANCZOS) for im in images]
        import numpy as np

        from .offload import unpark

        torch = self.torch
        parked = self._upscaler_room(hires["model"])
        try:
            up = self.ensure_upscaler(hires["model"])
            out = []
            for k, im in enumerate(images):
                what = f"Hi-res fix · upscaling{f' {k + 1} of {len(images)}' if len(images) > 1 else ''}" \
                    f" to {size[0]}×{size[1]} with {up.info.name}"
                note(what)
                x = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
                y = up.upscale(x.contiguous(), size, lambda text: note(f"{what}: {text}"), check)
                out.append(_to_image(y, "RGB"))
            return out
        finally:
            self.drop_upscaler(big_only=True)  # the second pass needs the room
            if parked:
                unpark(parked)


    # --- RefMods -------------------------------------------------------------
    def make_refmod(self, msg: dict, is_cancelled):
        """A RefMod (``engine.refmod``) of ``paths``, images and clips (with ``sheet``, the images on one sheet
        first, saved as sheet.png), through the VAE already loaded when the loaded profile's family and VAE are the
        profile's (a RefMod needs only the VAE, so the diffusion model stays), else the profile's: <out_dir>/<name>
        .safetensors, its preview (``refmod.save_preview``), and made.json describing it (the done event's
        ``made``), copied to ``refmod_copy_to`` if set."""
        from . import refmod
        from .offload import park
        from .sampling import Cancelled

        rid, out_dir, s, profile = msg["id"], msg["out_dir"], msg["settings"], msg["profile"]
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        emit("started", id=rid)
        self.drop_upscaler(big_only=True)
        d = self.driver
        if d is None or d.info.id != profile["family"] or d.profile.get("vae") != profile["data"].get("vae"):
            self.ensure_driver(profile)
            d = self.driver

        def check():
            if is_cancelled():
                raise Cancelled()

        def log(text):
            emit("progress", id=rid, message=str(text).strip())

        parked = self.vae_parked is not None  # low VRAM keeps it in system RAM between renders
        if not self.vae_loaded:
            log("Loading the VAE")
        self.vae_to_gpu()
        try:
            paths = list(msg["paths"])
            if msg.get("sheet"):
                log(f"Putting {len(paths)} images on a sheet")
                refmod.make_sheet(paths, int(s["refmod_size"])).save(os.path.join(out_dir, "sheet.png"))
                paths = [os.path.join(out_dir, "sheet.png")]
            latent, meta = refmod.make(d, paths, s, log=log, check=check)
            fname = msg["name"] + ".safetensors"
            meta["name"] = msg["name"]
            out = refmod.save(os.path.join(out_dir, fname), latent, meta)
            copy_to = str(s.get("refmod_copy_to") or "").strip()
            copied = None
            if copy_to:
                import shutil

                try:
                    os.makedirs(os.path.expanduser(copy_to), exist_ok=True)
                    shutil.copyfile(out, os.path.join(os.path.expanduser(copy_to), fname))
                    copied = copy_to
                except OSError as e:
                    log(f"Couldn't copy it to {copy_to}: {e}")
            t, h, w = (int(x) for x in latent.shape[2:])
            log(f"Decoding a preview of {min(t, refmod.PREVIEW_FRAMES)} frame(s)")
            made = {"name": msg["name"], "file": fname, "path": out, "family": profile["family"], "kind": meta["kind"],
                    "mode": s["refmod_mode"], "references": len(msg["paths"]), "sheet": bool(msg.get("sheet")),
                    "tokens": refmod.tokens(latent), "bytes": os.path.getsize(out), "frames": t, "height": h,
                    "width": w, "copied_to": copied, "created": time.time(),
                    "preview": refmod.save_preview(d, latent, out_dir, check)}
            with open(os.path.join(out_dir, "made.json"), "w", encoding="utf-8") as f:
                json.dump(made, f)
        finally:
            if parked:
                self.vae_parked = park(self._vae_modules())
        emit("done", id=rid, elapsed=round(time.time() - t0, 1), made=made)

    # --- upscaling -------------------------------------------------------------
    def drop_upscaler(self, big_only: bool = False, clean: bool = True):
        """Unload the upscaler; with ``big_only``, only one that took more than ``BIG_UPSCALER`` of VRAM (a pixel
        upscaler's tens of MB can stay beside the profile's models)."""
        if self.upscaler is None or (big_only and self.upscaler_bytes < BIG_UPSCALER):
            return
        if big_only:
            emit("status", message="Unloading the upscaler to make room")
        try:
            self.upscaler.unload()
        finally:
            self.upscaler, self.upscaler_path, self.upscaler_bytes = None, None, 0
        if clean:
            gc.collect()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def ensure_upscaler(self, path: str):
        """The upscaler of the file at ``path``, loaded unless it already is (another one is unloaded first)."""
        from . import upscale

        if self.upscaler is not None and self.upscaler_path == path:
            return self.upscaler
        self.drop_upscaler()
        info = upscale.identify(path)
        cuda = self.torch.cuda.is_available()
        before = self.torch.cuda.memory_allocated() if cuda else 0
        emit("status", message=f"Loading {info.name}")
        self.upscaler = upscale.load(info, self.device, log=lambda m: emit("status", message=str(m).strip()))
        self.upscaler_path = path
        self.upscaler_bytes = self.torch.cuda.memory_allocated() - before if cuda else 0
        return self.upscaler

    def _upscaler_room(self, path: str) -> list:
        """Park the profile's models (diffusion model, VAE) in system RAM when the upscaler of ``path`` hasn't room
        beside them: its working memory, plus its weights if it isn't loaded yet. What to unpark afterwards."""
        torch = self.torch
        if not torch.cuda.is_available() or not self._gpu_modules():
            return []
        from . import upscale
        from .offload import park

        info = upscale.identify(path)
        # the upscaler class says what it works in (SeedVR2's ``vram_bytes``); else the table's guess
        need = _upscaler_work(info)
        if self.upscaler_path != path:
            need += sum(os.path.getsize(p) for p in (info.path, info.companion) if p and os.path.exists(p))
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] >= need:
            return []
        emit("status", message="Moving the diffusion model to system RAM while the upscaler runs")
        return park(self._gpu_modules(), getattr(self.driver, "swapper", None))

    def upscale(self, msg: dict, is_cancelled):
        """Each of ``inputs`` (images and clips) upscaled by ``factor`` with the upscaler in ``model``, one output
        each: a fixed-scale model runs at its scale and is fitted to the size asked; clips keep their frame rate and
        sound, and are written as they're upscaled (``_upscale_clip``). Progress counts frames, an image as one."""
        from . import upscale
        from .dataset import image_size, media_kind
        from .media import save_media
        from .offload import unpark
        from .upscale import any_channels

        rid, out_dir, factor = msg["id"], msg["out_dir"], float(msg["factor"])
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        emit("started", id=rid)
        check = _checker(is_cancelled)
        if not factor > 0:
            raise ValueError(f"Can't upscale by {factor:g}")
        jobs = []  # (path, kind, its size, the size asked, frames, what was probed of a clip)
        for i in msg["inputs"]:
            path = i["path"]
            kind = i.get("kind") or media_kind(path)
            if kind == "video":
                clip = _probe_clip(path)
                w, h, frames = clip["width"], clip["height"], clip["frames"]
            elif kind == "image":
                (w, h), frames, clip = image_size(path), 1, None
            else:
                raise ValueError(f"{os.path.basename(path)} isn't an image or a video")
            # H.264 in 4:2:0 wants even sides
            jobs.append((path, kind, (w, h), upscale.target_size(w, h, factor, 2 if clip else 1), frames, clip))
        total = sum(j[4] for j in jobs)
        total_px = sum(j[4] * j[3][0] * j[3][1] for j in jobs) or 1
        # the upscaler's own settings (SeedVR2's), passed on as they are; a pixel upscaler has none
        kw = {"options": dict(msg["options"])} if msg.get("options") else {}
        state = {"done": 0, "px": 0, "input": 0, "since": None}

        def report(message: str):
            check()
            eta = None
            if state["px"] and state["since"]:  # from the output pixels made so far, model loading left out
                eta = (time.time() - state["since"]) / state["px"] * (total_px - state["px"])
            emit("progress", id=rid, done=state["done"], total=total, message=message, input=state["input"] + 1,
                 inputs=len(jobs), eta=eta, started=t0)

        parked = self._upscaler_room(msg["model"])
        try:
            up = self.ensure_upscaler(msg["model"])
            if kw and hasattr(up, "check_options"):  # once: a long clip's pieces would each log the same changes
                kw = {"options": up.check_options(kw["options"])}
            state["since"] = time.time()
            for n, (path, kind, (iw, ih), (w, h), frames, clip) in enumerate(jobs):
                state["input"] = n
                name = os.path.basename(path)
                report(f"{name}: {iw}×{ih} to {w}×{h}")
                stem = f"{n}_{_safe_stem(path)}_x{factor:g}"
                if kind == "image":
                    x, mode, _ = _read_image(path)
                    y = any_channels(up, x, (w, h), lambda text: report(f"{name}: {text}"), check, **kw)
                    saved = save_media(_to_image(y, mode), out_dir, stem)
                    state["done"] += 1
                    state["px"] += w * h
                else:
                    saved = self._upscale_clip(up, path, clip, (w, h), out_dir, stem, state, report, check, kw)
                emit("output", id=rid, item={**saved, "source": path, "width": w, "height": h})
        finally:
            if parked:
                unpark(parked)
        emit("done", id=rid, elapsed=round(time.time() - t0, 1))

    def _upscale_clip(self, up, path: str, clip: dict, size: tuple[int, int], out_dir: str, stem: str, state: dict,
                      report, check, kw: dict) -> dict:
        """A clip upscaled to ``size`` and written to ``out_dir`` as <stem>.mp4 with its frame rate and its sound
        (``media.VideoWriter``, which muxes the sound from the clip) -> {file, kind, poster}. It's read, upscaled
        and written a piece at a time, so no length is too long: a frame at a time for a pixel upscaler; for one
        that takes clips as clips, the whole clip when its frames fit in ``CLIP_RAM`` (the upscaler makes its own
        overlapping passes), else pieces of whole passes, each sharing the upscaler's ``overlap_frames`` with the
        piece before, crossfaded as its passes are (``upscale.crossfade``)."""
        import numpy as np
        import psutil

        from .media import VideoWriter
        from .upscale import crossfade

        torch = self.torch
        name, (w, h), count = os.path.basename(path), size, max(1, clip["frames"])
        per = 4 * 3 * (clip["width"] * clip["height"] + w * h)  # a frame in and out, as floats
        free = psutil.virtual_memory().available
        fits = int(min(CLIP_RAM, free / 4) // per)  # frames a piece can be
        chunk, overlap = 1, 0
        if up.info.video:
            per_pass, shared = getattr(up, "frames_per_pass", None), getattr(up, "overlap_frames", None)
            n = max(1, int(per_pass(size, kw.get("options")))) if per_pass else count
            overlap = max(0, min(int(shared(size, kw.get("options"))) if shared else 0, (n - 1) // 2))
            chunk = min(count, fits)
            if count > chunk >= n:  # pieces of whole passes: m * (n - overlap) + overlap frames
                chunk = overlap + (chunk - overlap) // (n - overlap) * (n - overlap)
        if fits < chunk or chunk <= overlap:
            least = overlap + 1
            raise ValueError(f"{name}: {least} frame{'s' * (least > 1)} at {w}×{h} take {least * per / 1e9:.1f} GB "
                             f"of system RAM to upscale together, more than a quarter of the {free / 1e9:.1f} GB "
                             "free. Try a smaller factor, or close what else is using RAM.")
        # a clip's pieces, as the header counts its frames (a pixel upscaler's frames aren't pieces)
        pieces = 1 if chunk >= count or not up.info.video else math.ceil((count - overlap) / (chunk - overlap))
        made, held = 0, None  # held: a piece's last frames, kept back to crossfade with the start of the next
        # the rate as ffprobe gave it (30000/1001, not 29.97002997...), passed to ffmpeg as it is
        with VideoWriter(out_dir, stem, clip["fps"], audio=path if clip["audio"] else None) as video, \
                contextlib.closing(_clip_frames(path, clip["width"], clip["height"], chunk, overlap)) as frames_in:
            for k, frames in enumerate(frames_in):
                check()
                repeated = overlap if k else 0  # the frames this piece shares with the one before
                part = f"part {k + 1} of {max(pieces, k + 1)}: " if pieces > 1 else ""
                x = torch.from_numpy(frames.astype(np.float32)).div_(255).permute(0, 3, 1, 2).contiguous()
                y = up.upscale(x, size, lambda text: report(f"{name}: {part}{text}"), check, **kw)
                del x
                if repeated:
                    with torch.inference_mode():  # SeedVR2's frames are inference tensors: changed only in it
                        y[:repeated] = crossfade(held, y[:repeated])
                keep = overlap if len(frames) == chunk else 0  # only a whole piece can have another after it
                _write_frames(video, y[:len(y) - keep])
                held = y[len(y) - keep:].clone() if keep else None
                del y
                made += len(frames) - repeated
                state["done"] += len(frames) - repeated
                state["px"] += (len(frames) - repeated) * w * h
                report(f"{name}: frame {made} of {max(count, made)}")
            if held is not None:
                _write_frames(video, held)
            if not made:
                raise ValueError(f"{name}: no frames could be read")
            report(f"{name}: writing the video")
            return video.close()

    def upscale_dataset(self, msg: dict, is_cancelled):
        """A dataset's small images upscaled in place: each of ``paths`` whose area is under ``resolution`` squared
        is upscaled to reach it (by ``max_factor`` at most), written as <stem>.png beside the original, which moves
        to the ``_originals`` folder beside it. Captions (<stem>.txt) are untouched. Clips, and images large enough
        already, are skipped. ``made``: {upscaled, skipped, errors, files: [{path, original, width, height}]}."""
        from . import upscale
        from .dataset import image_size, media_kind
        from .offload import unpark
        from .upscale import any_channels

        uid, paths = msg["id"], list(msg["paths"])
        resolution, max_factor = int(msg["resolution"]), float(msg["max_factor"])
        t0 = time.time()
        emit("started", id=uid)
        check = _checker(is_cancelled)
        made = {"upscaled": 0, "skipped": 0, "errors": [], "files": []}
        parked, up = [], None

        def report(done: int, message: str):
            emit("progress", id=uid, done=done, total=len(paths), message=message,
                 made={k: v for k, v in made.items() if k != "files"})

        try:
            for n, path in enumerate(paths):
                check()
                name = os.path.basename(path)
                if media_kind(path) != "image":
                    made["skipped"] += 1
                    continue
                try:
                    w, h = image_size(path)
                    size = upscale.target_size(w, h, upscale.factor_to_reach(w, h, resolution, max_factor))
                    if size[0] <= w or size[1] <= h:
                        made["skipped"] += 1
                        continue
                    report(n, f"{name}: {w}×{h} to {size[0]}×{size[1]}")
                    x, mode, icc = _read_image(path)
                except Exception as e:  # a file that isn't an image after all: noted, and on to the next
                    made["errors"].append(f"{name}: {e}")
                    continue
                if up is None:  # loaded once something needs it; failing to load or run it ends the job
                    parked = self._upscaler_room(msg["model"])
                    up = self.ensure_upscaler(msg["model"])
                y = any_channels(up, x, size, None, check)
                try:
                    original = _replace_image(path, _to_image(y, mode), icc)
                except (OSError, ValueError) as e:
                    made["errors"].append(f"{name}: {e}")
                    continue
                made["upscaled"] += 1
                made["files"].append({"path": os.path.splitext(path)[0] + ".png", "original": original,
                                      "width": size[0], "height": size[1]})
            report(len(paths), f"Upscaled {made['upscaled']}, skipped {made['skipped']}")
        finally:
            if parked:
                unpark(parked)
        emit("done", id=uid, elapsed=round(time.time() - t0, 1), made=made)


# VRAM an upscaler works in beyond its weights, by kind, to judge whether the profile's models must make room
UPSCALE_WORK = {"pixel": 3e9, "seedvr2": 9.7e9}
# system RAM a clip's piece may take while it's upscaled (its frames in and out, as floats), and at most a quarter
# of what's free: a longer clip goes in pieces (8 GB is 300 frames from 480p to 1080p)
CLIP_RAM = 8e9


def _upscaler_work(info) -> float:
    """Working VRAM an upscaler of ``info``'s kind needs beside its weights: the class's ``vram_bytes`` when it has
    one, else ``UPSCALE_WORK``."""
    if info.kind == "seedvr2":
        try:
            from .upscalers.seedvr2 import SeedVR2Upscaler
            return float(getattr(SeedVR2Upscaler, "vram_bytes", UPSCALE_WORK["seedvr2"]))
        except ImportError:
            pass
    return UPSCALE_WORK.get(info.kind, UPSCALE_WORK["pixel"])
BIG_UPSCALER = 1.5e9  # an upscaler taking more VRAM than this is unloaded when a generation comes


def _checker(is_cancelled):
    from .sampling import Cancelled

    def check():
        if is_cancelled():
            raise Cancelled()
    return check


def _safe_stem(path: str) -> str:
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in os.path.splitext(os.path.basename(path))[0])
    return stem[:60] or "input"


def _read_image(path: str):
    """An image file, turned by its EXIF orientation -> ((1, C, H, W) float in [0, 1], the mode to save the result
    in, its ICC profile). C is 3 (RGB), 4 (RGBA), 1 (greyscale) or 2 (greyscale with transparency)."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I", "F") or im.mode.startswith("I;16"):  # 16-bit greyscale
            arr, mode = np.clip(np.asarray(im, dtype=np.float32) / 65535, 0, 1), "L"
        else:
            alpha = im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info
            gray = im.mode in ("1", "L", "LA", "La")
            mode = ("LA" if gray else "RGBA") if alpha else ("L" if gray else "RGB")
            arr = np.asarray(im.convert(mode), dtype=np.float32) / 255
    arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    return torch.from_numpy(arr).permute(2, 0, 1)[None].contiguous(), mode, icc


def _to_image(frames, mode: str):
    """(1, C, H, W) float in [0, 1] -> a PIL image in ``mode`` (L, LA, RGB or RGBA, as C)."""
    from PIL import Image

    arr = (frames[0].permute(1, 2, 0) * 255).round_().clamp_(0, 255).byte().numpy()
    return Image.fromarray(arr[..., 0] if mode == "L" else arr)


def _pixels(image):
    """A PIL image -> (1, 3, H, W) float in [-1, 1], as ``encode_image`` takes it (``dataset.load_and_crop``'s)."""
    import numpy as np
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _replace_image(path: str, image, icc) -> str:
    """Put ``image`` in the place of the image at ``path``, as <stem>.png beside it, the original moved into the
    ``_originals`` folder there (never over an earlier original: <name>_2 and so on). Where it moved to."""
    from .dataset import ORIGINALS_DIR

    folder, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    target = os.path.join(folder, stem + ".png")
    if os.path.normcase(target) != os.path.normcase(path) and os.path.exists(target):
        raise ValueError(f"{stem}.png is there already, another image of the same name")
    keep = os.path.join(folder, ORIGINALS_DIR)
    os.makedirs(keep, exist_ok=True)
    kept, k = os.path.join(keep, name), 1
    while os.path.exists(kept):
        k += 1
        kept = os.path.join(keep, f"{stem}_{k}{ext}")
    tmp = os.path.join(folder, f".{stem}.upscaling.png")  # hidden: datasets skip it if it's ever left behind
    try:
        image.save(tmp, **({"icc_profile": icc} if icc else {}))
        os.replace(path, kept)
        try:
            os.replace(tmp, target)
        except OSError:
            os.replace(kept, path)  # put the original back
            raise
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return kept


def _probe_clip(path: str) -> dict:
    """{width, height (as shown, turned by its rotation), frames (from the header, else duration × rate), fps (as
    ffprobe gives it, e.g. "30000/1001"), rate (as a number), audio, sample_rate, channels (at most 2)} of a clip."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,sample_rate,channels"
                          ":stream_tags=rotate:stream_side_data=rotation:format=duration", "-of", "json", path],
                         capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        why = f" ({out.stderr.strip()})" if out.stderr.strip() else ""
        raise ValueError(f"{os.path.basename(path)}: no video stream{why}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps = next((r for r in (video.get("avg_frame_rate"), video.get("r_frame_rate"))
                if r and _rate(r) > 0), "24/1")
    w, h = int(video["width"]), int(video["height"])
    turns = [(video.get("tags") or {}).get("rotate")] + [d.get("rotation") for d in video.get("side_data_list") or []]
    if any(t is not None and abs(int(float(t))) % 180 == 90 for t in turns):
        w, h = h, w
    duration = float((info.get("format") or {}).get("duration") or 0.0)
    frames = int(video.get("nb_frames") or 0) or max(1, round(duration * _rate(fps)))
    return {"width": w, "height": h, "frames": frames, "fps": fps, "rate": _rate(fps), "audio": audio is not None,
            "sample_rate": int((audio or {}).get("sample_rate") or 48000),
            "channels": min(2, int((audio or {}).get("channels") or 2))}


def _rate(text: str) -> float:
    num, _, den = str(text).partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _write_frames(video, frames, batch: int = 8):
    """Upscaled frames (n, 3, h, w) float in [0, 1] into a ``media.VideoWriter`` as 8-bit, a few at a time, so
    there's never a second copy of them all."""
    import torch

    for a in range(0, frames.shape[0], batch):
        video.write((frames[a:a + batch].permute(0, 2, 3, 1) * 255).round_().clamp_(0, 255).to(torch.uint8).numpy())


def _clip_frames(path: str, width: int, height: int, chunk: int, overlap: int = 0):
    """A clip's frames as (n, height, width, 3) uint8 arrays of up to ``chunk``, decoded by ffmpeg as they're
    asked for: every frame as it is, none added or dropped to even out the frame rate. Each array after the first
    starts with the last ``overlap`` frames of the one before, then up to ``chunk - overlap`` new ones."""
    import numpy as np

    size = width * height * 3
    overlap = max(0, min(overlap, chunk - 1))
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough",
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=err)
        try:
            last = None
            while True:
                want = chunk if last is None else chunk - overlap
                data = proc.stdout.read(size * want)
                n = len(data) // size
                if n:
                    new = np.frombuffer(data, np.uint8, n * size).reshape(n, height, width, 3)
                    last = new if last is None or not overlap else np.concatenate([last[-overlap:], new])
                    yield last
                if len(data) < size * want:
                    break
            if proc.wait() != 0:
                err.seek(0)
                raise ValueError(f"{os.path.basename(path)}: ffmpeg couldn't read it: "
                                 f"{err.read().decode(errors='replace').strip()[-500:]}")
        finally:
            if proc.poll() is None:  # stopped early (cancelled, or an error upscaling)
                proc.kill()
            proc.stdout.close()
            proc.wait()


def _out_of_memory(cmd: str, msg: dict, session: Session) -> str:
    """What to try when ``cmd`` ran out of GPU memory, to go before the error."""
    if cmd == "generate":
        return "Out of GPU memory. Try a smaller size, fewer frames, or fewer LoRAs. "
    if cmd == "refmod":
        return "Out of GPU memory: lower the size or the frames per clip. "
    seedvr2 = session.upscaler is not None and session.upscaler.info.kind == "seedvr2"
    if session.upscaler is None:
        try:
            from .upscale import identify

            seedvr2 = identify(msg["model"]).kind == "seedvr2"
        except Exception:
            pass
    what = "a smaller factor" if cmd == "upscale" else "a lower resolution or max factor"
    if seedvr2:
        what += ", or a smaller SeedVR2 tile (its tile option) or fewer frames a pass (max frames)"
    return f"Out of GPU memory: try {what}. "


def main():
    global EVENTS
    # events get the real stdout; anything libraries print goes to stderr (the worker log)
    EVENTS = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    commands: queue.Queue = queue.Queue()
    cancelled: set = set()

    def reader():
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("cmd") == "cancel":
                cancelled.add(msg.get("id"))
                continue
            commands.put(msg)
        commands.put({"cmd": "shutdown"})  # the server went away

    threading.Thread(target=reader, daemon=True).start()
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()
    session = Session()
    from .sampling import Cancelled

    emit("ready", device=str(session.device))
    while True:
        msg = commands.get()
        cmd = msg.get("cmd")
        if cmd == "shutdown":
            break
        if cmd == "unload":
            session.unload_all()
            emit("idle")
        elif cmd in ("generate", "refmod", "upscale", "upscale_dataset"):
            rid = msg.get("id")
            run = {"generate": session.generate, "refmod": session.make_refmod, "upscale": session.upscale,
                   "upscale_dataset": session.upscale_dataset}[cmd]
            try:
                if rid in cancelled:
                    raise Cancelled()
                run(msg, lambda: rid in cancelled)
            except Cancelled:
                emit("cancelled", id=rid)
            except Exception as e:  # report and keep serving
                traceback.print_exc()
                text = f"{type(e).__name__}: {e}"
                if "out of memory" in str(e).lower():
                    text = _out_of_memory(cmd, msg, session) + text
                emit("error", id=rid, message=text)
                if session.torch.cuda.is_available():
                    session.torch.cuda.empty_cache()
            cancelled.discard(rid)
    session.unload_all()
    emit("exit")


if __name__ == "__main__":
    main()

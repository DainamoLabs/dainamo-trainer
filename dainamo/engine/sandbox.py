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
        """The render's media. In low VRAM mode the VAE comes to the GPU just for this (the diffusion model
        moving to system RAM meanwhile, if there isn't room for both), then goes back to system RAM unless
        ``keep_vae`` (the next render's input images need it)."""
        if not low_vram:
            return self.driver.to_media(lat)[0]
        from .memory import decode_bytes
        from .offload import make_room, park, unpark

        on_gpu = self.vae_loaded and self.vae_parked is None
        moved = make_room(decode_bytes(self.driver, w, h, frames, loaded=on_gpu), [self.model], self.driver.swapper)
        try:
            self.vae_to_gpu()
            return self.driver.to_media(lat)[0]
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
            for k, (prompt, images) in missing.items():
                self.texts[k] = self.driver.encode_condition(prompt, images)
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
        from .media import save_media
        from .sampling import Cancelled, model_calls
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
        audio = list(p.get("audio") or []) if frames > 1 else []
        refs = {**({"refmods": p["refmods"]} if p.get("refmods") else {}), **({"audio": audio} if audio else {}),
                # Reference-to-Video's images are references whatever the model was trained for (MiniMax H3's fl2va
                # would take one as its first frame)
                **({"references": True} if p.get("workflow") == "r2v" else {})}

        prompt, cond_images = d.sample_condition({"prompt": p["prompt"], "width": w, "height": h,
                                                  "images": images, "frames": frames,
                                                  "lyrics": p.get("lyrics") or "", "seconds": p.get("seconds"),
                                                  **refs})
        key = cond_key(prompt, cond_images)
        conds = {key: (prompt, cond_images)}
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
        self.ensure_models(vae=not low_vram or bool(images or audio))
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
        steps = int(p["steps"])
        per = model_calls(p.get("sampler") or "euler", steps) if d.supports_samplers else steps
        total = per * len(renders)
        # sampling and decoding times so far, for seconds per step and the time left
        clock = {"started": time.time(), "last": 0.0, "sampling": 0.0, "calls": 0, "decodes": []}
        n = 0  # the render in progress

        def progress(done: int, message: str | None = None):
            """``done`` model calls of ``total`` (all renders) -> a progress event with the step counter."""
            per_call = clock["sampling"] / clock["calls"] if clock["calls"] else None
            eta = None
            if per_call is not None:
                decode = sum(clock["decodes"]) / len(clock["decodes"]) if clock["decodes"] else 0.0
                eta = (total - done) * per_call + (len(renders) - len(clock["decodes"])) * decode
            emit("progress", id=rid, done=done, total=total, message=message, image=n + 1, images=len(renders),
                 step=min(steps, round((done - n * per) * steps / per)), steps=steps,
                 sec_per_step=per_call * per / steps if per_call else None, eta=eta, started=clock["started"])

        def note(text):
            if is_cancelled():
                raise Cancelled()
            progress(n * per, text)

        def callback(done, _total):
            if is_cancelled():
                raise Cancelled()
            now = time.time()
            clock["sampling"] += now - clock["last"]
            clock["calls"] += 1
            clock["last"] = now
            progress(n * per + done)

        d.progress_note = note
        try:
            for n, (i, seed, variant, c, s) in enumerate(renders):
                if variant == "grid":
                    ckpt, strength = grid["checkpoints"][c], grid["strengths"][s]
                    report = self.apply_loras(fixed + [{"path": ckpt["path"], "strength": strength}])
                    label = f"{ckpt['label']} at {strength:g}"
                else:
                    label = "Base model" if variant == "base" else "Sampling"
                self.adapters.set_enabled(variant != "base")
                progress(n * per, f"{label} · seed {seed}")
                clock["last"] = time.time()
                lat = d.sample(self.model, self.texts[key], self.texts.get(negative) if use_cfg else None,
                               w, h, steps, cfg, seed, frames=frames, images=images or None,
                               sampler=p.get("sampler") or "euler", scheduler=p.get("scheduler") or "default",
                               shift=p.get("shift"), callback=callback, **refs)
                progress((n + 1) * per, "Decoding")
                if is_cancelled():
                    raise Cancelled()
                t = time.time()
                media = self.decode(lat, low_vram, w, h, frames, keep_vae=bool(images or audio))
                clock["decodes"].append(time.time() - t)
                if variant == "grid":
                    saved = save_media(media, out_dir, f"{i}_c{c}_s{s}")
                    emit("output", id=rid, item={**saved, "seed": seed, "variant": variant, "checkpoint": c, "strength": s})
                else:
                    saved = save_media(media, out_dir, f"{i}_{variant}")
                    emit("output", id=rid, item={**saved, "seed": seed, "variant": variant})
        finally:
            self.adapters.set_enabled(True)
            d.progress_note = None
        emit("done", id=rid, elapsed=round(time.time() - t0, 1), loras=report)


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
        sound. Progress counts frames, an image as one."""
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
                    media = self._upscale_clip(up, path, clip, (w, h), state, report, check, kw)
                    report(f"{name}: writing the video")
                    saved = save_media(media, out_dir, stem)
                    del media
                emit("output", id=rid, item={**saved, "source": path, "width": w, "height": h})
        finally:
            if parked:
                unpark(parked)
        emit("done", id=rid, elapsed=round(time.time() - t0, 1))

    def _upscale_clip(self, up, path: str, clip: dict, size: tuple[int, int], state: dict, report, check,
                      kw: dict) -> dict:
        """A clip upscaled to ``size`` -> ``save_media``'s video dict, with its frame rate and sound. Frames are
        read and upscaled a chunk at a time (one at a time for a pixel upscaler, as much of the clip as fits for one
        that takes clips as clips), so only the result is held whole, as 8-bit frames."""
        import numpy as np
        import psutil

        from .dataset import read_audio

        torch = self.torch
        name, (w, h), count = os.path.basename(path), size, clip["frames"]
        free = psutil.virtual_memory().available
        need = 2 * count * w * h * 3  # the frames, and save_media's copy of them for ffmpeg
        if need + 2e9 > free:
            raise ValueError(f"{name}: {count} frames at {w}×{h} take {need / 1e9:.0f} GB of system RAM to write, and "
                             f"{free / 1e9:.0f} GB is free. Try a smaller factor or a shorter clip.")
        per = 4 * 3 * (clip["width"] * clip["height"] + w * h)  # a frame in and out, as floats
        chunk = max(1, min(count, int(min(8e9, free / 4) // per))) if up.info.video else 1
        per_pass = getattr(up, "frames_per_pass", None)
        if per_pass is not None and chunk < count:  # pieces end where the upscaler's own passes end
            n = max(1, int(per_pass(size)))
            chunk = max(n, chunk - chunk % n) if chunk >= n else chunk
        out, made = [], 0
        with contextlib.closing(_clip_frames(path, clip["width"], clip["height"], chunk)) as chunks:
            for frames in chunks:
                check()
                x = torch.from_numpy(frames.astype(np.float32)).div_(255).permute(0, 3, 1, 2).contiguous()
                y = up.upscale(x, size, lambda text: report(f"{name}: {text}"), check, **kw)
                out.append((y.permute(0, 2, 3, 1) * 255).round_().clamp_(0, 255).to(torch.uint8).numpy())
                del x, y
                made += len(frames)
                state["done"] += len(frames)
                state["px"] += len(frames) * w * h
                report(f"{name}: frame {made} of {max(count, made)}")
        if not out:
            raise ValueError(f"{name}: no frames could be read")
        audio = None
        if clip["audio"]:
            audio = read_audio(path, made / clip["rate"], clip["sample_rate"], clip["channels"]).numpy()
        frames = out[0] if len(out) == 1 else np.concatenate(out)
        del out
        # the rate as ffprobe gave it (30000/1001, not 29.97002997...): save_media passes it to ffmpeg as it is
        return {"frames": frames, "fps": clip["fps"], "audio": audio, "sample_rate": clip["sample_rate"]}

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


def _clip_frames(path: str, width: int, height: int, chunk: int):
    """A clip's frames as (n, height, width, 3) uint8 arrays, ``chunk`` at a time, decoded by ffmpeg as they're
    asked for: every frame as it is, none added or dropped to even out the frame rate."""
    import numpy as np

    size = width * height * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough",
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                data = proc.stdout.read(size * chunk)
                n = len(data) // size
                if n:
                    yield np.frombuffer(data, np.uint8, n * size).reshape(n, height, width, 3)
                if len(data) < size * chunk:
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
    return f"Out of GPU memory: try {what}" + (", or a smaller SeedVR2 tile (its tile option)" if seedvr2 else "") + ". "


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

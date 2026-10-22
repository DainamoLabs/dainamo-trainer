"""Sandbox worker: ``python -m dainamo.engine.sandbox`` (one GPU, chosen by CUDA_VISIBLE_DEVICES).

Keeps a profile's models loaded between generations so trying prompts, LoRAs and settings costs
sampling time only. Commands arrive as JSON lines on stdin; events go out as JSON lines on stdout
(library output is redirected to stderr, the worker log):

  in:  {"cmd": "generate", "id", "out_dir", "profile": {id, family, data}, "params": {...}}
       {"cmd": "cancel", "id"}   {"cmd": "unload"}   {"cmd": "shutdown"}
  out: ready | status {message} | started {id}
       | progress {id, done, total, message, image, images, step, steps, sec_per_step, eta, started}
       | output {id, item} | done {id, elapsed, loras} | error {id, message} | cancelled {id} | idle | exit

Memory: the diffusion model and VAE stay on the GPU, apart from the share of the diffusion model's
blocks that the request's ``offload`` keeps in system RAM (changing it reloads the model, as does a request's
``model``: a finetuned diffusion model in place of the profile's). The text
encoder is loaded only to encode new prompts (encodings are cached), then parked in system RAM when
there's room for it, else unloaded. When it doesn't fit next to the diffusion model, the diffusion
model and VAE are parked in RAM while it runs. A request with ``low_vram`` streams the text encoder's
layers from system RAM instead, so only the rest of it comes to the GPU, and keeps the VAE in system RAM
while sampling (unless input images or reference audio need it), moving the diffusion model there while a render decodes if
the GPU hasn't room for both. Closing stdin (the server going away) shuts the worker down.
"""

from __future__ import annotations

import gc
import json
import os
import queue
import sys
import threading
import time
import traceback
from collections import OrderedDict

EVENTS = None


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
        self.texts: dict = {}
        self.te_where: str | None = None  # "gpu", "cpu" (parked) or None (not loaded)
        self.te_parked: list | None = None  # what ``unpark`` puts back of a parked, streamed text encoder
        self.lora_files: OrderedDict = OrderedDict()  # path -> (mtime, state dict), most recent last
        self.lora_key = None
        self.lora_report: list = []

    # --- lifecycle -----------------------------------------------------------
    def unload_all(self):
        self.driver = self.model = self.adapters = None
        self.vae_loaded, self.te_where, self.te_parked, self.texts, self.lora_key = False, None, None, {}, None
        self.vae_parked = None
        self.profile_key = None
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
        only the diffusion model when it changes."""
        if path == self.driver.profile["diffusion_model"]:
            return
        if self.model is not None:
            emit("status", message="Switching diffusion model")
            self.drop_model()
        self.driver.profile["diffusion_model"] = path

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
        return os.path.getsize(self.driver.profile["text_encoder"]) * 1.25 + 1.5e9

    @staticmethod
    def _bytes(module) -> int:
        return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))

    def encode(self, conds: dict):
        """Encode the conditions not cached yet: {key: (prompt, images)}."""
        from .offload import park, unpark

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
            self.lora_report = self.adapters.set(
                [(os.path.basename(lr["path"]), self._lora_file(lr["path"]), float(lr["strength"])) for lr in loras])
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
        self.ensure_driver(msg["profile"])
        self.use_model(p.get("model") or msg["profile"]["data"]["diffusion_model"])
        d = self.driver
        w, h = int(p["width"]), int(p["height"])
        frames = int(p.get("frames") or 1)
        if frames > 1 and d.snap_frames:
            frames = d.snap_frames(frames)
        images = list(p.get("images") or [])
        audio = list(p.get("audio") or []) if frames > 1 else []
        refs = {**({"refmods": p["refmods"]} if p.get("refmods") else {}), **({"audio": audio} if audio else {})}

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
        elif cmd == "generate":
            rid = msg.get("id")
            try:
                if rid in cancelled:
                    raise Cancelled()
                session.generate(msg, lambda: rid in cancelled)
            except Cancelled:
                emit("cancelled", id=rid)
            except Exception as e:  # report and keep serving
                traceback.print_exc()
                text = f"{type(e).__name__}: {e}"
                if "out of memory" in str(e).lower():
                    text = "Out of GPU memory. Try a smaller size, fewer frames, or fewer LoRAs. " + text
                emit("error", id=rid, message=text)
                if session.torch.cuda.is_available():
                    session.torch.cuda.empty_cache()
            cancelled.discard(rid)
    session.unload_all()
    emit("exit")


if __name__ == "__main__":
    main()

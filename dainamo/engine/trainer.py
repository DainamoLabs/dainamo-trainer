"""Family-agnostic LoRA training loop.

A job directory holds ``job.json`` (written by the server). The trainer writes:
  status.json    latest progress snapshot (atomically replaced)
  metrics.jsonl  one line per step: step, loss, lr
  samples/       samples generated during training (PNG, or MP4 + poster for video families)
  output/        LoRA checkpoints (ComfyUI-compatible .safetensors); a finetune's are whole diffusion models
  state/         resumable optimizer state (and, with ``ema_enabled``, the EMA of the LoRA's weights)
  cache.json     the files of the encoding cache it uses (``engine.cache``), so they're kept while it runs
and polls ``control.json`` for stop / sample-now / save-now requests. SIGTERM is
treated like a stop request, so state is saved before exiting.

Encoded items and captions go to the cache that jobs share, ``job.json``'s ``cache_dir`` (else the job's own
``cache/``), and are reused by any job encoding them the same way.

A job.json with ``evaluate`` set is a checkpoint evaluation of a finished job instead (see
``run_evaluation``): the job's saved checkpoints are scored, and nothing is trained. Settings with
``slider_enabled`` train a concept slider (``engine.slider``) instead of learning the datasets, and with
``refmod_enabled`` the job makes a RefMod of them instead (``run_refmod``): nothing is trained. With
``finetune_enabled`` every weight of the diffusion model trains (``engine.finetune``) in place of a LoRA, and with
``extract_enabled`` the job extracts a LoRA from a finetune's checkpoint (``run_extract``), and with ``merge_enabled``
it merges LoRAs into one (``run_merge``), training nothing. With ``ema_enabled`` (not for a finetune) an average of
the LoRA's weights (``engine.ema``) is what's saved, sampled and evaluated, while training steps the live weights.

Everything model-specific goes through the family driver: which conditions an item trains with
(caption, or caption plus conditioning images), how an item is VAE-encoded, the loss, sampling and
decoding. Image families use the driver's defaults.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time

import torch

from ..fsutil import replace
from . import cache
from . import dataset as ds
from . import evaluation
from . import refmod
from . import similarity
from .ema import EMA
from .families import get_family
from .hf_assets import resolve_path
from .lora import LoRANetwork
from .media import save_media
from .memory import decode_bytes
from .offload import ActivationOffload, make_room, park_optimizer, unpark
from .slider import Slider, slider_loss




class StopRequested(Exception):
    pass


_stop_signalled = False


def request_stop() -> None:
    """Called from a signal handler; picked up at the next control check."""
    global _stop_signalled
    _stop_signalled = True


def cond_key(prompt: str, images: list | None) -> str:
    """Cache key for a text condition. Plain prompts key on the prompt itself (so existing caches
    stay valid); conditions with images add a fingerprint of the prepared images (and of any non-image
    marker among them, such as MiniMax H3's audio reference label)."""
    if not images:
        return prompt
    h = hashlib.sha1()
    for im in images:
        if not hasattr(im, "tobytes"):
            h.update(repr(im).encode())
            continue
        h.update(f"{im.size}{im.mode}".encode())
        h.update(im.tobytes())
    return f"{prompt}\x00img:{h.hexdigest()}"


class Trainer:
    def __init__(self, job_dir: str):
        self.job_dir = job_dir
        with open(os.path.join(job_dir, "job.json")) as f:
            self.job = json.load(f)
        self.s = self.job["settings"]
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"No CUDA GPU visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}). "
                "Check the GPU selection in Settings."
            )
        self.device = torch.device("cuda")
        self.text_files: dict[str, str] = {}
        self.opt = None  # the optimizer, once training has one
        self.ema: EMA | None = None  # the average of the LoRA's weights, with ema_enabled
        for sub in ("samples", "output", "state"):
            os.makedirs(os.path.join(job_dir, sub), exist_ok=True)
        self.cache_root = self.job.get("cache_dir") or os.path.join(job_dir, "cache")
        self.cache_dirs: dict[str, str] = {}  # kind -> its folder in the cache, once known
        self.cache_used: set[str] = set()  # the cache files this run uses
        self.status = {"phase": "starting", "step": 0, "total_steps": int(self.s["steps"]), "loss": None,
                       "avg_loss": None, "lr": None, "it_per_sec": None, "eta_seconds": None, "message": "",
                       "samples": [], "checkpoints": [], "started_at": time.time(), "vram_gb": None,
                       "evals": [], "recommended": None, "sweep": None, "likeness": []}
        self.validator: evaluation.Validator | None = None
        self.probe: evaluation.BleedProbe | None = None
        self.likeness: similarity.LikenessMeter | None = None  # scores the samples against the training images
        self.slider = Slider.from_settings(self.s)
        self.finetune = bool(self.s.get("finetune_enabled"))  # the whole model trains, not a LoRA
        self._last_status_write = 0.0
        self._progress_open = False
        old = self.read_status()
        self.evaluating = bool(self.job.get("evaluate"))
        if self.evaluating:
            # the job's status stays as training left it, apart from the phase while this runs and
            # status["evaluation"], which holds what this run finds
            self.status.update(old)
            self.prev_phase, self.prev_message = self.status.get("phase"), self.status.get("message", "")
            self.status["evaluation"] = {"state": "running", "settings": self.job["evaluate"]["settings"],
                                         "evals": [], "recommended": None, "sweep": None,
                                         "started_at": time.time(), "vram_gb": None}
        else:
            if old.get("evaluation"):  # a checkpoint evaluation's results are kept until the job restarts
                self.status["evaluation"] = old["evaluation"]
            if os.path.exists(os.path.join(job_dir, "state", "last.pt")):
                # resuming: what it made so far stays listed if it stops or fails before its state loads (which
                # lists them again from the state)
                for k in ("samples", "checkpoints", "evals", "likeness", "recommended", "sweep"):
                    if k in old:
                        self.status[k] = old[k]
        # where evaluation results go: the job's own status while training, else status["evaluation"]
        self.ev = self.status["evaluation"] if self.evaluating else self.status

    def make_driver(self):
        driver = get_family(self.job["family"])(dict(self.job["profile"]), self.device, log=self.log)
        driver.low_vram = bool(self.s.get("low_vram")) and driver.info.streams_text_encoder
        return driver

    # --- io helpers -------------------------------------------------------
    def log(self, msg: str):
        self._end_progress()
        print(msg, flush=True)
        self.status["message"] = msg
        self.write_status(force=True)

    def progress(self, text: str, keep: bool = False):
        """A status line that the next one overwrites in place (``\\r``), the way a terminal shows a
        progress bar; ``keep`` leaves it in the log as a line of its own."""
        print("\r" + text, end="\n" if keep else "", flush=True)
        self._progress_open = not keep

    def _end_progress(self):
        if self._progress_open:
            print(flush=True)
            self._progress_open = False

    def read_status(self) -> dict:
        try:
            with open(os.path.join(self.job_dir, "status.json")) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def write_status(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_status_write < 1.0:
            return
        self._last_status_write = now
        if torch.cuda.is_available():
            # what PyTorch held at its peak, in GiB like nvidia-smi and the VRAM projection
            self.ev["vram_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        path = os.path.join(self.job_dir, "status.json")
        with open(path + ".tmp", "w") as f:
            json.dump(self.status, f)
        replace(path + ".tmp", path)

    def phase(self, name: str, msg: str = ""):
        self.status["phase"] = name
        self.log(msg or name)

    def read_control(self) -> dict:
        if _stop_signalled:
            return {"action": "stop"}
        path = os.path.join(self.job_dir, "control.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                ctl = json.load(f)
            os.remove(path)
            return ctl
        except (OSError, json.JSONDecodeError):
            return {}

    # --- caching ----------------------------------------------------------
    def cache_dir(self, driver, kind: str) -> str:
        """The shared cache's folder for ``kind`` ("latents" or "text") as this job's model encodes it."""
        if kind not in self.cache_dirs:
            self.cache_dirs[kind] = cache.folder(self.cache_root, driver, kind, self.s)
        return self.cache_dirs[kind]

    def use_cache(self, files):
        """Record cache files this run uses in ``cache.json``, so clearing the cache keeps them while it runs."""
        self.cache_used.update(files)
        path = os.path.join(self.job_dir, "cache.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"root": self.cache_root,
                       "files": sorted(os.path.relpath(p, self.cache_root) for p in self.cache_used)}, f)
        replace(path + ".tmp", path)

    def cache_text(self, driver, conds: dict[str, tuple[str, list | None]]) -> dict[str, str]:
        """Encode text conditions {key: (prompt, images)} to the disk cache; returns key -> file.
        Embeddings are loaded per step rather than all held in RAM."""
        cache_dir = self.cache_dir(driver, "text")
        files, todo = {}, []
        for key, (prompt, images) in conds.items():
            f = os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest() + ".pt")
            files[key] = f
            if not os.path.exists(f):
                todo.append((prompt, images, f))
        self.use_cache(files.values())
        if todo:
            self.phase("caching_text", f"Encoding {len(todo)} captions")
            driver.load_text_encoder()
            for i, (prompt, images, f) in enumerate(todo):
                self.check_control_during_prep()
                # another job may be encoding the same caption: each writes its own temporary file
                tmp = f"{f}.{os.getpid()}.tmp"
                torch.save(driver.encode_condition(prompt, images), tmp)
                replace(tmp, f)
                self.status["message"] = f"Encoding captions {i + 1}/{len(todo)}"
                self.progress(self.status["message"], keep=i + 1 == len(todo))
                self.write_status()
            driver.unload_text_encoder()
        return files

    def text_emb(self, key: str):
        return torch.load(self.text_files[key])

    def cache_items(self, driver, items: list[ds.Item]) -> list[str]:
        """VAE-encode items through the driver; returns one cache file per item."""
        cache_dir = self.cache_dir(driver, "latents")
        files = [os.path.join(cache_dir, it.key() + ".pt") for it in items]
        # repeats share a cache file, so dedupe by file
        todo = [(it, f) for f, it in dict(zip(files, items)).items() if not os.path.exists(f)]
        self.use_cache(files)
        if todo:
            what = ("audio clips" if any(it.is_audio for it, _ in todo)
                    else "clips and images" if any(it.is_video for it, _ in todo) else "images")
            self.phase("caching_latents", f"Encoding {len(todo)} {what} with the VAE")
            driver.load_vae()
            for i, (it, f) in enumerate(todo):
                self.check_control_during_prep()
                tmp = f"{f}.{os.getpid()}.tmp"
                torch.save(driver.encode_item(it, self.s), tmp)
                replace(tmp, f)
                self.status["message"] = f"Encoding {i + 1}/{len(todo)}"
                self.progress(self.status["message"], keep=i + 1 == len(todo))
                self.write_status()
            driver.unload_vae()
        return files

    def item_cond_keys(self, items: list[ds.Item], conds: dict) -> list[dict[str, str]]:
        """Per item {role: cache key}; registers every condition in ``conds``."""
        out, memo = [], {}
        for it in items:
            sig = (it.path, it.flip, it.size, it.frames, it.caption, it.lyrics, tuple(it.refs))
            if sig not in memo:
                roles = {}
                for role, (prompt, images) in self.driver.item_conditions(it, self.s).items():
                    key = cond_key(prompt, images)
                    conds[key] = (prompt, images)
                    roles[role] = key
                memo[sig] = roles
            out.append(memo[sig])
        return out

    def check_control_during_prep(self):
        if self.read_control().get("action") == "stop":
            raise StopRequested()

    # --- optimizer --------------------------------------------------------
    def make_optimizer(self, params):
        name = self.s.get("optimizer", "adamw8bit")
        lr = float(self.s["learning_rate"])
        wd = float(self.s.get("weight_decay", 0.0))
        if self.finetune:  # each weight steps as backward reaches it
            from .finetune import FusedOptimizer
            opt = FusedOptimizer(params, name, lr, wd, device=self.device)
            if self.driver.swapper is not None:  # a streamed block is on the GPU to step, and goes back to RAM after
                opt.before_step = self.driver.swapper.prepare_step
            return opt
        if name == "adamw8bit":
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        if name == "adafactor":
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, weight_decay=wd, relative_step=False, scale_parameter=False,
                             warmup_init=False)
        if name == "prodigy":
            from prodigyopt import Prodigy
            return Prodigy(params, lr=lr, weight_decay=wd)
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def make_scheduler(self, opt):
        total = int(self.s["steps"])
        warm = int(self.s.get("warmup_steps", 0))
        kind = self.s.get("lr_scheduler", "constant")

        def fn(step):
            if warm and step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total - warm)
            if kind == "cosine":
                return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
            if kind == "linear":
                return max(0.0, 1.0 - p)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(opt, fn)

    # --- saving / sampling ------------------------------------------------
    def lora_file(self, step: int, final: bool = False) -> str:
        name = self.job.get("output_name") or self.job["name"]
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        return f"{safe}.safetensors" if final else f"{safe}_{step:06d}.safetensors"

    def keep_final(self):
        """Training on past the end of a finished job: its final LoRA is renamed like the other checkpoints
        (name_002000.safetensors), so the new final one doesn't overwrite it, and results naming it follow.
        It's marked ``kept``: "keep last N" never deletes a LoRA that was a finished result."""
        final = next((c for c in self.status["checkpoints"] if c.get("final")), None)
        if final is None:
            return
        old, new = final["file"], self.lora_file(final["step"])
        out = os.path.join(self.job_dir, "output")
        if os.path.exists(os.path.join(out, old)):
            replace(os.path.join(out, old), os.path.join(out, new))
        final.update(file=new, final=False, kept=True)
        for results in (self.status, self.status.get("evaluation") or {}):
            for e in results.get("evals") or []:
                if e.get("file") == old:
                    e["file"] = new
            for k in ("recommended", "sweep"):
                if (results.get(k) or {}).get("file") == old:
                    results[k]["file"] = new
        self.log(f"Continuing past step {final['step']}: its {'model' if self.finetune else 'LoRA'} is kept as {new}")

    def save_lora(self, network: LoRANetwork, step: int, final: bool = False):
        fname = self.lora_file(step, final)
        path = os.path.join(self.job_dir, "output", fname)
        meta = {} if network.kind == "full" else dict(self.driver.lora_metadata)
        meta.update({"dainamo.job": self.job["name"], "dainamo.step": str(step), "dainamo.network": network.kind,
                     "dainamo.trigger": ", ".join(d.get("trigger", "") for d in self.job["datasets"] if d.get("trigger"))})
        if network.kind == "full":  # a whole diffusion model, finetuned from the profile's
            meta.update({"dainamo.family": self.job["family"],
                         "dainamo.base": os.path.basename(self.job["profile"]["diffusion_model"])})
            self.log(f"Saving the finetuned model at step {step}")
        elif network.kind == "lokr":  # as LyCORIS's kohya-style trainers record it
            args = {"algo": "lokr", "factor": str(network.factor), "full_matrix": "True"}
            meta.update({"ss_network_module": "lycoris.kohya", "ss_network_args": json.dumps(args)})
        else:
            meta.update({"ss_network_dim": str(self.s["rank"]), "ss_network_alpha": str(self.s["alpha"])})
        if self.slider:
            meta.update(self.slider.metadata())
        if self.ema is not None:  # (the weights saved are the average, ``ema_applied``)
            meta["dainamo.ema"] = str(self.ema.decay)
        network.save(path, meta)
        self.status["checkpoints"].append({"step": step, "file": fname, "final": final})
        keep = int(self.s.get("keep_last", 0))
        if keep > 0 and not final:
            inter = [c for c in self.status["checkpoints"] if not c["final"]]
            # the picker's current recommendations (from training, or a checkpoint evaluation) are never deleted
            protected = {(rec or {}).get("file") for rec in
                         (self.status.get("recommended"), (self.status.get("evaluation") or {}).get("recommended"))}
            for c in inter[:-keep]:
                if c["file"] in protected or c.get("kept"):
                    continue
                try:
                    os.remove(os.path.join(self.job_dir, "output", c["file"]))
                except OSError:
                    pass
                self.status["checkpoints"].remove(c)
        self.log(f"Saved {fname}")
        return fname

    def save_state(self, network, opt, sched, step, batches=None):
        """The optimizer and schedule, and the LoRA's weights (the live ones, and their EMA if it's on); a finetune's
        weights are the checkpoint just saved, which ``weights`` names (the whole model would double the save).
        Also where training's randomness is (``rng_state``), so a resumed job draws on instead of replaying."""
        if network.kind == "full":
            weights = {"weights": next(c["file"] for c in reversed(self.status["checkpoints"]) if c["step"] == step)}
        else:
            weights = {"lora": network.state_dict(torch.float32)}
            if self.ema is not None:
                weights["ema"] = self.ema.state_dict()
        # written beside it and then swapped in, so a kill, OOM or full disk mid-save keeps the last good state
        path = os.path.join(self.job_dir, "state", "last.pt")
        torch.save({"step": step, **weights, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "status": self.status, "rng": self.rng_state(batches)}, path + ".tmp")
        replace(path + ".tmp", path)

    def rng_state(self, batches) -> dict | None:
        """Training's random state: the batch draws, caption dropout and slider directions (``train_rng``), the
        noise (torch's generators), and the batches left in this epoch, for the data they index (``data_key``)."""
        if getattr(self, "train_rng", None) is None:
            return None
        return {"python": self.train_rng.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "batches": [list(b) for b in batches or []], "data": self.data_key}

    def restore_rng(self, saved: dict | None, batches: list) -> None:
        """Put back ``rng_state``'s random state (a state saved before it was kept starts afresh, as it did); the
        epoch's batches only for the same data, else a new shuffle starts."""
        if not saved:
            self.log("Note: this job's saved state predates its random state, so the batch order starts afresh")
            return
        self.train_rng.setstate(saved["python"])
        torch.set_rng_state(saved["torch"])
        if torch.cuda.is_available() and len(saved["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved["cuda"])
        if saved.get("data") == self.data_key:
            batches[:] = saved["batches"]
        else:
            self.log("The datasets changed since the last save: the batch order starts a new epoch")

    def sample_specs(self) -> list[dict]:
        """The sample prompt cards with blank fields filled from the Sampling defaults.
        Blank seeds walk: default seed + card number. Plain strings (older jobs) are prompt-only cards."""
        s = self.s
        div = self.driver.bucket_divisibility
        base_w, base_h = int(s.get("sample_width", 1024)), int(s.get("sample_height", 1024))
        base_seed = int(s.get("sample_seed", 42))
        base_frames = int(s.get("sample_frames", 1) or 1)
        base_seconds = float(s.get("sample_seconds", 30) or 30)
        specs = []
        for i, entry in enumerate(s.get("sample_prompts", [])):
            card = {"prompt": entry} if isinstance(entry, str) else dict(entry)
            prompt = str(card.get("prompt") or "").strip()
            if not prompt:
                continue

            def value(key, default, kind):
                v = card.get(key)
                return kind(v) if v not in (None, "") else default

            frames = 1
            if self.driver.supports_video:
                frames = max(1, value("frames", base_frames, int))
                if frames > 1 and self.driver.snap_frames:
                    frames = self.driver.snap_frames(frames)
            images = card.get("images") or []
            if isinstance(images, str):
                images = [p.strip() for p in images.split(",") if p.strip()]
            specs.append({
                "prompt": prompt,
                "width": max(div, value("width", base_w, int) // div * div),
                "height": max(div, value("height", base_h, int) // div * div),
                "seed": value("seed", base_seed + i, int),
                "lora_scale": value("lora_scale", self.driver.info.lora_strength, float),
                "frames": frames,
                "images": images,
            })
            if self.driver.is_audio:
                specs[-1].update(seconds=value("seconds", base_seconds, float), lyrics=str(card.get("lyrics") or ""))
            key = getattr(self, "sample_cond_keys", {}).get(id_of(specs[-1]))
            if key:
                specs[-1]["cond_key"] = key
        slider = getattr(self, "slider", None)
        if slider:  # a slider's cards render at each of its sample strengths instead of their own
            specs = [dict(sp, lora_scale=st) for sp in specs for st in slider.sample_strengths]
        return specs

    @torch.no_grad()
    def render_media(self, model, network, specs: list[dict]) -> list:
        """Render specs with the profile's sampler: PIL images, or video dicts for video families. The
        training adapter is off (it's training-only) and the model is put back in its previous mode.

        The VAE is loaded only to decode (or from the start, to encode conditioning images). In low VRAM
        mode the optimizer state waits in system RAM meanwhile, and so do the diffusion model and LoRA while
        the samples decode, if the GPU hasn't room for the VAE beside them."""
        d = self.driver
        was_training = model.training
        model.eval()
        network.set_assistants_enabled(False)
        prof = self.job["profile"]
        steps = d.sample_steps()
        cfg = d.cfg_scale(d.profile_value(prof, "sample_cfg"))
        negative = self.text_emb(self.negative_prompt)
        lats = []
        conditioned = any(sp.get("images") for sp in specs)
        parked = park_optimizer(self.opt) if d.low_vram and self.opt is not None else []
        try:
            if conditioned:  # conditioning images are VAE-encoded inside the sampler
                d.load_vae()
            try:
                for sp in specs:
                    network.set_multiplier(sp["lora_scale"])
                    cond = self.text_emb(sp.get("cond_key") or sp["prompt"])
                    # each step rewrites the status file (at most once a second): a job asked to stop while it
                    # renders shows the queue it's still working, so isn't killed before it saves
                    lats.append(d.sample(model, cond, negative, sp["width"], sp["height"], steps, cfg,
                                         sp["seed"], frames=sp.get("frames", 1), images=sp.get("images"),
                                         callback=lambda *_: self.write_status()))
            finally:
                network.set_multiplier(1.0)
                network.set_assistants_enabled(True)
                model.train(was_training)
            if d.low_vram:
                need = max(decode_bytes(d, sp["width"], sp["height"], sp.get("frames", 1), loaded=conditioned)
                           for sp in specs)
                moved = make_room(need, [model], d.swapper)
                if moved:
                    self.log("  low VRAM: the diffusion model waits in system RAM while the samples decode")
                parked += moved
            if not conditioned:
                d.load_vae()
            media = [d.to_media(lat)[0] for lat in lats]
            d.unload_vae()
        finally:
            unpark(parked)
        return media

    def save_media(self, media, stem: str) -> dict:
        return save_media(media, os.path.join(self.job_dir, "samples"), stem)

    def make_samples(self, model, network, step: int):
        specs = self.sample_specs()
        if not specs:
            return
        prev = self.status["phase"]
        what = ("song(s)" if self.driver.is_audio
                else "clip(s)" if any(sp["frames"] > 1 for sp in specs) else "image(s)")
        self.phase("sampling", f"Sampling {len(specs)} {what} at step {step}")
        made = []
        for i, (sp, media) in enumerate(zip(specs, self.render_media(model, network, specs))):
            saved = self.save_media(media, f"step{step:06d}_{i}")
            entry = {k: v for k, v in sp.items() if k != "cond_key"}
            if self.likeness:
                entry.update(self.likeness.score(sp, media, step))
            made.append({"step": step, "index": i, **saved, **entry})
        self.status["samples"] += made
        note = ""
        if self.likeness and (summary := similarity.summarise(made)):
            self.status["likeness"] = [x for x in self.status["likeness"] if x["step"] != step] + [{"step": step, **summary}]
            note = f": likeness {summary['likeness']:.0f}%" + (" (a near copy of a training image)" if summary["near_copy"] else "")
        self.phase(prev, f"Samples saved for step {step}{note}")

    # --- checkpoint picker / concept bleed --------------------------------
    def evaluate(self, model, network, step: int, file: str | None, of: str = ""):
        """Score one checkpoint (or, at step 0 with no file, the untrained baseline)."""
        if not (self.validator or self.probe):
            return
        prev = self.status["phase"]
        what = "the base model" if file is None and step == 0 else f"checkpoint at step {step}"
        self.phase("evaluating", f"Evaluating {what}" + (f" ({of})" if of else ""))
        entry: dict = {"step": step, "file": file}
        notes = []
        if self.validator:
            entry["val_loss"] = round(self.validator.loss(model), 6)
            notes.append(f"validation loss {entry['val_loss']:.5f}")
        if self.probe:
            if file is None and step == 0:  # LoRA is still zero: the base model itself
                entry["bleed"], entry["drift"] = 0.0, 0.0
                entry["probes"] = [{"prompt": p, "file": self.probe.base_file(j), "bleed": 0.0, "drift": 0.0}
                                   for j, p in enumerate(self.probe.prompts)]
            else:
                entry.update(self.probe.measure(model, network, step))
            notes.append(f"concept bleed {entry['bleed']:.0f}% (drift {entry['drift']:.0f}%)")
        self.ev["evals"] = [e for e in self.ev["evals"] if e["step"] != step] + [entry]
        self.update_recommendation()
        self.phase(prev, f"Step {step}: " + ", ".join(notes))

    def update_recommendation(self):
        s = self.s
        available = {c["file"] for c in self.status["checkpoints"]}
        before = self.ev.get("recommended")
        # bleed counts while the probe runs: with it off (``make_evaluators``), the checkpoints it doesn't measure
        # would never count as under the limit
        rec = evaluation.recommend(self.ev["evals"], bool(s.get("picker_enabled")),
                                   bool(s.get("bleed_enabled")) and self.probe is not None,
                                   float(s.get("bleed_max", 25)), available)
        self.ev["recommended"] = rec
        if rec and (not before or before.get("file") != rec["file"]):
            self.log(f"Recommended checkpoint: {rec['file']} (step {rec['step']}): {rec['reason']}")

    def strength_sweep(self, model, network):
        """Render the recommended checkpoint at several LoRA strengths for each sample prompt."""
        s = self.s
        rec = self.ev.get("recommended")
        if not (s.get("picker_enabled") and s.get("sweep_enabled") and rec) or s.get("finetune_enabled"):
            return
        raw = s.get("sweep_strengths") or []  # the form's text ("0.6, 0.8") when the picker was off
        strengths = [float(x) for x in (raw.replace(",", " ").split() if isinstance(raw, str) else raw)]
        specs = self.sample_specs()
        if not (specs and strengths):
            return
        prev = self.status["phase"]
        self.phase("sampling", f"Rendering the recommended checkpoint at strengths {strengths}")
        final = network.state_dict(torch.float32)
        network.load(os.path.join(self.job_dir, "output", rec["file"]))
        try:
            grid = [dict(sp, lora_scale=st) for st in strengths for sp in specs]
            media = self.render_media(model, network, grid)
        finally:
            network.load_state_dict(final)  # the resume state must hold the final weights
        out = []
        for k, (sp, m) in enumerate(zip(grid, media)):
            saved = self.save_media(m, f"{'eval' if self.evaluating else ''}sweep_{k // len(specs)}_{k % len(specs)}")
            out.append({"strength": sp["lora_scale"], "index": k % len(specs), "prompt": sp["prompt"], **saved})
        self.ev["sweep"] = {"step": rec["step"], "file": rec["file"], "strengths": strengths, "images": out}
        self.phase(prev, "Strength sweep saved")

    # --- main -------------------------------------------------------------
    def training_datasets(self) -> list[dict]:
        """The job's datasets; a slider job without any trains on the images it renders of its target."""
        if self.job["datasets"] or not self.slider:
            return self.job["datasets"]
        return [{"path": self.slider.folder(self.job_dir), "name": "Rendered images", "trigger": "",
                 "repeats": 1, "resolution": self.s["resolution"]}]

    def resolutions(self) -> list[int]:
        """Every size the datasets train at (each dataset has its own; jobs from before that have the job's)."""
        return sorted({r for d in self.training_datasets()
                       for r in ds.resolutions(d.get("resolution") or self.s["resolution"])})

    def scan(self, driver) -> list[ds.Item]:
        """The training items of ``training_datasets``."""
        s = self.s
        if driver.is_audio:
            items = self.audio_items(self.training_datasets())
            if not items:
                raise RuntimeError("No usable audio found in the selected datasets")
            return items
        res = self.resolutions()
        video_frames = int(s.get("num_frames", 0) or 0) if driver.supports_video else None
        # flip and caption dropout are each dataset's own (jobs from before that have the job's)
        items = ds.build_items(self.training_datasets(), res, driver.bucket_divisibility, bool(s.get("flip", False)),
                               log=self.log, video_frames=video_frames, snap_frames=driver.snap_frames, fps=driver.fps,
                               caption_dropout=float(s.get("caption_dropout", 0.0)),
                               references=driver.trains_references, masks=driver.trains_masks)
        if s.get("masked_loss") and not driver.trains_masks:
            self.log(f"Note: {type(driver).info.label} trains clips or sound, so loss masks (_masks) aren't used")
        if len(res) > 1:
            per_res = {r: sum(1 for it in items if it.resolution == r) for r in res}
            self.log("Training at " + ", ".join(f"{r} ({n} items)" for r, n in per_res.items()))
        if not items:
            raise RuntimeError("No usable training media found in the selected datasets")
        return items

    def audio_items(self, datasets: list[dict], flip: bool | None = None) -> list[ds.Item]:
        s = self.s
        return ds.build_audio_items(datasets, float(s.get("clip_seconds", 30) or 0),
                                    bool(s.get("flip", False)) if flip is None else flip, log=self.log,
                                    caption_dropout=float(s.get("caption_dropout", 0.0)))

    def prepare(self, driver, validation_datasets: list[dict], training: bool = True):
        """Scan the datasets, set the validation items apart, and cache the text conditions and VAE
        payloads that are needed (the training items' own only when ``training``). A slider job that
        still has images to render returns no items: ``run`` renders them once the model is loaded."""
        s = self.s
        self.phase("preparing", "Scanning datasets")
        res = [] if driver.is_audio else self.resolutions()
        div, fps = driver.bucket_divisibility, driver.fps
        video_frames = int(s.get("num_frames", 0) or 0) if driver.supports_video else None
        rendering = (training and self.slider is not None and not self.job["datasets"]
                     and not self.slider.rendered(self.job_dir))
        items = [] if rendering else self.scan(driver)
        val_items: list[ds.Item] = []
        if s.get("picker_enabled"):
            if validation_datasets and driver.is_audio:  # each song's first clip
                val_items = list({it.path: it for it in reversed(self.audio_items(validation_datasets, False))}.values())
            elif validation_datasets:
                # validation loss is measured at one size: the largest
                val_items = list({it.path: it for it in ds.build_items(
                    validation_datasets, res[-1], div, False, log=self.log, video_frames=video_frames,
                    snap_frames=driver.snap_frames, fps=fps, references=driver.trains_references,
                    masks=driver.trains_masks)}.values())
                overlap = {it.path for it in val_items} & {it.path for it in items}
                if overlap:
                    self.log(f"Note: {len(overlap)} validation item(s) are also training items; the picker "
                             "then measures fit rather than generalisation")
            else:
                items, val_items = evaluation.split_holdout(items, int(s.get("val_holdout", 2)), int(s.get("seed", 42)))
            if not val_items:
                raise RuntimeError("The checkpoint picker found no validation items")
            self.log(f"Checkpoint picker: {len(val_items)} validation item(s)"
                     + ("" if validation_datasets else ", held out from training"))
        if training and items:
            for line in bucket_summary(items):
                self.log(line)

        # text conditions: item captions (some with conditioning images; a slider's prompts instead),
        # sample cards, bleed probes, "" (caption dropout) and the negative prompt (sampling's CFG branch)
        conds: dict[str, tuple[str, list | None]] = {}
        item_keys = self.item_cond_keys(items, conds) if training and not self.slider else []
        val_keys = self.item_cond_keys(val_items, conds)
        for prompts in self.slider.prompt_sets() if self.slider else ():
            for p in prompts.values():
                conds[p] = (p, None)
        self.sample_cond_keys = {}
        for sp in self.sample_specs():
            prompt, images = driver.sample_condition(sp)
            key = cond_key(prompt, images)
            conds[key] = (prompt, images)
            self.sample_cond_keys[id_of(sp)] = key
        self.negative_prompt = str(self.job["profile"].get("sample_negative") or "")
        bleed_prompts = [p for p in s.get("bleed_prompts") or [] if p.strip()] if s.get("bleed_enabled") else []
        for p in bleed_prompts + ["", self.negative_prompt]:
            conds[p] = (p, None)
        self.text_files = self.cache_text(driver, conds)
        payload_files = self.cache_items(driver, items) if training else []
        val_payload_files = self.cache_items(driver, val_items) if val_items else []
        return items, item_keys, payload_files, val_keys, val_payload_files, bleed_prompts

    def load_model(self, driver, path: str | None = None):
        """The diffusion model, from the profile's file or (a finetune carrying on) the checkpoint at ``path``."""
        self.phase("loading", "Loading diffusion model")
        driver.offload = float(self.s.get("offload", driver.info.training_defaults.get("offload", 0)) or 0) / 100
        if self.finetune:  # trained whole, in bf16 (its streamed blocks trained as they stream)
            if path:
                driver.profile["diffusion_model"] = path
            return driver.load_transformer(trainable=True)
        return driver.load_transformer()

    def render_slider_images(self, driver, model, network) -> tuple[list[ds.Item], list[str]]:
        """A slider job without datasets trains on images of its target prompt, rendered here by the base model
        with the profile's sampler, square at the largest training resolution. They're kept in
        ``slider_images/``, so a resumed job renders only the ones it's missing. -> (items, payload files)."""
        sl, s = self.slider, self.s
        div = driver.bucket_divisibility
        size = max(div, self.resolutions()[-1] // div * div)
        seed = int(s.get("seed", 42))
        todo = [i for i in range(sl.images) if not os.path.exists(sl.image_path(self.job_dir, i))]
        if todo:
            os.makedirs(sl.folder(self.job_dir), exist_ok=True)
            self.phase("rendering", f"Rendering {len(todo)} training image(s) of \"{sl.target}\" with the base model")
            for k in range(0, len(todo), 4):  # the VAE is loaded once per 4 images
                self.check_control_during_prep()
                chunk = todo[k:k + 4]
                specs = [{"prompt": sl.target, "width": size, "height": size, "seed": seed + i, "lora_scale": 0.0,
                          "frames": 1, "images": []} for i in chunk]
                for i, img in zip(chunk, self.render_media(model, network, specs)):
                    path = sl.image_path(self.job_dir, i)
                    img.save(path + ".tmp", format="PNG")
                    os.replace(path + ".tmp", path)
                done = min(k + 4, len(todo))
                self.status["message"] = f"Rendering {done}/{len(todo)}"
                self.progress(self.status["message"], keep=done == len(todo))
                self.write_status()
        items = self.scan(driver)
        for line in bucket_summary(items):
            self.log(line)
        return items, self.cache_items(driver, items)

    def make_network(self, driver, model) -> LoRANetwork:
        """The job's LoRA on ``model`` (zero, so the base model, until trained or loaded), with the
        profile's training adapter attached."""
        s = self.s
        network = LoRANetwork(model, driver.lora_targets, int(s["rank"]), float(s["alpha"]),
                              float(s.get("dropout", 0.0)), kind=s.get("network_type") or "lora",
                              factor=int(s.get("lokr_factor", -1)))
        network.to(self.device, torch.float32)
        what = f"LoKr (factor {network.factor})" if network.kind == "lokr" else "LoRA"
        self.log(f"{what}: {len(network.modules)} layers, {network.num_parameters() / 1e6:.1f}M trainable params")
        adapter = (self.job["profile"].get("training_adapter") or "").strip()
        if adapter:
            self.log(f"Attaching training adapter {adapter}")
            n = network.attach_assistant(resolve_path(adapter), self.device)
            self.log(f"  adapter applied to {n} layers")
        return network

    def make_ema(self, network, saved: dict | None) -> EMA | None:
        """The average of the LoRA's weights (``engine.ema``) when ``ema_enabled``: carried on from the resume state's
        ``saved``, else started from the weights as they are (also a job whose EMA an edit turned on). None for a
        finetune: its weights are the whole model."""
        if not self.s.get("ema_enabled"):
            return None
        if self.finetune:
            self.log("EMA weights aren't kept for a finetune (its weights are the whole model): it saves the trained "
                     "weights as they are")
            return None
        ema = EMA(list(network.parameters()), float(self.s.get("ema_decay", 0.99)))
        if saved:
            ema.load_state_dict(saved)
        what = "LoKr" if network.kind == "lokr" else "LoRA"
        self.log(f"EMA of the {what} weights (decay {ema.decay:g}): saves, samples and evaluations use the averaged "
                 "weights" + (f", carried on from {ema.num_updates} steps" if saved else ""))
        return ema

    def ema_applied(self):
        """Saves, samples and evaluations inside this see the EMA weights, if it's on; training's live weights are
        put back after."""
        return self.ema.applied() if self.ema is not None else contextlib.nullcontext()

    def make_full(self, driver, model):
        """A finetune's stand-in for the LoRA: the model's own weights (``finetune.FullModel``), with the profile's
        training adapter beside them."""
        from .finetune import FullModel, TrainingAdapter

        # saves are laid out like the profile's file (or, if it's gone, the checkpoint it carries on from)
        source = self.job["profile"]["diffusion_model"]
        adapter = None
        path = (self.job["profile"].get("training_adapter") or "").strip()
        if path:
            self.log(f"Attaching training adapter {path}")
            adapter = TrainingAdapter(model, resolve_path(path), self.device)
            self.log(f"  adapter applied to {len(adapter.handles)} layers")
        network = FullModel(model, source if os.path.exists(source) else driver.profile["diffusion_model"],
                            swapper=driver.swapper, adapter=adapter)
        n = network.num_parameters()
        self.log(f"Full finetune: every weight trains, " + (f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M")
                 + f" params ({self.s.get('optimizer')}, each weight updated as backward reaches it)")
        # a save is written beside the last one before "keep last N" deletes that
        size, free = n * 2, shutil.disk_usage(os.path.join(self.job_dir, "output")).free
        if free < 2 * size:
            self.log(f"Warning: {free / 1e9:.0f} GB free on the disk the saves go to, and each is about "
                     f"{size / 1e9:.0f} GB: a save may fail, and with it stopping the job")
        return network

    def make_evaluators(self, driver, model, network, items, val_keys, val_payload_files, bleed_prompts,
                        prefix: str = "bleed", base_weights: bool = True):
        """The checkpoint picker's validator and the concept-bleed probe, as the settings ask. Without
        ``base_weights`` (a resumed finetune's are a checkpoint's), the probe can't render the base model, so it runs
        only if its references from step 0 are still there."""
        s, div = self.s, driver.bucket_divisibility
        if s.get("picker_enabled"):
            self.validator = evaluation.Validator(
                driver, val_payload_files, lambda i: {r: self.text_emb(k) for r, k in val_keys[i].items()}, s,
                int(s.get("val_timesteps", 8)), int(s.get("seed", 42)), evaluation.latent_hw(val_payload_files[0]))
        if bleed_prompts:
            self.probe = evaluation.BleedProbe(
                bleed_prompts, int(s.get("bleed_width", 512)) // div * div, int(s.get("bleed_height", 512)) // div * div,
                int(s.get("bleed_seed", 1234)), os.path.join(self.job_dir, "samples"), self.render_media, self.log,
                prefix=prefix)
            if not (base_weights or self.probe.has_references()):
                self.log("Concept bleed: no baseline images from step 0, so the probe is off for this resumed finetune")
                self.probe = None
            else:
                self.probe.setup(model, network, self.device, [it.path for it in items])
        self.make_likeness(driver, model, network, items)

    def make_likeness(self, driver, model, network, items):
        """The likeness meter (``similarity.LikenessMeter``), scoring each sample against the training images, as
        ``likeness_enabled`` asks (on by default): for pictures only, and not for a slider (it has no dataset to look
        like) or a checkpoint evaluation. A finetune can render the base model only at step 0, before training."""
        s = self.s
        if (not s.get("likeness_enabled", True) or self.evaluating or self.slider or driver.is_audio
                or s.get("refmod_enabled") or not items):
            return
        self.likeness = similarity.LikenessMeter(
            self.device, os.path.join(self.job_dir, "state", "likeness.pt"),
            lambda spec: self.render_media(model, network, [spec])[0], self.log,
            can_render_base=not self.finetune)
        self.likeness.setup([it.path for it in items])

    def run_evaluation(self) -> str:
        """Score a finished job's saved checkpoints, as asked from its page (``job["evaluate"]``): the base
        model, then each checkpoint, with the checkpoint picker and/or concept bleed settings chosen then,
        and the strength sweep if it's on. Results go to status["evaluation"], apart from any the job made
        while training; the job's own phase is put back at the end."""
        ev = self.job["evaluate"]
        if self.finetune:
            raise RuntimeError("A finetune's checkpoints are whole models, which a checkpoint evaluation doesn't load")
        self.s = s = {**self.s, **ev["settings"]}
        out = os.path.join(self.job_dir, "output")
        ckpts = sorted((c for c in self.status["checkpoints"] if os.path.exists(os.path.join(out, c["file"]))),
                       key=lambda c: c["step"])
        if not ckpts:
            raise RuntimeError("The job has no saved checkpoints left to evaluate")
        torch.manual_seed(int(s.get("seed", 42)))
        driver = self.driver = self.make_driver()
        self.log(f"Checkpoint evaluation: the base model and {len(ckpts)} saved checkpoint(s)")
        # a new evaluation replaces the last one, images included
        samples = os.path.join(self.job_dir, "samples")
        for f in os.listdir(samples):
            if f.startswith(("evalbleed_", "evalsweep_")):
                os.remove(os.path.join(samples, f))
        try:
            items, _, _, val_keys, val_payload_files, bleed_prompts = self.prepare(
                driver, ev.get("validation_datasets") or [], training=False)
            model = self.load_model(driver)
            network = self.make_network(driver, model)
            self.make_evaluators(driver, model, network, items, val_keys, val_payload_files, bleed_prompts,
                                 prefix="evalbleed")
            self.evaluate(model, network, 0, None)  # the LoRA is still zero: the base model
            for n, c in enumerate(ckpts, 1):
                self.check_control_during_prep()
                network.load(os.path.join(out, c["file"]))
                self.evaluate(model, network, c["step"], c["file"], f"{n} of {len(ckpts)}")
            self.check_control_during_prep()
            self.strength_sweep(model, network)
        except StopRequested:
            self.end_evaluation("stopped")
            return "stopped"
        self.end_evaluation("completed")
        return "completed"

    def run_refmod(self) -> str:
        """Make a RefMod (``engine.refmod``) of the datasets' images and clips with only the video VAE loaded, or
        (``refmod_kind`` audio) of their sound with only the audio VAE: output/<name>.safetensors, copied to
        ``refmod_copy_to`` if set, and a preview in samples/ (``refmod.save_preview``; or the sound, decoded). status["refmod"] describes it."""
        s = self.s
        driver = self.driver = self.make_driver()
        if not driver.info.refmods:
            raise RuntimeError(f"{driver.info.label} doesn't make RefMods")
        sound = s.get("refmod_kind") == "audio"
        paths = (refmod.audio_sources if sound else refmod.sources)(self.job["datasets"])
        if not paths:
            raise RuntimeError("No sound files or clips with sound found in the selected datasets" if sound
                               else "No images or clips found in the selected datasets")
        self.phase("loading", "Loading the audio VAE" if sound else "Loading the video VAE")
        driver.load_vae(audio=sound, video=not sound)
        mode = "full" if sound or s["refmod_mode"] == "full" else "compressed"
        self.phase("encoding", f"Making an audio RefMod of up to {s['refmod_seconds']:g} s of sound" if sound
                   else f"Making a {mode} RefMod of {len(paths)} reference(s)")
        make = refmod.make_audio if sound else refmod.make
        latent, meta = make(driver, paths, s, log=self.log, check=self.check_control_during_prep)
        fname = self.lora_file(0, final=True)
        meta["name"] = fname[: -len(".safetensors")]
        out = refmod.save(os.path.join(self.job_dir, "output", fname), latent, meta)
        if sound:
            self.log(f"Saved {fname}: {latent.shape[-1] / refmod.AUDIO_PER_SECOND:.1f} s of sound, "
                     f"{refmod.tokens(latent)} tokens, {os.path.getsize(out) / 1e6:.1f} MB")
        else:
            t, h, w = latent.shape[2:]
            self.log(f"Saved {fname}: {t} latent frame(s) of {h}×{w}, {refmod.tokens(latent)} tokens, "
                     f"{os.path.getsize(out) / 1e6:.1f} MB")
        copy_to = str(s.get("refmod_copy_to") or "").strip()
        if copy_to:
            import shutil

            try:
                os.makedirs(os.path.expanduser(copy_to), exist_ok=True)
                shutil.copyfile(out, os.path.join(os.path.expanduser(copy_to), fname))
                self.log(f"Copied to {copy_to}")
            except OSError as e:
                self.log(f"Warning: couldn't copy it to {copy_to}: {e}")
        self.status["checkpoints"] = [{"step": 0, "file": fname, "final": True}]
        made = {"file": fname, "kind": meta["kind"], "mode": mode, "references": len(paths),
                "tokens": refmod.tokens(latent), "bytes": os.path.getsize(out), "copied_to": copy_to or None}
        if sound:
            self.phase("encoding", "Decoding the sound to listen to")
            with torch.no_grad():
                wav = driver.audio_vae.decode(latent[0].permute(1, 0, 2).to(self.device))[:, 0]  # (2, samples)
            made.update(seconds=round(latent.shape[-1] / refmod.AUDIO_PER_SECOND, 2), concept=meta["concept_type"],
                        audio=save_media({"audio": wav.float().cpu().numpy(), "sample_rate": refmod.AUDIO_RATE},
                                         os.path.join(self.job_dir, "samples"), "refmod_audio"),
                        frames=0, height=0, width=0, preview=[])
            return self._refmod_done(made)
        self.phase("encoding", f"Decoding a preview of {min(t, refmod.PREVIEW_FRAMES)} frame(s)")
        preview = refmod.save_preview(driver, latent, os.path.join(self.job_dir, "samples"),
                                      self.check_control_during_prep)
        made.update(frames=int(t), height=int(h), width=int(w), preview=preview)
        return self._refmod_done(made)

    def _refmod_done(self, made: dict) -> str:
        self.status["refmod"] = made
        self.status["phase"] = "completed"
        self.log(f"Done in {(time.time() - self.status['started_at']) / 60:.1f} min")
        return "completed"

    def extract_source(self) -> str:
        """The finetuned model a LoRA is extracted from: the source job's checkpoint at the step it was chosen at
        (a final one is renamed when that job trains on), else the file as it was named then."""
        src = self.s.get("extract_from") or {}
        if src.get("job_id"):
            # its folder: named after it, or (from before folders were named) its id
            folder = os.path.join(os.path.dirname(os.path.abspath(self.job_dir)), src.get("folder") or src["job_id"])
            try:
                with open(os.path.join(folder, "status.json")) as f:
                    ckpts = json.load(f).get("checkpoints") or []
                c = next((c for c in ckpts if c.get("step") == src.get("step")), None)
                if c:
                    return os.path.join(folder, "output", c["file"])
            except (OSError, json.JSONDecodeError):
                pass
        return self.s["extract_source"]

    def run_extract(self) -> str:
        """Extract a LoRA from a finetune (``engine.extract``): the change from ``extract_base`` (the model it was
        finetuned from, the profile's) to the finetuned model (``extract_source``), as output/<name>.safetensors.
        Nothing is trained and no model is loaded: the two files are compared a block at a time on the GPU.
        status["extract"] describes what it saved."""
        from . import extract

        s = self.s
        Family = get_family(self.job["family"])
        source = self.extract_source()
        base = s.get("extract_base") or self.job["profile"]["diffusion_model"]
        for what, path in (("finetuned model", source), ("model it was finetuned from", base)):
            if not os.path.isfile(path):
                raise RuntimeError(f"The {what}, {path}, is gone")
        with contextlib.suppress(Exception):
            from safetensors import safe_open

            with safe_open(source, "pt") as f:
                was = (f.metadata() or {}).get("dainamo.base")
            if was and was != os.path.basename(base):
                self.log(f"Warning: {os.path.basename(source)} was finetuned from {was}, not {os.path.basename(base)}")
        rank = int(s["extract_rank"])
        src = s.get("extract_from") or {}
        fname = self.lora_file(0, final=True)
        meta = {**Family.lora_metadata, "dainamo.job": self.job["name"], "dainamo.network": "lora",
                "dainamo.step": str(src.get("step", 0)), "dainamo.trigger": src.get("trigger", ""),
                "dainamo.extracted_from": os.path.basename(source), "dainamo.base": os.path.basename(base),
                "ss_network_dim": str(rank), "ss_network_alpha": str(rank)}
        self.phase("extracting", f"Extracting a rank {rank} LoRA: what {os.path.basename(source)} changed from "
                                 f"{os.path.basename(base)}")

        def progress(done: int, total: int):
            self.status.update(step=done, total_steps=total, message=f"Compared {done} of {total} parts")
            self.progress(self.status["message"], keep=done == total)
            self.write_status()

        made = extract.extract(base, source, os.path.join(self.job_dir, "output", fname), rank, Family.block_list,
                               self.device, meta, log=self.log, progress=progress,
                               check=self.check_control_during_prep)
        self.log(f"Saved {fname}: {made['layers']} layers as rank {rank} LoRAs and {made['whole']} changes kept whole "
                 f"({made['unchanged']} tensors unchanged), {made['bytes'] / 1e6:.0f} MB. It keeps {made['kept']:.1%} "
                 f"of the change, which moved the weights by {made['change']:.2%}.")
        self.status.update(step=0, total_steps=0)
        self.status["checkpoints"] = [{"step": int(src.get("step", 0)), "file": fname, "final": True}]
        self.status["extract"] = {"file": fname, **made, "source": os.path.basename(source),
                                  "base": os.path.basename(base)}
        self.status["phase"] = "completed"
        self.log(f"Done in {(time.time() - self.status['started_at']) / 60:.1f} min")
        return "completed"

    def run_merge(self) -> str:
        """Merge LoRAs into one (``engine.merge``): ``merge_loras`` [{path, strength}], kept exact (``merge_rank``
        0) or cut to that rank, as output/<name>.safetensors. Nothing is trained and no model is loaded: the
        LoRAs' layers are checked against the profile's model built without its weights, and merged a layer at a
        time on the GPU. status["merge"] describes what it saved."""
        from . import merge

        s = self.s
        Family = get_family(self.job["family"])
        loras = [(e["path"], float(e.get("strength", 1.0))) for e in s["merge_loras"]]
        for path, _ in loras:
            if not os.path.isfile(path):
                raise RuntimeError(f"{path} is gone")
        rank = int(s.get("merge_rank") or 0)
        fname = self.lora_file(0, final=True)
        triggers = [t for e in s["merge_loras"] for t in [e.get("trigger", "")] if t]
        meta = {**Family.lora_metadata, "dainamo.job": self.job["name"], "dainamo.network": "lora",
                "dainamo.step": "0", "dainamo.trigger": ", ".join(dict.fromkeys(triggers)),
                "dainamo.merged_from": json.dumps([{"file": os.path.basename(p), "strength": st} for p, st in loras])}
        if rank:
            meta.update(ss_network_dim=str(rank), ss_network_alpha=str(rank))
        what = " + ".join(f"{os.path.basename(p)} × {st:g}" for p, st in loras)
        self.phase("merging", f"Merging {what}" + (f", cut to rank {rank}" if rank else ", exactly"))
        layers = merge.model_layers(Family, self.job["profile"])

        def progress(done: int, total: int):
            self.status.update(step=done, total_steps=total, message=f"Merged {done} of {total} layers")
            self.progress(self.status["message"], keep=done == total)
            self.write_status()

        made = merge.merge(loras, os.path.join(self.job_dir, "output", fname), rank, layers, self.device, meta,
                           progress=progress, check=self.check_control_during_prep)
        for i in made["inputs"]:
            if i["matched"] < i["layers"]:
                self.log(f"Warning: {i['file']} changes {i['layers']} layers, {i['layers'] - i['matched']} of which "
                         "this model doesn't have (at that size); they're left out")
        worst = made["worst"]
        self.log(f"Saved {fname}: {made['layers']} layers as LoRAs"
                 + (f" ({made['cut']} cut to rank {rank})" if rank else "")
                 + f" and {made['whole']} changes kept whole, {made['bytes'] / 1e6:.0f} MB. It keeps "
                 f"{made['kept']:.1%} of the merged change"
                 + (f"; least in {worst['name']}, {worst['kept']:.1%}" if worst else "") + ".")
        self.status.update(step=0, total_steps=0)
        self.status["checkpoints"] = [{"step": 0, "file": fname, "final": True}]
        self.status["merge"] = {"file": fname, **made}
        self.status["phase"] = "completed"
        self.log(f"Done in {(time.time() - self.status['started_at']) / 60:.1f} min")
        return "completed"

    def end_evaluation(self, state: str, message: str = ""):
        """Record how a checkpoint evaluation ended, and give the job back the phase and message it had."""
        self.ev.update(state=state, message=message, finished_at=time.time())
        rec = self.ev.get("recommended")
        self.log(f"Checkpoint evaluation {state}" + (f": {message}" if message else "")
                 + (f"; recommended {rec['file']} (step {rec['step']})" if rec and state == "completed" else ""))
        self.status.update(phase=self.prev_phase, message=self.prev_message)
        self.write_status(force=True)

    def run(self):
        s = self.s
        seed = int(s.get("seed", 42))
        torch.manual_seed(seed)
        rng = self.train_rng = random.Random(seed)
        driver = self.driver = self.make_driver()
        items, item_keys, payload_files, val_keys, val_payload_files, bleed_prompts = self.prepare(
            driver, self.job.get("validation_datasets") or [])

        state_path = os.path.join(self.job_dir, "state", "last.pt")
        # a finetune's saved state is small beside its model (on the CPU until the optimizer takes it)
        st = torch.load(state_path, weights_only=False, map_location="cpu" if self.finetune else None) \
            if os.path.exists(state_path) else None
        resume_from = None
        if st is not None and self.finetune:
            if not st.get("weights"):
                raise RuntimeError("This job's saved state isn't a finetune's; restart it from step 0")
            resume_from = os.path.join(self.job_dir, "output", st["weights"])
            if not os.path.exists(resume_from):
                raise RuntimeError(f"The checkpoint this finetune carries on from, {st['weights']}, is gone; "
                                   "restart it from step 0")
        model = self.load_model(driver, resume_from)
        if driver.swapper is not None and not s.get("gradient_checkpointing", True):
            if self.finetune:  # (the server doesn't queue one)
                raise RuntimeError("Offloading a finetune needs gradient checkpointing: turn it on, or offload 0%")
            self.log("Note: layer offloading saves little VRAM in training without gradient checkpointing, "
                     "since every block's weights stay saved for the backward pass")
        if s.get("gradient_checkpointing", True):
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing(True)
            else:
                model.gradient_checkpointing = True
        network = self.make_full(driver, model) if self.finetune else self.make_network(driver, model)

        params = list(network.parameters())
        for p in params:
            p.requires_grad_(True)
        opt = self.opt = self.make_optimizer(params)
        # low VRAM: the blocks' saved activations wait in system RAM during each forward and backward pass
        saved = ActivationOffload(self.device) if driver.low_vram and s.get("gradient_checkpointing", True) \
            and torch.cuda.is_available() else None
        offloaded = saved.step if saved else contextlib.nullcontext
        sched = self.make_scheduler(opt)

        start = 0
        ema_saved = rng_saved = None
        if st is not None:
            if not self.finetune:  # a finetune's weights were loaded with the model
                network.load_state_dict(st["lora"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            # the learning rate and weight decay as the job has them now (an edit may have changed them), which the
            # saved state would otherwise put back to what it started with; the schedule follows the settings anyway
            lr, wd = float(s["learning_rate"]), float(s.get("weight_decay", 0.0))
            sched.base_lrs = [lr for _ in opt.param_groups]
            for g in opt.param_groups:
                g.update(initial_lr=lr, weight_decay=wd, lr=lr * sched.lr_lambdas[0](sched.last_epoch))
            start = st["step"]
            for k in ("samples", "checkpoints", "evals", "likeness"):
                self.status[k] = st["status"].get(k, [])
            for k in ("recommended", "sweep"):
                self.status[k] = st["status"].get(k)
            self.log(f"Resumed from step {start}")
            if start < int(s["steps"]):
                self.keep_final()  # a finished job given more steps
            ema_saved, rng_saved = st.get("ema"), st.get("rng")
            del st
        # after the resume state, so a new EMA starts from the weights the job carries on with
        self.ema = self.make_ema(network, ema_saved)

        if not items:  # a slider job without datasets renders its training images first
            items, payload_files = self.render_slider_images(driver, model, network)
        self.make_evaluators(driver, model, network, items, val_keys, val_payload_files, bleed_prompts,
                             base_weights=resume_from is None)
        slider_conds = [{k: self.text_emb(p) for k, p in prompts.items()} for prompts in self.slider.prompt_sets()] \
            if self.slider else []
        if self.slider:
            sl = self.slider
            self.log(f"Slider: \"{sl.target}\" toward \"{sl.positive}\" (+) or \"{sl.negative}\" (-), "
                     f"strength {sl.strength:g}" + (f"; preserving {', '.join(sl.preserve)}" if sl.preserve else "")
                     + (f"; layout kept above noise {sl.layout_sigma:g}" if sl.layout_sigma is not None else ""))

        total = int(s["steps"])
        bs = int(s.get("batch_size", 1))
        # a finetune's weights update during backward: one batch per step, and no global clipping
        accum = 1 if self.finetune else max(1, int(s.get("grad_accum", 1)))
        max_norm = 0.0 if self.finetune else float(s.get("max_grad_norm", 1.0))
        save_every = int(s.get("save_every", 0))
        sample_every = int(s.get("sample_every", 0))
        metrics = open(os.path.join(self.job_dir, "metrics.jsonl"), "a")
        keep_every = max(1, total // 50)  # about 50 progress lines stay in the log; the rest update in place

        if start == 0:
            with self.ema_applied():  # (the EMA is the initial weights yet)
                self.evaluate(model, network, 0, None)
                if s.get("sample_at_start", True):
                    self.make_samples(model, network, 0)
                elif self.likeness and self.finetune:  # its last chance to render the base model
                    self.likeness.prime(self.sample_specs())

        self.phase("training", "Training")
        model.train()
        batches: list[list[int]] = []
        # the data the batches index: its latents' files, in order
        self.data_key = hashlib.sha1("\n".join(map(str, payload_files)).encode()).hexdigest()
        if start:
            self.restore_rng(rng_saved, batches)
        ema_loss = None
        t0, step_times = time.time(), []
        step = start
        told = False  # logged what the activation offload holds
        try:
            while step < total:
                ctl = self.read_control()
                action = ctl.get("action")
                if action == "stop":
                    raise StopRequested()
                if action == "sample":
                    with self.ema_applied():
                        self.make_samples(model, network, step)
                if action == "save":
                    with self.ema_applied():
                        self.evaluate(model, network, step, self.save_lora(network, step))

                ts = time.time()
                loss_acc = 0.0
                for _ in range(accum):
                    if not batches:
                        batches = ds.make_batches(items, bs, rng)
                    idxs = batches.pop()
                    payloads = [torch.load(payload_files[i]) for i in idxs]
                    if self.slider:
                        direction = rng.choice((1.0, -1.0))
                        # the prompts as given, or with one of the preserved attributes
                        conds = slider_conds[0] if len(slider_conds) == 1 else rng.choice(slider_conds)
                        # the LoRA stays at that strength until backward has rerun the checkpointed blocks
                        network.set_multiplier(direction)
                        try:
                            with offloaded():
                                loss = slider_loss(driver, model, network, payloads, conds, s, direction,
                                                   self.slider.strength, self.slider.layout_sigma)
                                (loss / accum).backward()
                        finally:
                            network.set_multiplier(1.0)
                    else:
                        batch_conds = []
                        for i in idxs:
                            if rng.random() < items[i].caption_dropout:
                                batch_conds.append({"text": self.text_emb(""), "dropped": True})
                            else:
                                batch_conds.append({r: self.text_emb(k) for r, k in item_keys[i].items()})
                        with offloaded():
                            loss = driver.training_loss(model, payloads, batch_conds, s, rng=rng)
                            (loss / accum).backward()
                    loss_acc += loss.item() / accum
                    if saved and saved.last_bytes and not told:
                        told = True
                        self.log(f"  low VRAM: {saved.last_bytes / 1e9:.1f} GB of saved activations per step wait "
                                 "in system RAM")
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
                opt.step()
                if self.ema is not None:
                    self.ema.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if self.finetune and driver.swapper is not None:  # the last block backward ran goes back to RAM
                    driver.swapper.release_all()
                step += 1

                if not math.isfinite(loss_acc):
                    raise RuntimeError(f"Loss became {loss_acc} at step {step}; try a lower learning rate")
                ema_loss = loss_acc if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_acc
                step_times.append(time.time() - ts)
                step_times = step_times[-20:]
                spi = sum(step_times) / len(step_times)
                lr = sched.get_last_lr()[0]
                self.status.update(step=step, loss=round(loss_acc, 5), avg_loss=round(ema_loss, 5), lr=lr,
                                   it_per_sec=round(1 / spi, 3), eta_seconds=int(spi * (total - step)))
                metrics.write(json.dumps({"step": step, "loss": loss_acc, "lr": lr}) + "\n")
                metrics.flush()
                self.write_status()
                self.progress(progress_line(step, total, loss_acc, ema_loss, spi, lr),
                              keep=step % keep_every == 0 or step == total)

                if save_every and step % save_every == 0 and step < total:
                    with self.ema_applied():
                        fname = self.save_lora(network, step)
                        self.evaluate(model, network, step, fname)
                    self.save_state(network, opt, sched, step, batches)  # (the live weights, and the EMA beside them)
                if sample_every and step % sample_every == 0 and step < total:
                    with self.ema_applied():
                        self.make_samples(model, network, step)
        except StopRequested:
            if step > start:  # nothing new to save if no step ran in this session
                if not any(c["step"] == step for c in self.status["checkpoints"]):  # (else just saved)
                    with self.ema_applied():
                        self.save_lora(network, step)
                self.save_state(network, opt, sched, step, batches)
            self.status["phase"] = "stopped"
            self.log(f"Stopped at step {step}" + ("; state saved for resume" if step > start else ""))
            return "stopped"
        finally:
            metrics.close()

        with self.ema_applied():
            self.evaluate(model, network, step, self.save_lora(network, step, final=True))
            if s.get("sample_at_end", True):
                self.make_samples(model, network, step)
            self.strength_sweep(model, network)
        self.save_state(network, opt, sched, step, batches)
        self.status["phase"] = "completed"
        self.log(f"Done in {(time.time() - t0) / 60:.1f} min")
        return "completed"


def fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def progress_line(step: int, total: int, loss: float, avg: float, sec_per_step: float, lr: float) -> str:
    """e.g. "step  120/2000 █░░░░░░░░░░░░░░░░░░░   6%  loss 0.1234 (avg 0.1500)  8.41 s/step  lr 1.00e-04  ETA 4h 23m"."""
    width = 20
    filled = width * step // total if total else width
    return (f"step {step:>{len(str(total))}}/{total} {'█' * filled}{'░' * (width - filled)} {100 * step // max(total, 1):3d}%"
            f"  loss {loss:.4f} (avg {avg:.4f})  {sec_per_step:.2f} s/step  lr {lr:.2e}"
            f"  ETA {fmt_eta(sec_per_step * (total - step))}")


def bucket_summary(items: list[ds.Item]) -> list[str]:
    """Log lines on the training sizes: how many buckets, the range and the most used, then how many
    files train well below their resolution because they're smaller (``ds.trains_small``). Audio: the
    clip lengths."""
    if items and items[0].is_audio:
        lengths: dict[float, int] = {}
        for it in items:
            lengths[it.seconds] = lengths.get(it.seconds, 0) + 1
        songs = len({it.path for it in items})
        return [f"{len(items)} training item{'' if len(items) == 1 else 's'} from {songs} "
                f"song{'' if songs == 1 else 's'}, "
                + ", ".join(f"{n} of {sec:g} s" for sec, n in sorted(lengths.items(), key=lambda kv: -kv[1]))]
    def name(size, frames):
        return f"{size[0]}x{size[1]}" + (f"x{frames}f" if frames > 1 else "")

    counts: dict[tuple, int] = {}
    for it in items:
        counts[(it.size, it.frames)] = counts.get((it.size, it.frames), 0) + 1
    by_area = sorted(counts, key=lambda b: (b[0][0] * b[0][1], b))  # by picture size; clips name their frames
    line = (f"{len(items)} training item{'' if len(items) == 1 else 's'} in {len(counts)} "
            f"bucket size{'' if len(counts) == 1 else 's'}")
    if len(counts) > 1:
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        line += (f", from {name(*by_area[0])} to {name(*by_area[-1])}; the most used: "
                 + ", ".join(f"{name(*b)} ({n})" for b, n in top))
    else:
        line += f": {name(*by_area[0])}"
    lines = [line]
    small: dict[int, set] = {}
    for it in items:
        if it.small:
            small.setdefault(it.resolution, set()).add(it.path)
    if small:
        total = len({it.path for it in items})
        parts = [f"{len(paths)} of {total} files train below {res}" for res, paths in sorted(small.items(), reverse=True)]
        lines.append("; ".join(parts) + ": they're smaller than that, and files are never upscaled "
                     "(the dataset's Health check lists them)")
    return lines


def id_of(spec: dict) -> str:
    """Stable identity of a sample card (its position-independent content)."""
    keys = ("prompt", "width", "height", "frames", "images", "seconds", "lyrics")
    return json.dumps({k: spec[k] for k in keys if k in spec}, sort_keys=True)

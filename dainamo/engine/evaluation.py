"""Optional checkpoint evaluation: a best-checkpoint picker and a concept-bleed detector.

Validation loss
    Images held out from training (or a separate dataset) are scored at every checkpoint with the
    same noise and the same timesteps each time, so the curve is smooth enough to see where the
    LoRA starts to overfit. The timesteps are quantiles of the family's own training distribution,
    which keeps the number comparable with the training loss.

Concept bleed
    Prompts without the trigger word are rendered with a fixed seed by the base model and by each
    checkpoint, and compared in DINOv2 feature space with the training images:
      bleed = how far a probe image moved toward the training set, as a share of the gap between
              the base model's image and the training images themselves (0% = no closer to your
              dataset than the base model's image, 100% = as close as your own images are)
      drift = how much the probe image changed at all, relative to the change a new random seed
              causes (0% = identical, 100% = as different as another seed). Drift alone isn't
              leakage: a LoRA can nudge a composition while the image stays on-prompt.
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

from . import dataset as ds

DINO_REPO = "facebook/dinov2-small"
MAX_CONCEPT_IMAGES = 64  # training images used to describe the concept
DEFAULT_BLEED_PROMPTS = [
    "a portrait photo of a woman with short dark hair, natural window light",
    "a mountain lake at sunrise, wide landscape photo",
    "a bowl of fruit on a wooden kitchen table",
    "a watercolor painting of a city street in the rain",
]


def split_holdout(items: list[ds.Item], count: int, seed: int) -> tuple[list[ds.Item], list[ds.Item]]:
    """Hold ``count`` whole images (all their repeats/flips/resolutions) out of training, chosen
    reproducibly; each is validated once, at the largest resolution it has. Clips cut from one video
    (``Item.source``) are held out together, so none of the video trains, and its first clip is validated."""
    def whole(it):  # an image or clip, or the video a split dataset's clip was cut from
        return it.source or it.path

    keys = sorted({whole(it) for it in items})
    if count >= len(keys):
        raise ValueError(f"Can't hold out {count} image(s) for validation: the datasets only have {len(keys)}")
    held = set(random.Random(seed).sample(keys, count))
    train = [it for it in items if whole(it) not in held]
    val = {}
    for it in items:
        k = whole(it)
        if k in held and not it.flip and (k not in val or (it.path, -it.resolution) < (val[k].path, -val[k].resolution)):
            val[k] = it
    return train, list(val.values())


def latent_hw(payload_file: str) -> tuple[int, int]:
    """Spatial size of the main latent in a cached payload (for resolution-dependent timestep shifts)."""
    payload = torch.load(payload_file)
    if isinstance(payload, torch.Tensor):
        return tuple(payload.shape[-2:])
    for key in ("latent", "video"):
        if key in payload:
            return tuple(payload[key].shape[-2:])
    t = next(v for v in payload.values() if isinstance(v, torch.Tensor) and v.dim() >= 3)
    return tuple(t.shape[-2:])


def recommend(evals: list[dict], picker: bool, bleed: bool, max_bleed: float, available: set[str]) -> dict | None:
    """Pick the checkpoint to use. Only checkpoints whose files still exist are candidates."""
    cands = [e for e in evals if e.get("file") in available]
    if not cands:
        return None
    under = [e for e in cands if e.get("bleed") is not None and e["bleed"] <= max_bleed] if bleed else []
    if picker:
        pool = under if under else cands
        best = min(pool, key=lambda e: e["val_loss"])
        reason = "Lowest validation loss"
        if bleed:
            reason += f" among checkpoints under {max_bleed:g}% bleed" if under else \
                f"; every checkpoint is over the {max_bleed:g}% bleed limit"
    elif under:
        best = max(under, key=lambda e: e["step"])
        reason = f"The most-trained checkpoint under {max_bleed:g}% bleed"
    else:
        best = min(cands, key=lambda e: e["bleed"])
        reason = f"Lowest bleed; every checkpoint is over the {max_bleed:g}% limit"
    return {"step": best["step"], "file": best["file"], "reason": reason}


class Validator:
    """Deterministic validation loss on a fixed set of items, noise and timesteps, computed with the
    family's own ``training_loss`` (so video and audio families are scored the way they train)."""

    def __init__(self, driver, payload_files: list[str], cond_for, settings: dict, timesteps: int, seed: int,
                 latent_hw: tuple[int, int]):
        self.driver, self.payload_files, self.cond_for, self.settings = driver, payload_files, cond_for, settings
        self.seed = seed
        # quantiles of the family's training distribution, drawn reproducibly
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            draws = driver.sample_timesteps(8192, latent_hw, settings).double()
        q = (torch.arange(timesteps, dtype=torch.float64) + 0.5) / timesteps
        self.sigmas = torch.quantile(draws, q).clamp(1e-3, 1.0).tolist()

    @torch.no_grad()
    def loss(self, model) -> float:
        was_training = model.training
        model.eval()
        total, n = 0.0, 0
        for i, f in enumerate(self.payload_files):
            payload = torch.load(f)
            cond = self.cond_for(i)
            for k, sigma in enumerate(self.sigmas):
                g = torch.Generator(device="cpu").manual_seed(self.seed * 7919 + i * 101 + k)
                total += self.driver.training_loss(model, [payload], [cond], self.settings, sigma=sigma,
                                                   generator=g, deterministic=True).item()
                n += 1
        model.train(was_training)
        return total / n


class DinoEmbedder:
    """DINOv2-small image features (downloaded once, ~88 MB), loaded only while measuring."""

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def __call__(self, images: list[Image.Image]) -> torch.Tensor:
        from transformers import AutoImageProcessor, AutoModel

        from .hf_assets import from_pretrained

        proc = from_pretrained(AutoImageProcessor, DINO_REPO, "DINOv2 (for concept bleed)")
        model = from_pretrained(AutoModel, DINO_REPO, "DINOv2 (for concept bleed)").to(self.device).eval()
        inputs = proc(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
        emb = F.normalize(model(**inputs).pooler_output.float(), dim=-1).cpu()
        del model
        torch.cuda.empty_cache()
        return emb


class BleedProbe:
    """Renders trigger-free prompts and measures how far each checkpoint moves them from the base model."""

    def __init__(self, prompts: list[str], width: int, height: int, seed: int, out_dir: str, render, log,
                 prefix: str = "bleed"):
        self.prompts, self.out_dir, self.render, self.log = prompts, out_dir, render, log
        self.prefix = prefix  # of its image files: "bleed" in training, "evalbleed" in a checkpoint evaluation
        self.specs = [{"prompt": p, "width": width, "height": height, "seed": seed + j, "lora_scale": 1.0}
                      for j, p in enumerate(prompts)]
        self.embed = None
        self.base = None       # (P, D) base-model features
        self.variation = None  # (P,) feature distance base(seed) -> base(other seed)
        self.concept = None    # (D,) normalised mean feature of the training images
        self.base_sim = None   # (P,) cos(base image, concept)
        self.train_sim = None  # mean cos(training image, concept)
        os.makedirs(out_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    def base_file(self, j: int) -> str:
        return f"{self.prefix}_base_{j}.png"

    def setup(self, model, network, device, train_paths: list[str]):
        """Base-model references (rendered once, reused from disk when a job resumes) and the
        training-set concept they're measured against."""
        self.embed = DinoEmbedder(device)
        base_files = [self._path(self.base_file(j)) for j in range(len(self.specs))]
        alt_files = [self._path(f"{self.prefix}_alt_{j}.png") for j in range(len(self.specs))]
        if not all(os.path.exists(f) for f in base_files + alt_files):
            self.log(f"Concept bleed: rendering {len(self.specs)} reference prompt(s) with the base model")
            base = [dict(sp, lora_scale=0.0) for sp in self.specs]
            alt = [dict(sp, lora_scale=0.0, seed=sp["seed"] + 100_000) for sp in self.specs]
            for img, f in zip(self.render(model, network, base + alt), base_files + alt_files):
                img.save(f)
        base_imgs = [Image.open(f) for f in base_files]
        alt_imgs = [Image.open(f) for f in alt_files]
        e = self.embed(base_imgs + alt_imgs)
        n = len(self.specs)
        self.base = e[:n]
        self.variation = (1 - (e[:n] * e[n:]).sum(-1)).clamp_min(1e-4)
        paths = sorted(set(train_paths))
        if len(paths) > MAX_CONCEPT_IMAGES:
            paths = random.Random(0).sample(paths, MAX_CONCEPT_IMAGES)
        train = []
        for f in paths:
            if os.path.splitext(f)[1].lower() in ds.VIDEO_EXTS:  # a clip stands in as its first frame
                train.append(ds.first_frame(f, (448, 448), 24))
                continue
            with Image.open(f) as im:
                train.append(ds.to_rgb(im))
        t = self.embed(train)
        self.concept = F.normalize(t.mean(0), dim=-1)
        self.train_sim = (t @ self.concept).mean().item()
        self.base_sim = self.base @ self.concept

    def measure(self, model, network, step: int) -> dict:
        images = self.render(model, network, self.specs)
        files = []
        for j, img in enumerate(images):
            name = f"{self.prefix}_step{step:06d}_{j}.png"
            img.save(self._path(name))
            files.append(name)
        e = self.embed(images)
        drift = (100 * (1 - (e * self.base).sum(-1)) / self.variation).tolist()
        gap = (self.train_sim - self.base_sim).clamp_min(1e-3)
        bleed = (100 * ((e @ self.concept) - self.base_sim) / gap).clamp_min(0).tolist()
        return {
            "bleed": round(sum(bleed) / len(bleed), 1),
            "drift": round(sum(drift) / len(drift), 1),
            "probes": [{"prompt": p, "file": f, "bleed": round(b, 1), "drift": round(d, 1)}
                       for p, f, b, d in zip(self.prompts, files, bleed, drift)],
        }

"""Model-family driver interface.

The training loop is family-agnostic; everything model-specific lives in a
driver: which files a profile needs, how to load them, how to encode text and
images, the noise schedule, the forward pass, and sampling.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass
class ProfileField:
    key: str
    label: str
    kind: str = "path"  # path | text | number | bool | select
    required: bool = True
    help: str = ""
    default: Any = None
    options: list[str] | None = None
    hint: str = ""  # path fields: the model the file must be, e.g. "Qwen3-VL-4B" (any precision or file name)


@dataclass
class Download:
    """A file for a profile's path field that the model manager offers to download from Hugging Face, from a pinned
    commit so it's the file described here. Only files Dainamo can read are listed (no w4a8, GPTQ, w6a8 or mxfp8)."""

    field: str  # the profile field it fills
    repo: str
    path: str  # in the repo; its folder is the ComfyUI models folder it goes in
    size: int  # bytes
    label: str  # what tells it from the field's other files, e.g. "Turbo · int8 convrot"
    revision: str
    sha256: str = ""  # checked once it's downloaded; "" where the repo doesn't publish it (gated repos)
    folder: str = ""  # the ComfyUI models folder, for a file at the repo's top level
    gated: bool = False  # Hugging Face wants you signed in, having accepted the model's terms
    recommended: bool = False  # the one to take when unsure (one per field and variant)
    variant: str = ""  # the profile variant it's for ("": any)
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.repo}/{self.path}"

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def kind_folder(self) -> str:
        return self.folder or self.path.split("/")[0]

    def to_json(self) -> dict:
        return {"id": self.id, "field": self.field, "repo": self.repo, "path": self.path, "name": self.name,
                "size": self.size, "label": self.label, "folder": self.kind_folder, "gated": self.gated,
                "recommended": self.recommended, "variant": self.variant, "note": self.note,
                "url": f"https://huggingface.co/{self.repo}"}


@dataclass
class MemoryProfile:
    """Working memory of a family's models for the VRAM projection (``engine.memory``), measured with
    ``scripts/measure_memory.py``. Per-token values count every token of the sequence (image, text
    and conditioning)."""
    saved: float  # bytes per token per block kept for backward, with gradient checkpointing
    full: float  # bytes per token per block kept for backward, without it
    work: float  # bytes per token: one block's forward + backward working set, plus the outer layers
    infer: float  # bytes per token: working set of an inference forward pass
    vae_encode: tuple[float, float]  # (bytes, bytes per pixel per frame; audio: per latent frame) to VAE-encode
    vae_decode: tuple[float, float]  # the same to decode
    decode_tile_pixels: int | None = None  # decodes above this many pixels are tiled at this size
    encode_tile_pixels: int | None = None  # the same for encodes
    text_tokens: int = 256  # caption length the projection assumes
    text_work: float = 0.8  # GB the text encoder works in while it encodes, on top of its weights


def count_blocks(shapes: dict, block_list: str) -> int:
    """Number of ``<block_list>.<i>.`` blocks in a {key: shape} map (``block_list`` may be dotted)."""
    return 1 + max(block_index(k, block_list) for k in shapes if k.startswith(block_list + "."))


def block_index(key: str, block_list: str) -> int:
    """The block number of a ``<block_list>.<i>.…`` key."""
    return int(key[len(block_list) + 1:].split(".", 1)[0])


def transformer_shapes(path: str) -> tuple[str, dict[str, list[int]]]:
    """(prefix, {key: shape}) of a diffusion-model file without reading weights; ``prefix`` is
    "model.diffusion_model." for a full ComfyUI checkpoint (stripped from the keys), else ""."""
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        full = any(k.startswith("model.diffusion_model.") for k in keys)
        prefix = "model.diffusion_model." if full else ""
        return prefix, {k[len(prefix):]: list(f.get_slice(k).get_shape()) for k in keys if k.startswith(prefix)}


# What a Sandbox generation takes and makes. Families list the workflows they support.
WORKFLOWS = {
    "t2i": {"label": "Text-to-Image", "output": "image"},
    "edit": {"label": "Image Edit", "output": "image"},
    "t2v": {"label": "Text-to-Video", "output": "video"},
    "i2v": {"label": "Image-to-Video", "output": "video"},
    "r2v": {"label": "Reference-to-Video", "output": "video"},
    "t2a": {"label": "Text-to-Music", "output": "audio"},
}


@dataclass
class Workflow:
    id: str  # a WORKFLOWS key
    # conditioning images it takes: {"label", "min", "max", "help"}; "help" also explains the dataset's
    # _refs folder, and the sample cards of a job use the same input
    images: dict | None = None
    variants: list[str] | None = None  # profile variants it applies to (None: every variant)
    refmods: bool = False  # takes RefMods (``engine.refmod``) as references, as well as or instead of images
    # reference sound files it takes, for videos: {"label", "max", "help"}; with audio RefMods, at most "max" in all
    audio: dict | None = None

    def to_json(self) -> dict:
        return {"id": self.id, **WORKFLOWS[self.id], "images": self.images, "variants": self.variants,
                "refmods": self.refmods, "audio": self.audio}


@dataclass
class FamilyInfo:
    id: str
    label: str
    media: list[str]  # "image" and/or "video", or "audio"
    profile_fields: list[ProfileField]
    training_defaults: dict[str, Any] = field(default_factory=dict)
    # (id, label) pairs offered for settings.timestep_sampling; the first is the default
    timestep_modes: list[tuple[str, str]] = field(default_factory=lambda: [
        ("shift", "Resolution shift"), ("logit_normal", "Logit-normal + shift"), ("uniform", "Uniform")])
    workflows: list[Workflow] = field(default_factory=lambda: [Workflow("t2i")])
    frame_rule: str = ""  # video families: the clip lengths the VAE takes, e.g. "8n + 1 (9, 17, 25, …)"
    sliders: bool = False  # trains concept sliders (``engine.slider``) through slider_batch / slider_velocity
    # makes RefMods (``engine.refmod``): saved reference latents, through ``encode_reference``
    refmods: bool = False
    # the strength its LoRAs are meant to be used at; training samples, the Sandbox and the strength sweep start
    # there (YuE2's train without the composer's tokens, so they act at about half strength with them)
    lora_strength: float = 1.0
    # "Low VRAM" streams the text encoder's layers from system RAM (``FamilyDriver.stream_text_encoder``);
    # off for YuE2, whose composer writes songs a token at a time, far too often to stream
    streams_text_encoder: bool = True
    # trains as a full finetune (``engine.finetune``) as well as adapters: the settings a finetune starts from
    # (its learning rate and optimizer, how many saves to keep), or None
    finetune: dict[str, Any] | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "media": self.media,
            "profile_fields": [f.__dict__ for f in self.profile_fields],
            "training_defaults": self.training_defaults,
            "timestep_modes": [list(m) for m in self.timestep_modes],
            "workflows": [w.to_json() for w in self.workflows],
            "frame_rule": self.frame_rule,
            "sliders": self.sliders,
            "refmods": self.refmods,
            "lora_strength": self.lora_strength,
            "streams_text_encoder": self.streams_text_encoder,
            "finetune": self.finetune,
        }


class FamilyDriver:
    info: FamilyInfo
    # spatial multiple training images are cropped to (VAE factor * patch)
    bucket_divisibility: int = 16
    # fnmatch patterns of Linear module names to train
    lora_targets: list[str] = []
    # video families: snaps a requested clip length to a frame count the VAE accepts
    snap_frames = None
    fps: int = 24

    # attribute of the diffusion model holding its repeated blocks (the unit offloading streams)
    block_list: str = "blocks"
    # working memory for the VRAM projection; None: the projection counts weights only
    memory: MemoryProfile | None = None
    # text-encoder keys that are never loaded (substrings), for the projection
    text_encoder_skip: tuple[str, ...] = ("lm_head.",)
    # text-encoder keys loaded in fp32 rather than bf16 (substrings; a vision tower), for the projection
    text_encoder_fp32: tuple[str, ...] = ()
    # caption tokens the model reads; longer captions are cut off there (None: no limit)
    caption_tokens: int | None = None
    # key prefixes of the diffusion-model file that load with the text encoder instead (LTX's text
    # connectors): skipped when the diffusion model loads, and counted with the text encoder in the projection
    text_encoder_parts: tuple[str, ...] = ()
    # parts of a single-file checkpoint that load as the text encoder or VAE, for the projection:
    # {"text_encoder" | "vae": (key prefix, bytes per weight once loaded)}
    bundled: dict[str, tuple[str, int]] = {}
    # raised when how the family encodes items or captions changes, so the shared cache encodes them again
    cache_version: int = 1

    def __init__(self, profile: dict, device: torch.device, log=print):
        self.profile = profile
        self.device = device
        self.log = log
        self.dtype = torch.bfloat16
        # share of the diffusion model's blocks kept in system RAM and streamed in (0-1); set by
        # the trainer or the Sandbox before load_transformer
        self.offload = 0.0
        self.swapper = None
        # "Low VRAM": the text encoder loads to system RAM and its layers stream to the GPU as they run, and
        # samples are made in two steps (the trainer and the Sandbox see to that); set by them before loading
        self.low_vram = False
        self.text_swappers = []
        # optional fn(text) a family calls during long work that isn't sampling steps (YuE2's composer writing
        # the song), to show it; it may raise to cancel. The Sandbox sets it.
        self.progress_note = None

    # --- text -------------------------------------------------------------
    def load_text_encoder(self): ...
    def encode_prompt(self, prompt: str) -> torch.Tensor: ...

    @property
    def text_encoder_device(self) -> torch.device:
        """Where load_text_encoder reads the text encoder's weights to: system RAM in low VRAM mode, after which
        ``stream_text_encoder`` moves what isn't streamed to the GPU."""
        return torch.device("cpu") if self.low_vram and self.info.streams_text_encoder else self.device

    def stream_text_encoder(self, model: nn.Module) -> nn.Module:
        """In low VRAM mode, stream the layers of a text encoder loaded to ``text_encoder_device`` and move the
        rest of it to the GPU (``offload.stream_layers``); otherwise it's already on the GPU. Drop the loaded
        state dict first: its tensors would otherwise stay in RAM beside the pinned copies."""
        if self.text_encoder_device != self.device:
            from ..offload import stream_layers
            self.text_swappers += stream_layers(model, self.device, self.log)
        return model

    def unload_text_encoder(self):
        while self.text_swappers:  # (a loop variable would keep the last one's pinned RAM until this returns)
            self.text_swappers.pop().remove()
        self.text_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch._C._host_emptyCache()  # streamed layers' pinned RAM, which PyTorch would otherwise keep

    # --- vae --------------------------------------------------------------
    def load_vae(self): ...
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels (1, 3, H, W) in [-1, 1] -> latents (C, h, w)"""
        ...
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """latents (B, C, h, w) -> pixels (B, 3, H, W) in [-1, 1]"""
        ...
    def unload_vae(self): ...

    # --- transformer --------------------------------------------------------
    def build_transformer(self, shapes: dict[str, list[int]]) -> nn.Module:
        """The diffusion model on the meta device, sized from its tensor shapes (keys without a full
        checkpoint's prefix). Builds as many blocks as ``shapes`` has, so memory measurement can build
        a few."""
        raise NotImplementedError

    def transformer_dtype_for(self, key: str) -> torch.dtype | None:
        """A dtype other than ``self.dtype`` for an unquantized diffusion-model tensor, or None."""
        return None

    def check_transformer_state(self, path: str, loaded) -> None:
        """Adjust or reject the loaded diffusion-model tensors before they're assigned."""

    @classmethod
    def sequence_tokens(cls, width: int, height: int, frames: int = 1) -> int:
        """Tokens the diffusion model sees for one training item or sample of this size."""
        div = cls.bucket_divisibility
        text = cls.memory.text_tokens if cls.memory else 256
        return (width // div) * (height // div) + text

    def streamed_blocks(self, num_blocks: int) -> int:
        # halves round up, as in the web UI (Python's round() would round them to even)
        return max(0, min(num_blocks, math.floor(float(self.offload or 0) * num_blocks + 0.5)))

    def load_transformer(self, trainable: bool = False) -> nn.Module:
        """Load the diffusion model, keeping quantized layers quantized. With ``offload`` set, the
        last blocks are read to system RAM and streamed in as they run (``self.swapper``);
        everything else is read straight to the GPU, so the whole model never sits in RAM at once.
        ``trainable`` (a full finetune): quantized layers are expanded to bf16 instead, and the streamed blocks are
        trained as they stream (``offload.BlockSwapper``'s ``trainable``)."""
        import gc

        from .. import comfy_io
        from ..fp8_linear import assert_materialized, materialize
        from ..offload import BlockSwapper

        path = self.profile["diffusion_model"]
        self.log(f"Loading diffusion model {path}")
        prefix, shapes = transformer_shapes(path)
        model = self.build_transformer(shapes)
        blocks = model.get_submodule(self.block_list)
        n = len(blocks)
        k = self.streamed_blocks(n)
        streamed = tuple(f"{self.block_list}.{i}." for i in range(n - k, n))
        opts = {"strip_prefix": prefix or None, "only_prefixed": bool(prefix),
                "cast_to": lambda key: self.transformer_dtype_for(key) or self.dtype}
        loaded = comfy_io.load_state_dict(path, skip_prefixes=streamed + self.text_encoder_parts, device=self.device,
                                          **opts)
        if streamed:
            rest = comfy_io.load_state_dict(path, include=streamed, device="cpu", **opts)
            loaded.tensors.update(rest.tensors)
            loaded.quant.update(rest.quant)
            del rest
        self.check_transformer_state(path, loaded)
        kinds = sorted({q.label for q in loaded.quant.values()})
        n_quant = len(loaded.quant)
        materialize(model, loaded, self.dtype, keep_fp8=not trainable, device=None, dtype_for=self.transformer_dtype_for)
        del loaded
        gc.collect()
        assert_materialized(model)
        if trainable and torch.cuda.is_available():
            torch.cuda.empty_cache()  # what expanding the quantized layers took, which training would otherwise keep
        if trainable and n_quant:
            self.log(f"  {n} blocks; {n_quant} quantized layers ({', '.join(kinds)}) expanded to bf16 to train. "
                     "Training from the bf16 release is better: a quantized file's rounding is trained in.")
        else:
            self.log(f"  {n} blocks; " + (f"{n_quant} layers kept quantized ({', '.join(kinds)})" if n_quant
                                          else "bf16 weights"))
        self.swapper = BlockSwapper(blocks, k, self.device, self.log, trainable=trainable) if k else None
        return model

    def sample_timesteps(self, batch: int, latent_hw: tuple[int, int], settings: dict) -> torch.Tensor: ...
    def predict(self, model: nn.Module, noisy: torch.Tensor, t: torch.Tensor, text: list[torch.Tensor]) -> torch.Tensor: ...
    def loss_target(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - latents  # flow-matching velocity

    # --- sampling -----------------------------------------------------------
    # Image families give the starting noise, a velocity function and their default schedule; the
    # sampler and scheduler are generic (engine.sampling). A family with its own sampling loop
    # (MiniMax's joint video + audio) sets supports_samplers = False and overrides ``sample``.
    supports_samplers = True

    @classmethod
    def profile_value(cls, profile: dict, key: str, fallback=None):
        """A profile setting, or its field's default when the profile leaves it blank (as the profile form
        would fill it in), so training samples and the Sandbox agree on, e.g., the sample steps."""
        v = profile.get(key)
        if v not in (None, ""):
            return v
        field = next((f for f in cls.info.profile_fields if f.key == key), None)
        return field.default if field is not None and field.default is not None else fallback

    def sample_steps(self) -> int:
        return int(self.profile_value(self.profile, "sample_steps", 20))

    def cfg_scale(self, profile_cfg) -> float:
        """The profile's "Sampling CFG" as a standard guidance scale (1 = off)."""
        return float(profile_cfg) if profile_cfg not in (None, "") else 1.0

    def default_shift(self, width: int, height: int) -> float:
        """The shift the model's own schedule applies at this size (used by the simple/beta schedulers)."""
        return 1.0

    def default_sigmas(self, steps: int, width: int, height: int) -> list[float]:
        from ..sampling import make_sigmas
        return make_sigmas(self, "simple", steps, width, height, self.default_shift(width, height))

    def prepare_sampling(self, model: nn.Module, cond, uncond, width: int, height: int, cfg: float, seed: int,
                         images: list | None = None):
        """-> (starting noise, velocity(x, sigma)); ``cfg`` is a standard scale (1 = off)."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, model: nn.Module, cond, uncond, width: int, height: int, steps: int, cfg: float, seed: int,
               frames: int = 1, images: list | None = None, sampler: str = "euler", scheduler: str = "default",
               shift: float | None = None, callback=None):
        """Returns latents for ``to_media``. ``cfg`` is a standard scale (see ``cfg_scale``)."""
        from .. import sampling
        x, velocity = self.prepare_sampling(model, cond, uncond, width, height, cfg, seed, images)
        sigmas = sampling.make_sigmas(self, scheduler, int(steps), width, height, shift)
        return sampling.run(sampler, x, sigmas, velocity, callback)

    # --- generic hooks (image families use these defaults) -----------------
    @property
    def supports_video(self) -> bool:
        return "video" in self.info.media

    @property
    def is_audio(self) -> bool:
        """Trains on audio files alone (music): items are clips of songs, samples are songs."""
        return "audio" in self.info.media

    def item_conditions(self, item, settings: dict) -> dict[str, tuple[str, list | None]]:
        """Text conditions an item may train with: {role: (prompt, conditioning images or None)}."""
        return {"text": (item.caption, None)}

    def sample_condition(self, spec: dict) -> tuple[str, list | None]:
        """(prompt, conditioning images) for a sample card."""
        return spec["prompt"], None

    def encode_condition(self, prompt: str, images: list | None):
        return self.encode_prompt(prompt)

    def cache_inputs(self, kind: str, settings: dict) -> dict:
        """What the cached ``kind`` ("latents": ``encode_item``'s payloads, "text": ``encode_condition``'s) is made
        from: ``files``, the profile's files whose weights encode it, and any setting or profile value that
        changes it. Jobs whose inputs match share their encodings (``engine.cache``)."""
        return {"files": [self.profile.get("vae" if kind == "latents" else "text_encoder")]}

    def encode_item(self, item, settings: dict) -> dict:
        """VAE-encode one dataset item (VAE loaded). The payload is cached and passed to training_loss."""
        from .. import dataset as ds
        return {"latent": self.encode_image(ds.load_and_crop(item.path, item.size, item.flip))}

    def training_loss(self, model: nn.Module, payloads: list, conds: list[dict], settings: dict, rng=None,
                      sigma: float | None = None, generator: torch.Generator | None = None,
                      deterministic: bool = False) -> torch.Tensor:
        """Flow-matching MSE on a batch. ``sigma``/``generator`` pin the noise level and noise
        (validation); otherwise both are random."""
        lat = torch.stack([p["latent"] if isinstance(p, dict) else p for p in payloads]).to(self.device, torch.float32)
        text = [c["text"] for c in conds]
        if generator is not None:
            noise = torch.randn(lat.shape, generator=generator).to(self.device)
        else:
            noise = torch.randn_like(lat)
        if sigma is not None:
            t = torch.full((lat.shape[0],), float(sigma), device=self.device)
        else:
            t = self.sample_timesteps(lat.shape[0], lat.shape[-2:], settings).to(self.device)
        tb = t.view(-1, *([1] * (lat.dim() - 1)))
        noisy = (1 - tb) * lat + tb * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.predict(model, noisy, t, text)
        return F.mse_loss(pred.float(), self.loss_target(lat, noise).float())

    # --- concept sliders (``engine.slider``; families with info.sliders) ----
    def slider_batch(self, payloads: list, settings: dict) -> dict:
        """Training images noised at random timesteps, for ``slider_velocity`` (captions and any reference
        images of the payloads aren't used)."""
        lat = torch.stack([p["latent"] if isinstance(p, dict) else p for p in payloads]).to(self.device, torch.float32)
        t = self.sample_timesteps(lat.shape[0], lat.shape[-2:], settings).to(self.device)
        tb = t.view(-1, *([1] * (lat.dim() - 1)))
        return {"noisy": (1 - tb) * lat + tb * torch.randn_like(lat), "t": t}

    def slider_velocity(self, model: nn.Module, batch: dict, cond) -> torch.Tensor:
        """The model's velocity (fp32) for every image of a ``slider_batch`` under one text condition."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.predict(model, batch["noisy"], batch["t"], [cond] * batch["noisy"].shape[0]).float()

    def to_media(self, latents) -> list:
        """Decoded samples (VAE loaded): PIL images, or dicts {frames, audio, fps, sample_rate} for video."""
        out = []
        for img in self.decode_latents(latents):
            arr = ((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
            out.append(Image.fromarray(arr))
        return out

    lora_metadata: dict[str, str] = {}

    # tensor names expected in each path field, used to catch a wrong file early
    signature_keys: dict[str, list[str]] = {}
    # optional leading dims of signature tensors, for files that share key names with other models'
    # (e.g. Qwen3-VL 4B vs 8B): {field: {key suffix: (dim0, dim1, ...)}}
    signature_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    # the files the model manager can download for the path fields
    downloads: list[Download] = []

    @classmethod
    def matches(cls, field: str, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> bool:
        """Whether a file with these tensors ({key: (dtype, shape)}, ``memory.tensors``) has the signature of
        ``field``'s model (True for a field without one)."""
        for suffix in cls.signature_keys.get(field, []):
            if not any(k.endswith(suffix) for k in tensors):
                return False
        for suffix, want in cls.signature_shapes.get(field, {}).items():
            key = next((k for k in tensors if k.endswith(suffix)), None)
            if key is None:
                return False
            dtype, shape = tensors[key]
            shape = list(shape)
            if dtype == "U8" and len(shape) == 2:
                shape[1] *= 2  # nvfp4: two codes per byte
            if tuple(shape[: len(want)]) != tuple(want):
                return False
        return True

    @classmethod
    def validate_profile(cls, profile: dict) -> list[dict]:
        """Returns [{field, level: error|warning|ok, message}] without loading weights."""
        import os
        from ..comfy_io import describe_quant
        from ..memory import tensors

        out = []
        for f in cls.info.profile_fields:
            if f.kind != "path":
                continue
            val = (profile.get(f.key) or "").strip()
            if not val:
                if f.required:
                    out.append({"field": f.key, "level": "error", "message": "Required"})
                continue
            if not os.path.exists(os.path.expanduser(val)):
                is_hub = not val.startswith(("/", "~", ".")) and val.count("/") >= 2
                out.append({"field": f.key, "level": "warning" if is_hub else "error",
                            "message": "Downloads from Hugging Face on first use, into its cache, where ComfyUI "
                                       "doesn't look" if is_hub else "File not found"})
                continue
            try:
                keys = tensors(val)
            except Exception as e:
                out.append({"field": f.key, "level": "error", "message": f"Not a readable .safetensors file: {e}"})
                continue
            if not cls.matches(f.key, keys):
                want = f.hint or f"a {f.label.lower()} for {cls.info.label}"
                out.append({"field": f.key, "level": "error", "message": f"Wrong file? Expected {want}"})
                continue
            labels, unsupported = describe_quant(os.path.expanduser(val))
            if unsupported:
                out.append({"field": f.key, "level": "error",
                            "message": f"Uses quantization Dainamo can't read yet: {', '.join(sorted(unsupported))}"})
                continue
            msg = f"{len(keys)} tensors" + (f" · {', '.join(sorted(labels))} (ComfyUI)" if labels else "")
            out.append({"field": f.key, "level": "ok", "message": msg})
        # a Turbo (distilled) model trained without its adapter loses its few-step sampling
        dm = os.path.basename((profile.get("diffusion_model") or "").strip()).lower()
        if ("turbo" in dm and any(f.key == "training_adapter" for f in cls.info.profile_fields)
                and not (profile.get("training_adapter") or "").strip()):
            out.append({"field": "training_adapter", "level": "warning",
                        "message": "This looks like a Turbo model: without its training adapter, the LoRA breaks "
                                   "Turbo's few-step sampling"})
        return out

"""SeedVR2 as an ``engine.upscale.Upscaler``: the reference's one-step restoration (``projects/
inference_seedvr2_3b.py`` and ``projects/video_diffusion_sr/infer.py`` in SeedVR at e4de8c2, Apache License 2.0).

For each clip (or a part of a long one, ``max_tokens``): resize the frames to the asked size (bicubic,
antialiased, as torchvision's ``resize``), pad them to a multiple of 16 (the reference crops instead; padding keeps
the asked size), scale to [-1, 1], repeat the last frame up to 4n + 1 frames, encode them with the VAE (a posterior
sample), and take one Euler step from pure noise at t = 1000 with the DiT conditioned on those latents and the
fixed text embedding: x0 = noise - v. CFG is 1 in the release's inference, so the negative embedding never runs.
Decode, then the wavelet color fix the reference applies (the output's high frequencies over the resized input's
low ones), clip, and drop the padding.
"""

from __future__ import annotations

import gc
import math
import os
from typing import Callable

import torch
import torch.nn.functional as F

from ... import comfy_io
from ...fp8_linear import assert_materialized, materialize
from ...upscale import Info
from .model import SeedVR2Config, SeedVR2DiT
from .vae import SeedVR2VAE, fit_tiles, padded_length

SEED = 666  # the reference's default seed
TIMESTEP = 1000.0  # one step: the trailing schedule's only timestep, T * 1.0 (its resolution shift leaves 1 at 1)
MULTIPLE = 16  # VAE 8x, then 2x2 patches
# DiT tokens per pass: a longer clip is cut into parts of about equal length. 1080p is 8160 tokens a latent frame
MAX_TOKENS = 8160 * 11
# GiB of VAE activations per megapixel of frame (measured on an RTX 3090, plus 15%): (stage, frames) with
# frames 1 (an image), 5 (one slice) or more (sliced, each conv carrying frames to the next slice)
VAE_GIB_PER_MP = {("encode", 1): 1.7, ("decode", 1): 2.7, ("encode", 5): 11.0, ("decode", 5): 15.0,
                  ("encode", 9): 14.0, ("decode", 9): 22.0}
OVERLAP = 128  # pixels shared by neighbouring VAE tiles
MARGIN = 2**31  # VRAM left free: PyTorch's allocator reserves up to about 2 GB past its peak here
BATCH = 4  # frames resized or color-fixed at a time
CUDA_CONTEXT = 2**29  # what a process holds on the GPU beyond PyTorch's allocations
# settings ``upscale``'s ``options`` may set for one call: the attributes, plus "tile" (``vae_tile``) and
# "max_frames" (frames per DiT pass, setting ``max_tokens``)
OPTIONS = ("seed", "max_tokens", "vram_limit", "vae_tile", "vae_memory")
SMALL_KEYS = (".ada.", "_ada.", "norm_q.", "norm_k.", "vid_out_norm.", "rope.freqs")  # kept in fp32


def _blur(x: torch.Tensor, dilation: int) -> torch.Tensor:
    """A 3x3 binomial blur ([1, 2, 1] / 4 each way) with its taps ``dilation`` apart, edges replicated."""
    k = torch.tensor([1.0, 2.0, 1.0], device=x.device, dtype=x.dtype) / 4
    kernel = (k[:, None] * k[None, :]).expand(x.shape[1], 1, 3, 3)
    padded = F.pad(x, (dilation,) * 4, mode="replicate")
    return F.conv2d(padded, kernel, groups=x.shape[1], dilation=dilation)


def low_frequencies(x: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """What's left of (N, C, H, W) after an a-trous wavelet decomposition takes ``levels`` detail bands off it
    (blurs at dilations 1, 2, 4, ...)."""
    for i in range(levels):
        x = _blur(x, 2 ** i)
    return x


def color_fix(out: torch.Tensor, ref: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """The wavelet color fix SeedVR2's inference applies: ``out``'s detail bands on ``ref``'s low frequencies
    (the decomposition's details sum to the image minus its last low band). (N, C, H, W), computed in fp32."""
    out, ref = out.float(), ref.float()
    return out - low_frequencies(out, levels) + low_frequencies(ref, levels)


def chunks(frames: int, frames_per_chunk: int) -> list[tuple[int, int]]:
    """(start, end) parts of a clip, each at most ``frames_per_chunk`` (4n + 1) frames, about equal."""
    if frames <= frames_per_chunk:
        return [(0, frames)]
    n = math.ceil(frames / frames_per_chunk)
    size = 4 * math.ceil((math.ceil(frames / n) - 1) / 4) + 1
    return [(a, min(a + size, frames)) for a in range(0, frames, size)]


def _pad_amounts(n: int) -> tuple[int, int]:
    extra = -n % MULTIPLE
    return extra // 2, extra - extra // 2


class SeedVR2Upscaler:
    """SeedVR2 3B (ComfyUI's single file) and its VAE (``info.companion``) on ``device``, in bf16 like the
    reference. Settings (attributes, may be changed between calls): ``seed``, the noise (the same input gives the
    same output); ``max_tokens``, DiT tokens per pass, so the clip length a pass takes (``MAX_TOKENS``);
    ``vram_limit``, bytes this process may hold on the GPU (None: what's free); ``vae_tile``, the VAE's tile size
    in pixels (0: whole frames; None: tiles only when a frame wouldn't fit, sized to what's free);
    ``vae_memory``, where the VAE keeps the frames a clip's slice carries to the next ("gpu", or "cpu": less VRAM,
    about 3x slower). ``upscale``'s ``options`` sets them for one call (``OPTIONS``).

    A clip is upscaled in passes of ``frames_per_pass(size)`` frames (41 at 1080p), each on its own: give it whole
    clips, or pieces that are multiples of that, and it pads each to 4n + 1 frames and drops the padding."""

    scale = 0
    # working VRAM beyond the weights (6.8 GiB) for a 1080p clip. The VAE tiles itself to what's free, so this is
    # what a full 41-frame pass held (PyTorch's reserve) when held to the weights + 8 GiB: 7.7 GiB at its peak,
    # 8.6 reserved, at 2.3 s a frame (with room to spare, 11 GiB and 2.1 s). An image takes 4.9 GiB.
    vram_bytes = 9 * 2**30

    def __init__(self, info: Info, device: torch.device | str, log: Callable[[str], None] = print, *,
                 seed: int = SEED, max_tokens: int = MAX_TOKENS, vram_limit: int | None = None,
                 vae_tile: int | None = None, vae_memory: str = "gpu"):
        self.info = info
        self.device = torch.device(device)
        self.log = log
        self.seed, self.max_tokens, self.vram_limit = seed, max_tokens, vram_limit
        self.vae_tile, self.vae_memory = vae_tile, vae_memory
        if not info.companion:
            raise ValueError("SeedVR2 needs its VAE (seedvr2_ema_vae) in the same folder")
        self.dit, self.positive = self._load_dit(info.path)
        self.vae = self._load_vae(info.companion)
        self.weights_bytes = sum(t.numel() * t.element_size() for m in (self.dit, self.vae)
                                 for t in list(m.parameters()) + list(m.buffers()))

    # ---------------------------------------------------------------- loading
    def _load_dit(self, path: str) -> tuple[SeedVR2DiT, torch.Tensor]:
        self.log(f"Loading SeedVR2 {os.path.basename(path)}")
        small = lambda name: torch.float32 if any(s in name for s in SMALL_KEYS) else torch.bfloat16  # noqa: E731
        loaded = comfy_io.load_state_dict(path, device=self.device, cast_to=small)
        sd = loaded.tensors
        positive = sd.pop("positive_conditioning", None)
        sd.pop("negative_conditioning", None)  # CFG 1: never used
        if positive is None:
            raise ValueError(f"{os.path.basename(path)} has no positive_conditioning: Dainamo reads ComfyUI's "
                             "single-file SeedVR2, which carries the text embedding it was released with")
        if not any(k.endswith(".mlp.vid.proj_in_gate.weight") for k in sd):
            raise ValueError(f"{os.path.basename(path)} isn't SeedVR2 3B (the 7B is a different model, which "
                             "Dainamo doesn't run yet)")
        with torch.device("meta"):
            dit = SeedVR2DiT(SeedVR2Config.from_state_dict(sd))
        materialize(dit, loaded, torch.bfloat16, device=None, dtype_for=small)
        assert_materialized(dit)
        return dit.eval(), positive.to(self.device)

    def _load_vae(self, path: str) -> SeedVR2VAE:
        self.log(f"Loading SeedVR2 VAE {os.path.basename(path)}")
        loaded = comfy_io.load_state_dict(path, device=self.device, cast_to=lambda n: torch.bfloat16)
        with torch.device("meta"):
            vae = SeedVR2VAE()
        materialize(vae, loaded, torch.bfloat16, device=None)
        assert_materialized(vae)
        return vae.eval()

    def unload(self) -> None:
        self.dit = self.vae = self.positive = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- running
    def frames_per_pass(self, size: tuple[int, int]) -> int:
        """Frames (4n + 1) one DiT pass takes at ``size``."""
        w, h = size
        tokens = math.ceil(h / MULTIPLE) * math.ceil(w / MULTIPLE)
        return 4 * max(0, self.max_tokens // tokens - 1) + 1

    def available(self) -> int:
        """Bytes the VAE may still take on the GPU."""
        if self.device.type != "cuda":
            return 1 << 50
        free, _ = torch.cuda.mem_get_info(self.device)
        held = torch.cuda.memory_allocated(self.device)
        avail = free + torch.cuda.memory_reserved(self.device) - held  # PyTorch's cached blocks count as free
        if self.vram_limit:
            avail = min(avail, self.vram_limit - CUDA_CONTEXT - held)
        return avail - MARGIN

    def tile_for(self, frame: tuple[int, int], frames: int, stage: str) -> tuple[int, int] | None:
        """The VAE tile (height, width in pixels, multiples of 16) for ``frames`` of ``frame`` (h, w) at
        ``stage``, or None for whole frames."""
        h, w = frame
        if self.vae_tile is not None:
            if not self.vae_tile:
                return None
            side = max(2 * MULTIPLE, self.vae_tile // MULTIPLE * MULTIPLE)
            return min(h, side), min(w, side)
        per_mp = VAE_GIB_PER_MP[(stage, 1 if frames == 1 else 5 if frames <= 5 else 9)]
        area = max(0, self.available()) / 2**30 / per_mp * 1e6
        return fit_tiles(h, w, area, OVERLAP, MULTIPLE)

    def upscale(self, frames: torch.Tensor, size: tuple[int, int], progress: Callable[[str], None] | None = None,
                check: Callable[[], None] | None = None, options: dict | None = None) -> torch.Tensor:
        """frames (T, 3, H, W) in [0, 1] on the CPU -> (T, 3, size[1], size[0]) the same way. ``options``: settings
        for this call (``OPTIONS``; others are ignored)."""
        saved = {k: getattr(self, k) for k in OPTIONS}
        try:
            for k, v in (options or {}).items():
                if k == "max_frames" and v:
                    self.max_tokens = self.tokens_for(size, int(v))
                elif k == "tile":
                    self.vae_tile = None if v is None else int(v)
                elif k in OPTIONS and v is not None:
                    setattr(self, k, v)
            return self._upscale(frames, size, progress, check)
        finally:
            for k, v in saved.items():
                setattr(self, k, v)

    def tokens_for(self, size: tuple[int, int], frames: int) -> int:
        """The ``max_tokens`` that makes passes of ``frames`` (rounded up to 4n + 1) at ``size``."""
        w, h = size
        return (math.ceil((frames - 1) / 4) + 1) * math.ceil(h / MULTIPLE) * math.ceil(w / MULTIPLE)

    @torch.inference_mode()
    def _upscale(self, frames: torch.Tensor, size: tuple[int, int], progress, check) -> torch.Tensor:
        w, h = size
        total = frames.shape[0]
        out = torch.empty(total, 3, h, w)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        parts = chunks(total, self.frames_per_pass(size))
        self.vae.set_memory_device("cpu" if self.vae_memory == "cpu" else None)
        for a, b in parts:
            if check:
                check()
            if progress:
                progress("Upscaling" if total == 1 else f"Upscaling frames {a + 1}-{b} of {total}")
            out[a:b] = self._clip(frames[a:b], size, gen, check)
        return out

    def _prepare(self, frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """frames (t, 3, H, W) [0, 1] CPU -> (t, 3, H16, W16) on the GPU in [-1, 1]: resized to ``size`` (bicubic,
        antialiased, as torchvision's ``resize``), clamped, then padded to multiples of 16 (mirrored)."""
        w, h = size
        top, bottom = _pad_amounts(h)
        left, right = _pad_amounts(w)
        x = frames.to(self.device, torch.float32)
        x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
        mode = "reflect" if min(h, w) > MULTIPLE else "replicate"
        return F.pad(x, (left, right, top, bottom), mode=mode).mul_(2).sub_(1)

    def _clip(self, frames: torch.Tensor, size: tuple[int, int], gen: torch.Generator,
              check: Callable[[], None] | None) -> torch.Tensor:
        """One DiT pass: frames (t, 3, H, W) [0, 1] CPU -> (t, 3, h, w) [0, 1] CPU. The resized frames are made
        a few at a time, twice (for the VAE, then for the color fix), so a long clip is never whole on the GPU
        in fp32."""
        w, h = size
        t = frames.shape[0]
        top, left = _pad_amounts(h)[0], _pad_amounts(w)[0]
        hp, wp = h + sum(_pad_amounts(h)), w + sum(_pad_amounts(w))
        n = padded_length(t)
        clip = torch.empty(1, 3, n, hp, wp, dtype=self.vae.dtype, device=self.device)
        for a in range(0, t, BATCH):
            clip[0, :, a:min(a + BATCH, t)] = self._prepare(frames[a:a + BATCH], size).transpose(0, 1)
        if n > t:
            clip[:, :, t:] = clip[:, :, t - 1:t]  # the reference's cut_videos: the last frame repeated
        tile = self.tile_for((hp, wp), n, "encode")
        latents = self.vae.encode(clip, generator=gen, tile=tile, overlap=OVERLAP, check=check)  # (1, 16, l, ., .)
        del clip
        if check:
            check()
        cond = latents[0].permute(1, 2, 3, 0)  # (l, hp / 8, wp / 8, 16)
        noise = torch.randn(cond.shape, generator=gen, device=cond.device, dtype=torch.float32)
        vid = torch.cat([noise, cond, torch.ones_like(cond[..., :1])], dim=-1)
        del cond, latents
        v = self.dit(vid, self.positive, TIMESTEP)
        x0 = (noise - v.float()).permute(3, 0, 1, 2)[None]  # the lerp schedule's endpoint at t = T: noise - v
        del vid, v, noise
        if check:
            check()
        result = torch.empty(t, 3, h, w)
        done = 0
        tile = self.tile_for((hp, wp), n, "decode")
        tile = tile and (tile[0] // 8, tile[1] // 8)
        for y in self.vae.decode_slices(x0, tile=tile, overlap=OVERLAP // 8, check=check):
            for a in range(0, min(y.shape[2], t - done), BATCH):
                b = min(a + BATCH, y.shape[2], t - done)
                ref = self._prepare(frames[done + a:done + b], size)
                fixed = color_fix(y[0, :, a:b].transpose(0, 1), ref).clamp_(-1, 1).add_(1).mul_(0.5)
                result[done + a:done + b] = fixed[:, :, top:top + h, left:left + w].cpu()
                del ref, fixed
            done += y.shape[2]
            del y
        return result

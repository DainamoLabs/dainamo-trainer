"""Upscalers: the files of the ``upscale_models`` folder, told apart by their tensors.

* Pixel upscalers (the ESRGAN family and the other single-image architectures ``spandrel`` loads: Real-ESRGAN,
  4x-UltraSharp, DAT, HAT, SwinIR, ...): ``upscalers/pixel.py``. A fixed integer scale, run tiled, frame by frame.
* SeedVR2 (ByteDance's one-step diffusion video restorer, the 3B release as ComfyUI's single file beside its
  VAE): ``upscalers/seedvr2/``. Any output size; a clip's frames go through together, in passes of as many as fit
  (41 at 1080p), so they stay consistent, neighbouring passes sharing a few frames that are crossfaded
  (``crossfade``) so the join doesn't show.

Every upscaler takes frames as a float tensor (T, 3, H, W) in [0, 1] on the CPU and gives frames the same way at
the asked size (``Upscaler.upscale``). The Sandbox worker (``engine.sandbox``) runs them: on a run's output, an
uploaded file, or a dataset's small images (``upscale_dataset``), which it replaces, keeping the originals in the
dataset's ``_originals`` folder (``dataset.ORIGINALS_DIR``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from . import comfy_io

FOLDER = "upscale_models"  # under the models folder, as in ComfyUI
EXTS = (".safetensors", ".pth", ".pt", ".ckpt")
# tensors that mark a file's kind (any key ending with one of them)
SEEDVR2_KEYS = ("blocks.0.attn.proj_qkv.vid.weight", "blocks.0.mlp.vid.proj_in_gate.weight", "vid_in.proj.weight")  # 3B
SEEDVR2_VAE_KEYS = ("decoder.conv_in.weight", "encoder.conv_out.weight")


@dataclass
class Info:
    """What a file in the folder is: its kind ("pixel", "seedvr2", "seedvr2_vae" or "unknown"), the fixed scale of
    a pixel upscaler (0: any size), whether it takes clips as clips, and for SeedVR2 the VAE file it needs."""
    path: str
    kind: str
    name: str
    scale: int = 0
    video: bool = False
    companion: str | None = None
    error: str | None = None

    def to_json(self) -> dict:
        return {"path": self.path, "kind": self.kind, "name": self.name, "scale": self.scale, "video": self.video,
                "companion": self.companion, "error": self.error}


class Upscaler(Protocol):
    """A loaded upscaler. ``scale`` is 0 when it makes any size (SeedVR2), else its fixed factor.

    One that takes clips as clips (``info.video``) may say how, for a caller that has to cut a long clip into
    pieces (the Sandbox worker): ``frames_per_pass(size, options)``, the frames it takes together;
    ``overlap_frames(size, options)``, the frames neighbouring passes share (pieces share as many, and are
    ``crossfade``d); ``check_options(options)``, its options checked once (out-of-range values clamped, unknown
    ones dropped, each change logged)."""
    info: Info
    scale: int

    def upscale(self, frames: torch.Tensor, size: tuple[int, int], progress: Callable[[str], None] | None = None,
                check: Callable[[], None] | None = None, options: dict | None = None) -> torch.Tensor:
        """frames (T, 3, H, W) float in [0, 1] on the CPU -> frames (T, 3, size[1], size[0]) the same way.
        ``size`` is (width, height). A fixed-scale upscaler runs at its scale and is fitted to ``size`` afterwards
        (``fit``). ``progress`` takes a line of text; ``check`` may raise to cancel; ``options`` are the upscaler's
        own settings (SeedVR2's), None for the defaults. Frames of other channel counts go through
        ``any_channels``."""
        ...

    def unload(self) -> None: ...


def any_channels(up: Upscaler, frames: torch.Tensor, size: tuple[int, int], progress=None, check=None,
                 **kw) -> torch.Tensor:
    """Frames of any channels through an upscaler that takes RGB (``up.upscale``, given ``kw`` too): greyscale (1),
    greyscale and alpha (2), RGB (3) or RGBA (4). Grey runs as RGB and comes back as the mean of the channels; alpha
    isn't shown to the model but resized to ``size`` beside it (``fit``), so transparency carries through."""
    c = frames.shape[1]
    if c == 3:
        return up.upscale(frames, size, progress, check, **kw)
    color, alpha = (frames[:, :-1], frames[:, -1:]) if c in (2, 4) else (frames, None)
    gray = color.shape[1] == 1
    out = up.upscale(color.expand(-1, 3, -1, -1).contiguous() if gray else color, size, progress, check, **kw)
    if gray:
        out = out.mean(1, keepdim=True)
    return torch.cat([out, fit(alpha, size)], 1) if alpha is not None else out


def is_seedvr2_vae(keys: list[str]) -> bool:
    return all(any(k.endswith(s) for k in keys) for s in SEEDVR2_VAE_KEYS) and \
        not any(k.endswith(SEEDVR2_KEYS[0]) for k in keys)


def is_seedvr2(keys: list[str]) -> bool:
    return all(any(k.endswith(s) for k in keys) for s in SEEDVR2_KEYS)


def _keys(path: str) -> list[str] | None:
    """A .safetensors file's tensor names (None for other formats, which spandrel reads itself)."""
    return comfy_io.read_header(path)[0] if path.endswith(".safetensors") else None


def identify(path: str, vae_files: list[str] | None = None) -> Info:
    """What ``path`` is, from its tensor names (a .pth is read by spandrel). A SeedVR2 DiT's ``companion`` is the
    SeedVR2 VAE among ``vae_files`` (else the files of its own folder), or None when there isn't one, which
    ``error`` explains."""
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        keys = _keys(path)
    except (OSError, ValueError) as e:
        return Info(path, "unknown", name, error=f"Couldn't read it: {e}")
    if keys is not None and is_seedvr2(keys):
        files = vae_files if vae_files is not None else _folder_files(os.path.dirname(path))
        vae = None
        for f in files:
            try:
                if f != path and f.endswith(".safetensors") and is_seedvr2_vae(_keys(f) or []):
                    vae = f
                    break
            except (OSError, ValueError):
                continue
        return Info(path, "seedvr2", name, scale=0, video=True, companion=vae,
                    error=None if vae else "SeedVR2 needs its VAE (seedvr2_ema_vae) in upscale_models or vae")
    if keys is not None and is_seedvr2_vae(keys):
        return Info(path, "seedvr2_vae", name)
    from .upscalers import pixel
    return pixel.identify(path, name)


def _folder_files(folder: str) -> list[str]:
    try:
        return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(EXTS))
    except OSError:
        return []


def list_models(models_folder: str) -> list[Info]:
    """The upscalers of ``models_folder``: every model file in ``upscale_models``, with what it is, plus SeedVR2 DiTs
    kept the ComfyUI way in ``diffusion_models`` (their VAE may sit in ``vae``). SeedVR2 VAEs themselves are left out
    (they belong to their DiT)."""
    files = _folder_files(os.path.join(models_folder, FOLDER))
    for f in _folder_files(os.path.join(models_folder, "diffusion_models")):
        try:
            if f.endswith(".safetensors") and is_seedvr2(_keys(f) or []):
                files.append(f)
        except (OSError, ValueError):
            continue
    vae_files = files + _folder_files(os.path.join(models_folder, "vae"))
    out = [identify(f, vae_files) for f in files]
    return [i for i in out if i.kind != "seedvr2_vae"]


def load(info: Info, device: torch.device, log=print) -> Upscaler:
    """The upscaler of ``info`` on ``device``."""
    if info.kind == "seedvr2":
        if not info.companion:
            raise ValueError(info.error or "SeedVR2 needs its VAE file")
        from .upscalers.seedvr2 import SeedVR2Upscaler
        return SeedVR2Upscaler(info, device, log)
    if info.kind == "pixel":
        from .upscalers.pixel import PixelUpscaler
        return PixelUpscaler(info, device, log)
    raise ValueError(info.error or f"{os.path.basename(info.path)} isn't an upscaler this app knows")


def fit(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Frames (T, C, H, W), any channels, resized to ``size`` (width, height) with antialiasing, if they aren't that
    size."""
    w, h = size
    if frames.shape[-1] == w and frames.shape[-2] == h:
        return frames
    import torch.nn.functional as F
    return F.interpolate(frames, size=(h, w), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)


def crossfade(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """The frames two overlapping passes both made, (n, ...) each, going linearly from the earlier pass's
    (``before``) to the later one's (``after``) so the join doesn't show: frame i weighs (i + 0.5) / n of
    ``after``. In fp32."""
    n = before.shape[0]
    w = ((torch.arange(n, dtype=torch.float32, device=before.device) + 0.5) / max(n, 1))
    return torch.lerp(before.float(), after.float(), w.view(-1, *[1] * (before.dim() - 1)))


def target_size(width: int, height: int, factor: float, multiple: int = 1) -> tuple[int, int]:
    """The output size of upscaling ``width`` x ``height`` by ``factor``, each side rounded to ``multiple``."""
    w = max(multiple, int(round(width * factor / multiple)) * multiple)
    h = max(multiple, int(round(height * factor / multiple)) * multiple)
    return w, h


def factor_to_reach(width: int, height: int, resolution: int, max_factor: float) -> float:
    """How much to upscale a ``width`` x ``height`` image so its area reaches ``resolution`` squared (the size a job
    trains at), at most ``max_factor``; 1 when it's already there."""
    have = math.sqrt(width * height)
    if have >= resolution:
        return 1.0
    return min(max_factor, resolution / have)

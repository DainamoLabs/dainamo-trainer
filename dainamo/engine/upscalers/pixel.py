"""Pixel upscalers: the single-image super-resolution models ``spandrel`` loads (the ESRGAN family, such as
Real-ESRGAN and 4x-UltraSharp, and DAT, HAT, SwinIR, ...). Each has a fixed scale (``Info.scale``).

Frames are upscaled one at a time in overlapping tiles, so a large image fits in a few GB of VRAM. A model sees
less around the pixels at a tile's edge, so neighbouring tiles fade into each other across their overlap (a
linear crossfade, the edge pixels weighing next to nothing) and no seams show. ``upscale.fit`` then brings the
result to the asked size: a 4x model asked for 2x runs at 4x and is scaled down, which looks sharper than 2x of
a plain resize.
"""

from __future__ import annotations

import dataclasses
import gc
import math
import os
from typing import Callable

import torch

from ..upscale import Info, any_channels, fit  # noqa: F401  (any_channels: kept importable from here)

TILE = 512  # input pixels a tile side holds at most
OVERLAP = 32  # input pixels neighbouring tiles share at least
# bigger files aren't pixel upscalers (the largest spandrel loads are a few hundred MB), and reading one to find
# out would take a minute or more on a slow disk
MAX_BYTES = 1 << 30
_identified: dict[tuple, Info] = {}  # (path, size, mtime) -> what it is: listing the folder needn't reload each file


def _load(path: str):
    """The file's spandrel model descriptor, on the CPU. (Tests put a stand-in in its place.)"""
    from spandrel import ImageModelDescriptor, ModelLoader

    desc = ModelLoader().load_from_file(path)
    if not isinstance(desc, ImageModelDescriptor):
        raise ValueError(f"it's a {desc.architecture.name} inpainting model, not an upscaler")
    if desc.input_channels not in (1, 3) or desc.output_channels != desc.input_channels:
        raise ValueError(f"its {desc.architecture.name} model takes {desc.input_channels} channels and makes "
                         f"{desc.output_channels}: only RGB and greyscale models work here")
    return desc


def identify(path: str, name: str) -> Info:
    """What ``path`` is, by loading it with spandrel (on the CPU; a few hundred ms for an ESRGAN file): a "pixel"
    upscaler and its scale, or "unknown" with the reason in ``error``. Remembered while the file is unchanged."""
    try:
        st = os.stat(path)
    except OSError as e:
        return Info(path, "unknown", name, error=f"Couldn't read it: {e}")
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _identified:
        _identified[key] = _identify(path, name, st.st_size)
    return dataclasses.replace(_identified[key], name=name)


def _identify(path: str, name: str, size: int) -> Info:
    if size > MAX_BYTES:
        return Info(path, "unknown", name, error=f"Not an upscaler this app knows (at {size / 2**30:.1f} GB it's too "
                                                 "big for a pixel upscaler)")
    from spandrel import UnsupportedModelError

    try:
        desc = _load(path)
    except UnsupportedModelError:
        return Info(path, "unknown", name, error="Not an upscaler this app knows: no architecture spandrel supports "
                                                 "matches its tensors")
    except Exception as e:  # a bad extension, a damaged file, an inpainting model, ...
        return Info(path, "unknown", name, error=f"Couldn't load it: {e}")
    return Info(path, "pixel", name, scale=int(desc.scale))


def spans(n: int, tile: int, overlap: int, multiple: int = 1) -> list[tuple[int, int]]:
    """(start, end) of the tiles covering 0..n along one side: as few as hold at most ``tile`` pixels with at least
    ``overlap`` shared between neighbours, all the same size (rounded up to ``multiple``, what the model pads to,
    when that fits) and spread evenly, so none is a sliver."""
    if n <= tile:
        return [(0, n)]
    overlap = min(overlap, tile // 2)
    count = math.ceil((n - overlap) / (tile - overlap))
    size = math.ceil((n + (count - 1) * overlap) / count)
    size = min(n, -(-size // multiple) * multiple)
    return [(s, s + size) for s in (round(i * (n - size) / (count - 1)) for i in range(count))]


def ramps(tiles: list[tuple[int, int]], scale: int) -> list[torch.Tensor]:
    """Each tile's blending weights along one side of the output: 1, fading linearly across the overlap with each
    neighbour (the two tiles' weights add up to 1 there), never quite to 0, so every pixel has some weight."""
    out = []
    for i, (a, b) in enumerate(tiles):
        w = torch.ones((b - a) * scale)
        k = (tiles[i - 1][1] - a) * scale if i > 0 else 0
        if k > 0:
            w[:k] = (torch.arange(k) + 0.5) / k
        k = (b - tiles[i + 1][0]) * scale if i + 1 < len(tiles) else 0
        if k > 0:
            w[-k:] = torch.minimum(w[-k:], (torch.arange(k, 0, -1) - 0.5) / k)
        out.append(w)
    return out


class PixelUpscaler:
    """A pixel upscaler on ``device``: fp16 when its architecture supports it (fp32 on the CPU, or if fp16
    overflows), run tile by tile (``tile`` and ``overlap`` in input pixels)."""

    def __init__(self, info: Info, device: torch.device, log: Callable = print, tile: int = TILE,
                 overlap: int = OVERLAP):
        self.info, self.device, self.log = info, torch.device(device), log
        self.tile, self.overlap = tile, overlap
        log(f"Loading {os.path.basename(info.path)}")
        desc = _load(info.path)
        self.scale = int(desc.scale)
        self.channels = int(getattr(desc, "input_channels", 3))
        req = getattr(desc, "size_requirements", None)
        self.multiple = int(getattr(req, "multiple_of", 1) or 1)
        # some architectures look at the whole image and say not to tile them
        self.whole = getattr(getattr(desc, "tiling", None), "name", "") == "INTERNAL"
        half = self.device.type == "cuda" and bool(getattr(desc, "supports_half", False))
        self.dtype = torch.float16 if half else torch.float32
        self.model = desc.to(self.device, self.dtype)

    def upscale(self, frames: torch.Tensor, size: tuple[int, int], progress: Callable[[str], None] | None = None,
                check: Callable[[], None] | None = None, options: dict | None = None) -> torch.Tensor:
        """(T, C, H, W) float in [0, 1] on the CPU -> (T, C, size[1], size[0]); C is 3, or 1, 2 or 4 (``any_channels``).
        ``progress`` hears of each frame of a clip, or each tile of a single large image. A pixel upscaler has no
        ``options`` (SeedVR2's settings)."""
        if frames.shape[1] != 3:
            return any_channels(self, frames, size, progress, check)
        t = frames.shape[0]
        out = torch.empty(t, 3, size[1], size[0])
        for i in range(t):
            if check:
                check()
            if progress and t > 1:
                progress(f"Frame {i + 1} of {t}")
            out[i] = fit(self._tiled(frames[i:i + 1], check, progress if t == 1 else None), size)[0]
        return out

    @torch.inference_mode()
    def _tiled(self, x: torch.Tensor, check=None, progress=None) -> torch.Tensor:
        """(1, 3, H, W) -> (1, 3, H * scale, W * scale) float32 on the CPU, in tiles blended across their overlaps.
        The weights are a product of one per side, so the total weight is too."""
        _, c, h, w = x.shape
        s = self.scale
        if self.whole:
            ys, xs = [(0, h)], [(0, w)]
        else:
            ys, xs = spans(h, self.tile, self.overlap, self.multiple), spans(w, self.tile, self.overlap, self.multiple)
        if len(ys) == len(xs) == 1:
            return self._run(x)
        wys, wxs = ramps(ys, s), ramps(xs, s)
        out = torch.zeros(1, c, h * s, w * s)
        total_y, total_x = torch.zeros(h * s), torch.zeros(w * s)
        for (y0, y1), wy in zip(ys, wys):
            total_y[y0 * s:y1 * s] += wy
        for (x0, x1), wx in zip(xs, wxs):
            total_x[x0 * s:x1 * s] += wx
        n = 0
        for (y0, y1), wy in zip(ys, wys):
            for (x0, x1), wx in zip(xs, wxs):
                if check:
                    check()
                n += 1
                if progress:
                    progress(f"Tile {n} of {len(ys) * len(xs)}")
                piece = self._run(x[:, :, y0:y1, x0:x1])
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] += piece * (wy[:, None] * wx[None, :])
        return out.div_(total_y[:, None] * total_x[None, :])

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """One tile through the model -> float32 on the CPU. A grey model takes the channels one at a time."""
        if self.channels == 1:
            return torch.cat([self._call(x[:, i:i + 1]) for i in range(x.shape[1])], 1)
        return self._call(x)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.to(self.device, self.dtype))
        if self.dtype == torch.float16 and not torch.isfinite(y).all():
            # a few ESRGAN finetunes overflow fp16; spandrel can't tell which, so find out on the first tile
            self.log(f"{self.info.name} overflowed in fp16: running it in fp32")
            self.dtype = torch.float32
            self.model.to(self.device, torch.float32)
            y = self.model(x.to(self.device, self.dtype))
        return y.float().cpu()

    def unload(self) -> None:
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

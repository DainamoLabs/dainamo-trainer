"""SeedVR2's video VAE (``video_vae_v3``, "s8_c16_t4"): a causal 3D-conv autoencoder, 8x smaller spatially, 4x in
time (the first frame alone, then 4 frames per latent frame), 16 latent channels.

Ported from ByteDance Seed's reference code, SeedVR (github.com/ByteDance-Seed/SeedVR at e4de8c2, Apache License
2.0: ``models/video_vae_v3/modules/attn_video_vae.py``, ``causal_inflation_lib.py`` and
``s8_c16_t4_inflation_sd3.yaml``). See THIRD_PARTY_NOTICES.md.

The layout is diffusers' AutoencoderKL with every conv made a causal 3D conv: no temporal padding, the first frame
repeated in front instead; GroupNorm and the mid-block attention work frame by frame. The encoder halves time in
its 2nd and 3rd down blocks, the decoder doubles it in its 1st and 2nd up blocks (a 1x1x1 conv to 8x the channels,
a pixel shuffle in t, h, w, and the first frame's duplicate dropped). Like the reference, a clip is encoded in
slices (the first 5 frames, then 4 at a time) and decoded a latent frame at a time; each conv keeps the last
frames of its input for the next slice, so the result is the same as in one go.

Changes for Dainamo (the arithmetic is the reference's):
  * module names are the checkpoint's keys (ComfyUI's repack uses diffusers' current attention names);
  * a single frame (an image) runs each causal conv as a 2D conv with its kernel summed over time (the conv would
    see the frame three times), so images cost a third;
  * a big conv input is run in bands of rows (exact: each band takes the rows its outputs need) rather than in the
    reference's recursive 0.5 GB splits, and GroupNorm runs a frame at a time;
  * the slices' carried frames can be kept in system RAM (``memory_device``), for big frames.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FIRST, NEXT = "first", "next"  # a slice's place in a sliced clip; None: the whole clip at once
CONV_BAND_BYTES = 768 * 2**20  # a conv's (extended, padded) input above this runs in bands of rows


class CausalConv3d(nn.Module):
    """A 3D conv that's causal in time: ``2 * temporal_pad`` copies of the first frame go in front (or, for a
    slice after the first, the frames the previous slice left), never anything after. ``pad`` is the spatial zero
    padding (left, right, top, bottom)."""

    def __init__(self, cin: int, cout: int, kernel: int | tuple[int, int, int], stride: int | tuple[int, int, int] = 1,
                 temporal_pad: int = 0, pad: tuple[int, int, int, int] = (0, 0, 0, 0), bias: bool = True):
        super().__init__()
        kernel = (kernel,) * 3 if isinstance(kernel, int) else tuple(kernel)
        stride = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        self.kernel, self.stride, self.temporal_pad, self.pad = kernel, stride, temporal_pad, pad
        self.weight = nn.Parameter(torch.empty(cout, cin, *kernel))
        self.bias = nn.Parameter(torch.zeros(cout)) if bias else None
        self.memory: Tensor | None = None
        self.memory_device: str | None = None  # None: beside the activations; "cpu": in system RAM
        self._summed: Tensor | None = None

    def forward(self, x: Tensor, state: str | None = None) -> Tensor:
        kt, st = self.kernel[0], self.stride[0]
        keep = kt - st  # input frames the next slice needs
        if state == NEXT and self.memory is not None:
            prefix = self.memory.to(x.device, x.dtype)
        else:
            prefix = x[:, :, :1].expand(-1, -1, 2 * self.temporal_pad, -1, -1) if self.temporal_pad else None
        if state is not None and keep > 0:
            tail = x[:, :, -keep:] if x.shape[2] >= keep else torch.cat([prefix, x], dim=2)[:, :, -keep:]
            self.memory = tail.detach().to(self.memory_device) if self.memory_device else tail.detach().clone()
        elif state is not None:
            self.memory = None
        if prefix is not None and state != NEXT and x.shape[2] == 1 and kt == 2 * self.temporal_pad + 1:
            return self._single_frame(x)  # the conv would see the same frame kt times
        return self._conv(prefix, x)

    def _single_frame(self, x: Tensor) -> Tensor:
        if self._summed is None or self._summed.dtype != self.weight.dtype or self._summed.device != self.weight.device:
            self._summed = self.weight.float().sum(2).to(self.weight.dtype)
        left, right, top, bottom = self.pad
        xs = x[:, :, 0]
        if left == right and top == bottom:
            out = F.conv2d(xs, self._summed, self.bias, stride=self.stride[1:], padding=(top, left))
        else:
            out = F.conv2d(F.pad(xs, self.pad), self._summed, self.bias, stride=self.stride[1:])
        return out[:, :, None]

    def _conv(self, prefix: Tensor | None, x: Tensor) -> Tensor:
        """The conv of ``prefix`` + ``x`` along time; a big input in bands of rows, each band built (joined and
        padded) on its own, so neither the joined clip nor its padded copy is ever made whole."""
        left, right, top, bottom = self.pad
        (kt, kh, kw), (st, sh, sw) = self.kernel, self.stride
        b, _, t, h, w = x.shape
        t_in = t + (prefix.shape[2] if prefix is not None else 0)
        size = x.element_size() * x.shape[1] * t_in * (h + top + bottom) * (w + left + right) * b
        if size <= CONV_BAND_BYTES:
            ext = x if prefix is None else torch.cat([prefix, x], dim=2)
            if left == right and top == bottom:
                return F.conv3d(ext, self.weight, self.bias, stride=self.stride, padding=(0, top, left))
            return F.conv3d(F.pad(ext, self.pad), self.weight, self.bias, stride=self.stride)
        out_t = (t_in - kt) // st + 1
        out_h = (h + top + bottom - kh) // sh + 1
        out_w = (w + left + right - kw) // sw + 1
        out = x.new_empty(b, self.weight.shape[0], out_t, out_h, out_w)
        rows = max(1, int(out_h * CONV_BAND_BYTES / size))
        for r0 in range(0, out_h, rows):
            r1 = min(out_h, r0 + rows)
            a, e = r0 * sh - top, (r1 - 1) * sh - top + kh  # the input rows these outputs read
            rs = slice(max(a, 0), min(e, h))
            band = x[:, :, :, rs] if prefix is None else torch.cat([prefix[:, :, :, rs], x[:, :, :, rs]], dim=2)
            band = F.pad(band, (left, right, max(0, -a), max(0, e - h)))
            out[:, :, :, r0:r1] = F.conv3d(band, self.weight, self.bias, stride=self.stride)
            del band
        return out


def group_norm(norm: nn.GroupNorm, x: Tensor, silu: bool, inplace: bool = False) -> Tensor:
    """GroupNorm of each frame of (B, C, T, H, W) on its own, as the reference's ``causal_norm_wrapper``, then
    SiLU if asked; a frame at a time, into ``x`` itself with ``inplace``."""
    out = x if inplace else torch.empty_like(x)
    for t in range(x.shape[2]):
        y = F.group_norm(x[:, :, t], norm.num_groups, norm.weight, norm.bias, norm.eps)
        out[:, :, t] = F.silu(y, inplace=True) if silu else y
        del y
    return out


class ResnetBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin, eps=eps)
        self.conv1 = CausalConv3d(cin, cout, 3, temporal_pad=1, pad=(1, 1, 1, 1))
        self.norm2 = nn.GroupNorm(groups, cout, eps=eps)
        self.conv2 = CausalConv3d(cout, cout, 3, temporal_pad=1, pad=(1, 1, 1, 1))
        self.conv_shortcut = CausalConv3d(cin, cout, 1) if cin != cout else None

    def forward(self, x: Tensor, state: str | None) -> Tensor:
        h = self.conv1(group_norm(self.norm1, x, silu=True), state)
        h = self.conv2(group_norm(self.norm2, h, silu=True, inplace=True), state)
        return h.add_(self.conv_shortcut(x, state) if self.conv_shortcut is not None else x)


class Attention(nn.Module):
    """The mid block's single-head self-attention over each frame's pixels, with a residual (diffusers'
    ``Attention`` as the reference builds it)."""

    def __init__(self, dim: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim, eps=eps)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T, H, W), updated in place."""
        for t in range(x.shape[2]):
            f = x[:, :, t]
            b, c, h, w = f.shape
            y = self.group_norm(f).flatten(2).transpose(1, 2)  # (B, HW, C)
            q, k, v = (lin(y)[:, None] for lin in (self.to_q, self.to_k, self.to_v))
            y = F.scaled_dot_product_attention(q, k, v)[:, 0]
            f += self.to_out[0](y).transpose(1, 2).reshape(b, c, h, w)
        return x


class MidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(dim, dim), ResnetBlock3D(dim, dim)])
        self.attentions = nn.ModuleList([Attention(dim)])

    def forward(self, x: Tensor, state: str | None) -> Tensor:
        x = self.resnets[0](x, state)
        x = self.attentions[0](x)
        return self.resnets[1](x, state)


class Downsample3D(nn.Module):
    """Stride 2 in h and w (and in t for ``temporal``), zero padding only on the right and bottom."""

    def __init__(self, dim: int, temporal: bool):
        super().__init__()
        self.conv = CausalConv3d(dim, dim, (3 if temporal else 1, 3, 3), stride=(2 if temporal else 1, 2, 2),
                                 temporal_pad=1 if temporal else 0, pad=(0, 1, 0, 1))

    def forward(self, x: Tensor, state: str | None) -> Tensor:
        return self.conv(x, state)


class Upsample3D(nn.Module):
    """A 1x1x1 conv to 4x (8x with ``temporal``) the channels, shuffled into 2x h and w (and t), the first frame's
    duplicate dropped, then a causal 3x3x3 conv."""

    def __init__(self, dim: int, temporal: bool):
        super().__init__()
        self.temporal = temporal
        self.upscale_conv = nn.Conv3d(dim, dim * (8 if temporal else 4), kernel_size=1)
        self.conv = CausalConv3d(dim, dim, 3, temporal_pad=1, pad=(1, 1, 1, 1))

    def forward(self, x: Tensor, state: str | None) -> Tensor:
        b, c, t, h, w = x.shape
        z = 2 if self.temporal else 1
        drop = 1 if self.temporal and state != NEXT else 0  # the first frame's temporal duplicate
        out = x.new_empty(b, c, t * z - drop, 2 * h, 2 * w)
        for f in range(t):  # a frame at a time, so the shuffle never copies the whole clip
            y = self.upscale_conv(x[:, :, f:f + 1]).view(b, 2, 2, z, c, h, w)  # channels as (x y z c)
            y = y.permute(0, 4, 3, 5, 1, 6, 2).reshape(b, c, z, 2 * h, 2 * w)
            if f == 0 and drop:
                y = y[:, :, :1]
            start = max(0, f * z - drop) if f else 0
            out[:, :, start:start + y.shape[2]] = y
            del y
        return self.conv(out, state)


class DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, down: bool, temporal: bool, layers: int = 2):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(cin if i == 0 else cout, cout) for i in range(layers)])
        self.downsamplers = nn.ModuleList([Downsample3D(cout, temporal)]) if down else None

    def forward(self, x: Tensor, state: str | None) -> Tensor:
        for r in self.resnets:
            x = r(x, state)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x, state)
        return x


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int, up: bool, temporal: bool, layers: int = 3):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(cin if i == 0 else cout, cout) for i in range(layers)])
        self.upsamplers = nn.ModuleList([Upsample3D(cout, temporal)]) if up else None

    def forward(self, x: Tensor, state: str | None) -> Tensor:
        for r in self.resnets:
            x = r(x, state)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, state)
        return x


class Encoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], latent: int, temporal_downs: int = 2):
        super().__init__()
        self.conv_in = CausalConv3d(3, channels[0], 3, temporal_pad=1, pad=(1, 1, 1, 1))
        n = len(channels)
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[max(i - 1, 0)], channels[i], down=i < n - 1, temporal=i >= n - temporal_downs - 1)
            for i in range(n)])
        self.mid_block = MidBlock(channels[-1])
        self.conv_norm_out = nn.GroupNorm(32, channels[-1], eps=1e-6)
        self.conv_out = CausalConv3d(channels[-1], 2 * latent, 3, temporal_pad=1, pad=(1, 1, 1, 1))

    def forward(self, x: Tensor, state: str | None) -> Tensor:
        x = self.conv_in(x, state)
        for block in self.down_blocks:
            x = block(x, state)
        x = self.mid_block(x, state)
        return self.conv_out(group_norm(self.conv_norm_out, x, silu=True, inplace=True), state)


class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], latent: int, temporal_ups: int = 2):
        super().__init__()
        rev = tuple(reversed(channels))
        self.conv_in = CausalConv3d(latent, rev[0], 3, temporal_pad=1, pad=(1, 1, 1, 1))
        self.mid_block = MidBlock(rev[0])
        n = len(rev)
        self.up_blocks = nn.ModuleList([
            UpBlock(rev[max(i - 1, 0)], rev[i], up=i < n - 1, temporal=i < temporal_ups) for i in range(n)])
        self.conv_norm_out = nn.GroupNorm(32, channels[0], eps=1e-6)
        self.conv_out = CausalConv3d(channels[0], 3, 3, temporal_pad=1, pad=(1, 1, 1, 1))

    def forward(self, z: Tensor, state: str | None) -> Tensor:
        x = self.conv_in(z, state)
        x = self.mid_block(x, state)
        for block in self.up_blocks:
            x = block(x, state)
        return self.conv_out(group_norm(self.conv_norm_out, x, silu=True, inplace=True), state)


class SeedVR2VAE(nn.Module):
    scaling_factor = 0.9152  # configs_3b/main.yaml
    slice_frames = 4  # the reference's causal slicing (vae.slicing.split_size): frames per slice after the first

    def __init__(self, channels: tuple[int, ...] = (128, 256, 512, 512), latent: int = 16):
        super().__init__()
        self.encoder = Encoder(channels, latent)
        self.decoder = Decoder(channels, latent)
        self.latent_channels = latent

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.conv_in.weight.dtype

    def set_memory_device(self, device: str | None):
        """Where convs keep the frames a slice leaves for the next: None (on the GPU) or "cpu"."""
        for m in self.modules():
            if isinstance(m, CausalConv3d):
                m.memory_device = device

    def clear_memory(self):
        for m in self.modules():
            if isinstance(m, CausalConv3d):
                m.memory = None

    def _moments(self, x: Tensor) -> Tensor:
        """(B, 3, T, H, W) -> the posterior's (mean, logvar) (B, 32, T', H/8, W/8), sliced as the reference."""
        t, n = x.shape[2], self.slice_frames
        try:
            if t - 1 > n:
                parts = [self.encoder(x[:, :, :1 + n], FIRST)]
                parts += [self.encoder(x[:, :, a:a + n], NEXT) for a in range(1 + n, t, n)]
                return torch.cat(parts, dim=2)
            return self.encoder(x, None)
        finally:
            self.clear_memory()

    def encode(self, x: Tensor, generator: torch.Generator | None = None, sample: bool = True,
               tile: int | tuple[int, int] | None = None, overlap: int = 128, check=None) -> Tensor:
        """(B, 3, 1 + 4n, H, W) in [-1, 1] -> scaled latents (B, 16, 1 + n, H/8, W/8): a sample of the posterior,
        as the reference's ``vae_encode`` (``mean`` with ``sample`` off). ``tile`` (pixels, multiples of 8; a size
        or (height, width)): encode in overlapping tiles that size, blended (to bound memory on big frames)."""
        x = x.to(self.dtype)
        th, tw = (tile, tile) if isinstance(tile, int) else tile or (0, 0)
        if tile and (x.shape[3] > th or x.shape[4] > tw):
            h = tiled(self._moments, x, tile, overlap, align=8, factor=1 / 8, check=check)
        else:
            h = self._moments(x)
        mean, logvar = h.float().chunk(2, dim=1)
        del h
        if sample:
            std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
            noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
            mean = mean + std * noise
        return mean * self.scaling_factor

    def _decode_slices(self, z: Tensor):
        t = z.shape[2]
        try:
            if t - 1 > 1:
                yield self.decoder(z[:, :, :2], FIRST)
                for a in range(2, t):
                    yield self.decoder(z[:, :, a:a + 1], NEXT)
            else:
                yield self.decoder(z, None)
        finally:
            self.clear_memory()

    def decode_slices(self, z: Tensor, tile: int | tuple[int, int] | None = None, overlap: int = 16, check=None):
        """Scaled latents (B, 16, 1 + n, h, w) -> frames (B, 3, ., 8h, 8w) in about [-1, 1], yielded slice by slice
        (5 frames, then 4 per latent frame), as the reference's sliced ``decode``. ``tile`` (latent pixels; a size or
        (height, width)): decode in overlapping tiles that size, blended, all yielded at once."""
        z = (z / self.scaling_factor).to(self.dtype)
        th, tw = (tile, tile) if isinstance(tile, int) else tile or (0, 0)
        if tile and (z.shape[3] > th or z.shape[4] > tw):
            yield tiled(lambda part: torch.cat(list(self._decode_slices(part)), dim=2), z, tile, overlap, align=1,
                        factor=8, check=check)
        else:
            yield from self._decode_slices(z)

    def decode(self, z: Tensor, **kw) -> Tensor:
        return torch.cat(list(self.decode_slices(z, **kw)), dim=2)


def spans(n: int, tile: int, overlap: int, align: int = 1) -> list[tuple[int, int]]:
    """(start, end) of tiles at most ``tile`` long covering [0, n), neighbours sharing at least ``overlap`` (at most
    half a tile); starts on multiples of ``align`` (``tile`` and ``overlap`` should be too)."""
    if n <= tile:
        return [(0, n)]
    overlap = min(overlap, tile // 2 // align * align)  # so each tile moves on by at least half its length
    count = max(2, math.ceil((n - overlap) / (tile - overlap)))
    out = []
    for i in range(count):
        a = min(n - tile, round(i * (n - tile) / (count - 1)))
        a -= a % align
        out.append((a, min(n, a + tile)))
    out[-1] = (out[-1][0], n)
    return out


def fit_tiles(h: int, w: int, area: float, overlap: int, align: int) -> tuple[int, int] | None:
    """The tile (height, width) of at most ``area`` pixels, multiples of ``align``, that covers an h x w frame
    with the least work (tiles' total area, overlaps counted twice); None when the frame fits whole."""
    if h * w <= area:
        return None
    best = None
    for ny in range(1, 33):
        th = _round_up(math.ceil((h + (ny - 1) * overlap) / ny), align)
        if ny > 1 and th <= 2 * overlap:
            break
        for nx in range(1, 33):
            tw = _round_up(math.ceil((w + (nx - 1) * overlap) / nx), align)
            if nx > 1 and tw <= 2 * overlap:
                break
            if th * tw <= area:
                cost = ny * th * nx * tw
                if best is None or cost < best[0]:
                    best = (cost, (min(th, h), min(tw, w)))
                break
    if best is None:  # even the smallest tiles are too big: take them anyway
        return (min(h, _round_up(2 * overlap + align, align)), min(w, _round_up(2 * overlap + align, align)))
    return best[1]


def _round_up(n: int, k: int) -> int:
    return -(-n // k) * k


def ramp_weights(parts: list[tuple[int, int]], n: int, scale: float) -> list[Tensor]:
    """Each tile's blend weight along one axis (in output units, ``scale`` x the input's): rising over its overlap
    with the previous tile, falling over the next; overlapping neighbours sum to 1."""
    out = []
    for i, (a, b) in enumerate(parts):
        a, b = round(a * scale), round(b * scale)
        w = torch.ones(b - a)
        pos = torch.arange(a, b, dtype=torch.float32) + 0.5
        if i > 0:
            pe = round(parts[i - 1][1] * scale)
            if pe > a:
                w = torch.minimum(w, ((pos - a) / (pe - a)).clamp(0, 1))
        if i < len(parts) - 1:
            ns = round(parts[i + 1][0] * scale)
            if ns < b:
                w = torch.minimum(w, ((b - pos) / (b - ns)).clamp(0, 1))
        out.append(w)
    return out


def tiled(fn, x: Tensor, tile: int | tuple[int, int], overlap: int, align: int, factor: float, check=None) -> Tensor:
    """``fn`` on overlapping spatial tiles of (B, C, T, H, W) (each at most ``tile``, or (height, width)), its
    outputs (``factor`` x the size) blended with linear ramps across the overlaps. Accumulated in fp32."""
    h, w = x.shape[3:]
    th, tw = (tile, tile) if isinstance(tile, int) else tile
    ys, xs = spans(h, th, overlap, align), spans(w, tw, overlap, align)
    wy, wx = ramp_weights(ys, h, factor), ramp_weights(xs, w, factor)
    out = None
    for (y0, y1), my in zip(ys, wy):
        for (x0, x1), mx in zip(xs, wx):
            if check:
                check()
            part = fn(x[:, :, :, y0:y1, x0:x1])
            if out is None:
                out = torch.zeros(*part.shape[:3], round(h * factor), round(w * factor), dtype=torch.float32,
                                  device=part.device)
            oy, ox = round(y0 * factor), round(x0 * factor)
            mask = (my[:, None] * mx[None, :]).to(part.device)
            out[:, :, :, oy:oy + part.shape[3], ox:ox + part.shape[4]] += part.float() * mask
            del part
    return out


def frames_for_latents(latent_frames: int) -> int:
    return 4 * (latent_frames - 1) + 1


def padded_length(t: int) -> int:
    """The reference's ``cut_videos``: a clip's length padded to 4n + 1 (a single frame stays one; 2 to 4 frames
    become 5), by repeating its last frame."""
    if t == 1:
        return 1
    return max(5, 4 * math.ceil((t - 1) / 4) + 1)

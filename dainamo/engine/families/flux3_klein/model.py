"""FLUX 3 Image Klein DiT ("f3-image-small", 5.6B parameters).

Ported from Black Forest Labs' reference inference code, ``flux3`` (``src/flux3/dit.py``, Apache License 2.0).
See THIRD_PARTY_NOTICES.md.

Tokens come in three modes: ``image`` (the noisy target's latents), ``image_cond`` (reference images' latents, for
edits) and ``text`` (the Qwen3-VL-4B features). Each mode is embedded by its own input layer and runs through its
own 5 blocks, attending only within itself (``content_mode_blocks.image``, ``content_mode_blocks.image_cond``,
``txt_mode_blocks``); then each sample's tokens are joined (image, image_cond, text) and run through 28 shared
``single_blocks``. Samples never attend to each other. Every block is a parallel attention + SwiGLU block under one
shift / scale / gate, which comes from its mode's modulation of the timestep embedding (one set for the mode blocks,
``early_stream_modulations``, one for the shared blocks, ``single_stream_modulations``). Only the image tokens are
read out (``final_layer.image``): the velocity, noise - clean.

Changes for Dainamo (the arithmetic is the reference's):
  * q, k and v are one ``qkv`` Linear per block (rows [q; k; v]), the layout of the ComfyUI files, so module names
    are the checkpoint's keys and LoRAs saved on them load in ComfyUI;
  * a mode's timestep embedding and modulations are computed once per sample and repeated over its tokens (every
    token of a mode in a sample shares its timestep in every use of the reference);
  * a mode a pack has no tokens of is skipped, rather than run on a placeholder token;
  * samples of equal length attend in one batched call;
  * per-block gradient checkpointing; sizes from a config, so tests can build a small one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

MODES = ("image", "image_cond", "text")  # the order a sample's tokens take in the shared blocks
MOD_KEYS = {"image": "image", "image_cond": "image_cond", "text": "txt"}  # the checkpoint's names for the modes
FLOW_MAP_KEY = "ref_time_in.in_layer.weight"  # only the distilled (flow-map) checkpoint has it


@dataclass
class Flux3Config:
    hidden: int = 3072
    heads: int = 24
    depth: int = 5  # blocks of each mode
    depth_single: int = 28
    mlp_ratio: float = 3.0
    in_channels: int = 128  # FLUX.2 VAE latents, packed 2x2
    text_channels: int = 20480  # 8 Qwen3-VL-4B layers x 2560
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    theta: int = 10000
    time_dim: int = 256


@dataclass
class Pack:
    """Several samples' tokens, mode by mode. ``data[m]``: (N_m, C_m) every sample's tokens of mode m, one sample
    after another; ``ids[m]``: (N_m, 4) their RoPE positions; ``spans[m][s]``: (start, end) of sample s in
    ``data[m]`` (start == end: none); ``timesteps[m]``: (S,) each sample's noise level for its mode-m tokens
    (1 = noise; references and text are clean, 0)."""

    data: dict[str, Tensor]
    ids: dict[str, Tensor]
    spans: dict[str, list[tuple[int, int]]]
    timesteps: dict[str, Tensor]

    @property
    def num_samples(self) -> int:
        return len(self.spans["image"])


def timestep_embedding(t: Tensor, dim: int = 256, max_period: int = 10000,
                       dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """The sinusoidal embedding of 1000 * t. The reference keeps timesteps in bf16 and scales them there (``dtype``),
    then embeds in fp32 and returns ``dtype``."""
    t = 1000.0 * t.to(dtype)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """pos (L,) -> rotation matrices (L, dim // 2, 2, 2), angles in float64."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    angles = pos.double()[:, None] / (theta ** scale)[None]
    out = torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_embed(ids: Tensor, axes_dim: tuple[int, ...], theta: int) -> Tensor:
    """ids (L, 4) -> (L, head_dim // 2, 2, 2): each axis rotates its own share of the head's pairs."""
    with torch.autocast(device_type=ids.device.type, enabled=False):
        return torch.cat([rope(ids[:, i], d, theta) for i, d in enumerate(axes_dim)], dim=-3)


def rmsnorm_rope(x: Tensor, scale: Tensor, freqs: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm (weight ``scale``) then RoPE over (heads, L, D), in fp32; returns x's dtype."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        h, length, d = x.shape
        xf = x.float()
        xn = xf * xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * scale.float()
        pairs = xn.reshape(h, length, d // 2, 1, 2)
        rotated = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        return rotated.reshape(h, length, d).to(x.dtype)


def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """(1 + scale) * LayerNorm(x) + shift (no affine, eps 1e-6), in fp32; returns x's dtype."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6)
        return ((1.0 + scale.float()) * xf + shift.float()).to(x.dtype)


def attention(q: Tensor, k: Tensor, v: Tensor, bounds: list[tuple[int, int]]) -> Tensor:
    """(heads, L, D) each -> (L, heads * D), every sample (``bounds``: its (start, end)) attending to itself only."""
    heads, length, dim = q.shape
    lengths = {b - a for a, b in bounds}
    if len(lengths) == 1 and bounds[0][0] == 0 and bounds[-1][1] == length:  # equal lengths, back to back: one call
        n = len(bounds)
        split = lambda t: t.reshape(heads, n, length // n, dim).transpose(0, 1)  # noqa: E731
        out = F.scaled_dot_product_attention(split(q), split(k), split(v))  # (n, heads, l, D)
        return out.permute(0, 2, 1, 3).reshape(length, heads * dim)
    parts = []
    for a, b in bounds:
        if b > a:
            out = F.scaled_dot_product_attention(q[None, :, a:b], k[None, :, a:b], v[None, :, a:b])
            parts.append(out[0].transpose(0, 1).reshape(b - a, heads * dim))
    return torch.cat(parts)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(self.lin(F.silu(vec)).chunk(3, dim=-1))  # shift, scale, gate


class LastLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(modulate(x, scale, shift))


class Block(nn.Module):
    """Attention and a SwiGLU MLP side by side, both on the modulated input, under one gate."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, 2 * mlp_hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def forward(self, x: Tensor, freqs: Tensor, bounds: list[tuple[int, int]], shift: Tensor, scale: Tensor,
                gate: Tensor) -> Tensor:
        x_mod = modulate(x, scale, shift)
        length = x.shape[0]
        q, k, v = self.qkv(x_mod).view(length, 3, self.heads, -1).permute(1, 2, 0, 3)  # 3 x (heads, L, D)
        q = rmsnorm_rope(q, self.norm.query_norm.scale, freqs)
        k = rmsnorm_rope(k, self.norm.key_norm.scale, freqs)
        attn = attention(q, k, v.to(q.dtype), bounds)
        gate_in, up = self.mlp_in(x_mod).chunk(2, dim=-1)
        return x + gate * (self.attn_out(attn) + self.mlp_out(F.silu(gate_in) * up))


class Flux3KleinDiT(nn.Module):
    def __init__(self, config: Flux3Config):
        super().__init__()
        c = self.config = config
        self.compute_dtype = torch.bfloat16
        # the dtype timesteps are rounded to before they're embedded (the reference's bf16; tests compare in fp32)
        self.timestep_dtype = torch.bfloat16
        self.gradient_checkpointing = False
        block = lambda: Block(c.hidden, c.heads, c.mlp_ratio)  # noqa: E731
        self.emb_in = nn.ModuleDict({m: nn.Linear(c.in_channels, c.hidden, bias=False)
                                     for m in ("image", "image_cond")})
        self.txt_in = nn.Linear(c.text_channels, c.hidden, bias=False)
        self.time_in = MLPEmbedder(c.time_dim, c.hidden)
        self.early_stream_modulations = nn.ModuleDict({k: ModeModulation(c.hidden) for k in MOD_KEYS.values()})
        self.single_stream_modulations = nn.ModuleDict({k: ModeModulation(c.hidden) for k in MOD_KEYS.values()})
        self.content_mode_blocks = nn.ModuleDict({
            m: nn.ModuleList([block() for _ in range(c.depth)]) for m in ("image", "image_cond")})
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(c.depth)])
        self.single_blocks = nn.ModuleList([block() for _ in range(c.depth_single)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(c.hidden, c.in_channels)})

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    def _block(self, block: Block, x: Tensor, freqs: Tensor, bounds, mod) -> Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, x, freqs, bounds, *mod, use_reentrant=False)
        return block(x, freqs, bounds, *mod)

    def forward(self, pack: Pack) -> Tensor:
        """The velocity of the image tokens, (N_image, in_channels) in ``pack.data["image"]``'s order."""
        c, dtype = self.config, self.compute_dtype
        modes = [m for m in MODES if pack.data[m].shape[0] > 0]
        device = pack.data["image"].device
        lengths = {m: torch.tensor([b - a for a, b in pack.spans[m]], device=device) for m in modes}

        def per_token(t: Tensor, m: str) -> Tensor:  # (S, ...) -> one row per token of mode m
            return t.repeat_interleave(lengths[m], dim=0)

        emb = {m: timestep_embedding(pack.timesteps[m].to(device), c.time_dim, dtype=self.timestep_dtype).to(dtype)
               for m in modes}
        vec = {m: self.time_in(emb[m]) for m in modes}  # (S, hidden)
        early = {m: tuple(per_token(p, m) for p in self.early_stream_modulations[MOD_KEYS[m]](vec[m])) for m in modes}
        single = {m: tuple(per_token(p, m) for p in self.single_stream_modulations[MOD_KEYS[m]](vec[m]))
                  for m in modes}
        x = {m: (self.txt_in if m == "text" else self.emb_in[m])(pack.data[m].to(dtype)) for m in modes}

        for m in modes:  # each mode attends within itself (and each sample within itself)
            freqs = rope_embed(pack.ids[m].to(device), c.axes_dim, c.theta)
            blocks = self.txt_mode_blocks if m == "text" else self.content_mode_blocks[m]
            for block in blocks:
                x[m] = self._block(block, x[m], freqs, pack.spans[m], early[m])

        chunks, ids, mods, bounds, images, offset = [], [], ([], [], []), [], [], 0
        for s in range(pack.num_samples):
            start = offset
            for m in modes:
                a, b = pack.spans[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    ids.append(pack.ids[m][a:b].to(device))
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    if m == "image":
                        images.append((offset, offset + b - a))
                    offset += b - a
            bounds.append((start, offset))
        h = torch.cat(chunks)
        freqs = rope_embed(torch.cat(ids), c.axes_dim, c.theta)
        mods = tuple(torch.cat(p) for p in mods)
        for block in self.single_blocks:
            h = self._block(block, h, freqs, bounds, mods)
        image = torch.cat([h[a:b] for a, b in images])
        return self.final_layer["image"](image, per_token(vec["image"], "image"))

"""FLUX.2 Klein DiT (4B and 9B).

Ported from Black Forest Labs' FLUX.2 reference code, ``src/flux2/model.py`` (``Flux2`` with ``Klein4BParams`` /
``Klein9BParams``; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

A Flux-style MMDiT on FLUX.2's packed 128-channel latent tokens: ``double_blocks`` (separate image and text
streams, joint attention) then ``single_blocks`` (one stream, parallel attention + MLP), with the modulations
shared by every block of a kind (``double_stream_modulation_img/txt``, ``single_stream_modulation``) and 4-axis
RoPE over (t, h, w, l) position ids. Klein has no guidance embedding and no biases.

Changes for Dainamo:
  * One training forward. The reference's KV-cache paths (``forward_kv_extract`` / ``forward_kv_cached``) are left
    out; reference-image tokens come in as ``ref``. By default they are appended to the image tokens and attend
    like them, as the reference's ``denoise`` / ``denoise_cfg`` run the Klein checkpoints with references.
    ``causal_refs`` lays them out as the 9B KV checkpoint was trained (``forward_kv_extract`` without the cache):
    before the image tokens, modulated at timestep 0, attending only to themselves (``causal_attn_fn``).
  * Per-block gradient checkpointing (``gradient_checkpointing``).
  * Sizes are read from a checkpoint's tensor shapes (``Flux2KleinConfig.from_shapes``).
  * RoPE angles are computed in float64 (as diffusers' FLUX.2 port does), then used in float32 as the reference.
Module and parameter names are the reference's, which are the checkpoints' keys, so files load as they are and
LoRAs saved on these names load in ComfyUI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

TIME_DIM = 256  # sinusoidal timestep features


@dataclass
class Flux2KleinConfig:
    in_channels: int = 128
    context_in_dim: int = 7680  # 3 Qwen3 layers' hidden states side by side
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5  # double blocks
    depth_single_blocks: int = 20
    mlp_ratio: float = 3.0
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    theta: int = 2000

    @property
    def mlp_hidden(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    @classmethod
    def from_shapes(cls, shapes: dict[str, list[int]]) -> "Flux2KleinConfig":
        """The config of a checkpoint from its {key: shape} (as many blocks as it has; quantized weights keep
        their output dimension, which is all that's read of them)."""
        if "guidance_in.in_layer.weight" in shapes:
            raise ValueError("This is FLUX.2 [dev] (it has a guidance embedding), not FLUX.2 Klein")
        if "img_in.weight" not in shapes or "txt_in.weight" not in shapes:
            raise ValueError("Not a FLUX.2 Klein diffusion model (no img_in / txt_in)")
        hidden, in_channels = shapes["img_in.weight"]
        norm = next((k for k in ("single_blocks.0.norm.query_norm.scale", "double_blocks.0.img_attn.norm.query_norm.scale")
                     if k in shapes), None)
        if norm is None:
            raise ValueError("Not a FLUX.2 Klein diffusion model (no attention blocks)")
        head_dim = shapes[norm][0]

        def count(prefix):
            idx = [int(k.split(".")[1]) for k in shapes if k.startswith(prefix + ".")]
            return 1 + max(idx) if idx else 0

        if "single_blocks.0.linear1.weight" in shapes:
            mlp_hidden = (shapes["single_blocks.0.linear1.weight"][0] - 3 * hidden) // 2
        else:
            mlp_hidden = shapes["double_blocks.0.img_mlp.0.weight"][0] // 2
        return cls(in_channels=in_channels, context_in_dim=shapes["txt_in.weight"][1], hidden_size=hidden,
                   num_heads=hidden // head_dim, depth=count("double_blocks"),
                   depth_single_blocks=count("single_blocks"), mlp_ratio=mlp_hidden / hidden,
                   axes_dim=(head_dim // 4,) * 4)


KLEIN_4B = Flux2KleinConfig()
KLEIN_9B = Flux2KleinConfig(context_in_dim=12288, hidden_size=4096, num_heads=32, depth=8, depth_single_blocks=24)


# --- pieces -------------------------------------------------------------------
def timestep_embedding(t: Tensor, dim: int = TIME_DIM, max_period: float = 10000, time_factor: float = 1000.0) -> Tensor:
    """Sinusoidal features of ``time_factor * t`` (cos then sin), in t's dtype."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype if t.is_floating_point() else torch.float32)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """(B, L) positions -> (B, L, dim/2, 2, 2) rotation matrices."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = pos.to(torch.float64)[..., None] * omega
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, num_txt: int = 0, num_causal_refs: int = 0) -> Tensor:
    """RoPE then scaled dot-product attention over (B, H, L, D) -> (B, L, H*D). With ``num_causal_refs``, the tokens
    right after the text attend only to each other (the reference's ``causal_attn_fn``); all others see everything."""
    q, k = apply_rope(q, k, pe)
    if not num_causal_refs:
        out = F.scaled_dot_product_attention(q, k, v)
    else:
        s, e = num_txt, num_txt + num_causal_refs
        rest = F.scaled_dot_product_attention(torch.cat([q[:, :, :s], q[:, :, e:]], 2), k, v)
        refs = F.scaled_dot_product_attention(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e])
        out = torch.cat([rest[:, :, :s], refs, rest[:, :, s:]], 2)
    b, h, n, d = out.shape
    return out.transpose(1, 2).reshape(b, n, h * d)


def split_heads(qkv: Tensor, num_heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """(B, L, 3*H*D) -> q, k, v (B, H, L, D) ("B L (K H D) -> K B H L D")."""
    b, n, _ = qkv.shape
    qkv = qkv.view(b, n, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, ...]):
        super().__init__()
        self.dim, self.theta, self.axes_dim = dim, theta, tuple(axes_dim)

    def forward(self, ids: Tensor) -> Tensor:
        """(B, L, n_axes) ids -> (B, 1, L, dim/2, 2, 2)."""
        emb = torch.cat([rope(ids[..., i], d, self.theta) for i, d in enumerate(self.axes_dim)], dim=-3)
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6).to(x.dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SelfAttention(nn.Module):
    """The weights of one stream's attention (its maths is in the blocks, which attend across streams)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUActivation(nn.Module):
    """SwiGLU's gate: silu(first half) * second half."""

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=False)

    def forward(self, vec: Tensor):
        """(B, D) -> ((shift, scale, gate), second (shift, scale, gate) or None), each (B, 1, D)."""
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


def _blend(mods: tuple[Tensor, ...], ref_mods: tuple[Tensor, ...], start: int, count: int, length: int):
    """Per-token modulations: ``ref_mods`` for tokens start..start+count, ``mods`` for the rest (B, length, D)."""
    out = []
    for m, r in zip(mods, ref_mods):
        b = m.shape[0]
        m = m.expand(b, length, -1)
        out.append(torch.cat([m[:, :start], r.expand(b, count, -1), m[:, start + count:]], 1))
    return tuple(out)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden * 2, bias=False), SiLUActivation(),
                                     nn.Linear(mlp_hidden, hidden_size, bias=False))
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden * 2, bias=False), SiLUActivation(),
                                     nn.Linear(mlp_hidden, hidden_size, bias=False))

    def forward(self, img: Tensor, txt: Tensor, pe: Tensor, mod_img, mod_txt, num_causal_refs: int = 0):
        """``pe``: the text's then the image's RoPE; ``mod_img`` / ``mod_txt``: ((shift, scale, gate) x 2)."""
        (i_shift1, i_scale1, i_gate1), (i_shift2, i_scale2, i_gate2) = mod_img
        (t_shift1, t_scale1, t_gate1), (t_shift2, t_scale2, t_gate2) = mod_txt
        iq, ik, iv = split_heads(self.img_attn.qkv((1 + i_scale1) * self.img_norm1(img) + i_shift1), self.num_heads)
        iq, ik = self.img_attn.norm(iq, ik, iv)
        tq, tk, tv = split_heads(self.txt_attn.qkv((1 + t_scale1) * self.txt_norm1(txt) + t_shift1), self.num_heads)
        tq, tk = self.txt_attn.norm(tq, tk, tv)
        n_txt = txt.shape[1]
        attn = attention(torch.cat((tq, iq), 2), torch.cat((tk, ik), 2), torch.cat((tv, iv), 2), pe, n_txt,
                         num_causal_refs)
        txt_attn, img_attn = attn[:, :n_txt], attn[:, n_txt:]
        img = img + i_gate1 * self.img_attn.proj(img_attn)
        img = img + i_gate2 * self.img_mlp((1 + i_scale2) * self.img_norm2(img) + i_shift2)
        txt = txt + t_gate1 * self.txt_attn.proj(txt_attn)
        txt = txt + t_gate2 * self.txt_mlp((1 + t_scale2) * self.txt_norm2(txt) + t_shift2)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def forward(self, x: Tensor, pe: Tensor, mod, num_txt: int, num_causal_refs: int = 0) -> Tensor:
        shift, scale, gate = mod
        qkv, mlp = torch.split(self.linear1((1 + scale) * self.pre_norm(x) + shift),
                               [3 * self.hidden_size, self.mlp_hidden_dim * 2], dim=-1)
        q, k, v = split_heads(qkv, self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe, num_txt, num_causal_refs)
        return x + gate * self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec)[:, None, :].chunk(2, dim=-1)
        return self.linear((1 + scale) * self.norm_final(x) + shift)


class Flux2KleinDiT(nn.Module):
    def __init__(self, config: Flux2KleinConfig):
        super().__init__()
        self.config = config
        c = config
        if c.hidden_size % c.num_heads:
            raise ValueError(f"hidden size {c.hidden_size} isn't divisible by {c.num_heads} heads")
        head_dim = c.hidden_size // c.num_heads
        if sum(c.axes_dim) != head_dim:
            raise ValueError(f"RoPE axes {c.axes_dim} don't add up to the head size {head_dim}")
        self.gradient_checkpointing = False
        self.pe_embedder = EmbedND(head_dim, c.theta, c.axes_dim)
        self.img_in = nn.Linear(c.in_channels, c.hidden_size, bias=False)
        self.time_in = MLPEmbedder(TIME_DIM, c.hidden_size)
        self.txt_in = nn.Linear(c.context_in_dim, c.hidden_size, bias=False)
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(c.hidden_size, c.num_heads, c.mlp_ratio) for _ in range(c.depth)])
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(c.hidden_size, c.num_heads, c.mlp_ratio) for _ in range(c.depth_single_blocks)])
        self.double_stream_modulation_img = Modulation(c.hidden_size, double=True)
        self.double_stream_modulation_txt = Modulation(c.hidden_size, double=True)
        self.single_stream_modulation = Modulation(c.hidden_size, double=False)
        self.final_layer = LastLayer(c.hidden_size, c.in_channels)

    def _run(self, block: nn.Module, *args):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, timesteps: Tensor,
                ref: Tensor | None = None, ref_ids: Tensor | None = None, causal_refs: bool = False) -> Tensor:
        """Velocity of the image tokens.

        img (B, N, 128) noisy image tokens with ``img_ids`` (B, N, 4) (t, h, w, l) positions; txt (B, T,
        context_in_dim) with ``txt_ids`` (B, T, 4); timesteps (B,) noise levels (1 = noise); ``ref`` (B, R, 128)
        clean reference-image tokens with ``ref_ids``. -> (B, N, 128)."""
        n_img, n_txt = img.shape[1], txt.shape[1]
        n_ref = 0 if ref is None else ref.shape[1]
        vec = self.time_in(timestep_embedding(timesteps))
        mod_img = self.double_stream_modulation_img(vec)
        mod_txt = self.double_stream_modulation_txt(vec)
        mod_single, _ = self.single_stream_modulation(vec)
        causal = n_ref if causal_refs else 0
        if n_ref and causal_refs:  # [ref, img], the references modulated as clean
            img, img_ids = torch.cat([ref, img], 1), torch.cat([ref_ids, img_ids], 1)
            ref_vec = self.time_in(timestep_embedding(torch.zeros_like(timesteps)))
            ref_img = self.double_stream_modulation_img(ref_vec)
            ref_single, _ = self.single_stream_modulation(ref_vec)
            length = n_ref + n_img
            mod_img = tuple(_blend(m, r, 0, n_ref, length) for m, r in zip(mod_img, ref_img))
            mod_single = _blend(mod_single, ref_single, n_txt, n_ref, n_txt + length)
        elif n_ref:  # [img, ref], as the reference's denoise appends them
            img, img_ids = torch.cat([img, ref], 1), torch.cat([img_ids, ref_ids], 1)

        x = self.img_in(img)
        t = self.txt_in(txt)
        pe = torch.cat((self.pe_embedder(txt_ids), self.pe_embedder(img_ids)), dim=2)
        for block in self.double_blocks:
            x, t = self._run(block, x, t, pe, mod_img, mod_txt, causal)
        x = torch.cat((t, x), 1)
        for block in self.single_blocks:
            x = self._run(block, x, pe, mod_single, n_txt, causal)
        x = x[:, n_txt + causal: n_txt + causal + n_img]
        return self.final_layer(x, vec)

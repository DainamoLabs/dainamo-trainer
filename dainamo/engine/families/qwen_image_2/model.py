"""Qwen-Image 2.1 single-stream DiT.

Ported from diffusers' ``QwenImage21Transformer2DModel`` (Copyright 2026 Qwen-Image Team and The
HuggingFace Team, Apache License 2.0) as vendored by ai-toolkit (Ostris, LLC — MIT),
``extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py``. See THIRD_PARTY_NOTICES.md.

Changes for Dainamo: a plain ``nn.Module`` (no diffusers mixins); the fused ComfyUI ``img_mlp.gate_up``
projection as in ai-toolkit, so ComfyUI checkpoints load as-is and LoRA keys match ComfyUI's module
names; attention is the exact multi-pass SDPA form of the block-causal mask (the flex-attention and
prefix KV-cache inference paths are not ported); RoPE tables are built lazily per device.

Text and image tokens share one sequence. Qwen3-VL reserves an ``<|image_pad|>`` slot per 32 px of
each reference image; here every slot becomes the 2x2 latent tokens of that image. The target
image's tokens are appended last. Attention is causal over text, bidirectional within each image
block, and every token sees everything before it. With ``causal_condition`` the text and reference
tokens are modulated at t = 0, so only the target image sees the sampled timestep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.normalization import RMSNorm
from torch.utils.checkpoint import checkpoint

IMG_TOKENS_PER_SLOT = 4  # one vision-language slot = 2x2 latent tokens


@dataclass
class QwenImage21Config:
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True

    @classmethod
    def from_shapes(cls, shapes: dict[str, list[int]]) -> "QwenImage21Config":
        """Read the sizes that vary between checkpoints from tensor shapes."""
        layers = 1 + max(int(k.split(".")[1]) for k in shapes if k.startswith("transformer_blocks."))
        inner, in_ch = shapes["img_in.weight"]
        head_dim = shapes["transformer_blocks.0.attn.norm_q.weight"][0]
        return cls(in_channels=in_ch, out_channels=shapes["proj_out.weight"][0], num_layers=layers,
                   attention_head_dim=head_dim, num_attention_heads=inner // head_dim,
                   context_in_dim=shapes["txt_in.in_layer.weight"][1],
                   mlp_ratio=shapes["transformer_blocks.0.img_mlp.out.weight"][1] // inner)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x (B, S, H, D) with interleaved (re, im) pairs; freqs (S, D/2) complex."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs.unsqueeze(1)).flatten(3).type_as(x)


class Timesteps(nn.Module):
    """Sinusoidal embedding of 1000 * t, cos half first."""

    def __init__(self, dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.dim, self.max_period, self.time_factor = dim, max_period, time_factor

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = (self.time_factor * t.float())[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb.to(t.dtype)


class TimestepProj(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.time_proj = Timesteps(256)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim, sample_proj_bias=False)

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.timestep_embedder(self.time_proj(t).to(dtype))


class ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose stored weight is ``scale - 1``; computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (self.weight.float() + 1)).to(dt)


class TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden: int, eps: float):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class SwiGLU(nn.Module):
    """``gate_up`` is [gate; up] concatenated on the out dim (ComfyUI layout)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * up)


def select_rows(params: torch.Tensor, target_mask: torch.Tensor | None) -> torch.Tensor:
    """Per-sample modulation over the token axis. With ``causal_condition`` ``params`` has B + 1 rows,
    the last one from t = 0: target tokens read their sample's row, every other token the t = 0 row."""
    if target_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_mask.view(1, -1, 1), real, zero)


class FinalNorm(nn.Module):
    """Adaptive LayerNorm, scale only."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False, bias=False)

    def forward(self, x, temb, target_mask):
        scale = select_rows(self.linear(F.silu(temb).to(x.dtype)), target_mask)
        return self.norm(x) * (1 + scale)


def prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> list[tuple[int, int, bool]]:
    """(start, end, is_text) runs of equal image id over the prefix (everything before the target)."""
    ids = image_ids[:prefix_len].tolist()
    out, start = [], 0
    for i in range(1, prefix_len + 1):
        if i == prefix_len or ids[i] != ids[start]:
            out.append((start, i, ids[start] < 0))
            start = i
    return out


def block_causal_attention(q, k, v, segments, key_valid):
    """q, k, v (B, H, S, D). Each prefix segment attends to all keys up to its own end (text segments
    causally within themselves); the target block attends to everything. ``key_valid`` (B, S) drops
    padded text keys. Exact; equivalent to the (q >= k) | same_image_block mask."""
    outs = []
    prefix_len = segments[-1][1] if segments else 0
    for start, end, is_text in segments:
        mask = None
        if is_text:
            n = end - start
            mask = torch.cat([torch.ones(n, start, dtype=torch.bool, device=q.device),
                              torch.ones(n, n, dtype=torch.bool, device=q.device).tril()], dim=1)[None, None]
        if key_valid is not None:
            kv = key_valid[:, None, None, :end]
            mask = kv if mask is None else (mask & kv)
        outs.append(F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, :end], v[:, :, :end], attn_mask=mask))
    mask = None if key_valid is None else key_valid[:, None, None, :]
    outs.append(F.scaled_dot_product_attention(q[:, :, prefix_len:], k, v, attn_mask=mask))
    return torch.cat(outs, dim=2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)

    def forward(self, x, rope, segments, key_valid):
        q = self.to_q(x).unflatten(-1, (self.heads, -1))
        k = self.to_k(x).unflatten(-1, (self.heads, -1))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        q = self.norm_q(q).to(v.dtype)
        k = self.norm_k(k).to(v.dtype)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        out = block_causal_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), segments, key_valid)
        out = out.transpose(1, 2).flatten(2, 3).type_as(q)
        return self.to_out[0](out)


class Block(nn.Module):
    """Single-stream block; its scales and gates come from the model's one shared modulation."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: int, eps: float):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, heads, head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLU(dim, dim * mlp_ratio)

    @staticmethod
    def _modulate(x, params, target_mask):
        scale, gate = params.chunk(2, dim=-1)
        return x * (1 + select_rows(scale, target_mask)), select_rows(gate, target_mask)

    def forward(self, x, modulation, rope, target_mask, segments, key_valid):
        mod1, mod2 = modulation.chunk(2, dim=-1)
        h, gate1 = self._modulate(self.img_norm1(x), mod1, target_mask)
        x = x + gate1.tanh() * self.attn(h, rope, segments, key_valid)
        h, gate2 = self._modulate(self.img_norm2(x), mod2, target_mask)
        x = x + gate2.tanh() * self.img_mlp(h)
        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)
        return x


class QwenImage21Rope:
    """3-axis (frame, height, width) RoPE over the joint sequence. Text advances one shared position on
    all axes; each image block sits at the frame position its preceding text reached, on a height/width
    grid centred on zero, and then advances the position by max(h, w)."""

    def __init__(self, theta: int, axes_dim: tuple[int, int, int]):
        self.theta, self.axes_dim = theta, axes_dim
        self.tables: dict[torch.device, list[torch.Tensor]] = {}

    def _table(self, device) -> list[torch.Tensor]:
        if device not in self.tables:
            pos = torch.arange(8192)
            neg = torch.arange(1024).flip(0) * -1 - 1
            out = []
            for dim in self.axes_dim:
                inv = 1.0 / torch.pow(self.theta, torch.arange(0, dim, 2).float().div(dim))
                f = torch.cat([torch.outer(pos, inv), torch.outer(neg, inv)])
                out.append(torch.polar(torch.ones_like(f), f).to(device))
            self.tables[device] = out
        return self.tables[device]

    def __call__(self, img_shapes, image_pad_mask: torch.Tensor) -> torch.Tensor:
        dev = image_pad_mask.device
        frame, img_h, img_w = [], [], []
        cursor = position = 0
        total = image_pad_mask.shape[-1]
        is_img = image_pad_mask.tolist()
        for _, h, w in img_shapes:
            start = is_img.index(True, cursor)
            frame.extend(range(position, position + start - cursor))
            position += start - cursor
            cursor = start + h * w
            frame.extend([position] * (h * w))
            position += max(h, w)
            img_h.extend(r for r in range(-(h - h // 2), h // 2) for _ in range(w))
            img_w.extend(c for _ in range(h) for c in range(-(w - w // 2), w // 2))
        if cursor < total:
            frame.extend(range(position, position + total - cursor))
        frame = torch.tensor(frame, dtype=torch.long, device=dev)
        hi, wi = frame.clone(), frame.clone()
        hi[image_pad_mask] = torch.tensor(img_h, dtype=torch.long, device=dev)
        wi[image_pad_mask] = torch.tensor(img_w, dtype=torch.long, device=dev)
        t = self._table(dev)
        return torch.cat([t[0][frame], t[1][hi], t[2][wi]], dim=-1)


class QwenImage21DiT(nn.Module):
    def __init__(self, cfg: QwenImage21Config):
        super().__init__()
        self.cfg = cfg
        inner = cfg.num_attention_heads * cfg.attention_head_dim
        self.pos_embed = QwenImage21Rope(10000, cfg.axes_dims_rope)
        self.time_text_embed = TimestepProj(inner)
        self.txt_in = TextProjection(cfg.context_in_dim, inner, cfg.eps)
        self.img_in = nn.Linear(cfg.in_channels, inner, bias=False)
        # one modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate]
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(inner, 4 * inner, bias=False))
        self.transformer_blocks = nn.ModuleList(
            Block(inner, cfg.num_attention_heads, cfg.attention_head_dim, cfg.mlp_ratio, cfg.eps)
            for _ in range(cfg.num_layers))
        self.norm_out = FinalNorm(inner, cfg.eps)
        self.proj_out = nn.Linear(inner, cfg.out_channels, bias=False)
        self.gradient_checkpointing = False

    @staticmethod
    def token_metadata(image_pad_mask, img_shapes):
        """(image_ids, target_mask): -1 at text, a unique id per image block; True at the target's tokens.
        Block boundaries come from ``img_shapes``, so adjacent references stay separate blocks."""
        positions = image_pad_mask.nonzero(as_tuple=True)[0]
        lengths = [math.prod(s) for s in img_shapes]
        if sum(lengths) != positions.numel():
            raise ValueError(f"img_shapes account for {sum(lengths)} image tokens but the mask marks {positions.numel()}")
        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        image_ids[positions] = torch.repeat_interleave(torch.arange(len(lengths), device=positions.device),
                                                       torch.tensor(lengths, device=positions.device))
        target = torch.zeros_like(image_pad_mask)
        target[positions[-lengths[-1]:]] = True
        return image_ids, target

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask,
                encoder_hidden_states_mask=None):
        """hidden_states (B, N, C): reference latents in slot order, then the target's. encoder_hidden_states
        (B, L, ctx). timestep (B,) in [0, 1] (1 = noise). img_shapes: [(1, h, w), ...] in latent tokens,
        references first, shared by the batch. img_mask (B, L + target/4): True at image slots.
        Returns (B, joint_len, C); the target's velocity is the last h*w rows."""
        b = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        text = self.txt_in(encoder_hidden_states)

        repeats = torch.where(img_mask, IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        target_tokens = math.prod(img_shapes[-1])
        joint = torch.cat([text, text.new_zeros(b, target_tokens // IMG_TOKENS_PER_SLOT, text.shape[2])], dim=1)
        joint = joint.repeat_interleave(repeats, dim=1)
        joint[:, image_pad_mask] = hidden_states

        rope = self.pos_embed(img_shapes, image_pad_mask)
        image_ids, target_mask = self.token_metadata(image_pad_mask, img_shapes)

        timestep = timestep.to(hidden_states.dtype)
        if self.cfg.causal_condition:  # extra t = 0 row for text and references
            timestep = torch.cat([timestep, timestep.new_zeros(1)])
            mod_mask = target_mask
        else:
            mod_mask = None
        temb = self.time_text_embed(timestep, hidden_states.dtype)
        modulation = self.modulation(temb)

        key_valid = None
        if encoder_hidden_states_mask is not None:
            # text positions of the joint sequence line up, in order, with the non-slot positions of
            # the vision-language sequence
            key_valid = torch.ones(b, image_pad_mask.shape[0], dtype=torch.bool, device=joint.device)
            text_pos = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            key_valid[:, text_pos] = encoder_hidden_states_mask.bool()[:, vlm_text]
        segments = prefix_segments(image_ids, int((~target_mask).sum()))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                joint = checkpoint(block, joint, modulation, rope, mod_mask, segments, key_valid, use_reentrant=False)
            else:
                joint = block(joint, modulation, rope, mod_mask, segments, key_valid)

        return self.proj_out(self.norm_out(joint, temb, mod_mask))

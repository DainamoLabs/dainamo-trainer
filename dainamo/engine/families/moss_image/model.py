"""MOSS-Image DiT.

Ported from OpenMOSS-Team/MOSS-Image-SFT ``pipeline_moss_image.py`` (Copyright 2026 OpenMOSS,
Apache License 2.0). See THIRD_PARTY_NOTICES.md.

Changes for Dainamo:
  * q/k/v are one fused ``proj.qkv`` Linear ([q; k; v]), the layout of ComfyUI's single-file
    checkpoints, so those load as-is and saved LoRAs use the same names.
  * The compute dtype is a model attribute instead of being read from ``.weight`` (base weights
    may be int8/fp8 here).
  * Optional per-block gradient checkpointing; text conditioning is passed as a list of
    per-sample tensors (no padding), which is what the reference does internally.
  * RoPE bands are a non-persistent fp32 buffer, always derived (the reference re-derives them
    on load for the same reason: bf16 would round the frequencies).
The maths of every module is the reference's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class MossConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            "qkv": nn.Linear(dim, 3 * dim, bias=False),
            "o": nn.Linear(dim, dim, bias=False),
            "ffn_up_gate": nn.Linear(dim, 2 * ffn_dim, bias=False),
            "ffn_down": nn.Linear(ffn_dim, dim, bias=False),
        })

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = (self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]).to(x.dtype)
        q, k, v = self.proj["qkv"](y).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(q, k, v, rope, self.heads)) * gate_a[:, None]
        y = (self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]).to(x.dtype)
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base

    def bands(self, device):
        # derived in fp32 on the CPU, as in training, then moved
        exponent = torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def forward(self, height, width, text_length, device):
        # coordinates (r, f, h, w); sequence order is image followed by text
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands(device)
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MossImageDiT(nn.Module):
    def __init__(self, config: MossConfig):
        super().__init__()
        c = self.config = config
        self.compute_dtype = torch.bfloat16
        self.gradient_checkpointing = False
        self.video_embedder = nn.Linear(c.latent_channels, c.dim)
        self.text_embedder = TextProjection(c.text_dim, c.dim, c.ffn_dim, c.num_heads)
        self.rope = RoPE(c.dim // c.num_heads, c.rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(c.freq_dim, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim))
        self.shared_block_modulation = nn.Linear(c.dim, 6 * c.dim)
        self.layers = nn.ModuleList([Block(c.dim, c.ffn_dim, c.num_heads) for _ in range(c.num_layers)])
        self.final_norm_video = RMSNorm(c.dim)
        self.final_linear_video = nn.Linear(c.dim, c.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(c.dim, 2 * c.dim)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, texts: list[torch.Tensor],
                skip_layers: frozenset[int] | tuple = ()) -> torch.Tensor:
        """hidden_states (B, 128, H, W) packed latents; timestep (B,) = 1000 * sigma;
        texts: B tensors (L_i, text_dim) at their true lengths."""
        batch, _, height, width = hidden_states.shape
        dtype = self.compute_dtype
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
        # the sinusoid is computed in float64 and fed to the MLP in float32, as in training
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)
        outputs = []
        for i in range(batch):
            context = texts[i].to(hidden_states.device, dtype)[None]
            video = hidden_states[i : i + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(dtype).contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            m = mod[i : i + 1]
            for index, layer in enumerate(self.layers):
                if index in skip_layers:
                    continue
                if self.gradient_checkpointing and torch.is_grad_enabled():
                    x = checkpoint(layer, x, rope, m, use_reentrant=False)
                else:
                    x = layer(x, rope, m)
            x = self.final_norm_video(x[:, : height * width])
            x = x * (1 + head[i : i + 1, 1, None]) + head[i : i + 1, 0, None]
            x = self.final_linear_video(x.to(dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)

"""SeedVR2 3B DiT ("NaDiT"), ByteDance's one-step video restorer.

Ported from ByteDance Seed's reference code, SeedVR (github.com/ByteDance-Seed/SeedVR at e4de8c2, Apache License
2.0: ``models/dit_v2`` and ``configs_3b/main.yaml``). See THIRD_PARTY_NOTICES.md.

A clip's latents (T, h, w, 33: the noise, the low-res clip's latents and a mask of ones) are cut into 2x2 patches,
one token each, and run with the fixed text embedding (``positive_conditioning``, 58 tokens) through 32 blocks.
The first 10 keep separate weights for the video and text tokens (``.vid`` / ``.txt``), the other 22 share them
(``.all``). Each block is RMSNorm, a shift / scale from the timestep embedding plus the block's own learned vectors
(``ada``), attention, a gate, the residual; then the same around a SwiGLU MLP. Attention is windowed: the video
tokens are cut into windows about 1/9 of a 720p frame's area (and a quarter of the clip's length, at most 30 latent
frames), alternately offset by half a window (``win`` / ``swin``, as in Swin), and every window attends to itself
and to all the text tokens; a text token's output is the mean over the windows. RoPE is 3D for the video tokens
(t, h, w, each rotating 21 of the head's 64 pairs) at positions local to the window, the time axis starting after
the text; text tokens take their index on all three axes. The output is the velocity (noise - clean).

Changes for Dainamo (the arithmetic is the reference's):
  * module names are the checkpoint's keys (ComfyUI's single-file repack keeps the reference's names), so the
    shared loader assigns them directly;
  * one sample at a time (the reference packs several of different sizes); windows of the same shape attend in one
    batched SDPA call instead of flash-attn's varlen kernel;
  * the per-token work (norms, projections, MLP) runs in slices of ``chunk`` tokens, so a long clip's activations
    stay a few residual-sized buffers;
  * RoPE angles are computed in fp32 from the formula (``1 / 10000 ** (2i / 42)``, what the checkpoint's
    ``attn.rope.rope.freqs`` hold, rounded to fp16 in this file); norms and modulation in fp32.

One quirk kept on purpose: the reference caches each modulation's per-token expansion under a key that the output
layer's modulation shares with the blocks' attention one, so the output layer is shifted and scaled by the
timestep embedding's *attention* shift / scale plus its own ``vid_out_ada`` vectors. The checkpoint was trained
that way (the output layer's own slice of the embedding wouldn't even have the right width). Likewise the last
block skips the text branch of its modulation and MLP, so there the video attends to the text's plain norm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

WIN, SWIN = "720pwin_by_size_bysize", "720pswin_by_size_bysize"  # the reference's window methods, alternating


@dataclass
class SeedVR2Config:
    in_channels: int = 33  # noise latent (16) + condition latent (16) + mask (1)
    out_channels: int = 16
    dim: int = 2560
    text_dim: int = 5120
    heads: int = 20
    head_dim: int = 128
    layers: int = 32
    mm_layers: int = 10  # the first blocks keep separate video / text weights
    mlp_hidden: int = 6912
    patch: int = 2  # spatial; one latent frame per patch
    windows: tuple[int, int, int] = (4, 3, 3)  # windows per (t, h, w) at 720p
    eps: float = 1e-5
    time_dim: int = 256
    theta: float = 10000.0

    @property
    def rope_axis(self) -> int:
        """Dims each of the three RoPE axes rotates (42 of 128: the last 2 are left alone)."""
        return self.head_dim // 3

    @classmethod
    def from_state_dict(cls, sd: dict[str, Tensor]) -> "SeedVR2Config":
        """The config of a checkpoint, from its tensor shapes."""
        dim, patch_in = sd["vid_in.proj.weight"].shape
        layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        mm = sum(1 for i in range(layers) if f"blocks.{i}.attn.proj_qkv.vid.weight" in sd)
        b = "vid" if mm else "all"
        head_dim = sd[f"blocks.0.attn.norm_q.{b}.weight"].shape[0]
        qkv = sd[f"blocks.0.attn.proj_qkv.{b}.weight"].shape[0]
        return cls(in_channels=patch_in // 4, out_channels=sd["vid_out.proj.weight"].shape[0] // 4, dim=dim,
                   text_dim=sd["txt_in.weight"].shape[1], heads=qkv // 3 // head_dim, head_dim=head_dim,
                   layers=layers, mm_layers=mm, mlp_hidden=sd[f"blocks.0.mlp.{b}.proj_in.weight"].shape[0],
                   time_dim=sd["emb_in.proj_in.weight"].shape[1])


# ---------------------------------------------------------------- pieces


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """x / sqrt(mean(x^2) + eps) (times ``weight``), in fp32."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * weight.float() if weight is not None else xf


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """The sinusoidal embedding the reference takes from diffusers (sin first, no frequency shift), in fp32."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def rope_freqs(axis_dim: int, theta: float) -> Tensor:
    """The rotation frequencies of one RoPE axis (``axis_dim // 2`` of them), in fp32."""
    return 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float64)[: axis_dim // 2] / axis_dim)).float()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate (n, heads, head_dim) by (n, pairs) angles: interleaved pairs (2i, 2i+1), the leading 2 * pairs
    dims; the rest pass through. In fp32; returns x's dtype."""
    n, h, d = x.shape
    p = cos.shape[-1]
    xf = x.float()
    pairs = xf[..., : 2 * p].reshape(n, h, p, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    c, s = cos[:, None], sin[:, None]
    rotated = torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1).reshape(n, h, 2 * p)
    return torch.cat([rotated, xf[..., 2 * p:]], dim=-1).to(x.dtype)


def window_slices(size: tuple[int, int, int], num_windows: tuple[int, int, int],
                  shifted: bool) -> list[tuple[slice, slice, slice]]:
    """The reference's ``make_720Pwindows_bysize`` / ``make_shifted_720Pwindows_bysize``: windows over a
    (t, h, w) token grid, sized as ``num_windows`` would cut it at 720p's area (45 x 80 tokens) and at most 30
    latent frames; shifted by half a window along each axis that has more than one. Order: w, then h, then t."""
    t, h, w = size
    nt0, nh0, nw0 = num_windows
    scale = math.sqrt((45 * 80) / (h * w))
    rh, rw = round(h * scale), round(w * scale)
    wh, ww = math.ceil(rh / nh0), math.ceil(rw / nw0)
    wt = math.ceil(min(t, 30) / nt0)
    if not shifted:
        nt, nh, nw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
        return [(slice(it * wt, min((it + 1) * wt, t)), slice(ih * wh, min((ih + 1) * wh, h)),
                 slice(iw * ww, min((iw + 1) * ww, w)))
                for iw in range(nw) if min((iw + 1) * ww, w) > iw * ww
                for ih in range(nh) if min((ih + 1) * wh, h) > ih * wh
                for it in range(nt) if min((it + 1) * wt, t) > it * wt]
    st, sh, sw = (0.5 if wt < t else 0), (0.5 if wh < h else 0), (0.5 if ww < w else 0)
    nt, nh, nw = math.ceil((t - st) / wt), math.ceil((h - sh) / wh), math.ceil((w - sw) / ww)
    nt, nh, nw = (nt + 1 if st > 0 else 1), (nh + 1 if sh > 0 else 1), (nw + 1 if sw > 0 else 1)

    def span(i: int, s: float, size_: int, n: int) -> tuple[int, int]:
        return max(int((i - s) * size_), 0), min(int((i - s + 1) * size_), n)

    out = []
    for iw in range(nw):
        aw, bw = span(iw, sw, ww, w)
        if bw <= aw:
            continue
        for ih in range(nh):
            ah, bh = span(ih, sh, wh, h)
            if bh <= ah:
                continue
            for it in range(nt):
                at, bt = span(it, st, wt, t)
                if bt > at:
                    out.append((slice(at, bt), slice(ah, bh), slice(aw, bw)))
    return out


@dataclass
class WindowPlan:
    """How one window method cuts a clip's tokens: ``groups`` of same-shaped windows ((n, L) token indices, in
    each window's (t, h, w) order), every token's RoPE ``cos`` / ``sin`` at its window-local position (N, 63),
    and the number of windows (a text token's output is its mean over them)."""
    groups: list[Tensor]
    cos: Tensor
    sin: Tensor
    count: int


def plan_windows(grid: tuple[int, int, int], method: str, windows: tuple[int, int, int], text_len: int,
                 freqs: Tensor, device: torch.device) -> WindowPlan:
    t, h, w = grid
    index = torch.arange(t * h * w).view(t, h, w)
    slices = window_slices(grid, windows, shifted=method == SWIN)
    shapes: dict[tuple[int, int, int], list[Tensor]] = {}
    pairs = freqs.shape[0]
    cos = torch.empty(t * h * w, 3 * pairs)
    sin = torch.empty(t * h * w, 3 * pairs)
    tables: dict[tuple[int, int, int], tuple[Tensor, Tensor]] = {}
    for st, sh, sw in slices:
        ids = index[st, sh, sw]
        shape = tuple(ids.shape)
        shapes.setdefault(shape, []).append(ids.reshape(-1))
        if shape not in tables:  # window-local positions; time starts after the text's positions
            gt, gh, gw = torch.meshgrid(torch.arange(shape[0]) + text_len, torch.arange(shape[1]),
                                        torch.arange(shape[2]), indexing="ij")
            ang = torch.cat([p.reshape(-1, 1).float() * freqs[None] for p in (gt, gh, gw)], dim=-1)
            tables[shape] = (torch.cos(ang), torch.sin(ang))
        c, s = tables[shape]
        cos[ids.reshape(-1)] = c
        sin[ids.reshape(-1)] = s
    groups = [torch.stack(v).to(device) for v in shapes.values()]
    return WindowPlan(groups, cos.to(device), sin.to(device), len(slices))


def text_rope(length: int, freqs: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    """Text tokens take their index on all three axes."""
    ang = torch.arange(length, dtype=torch.float32)[:, None] * freqs[None]
    ang = ang.repeat(1, 3)
    return torch.cos(ang).to(device), torch.sin(ang).to(device)


def window_attention(q: Tensor, k: Tensor, v: Tensor, tq: Tensor, tk: Tensor, tv: Tensor, plan: WindowPlan,
                     max_tokens: int = 65536) -> tuple[Tensor, Tensor]:
    """Video (N, heads, D) and text (L, heads, D) queries / keys / values -> outputs: each window attends to its
    tokens and all the text; a text token's output is the mean of its outputs over the windows."""
    out = torch.empty_like(q)
    length = tq.shape[0]
    text_sum = torch.zeros(tq.shape, dtype=torch.float32, device=q.device)
    for idx in plan.groups:
        n, win = idx.shape
        step = max(1, max_tokens // (win + length))
        for a in range(0, n, step):
            ids = idx[a:a + step]
            m = ids.shape[0]

            def join(x: Tensor, tx: Tensor) -> Tensor:  # (m, win + L, heads, D) -> (m, heads, win + L, D)
                return torch.cat([x[ids], tx.expand(m, *tx.shape)], dim=1).transpose(1, 2)

            o = F.scaled_dot_product_attention(join(q, tq), join(k, tk), join(v, tv)).transpose(1, 2)
            out[ids.reshape(-1)] = o[:, :win].reshape(-1, *q.shape[1:])
            text_sum += o[:, win:].float().sum(0)
    return out, (text_sum / plan.count).to(q.dtype)


# ---------------------------------------------------------------- modules (names are the checkpoint's keys)


class Branches(nn.Module):
    """A module per token stream (``vid``, ``txt``), or one both share (``all``): the reference's MMModule."""

    def __init__(self, make, shared: bool):
        super().__init__()
        self.shared = shared
        if shared:
            self.all = make()
        else:
            self.vid = make()
            self.txt = make()

    def __getitem__(self, branch: str) -> nn.Module:
        return self.all if self.shared else getattr(self, branch)


class Weight(nn.Module):
    """An RMSNorm's scale."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class AdaVectors(nn.Module):
    """A block's learned shift / scale / gate vectors, added to the timestep embedding's (``AdaSingle``)."""

    def __init__(self, dim: int, layers: tuple[str, ...] = ("attn", "mlp"), gates: bool = True):
        super().__init__()
        for name in layers:
            setattr(self, f"{name}_shift", nn.Parameter(torch.zeros(dim)))
            setattr(self, f"{name}_scale", nn.Parameter(torch.ones(dim)))
            if gates:
                setattr(self, f"{name}_gate", nn.Parameter(torch.zeros(dim)))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.proj_in_gate = nn.Linear(dim, hidden, bias=False)
        self.proj_out = nn.Linear(hidden, dim, bias=False)
        self.proj_in = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))


class _Freqs(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("freqs", torch.zeros(n))


class Rope(nn.Module):
    """Holds the checkpoint's ``attn.rope.rope.freqs`` (the angles come from the formula, ``rope_freqs``)."""

    def __init__(self, n: int):
        super().__init__()
        self.rope = _Freqs(n)


class Attention(nn.Module):
    def __init__(self, c: SeedVR2Config, shared: bool):
        super().__init__()
        inner = c.heads * c.head_dim
        self.proj_qkv = Branches(lambda: nn.Linear(c.dim, 3 * inner, bias=False), shared)
        self.proj_out = Branches(lambda: nn.Linear(inner, c.dim), shared)
        self.norm_q = Branches(lambda: Weight(c.head_dim), shared)
        self.norm_k = Branches(lambda: Weight(c.head_dim), shared)
        self.rope = Rope(c.rope_axis // 2)


@dataclass
class Modulation:
    """The timestep embedding's shift / scale / gate for the attention and MLP halves, (dim,) each, fp32."""
    attn: tuple[Tensor, Tensor, Tensor]
    mlp: tuple[Tensor, Tensor, Tensor]


class Block(nn.Module):
    def __init__(self, c: SeedVR2Config, shared: bool, last: bool):
        super().__init__()
        self.c = c
        # In the last block the reference skips the text branch of ``ada`` and ``mlp`` (MMModule's vid_only): the
        # text's keys and values come from its unmodulated norm, and nothing reads the text after the block.
        self.last = last
        self.attn = Attention(c, shared)
        self.mlp = Branches(lambda: SwiGLU(c.dim, c.mlp_hidden), shared)
        self.ada = Branches(lambda: AdaVectors(c.dim), shared)

    def _modulated(self, x: Tensor, branch: str, half: str, mod: Modulation) -> Tensor:
        ada = self.ada[branch]
        shift, scale, _ = getattr(mod, half)
        out = rms_norm(x, None, self.c.eps) * (scale + getattr(ada, f"{half}_scale")) \
            + (shift + getattr(ada, f"{half}_shift"))
        return out.to(x.dtype)

    def _gated(self, y: Tensor, branch: str, half: str, mod: Modulation) -> Tensor:
        return (y.float() * (getattr(mod, half)[2] + getattr(self.ada[branch], f"{half}_gate"))).to(y.dtype)

    def _qkv(self, x: Tensor, branch: str, mod: Modulation, cos: Tensor, sin: Tensor, modulate: bool = True):
        c = self.c
        h = self._modulated(x, branch, "attn", mod) if modulate else rms_norm(x, None, c.eps).to(x.dtype)
        qkv = self.attn.proj_qkv[branch](h).view(-1, 3, c.heads, c.head_dim)
        q = apply_rope(rms_norm(qkv[:, 0], self.attn.norm_q[branch].weight, c.eps).to(x.dtype), cos, sin)
        k = apply_rope(rms_norm(qkv[:, 1], self.attn.norm_k[branch].weight, c.eps).to(x.dtype), cos, sin)
        return q, k, qkv[:, 2]

    def forward(self, vid: Tensor, txt: Tensor, mod: Modulation, plan: WindowPlan, txt_rope: tuple[Tensor, Tensor],
                chunk: int) -> tuple[Tensor, Tensor]:
        """vid (N, dim), txt (L, dim) -> the same; ``vid`` is updated in place."""
        c = self.c
        n = vid.shape[0]
        q = vid.new_empty(n, c.heads, c.head_dim)
        k, v = torch.empty_like(q), torch.empty_like(q)
        for a in range(0, n, chunk):
            s = slice(a, min(a + chunk, n))
            q[s], k[s], v[s] = self._qkv(vid[s], "vid", mod, plan.cos[s], plan.sin[s])
        tq, tk, tv = self._qkv(txt, "txt", mod, *txt_rope, modulate=not self.last)
        out, txt_out = window_attention(q, k, v, tq, tk, tv, plan)
        del q, k, v
        out = out.view(n, -1)
        for a in range(0, n, chunk):
            s = slice(a, min(a + chunk, n))
            vid[s] += self._gated(self.attn.proj_out["vid"](out[s]), "vid", "attn", mod)
            h = self._modulated(vid[s], "vid", "mlp", mod)
            vid[s] += self._gated(self.mlp["vid"](h), "vid", "mlp", mod)
        if self.last:
            return vid, txt
        txt = txt + self._gated(self.attn.proj_out["txt"](txt_out.reshape(txt.shape[0], -1)), "txt", "attn", mod)
        txt = txt + self._gated(self.mlp["txt"](self._modulated(txt, "txt", "mlp", mod)), "txt", "mlp", mod)
        return vid, txt


class PatchLinear(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.proj = nn.Linear(cin, cout)


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, hidden: int, out: int):
        super().__init__()
        self.time_dim = time_dim
        self.proj_in = nn.Linear(time_dim, hidden)
        self.proj_hid = nn.Linear(hidden, hidden)
        self.proj_out = nn.Linear(hidden, out)

    def forward(self, t: Tensor, dtype: torch.dtype) -> Tensor:
        emb = timestep_embedding(t, self.time_dim).to(dtype)
        return self.proj_out(F.silu(self.proj_hid(F.silu(self.proj_in(emb)))))


class SeedVR2DiT(nn.Module):
    def __init__(self, config: SeedVR2Config | None = None):
        super().__init__()
        c = self.config = config or SeedVR2Config()
        self.compute_dtype = torch.bfloat16
        p = c.patch
        self.vid_in = PatchLinear(c.in_channels * p * p, c.dim)
        self.txt_in = nn.Linear(c.text_dim, c.dim)
        self.emb_in = TimeEmbedding(c.time_dim, c.dim, 6 * c.dim)
        self.blocks = nn.ModuleList([Block(c, shared=i >= c.mm_layers, last=i == c.layers - 1)
                                     for i in range(c.layers)])
        self.vid_out_norm = Weight(c.dim)
        self.vid_out_ada = AdaVectors(c.dim, layers=("out",), gates=False)
        self.vid_out = PatchLinear(c.dim, c.out_channels * p * p)

    def modulation(self, timestep: float, device: torch.device) -> Modulation:
        """The timestep embedding, split as the reference's AdaSingle reads it: (dim, 2 halves, shift/scale/gate)."""
        emb = self.emb_in(torch.tensor([float(timestep)], device=device), self.compute_dtype)
        emb = emb.float().view(self.config.dim, 2, 3)
        return Modulation(attn=tuple(emb[:, 0].unbind(-1)), mlp=tuple(emb[:, 1].unbind(-1)))

    def forward(self, vid: Tensor, txt: Tensor, timestep: float = 1000.0, chunk: int = 16384) -> Tensor:
        """vid (T, h, w, in_channels) latents, txt (L, text_dim) -> (T, h, w, out_channels)."""
        c, dtype = self.config, self.compute_dtype
        p = c.patch
        t, h, w, ch = vid.shape
        grid = (t, h // p, w // p)
        x = vid.to(dtype).view(t, h // p, p, w // p, p, ch).permute(0, 1, 3, 2, 4, 5).reshape(-1, p * p * ch)
        x = self.vid_in.proj(x)
        txt = self.txt_in(txt.to(dtype))
        mod = self.modulation(timestep, x.device)
        freqs = rope_freqs(c.rope_axis, c.theta)
        plans = {m: plan_windows(grid, m, c.windows, txt.shape[0], freqs, x.device) for m in (WIN, SWIN)}
        trope = text_rope(txt.shape[0], freqs, x.device)
        for i, block in enumerate(self.blocks):
            x, txt = block(x, txt, mod, plans[WIN if i % 2 == 0 else SWIN], trope, chunk)
        out = torch.empty(x.shape[0], self.vid_out.proj.out_features, dtype=dtype, device=x.device)
        shift, scale, _ = mod.attn  # the reference's shared cache entry (see the module docstring)
        for a in range(0, x.shape[0], chunk):
            s = slice(a, min(a + chunk, x.shape[0]))
            y = rms_norm(x[s], self.vid_out_norm.weight, c.eps) * (scale + self.vid_out_ada.out_scale) \
                + (shift + self.vid_out_ada.out_shift)
            out[s] = self.vid_out.proj(y.to(dtype))
        oc = c.out_channels
        return out.view(t, h // p, w // p, p, p, oc).permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, oc)

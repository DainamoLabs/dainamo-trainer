"""SeedVR2 (the 3B upscaler): the DiT against ByteDance's reference code on a small config (dual-stream and shared
blocks, windows and shifted windows, the output layer's shared modulation), RoPE and the window cuts, the causal
VAE's convs, shuffles and slicing, its tiles, the 4n + 1 padding, the color fix, the one-step schedule, the
upscaler end to end on tiny models (passes that overlap and crossfade, its options checked and clamped), and that
the real files' tensors are the modules' parameters.

The reference pieces below are copied from ByteDance Seed's SeedVR (https://github.com/ByteDance-Seed/SeedVR at
e4de8c2, ``models/dit_v2``; Copyright (c) 2025 Bytedance Ltd. and/or its affiliates, Apache License 2.0) and from
rotary-embedding-torch 0.8.9 (Phil Wang, MIT License), trimmed to one GPU-less process: no sequence parallelism,
flash-attn's varlen call as a loop of SDPA calls, apex's fused RMSNorm as its formula. They're used only here, as
the yardstick.

    .venv/bin/python -m pytest tests/test_seedvr2.py -v
"""

import math
import os
from itertools import chain

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from dainamo.engine import comfy_io
from dainamo.engine.upscale import Info
from dainamo.engine.upscalers.seedvr2 import model as M
from dainamo.engine.upscalers.seedvr2 import pipeline as P
from dainamo.engine.upscalers.seedvr2 import vae as V

FOLDER = "/home/william/AI_APPS/COMFY_CACHE/models/upscale_models"
DIT_FILE = f"{FOLDER}/seedvr2_3b_fp16.safetensors"
VAE_FILE = f"{FOLDER}/seedvr2_ema_vae_fp16.safetensors"


# --- the reference, small (SeedVR models/dit_v2 and rotary_embedding_torch) -------------------------------------
class Cache:
    def __init__(self, disable=False, prefix="", cache=None):
        self.cache = cache if cache is not None else {}
        self.disable = disable
        self.prefix = prefix

    def __call__(self, key, fn):
        if self.disable:
            return fn()
        key = self.prefix + key
        try:
            result = self.cache[key]
        except KeyError:
            result = fn()
            self.cache[key] = result
        return result

    def namespace(self, namespace):
        return Cache(disable=self.disable, prefix=self.prefix + namespace + ".", cache=self.cache)


def na_flatten(hid):
    shape = torch.stack([torch.tensor(x.shape[:-1], device=hid[0].device) for x in hid])
    hid = torch.cat([x.flatten(0, -2) for x in hid])
    return hid, shape


def na_unflatten(hid, hid_shape):
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    return [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]


def na_repeat_concat(vid, txt, vid_len, txt_len, txt_repeat):
    vid = torch.split(vid, vid_len.tolist())
    txt = torch.split(txt, txt_len.tolist())
    txt = [[x] * n for x, n in zip(txt, txt_repeat)]
    txt = list(chain(*txt))
    return torch.cat(list(chain(*zip(vid, txt))))


def na_repeat_concat_idx(vid_len, txt_len, txt_repeat):
    vid_idx = torch.arange(vid_len.sum())
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum())
    txt_repeat_list = txt_repeat.tolist()
    tgt_idx = na_repeat_concat(vid_idx, txt_idx, vid_len, txt_len, txt_repeat)
    src_idx = torch.argsort(tgt_idx)
    txt_idx_len = len(tgt_idx) - len(vid_idx)
    repeat_txt_len = (txt_len * txt_repeat).tolist()

    def unconcat_coalesce(all):
        vid_out, txt_out = all[src_idx].split([len(vid_idx), txt_idx_len])
        txt_out_coalesced = []
        for txt, repeat_time in zip(txt_out.split(repeat_txt_len), txt_repeat_list):
            txt = txt.reshape(-1, repeat_time, *txt.shape[1:]).mean(1)
            txt_out_coalesced.append(txt)
        return vid_out, torch.cat(txt_out_coalesced)

    return (lambda vid, txt: torch.cat([vid, txt])[tgt_idx], lambda all: unconcat_coalesce(all))


def na_window(hid, hid_shape, window_fn):
    hid = na_unflatten(hid, hid_shape)
    hid = list(map(window_fn, hid))
    hid_windows = torch.tensor(list(map(len, hid)))
    hid, hid_shape = na_flatten(list(chain(*hid)))
    return hid, hid_shape, hid_windows


def na_window_idx(hid_shape, window_fn):
    hid_idx = torch.arange(hid_shape.prod(-1).sum()).unsqueeze(-1)
    tgt_idx, tgt_shape, tgt_windows = na_window(hid_idx, hid_shape, window_fn)
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    return (lambda hid: torch.index_select(hid, 0, tgt_idx), lambda hid: torch.index_select(hid, 0, src_idx),
            tgt_shape, tgt_windows)


def make_720Pwindows_bysize(size, num_windows):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = math.ceil(resized_h / resized_nh), math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, 30) / resized_nt)
    nt, nh, nw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
    return [(slice(it * wt, min((it + 1) * wt, t)), slice(ih * wh, min((ih + 1) * wh, h)),
             slice(iw * ww, min((iw + 1) * ww, w)))
            for iw in range(nw) if min((iw + 1) * ww, w) > iw * ww
            for ih in range(nh) if min((ih + 1) * wh, h) > ih * wh
            for it in range(nt) if min((it + 1) * wt, t) > it * wt]


def make_shifted_720Pwindows_bysize(size, num_windows):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = math.ceil(resized_h / resized_nh), math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, 30) / resized_nt)
    st, sh, sw = (0.5 if wt < t else 0, 0.5 if wh < h else 0, 0.5 if ww < w else 0)
    nt, nh, nw = math.ceil((t - st) / wt), math.ceil((h - sh) / wh), math.ceil((w - sw) / ww)
    nt, nh, nw = (nt + 1 if st > 0 else 1, nh + 1 if sh > 0 else 1, nw + 1 if sw > 0 else 1)
    return [(slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
             slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
             slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)))
            for iw in range(nw) if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
            for ih in range(nh) if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
            for it in range(nt) if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)]


WINDOW_OPS = {M.WIN: make_720Pwindows_bysize, M.SWIN: make_shifted_720Pwindows_bysize}


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    t_left, t_middle, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    return torch.cat((t_left, t_transformed, t_right), dim=-1).type(dtype)


class RotaryEmbedding(nn.Module):  # "lang" frequencies only
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.register_buffer("freqs", 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)))

    def forward(self, t):
        freqs = torch.einsum("..., f -> ... f", t.type(self.freqs.dtype), self.freqs)
        return repeat(freqs, "... n -> ... (n r)", r=2)

    def get_axial_freqs(self, *dims):
        all_freqs = []
        for ind, dim in enumerate(dims):
            freqs = self.forward(torch.arange(dim))
            all_axis = [None] * len(dims)
            all_axis[ind] = slice(None)
            all_freqs.append(freqs[(Ellipsis, *all_axis, slice(None))])
        return torch.cat(torch.broadcast_tensors(*all_freqs), dim=-1)


class NaMMRotaryEmbedding3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rope = RotaryEmbedding(dim=dim // 3)
        self.mm = True

    def forward(self, vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache):
        vid_freqs, txt_freqs = cache("mmrope_freqs_3d", lambda: self.get_freqs(vid_shape, txt_shape))
        vid_q = rearrange(vid_q, "L h d -> h L d")
        vid_k = rearrange(vid_k, "L h d -> h L d")
        vid_q = apply_rotary_emb(vid_freqs, vid_q.float()).to(vid_q.dtype)
        vid_k = apply_rotary_emb(vid_freqs, vid_k.float()).to(vid_k.dtype)
        vid_q = rearrange(vid_q, "h L d -> L h d")
        vid_k = rearrange(vid_k, "h L d -> L h d")
        txt_q = rearrange(txt_q, "L h d -> h L d")
        txt_k = rearrange(txt_k, "L h d -> h L d")
        txt_q = apply_rotary_emb(txt_freqs, txt_q.float()).to(txt_q.dtype)
        txt_k = apply_rotary_emb(txt_freqs, txt_k.float()).to(txt_k.dtype)
        txt_q = rearrange(txt_q, "h L d -> L h d")
        txt_k = rearrange(txt_k, "h L d -> L h d")
        return vid_q, vid_k, txt_q, txt_k

    def get_freqs(self, vid_shape, txt_shape):
        vid_freqs = self.rope.get_axial_freqs(1024, 128, 128)
        txt_freqs = self.rope.get_axial_freqs(1024)
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq_list.append(vid_freqs[l: l + f, :h, :w].reshape(-1, vid_freqs.size(-1)))
            txt_freq_list.append(txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1)))
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


class MMArg:
    def __init__(self, vid, txt):
        self.vid, self.txt = vid, txt


def get_args(key, args):
    return [getattr(v, key) if isinstance(v, MMArg) else v for v in args]


def get_kwargs(key, kwargs):
    return {k: getattr(v, key) if isinstance(v, MMArg) else v for k, v in kwargs.items()}


class MMModule(nn.Module):
    def __init__(self, module, *args, shared_weights=False, vid_only=False, **kwargs):
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if self.shared_weights:
            self.all = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
        else:
            self.vid = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
            self.txt = module(*get_args("txt", args), **get_kwargs("txt", kwargs)) if not vid_only else None

    def forward(self, vid, txt, *args, **kwargs):
        vid_module = self.vid if not self.shared_weights else self.all
        vid = vid_module(vid, *get_args("vid", args), **get_kwargs("vid", kwargs))
        if not self.vid_only:
            txt_module = self.txt if not self.shared_weights else self.all
            txt = txt_module(txt, *get_args("txt", args), **get_kwargs("txt", kwargs))
        return vid, txt


class RMSNorm(nn.Module):  # apex's FusedRMSNorm
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.weight if self.weight is not None else y


class AdaSingle(nn.Module):
    def __init__(self, dim, emb_dim, layers, modes=("in", "out")):
        super().__init__()
        self.dim, self.emb_dim, self.layers = dim, emb_dim, layers
        for layer in layers:
            if "in" in modes:
                self.register_parameter(f"{layer}_shift", nn.Parameter(torch.randn(dim) / dim ** 0.5))
                self.register_parameter(f"{layer}_scale", nn.Parameter(torch.randn(dim) / dim ** 0.5 + 1))
            if "out" in modes:
                self.register_parameter(f"{layer}_gate", nn.Parameter(torch.randn(dim) / dim ** 0.5))

    def forward(self, hid, emb, layer, mode, cache=Cache(disable=True), branch_tag="", hid_len=None):
        idx = self.layers.index(layer)
        emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., idx, :]
        shape = emb.shape
        emb = emb.reshape(shape[:1] + (1,) * (hid.ndim + 1 - len(shape)) + shape[1:])
        if hid_len is not None:
            emb = cache(f"emb_repeat_{idx}_{branch_tag}",
                        lambda: torch.cat([e.repeat(n, *([1] * e.ndim)) for e, n in zip(emb, hid_len)]))
        shiftA, scaleA, gateA = emb.unbind(-1)
        shiftB, scaleB, gateB = (getattr(self, f"{layer}_shift", None), getattr(self, f"{layer}_scale", None),
                                 getattr(self, f"{layer}_gate", None))
        if mode == "in":
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            return hid.mul_(gateA + gateB)
        raise NotImplementedError


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, expand_ratio, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))


def varlen_attention(q, k, v, cu_seqlens):  # flash_attn_varlen_func, without its dtype
    out = []
    for a, b in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        o = F.scaled_dot_product_attention(q[None, a:b].transpose(1, 2), k[None, a:b].transpose(1, 2),
                                           v[None, a:b].transpose(1, 2))
        out.append(o[0].transpose(0, 1))
    return torch.cat(out)


class NaSwinAttention(nn.Module):
    def __init__(self, vid_dim, txt_dim, heads, head_dim, qk_norm_eps, rope_dim, shared_weights, window,
                 window_method):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        self.head_dim = head_dim
        self.proj_qkv = MMModule(nn.Linear, dim, inner_dim * 3, bias=False, shared_weights=shared_weights)
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(RMSNorm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True,
                               shared_weights=shared_weights)
        self.norm_k = MMModule(RMSNorm, dim=head_dim, eps=qk_norm_eps, elementwise_affine=True,
                               shared_weights=shared_weights)
        self.rope = NaMMRotaryEmbedding3d(dim=rope_dim)
        self.window, self.window_method = window, window_method
        self.window_op = WINDOW_OPS[window_method]

    def forward(self, vid, txt, vid_shape, txt_shape, cache):
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform", lambda: na_window_idx(vid_shape, make_window))
        vid_qkv_win = window_partition(vid_qkv)
        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na_repeat_concat_idx(vid_len_win, txt_len, window_count))
        _, num_h, _ = txt_q.shape
        txt_q_repeat = rearrange(txt_q, "l h d -> l (h d)")
        txt_q_repeat = na_unflatten(txt_q_repeat, txt_shape)
        txt_q_repeat = list(chain(*[[x] * n for x, n in zip(txt_q_repeat, window_count)]))
        txt_q_repeat, txt_shape_repeat = na_flatten(txt_q_repeat)
        txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=num_h)
        txt_k_repeat = rearrange(txt_k, "l h d -> l (h d)")
        txt_k_repeat = na_unflatten(txt_k_repeat, txt_shape)
        txt_k_repeat = list(chain(*[[x] * n for x, n in zip(txt_k_repeat, window_count)]))
        txt_k_repeat, _ = na_flatten(txt_k_repeat)
        txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=num_h)
        vid_q, vid_k, txt_q, txt_k = self.rope(vid_q, vid_k, window_shape, txt_q_repeat, txt_k_repeat,
                                               txt_shape_repeat, cache_win)
        cu = F.pad(all_len_win.cumsum(0), (1, 0)).int()
        out = varlen_attention(concat_win(vid_q, txt_q), concat_win(vid_k, txt_k), concat_win(vid_v, txt_v), cu)
        vid_out, txt_out = unconcat_win(out)
        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)
        return self.proj_out(vid_out, txt_out)


class NaMMSRTransformerBlock(nn.Module):
    def __init__(self, *, vid_dim, txt_dim, emb_dim, heads, head_dim, expand_ratio, norm_eps, shared_weights,
                 rope_dim, is_last_layer, window, window_method):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(RMSNorm, dim=dim, eps=norm_eps, elementwise_affine=False,
                                  shared_weights=shared_weights)
        self.attn = NaSwinAttention(vid_dim, txt_dim, heads, head_dim, norm_eps, rope_dim, shared_weights, window,
                                    window_method)
        self.mlp_norm = MMModule(RMSNorm, dim=dim, eps=norm_eps, elementwise_affine=False,
                                 shared_weights=shared_weights, vid_only=is_last_layer)
        self.mlp = MMModule(SwiGLUMLP, dim=dim, expand_ratio=expand_ratio, shared_weights=shared_weights,
                            vid_only=is_last_layer)
        self.ada = MMModule(AdaSingle, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"],
                            shared_weights=shared_weights, vid_only=is_last_layer)

    def forward(self, vid, txt, vid_shape, txt_shape, emb, cache):
        hid_len = MMArg(cache("vid_len", lambda: vid_shape.prod(-1)), cache("txt_len", lambda: txt_shape.prod(-1)))
        ada_kwargs = {"emb": emb, "hid_len": hid_len, "cache": cache, "branch_tag": MMArg("vid", "txt")}
        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)
        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        return (vid_mlp + vid_attn), (txt_mlp + txt_attn), vid_shape, txt_shape


class TimeEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim, hidden_dim, output_dim):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, timestep, device, dtype):
        from diffusers.models.embeddings import get_timestep_embedding
        emb = get_timestep_embedding(timesteps=timestep, embedding_dim=self.sinusoidal_dim, flip_sin_to_cos=False,
                                     downscale_freq_shift=0).to(dtype)
        return self.proj_out(self.act(self.proj_hid(self.act(self.proj_in(emb)))))


class Patch(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Linear(cin, cout)


class NaDiT(nn.Module):
    def __init__(self, vid_in_channels, vid_out_channels, vid_dim, txt_in_dim, heads, head_dim, expand_ratio,
                 num_layers, mm_layers, norm_eps=1e-5, window=(4, 3, 3)):
        super().__init__()
        emb_dim = 6 * vid_dim
        self.vid_in = Patch(vid_in_channels * 4, vid_dim)
        self.txt_in = nn.Linear(txt_in_dim, vid_dim)
        self.emb_in = TimeEmbedding(256, vid_dim, emb_dim)
        methods = num_layers // 2 * [M.WIN, M.SWIN]
        self.blocks = nn.ModuleList([NaMMSRTransformerBlock(
            vid_dim=vid_dim, txt_dim=vid_dim, emb_dim=emb_dim, heads=heads, head_dim=head_dim,
            expand_ratio=expand_ratio, norm_eps=norm_eps, shared_weights=not i < mm_layers, rope_dim=head_dim,
            is_last_layer=i == num_layers - 1, window=window, window_method=methods[i]) for i in range(num_layers)])
        self.vid_out_norm = RMSNorm(vid_dim, norm_eps, True)
        self.vid_out_ada = AdaSingle(dim=vid_dim, emb_dim=emb_dim, layers=["out"], modes=["in"])
        self.vid_out = Patch(vid_dim, vid_out_channels * 4)

    def forward(self, vid, txt, vid_shape, txt_shape, timestep):
        cache = Cache()
        txt = self.txt_in(txt)
        vid = na_unflatten(vid, vid_shape)
        vid = [rearrange(v, "(T t) (H h) (W w) c -> T H W (t h w c)", t=1, h=2, w=2) for v in vid]
        vid, vid_shape = na_flatten(vid)
        vid = self.vid_in.proj(vid)
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)
        for block in self.blocks:
            vid, txt, vid_shape, txt_shape = block(vid=vid, txt=txt, vid_shape=vid_shape, txt_shape=txt_shape,
                                                   emb=emb, cache=cache)
        vid = self.vid_out_norm(vid)
        vid = self.vid_out_ada(vid, emb=emb, layer="out", mode="in",
                               hid_len=cache("vid_len", lambda: vid_shape.prod(-1)), cache=cache, branch_tag="vid")
        vid = self.vid_out.proj(vid)
        vid = na_unflatten(vid, vid_shape)
        vid = [rearrange(v, "T H W (t h w c) -> (T t) (H h) (W w) c", t=1, h=2, w=2) for v in vid]
        return na_flatten(vid)[0]


# --- the DiT ------------------------------------------------------------------------------------------------------
SMALL = dict(in_channels=5, out_channels=2, dim=64, text_dim=24, heads=2, head_dim=32, layers=4, mm_layers=2,
             mlp_hidden=256)


def _pair(seed=0):
    """The reference and Dainamo's DiT on SMALL, sharing randomized weights (so every norm and vector counts)."""
    torch.manual_seed(seed)
    ref = NaDiT(vid_in_channels=SMALL["in_channels"], vid_out_channels=SMALL["out_channels"], vid_dim=SMALL["dim"],
                txt_in_dim=SMALL["text_dim"], heads=SMALL["heads"], head_dim=SMALL["head_dim"], expand_ratio=4,
                num_layers=SMALL["layers"], mm_layers=SMALL["mm_layers"]).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.2 * torch.randn_like(p))
    ours = M.SeedVR2DiT(M.SeedVR2Config(**SMALL)).eval()
    ours.compute_dtype = torch.float32
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return ref, ours


@pytest.mark.parametrize("grid", [(2, 40, 72), (1, 34, 60), (3, 6, 10)])
def test_dit_matches_reference(grid):
    """Several windows and shifted windows (the 40 x 72 latent grid cuts into 2 x 2 x 2 windows, then 3 x 3 x 3
    shifted), one frame, and a grid that's one window; text of 7 tokens."""
    ref, ours = _pair()
    t, h, w = grid
    vid = torch.randn(t, h, w, SMALL["in_channels"])
    txt = torch.randn(7, SMALL["text_dim"])
    with torch.no_grad():
        want = ref(vid.reshape(-1, vid.shape[-1]), txt, torch.tensor([[t, h, w]]), torch.tensor([[7]]),
                   torch.tensor([1000.0]))
        got = ours(vid, txt, 1000.0, chunk=500)  # chunks smaller than the clip
    assert got.shape == (t, h, w, SMALL["out_channels"])
    torch.testing.assert_close(got.reshape(-1, got.shape[-1]), want, rtol=1e-4, atol=1e-4)


def test_output_layer_uses_attention_modulation():
    """The reference's output layer reads the blocks' cached attention shift / scale (its own slice of the
    embedding would be 2 x dim wide): changing the embedding's attention half moves the output layer."""
    _, ours = _pair()
    mod = ours.modulation(1000.0, torch.device("cpu"))
    assert mod.attn[0].shape == (SMALL["dim"],) and mod.mlp[2].shape == (SMALL["dim"],)
    emb = ours.emb_in(torch.tensor([1000.0]), torch.float32).view(SMALL["dim"], 2, 3)
    torch.testing.assert_close(mod.attn[1], emb[:, 0, 1])  # (dim, layer, shift/scale/gate)
    torch.testing.assert_close(mod.mlp[0], emb[:, 1, 0])


def test_config_from_state_dict():
    ours = M.SeedVR2DiT(M.SeedVR2Config(**SMALL))
    c = M.SeedVR2Config.from_state_dict(ours.state_dict())
    assert (c.in_channels, c.out_channels, c.dim, c.text_dim, c.heads, c.head_dim, c.layers, c.mm_layers,
            c.mlp_hidden) == tuple(SMALL.values())


@pytest.mark.parametrize("size", [(1, 68, 120), (11, 68, 120), (1, 45, 80), (5, 23, 41), (40, 30, 30), (2, 3, 5)])
@pytest.mark.parametrize("method", [M.WIN, M.SWIN])
def test_windows_match_reference(size, method):
    got = M.window_slices(size, (4, 3, 3), shifted=method == M.SWIN)
    assert got == WINDOW_OPS[method](size, (4, 3, 3))
    cover = torch.zeros(size, dtype=torch.int)
    for st, sh, sw in got:
        cover[st, sh, sw] += 1
    assert (cover == 1).all()  # every token in exactly one window


def test_windows_at_1080p():
    """1080p (68 x 120 tokens): windows of 15 x 27 tokens, 5 x 5 of them; the shifted cut starts with 7 rows."""
    win = M.window_slices((1, 68, 120), (4, 3, 3), shifted=False)
    assert len(win) == 25 and (win[0][1], win[0][2]) == (slice(0, 15), slice(0, 27))
    shifted = M.window_slices((1, 68, 120), (4, 3, 3), shifted=True)
    assert shifted[0][1] == slice(0, 7) and shifted[0][2] == slice(0, 13)


def test_rope_matches_rotary_embedding():
    """Window-local 3D positions (time after the text), text on all three axes, 2 dims of 32 left alone."""
    head_dim, text_len, grid = 32, 7, (2, 40, 72)
    freqs = M.rope_freqs(head_dim // 3, 10000.0)
    torch.testing.assert_close(freqs, RotaryEmbedding(head_dim // 3).freqs)
    plan = M.plan_windows(grid, M.SWIN, (4, 3, 3), text_len, freqs, torch.device("cpu"))
    q = torch.randn(math.prod(grid), 2, head_dim)
    got = M.apply_rope(q, plan.cos, plan.sin)
    rope = NaMMRotaryEmbedding3d(head_dim)
    index = torch.arange(math.prod(grid)).view(grid)
    for st, sh, sw in make_shifted_720Pwindows_bysize(grid, (4, 3, 3)):
        ids = index[st, sh, sw].reshape(-1)
        shape = torch.tensor([index[st, sh, sw].shape])
        want, _, _, _ = rope(q[ids], q[ids], shape, q[:text_len], q[:text_len], torch.tensor([[text_len]]),
                             Cache(disable=True))
        torch.testing.assert_close(got[ids], want)
    tq = torch.randn(text_len, 2, head_dim)
    _, _, want, _ = rope(q[:1], q[:1], torch.tensor([[1, 1, 1]]), tq, tq, torch.tensor([[text_len]]),
                         Cache(disable=True))
    torch.testing.assert_close(M.apply_rope(tq, *M.text_rope(text_len, freqs, torch.device("cpu"))), want)


# --- the VAE ------------------------------------------------------------------------------------------------------
TINY_VAE = dict(channels=(32, 32, 64, 64), latent=4)


def _vae(seed=0):
    torch.manual_seed(seed)
    vae = V.SeedVR2VAE(**TINY_VAE).eval()
    with torch.no_grad():
        for m in vae.modules():
            if isinstance(m, V.CausalConv3d):
                nn.init.normal_(m.weight, std=1 / math.sqrt(m.weight[0].numel()))
                nn.init.normal_(m.bias, std=0.1)
            elif isinstance(m, nn.GroupNorm):
                nn.init.normal_(m.weight, 1, 0.1)
                nn.init.normal_(m.bias, 0, 0.1)
    return vae


def _reference_causal_conv(conv, x):
    """The reference's ``InflatedCausalConv3d`` on a whole clip: the first frame repeated 2 * temporal padding
    times in front, then the conv with its spatial padding."""
    if conv.temporal_pad:
        x = torch.cat([x[:, :, :1]] * (2 * conv.temporal_pad) + [x], dim=2)
    return F.conv3d(F.pad(x, conv.pad), conv.weight, conv.bias, stride=conv.stride)


@pytest.mark.parametrize("kind", ["k3", "down_t", "down_s", "k1"])
def test_causal_conv(kind, monkeypatch):
    """Against the reference's formula on a whole clip; sliced as the reference slices (5, then 4 frames) with the
    carried frames; a single frame through the time-summed 2D kernel; and in bands of rows."""
    torch.manual_seed(1)
    make = {"k3": lambda: V.CausalConv3d(6, 5, 3, temporal_pad=1, pad=(1, 1, 1, 1)),
            "down_t": lambda: V.CausalConv3d(6, 6, 3, stride=2, temporal_pad=1, pad=(0, 1, 0, 1)),
            "down_s": lambda: V.CausalConv3d(6, 6, (1, 3, 3), stride=(1, 2, 2), pad=(0, 1, 0, 1)),
            "k1": lambda: V.CausalConv3d(6, 5, 1)}[kind]
    conv = make()
    nn.init.normal_(conv.weight)
    nn.init.normal_(conv.bias)
    x = torch.randn(1, 6, 13, 10, 12)
    want = _reference_causal_conv(conv, x)
    torch.testing.assert_close(conv(x), want)
    if kind == "down_t":  # its input is already halved in time in the VAE: slices of 3, then 2
        parts = [conv(x[:, :, :3], V.FIRST)] + [conv(x[:, :, a:a + 2], V.NEXT) for a in range(3, 13, 2)]
    else:
        parts = [conv(x[:, :, :5], V.FIRST)] + [conv(x[:, :, a:a + 4], V.NEXT) for a in range(5, 13, 4)]
    torch.testing.assert_close(torch.cat(parts, dim=2), want)
    one = x[:, :, :1]
    torch.testing.assert_close(conv(one), _reference_causal_conv(conv, one), rtol=1e-5, atol=1e-5)
    monkeypatch.setattr(V, "CONV_BAND_BYTES", 2000)  # bands of a row or two
    torch.testing.assert_close(conv(x), want)


@pytest.mark.parametrize("temporal", [True, False])
def test_upsample_matches_reference(temporal):
    """The per-frame shuffle against the reference's rearrange and its dropped first duplicate."""
    torch.manual_seed(2)
    up = V.Upsample3D(8, temporal)
    for conv in (up.upscale_conv, up.conv):  # CausalConv3d's tensors are left empty for the loader
        nn.init.normal_(conv.weight, std=0.1)
        nn.init.normal_(conv.bias, std=0.1)
    x = torch.randn(1, 8, 3, 4, 5)
    y = rearrange(up.upscale_conv(x), "b (x y z c) f h w -> b c (f z) (h x) (w y)", x=2, y=2, z=2 if temporal else 1)
    if temporal:
        y = torch.cat([y[:, :, :1], y[:, :, 2:]], dim=2)
    torch.testing.assert_close(up(x, None), up.conv(y, None))


def test_vae_shapes_and_slicing():
    """5 frames per slice then 4 encode as the whole clip does, and decoding a latent frame at a time as all at
    once; 1 + 4n frames <-> 1 + n latent frames, 8x smaller."""
    vae = _vae()
    x = torch.rand(1, 3, 13, 32, 48) * 2 - 1
    with torch.no_grad():
        sliced = vae._moments(x)
        vae.slice_frames = 100
        whole = vae._moments(x)
        assert sliced.shape == (1, 8, 4, 4, 6)
        torch.testing.assert_close(sliced, whole, rtol=1e-4, atol=1e-4)
        z = vae.encode(x, sample=False)
        assert z.shape == (1, 4, 4, 4, 6)
        torch.testing.assert_close(z, whole[:, :4] * vae.scaling_factor, rtol=1e-4, atol=1e-4)
        parts = list(vae.decode_slices(z))
        assert [p.shape[2] for p in parts] == [5, 4, 4]
        whole_dec = vae.decoder(z / vae.scaling_factor, None)
        torch.testing.assert_close(torch.cat(parts, dim=2), whole_dec, rtol=1e-4, atol=1e-4)
        assert whole_dec.shape == (1, 3, 13, 32, 48)
        assert vae.decode(z[:, :, :1]).shape == (1, 3, 1, 32, 48)  # an image


def test_vae_group_norm_is_per_frame():
    """The reference's causal_norm_wrapper: GroupNorm over (b t) c h w."""
    norm = nn.GroupNorm(4, 8)
    nn.init.normal_(norm.weight)
    x = torch.randn(2, 8, 3, 5, 6) * torch.arange(1, 4).view(1, 1, 3, 1, 1)
    want = rearrange(norm(rearrange(x, "b c t h w -> (b t) c h w")), "(b t) c h w -> b c t h w", t=3)
    torch.testing.assert_close(V.group_norm(norm, x, silu=False), want)
    torch.testing.assert_close(V.group_norm(norm, x.clone(), silu=True, inplace=True), F.silu(want))


def test_tiles():
    for n, tile, overlap, align in [(1088, 608, 128, 16), (240, 76, 16, 1), (1920, 736, 128, 16), (100, 100, 8, 1)]:
        parts = V.spans(n, tile, overlap, align)
        assert parts[0][0] == 0 and parts[-1][1] == n
        for (a0, b0), (a1, b1) in zip(parts, parts[1:]):
            assert b0 - a1 >= overlap - align and a1 > a0 and a1 % align == 0
        weights = torch.zeros(n)
        for (a, b), wgt in zip(parts, V.ramp_weights(parts, n, 1)):
            weights[a:b] += wgt
        torch.testing.assert_close(weights, torch.ones(n))  # a partition of unity
    assert V.spans(48, 32, 128, 8) == [(0, 32), (16, 48)]  # an overlap past half the tile is cut to half
    x = torch.randn(1, 2, 3, 40, 56)
    torch.testing.assert_close(V.tiled(lambda p: p, x, (24, 32), 8, 8, 1), x)  # blends back exactly
    up = V.tiled(lambda p: F.interpolate(p, scale_factor=(1, 2, 2)), x, 16, 8, 8, 2)
    torch.testing.assert_close(up, F.interpolate(x, scale_factor=(1, 2, 2)))
    # a fixed tile much larger than its step: three tiles cover some rows, and the blend still sums to 1 there
    for n, tile in [(912, 512), (1440, 768)]:
        x = torch.randn(1, 1, 1, n, 64)
        torch.testing.assert_close(V.tiled(lambda p: p, x, (tile, 64), 128, 8, 1), x)
    assert V.fit_tiles(1088, 1920, 3e6, 128, 16) is None
    for area in (2e5, 4e5, 7.3e5, 1.5e6):
        th, tw = V.fit_tiles(1088, 1920, area, 128, 16)
        assert th * tw <= area and th % 16 == 0 and tw % 16 == 0


def test_tiled_vae_close_to_whole():
    """Tiles of a tiny VAE's encode and decode stay close to whole frames (they differ only near the seams)."""
    vae = _vae()
    x = torch.rand(1, 3, 5, 64, 96) * 2 - 1
    with torch.no_grad():
        z_whole = vae.encode(x, sample=False)
        z_tiled = vae.encode(x, sample=False, tile=(48, 64), overlap=16)
        assert z_tiled.shape == z_whole.shape
        assert (z_tiled - z_whole).abs().mean() < 0.25 * z_whole.abs().mean()
        y_tiled = vae.decode(z_whole, tile=(6, 8), overlap=2)
        y_whole = vae.decode(z_whole)
        assert y_tiled.shape == y_whole.shape == (1, 3, 5, 64, 96)


# --- the pipeline -------------------------------------------------------------------------------------------------
def test_padding_to_4n_plus_1():
    """The reference's cut_videos: 1 stays 1, 2 to 5 become 5, then up to the next 4n + 1."""
    assert [V.padded_length(t) for t in (1, 2, 4, 5, 6, 9, 10, 13, 81)] == [1, 5, 5, 5, 9, 9, 13, 13, 81]


def test_passes_cover_a_clip_sharing_frames():
    """As few passes as cover the clip, each at most a pass long and 4n + 1 frames (all but the last the same),
    neighbours sharing the overlap (at most half a pass, less one), the last longer than it."""
    for frames, per in [(1, 41), (41, 41), (42, 41), (60, 41), (100, 41), (7, 5), (300, 9), (12, 1), (2, 5)]:
        for overlap in (0, 2, 8):
            parts = P.passes(frames, per, overlap)
            o = min(overlap, (per - 1) // 2) if frames > per else 0
            assert parts[0][0] == 0 and parts[-1][1] == frames
            assert all(b - a == parts[0][1] - parts[0][0] <= per and (b - a - 1) % 4 == 0 for a, b in parts[:-1])
            assert all(b0 - a1 == o for (_, b0), (a1, _) in zip(parts, parts[1:]))
            assert parts[-1][1] - parts[-1][0] > o
            assert len(parts) == (1 if frames <= per else math.ceil((frames - o) / (per - o)))
    assert P.passes(60, 41, 8) == [(0, 37), (29, 60)]  # 60 frames at 1080p
    assert P.passes(100, 41, 0) == [(0, 37), (37, 74), (74, 100)]  # no overlap: side by side, as before
    assert P.passes(8 + 3 * 33, 41, 8) == [(0, 41), (33, 74), (66, 107)]  # m * (41 - 8) + 8: m whole passes


def test_color_fix_keeps_input_colors():
    """The output's details on the input's low frequencies: a color cast in the output is gone, its details stay."""
    torch.manual_seed(3)
    ref = F.avg_pool2d(torch.rand(2, 3, 96, 128), 9, stride=1, padding=4) * 2 - 1
    detail = 0.1 * torch.randn(2, 3, 96, 128)
    cast = torch.tensor([0.3, -0.2, 0.1]).view(1, 3, 1, 1)
    fixed = P.color_fix(ref + detail + cast, ref)
    torch.testing.assert_close(fixed.mean((2, 3)), ref.mean((2, 3)), rtol=0, atol=2e-3)
    kept = detail - P.low_frequencies(detail)  # the details' bands, which the fix leaves alone
    torch.testing.assert_close(fixed, ref + kept, rtol=1e-5, atol=1e-5)
    assert kept.pow(2).mean() > 0.95 * detail.pow(2).mean()


def test_one_step_schedule():
    """configs_3b: lerp schedule (T 1000), uniform trailing timesteps, v_lerp, one Euler step. One step means the
    single timestep T (the resolution shift maps 1 to 1), and the endpoint at t = T is x_T - v."""
    steps, T = 1, 1000.0
    timesteps = torch.arange(1.0, 0.0, -1.0 / steps) * T  # UniformTrailingSamplingTimesteps
    assert timesteps.tolist() == [P.TIMESTEP]
    for shift in (1.0, 3.2, 5.0):  # timestep_transform
        t = timesteps / T
        assert (shift * t / (1 + (shift - 1) * t) * T).tolist() == [P.TIMESTEP]
    x_t, v = torch.randn(4), torch.randn(4)
    a, b = 1 - P.TIMESTEP / T, P.TIMESTEP / T  # lerp: x_t = (1 - t/T) x_0 + t/T x_T
    assert torch.allclose((x_t - b * v) / (a + b), x_t - v)  # v_lerp's x_0
    cond = torch.randn(4)
    assert torch.equal(1.0 * cond + 0.0 * torch.randn(4), cond)  # condition noise scale 0: the clean latents
    from diffusers.models.embeddings import get_timestep_embedding
    t = torch.tensor([P.TIMESTEP])
    torch.testing.assert_close(M.timestep_embedding(t, 256), get_timestep_embedding(
        t, 256, flip_sin_to_cos=False, downscale_freq_shift=0))
    assert V.SeedVR2VAE.scaling_factor == 0.9152


def _tiny_upscaler():
    up = object.__new__(P.SeedVR2Upscaler)
    up.info = Info("seedvr2.safetensors", "seedvr2", "seedvr2", video=True, companion="vae.safetensors")
    up.device, up.log = torch.device("cpu"), print
    up.seed, up.max_tokens, up.vram_limit, up.vae_tile, up.vae_memory = P.SEED, P.MAX_TOKENS, None, None, "gpu"
    up.pass_overlap = P.PASS_OVERLAP
    torch.manual_seed(4)
    cfg = dict(SMALL, in_channels=9, out_channels=4)
    up.dit = M.SeedVR2DiT(M.SeedVR2Config(**cfg)).eval()
    up.dit.compute_dtype = torch.float32
    up.vae = _vae()
    up.positive = torch.randn(7, SMALL["text_dim"])
    return up


def test_upscaler_end_to_end():
    """Any frame count and size: padded to 4n + 1 frames and multiples of 16 inside, cut back; [0, 1]; the same
    seed gives the same frames; options apply to one call."""
    up = _tiny_upscaler()
    frames = torch.rand(3, 3, 20, 30)
    out = up.upscale(frames, (45, 37))
    assert out.shape == (3, 3, 37, 45) and out.min() >= 0 and out.max() <= 1
    torch.testing.assert_close(up.upscale(frames, (45, 37)), out)
    other = up.upscale(frames, (45, 37), options={"seed": 1, "tile": 32, "max_frames": 1, "unknown": 2})
    assert other.shape == out.shape and not torch.equal(other, out)
    assert (up.seed, up.vae_tile, up.max_tokens) == (P.SEED, None, P.MAX_TOKENS)  # restored
    assert up.upscale(frames[:1], (32, 32)).shape == (1, 3, 32, 32)
    seen = []
    up.max_tokens = up.tokens_for((32, 32), 5)
    assert up.frames_per_pass((32, 32)) == 5 and up.overlap_frames((32, 32)) == 2
    assert up.upscale(torch.rand(7, 3, 16, 16), (32, 32), progress=seen.append).shape == (7, 3, 32, 32)
    assert seen == ["Upscaling frames 1-5 of 7", "Upscaling frames 4-7 of 7"]  # sharing two frames
    for t in range(1, 13):  # any frame count, in passes of 5 (padded to 4n + 1 inside) sharing 2
        out = up.upscale(torch.rand(t, 3, 16, 16), (24, 40))
        assert out.shape == (t, 3, 40, 24) and torch.isfinite(out).all() and 0 <= out.min() and out.max() <= 1


def test_overlapping_passes_crossfade():
    """Passes that make the same frames give what one pass gives; passes that differ (here by a constant per pass)
    go from one to the next in a straight line over the frames they share, so there's no step. With a pass of one
    nearest-neighbour resize, each frame a flat grey of its own."""
    up = _tiny_upscaler()
    offset = []  # added to each pass's frames

    def one_pass(frames, size, gen, check):
        y = F.interpolate(frames, size=(size[1], size[0]), mode="nearest")
        offset.append(len(offset))
        return y + 0.1 * offset[-1] if shift else y

    up._clip = one_pass
    up.max_tokens = up.tokens_for((8, 8), 9)  # passes of 9 frames sharing 4
    for t in range(1, 13):
        x = torch.linspace(0, 0.5, t).view(t, 1, 1, 1).expand(t, 3, 4, 4).contiguous()
        shift = False
        offset.clear()
        out = up.upscale(x, (8, 8))
        assert out.shape == (t, 3, 8, 8)
        torch.testing.assert_close(out, F.interpolate(x, size=(8, 8), mode="nearest"))  # as one pass
        assert len(offset) == len(P.passes(t, 9, 4)) == (1 if t <= 9 else 2)
    shift = True
    offset.clear()
    x = torch.linspace(0, 0.5, 12).view(12, 1, 1, 1).expand(12, 3, 4, 4).contiguous()
    out = up.upscale(x, (8, 8), options={"overlap": 4})
    assert P.passes(12, 9, 4) == [(0, 9), (5, 12)]
    step = (out - F.interpolate(x, size=(8, 8), mode="nearest")).mean((1, 2, 3))
    ramp = [0.0] * 5 + [0.1 * (j + 0.5) / 4 for j in range(4)] + [0.1] * 3  # frames 5-8 are both passes'
    torch.testing.assert_close(step, torch.tensor(ramp), rtol=0, atol=1e-6)
    offset.clear()
    out = up.upscale(x, (8, 8), options={"overlap": 0})  # side by side: a step of the whole 0.1 between 8 and 9
    step = (out - F.interpolate(x, size=(8, 8), mode="nearest")).mean((1, 2, 3))
    torch.testing.assert_close(step, torch.tensor([0.0] * 9 + [0.1] * 3), rtol=0, atol=1e-6)


def test_options_are_checked_and_clamped():
    """The Sandbox's names (max_frames, tile, seed, overlap) and the attributes: unknown keys and values that aren't
    numbers dropped, numbers clamped to their range, each with a line in the log and never an error; checked
    options come back the same without a line."""
    up = _tiny_upscaler()
    said = []
    up.log = said.append
    opts = up.check_options({"max_frames": 5000, "tile": 10, "seed": -3, "overlap": 99, "bogus": 1})
    assert opts == {"max_frames": 1001, "tile": P.MIN_TILE, "seed": 0, "overlap": 32}
    assert len(said) == 5 and "bogus" in said[-1] and "1001" in said[0]
    said.clear()
    assert up.check_options(opts) == opts and not said
    assert up.check_options({"max_frames": "20", "tile": "0", "seed": "7", "overlap": None, "vae_memory": "cpu",
                             "max_tokens": 4096}) == {"max_frames": 17, "tile": 0, "seed": 7, "vae_memory": "cpu",
                                                      "max_tokens": 4096}
    assert not said  # 20 frames a pass is 17 (4n + 1, at most what was asked), a tile of 0 whole frames: no line
    assert up.check_options({"max_frames": "lots", "seed": True, "tile": -1, "vae_memory": "disk", "seed2": 1,
                             "overlap": float("nan")}) == {}
    assert len(said) == 6
    said.clear()
    out = up.upscale(torch.rand(3, 3, 16, 16), (32, 32), options={"max_frames": 0, "tile": 1, "seed": 2**70, "x": 1})
    assert out.shape == (3, 3, 32, 32) and len(said) == 4
    assert (up.seed, up.vae_tile, up.max_tokens, up.pass_overlap) == (P.SEED, None, P.MAX_TOKENS, P.PASS_OVERLAP)
    assert up.frames_per_pass((1920, 1080)) == 41 and up.overlap_frames((1920, 1080)) == 8
    assert up.frames_per_pass((32, 32), {"max_frames": 9}) == 9 and up.overlap_frames((32, 32), {"max_frames": 9}) == 4
    assert up.overlap_frames((32, 32), {"max_frames": 9, "overlap": 1}) == 1
    assert up.overlap_frames((32, 32), {"max_frames": 1}) == 0  # a pass of one frame shares none
    assert up.max_tokens == P.MAX_TOKENS and up.pass_overlap == P.PASS_OVERLAP


def test_upscaler_cancels():
    up = _tiny_upscaler()

    class Stop(Exception):
        pass

    def check():
        raise Stop

    with pytest.raises(Stop):
        up.upscale(torch.rand(1, 3, 16, 16), (32, 32), check=check)


# --- the real files -----------------------------------------------------------------------------------------------
def _header_shapes(path):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


@pytest.mark.skipif(not os.path.exists(DIT_FILE), reason="no SeedVR2 3B file")
def test_real_dit_keys_are_parameters():
    shapes = _header_shapes(DIT_FILE)
    assert shapes.pop("positive_conditioning") == (58, 5120) and shapes.pop("negative_conditioning") == (64, 5120)
    meta = {k: torch.empty(s, device="meta") for k, s in shapes.items()}
    config = M.SeedVR2Config.from_state_dict(meta)
    assert config == M.SeedVR2Config()  # configs_3b/main.yaml
    with torch.device("meta"):
        dit = M.SeedVR2DiT(config)
    mine = {k: tuple(v.shape) for k, v in dit.state_dict().items()}
    assert mine == shapes
    keys, _ = comfy_io.read_header(DIT_FILE)
    assert all(k in keys for k in ("blocks.0.attn.proj_qkv.vid.weight", "vid_in.proj.weight"))


@pytest.mark.skipif(not os.path.exists(VAE_FILE), reason="no SeedVR2 VAE file")
def test_real_vae_keys_are_parameters():
    shapes = _header_shapes(VAE_FILE)
    with torch.device("meta"):
        vae = V.SeedVR2VAE()
    assert {k: tuple(v.shape) for k, v in vae.state_dict().items()} == shapes


@pytest.mark.skipif(not os.path.exists(DIT_FILE), reason="no SeedVR2 3B file")
def test_real_rope_freqs_are_the_formula():
    from safetensors import safe_open
    with safe_open(DIT_FILE, "pt") as f:
        stored = f.get_tensor("blocks.0.attn.rope.rope.freqs")
    assert torch.equal(M.rope_freqs(42, 10000.0).half(), stored)

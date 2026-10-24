"""FLUX 3 Image Klein: the DiT and the driver's token layout against Black Forest Labs' reference code, the text
encoding, the schedules, references for edits, and the profile's files.

The reference classes below are copied from Black Forest Labs' reference code
(https://github.com/black-forest-labs/flux-3-image-klein, ``src/flux3/dit.py``, ``pipeline.py``, ``sampling.py``,
``text_encoder.py``; Copyright 2026 Black Forest Labs, Inc., Apache License 2.0), with the model's size constants
made small so the tests run on a CPU. They're used only here, as the yardstick.

    .venv/bin/python -m pytest tests/test_flux3_klein.py -v
"""

import math
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from conftest import header_only_safetensors
from dainamo.engine import comfy_io
from dainamo.engine.families.base import transformer_shapes
from dainamo.engine.families.flux3_klein import driver as D
from dainamo.engine.families.flux3_klein.driver import Flux3KleinDriver
from dainamo.engine.families.flux3_klein.model import Flux3Config, Flux3KleinDiT, timestep_embedding
from dainamo.engine.lora import LoRANetwork

CPU = torch.device("cpu")
MODELS = "/home/william/AI_APPS/COMFY_CACHE/models"
REAL = {
    "diffusion_model": f"{MODELS}/diffusion_models/flux3-image-klein-base-int8-convrot.safetensors",
    "text_encoder": f"{MODELS}/clip/qwen3vl_4b_fp8_scaled.safetensors",
    "vae": f"{MODELS}/vae/flux2-vae.safetensors",
}

# --- the reference, small (flux3/src/flux3/dit.py, Apache 2.0) --------------------------------------------------
MODES = ("image", "image_cond", "text")
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 96}
HIDDEN = 64
HEADS = 4
DEPTH = 2
DEPTH_SINGLE = 3
AXES_DIM = (4, 4, 4, 4)
MLP_RATIO = 3.0
THETA = 10000
SMALL = Flux3Config(hidden=HIDDEN, heads=HEADS, depth=DEPTH, depth_single=DEPTH_SINGLE, text_channels=96,
                    axes_dim=AXES_DIM)


@dataclass
class Packed:
    data: dict
    ids: dict
    timesteps: dict
    mode_indices: dict

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])


def ref_timestep_embedding(t: Tensor, dim: int = 256, max_period: int = 10000) -> Tensor:
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)


def ref_rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = torch.einsum("...n,d->...nd", pos, 1.0 / (theta**scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def ref_rope_embed(ids: Tensor) -> Tensor:
    with torch.autocast(device_type=ids.device.type, enabled=False):
        emb = torch.cat([ref_rope(ids[..., i], AXES_DIM[i], THETA) for i in range(len(AXES_DIM))], dim=-3)
    return emb.unsqueeze(1)


def ref_rmsnorm_rope(x: Tensor, scale: Tensor, freqs: Tensor, eps: float = 1e-6) -> Tensor:
    b, h, l, d = x.shape
    x_f = x.float()
    x_norm = x_f * x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * scale.float()
    x_r, freqs = x_norm.reshape(b, h, l, d // 2, 1, 2), freqs.float()
    rotated = freqs[..., 0] * x_r[..., 0] + freqs[..., 1] * x_r[..., 1]
    return rotated.reshape(b, h, l, d).to(x.dtype)


def ref_layernorm_modulation(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return (1.0 + scale) * F.layer_norm(x, (x.shape[-1],), eps=1e-6) + shift


def ref_packed_attention(q: Tensor, k: Tensor, v: Tensor, bounds) -> Tensor:
    _, heads, length, dim = q.shape
    out = torch.zeros(length, heads * dim, device=q.device, dtype=q.dtype)
    for start, end in bounds:
        if start == end:
            continue
        chunk = F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end])
        out[start:end] = chunk[0].transpose(0, 1).reshape(end - start, heads * dim)
    return out


class RefRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class RefQKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RefRMSNorm(dim)
        self.key_norm = RefRMSNorm(dim)

    def forward(self, q, k, freqs):
        return ref_rmsnorm_rope(q, self.query_norm.scale, freqs), ref_rmsnorm_rope(k, self.key_norm.scale, freqs)


class RefMLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class RefModeModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec):
        return tuple(self.lin(F.silu(vec)).chunk(3, dim=-1))


class RefLastLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(ref_layernorm_modulation(x, scale, shift))


class RefBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = RefQKNorm(hidden // heads)

    def forward(self, x, freqs, bounds, mod):
        shift, scale, gate = mod
        x_mod = ref_layernorm_modulation(x, scale, shift)
        split = lambda t: t.reshape(1, t.shape[0], self.heads, -1).transpose(1, 2)  # noqa: E731
        q, k = self.norm(split(self.q_proj(x_mod)), split(self.k_proj(x_mod)), freqs)
        v = split(self.v_proj(x_mod))
        attn = ref_packed_attention(q, k, v.to(q.dtype), bounds)
        gate_in, up = self.mlp_in(x_mod).chunk(2, dim=-1)
        return x + gate * (self.attn_out(attn) + self.mlp_out(F.silu(gate_in) * up))


class RefDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_in = nn.ModuleDict({m: nn.Linear(IN_CHANNELS[m], HIDDEN, bias=False) for m in ("image", "image_cond")})
        self.txt_in = nn.Linear(IN_CHANNELS["text"], HIDDEN, bias=False)
        self.time_in = RefMLPEmbedder(256, HIDDEN)
        self.early_stream_modulations = nn.ModuleDict({k: RefModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")})
        self.single_stream_modulations = nn.ModuleDict({k: RefModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")})
        self.content_mode_blocks = nn.ModuleDict({
            m: nn.ModuleList([RefBlock(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)]) for m in ("image", "image_cond")
        })
        self.txt_mode_blocks = nn.ModuleList([RefBlock(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)])
        self.single_blocks = nn.ModuleList([RefBlock(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH_SINGLE)])
        self.final_layer = nn.ModuleDict({"image": RefLastLayer(HIDDEN, IN_CHANNELS["image"])})

    def forward(self, packed: Packed) -> Tensor:
        data, ids, timesteps = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
        for m in MODES:
            if data[m].shape[0] == 0:
                data[m], ids[m], timesteps[m] = data[m].new_zeros(1, data[m].shape[1]), ids[m].new_zeros(1, 4), timesteps[m].new_zeros(1)
        mod_key = lambda m: "txt" if m == "text" else m  # noqa: E731
        vec = {m: self.time_in(ref_timestep_embedding(timesteps[m])) for m in MODES}
        early = {m: self.early_stream_modulations[mod_key(m)](vec[m]) for m in MODES}
        single = {m: self.single_stream_modulations[mod_key(m)](vec[m]) for m in MODES}
        x = {m: (self.txt_in if m == "text" else self.emb_in[m])(data[m]) for m in MODES}
        freqs = {m: ref_rope_embed(ids[m][None]) for m in MODES}
        for d in range(DEPTH):
            for m in ("image", "image_cond"):
                x[m] = self.content_mode_blocks[m][d](x[m], freqs[m], packed.mode_indices[m], early[m])
            x["text"] = self.txt_mode_blocks[d](x["text"], freqs["text"], packed.mode_indices["text"], early["text"])

        chunks, id_chunks, mods, bounds, placement, offset = [], [], ([], [], []), [], [], 0
        for s in range(packed.num_samples):
            start = offset
            for m in MODES:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    id_chunks.append(ids[m][a:b])
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    placement.append((m, a, b, offset))
                    offset += b - a
            bounds.append((start, offset))
        h = torch.cat(chunks)
        freqs_sm = ref_rope_embed(torch.cat(id_chunks)[None])
        mods = tuple(torch.cat(c) for c in mods)
        for block in self.single_blocks:
            h = block(h, freqs_sm, bounds, mods)
        image = torch.empty_like(x["image"])
        for m, a, b, start in placement:
            if m == "image":
                image[a:b] = h[start : start + b - a]
        return self.final_layer["image"](image, vec["image"])


def ref_pack(samples) -> Packed:
    """flux3/pipeline.py ``pack``, with fp32 timesteps (the reference's are bf16, which the CPU can't mix with
    fp32 weights; ``test_timesteps_are_rounded_to_bf16_as_in_the_reference`` covers the rounding)."""
    data = {m: [] for m in MODES}
    ids = {m: [] for m in MODES}
    mode_indices = {m: [] for m in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))
    cat = lambda parts, width: torch.cat(parts) if parts else torch.zeros((0, width))  # noqa: E731
    data = {m: cat(data[m], IN_CHANNELS[m]) for m in MODES}
    ids = {m: cat(ids[m], 4) for m in MODES}
    return Packed(data=data, ids=ids, timesteps={m: torch.zeros(data[m].shape[0]) for m in MODES},
                  mode_indices=mode_indices)


def ref_latent_ids(h: int, w: int) -> Tensor:
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4)


def ref_scaled_ids(ids: Tensor) -> Tensor:
    ids = ids.float()
    ids[:, 1:3] *= 0.5
    return ids


def ref_timesteps(steps: int, alpha: float) -> Tensor:
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


# --- helpers ----------------------------------------------------------------------------------------------------
def reference_model() -> RefDiT:
    torch.manual_seed(0)
    ref = RefDiT().eval()
    with torch.no_grad():  # nothing at its init value: norms and modulations random too
        for n, p in ref.named_parameters():
            if n.endswith(".scale"):
                p.copy_(1 + 0.3 * torch.randn_like(p))
    return ref


def ours_from(ref: RefDiT) -> Flux3KleinDiT:
    """The same weights with q/k/v fused into qkv = [q; k; v], as ComfyUI's files store them."""
    sd = dict(ref.state_dict())
    for name in [k[: -len(".q_proj.weight")] for k in sd if k.endswith(".q_proj.weight")]:
        sd[f"{name}.qkv.weight"] = torch.cat([sd.pop(f"{name}.{p}_proj.weight") for p in "qkv"])
    ours = Flux3KleinDiT(SMALL).eval()
    ours.load_state_dict(sd, strict=True)
    ours.compute_dtype = ours.timestep_dtype = torch.float32
    return ours


def rel(a, b) -> float:
    return ((a - b).abs().max() / b.abs().max()).item()


def tiny_driver() -> Flux3KleinDriver:
    d = Flux3KleinDriver({}, CPU)
    d.dtype = torch.float32
    return d


def text(n: int) -> Tensor:
    return torch.randn(n, IN_CHANNELS["text"])


# --- the DiT ----------------------------------------------------------------------------------------------------
def test_dit_matches_the_reference_text_to_image_and_edits():
    """A CFG-style pack (two samples, different texts), and an edit pack (two references of different sizes in one
    sample, one in another, none in a third), against the reference model on the same weights."""
    ref = reference_model()
    ours = ours_from(ref)
    g = torch.Generator().manual_seed(1)
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731

    def image(h, w, index=None):
        ids = ref_scaled_ids(ref_latent_ids(h, w))
        if index is not None:
            ids[:, 3] = -(1 + index)
        return rnd(h * w, 128), ids

    def txt(n):
        return rnd(n, 96), torch.stack([torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.arange(n).float()], -1)

    def both(samples, t_image):
        packed = ref_pack(samples)
        packed.timesteps["image"] = torch.cat([torch.full((s[2][0].shape[0],), t) for s, t in zip(samples, t_image)])
        return packed

    def our_pack(packed: Packed, t_image):
        from dainamo.engine.families.flux3_klein.model import Pack
        s = packed.num_samples
        return Pack(packed.data, packed.ids, packed.mode_indices,
                    {"image": torch.tensor(t_image), "image_cond": torch.zeros(s), "text": torch.zeros(s)})

    img = image(4, 6)
    cases = {
        "cfg": ([[txt(5), None, img], [txt(11), None, img]], [0.9, 0.9]),
        "edit": ([[txt(7), _cat([image(4, 4, 0), image(2, 6, 1)]), image(4, 6)],
                  [txt(3), image(6, 2, 0), image(4, 6)],
                  [txt(9), None, image(4, 6)]], [0.7, 0.2, 0.95]),
        "one": ([[txt(4), image(3, 3, 0), image(2, 2)]], [0.5]),
    }
    with torch.no_grad():
        for name, (samples, ts) in cases.items():
            packed = both(samples, ts)
            expected = ref(packed)
            got = ours(our_pack(packed, ts))
            assert got.shape == expected.shape and rel(got, expected) < 1e-5, name
        # negative control: the fused q/k/v order matters
        w = ours.single_blocks[0].qkv.weight
        w.data = torch.cat([w[HIDDEN : 2 * HIDDEN], w[:HIDDEN], w[2 * HIDDEN :]])
        packed = both(*cases["cfg"])
        assert rel(ours(our_pack(packed, cases["cfg"][1])), ref(packed)) > 1e-3


def _cat(parts):
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def test_timesteps_are_rounded_to_bf16_as_in_the_reference():
    t = torch.tensor([0.9643, 0.31415, 0.0, 1.0])
    expected = ref_timestep_embedding(t.to(torch.bfloat16))  # the reference's pack holds them in bf16
    assert expected.dtype == torch.bfloat16
    assert torch.equal(timestep_embedding(t), expected)
    assert torch.equal(timestep_embedding(t, dtype=torch.float32), ref_timestep_embedding(t))


def test_driver_lays_out_tokens_as_the_reference_pipeline():
    """The driver's pack from (B, 128, h, w) latents, cached texts and reference latents: image tokens row by row
    at scaled ids, text at (0, 0, 0, position), references at l = -(1 + index), clean; the same output as the
    reference fed the pipeline's own pack."""
    ref = reference_model()
    ours = ours_from(ref)
    d = tiny_driver()
    torch.manual_seed(3)
    noisy = torch.randn(2, 128, 3, 5)
    texts = [text(6), text(9)]
    refs = [[torch.randn(128, 2, 4), torch.randn(128, 3, 3)], []]
    t = torch.tensor([0.8, 0.4])
    with torch.no_grad():
        got = d._forward(ours, noisy, t, texts, refs)

    def tokens(lat):  # the reference autoencoder's encode: (h * w, 128) row by row
        return lat.permute(1, 2, 0).reshape(-1, 128)

    samples = []
    for i in range(2):
        image = (tokens(noisy[i]), ref_scaled_ids(ref_latent_ids(3, 5)))
        n = texts[i].shape[0]
        tx = (texts[i], torch.stack([torch.zeros(n)] * 3 + [torch.arange(n).float()], -1))
        cond = None
        if refs[i]:
            parts = []
            for j, r in enumerate(refs[i]):
                ids = ref_scaled_ids(ref_latent_ids(*r.shape[1:]))
                ids[:, 3] = -(1 + j)
                parts.append((tokens(r), ids))
            cond = _cat(parts)
        samples.append([tx, cond, image])
    packed = ref_pack(samples)
    packed.timesteps["image"] = torch.cat([torch.full((15,), 0.8), torch.full((15,), 0.4)])
    with torch.no_grad():
        expected = ref(packed).reshape(2, 3, 5, 128).permute(0, 3, 1, 2)
    assert rel(got, expected) < 1e-5


def test_lora_reaches_every_target_and_saves_comfyui_names():
    torch.manual_seed(0)
    model = Flux3KleinDiT(SMALL)
    model.compute_dtype = torch.float32
    net = LoRANetwork(model, Flux3KleinDriver.lora_targets, rank=4, alpha=4)
    blocks = 2 * DEPTH + DEPTH + DEPTH_SINGLE
    assert len(net.modules) == 4 * blocks
    for m in net.modules.values():
        nn.init.normal_(m.lora_B.weight, std=0.01)
    d = tiny_driver()
    payloads = [{"latent": torch.randn(128, 2, 3), "refs": [torch.randn(128, 2, 2)]}]
    d.training_loss(model, payloads, [{"text": text(5)}], {}, sigma=0.6).backward()
    missing = [n for n, m in net.modules.items()
               if m.lora_A.weight.grad is None or not m.lora_A.weight.grad.abs().sum()]
    assert not missing
    keys = net.state_dict()
    assert "diffusion_model.single_blocks.0.qkv.lora_A.weight" in keys
    assert "diffusion_model.content_mode_blocks.image_cond.1.mlp_out.lora_B.weight" in keys
    assert "diffusion_model.txt_mode_blocks.0.attn_out.lora_A.weight" in keys


def test_gradient_checkpointing_gives_the_same_gradients():
    torch.manual_seed(0)
    model = Flux3KleinDiT(SMALL)
    model.compute_dtype = torch.float32
    d = tiny_driver()
    payloads = [{"latent": torch.randn(128, 2, 2), "refs": [torch.randn(128, 2, 2)]}]
    conds = [{"text": text(4)}]
    grads = []
    for on in (False, True):
        model.enable_gradient_checkpointing(on)
        model.zero_grad()
        d.training_loss(model, payloads, conds, {}, sigma=0.5, generator=torch.Generator().manual_seed(0)).backward()
        grads.append(model.single_blocks[1].qkv.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


# --- the driver -------------------------------------------------------------------------------------------------
class Recorder(nn.Module):
    """Records the pack the driver builds and predicts zeros."""

    def __init__(self):
        super().__init__()
        self.packs = []

    def forward(self, pack):
        self.packs.append(pack)
        return torch.zeros_like(pack.data["image"])


def test_batched_edit_and_text_to_image_items_share_one_forward():
    d = tiny_driver()
    m = Recorder()
    edit = {"latent": torch.randn(128, 4, 6), "refs": [torch.randn(128, 4, 6), torch.randn(128, 2, 2)]}
    plain = {"latent": torch.randn(128, 4, 6)}
    d.training_loss(m, [edit, plain, edit], [{"text": text(80)}, {"text": text(160)}, {"text": text(80)}], {},
                    sigma=0.5)
    assert len(m.packs) == 1
    p = m.packs[0]
    assert p.spans["image"] == [(0, 24), (24, 48), (48, 72)]
    assert p.spans["image_cond"] == [(0, 28), (28, 28), (28, 56)]
    assert p.spans["text"] == [(0, 80), (80, 240), (240, 320)]
    assert p.ids["image_cond"][:24, 3].eq(-1).all() and p.ids["image_cond"][24:28, 3].eq(-2).all()
    assert torch.equal(p.ids["image"][:6, 2], torch.arange(6) * 0.5) and p.ids["image"][6, 1] == 0.5
    assert torch.equal(p.ids["text"][80:240, 3], torch.arange(160).float())
    assert p.timesteps["image"].tolist() == [0.5] * 3 and not p.timesteps["image_cond"].any()
    assert not p.timesteps["text"].any()
    # a dropped caption keeps its references (the reference's unconditional edit branch)
    d.training_loss(m, [edit], [{"text": text(80), "dropped": True}], {}, sigma=0.5)
    assert m.packs[-1].data["image_cond"].shape[0] == 28


class Oracle(nn.Module):
    """Predicts the exact velocity (noise - clean) toward fixed clean latents, reading sigma = t."""

    def __init__(self, x0):
        super().__init__()
        self.x0 = x0

    def forward(self, pack):
        x = pack.data["image"].float()
        return (x - self.x0) / pack.timesteps["image"].float().clamp_min(1e-6)


def test_sampler_lands_on_the_clean_latent_given_the_true_velocity():
    d = tiny_driver()
    x0 = torch.randn(4 * 6, 128)
    for scheduler in ("default", "simple", "beta"):
        out = d.sample(Oracle(x0), text(80), text(80), 96, 64, 12, 4.0, 3, scheduler=scheduler)
        assert torch.allclose(D.to_tokens(out), x0, atol=1e-4), scheduler


@pytest.mark.parametrize("steps", [1, 28, 50])
def test_sampling_schedule_is_the_references(steps):
    d = tiny_driver()
    assert torch.allclose(D.sampling_sigmas(steps), ref_timesteps(steps, 16.0))
    sig = d.default_sigmas(steps, 1024, 1024)
    assert len(sig) == steps + 1 and sig[0] == 1.0 and sig[-1] == 0.0
    from dainamo.engine.sampling import make_sigmas
    # the simple scheduler at the model's shift is the same schedule
    assert torch.allclose(torch.tensor(make_sigmas(d, "simple", steps, 1024, 1024)), torch.tensor(sig), atol=1e-6)


def test_training_noise_levels():
    d = tiny_driver()
    torch.manual_seed(0)
    for mode in ("shift", "logit_normal", "uniform"):
        t = d.sample_timesteps(20000, (64, 64), {"timestep_sampling": mode})
        assert t.min() > 0 and t.max() <= 1, mode
    t = d.sample_timesteps(200000, (64, 64), {"timestep_sampling": "shift"})
    # uniform u through the shift: the median lands at shift(0.5) = 8 / 8.5
    assert abs(t.median().item() - 8 / 8.5) < 3e-3
    assert abs((t < D.shift(torch.tensor(0.1))).float().mean().item() - 0.1) < 5e-3


def test_masked_loss_and_weighting():
    d = tiny_driver()
    lat = torch.randn(128, 2, 4)
    mask = torch.zeros(2, 4, dtype=torch.float16)
    mask[:, :2] = 1
    payload = {"latent": lat, "mask": mask}

    def loss(settings, **kw):
        return d.training_loss(Recorder(), [payload], [{"text": text(80)}], settings,
                               generator=torch.Generator().manual_seed(0), **kw)

    target = torch.randn((1, 128, 2, 4), generator=torch.Generator().manual_seed(0)) - lat[None]
    assert torch.equal(loss({}, sigma=0.5), F.mse_loss(torch.zeros_like(target), target))
    masked = {"masked_loss": True, "mask_min_weight": 0.0}
    assert torch.allclose(loss(masked, sigma=0.5), (target[..., :2] ** 2).mean())


def test_slider_batch_and_velocity_run_through_predict():
    torch.manual_seed(0)
    model = Flux3KleinDiT(SMALL)
    model.compute_dtype = torch.float32
    d = tiny_driver()
    batch = d.slider_batch([{"latent": torch.randn(128, 2, 3)}, {"latent": torch.randn(128, 2, 3)}], {})
    v = d.slider_velocity(model, batch, text(80))
    assert v.shape == (2, 128, 2, 3) and torch.isfinite(v).all()


def test_sequence_tokens_count_image_and_text():
    text_tokens = Flux3KleinDriver.memory.text_tokens if Flux3KleinDriver.memory else 256
    assert Flux3KleinDriver.sequence_tokens(1024, 768) == 64 * 48 + text_tokens
    assert Flux3KleinDriver.reference_tokens(512, 512) == 32 * 32


def test_reference_sizing():
    for (w, h), target, count in (((1920, 1080), 1024 * 1024, 1), ((500, 800), 512 * 512, 2),
                                  ((3000, 3000), 2048 * 2048, 3), ((64, 40), 256 * 256, 1)):
        rw, rh = D.reference_size(w, h, target, count)
        assert rw % 16 == 0 and rh % 16 == 0
        assert abs(rw / rh - w / h) / (w / h) < 0.12
        cap = D.SINGLE_REFERENCE_MAX_PIXELS if count == 1 else D.MULTI_REFERENCE_MAX_PIXELS
        assert abs(rw * rh - min(target, cap)) / min(target, cap) < 0.08
    img = Image.new("RGB", (300, 200), (255, 0, 0))
    x = D.prepare_reference(img, 256 * 256, flip=True)
    assert x.shape == (1, 3, 208, 320) and x.min() >= -1 and x.max() <= 1


def test_encode_prompt_matches_the_reference_text_encoder(tmp_path, monkeypatch):
    """A tiny random Qwen3-VL text model in ComfyUI's layout: right-padded to 80, positions the mask's running
    count, layers 4..32 side by side, exactly as the reference's TextEncoder.encode reads them."""
    from safetensors.torch import save_file
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    from dainamo.engine.hf_assets import qwen3vl_tokenizer_and_config

    tokenizer, real = qwen3vl_tokenizer_and_config()
    cfg = real.text_config.to_dict()
    cfg.update(hidden_size=16, intermediate_size=32, num_hidden_layers=33, num_attention_heads=2,
               num_key_value_heads=1, head_dim=8,
               rope_parameters={**cfg["rope_parameters"], "mrope_section": [2, 1, 1]})
    cfg = Qwen3VLTextConfig(**cfg)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3VLTextModel(cfg).eval()
    path = str(tmp_path / "te.safetensors")
    save_file({"model." + k: v.contiguous() for k, v in model.state_dict().items()}, path)

    class Config:
        text_config = cfg

    monkeypatch.setattr(D, "qwen3vl_tokenizer_and_config", lambda: (tokenizer, Config))
    d = Flux3KleinDriver({"text_encoder": path}, CPU, log=lambda *_: None)
    d.dtype = torch.float32
    d.load_text_encoder()
    for prompt in ("", "A red apple on a wooden table, soft window light, shallow depth of field, photograph. " * 9):
        got = d.encode_prompt(prompt)
        # the reference (flux3/text_encoder.py), on the same model
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        n = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].shape[1]
        length = min(math.ceil(n / 80) * 80, 4096)
        inputs = tokenizer(chat, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"]
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"], attention_mask=mask,
                        position_ids=positions[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False)
        assert len(out.hidden_states) == 34  # the embeddings, then each layer's output
        stacked = torch.stack([out.hidden_states[k] for k in D.OUTPUT_LAYERS], dim=1)
        expected = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
        assert got.shape == (length, 8 * 16) and length % 80 == 0 and length >= n
        assert torch.equal(got, expected)
    assert d.encode_prompt("").shape[0] == 80


# --- files --------------------------------------------------------------------------------------------------------
def test_distilled_and_unfused_files_are_refused(tmp_path):
    d = Flux3KleinDriver({"diffusion_model": "x.safetensors"}, CPU)
    shapes = {"single_blocks.0.qkv.weight": [9216, 3072], "txt_mode_blocks.0.qkv.weight": [9216, 3072]}
    with pytest.raises(ValueError, match="distilled"):
        d.build_transformer({**shapes, "ref_time_in.in_layer.weight": [3072, 256]})
    with pytest.raises(ValueError, match="separately"):
        d.build_transformer({"single_blocks.0.q_proj.weight": [3072, 3072], "txt_mode_blocks.0.q_proj.weight": [1, 1]})
    sig = {k: ("BF16", (8, 8)) for k in Flux3KleinDriver.signature_keys["diffusion_model"]}
    base, distilled = tmp_path / "base.safetensors", tmp_path / "distilled.safetensors"
    header_only_safetensors(base, sig)
    header_only_safetensors(distilled, {**sig, "ref_time_in.in_layer.weight": ("BF16", (8, 8))})
    check = lambda p: next(c for c in Flux3KleinDriver.validate_profile({"diffusion_model": str(p)})  # noqa: E731
                           if c["field"] == "diffusion_model")
    assert check(base)["level"] == "ok"
    assert check(distilled)["level"] == "error" and "distilled" in check(distilled)["message"]


def test_comfy_attention_hints_are_dropped():
    loaded = comfy_io.LoadedWeights({"single_blocks.0.comfy_attention.config": torch.zeros(3, dtype=torch.uint8),
                                     "single_blocks.0.qkv.weight": torch.zeros(1)})
    Flux3KleinDriver({}, CPU).check_transformer_state("x", loaded)
    assert list(loaded.tensors) == ["single_blocks.0.qkv.weight"]


def test_downloads_are_pinned():
    seen = set()
    for dl in Flux3KleinDriver.downloads:
        assert dl.size > 0 and len(dl.sha256) == 64 and len(dl.revision) == 40
        seen.add(dl.field)
    assert seen == {"diffusion_model", "text_encoder", "vae"}


@pytest.mark.parametrize("field", list(REAL))
def test_signatures_match_the_real_files(field):
    from dainamo.engine.memory import tensors

    path = REAL[field]
    if not os.path.exists(path):
        pytest.skip(f"{path} isn't here")
    keys, _ = comfy_io.read_header(path)
    for suffix in Flux3KleinDriver.signature_keys[field]:
        assert any(k.endswith(suffix) for k in keys), suffix
    assert Flux3KleinDriver.matches(field, tensors(path))
    if field == "diffusion_model":  # the model as the driver builds it has exactly the file's weights
        _, shapes = transformer_shapes(path)
        model = Flux3KleinDriver({"diffusion_model": path}, CPU).build_transformer(shapes)
        weights = {k for k in shapes if not k.endswith(comfy_io.SIDECAR_SUFFIXES + comfy_io.METADATA_SUFFIXES)}
        assert weights == set(model.state_dict())


def test_other_vaes_are_not_taken_for_the_flux2_vae():
    from dainamo.engine.memory import tensors

    for name in ("qwen_image_vae.safetensors", "ae.safetensors"):
        path = f"{MODELS}/vae/{name}"
        if os.path.exists(path):
            assert not Flux3KleinDriver.matches("vae", tensors(path)), name

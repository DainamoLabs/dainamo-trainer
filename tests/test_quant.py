"""Tests for loading ComfyUI-quantized weights (fp8_scaled, int8, int8 convrot).

Synthetic tests build small files in the same layout ComfyUI uses. The
``real`` tests compare the user's int8_convrot and fp8_scaled Krea 2 Turbo
checkpoints against each other; they're skipped when those files are missing.

    .venv/bin/python -m pytest tests -v
"""

import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from dainamo.engine import comfy_io
from dainamo.engine.comfy_io import dequantize, load_state_dict, regular_hadamard, rotate
from dainamo.engine.fp8_linear import QuantLinear, materialize
from dainamo.engine.lora import LoRANetwork

MODELS = os.environ.get("DAINAMO_TEST_MODELS", os.path.expanduser("~/AI_APPS/COMFY_CACHE/models/diffusion_models"))
INT8_FILE = os.path.join(MODELS, "krea2_turbo_int8_convrot.safetensors")
FP8_FILE = os.path.join(MODELS, "krea2_turbo_fp8_scaled.safetensors")


def marker(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def quantize_int8_rows(w: torch.Tensor, rot: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference quantizer: rotate along the input dim, symmetric per-row int8."""
    wr = rotate(w.float(), rot)
    scale = (wr.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
    return torch.round(wr / scale).clamp(-127, 127).to(torch.int8), scale


def quantize_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = w.abs().max().float() / 448
    return (w.float() / scale).to(torch.float8_e4m3fn), scale


class Tiny(nn.Module):
    def __init__(self, d_in=512, d_hidden=256, vocab=100):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_in)
        self.a = nn.Linear(d_in, d_hidden, bias=True)  # int8 convrot
        self.b = nn.Linear(d_hidden, d_in, bias=False)  # int8 convrot
        self.c = nn.Linear(d_in, 64, bias=False)  # fp8 scaled
        self.d = nn.Linear(64, 32, bias=True)  # plain bf16

    def forward(self, x):
        return self.d(self.c(self.b(F.gelu(self.a(x)))))


@pytest.fixture
def tiny_file(tmp_path):
    torch.manual_seed(0)
    ref = Tiny()
    sd = {}
    for name in ("a", "b"):
        lin = getattr(ref, name)
        q, s = quantize_int8_rows(lin.weight.data, 256)
        sd[f"{name}.weight"], sd[f"{name}.weight_scale"] = q, s
        sd[f"{name}.comfy_quant"] = marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
        if lin.bias is not None:
            sd[f"{name}.bias"] = lin.bias.data.to(torch.bfloat16)
    q, s = quantize_int8_rows(ref.emb.weight.data, 1)  # embeddings: int8 without rotation
    sd["emb.weight"], sd["emb.weight_scale"] = q, s
    sd["emb.comfy_quant"] = marker({"format": "int8_tensorwise"})
    q, s = quantize_fp8(ref.c.weight.data)
    sd["c.weight"], sd["c.weight_scale"] = q, s
    sd["c.comfy_quant"] = marker({"format": "float8_e4m3fn"})
    sd["d.weight"], sd["d.bias"] = ref.d.weight.data.to(torch.bfloat16), ref.d.bias.data.to(torch.bfloat16)
    path = tmp_path / "tiny.safetensors"
    save_file(sd, str(path))
    return str(path), ref


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---------------------------------------------------------------- rotation ---
@pytest.mark.parametrize("n", [4, 16, 64, 256])
def test_hadamard_is_symmetric_orthonormal_self_inverse(n):
    h = regular_hadamard(n, dtype=torch.float64)
    assert torch.equal(h, h.T)
    assert torch.allclose(h @ h, torch.eye(n, dtype=torch.float64), atol=1e-12)


def test_rotate_twice_is_identity():
    x = torch.randn(3, 7, 1024, dtype=torch.float64)
    assert torch.allclose(rotate(rotate(x, 256), 256), x, atol=1e-12)


def test_rotate_rejects_bad_group_size():
    with pytest.raises(ValueError):
        regular_hadamard(128)  # not a power of 4
    with pytest.raises(ValueError):
        rotate(torch.randn(2, 300), 256)


# ----------------------------------------------------------------- loading ---
def test_formats_detected(tiny_file):
    path, _ = tiny_file
    loaded = load_state_dict(path)
    assert loaded.quant["a"].format == "int8_tensorwise" and loaded.quant["a"].rot_size == 256
    assert loaded.quant["emb"].rot_size == 1
    assert loaded.quant["c"].format == "float8_e4m3fn"
    assert "d" not in loaded.quant
    labels, unsupported = comfy_io.describe_quant(path)
    assert labels == {"int8 convrot", "int8", "fp8 scaled"} and not unsupported


def test_dequantized_weights_match_original(tiny_file):
    path, ref = tiny_file
    loaded = load_state_dict(path)
    for name in ("a", "b"):
        w = dequantize(loaded.tensors[f"{name}.weight"], loaded.quant[name], torch.float32)
        assert rel_err(w, getattr(ref, name).weight.data) < 0.02, name
        # without un-rotating, the weight is unrelated to the original
        q = loaded.quant[name]
        wrong = loaded.tensors[f"{name}.weight"].float() * q.weight_scale
        assert rel_err(wrong, getattr(ref, name).weight.data) > 0.5
    w = dequantize(loaded.tensors["c.weight"], loaded.quant["c"], torch.float32)
    assert rel_err(w, ref.c.weight.data) < 0.05


def test_materialize_and_forward_backward_parity(tiny_file):
    path, ref = tiny_file
    loaded = load_state_dict(path)
    with torch.device("meta"):
        model = Tiny()
    materialize(model, loaded, torch.float32, keep_fp8=True)
    assert isinstance(model.a, QuantLinear) and model.a.rot_size == 256 and model.a.weight.dtype == torch.int8
    assert isinstance(model.c, QuantLinear) and model.c.rot_size == 1
    assert isinstance(model.emb, nn.Embedding) and model.emb.weight.dtype == torch.float32
    assert isinstance(model.d, nn.Linear)

    # reference: the same quantized weights expanded to a plain float model
    plain = load_state_dict(path)
    with torch.device("meta"):
        expanded = Tiny()
    materialize(expanded, plain, torch.float32, keep_fp8=False)

    x1 = torch.randn(2, 9, 512, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = model(x1), expanded(x2)
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)
    y1.square().sum().backward()
    y2.square().sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4, rtol=1e-4)

    # and both are close to the original unquantized model
    y_ref = ref(x1.detach())
    assert rel_err(y1.detach(), y_ref) < 0.05


def test_quantized_weights_get_no_gradients_but_lora_does(tiny_file):
    path, _ = tiny_file
    with torch.device("meta"):
        model = Tiny()
    materialize(model, load_state_dict(path), torch.float32, keep_fp8=True)
    net = LoRANetwork(model, ["a", "b", "c"], rank=4, alpha=4)
    for m in net.modules.values():
        nn.init.normal_(m.lora_B.weight, std=0.01)
    model(torch.randn(1, 5, 512)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    assert all(not b.requires_grad for b in model.buffers())


def test_dequantized_weight_helper_matches(tiny_file):
    path, _ = tiny_file
    loaded = load_state_dict(path)
    with torch.device("meta"):
        model = Tiny()
    materialize(model, load_state_dict(path), torch.float32, keep_fp8=True)
    expected = dequantize(loaded.tensors["a.weight"], loaded.quant["a"], torch.float32)
    assert torch.allclose(model.a.dequantized_weight(), expected, atol=1e-5)


def test_unsupported_format_fails_clearly(tmp_path):
    path = str(tmp_path / "nvfp4.safetensors")
    save_file({"x.weight": torch.zeros(4, 2, dtype=torch.uint8), "x.weight_scale": torch.zeros(4, 1),
               "x.comfy_quant": marker({"format": "mxfp4"})}, path)
    with pytest.raises(NotImplementedError, match="mxfp4"):
        load_state_dict(path)
    assert comfy_io.describe_quant(path)[1] == {"mxfp4"}


def test_int8_without_scale_fails(tmp_path):
    path = str(tmp_path / "noscale.safetensors")
    save_file({"x.weight": torch.zeros(4, 4, dtype=torch.int8)}, path)
    with pytest.raises(ValueError, match="weight_scale"):
        load_state_dict(path)


# ------------------------------------------------- real Krea 2 checkpoints ---
real = pytest.mark.skipif(not (os.path.exists(INT8_FILE) and os.path.exists(FP8_FILE)),
                          reason="Krea 2 int8_convrot / fp8_scaled checkpoints not present")

LAYERS = ["blocks.0.attn.wq", "blocks.0.attn.gate", "blocks.13.mlp.down", "blocks.27.attn.wo", "blocks.27.mlp.up"]


@real
def test_real_int8_convrot_matches_fp8_checkpoint():
    """Both files are quantizations of the same model, so the dequantized weights must agree
    to within quantization error — and only if the convrot rotation is undone correctly."""
    inc = tuple(LAYERS)
    i8 = load_state_dict(INT8_FILE, include=inc)
    f8 = load_state_dict(FP8_FILE, include=inc)
    for name in LAYERS:
        assert i8.quant[name].format == "int8_tensorwise" and i8.quant[name].rot_size == 256
        w_i8 = dequantize(i8.tensors[f"{name}.weight"], i8.quant[name], torch.float32)
        w_f8 = dequantize(f8.tensors[f"{name}.weight"], f8.quant[name], torch.float32)
        err = rel_err(w_i8, w_f8)
        q = i8.quant[name]
        wrong = rel_err(i8.tensors[f"{name}.weight"].float() * q.weight_scale, w_f8)
        print(f"{name}: int8 vs fp8 rel err {err:.4f} (without un-rotation: {wrong:.3f})")
        assert err < 0.05, name
        assert wrong > 0.5, name


@real
def test_real_unquantized_tensors_identical():
    names = ("first.weight", "txtfusion.projector.weight", "blocks.5.mod.lin", "last.linear.weight")
    i8 = load_state_dict(INT8_FILE, include=names)
    f8 = load_state_dict(FP8_FILE, include=names)
    for n in names:
        a = dequantize(i8.tensors[n], i8.quant.get(n.removesuffix(".weight")), torch.float32)
        b = dequantize(f8.tensors[n], f8.quant.get(n.removesuffix(".weight")), torch.float32)
        assert rel_err(a, b) < 0.05, n


# ---------------------------------------------------------- sharded files ---
def _shard(tiny_file, folder, header_meta: bool = False):
    """The tiny file split as Hugging Face shards it: two files and a model.safetensors.index.json. With
    ``header_meta``, each shard declares its quantized layers in its header instead of comfy_quant markers."""
    from safetensors.torch import load_file

    path, _ = tiny_file
    sd = load_file(path)
    if header_meta:
        layers = {k[: -len(".comfy_quant")]: json.loads(bytes(v.tolist())) for k, v in sd.items()
                  if k.endswith(".comfy_quant")}
        sd = {k: v for k, v in sd.items() if not k.endswith(".comfy_quant")}
    first = {k: v for k, v in sd.items() if k.startswith(("a.", "emb."))}
    parts = {"model-00001-of-00002.safetensors": first,
             "model-00002-of-00002.safetensors": {k: v for k, v in sd.items() if k not in first}}
    folder.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        meta = None
        if header_meta:
            mine = {p: c for p, c in layers.items() if f"{p}.weight" in part}
            meta = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": mine})}
        save_file(part, str(folder / name), metadata=meta)
    index = {"metadata": {"total_size": 0}, "weight_map": {k: n for n, part in parts.items() for k in part}}
    (folder / comfy_io.INDEX_NAME).write_text(json.dumps(index))
    return sd


@pytest.mark.parametrize("header_meta", [False, True])
def test_a_sharded_checkpoint_loads_as_one_file(tiny_file, tmp_path, header_meta):
    """A Hugging Face folder of shards, its index, or a cache folder's main snapshot: the same tensors and quantized
    layers as the single file, from either shard."""
    path, _ = tiny_file
    snap = tmp_path / "models--owner--repo" / "snapshots" / "abc123"
    _shard(tiny_file, snap, header_meta)
    (tmp_path / "models--owner--repo" / "refs").mkdir()
    (tmp_path / "models--owner--repo" / "refs" / "main").write_text("abc123")
    single = load_state_dict(path)
    for where in (snap, snap / comfy_io.INDEX_NAME, tmp_path / "models--owner--repo"):
        assert comfy_io.is_sharded(str(where))
        assert [os.path.basename(p) for p in comfy_io.shard_files(str(where))] == [
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        loaded = load_state_dict(str(where))
        assert loaded.tensors.keys() == single.tensors.keys()
        assert all(torch.equal(loaded.tensors[k], single.tensors[k]) for k in single.tensors)
        assert {k: (q.format, q.rot_size) for k, q in loaded.quant.items()} == \
            {k: (q.format, q.rot_size) for k, q in single.quant.items()}
        assert sorted(comfy_io.read_header(str(where))[0]) == sorted(
            k for k in comfy_io.read_header(path)[0] if not (header_meta and k.endswith(".comfy_quant")))
        assert comfy_io.describe_quant(str(where)) == comfy_io.describe_quant(path)
        part = load_state_dict(str(where), include=("c.",))  # from the second shard only
        assert set(part.tensors) == {"c.weight"} and set(part.quant) == {"c"}
    assert not comfy_io.is_sharded(path) and comfy_io.shard_files(path) == [path]


def test_a_folder_of_one_unindexed_file(tmp_path):
    folder = tmp_path / "repo"
    folder.mkdir()
    save_file({"x.weight": torch.ones(2, 2)}, str(folder / "model.safetensors"))
    assert comfy_io.shard_files(str(folder)) == [str(folder / "model.safetensors")]
    assert torch.equal(load_state_dict(str(folder)).tensors["x.weight"], torch.ones(2, 2))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        comfy_io.shard_files(str(tmp_path / "empty"))


# ------------------------------------------------------- format edge cases ---
def test_legacy_fp8_scaled_names(tmp_path):
    """Older ComfyUI fp8_scaled files: ``scale_weight`` + a root ``scaled_fp8`` marker."""
    torch.manual_seed(0)
    lin = nn.Linear(256, 64, bias=False)
    q, s = quantize_fp8(lin.weight.data)
    path = str(tmp_path / "legacy.safetensors")
    save_file({"lin.weight": q, "lin.scale_weight": s.reshape(1), "lin.scale_input": torch.ones(1),
               "scaled_fp8": torch.zeros(2, dtype=torch.float8_e4m3fn)}, path)
    loaded = load_state_dict(path)
    assert "scaled_fp8" not in loaded.tensors
    assert rel_err(dequantize(loaded.tensors["lin.weight"], loaded.quant["lin"], torch.float32), lin.weight.data) < 0.05
    assert comfy_io.describe_quant(path) == ({"fp8 scaled"}, set())


def test_header_declared_unsupported_format(tmp_path):
    path = str(tmp_path / "hdr.safetensors")
    meta = {"_quantization_metadata": json.dumps({"layers": {"x": {"format": "mxfp4"}}})}
    save_file({"x.weight": torch.zeros(4, 2, dtype=torch.uint8), "x.weight_scale": torch.ones(1)}, path, metadata=meta)
    with pytest.raises(NotImplementedError, match="mxfp4"):
        load_state_dict(path)
    assert comfy_io.describe_quant(path)[1] == {"mxfp4"}


def test_blockwise_scale_rejected(tmp_path):
    path = str(tmp_path / "block.safetensors")
    save_file({"x.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn), "x.weight_scale": torch.ones(2, 2)}, path)
    with pytest.raises(NotImplementedError, match="block-wise"):
        load_state_dict(path)


def test_per_row_scale_applied_in_fp32():
    """bf16 compute must not round the per-row scale (it would bias whole output channels)."""
    torch.manual_seed(0)
    codes = torch.randint(-127, 128, (64, 256), dtype=torch.int8)
    scale = torch.rand(64, 1) * 1e-3 + 1.0001e-3  # values that bf16 can't represent exactly
    lin = QuantLinear(256, 64, codes, scale, None, torch.bfloat16)
    x = torch.randn(4096, 256)
    ref = x.to(torch.bfloat16).float() @ (codes.float() * scale).T

    def channel_bias(y):  # systematic error per output channel, relative to its magnitude
        return (((y.float() - ref) * ref.sign()).sum(0) / ref.abs().sum(0)).abs().max().item()

    rounded = x.to(torch.bfloat16) @ (codes.to(torch.bfloat16) * scale.to(torch.bfloat16)).T  # bf16 scale
    ours, naive = channel_bias(lin(x)), channel_bias(rounded)
    assert ours < 1e-3 and ours < naive / 3, (ours, naive)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_load_straight_to_gpu(tiny_file):
    path, _ = tiny_file
    loaded = load_state_dict(path, device=torch.device("cuda"))
    assert all(t.is_cuda for t in loaded.tensors.values())
    assert loaded.quant["a"].weight_scale.is_cuda
    with torch.device("meta"):
        model = Tiny()
    materialize(model, loaded, torch.bfloat16, device="cuda")
    out = model(torch.randn(1, 3, 512, device="cuda"))
    assert out.is_cuda and torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fp16_autocast_does_not_overflow_unscaled_codes():
    """Under fp16 autocast (the MiniMax video VAE's decode recipe) raw code products pass 65504; the
    scale has to be folded into the weight before the fp16 matmul."""
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.manual_seed(0)
    codes = torch.randint(-127, 128, (256, 8192), dtype=torch.int8, device=dev)
    scale = torch.full((256, 1), 5e-4, device=dev)
    lin = QuantLinear(8192, 256, codes, scale, None, torch.float32)
    x = torch.randn(64, 8192, device=dev) * 300
    ref = x @ (codes.float() * scale).T
    assert not torch.isfinite(x.half() @ codes.half().T).all()  # the unscaled product really overflows
    with torch.autocast("cuda", dtype=torch.float16):
        y = lin(x)
    assert torch.isfinite(y).all() and rel_err(y.float(), ref) < 1e-2


# ------------------------------------------------------------------ nvfp4 ---
E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Row-major (rows, cols) -> cuBLAS 128x4 tile layout (inverse of the loader's unswizzle)."""
    rows, cols = scales.shape
    nr, nc = (rows + 127) // 128, (cols + 3) // 4
    x = scales.reshape(nr, 128, nc, 4).permute(0, 2, 1, 3).reshape(nr, nc, 4, 32, 4)
    return x.transpose(2, 3).reshape(rows, cols)


def quantize_nvfp4(w: torch.Tensor):
    """Reference quantizer: per-tensor scale, E4M3 block scales per 16, nearest E2M1 code,
    packed high nibble first, scales swizzled — the ComfyUI layout."""
    rows, cols = w.shape
    blocks = w.float().view(rows, cols // 16, 16)
    pts = blocks.abs().amax() / (6.0 * 448.0)
    bscale = (blocks.abs().amax(-1) / 6.0 / pts).clamp(min=2**-9).to(torch.float8_e4m3fn)
    scaled = blocks / (bscale.float() * pts).unsqueeze(-1)
    idx = (scaled.abs().unsqueeze(-1) - E2M1_POS).abs().argmin(-1)
    codes = (idx | ((scaled < 0).long() << 3)).view(rows, cols).to(torch.uint8)
    packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
    return packed, swizzle_scales(bscale), pts.reshape(())


def test_nvfp4_swizzle_roundtrip():
    s = torch.randn(256, 8)
    assert torch.equal(comfy_io.unswizzle_nvfp4_scales(swizzle_scales(s), 256, 8), s)


@pytest.mark.parametrize("awq", [False, True])
def test_nvfp4_load_and_forward(tmp_path, awq):
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.05
    pre = (torch.rand(512) + 0.5) if awq else None
    stored = w / pre.unsqueeze(0) if awq else w  # AWQ: activations are multiplied by pre, weights divided
    packed, bscale, pts = quantize_nvfp4(stored)
    sd = {"lin.weight": packed, "lin.weight_scale": bscale, "lin.weight_scale_2": pts.float(),
          "lin.comfy_quant": marker({"format": "nvfp4", "full_precision_matrix_mult": True})}
    if awq:
        sd["lin.pre_quant_scale"] = pre.to(torch.bfloat16)
    path = str(tmp_path / "nv.safetensors")
    save_file(sd, path)
    loaded = load_state_dict(path)
    assert loaded.quant["lin"].format == "nvfp4" and comfy_io.describe_quant(path) == ({"nvfp4"}, set())
    real = dequantize(loaded.tensors["lin.weight"], loaded.quant["lin"], torch.float32)
    assert rel_err(real, w) < 0.12  # fp4 is coarse; the structure must still match

    from dainamo.engine.fp8_linear import Nvfp4Linear
    with torch.device("meta"):
        meta = nn.Module()
        meta.lin = nn.Linear(512, 256, bias=False)
    materialize(meta, loaded, torch.float32)
    assert isinstance(meta.lin, Nvfp4Linear)
    x = torch.randn(4, 512, requires_grad=True)
    y = meta.lin(x)
    x2 = x.detach().clone().requires_grad_(True)
    y2 = x2 @ real.T  # the folded weight on raw activations
    assert torch.allclose(y, y2, atol=1e-4, rtol=1e-4)
    y.square().sum().backward()
    y2.square().sum().backward()
    # gradient w.r.t. the raw input goes through the AWQ input scale as well
    assert torch.allclose(x.grad, x2.grad, atol=1e-3, rtol=1e-3)

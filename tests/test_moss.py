"""MOSS-Image: parity with OpenMOSS's reference code and the report's training distribution.

The parity tests import ``pipeline_moss_image.py`` straight from the Hugging Face cache (downloaded
once, ~25 kB) and are skipped when it can't be fetched.

    .venv/bin/python -m pytest tests/test_moss.py -v
"""

import importlib.util
import json
import math

import pytest
import torch
from safetensors.torch import save_file

from dainamo.engine import comfy_io
from dainamo.engine.families.moss_image.driver import (
    PRETRAIN_CUTOFF, PRETRAIN_MU, PRETRAIN_S, PRETRAIN_UNIFORM, T_TRAIN, pack_latents, pretrain_sigmas,
    sampling_sigmas, unpack_latents,
)
from dainamo.engine.families.moss_image.model import MossConfig, MossImageDiT
from dainamo.engine.lora import LoRANetwork


@pytest.fixture(scope="module")
def reference():
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("OpenMOSS-Team/MOSS-Image-SFT", "pipeline_moss_image.py")
        spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:  # offline, or the reference needs something we don't have
        pytest.skip(f"reference pipeline unavailable: {e}")


SMALL = dict(dim=256, ffn_dim=512, num_heads=4, num_layers=3, text_dim=96, freq_dim=64, latent_channels=128)


def _ours_from_reference(ref_model) -> MossImageDiT:
    """Same weights, with the reference's separate q/k/v fused into proj.qkv = [q; k; v]."""
    sd = dict(ref_model.state_dict())
    sd.pop("rope.bands")
    for i in range(SMALL["num_layers"]):
        p = f"layers.{i}.proj."
        sd[p + "qkv.weight"] = torch.cat([sd.pop(p + n + ".weight") for n in ("q", "k", "v")])
    ours = MossImageDiT(MossConfig(**SMALL))
    ours.compute_dtype = torch.float32
    ours.load_state_dict(sd, strict=True)
    return ours


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_dit_matches_reference(reference):
    torch.manual_seed(0)
    ref = reference.MOSSImageTransformer(**SMALL).float().eval()  # default (random) init, nothing zeroed
    ours = _ours_from_reference(ref).eval()
    x = torch.randn(2, 128, 6, 4)
    t = torch.tensor([970.0, 120.0])
    texts = [torch.randn(7, SMALL["text_dim"]), torch.randn(3, SMALL["text_dim"])]
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[0, :7], mask[1, :3] = True, True
    padded = torch.zeros(2, 7, SMALL["text_dim"])
    padded[0], padded[1, :3] = texts[0], texts[1]
    with torch.no_grad():
        expected = ref(x, t, padded, mask).sample
        assert _rel(ours(x, t, texts), expected) < 1e-5
        # skipping blocks (the negative branch's uncond skip) matches too
        skipped = ref(x, t, padded, mask, skip_layers=[1]).sample
        assert _rel(ours(x, t, texts, skip_layers=frozenset({1})), skipped) < 1e-5
        assert _rel(skipped, expected) > 1e-2  # and skipping actually changes the output
        # negative control: the comparison is sensitive to the fused q/k/v order
        w = ours.layers[0].proj["qkv"].weight
        d = SMALL["dim"]
        w.data = torch.cat([w[d : 2 * d], w[:d], w[2 * d :]])
        assert _rel(ours(x, t, texts), expected) > 1e-3


@pytest.mark.parametrize("steps", [8, 30, 50])
def test_sampling_sigmas_match_reference_scheduler(reference, steps):
    sched = reference.MOSSImageScheduler()
    sched.set_timesteps(steps)
    assert torch.allclose(sampling_sigmas(steps, 5.0), sched.sigmas.float(), atol=1e-7)
    assert torch.allclose(sampling_sigmas(steps, 5.0)[:-1] * T_TRAIN, sched.timesteps.float(), atol=1e-3)


def test_pretrain_sigmas_follow_report_eq4():
    """Empirical frequencies of the 1000 grid points vs p_j from the report's eq. (4)."""
    n = 2_000_000
    g = torch.Generator().manual_seed(0)
    s = pretrain_sigmas(n, g)
    j = torch.round(s * T_TRAIN).long()
    assert j.min() >= 1 and j.max() <= T_TRAIN
    assert torch.allclose(s, j.double() / T_TRAIN)
    counts = torch.bincount(j, minlength=T_TRAIN + 1)[1:].double()

    def F(x):
        return 1 - 0.5 * math.erfc(-((math.log((1 - x) / x) - PRETRAIN_MU) / PRETRAIN_S) / math.sqrt(2))

    a, rho = PRETRAIN_CUTOFF, PRETRAIN_UNIFORM
    Fa = F(a)
    edges = [0.0] + [F(a * k / T_TRAIN) for k in range(1, T_TRAIN + 1)]
    p = torch.tensor([(1 - rho) * (edges[k] - edges[k - 1]) / Fa + rho / T_TRAIN for k in range(1, T_TRAIN + 1)],
                     dtype=torch.float64)
    assert abs(p.sum().item() - 1) < 1e-9
    # compare in 20 coarse bins (each ~100k samples) to keep sampling noise small
    emp, exp = counts.view(20, 50).sum(1) / n, p.view(20, 50).sum(1)
    assert (emp - exp).abs().max().item() < 2e-3, (emp, exp)


def test_pack_unpack_inverse():
    z = torch.randn(2, 32, 8, 12)
    assert torch.equal(unpack_latents(pack_latents(z)), z)
    packed = pack_latents(z)
    # channel = c * 4 + 2 * dy + dx at pixel (2h + dy, 2w + dx)
    assert packed[0, 3 * 4 + 2 * 1 + 0, 2, 5] == z[0, 3, 2 * 2 + 1, 2 * 5 + 0]


def test_lora_on_fused_qkv_and_comfy_hints_skipped(tmp_path):
    torch.manual_seed(0)
    model = MossImageDiT(MossConfig(**SMALL))
    sd = {k: v for k, v in model.state_dict().items()}
    sd["rope.bands"] = torch.zeros(SMALL["dim"] // SMALL["num_heads"] // 8)  # present in ComfyUI files
    for i in range(SMALL["num_layers"]):
        sd[f"layers.{i}.comfy_attention.config"] = torch.tensor(list(json.dumps({"attention": "x"}).encode()),
                                                                dtype=torch.uint8)
    path = str(tmp_path / "moss.safetensors")
    save_file(sd, path)
    loaded = comfy_io.load_state_dict(path)
    assert not any(k.endswith("comfy_attention.config") for k in loaded.tensors)
    assert "rope.bands" in loaded.tensors  # the driver drops it; RoPE bands are derived in fp32
    net = LoRANetwork(model, ["layers.*.proj.*"], rank=4, alpha=4)
    assert len(net.modules) == 4 * SMALL["num_layers"]
    for m in net.modules.values():
        torch.nn.init.normal_(m.lora_B.weight, std=0.01)
    model.compute_dtype = torch.float32
    model(torch.randn(1, 128, 4, 4), torch.tensor([500.0]), [torch.randn(5, SMALL["text_dim"])]).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    keys = net.state_dict()
    assert "diffusion_model.layers.0.proj.qkv.lora_A.weight" in keys

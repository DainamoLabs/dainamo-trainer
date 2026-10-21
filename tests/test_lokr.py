"""LoKr (``engine.lora``): the Kronecker-product layer against the product built in full, its saved keys, the
Sandbox applying LoKr files (full, low-rank and kohya-named), and a LoKr job trained end to end.

    .venv/bin/python -m pytest tests/test_lokr.py -v
"""

import json

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from dainamo.engine import memory
from dainamo.engine.lora import AdapterStack, LoKrLinear, LoRANetwork, factorize


class Net(nn.Module):
    """Two Linears with sizes that split several ways."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(48, 96), nn.Linear(96, 48)])

    def forward(self, x):
        return self.blocks[1](torch.relu(self.blocks[0](x)))


def folded(model, deltas):
    """``model`` with each {name: delta W} added to its weights (the reference the adapters must match)."""
    ref = Net().double()
    ref.load_state_dict({k.replace(".base.", "."): v for k, v in model.state_dict().items()})
    with torch.no_grad():
        for name, d in deltas.items():
            ref.get_submodule(name).weight += d
    return ref


def test_factorize_splits_like_lycoris():
    assert factorize(3072, 16) == (16, 192)
    assert factorize(12288, 8) == (8, 1536)
    assert factorize(3072, -1) == (48, 64)  # the most even split
    assert factorize(360, 16) == (15, 24)  # 16 doesn't divide it: the largest divisor below
    assert factorize(128, 16) == (8, 16)  # no further than the square root
    assert factorize(127, 4) == (1, 127)  # a prime only splits as 1 x itself


def test_a_lokr_layer_adds_the_kronecker_product_of_its_two_matrices():
    torch.manual_seed(0)
    base = nn.Linear(48, 96).double()
    layer = LoKrLinear(base, factor=4).double()
    assert layer.lokr_w1.shape == (4, 4) and layer.lokr_w2.shape == (24, 12)
    x = torch.randn(2, 5, 48, dtype=torch.float64)
    assert torch.equal(layer(x), base(x))  # w2 starts at zero
    layer(x).sum().backward()
    assert layer.lokr_w2.grad.abs().sum() > 0  # and w1, random, passes it gradients

    with torch.no_grad():
        layer.lokr_w2.normal_()
    layer.multiplier = -0.5
    expected = base(x) + x @ torch.kron(layer.lokr_w1, layer.lokr_w2).t() * -0.5
    assert torch.allclose(layer(x), expected)


def test_a_lokr_network_saves_lycoris_keys_and_loads_them_back(tmp_path):
    torch.manual_seed(0)
    net = LoRANetwork(Net().double(), ["blocks.*"], 4, 4, kind="lokr", factor=8)
    with torch.no_grad():
        for m in net.modules.values():
            m.lokr_w2.normal_()
    path = str(tmp_path / "l.safetensors")
    net.save(path, {"k": "v"}, dtype=torch.float64)
    sd = load_file(path)
    # 48 splits as 6 x 8 (no further than its square root) and 96 as 8 x 12; no alpha (full matrices)
    assert {k: tuple(v.shape) for k, v in sd.items()} == {
        "diffusion_model.blocks.0.lokr_w1": (8, 6), "diffusion_model.blocks.0.lokr_w2": (12, 8),
        "diffusion_model.blocks.1.lokr_w1": (6, 8), "diffusion_model.blocks.1.lokr_w2": (8, 12)}
    assert net.num_parameters() == sum(v.numel() for v in sd.values())
    assert net.num_parameters() == memory.lora_params(
        {"blocks.0.weight": ("F32", (96, 48)), "blocks.1.weight": ("F32", (48, 96))}, ["blocks.*"], 4, "lokr", 8)

    other = LoRANetwork(Net().double(), ["blocks.*"], 4, 4, kind="lokr", factor=8)
    other.model.load_state_dict({k: v for k, v in net.model.state_dict().items() if "lokr" not in k}, strict=False)
    other.load(path)
    x = torch.randn(3, 48, dtype=torch.float64)
    assert torch.equal(other.model(x), net.model(x))


def lokr_sd(kind, seed=0, alpha=2.0):
    """A LoKr file for Net: full matrices, low-rank w2 with an alpha, or the same with kohya-style names."""
    g = torch.Generator().manual_seed(seed)
    sd, deltas = {}, {}
    for name, (out_f, in_f) in (("blocks.0", (96, 48)), ("blocks.1", (48, 96))):
        (om, on), (im, inn) = factorize(out_f, 4), factorize(in_f, 4)
        w1 = torch.randn(om, im, generator=g, dtype=torch.float64)
        k = "lora_unet_" + name.replace(".", "_") if kind == "kohya" else f"diffusion_model.{name}"
        if kind == "full":
            w2 = torch.randn(on, inn, generator=g, dtype=torch.float64)
            sd |= {f"{k}.lokr_w1": w1, f"{k}.lokr_w2": w2}
            deltas[name] = torch.kron(w1, w2)
        else:
            a = torch.randn(on, 2, generator=g, dtype=torch.float64)
            b = torch.randn(2, inn, generator=g, dtype=torch.float64)
            sd |= {f"{k}.lokr_w1": w1, f"{k}.lokr_w2_a": a, f"{k}.lokr_w2_b": b, f"{k}.alpha": torch.tensor(alpha)}
            deltas[name] = torch.kron(w1, a @ b) * alpha / 2
    return sd, deltas


@pytest.mark.parametrize("kind", ["full", "lowrank", "kohya"])
def test_the_sandbox_applies_lokr_files(kind):
    torch.manual_seed(0)
    model = Net().double()
    x = torch.randn(3, 48, dtype=torch.float64)
    base = model(x)
    stack = AdapterStack(model, torch.device("cpu"), torch.float64)
    sd, deltas = lokr_sd(kind)
    report = stack.set([("lokr", sd, 0.6)])
    assert report == [{"label": "lokr", "matched": 2, "total": 2}]
    expected = folded(model, {n: 0.6 * d for n, d in deltas.items()})(x)
    assert torch.allclose(model(x), expected)
    stack.set_enabled(False)
    assert torch.allclose(model(x), base)


def test_the_sandbox_explains_lokr_files_it_cant_apply():
    stack = AdapterStack(Net().double(), torch.device("cpu"), torch.float64)
    sd, _ = lokr_sd("full")
    with pytest.raises(ValueError, match="DoRA"):
        stack.set([("dora", {**sd, "diffusion_model.blocks.0.dora_scale": torch.ones(96, 1)}, 1.0)])
    with pytest.raises(ValueError, match="lokr_t2"):
        stack.set([("conv", {**sd, "diffusion_model.blocks.0.lokr_t2": torch.ones(2, 2, 3, 3)}, 1.0)])
    with pytest.raises(ValueError, match="lokr_w1/lokr_w2"):
        stack.set([("loha", {"diffusion_model.blocks.0.hada_w1_a": torch.ones(96, 2)}, 1.0)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer needs a GPU")
def test_a_lokr_job_trains_saves_and_continues(tmp_path, monkeypatch, capsys):
    from dainamo.engine import trainer as trainer_mod

    from test_trainer import TinyFamily, _images

    monkeypatch.setattr(trainer_mod, "get_family", lambda name: TinyFamily)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _images(str(tmp_path / "train"), [(200, 30, 30), (190, 40, 20), (60, 160, 60)])
    job = {"name": "t", "family": "tiny", "profile": {},
           "datasets": [{"path": str(tmp_path / "train"), "trigger": "", "repeats": 1, "resolution": 64}],
           "settings": {"steps": 6, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1, "optimizer": "adamw",
                        "learning_rate": 0.01, "save_every": 0, "sample_every": 0, "network_type": "lokr",
                        "lokr_factor": 4}}
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    assert "LoKr (factor 4): 1 layers" in capsys.readouterr().out
    path = str(job_dir / "output" / "t.safetensors")
    first = load_file(path)
    assert sorted(first) == ["diffusion_model.blocks.0.lokr_w1", "diffusion_model.blocks.0.lokr_w2"]
    assert first["diffusion_model.blocks.0.lokr_w2"].abs().sum() > 0  # it trained
    with safe_open(path, "pt") as f:
        meta = f.metadata()
    assert meta["dainamo.network"] == "lokr" and meta["ss_network_module"] == "lycoris.kohya"
    assert json.loads(meta["ss_network_args"]) == {"algo": "lokr", "factor": "4", "full_matrix": "True"}
    assert "ss_network_dim" not in meta

    job["settings"]["steps"] = 9  # continued, it picks up its LoKr where it stopped
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    after = load_file(path)
    assert not torch.equal(after["diffusion_model.blocks.0.lokr_w2"], first["diffusion_model.blocks.0.lokr_w2"])
    assert torch.equal(load_file(str(job_dir / "output" / "t_000006.safetensors"))["diffusion_model.blocks.0.lokr_w2"],
                       first["diffusion_model.blocks.0.lokr_w2"])

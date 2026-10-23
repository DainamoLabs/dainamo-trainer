"""Full finetuning (``engine.finetune``): the fused per-weight optimizer step, bf16 stochastic rounding, the saved
model's layout, the trainer's finetune job end to end with a tiny model, and the server's rules for one.

    .venv/bin/python -m pytest tests/test_finetune.py -v
"""

import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from dainamo.engine import comfy_io
from dainamo.engine.finetune import FullModel, FusedOptimizer, save_model, stochastic_round_, write_safetensors

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_stochastic_rounding_is_unbiased_and_exact_on_bf16_values():
    torch.manual_seed(0)
    # a quarter of the way between 1 and the next bf16 value up (2^-7 apart), either sign
    for v in (1 + 2**-9, -(1 + 2**-9)):
        out = torch.empty(200_000, dtype=torch.bfloat16)
        stochastic_round_(out, torch.full((200_000,), v))
        assert set(out.float().abs().unique().tolist()) == {1.0, 1 + 2**-7}
        assert out.float().mean().item() == pytest.approx(v, abs=2e-5)
    exact = torch.randn(1000).bfloat16().float()  # values bf16 holds stay as they are
    out = torch.empty(1000, dtype=torch.bfloat16)
    assert torch.equal(stochastic_round_(out, exact).float(), exact)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(6, 16)
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
        self.norm = nn.LayerNorm(16)
        self.out = nn.Linear(16, 6, bias=False)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:  # checkpointed, as the families' blocks are in training
            h = h + torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
        return self.out(self.norm(h))


def _steps(model, opt, sched, n, fused):
    torch.manual_seed(1)
    x, y = torch.randn(4, 6), torch.randn(4, 6)
    for _ in range(n):
        F.mse_loss(model(x), y).backward()
        if fused:
            assert all(p.grad is None for p in model.parameters())  # each was stepped and dropped in backward
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)


def test_the_fused_step_matches_the_optimizer_stepped_as_usual():
    """On fp32 weights (no rounding) each weight's own Adafactor step during backward is the same update as one
    Adafactor stepping them all after it, warmup and weight decay included."""
    from transformers.optimization import Adafactor

    torch.manual_seed(0)
    a = Net()
    b = Net()
    b.load_state_dict(a.state_dict())
    warm = lambda s: min(1.0, (s + 1) / 3)  # noqa: E731
    fused = FusedOptimizer(a.parameters(), "adafactor", lr=0.01, weight_decay=0.1)
    plain = Adafactor(b.parameters(), lr=0.01, weight_decay=0.1, relative_step=False, scale_parameter=False,
                      warmup_init=False)
    _steps(a, fused, torch.optim.lr_scheduler.LambdaLR(fused, warm), 6, True)
    _steps(b, plain, torch.optim.lr_scheduler.LambdaLR(plain, warm), 6, False)
    for (k, p), q in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6), k
    # all of its state is in one place (for parking it in system RAM), and nothing is left in the proxies
    assert len(fused.state) == len(list(a.parameters()))
    assert all(px.numel() == 0 for px in fused.proxies)


def test_bf16_weights_keep_updates_far_smaller_than_their_spacing():
    """At a finetune's learning rate each update is a fraction of bf16's spacing; rounded to nearest every one is
    lost, stochastically rounded they add up."""
    w = nn.Parameter(torch.ones(20_000, dtype=torch.bfloat16))
    opt = FusedOptimizer([w], "adafactor", lr=1e-4)
    for _ in range(40):
        (w.float() * 1.0).sum().backward()  # a constant gradient: Adafactor's normalized update is about lr
    assert 0.994 < w.float().mean().item() < 0.998  # 40 updates of 1e-4 from 1.0; bf16 steps by 2^-8 below 1


def test_optimizer_state_survives_a_save_and_load():
    torch.manual_seed(0)
    a = Net()
    opt = FusedOptimizer(a.parameters(), "adafactor", lr=0.01)
    _steps(a, opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), 3, True)
    saved = opt.state_dict()
    opt.remove_hooks()
    again = FusedOptimizer(a.parameters(), "adafactor", lr=0.01)
    again.load_state_dict(saved)
    assert [s["step"] for s in again.state.values()] == [3] * len(again.proxies)
    with pytest.raises(ValueError):
        FusedOptimizer(a.parameters(), "adamw8bit", lr=0.01).load_state_dict(saved)


def _int8(w: torch.Tensor):
    scale = w.abs().amax(1).clamp_min(1e-8) / 127
    return (w / scale[:, None]).round().to(torch.int8), scale.float()


def _blob(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def test_a_saved_model_is_laid_out_like_its_source(tmp_path):
    """A full checkpoint's prefix and other parts, and the non-weight tensors, are kept; the model's own layers
    are written in bf16 without their quantization scales and markers (in the header too)."""
    torch.manual_seed(0)
    model = Net().to(torch.bfloat16)
    sd = model.state_dict()
    p = "model.diffusion_model."
    src = {f"{p}{k}": v for k, v in sd.items()}
    q, scale = _int8(sd["blocks.0.weight"].float())
    src.update({f"{p}blocks.0.weight": q, f"{p}blocks.0.weight_scale": scale,
                f"{p}blocks.0.comfy_quant": _blob({"format": "int8_tensorwise"}),
                f"{p}blocks.0.comfy_attention.config": _blob({"attention": "comfy_kitchen_int8"}),
                f"{p}blocks.comfy_attention.config": _blob({"attention": "comfy_kitchen_int8"}),  # on a block
                f"{p}rope.bands": torch.arange(4, dtype=torch.float32), "vae.decoder.weight": torch.ones(2, 2)})
    meta = {"_quantization_metadata": json.dumps({"layers": {"blocks.1": {"format": "float8_e4m3fn"},
                                                             "text.proj": {"format": "float8_e4m3fn"}}}),
            "modelspec.title": "tiny"}
    source = str(tmp_path / "src.safetensors")
    save_file(src, source, metadata=meta)

    out = str(tmp_path / "out.safetensors")
    FullModel(model, source).save(out, {"dainamo.network": "full"})
    got = load_file(out)
    assert set(got) == {f"{p}{k}" for k in sd} | {f"{p}rope.bands", "vae.decoder.weight"}
    for k, v in sd.items():
        assert got[f"{p}{k}"].dtype == torch.bfloat16 and torch.equal(got[f"{p}{k}"], v), k
    assert torch.equal(got[f"{p}rope.bands"], src[f"{p}rope.bands"])
    with safe_open(out, "pt") as f:
        m = f.metadata()
    assert m["modelspec.title"] == "tiny" and m["dainamo.network"] == "full"
    assert json.loads(m["_quantization_metadata"])["layers"] == {"text.proj": {"format": "float8_e4m3fn"}}
    # and it loads the way any diffusion model does
    loaded = comfy_io.load_state_dict(out, strip_prefix=p, only_prefixed=True)
    assert not loaded.quant and set(loaded.tensors) == set(sd) | {"rope.bands"}


def test_safetensors_written_a_tensor_at_a_time_read_back(tmp_path):
    tensors = [("b", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)), ("a", torch.tensor(3.5)),
               ("c", torch.zeros(0, 4, dtype=torch.int8)), ("d", torch.arange(5, dtype=torch.float32)[::2])]
    path = str(tmp_path / "x.safetensors")
    write_safetensors(path, tensors, {"k": "v"})
    got = load_file(path)
    for k, t in tensors:
        assert got[k].dtype == t.dtype and torch.equal(got[k], t), k
    with safe_open(path, "pt") as f:
        assert f.metadata() == {"k": "v"}
    assert os.listdir(tmp_path) == ["x.safetensors"]  # no temporary file left


# --- the trainer ---------------------------------------------------------------------------------------------


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        self.gradient_checkpointing = False

    def forward(self, x):
        h = x.permute(0, 2, 3, 1)
        for b in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = h + torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
            else:
                h = h + b(h)
        return h.permute(0, 3, 1, 2)


def _tiny_family():
    from tests.test_trainer import TinyFamily

    class TinyFull(TinyFamily):
        """TinyFamily's encoders with a diffusion model loaded from its file as the real families' are."""
        from dainamo.engine.families.base import FamilyInfo
        info = FamilyInfo(id="tiny", label="Tiny", media=["image"], profile_fields=[], finetune={})
        load_transformer = TinyFamily.__mro__[1].load_transformer  # FamilyDriver's

        def build_transformer(self, shapes):
            with torch.device("meta"):
                return TinyDiT()

        def check_transformer_state(self, path, loaded):
            loaded.tensors.pop("rope.bands", None)

    return TinyFull


@gpu
def test_a_finetune_job_trains_saves_whole_models_and_continues(tmp_path, monkeypatch):
    from dainamo.engine import trainer as trainer_mod
    from tests.test_trainer import _images

    Tiny = _tiny_family()
    monkeypatch.setattr(trainer_mod, "get_family", lambda name: Tiny)
    torch.manual_seed(0)
    base = TinyDiT().to(torch.bfloat16).state_dict()
    q, scale = _int8(base["blocks.0.weight"].float())  # an int8 layer, expanded to bf16 to train
    source = str(tmp_path / "tiny_int8.safetensors")
    save_file({**base, "blocks.0.weight": q, "blocks.0.weight_scale": scale,
               "blocks.0.comfy_quant": _blob({"format": "int8_tensorwise"}),
               "rope.bands": torch.arange(4, dtype=torch.float32)}, source)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _images(str(tmp_path / "train"), [(200, 30, 30), (190, 40, 20), (60, 160, 60)])
    job = {"name": "t", "family": "tiny", "profile": {"diffusion_model": source},
           "datasets": [{"path": str(tmp_path / "train"), "trigger": "", "repeats": 1, "resolution": 64}],
           "settings": {"steps": 6, "resolution": 64, "seed": 1, "optimizer": "adafactor", "learning_rate": 0.01,
                        "save_every": 3, "sample_every": 0, "keep_last": 1, "finetune_enabled": True,
                        "grad_accum": 1, "max_grad_norm": 1.0}}
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    st = json.loads((job_dir / "status.json").read_text())
    assert [(c["step"], c["file"], c["final"]) for c in st["checkpoints"]] == [
        (3, "t_000003.safetensors", False), (6, "t.safetensors", True)]
    first = load_file(str(job_dir / "output" / "t.safetensors"))
    assert set(first) == set(base) | {"rope.bands"}  # no int8 scale or marker: every layer is bf16
    assert all(first[k].dtype == torch.bfloat16 for k in base)
    expanded = comfy_io.dequantize(q, comfy_io.QuantLayer("int8_tensorwise", scale), torch.bfloat16)
    assert not torch.equal(first["blocks.0.weight"], expanded) and not torch.equal(first["blocks.1.bias"], base["blocks.1.bias"])
    with safe_open(str(job_dir / "output" / "t.safetensors"), "pt") as f:
        meta = f.metadata()
    assert meta["dainamo.network"] == "full" and meta["dainamo.base"] == "tiny_int8.safetensors"
    state = torch.load(job_dir / "state" / "last.pt", weights_only=False)
    assert state["weights"] == "t.safetensors" and "lora" not in state  # the weights are the checkpoint's

    # given more steps it carries on from the final model (kept under its step), not the profile's file
    loaded = []
    load = Tiny.load_transformer

    def spy(self, trainable=False):
        loaded.append((self.profile["diffusion_model"], trainable))
        return load(self, trainable)

    monkeypatch.setattr(Tiny, "load_transformer", spy)
    job["settings"]["steps"] = 9
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    assert loaded == [(str(job_dir / "output" / "t.safetensors"), True)]
    st = json.loads((job_dir / "status.json").read_text())
    assert [(c["step"], c["file"], c["final"], c.get("kept", False)) for c in st["checkpoints"]] == [
        (3, "t_000003.safetensors", False, False), (6, "t_000006.safetensors", False, True),
        (9, "t.safetensors", True, False)]
    kept = load_file(str(job_dir / "output" / "t_000006.safetensors"))
    assert all(torch.equal(kept[k], first[k]) for k in first)
    assert not torch.equal(load_file(str(job_dir / "output" / "t.safetensors"))["blocks.0.weight"], first["blocks.0.weight"])
    metrics = [json.loads(line) for line in (job_dir / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == list(range(1, 10))

    # its checkpoint gone, it can't carry on
    os.remove(job_dir / "output" / "t.safetensors")
    job["settings"]["steps"] = 12
    (job_dir / "job.json").write_text(json.dumps(job))
    with pytest.raises(RuntimeError, match="is gone"):
        trainer_mod.Trainer(str(job_dir)).run()


def _finetune_job(tmp_path, name: str, **settings) -> tuple:
    """A tiny finetune job, its model in bf16, and the profile's training adapter: a LoRA on ``blocks.0``."""
    from tests.test_trainer import _images

    torch.manual_seed(0)
    base = TinyDiT().to(torch.bfloat16).state_dict()
    source = str(tmp_path / "tiny.safetensors")
    if not os.path.exists(source):
        save_file(base, source)
        save_file({"diffusion_model.blocks.0.lora_A.weight": torch.randn(2, 3),
                   "diffusion_model.blocks.0.lora_B.weight": torch.randn(3, 2), "diffusion_model.blocks.0.alpha":
                   torch.tensor(2.0)}, str(tmp_path / "adapter.safetensors"))
        _images(str(tmp_path / "train"), [(200, 30, 30), (190, 40, 20), (60, 160, 60)])
    job_dir = tmp_path / name
    job_dir.mkdir()
    job = {"name": "t", "family": "tiny",
           "profile": {"diffusion_model": source, "training_adapter": str(tmp_path / "adapter.safetensors")},
           "datasets": [{"path": str(tmp_path / "train"), "trigger": "", "repeats": 1, "resolution": 64}],
           "settings": {"steps": 6, "resolution": 64, "seed": 1, "optimizer": "adafactor", "learning_rate": 0.01,
                        "save_every": 3, "sample_every": 0, "keep_last": 0, "finetune_enabled": True,
                        "grad_accum": 1, **settings}}
    (job_dir / "job.json").write_text(json.dumps(job))
    return job_dir, job, base


@gpu
def test_a_finetune_trains_its_streamed_blocks_and_the_training_adapter_applies(tmp_path, monkeypatch):
    """Offloaded (both blocks stream, so does the first), a finetune trains, saves and carries on exactly as it
    does on the GPU. The training adapter applies while it trains: the result differs from training without it,
    and isn't in the saved model."""
    from dainamo.engine import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "get_family", lambda name: _tiny_family())
    runs = {}
    for name, offload in (("gpu", 0), ("streamed", 100)):
        job_dir, job, base = _finetune_job(tmp_path, name, offload=offload)
        assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
        job["settings"]["steps"] = 9  # resumed from its final model, whose blocks stream again
        (job_dir / "job.json").write_text(json.dumps(job))
        assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
        runs[name] = {f: load_file(str(job_dir / "output" / f)) for f in sorted(os.listdir(job_dir / "output"))}
    assert list(runs["gpu"]) == list(runs["streamed"]) == ["t.safetensors", "t_000003.safetensors",
                                                         "t_000006.safetensors"]
    for f, tensors in runs["gpu"].items():
        assert set(tensors) == set(base)  # the model's own layers only: the adapter isn't saved
        for k, t in tensors.items():
            assert torch.equal(t, runs["streamed"][f][k]), (f, k)

    # without the adapter, the same job trains to other weights
    job_dir, job, _ = _finetune_job(tmp_path, "plain", offload=100)
    job["profile"]["training_adapter"] = ""
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    plain = load_file(str(job_dir / "output" / "t.safetensors"))
    assert not torch.equal(plain["blocks.0.weight"], runs["gpu"]["t_000006.safetensors"]["blocks.0.weight"])


def test_the_training_adapter_adds_its_change_while_on_and_leaves_the_layers_as_they_are(tmp_path):
    from dainamo.engine.finetune import TrainingAdapter

    torch.manual_seed(0)
    model = Net()
    names = set(model.state_dict())
    down, up = torch.randn(2, 16), torch.randn(16, 2)
    path = str(tmp_path / "a.safetensors")
    save_file({"diffusion_model.blocks.1.lora_A.weight": down, "diffusion_model.blocks.1.lora_B.weight": up,
               "diffusion_model.blocks.1.alpha": torch.tensor(1.0), "diffusion_model.norm.diff": torch.ones(16)}, path)
    adapter = TrainingAdapter(model, path, "cpu", torch.float32)
    assert len(adapter.handles) == 1 and set(model.state_dict()) == names  # a norm's change doesn't apply
    h = torch.randn(4, 16)
    assert torch.allclose(model.blocks[1](h), F.linear(h, model.blocks[1].weight, model.blocks[1].bias)
                          + h @ down.t() @ up.t() * 0.5, atol=1e-5)
    model.blocks[1](h).sum().backward()  # the layer's own weights still train through it
    assert model.blocks[1].weight.grad is not None
    adapter.enabled = False
    assert torch.equal(model.blocks[1](h), F.linear(h, model.blocks[1].weight, model.blocks[1].bias))
    with pytest.raises(ValueError, match="matched no layers"):
        save_file({"diffusion_model.nothing.lora_A.weight": down, "diffusion_model.nothing.lora_B.weight": up}, path)
        TrainingAdapter(model, path, "cpu")


# --- the server ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from dainamo.server.app import app

    with TestClient(app) as c:
        yield c


def test_finetune_settings_are_checked(client, tmp_path, monkeypatch):
    from dainamo.server.queue import scheduler

    monkeypatch.setattr(scheduler, "_launch", lambda: None)
    moss = client.post("/api/profiles", json={"name": "ft moss", "family": "moss_image", "data": {}}).json()["id"]
    krea = client.post("/api/profiles", json={"name": "ft krea", "family": "krea2", "data": {}}).json()["id"]
    qwen = client.post("/api/profiles", json={"name": "ft qwen", "family": "qwen_image_2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "ft", "path": str(tmp_path)}).json()["id"]

    def create(pid, **settings):
        return client.post("/api/jobs", json={"name": "ft", "profile_id": pid, "datasets": [{"id": did}],
                                              "settings": {"finetune_enabled": True, "optimizer": "adafactor", **settings}})

    assert "can't be finetuned" in create(qwen).json()["detail"]
    assert create(krea).status_code == 200
    assert "Adafactor or AdamW 8-bit" in create(moss, optimizer="adamw").json()["detail"]
    assert "slider" in create(moss, slider_enabled=True, slider_target="a", slider_positive="b",
                              slider_negative="c").json()["detail"]
    assert "Concept bleed" in create(moss, bleed_enabled=True, bleed_prompts=["a dog"]).json()["detail"]
    # offloaded blocks come back for backward through gradient checkpointing's recomputation
    assert "gradient checkpointing" in create(moss, offload=30, gradient_checkpointing=False).json()["detail"]
    r = create(moss, grad_accum=4, offload=30, picker_enabled=True, sweep_enabled=True)
    assert r.status_code == 200
    s = r.json()["config"]["settings"]
    assert (s["grad_accum"], s["offload"], s["picker_enabled"], s["sweep_enabled"]) == (1, 30, True, False)
    listed = next(j for j in client.get("/api/jobs").json() if j["id"] == r.json()["id"])
    assert listed["network"] == "full"
    jid = r.json()["id"]
    assert "whole models" in client.post(f"/api/jobs/{jid}/evaluate", json={"settings": {}}).json()["detail"]


def test_a_finetune_and_a_lora_dont_carry_on_from_each_other(client, tmp_path, monkeypatch):
    from dainamo.server import db
    from dainamo.server.queue import job_dir, scheduler

    monkeypatch.setattr(scheduler, "_launch", lambda: None)
    pid = client.post("/api/profiles", json={"name": "ft edit", "family": "moss_image", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "ft edit", "path": str(tmp_path)}).json()["id"]
    jid = client.post("/api/jobs", json={"name": "fe", "profile_id": pid, "datasets": [{"id": did}],
                                         "settings": {"finetune_enabled": True, "optimizer": "adafactor"}}).json()["id"]
    d = job_dir(jid)
    os.makedirs(os.path.join(d, "state"))
    open(os.path.join(d, "state", "last.pt"), "wb").close()
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"phase": "stopped", "step": 100}, f)
    db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (jid,))

    def edit(**settings):
        cfg = client.get(f"/api/jobs/{jid}/config").json()
        return client.put(f"/api/jobs/{jid}", json={"name": cfg["name"], "profile_id": pid, "gpu": "auto",
                                                    "datasets": [{"id": did}], "settings": {**cfg["settings"], **settings}})

    assert edit(rank=64, learning_rate=2e-5).status_code == 200  # a finetune has no rank to keep
    r = edit(finetune_enabled=False, optimizer="adafactor")
    assert r.status_code == 409 and "a LoRA or the whole model" in r.json()["detail"]


def test_a_finetune_loads_every_layer_in_bf16_as_the_projection_counts_it(tmp_path):
    from dainamo.engine import memory
    from dainamo.engine.fp8_linear import QuantLinear
    from tests.test_memory import D, N, TinyDriver, write_model

    path = write_model(tmp_path / "m.safetensors")  # its attn layers are int8
    cuda = torch.cuda.is_available()
    d = TinyDriver({"diffusion_model": path}, torch.device("cuda" if cuda else "cpu"), log=lambda *_: None)
    d.offload = 0.5  # half its blocks expanded in system RAM, and trained as they stream
    model = d.load_transformer(trainable=True)
    assert d.swapper is not None and d.swapper.trainable and d.swapper.swapped == list(range(N // 2, N))
    assert not any(isinstance(m, QuantLinear) for m in model.modules())
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(not p.is_cuda and p.is_pinned() == cuda for i in d.swapper.swapped for p in model.blocks[i].parameters())
    assert all(p.is_cuda == cuda for p in model.blocks[0].parameters())
    n = sum(p.numel() for p in model.parameters())
    p = memory.project(TinyDriver, {"diffusion_model": path}, {"finetune_enabled": True, "optimizer": "adafactor",
                                                               "resolution": 512, "offload": 50})
    assert p["adapter_params"] == n
    assert p["model_gb"] == pytest.approx(n * 2 / memory.GB)
    # a block stepped in backward is held until it's copied back to RAM: one more in flight than for a LoRA
    assert p["inflight_gb"] == pytest.approx((memory.IN_FLIGHT_BLOCKS + 1) * max(p["block_gb"]))
    train = dict(p["phases"][0]["parts"])
    vectors = N * 3 * D + D  # each block's norm weight and bias and mlp bias, and proj_in's bias
    matrices = N * 2 + 1
    assert train["Optimizer"] == pytest.approx(4 * (matrices * 2 * D + vectors) / memory.GB)
    assert train["Weight updates"] == pytest.approx(18 * D * D / memory.GB)
    assert any("quantized" in note for note in p["notes"])
    eight = memory.project(TinyDriver, {"diffusion_model": path}, {"finetune_enabled": True, "optimizer": "adamw8bit",
                                                                   "resolution": 512})
    assert dict(eight["phases"][0]["parts"])["Optimizer"] == pytest.approx(2 * n / memory.GB)


def test_the_sandbox_takes_a_finetuned_model_of_the_profiles_family(client, tmp_path):
    from tests.test_memory import write_model

    pid = client.post("/api/profiles", json={"name": "ft sbx", "family": "moss_image", "data": {}}).json()["id"]
    body = {"profile_id": pid, "workflow": "t2i", "prompt": "a cat", "steps": 4}
    from dainamo.server.routes import sandbox as routes

    other = write_model(tmp_path / "other.safetensors")  # not a MOSS-Image model
    with pytest.raises(Exception, match="Wrong file"):
        routes.check_sandbox(routes.SandboxIn(**body, model=other))
    with pytest.raises(Exception, match="not found"):
        routes.check_sandbox(routes.SandboxIn(**body, model=str(tmp_path / "gone.safetensors")))
    assert routes.check_sandbox(routes.SandboxIn(**body))[1]["model"] is None

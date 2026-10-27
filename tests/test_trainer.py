"""The trainer end to end with a tiny model standing in for a family's: training a job, continuing a finished
one, evaluating a finished job's saved checkpoints afterwards (Evaluate checkpoints), and saving an EMA of the
LoRA's weights.

    .venv/bin/python -m pytest tests/test_trainer.py -v
"""

import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from dainamo.engine import trainer as trainer_mod
from dainamo.engine.families.base import FamilyDriver, FamilyInfo
from dainamo.engine.lora import LoRANetwork

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer needs a GPU")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3)])

    def forward(self, x):
        return self.blocks[0](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TinyFamily(FamilyDriver):
    """Captions encode to a number, images to their 16x16-pixel averages, and the model is one Linear."""
    info = FamilyInfo(id="tiny", label="Tiny", media=["image"], profile_fields=[])
    lora_targets = ["blocks.*"]

    def load_text_encoder(self): ...
    def unload_text_encoder(self): ...
    def load_vae(self): ...
    def unload_vae(self): ...

    def encode_prompt(self, prompt):
        return torch.full((1, 4), float(len(prompt)))

    def encode_image(self, pixels):
        return F.avg_pool2d(pixels, 16)[0]

    def load_transformer(self):
        torch.manual_seed(0)
        return TinyModel().to(self.device)

    def sample_timesteps(self, n, hw, settings):
        return torch.rand(n)

    def predict(self, model, noisy, t, text):
        return model(noisy)


def _images(folder, colours):
    os.makedirs(folder, exist_ok=True)
    for i, c in enumerate(colours):
        Image.new("RGB", (64, 64), c).save(os.path.join(folder, f"{i}.png"))


def _finished_job(tmp_path, evaluate):
    """A completed job with checkpoints at steps 10 and 20, and what its training recorded."""
    job_dir = tmp_path / "job"
    (job_dir / "output").mkdir(parents=True)
    _images(str(tmp_path / "train"), [(200, 30, 30), (190, 40, 20), (210, 20, 40)])
    _images(str(tmp_path / "val"), [(60, 160, 60)])
    for step, scale in ((10, 0.05), (20, 0.5)):
        torch.manual_seed(step)
        net = LoRANetwork(TinyModel(), TinyFamily.lora_targets, 2, 2)
        for m in net.modules.values():
            nn.init.normal_(m.lora_B.weight, std=scale)
        net.save(str(job_dir / "output" / f"t_{step:06d}.safetensors"))
    (job_dir / "job.json").write_text(json.dumps({
        "name": "t", "family": "tiny", "profile": {},
        "datasets": [{"path": str(tmp_path / "train"), "trigger": "", "repeats": 1, "resolution": 64}],
        "settings": {"steps": 20, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1},
        "evaluate": evaluate}))
    training = {"phase": "completed", "message": "Done in 1.0 min", "step": 20, "total_steps": 20, "vram_gb": 5.0,
                "checkpoints": [{"step": 10, "file": "t_000010.safetensors", "final": False},
                                {"step": 20, "file": "t_000020.safetensors", "final": True}],
                "evals": [{"step": 20, "file": "t_000020.safetensors", "val_loss": 9.0}],
                "recommended": {"step": 20, "file": "t_000020.safetensors", "reason": "from training"}}
    (job_dir / "status.json").write_text(json.dumps(training))
    return str(job_dir), training


EVALUATE = {"settings": {"picker_enabled": True, "val_dataset": "val", "val_timesteps": 3, "sweep_enabled": False,
                         "bleed_enabled": False},
            "validation_datasets": None}


def test_checkpoints_are_scored_without_touching_the_jobs_own_results(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_mod, "get_family", lambda name: TinyFamily)
    job_dir, training = _finished_job(tmp_path, {**EVALUATE, "validation_datasets": [
        {"path": str(tmp_path / "val"), "trigger": "", "repeats": 1}]})
    t = trainer_mod.Trainer(job_dir)
    assert t.evaluating
    assert t.run_evaluation() == "completed"
    st = json.loads(open(os.path.join(job_dir, "status.json")).read())
    for k in ("phase", "message", "vram_gb", "checkpoints", "evals", "recommended"):
        assert st[k] == training[k], k  # the job stays as training left it
    ev = st["evaluation"]
    assert ev["state"] == "completed" and ev["vram_gb"] > 0 and ev["finished_at"] >= ev["started_at"]
    assert [e["step"] for e in sorted(ev["evals"], key=lambda e: e["step"])] == [0, 10, 20]
    losses = {e["step"]: e["val_loss"] for e in ev["evals"]}
    assert len(set(losses.values())) == 3  # the base model and each checkpoint score differently
    best = min((10, 20), key=losses.get)
    assert ev["recommended"]["file"] == f"t_{best:06d}.safetensors"

    # the same evaluation again gives the same numbers (fixed noise and timesteps)
    t = trainer_mod.Trainer(job_dir)
    t.run_evaluation()
    again = json.loads(open(os.path.join(job_dir, "status.json")).read())["evaluation"]
    assert {e["step"]: e["val_loss"] for e in again["evals"]} == pytest.approx(losses)


def test_a_stopped_evaluation_keeps_what_it_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_mod, "get_family", lambda name: TinyFamily)
    # the job held 1 image out in training; the evaluation uses the same one
    job_dir, training = _finished_job(tmp_path, {**EVALUATE, "settings": {**EVALUATE["settings"], "val_dataset": "",
                                                                         "val_holdout": 1}})
    t = trainer_mod.Trainer(job_dir)
    score = t.evaluate

    def score_then_stop(*args, **kwargs):  # Stop pressed while the base model is scored
        score(*args, **kwargs)
        with open(os.path.join(job_dir, "control.json"), "w") as f:
            json.dump({"action": "stop"}, f)

    t.evaluate = score_then_stop
    assert t.run_evaluation() == "stopped"
    st = json.loads(open(os.path.join(job_dir, "status.json")).read())
    assert st["phase"] == "completed" and st["message"] == training["message"]
    assert st["evaluation"]["state"] == "stopped" and [e["step"] for e in st["evaluation"]["evals"]] == [0]

    t = trainer_mod.Trainer(job_dir)
    t.end_evaluation("failed", "RuntimeError: out of memory")  # as run.py records a failure
    st = json.loads(open(os.path.join(job_dir, "status.json")).read())
    assert st["phase"] == "completed" and st["evaluation"]["state"] == "failed"
    assert st["evaluation"]["message"] == "RuntimeError: out of memory"


def test_a_job_trains_and_a_finished_one_continues(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    monkeypatch.setattr(trainer_mod, "get_family", lambda name: TinyFamily)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _images(str(tmp_path / "train"), [(200, 30, 30), (190, 40, 20), (60, 160, 60)])
    job = {"name": "t", "family": "tiny", "profile": {},
           "datasets": [{"path": str(tmp_path / "train"), "trigger": "", "repeats": 1, "resolution": 64,
                         "flip": True, "caption_dropout": 0.2}],
           "settings": {"steps": 6, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1, "optimizer": "adamw",
                        "learning_rate": 0.01, "save_every": 3, "sample_every": 0, "keep_last": 1}}
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    st = json.loads((job_dir / "status.json").read_text())
    assert [(c["step"], c["file"], c["final"]) for c in st["checkpoints"]] == [
        (3, "t_000003.safetensors", False), (6, "t.safetensors", True)]
    first = load_file(str(job_dir / "output" / "t.safetensors"))
    assert any(v.abs().sum() > 0 for k, v in first.items() if "lora_B" in k)  # it trained

    # given more steps, it carries on from step 6; the finished LoRA is kept under its step
    job["settings"].update(steps=12, learning_rate=0.005)  # and a learning rate edited meanwhile
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    st = json.loads((job_dir / "status.json").read_text())
    # keep_last 1 deleted step 3 but spared the kept one
    assert [(c["step"], c["file"], c["final"], c.get("kept", False)) for c in st["checkpoints"]] == [
        (6, "t_000006.safetensors", False, True), (9, "t_000009.safetensors", False, False),
        (12, "t.safetensors", True, False)]
    assert sorted(os.listdir(job_dir / "output")) == ["t.safetensors", "t_000006.safetensors", "t_000009.safetensors"]
    kept = load_file(str(job_dir / "output" / "t_000006.safetensors"))
    assert all(torch.equal(kept[k], first[k]) for k in first)
    assert not torch.equal(load_file(str(job_dir / "output" / "t.safetensors"))["diffusion_model.blocks.0.lora_B.weight"],
                           first["diffusion_model.blocks.0.lora_B.weight"])
    metrics = [json.loads(line) for line in (job_dir / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == list(range(1, 13))
    assert [m["lr"] for m in metrics] == [0.01] * 6 + [0.005] * 6  # not the one the saved state started with


def test_jobs_share_encodings_and_encode_only_what_changed(tmp_path, monkeypatch):
    encoded = {"images": 0, "captions": []}

    class Counting(TinyFamily):
        def encode_prompt(self, prompt):
            encoded["captions"].append(prompt)
            return super().encode_prompt(prompt)

        def encode_image(self, pixels):
            encoded["images"] += 1
            return super().encode_image(pixels)

    monkeypatch.setattr(trainer_mod, "get_family", lambda name: Counting)
    data = tmp_path / "train"
    _images(str(data), [(200, 30, 30), (190, 40, 20)])
    for i in range(2):
        (data / f"{i}.txt").write_text(f"photo {i}")

    def run(name):
        """A job on the dataset -> what it encoded."""
        encoded.update(images=0, captions=[])
        job_dir = tmp_path / name
        job_dir.mkdir()
        (job_dir / "job.json").write_text(json.dumps({
            "name": name, "family": "tiny", "profile": {}, "cache_dir": str(tmp_path / "cache"),
            "datasets": [{"path": str(data), "trigger": "", "repeats": 1, "resolution": 64}],
            "settings": {"steps": 2, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1, "optimizer": "adamw",
                         "learning_rate": 0.01, "sample_every": 0}}))
        assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
        return dict(encoded)

    first = run("a")
    assert first["images"] == 2 and {"photo 0", "photo 1"} <= set(first["captions"])
    assert run("b") == {"images": 0, "captions": []}  # another job on the same dataset encodes nothing
    (data / "1.txt").write_text("photo 1, red")  # a caption changes and an image is added
    Image.new("RGB", (64, 64), (10, 20, 200)).save(data / "2.png")
    (data / "2.txt").write_text("photo 2")
    third = run("c")
    assert third["images"] == 1 and sorted(third["captions"]) == ["photo 1, red", "photo 2"]
    # each job lists the cache files it uses, so clearing the cache spares them while it runs
    used = json.loads((tmp_path / "c" / "cache.json").read_text())["files"]
    assert sum(f.startswith("latents/") for f in used) == 3 and all(os.path.exists(tmp_path / "cache" / f) for f in used)
    assert not (tmp_path / "c" / "cache").exists()


def test_a_job_with_ema_saves_the_average_and_carries_it_on(tmp_path, monkeypatch):
    from safetensors import safe_open
    from safetensors.torch import load_file

    from dainamo.engine.ema import EMA

    monkeypatch.setattr(trainer_mod, "get_family", lambda name: TinyFamily)
    _images(str(tmp_path / "train"), [(200, 30, 30), (190, 40, 20), (60, 160, 60)])

    def run(name, steps, ema):
        job_dir = tmp_path / name
        job_dir.mkdir(exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps({
            "name": "t", "family": "tiny", "profile": {},
            "datasets": [{"path": str(tmp_path / "train"), "trigger": "", "repeats": 1, "resolution": 64}],
            "settings": {"steps": steps, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1, "optimizer": "adamw",
                         "learning_rate": 0.05, "save_every": 3, "sample_every": 0, "ema_enabled": ema,
                         "ema_decay": 0.9}}))
        assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
        state = torch.load(job_dir / "state" / "last.pt", weights_only=False)
        return job_dir, state

    plain_dir, plain = run("plain", 6, False)
    ema_dir, state = run("ema", 6, True)
    # training steps the live weights just the same; only what's saved is the average
    assert plain["lora"].keys() == state["lora"].keys()
    assert all(torch.allclose(plain["lora"][k], state["lora"][k], atol=1e-6) for k in plain["lora"])
    assert "ema" not in plain and state["ema"]["num_updates"] == 6

    net = LoRANetwork(TinyModel(), TinyFamily.lora_targets, 2, 2)
    net.load_state_dict(state["lora"])
    live = net.state_dict()
    ema = EMA(list(net.parameters()), 0.9)
    ema.load_state_dict(state["ema"])
    with ema.applied():
        averaged = net.state_dict()
    saved = load_file(str(ema_dir / "output" / "t.safetensors"))
    assert all(torch.equal(saved[k], averaged[k]) for k in averaged)  # the final file is the average
    assert not torch.equal(saved["diffusion_model.blocks.0.lora_B.weight"],
                           live["diffusion_model.blocks.0.lora_B.weight"])
    assert all(torch.equal(load_file(str(plain_dir / "output" / "t.safetensors"))[k], live[k]) for k in live)
    for f in ("t_000003.safetensors", "t.safetensors"):
        with safe_open(str(ema_dir / "output" / f), "pt") as fh:
            assert fh.metadata()["dainamo.ema"] == "0.9"
    with safe_open(str(plain_dir / "output" / "t.safetensors"), "pt") as fh:
        assert "dainamo.ema" not in fh.metadata()

    # continued, the average carries on counting; turned on by an edit, it starts from the weights then
    _, state = run("ema", 9, True)
    assert state["ema"]["num_updates"] == 9
    _, plain = run("plain", 9, True)
    assert plain["ema"]["num_updates"] == 3


def test_a_resumed_job_trains_on_as_if_it_never_stopped(tmp_path, monkeypatch):
    """Stopped at step 3 and resumed, a job draws the same batches, caption dropout and noise as one that ran
    straight through (it reseeded from the job's seed, replaying the first steps' draws); a stop before the saved
    state loads keeps what it made listed."""
    from safetensors.torch import load_file

    monkeypatch.setattr(trainer_mod, "get_family", lambda name: TinyFamily)
    _images(str(tmp_path / "train"), [(200, 30, 30), (190, 40, 20), (60, 160, 60), (20, 20, 200)])
    job = {"name": "t", "family": "tiny", "profile": {},
           "datasets": [{"path": str(tmp_path / "train"), "trigger": "", "repeats": 1, "resolution": 64,
                         "flip": True, "caption_dropout": 0.4}],
           "settings": {"steps": 6, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1, "optimizer": "adamw",
                        "learning_rate": 0.01, "save_every": 0, "sample_every": 0, "sample_at_start": False,
                        "sample_at_end": False}}

    def run(name, stop_at=None):
        job_dir = tmp_path / name
        job_dir.mkdir(exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(job))
        t = trainer_mod.Trainer(str(job_dir))
        if stop_at is not None:
            read = t.read_control
            t.read_control = lambda: {"action": "stop"} if t.status["step"] >= stop_at else read()
        return t.run(), job_dir

    assert run("straight")[0] == "completed"
    assert run("resumed", stop_at=3)[0] == "stopped"
    # stopped again before its saved state loads: the checkpoint it made stays listed
    job_dir = tmp_path / "resumed"
    t = trainer_mod.Trainer(str(job_dir))
    t.write_status(force=True)
    assert [c["step"] for c in json.loads((job_dir / "status.json").read_text())["checkpoints"]] == [3]
    assert run("resumed")[0] == "completed"

    def losses(name):
        return [json.loads(line)["loss"] for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]

    assert losses("resumed") == pytest.approx(losses("straight"))
    a, b = (load_file(str(tmp_path / n / "output" / "t.safetensors")) for n in ("straight", "resumed"))
    assert all(torch.allclose(a[k], b[k]) for k in a)


class SamplingFamily(TinyFamily):
    """The tiny family with a sampler: the latents' noise, stepped by the prediction, decoded by upscaling."""

    def prepare_sampling(self, model, cond, uncond, width, height, cfg, seed, images=None):
        g = torch.Generator().manual_seed(int(seed))
        x = torch.randn(1, 3, height // 16, width // 16, generator=g).to(self.device)
        return x, lambda x, s: self.predict(model, x, torch.full((1,), s, device=self.device), [cond]) * 0.1

    def decode_latents(self, latents):
        return F.interpolate(latents.float(), scale_factor=16).clamp(-1, 1)


def test_a_job_resumed_after_a_crash_keeps_its_checkpoints_samples_and_its_time(tmp_path, monkeypatch, capsys):
    """A checkpoint's samples are made after its state is saved: a job that dies then and resumes from that state
    still lists them; the time it took counts every run; a sample prompt that's a training caption is noted."""
    monkeypatch.setattr(trainer_mod, "get_family", lambda name: SamplingFamily)
    data = tmp_path / "train"
    _images(str(data), [(200, 30, 30), (190, 40, 20)])
    for i in range(2):
        (data / f"{i}.txt").write_text(f"photo {i}")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({
        "name": "t", "family": "tiny", "profile": {},
        "datasets": [{"path": str(data), "trigger": "", "repeats": 1, "resolution": 64}],
        "settings": {"steps": 6, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1, "optimizer": "adamw",
                     "learning_rate": 0.01, "save_every": 3, "sample_every": 3, "sample_at_start": False,
                     "sample_steps": 2, "sample_width": 64, "sample_height": 64, "likeness_enabled": False,
                     "sample_prompts": [{"prompt": "Photo 0."}, {"prompt": "a photo nobody took"}]}}))

    t = trainer_mod.Trainer(str(job_dir))
    read = t.read_control

    def crash():
        if t.status["step"] >= 4:
            raise RuntimeError("out of memory")
        return read()
    t.read_control = crash
    with pytest.raises(RuntimeError):
        t.run()
    assert [s["step"] for s in json.loads((job_dir / "status.json").read_text())["samples"]] == [3, 3]
    out = capsys.readouterr().out
    assert "Note: sample prompt 1 is the caption of 0.png" in out and "sample prompt 2" not in out

    # the state records the time to reach it; say the first run took an hour
    state_file = job_dir / "state" / "last.pt"
    state = torch.load(state_file, weights_only=False)
    assert "run_seconds" in state["status"]
    state["status"]["run_seconds"] = 3600.0
    torch.save(state, state_file)

    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    status = json.loads((job_dir / "status.json").read_text())
    assert [s["step"] for s in status["samples"]] == [3, 3, 6, 6]
    assert all(os.path.exists(job_dir / "samples" / s["file"]) for s in status["samples"])
    assert "Done in 1h 00m (" in capsys.readouterr().out and status["run_seconds"] >= 3600

"""The trainer end to end with a tiny model standing in for a family's: training a job, continuing a finished
one, and evaluating a finished job's saved checkpoints afterwards (Evaluate checkpoints).

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
    job["settings"]["steps"] = 12
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
    steps = [json.loads(line)["step"] for line in (job_dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == list(range(1, 13))

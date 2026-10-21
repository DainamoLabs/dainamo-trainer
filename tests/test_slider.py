"""Concept sliders (``engine.slider``): the step's targets and LoRA states, and a slider job end to end with
a tiny model, training on images the base model renders because the job has no dataset.

    .venv/bin/python -m pytest tests/test_slider.py -v
"""

import json
import os

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from dainamo.engine import trainer as trainer_mod
from dainamo.engine.lora import LoRALinear, LoRANetwork
from dainamo.engine.slider import Slider, slider_loss

from test_trainer import TinyFamily  # the trainer's tests' stand-in family: one Linear, 16x16-pixel latents

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer needs a GPU")

PROMPTS = {"target": "a person", "positive": "a person, smiling broadly", "negative": "a person, frowning"}


class SliderFamily(TinyFamily):
    """A prompt scales the noisy input into the prediction by its length, so the positive prompt's velocity
    differs from the negative one's by a multiple of the input, which a LoRA on the Linear can learn.
    Samples decode by upscaling the 16x16-pixel latents. ``backward_strengths`` records the LoRA's strength
    whenever backward reaches a prediction (when checkpointed blocks would run their forward again), and
    ``trained_on`` the length of the prompt each trained prediction had."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backward_strengths, self.trained_on = [], []

    def predict(self, model, noisy, t, text):
        k = torch.stack([c.flatten()[0] for c in text]).to(noisy) * 0.01
        out = model(noisy) + k.view(-1, 1, 1, 1) * noisy
        if out.requires_grad:
            lora = next(m for m in model.modules() if isinstance(m, LoRALinear))
            out.register_hook(lambda g: self.backward_strengths.append(lora.multiplier))
            self.trained_on.append(int(text[0].flatten()[0]))
        return out

    def prepare_sampling(self, model, cond, uncond, width, height, cfg, seed, images=None):
        g = torch.Generator().manual_seed(int(seed))
        x = torch.randn(1, 3, height // 16, width // 16, generator=g).to(self.device)

        def velocity(x, s):
            return self.predict(model, x, torch.full((1,), s, device=self.device), [cond]) * 0.1
        return x, velocity

    def decode_latents(self, latents):
        return F.interpolate(latents.float(), scale_factor=16).clamp(-1, 1)


def test_the_step_trains_the_lora_at_its_direction_against_the_base_models_guidance():
    d = SliderFamily({}, torch.device("cuda"))
    model = d.load_transformer()
    net = LoRANetwork(model, d.lora_targets, 2, 2)
    net.to(d.device)
    for m in net.modules.values():
        torch.nn.init.normal_(m.lora_B.weight, std=0.1)  # a LoRA that already does something
    conds = {k: d.encode_prompt(p) for k, p in PROMPTS.items()}
    payloads = [{"latent": torch.randn(3, 4, 4)} for _ in range(2)]
    seen = []
    velocity = d.slider_velocity

    def spy(model_, batch, cond):
        lora = next(iter(net.modules.values()))
        seen.append((float(cond.flatten()[0]), lora.enabled, lora.multiplier, torch.is_grad_enabled()))
        return velocity(model_, batch, cond)
    d.slider_velocity = spy

    for direction in (1.0, -1.0):
        seen.clear()
        net.set_multiplier(direction)
        torch.manual_seed(7)
        loss = slider_loss(d, model, net, payloads, conds, {}, direction, 3.0)
        lens = {k: float(len(p)) for k, p in PROMPTS.items()}
        # the base model (LoRA off, no gradients) under each prompt, then the LoRA at the direction with the target's
        assert seen == [(lens["positive"], False, direction, False), (lens["negative"], False, direction, False),
                        (lens["target"], False, direction, False), (lens["target"], True, direction, True)]

        torch.manual_seed(7)  # the same noise and timesteps, worked out by hand
        batch = d.slider_batch(payloads, {})
        net.set_enabled(False)
        with torch.no_grad():
            pos, neg, neu = (velocity(model, batch, conds[k]) for k in ("positive", "negative", "target"))
        net.set_enabled(True)
        with torch.no_grad():
            pred = velocity(model, batch, conds["target"])
        assert loss.item() == pytest.approx(F.mse_loss(pred, neu + direction * 3.0 * (pos - neg)).item(), rel=1e-5)

        # keeping the layout: an image noised above the cutoff is trained toward the base model's own prediction
        torch.manual_seed(7)
        t = d.slider_batch(payloads, {})["t"]
        cutoff = float(t.mean())  # one image either side
        assert (t > cutoff).sum() == 1
        torch.manual_seed(7)
        loss = slider_loss(d, model, net, payloads, conds, {}, direction, 3.0, layout_sigma=cutoff)
        goal = neu + direction * 3.0 * (pos - neg) * (t <= cutoff).view(-1, 1, 1, 1)
        assert loss.item() == pytest.approx(F.mse_loss(pred, goal).item(), rel=1e-5)
        # and a batch noised only above it doesn't need the positive and negative predictions
        seen.clear()
        torch.manual_seed(7)
        loss = slider_loss(d, model, net, payloads, conds, {}, direction, 3.0, layout_sigma=0.0)
        assert [c for c, *_ in seen] == [lens["target"], lens["target"]]
        assert loss.item() == pytest.approx(F.mse_loss(pred, neu).item(), rel=1e-5)
    net.set_multiplier(1.0)


def test_a_slider_renders_its_images_and_learns_both_directions(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_mod, "get_family", lambda name: SliderFamily)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {"name": "s", "family": "tiny", "profile": {"sample_steps": 2}, "datasets": [],
           "settings": {"steps": 40, "rank": 2, "alpha": 2, "resolution": 64, "seed": 1, "optimizer": "adamw",
                        "learning_rate": 0.02, "save_every": 0, "sample_every": 0, "sample_at_start": False,
                        "sample_prompts": [{"prompt": "a person at the beach"}], "slider_enabled": True,
                        "slider_target": PROMPTS["target"], "slider_positive": PROMPTS["positive"],
                        "slider_negative": PROMPTS["negative"], "slider_strength": 3.0,
                        "slider_sample_strengths": [-1.0, 1.0], "slider_images": 3,
                        "slider_preserve": ["outdoors"], "slider_keep_layout": True, "slider_layout_sigma": 0.9}}
    (job_dir / "job.json").write_text(json.dumps(job))
    t = trainer_mod.Trainer(str(job_dir))
    assert t.run() == "completed"
    # each step trained the LoRA at +1 or -1, and it was still at that strength during backward
    assert len(t.driver.backward_strengths) == 40 and set(t.driver.backward_strengths) == {1.0, -1.0}
    # with the target as given, or with the preserved attribute added
    assert set(t.driver.trained_on) == {len(PROMPTS["target"]), len(PROMPTS["target"] + ", outdoors")}
    rendered = sorted(os.listdir(job_dir / "slider_images"))
    assert rendered == ["000.png", "001.png", "002.png"]
    st = json.loads((job_dir / "status.json").read_text())
    # the sample card at each slider strength, with the same seed
    assert [(s["index"], s["lora_scale"], s["seed"]) for s in st["samples"]] == [(0, -1.0, 42), (1, 1.0, 42)]
    with safe_open(str(job_dir / "output" / "s.safetensors"), "pt") as f:
        meta = f.metadata()
    assert meta["dainamo.slider.positive"] == PROMPTS["positive"] and meta["dainamo.slider.strength"] == "3"
    assert (meta["dainamo.slider.preserve"], meta["dainamo.slider.layout_sigma"]) == ("outdoors", "0.9")

    # +1 moves the target's prediction the way positive - negative points, -1 the other way
    d = t.driver
    model = d.load_transformer()
    net = LoRANetwork(model, d.lora_targets, 2, 2)
    net.to(d.device)
    net.load(str(job_dir / "output" / "s.safetensors"))
    conds = {k: d.encode_prompt(p) for k, p in PROMPTS.items()}
    batch = {"noisy": torch.randn(4, 3, 4, 4, device=d.device), "t": torch.rand(4, device=d.device)}
    with torch.no_grad():
        net.set_enabled(False)
        pos, neg, base = (d.slider_velocity(model, batch, conds[k]) for k in ("positive", "negative", "target"))
        net.set_enabled(True)
        moved = {}
        for direction in (1.0, -1.0):
            net.set_multiplier(direction)
            moved[direction] = d.slider_velocity(model, batch, conds["target"]) - base
    assert (moved[1.0] * (pos - neg)).sum() > 0 > (moved[-1.0] * (pos - neg)).sum()

    # trained on, it keeps the images it rendered
    stamps = [os.stat(job_dir / "slider_images" / f).st_mtime_ns for f in rendered]
    job["settings"]["steps"] = 44
    (job_dir / "job.json").write_text(json.dumps(job))
    assert trainer_mod.Trainer(str(job_dir)).run() == "completed"
    assert [os.stat(job_dir / "slider_images" / f).st_mtime_ns for f in rendered] == stamps


def test_slider_settings_round_trip():
    s = Slider.from_settings({"slider_enabled": True, "slider_target": "t", "slider_positive": "p",
                              "slider_negative": "n", "slider_sample_strengths": [-2, 2]})
    assert (s.strength, s.sample_strengths, s.images, s.preserve, s.layout_sigma) == (3.0, [-2.0, 2.0], 16, [], None)
    assert Slider.from_settings({"slider_enabled": False}) is None
    s = Slider.from_settings({"slider_enabled": True, "slider_target": "a cat", "slider_positive": "a fat cat",
                              "slider_negative": "a thin cat", "slider_preserve": ["indoors", "black"],
                              "slider_keep_layout": True, "slider_layout_sigma": 0.8})
    assert s.layout_sigma == 0.8 and s.prompt_sets() == [
        {"target": "a cat", "positive": "a fat cat", "negative": "a thin cat"},
        {"target": "a cat, indoors", "positive": "a fat cat, indoors", "negative": "a thin cat, indoors"},
        {"target": "a cat, black", "positive": "a fat cat, black", "negative": "a thin cat, black"}]

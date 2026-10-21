"""Concept sliders: a LoRA whose strength moves one attribute of a subject, either way.

A slider job names a target prompt (the subject, e.g. "a photo of a person"), a positive and a negative
prompt (the subject at either end: "... smiling broadly" / "... frowning") and a strength. Each step
noises a training image, and the base model (the LoRA switched off) predicts its velocity under all
three prompts. The LoRA, applied at +1 or -1 (one of the two at random), is then trained so that with
the target prompt the model predicts

    target ± strength * (positive - negative)

At +1 it pushes the target toward the positive prompt, at -1 toward the negative one, and strengths in
between scale it. This is the objective of Concept Sliders (Gandikota et al., 2023, arXiv:2311.12092),
on flow-matching velocities instead of noise predictions.

Two optional ways to keep the rest of the image as it is:
  * preserved attributes (the paper's): each step adds one of them, or none, to all three prompts
    ("..., in a park"). With the setting or clothes named on both sides, the difference between the
    positive and negative prompts no longer carries whatever tends to come with the attribute.
  * keeping the layout: above a noise level, where the layout, clothes and background are decided, the
    LoRA is trained to change nothing (the target is the base model's own prediction).

Captions aren't used. The training images are a dataset of the subject or, without one, images of the
target prompt that the base model renders before training (kept in ``<job>/slider_images``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

FOLDER = "slider_images"


@dataclass
class Slider:
    target: str
    positive: str
    negative: str
    strength: float = 3.0
    sample_strengths: list[float] = field(default_factory=lambda: [-1.0, 1.0])  # each sample card renders at these
    images: int = 16  # rendered when the job has no datasets
    preserve: list[str] = field(default_factory=list)  # attributes added to all three prompts, one a step
    layout_sigma: float | None = None  # above this noise level the LoRA learns to change nothing (None: off)

    @classmethod
    def from_settings(cls, s: dict) -> Slider | None:
        if not s.get("slider_enabled"):
            return None
        return cls(str(s["slider_target"]), str(s["slider_positive"]), str(s["slider_negative"]),
                   float(s.get("slider_strength", 3.0)),
                   [float(x) for x in s.get("slider_sample_strengths") or [-1, 1]], int(s.get("slider_images", 16)),
                   [str(p) for p in s.get("slider_preserve") or []],
                   float(s.get("slider_layout_sigma", 0.9)) if s.get("slider_keep_layout") else None)

    def prompt_sets(self) -> list[dict[str, str]]:
        """{target, positive, negative} as given, then with each preserved attribute added to all three.
        Each step trains with one of them."""
        sets = [{"target": self.target, "positive": self.positive, "negative": self.negative}]
        for p in self.preserve:
            sets.append({k: f"{v}, {p}" for k, v in sets[0].items()})
        return sets

    def folder(self, job_dir: str) -> str:
        return os.path.join(job_dir, FOLDER)

    def image_path(self, job_dir: str, i: int) -> str:
        return os.path.join(self.folder(job_dir), f"{i:03d}.png")

    def rendered(self, job_dir: str) -> bool:
        return all(os.path.exists(self.image_path(job_dir, i)) for i in range(self.images))

    def metadata(self) -> dict[str, str]:
        """Saved in the LoRA file, so a slider says what it does."""
        meta = {"dainamo.slider.target": self.target, "dainamo.slider.positive": self.positive,
                "dainamo.slider.negative": self.negative, "dainamo.slider.strength": f"{self.strength:g}"}
        if self.preserve:
            meta["dainamo.slider.preserve"] = " | ".join(self.preserve)
        if self.layout_sigma is not None:
            meta["dainamo.slider.layout_sigma"] = f"{self.layout_sigma:g}"
        return meta


def slider_loss(driver, model, network, payloads: list, conds: dict, settings: dict, direction: float,
                strength: float, layout_sigma: float | None = None) -> torch.Tensor:
    """One slider step's loss with the LoRA applied at ``direction`` (+1 or -1), which the caller sets
    and keeps set until backward is done: checkpointed blocks run their forward again then. ``conds``:
    the cached text conditions of one of ``Slider.prompt_sets``. Images noised above ``layout_sigma``
    are trained toward the base model's prediction instead; a batch of only those skips the positive
    and negative predictions."""
    batch = driver.slider_batch(payloads, settings)
    guided = batch["t"] <= layout_sigma if layout_sigma is not None else torch.ones_like(batch["t"], dtype=torch.bool)
    with torch.no_grad():
        network.set_enabled(False)
        try:
            names = ("positive", "negative", "target") if guided.any() else ("target",)
            preds = {k: driver.slider_velocity(model, batch, conds[k]) for k in names}
        finally:
            network.set_enabled(True)
    goal = preds["target"]
    if guided.any():
        push = direction * strength * (preds["positive"] - preds["negative"])
        goal = goal + push * guided.view(-1, *([1] * (goal.dim() - 1)))
    return F.mse_loss(driver.slider_velocity(model, batch, conds["target"]), goal)

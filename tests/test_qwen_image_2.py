"""Qwen-Image 2.1: DiT parity with the release's diffusers model, the sampling schedule, reference
sizing, and the driver's text-to-image / edit plumbing.

``data/qwen_image_2_dit_ref.pt`` holds outputs of ai-toolkit's vendored diffusers
``QwenImage21Transformer2DModel`` on a small config whose weights and inputs are rebuilt here from
fixed seeds (sorted-key order), so the fixture is only the outputs.

    .venv/bin/python -m pytest tests/test_qwen_image_2.py -v
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from dainamo.engine.families.qwen_image_2 import driver as Q
from dainamo.engine.families.qwen_image_2.driver import QwenImage2Driver
from dainamo.engine.families.qwen_image_2.model import QwenImage21Config, QwenImage21DiT

CPU = torch.device("cpu")
REF = os.path.join(os.path.dirname(__file__), "data", "qwen_image_2_dit_ref.pt")
SMALL = dict(in_channels=16, out_channels=16, num_layers=2, attention_head_dim=32, num_attention_heads=4,
             context_in_dim=64, mlp_ratio=3, axes_dims_rope=(8, 12, 12))


def small_model() -> QwenImage21DiT:
    m = QwenImage21DiT(QwenImage21Config(**SMALL)).eval()
    g = torch.Generator().manual_seed(1234)
    m.load_state_dict({k: torch.randn(v.shape, generator=g) * 0.05 for k, v in sorted(m.state_dict().items())})
    return m


def test_dit_matches_diffusers_reference():
    """Text-to-image, an edit with two references, two adjacent references (separate bidirectional
    blocks), and a padded batch: all exact against the release's model."""
    expected = torch.load(REF)
    m = small_model()
    g = torch.Generator().manual_seed(1)
    got = {}

    def run(name, b, text_len, slots_layout, ref_shapes, target_hw, mask=None):
        img_mask = torch.zeros(b, text_len, dtype=torch.bool)
        pos = 0
        for t, s in slots_layout:
            pos += t
            img_mask[:, pos:pos + s] = True
            pos += s
        h, w = target_hw
        n_ref = sum(rh * rw for rh, rw in ref_shapes)
        x = torch.randn(b, n_ref + h * w, 16, generator=g)
        ctx = torch.randn(b, text_len, 64, generator=g)
        t = torch.rand(b, generator=g)
        full = torch.cat([img_mask, torch.ones(b, h * w // 4, dtype=torch.bool)], 1)
        shapes = [*[(1, rh, rw) for rh, rw in ref_shapes], (1, h, w)]
        with torch.no_grad():
            got[name] = m(x, ctx, t, shapes, full, mask)

    run("t2i", 1, 9, [], [], (4, 6))
    run("edit", 1, 3 + 6 + 2 + 1 + 4, [(3, 6), (2, 1)], [(4, 6), (2, 2)], (4, 4))
    run("edit_adjacent", 1, 3 + 6 + 1 + 4, [(3, 6), (0, 1)], [(4, 6), (2, 2)], (4, 4))
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 9:] = False
    run("t2i_padded", 2, 12, [], [], (4, 4), mask=mask)
    for name, y in expected.items():
        assert torch.allclose(got[name], y, atol=1e-6, rtol=1e-5), name


def test_config_from_checkpoint_shapes():
    m = QwenImage21DiT(QwenImage21Config(**SMALL))
    shapes = {k: list(v.shape) for k, v in m.state_dict().items()}
    cfg = QwenImage21Config.from_shapes(shapes)
    assert (cfg.num_layers, cfg.num_attention_heads, cfg.attention_head_dim, cfg.context_in_dim, cfg.mlp_ratio,
            cfg.in_channels) == (2, 4, 32, 64, 3, 16)


def test_sampling_sigmas_match_the_release_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9, num_train_timesteps=1000,
        shift=1.0, shift_terminal=0.02, time_shift_type="exponential", use_dynamic_shifting=True)
    for steps, seq in ((40, 64 * 64), (20, 84 * 48), (8, 32 * 32)):
        sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=Q.mu_for(seq))
        assert np.allclose(Q.sampling_sigmas(steps, seq), sched.sigmas.numpy(), atol=1e-6)
    assert Q.sampling_sigmas(1, 4096) == [1.0, 0.0]  # one step: noise to image (it was NaN)
    turbo = Q.sampling_sigmas(40, 4096, turbo=True)
    assert len(turbo) == 9 and turbo[0] == 1.0 and turbo[-1] == 0.0


def test_reference_sizing_keeps_aspect_on_the_vision_grid():
    for (w, h), target in (((1920, 1080), 1344 * 768), ((500, 800), 1024 * 1024), ((300, 200), 512 * 512),
                           ((64, 64), 128 * 128)):
        rw, rh = Q.reference_size(w, h, target)
        assert rw % 32 == 0 and rh % 32 == 0
        assert rw * rh >= Q.MIN_REF_PIXELS * 0.85  # never below what the processor would upsample
        assert abs(rw / rh - w / h) / (w / h) < 0.1
        assert abs(rw * rh - max(target, Q.MIN_REF_PIXELS)) / max(target, Q.MIN_REF_PIXELS) < 0.1
    img = Image.new("RGBA", (300, 200), (255, 0, 0, 0))  # fully transparent red
    x = Q.prepare_reference(img, 256 * 256)
    assert x.shape[1] == 4 and Q.vision_image(x).getpixel((5, 5)) == (255, 255, 255)  # composited over white


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden, embeds, t, img_shapes, img_mask, mask):
        self.calls.append(dict(hidden=hidden.shape, embeds=embeds.shape, img_shapes=img_shapes,
                               slots=int(img_mask[0].sum()), mask=mask))
        return torch.zeros(hidden.shape[0], img_mask.shape[1] + 3 * int(img_mask[0].sum()), hidden.shape[2])


def cond(n_text, slots=0):
    s = torch.zeros(n_text + slots, dtype=torch.bool)
    s[2:2 + slots] = True
    return {"embeds": torch.randn(n_text + slots, 4096).bfloat16(), "slots": s}


def test_edit_items_carry_references_and_dropped_captions_do_not():
    d = QwenImage2Driver({}, CPU)
    m = Recorder()
    lat = torch.randn(64, 4, 6)
    edit = {"latent": lat, "refs": [torch.randn(64, 4, 6), torch.randn(64, 2, 2)]}
    plain = {"latent": lat}
    ref_cond = {"ref": cond(9, slots=(24 + 4) // 4)}
    d.training_loss(m, [edit, plain, edit], [ref_cond, {"text": cond(5)}, {"text": cond(1)}], {}, sigma=0.5)
    # two layouts: the edit (references + target) and the two text-to-image rows (padded together)
    assert len(m.calls) == 2
    e, t = m.calls
    assert e["hidden"][1] == 24 + 4 + 24 and e["img_shapes"] == [(1, 4, 6), (1, 2, 2), (1, 4, 6)] and e["slots"] == 7 + 6
    assert t["hidden"][:2] == (2, 24) and t["img_shapes"] == [(1, 4, 6)] and t["mask"] is not None


def test_slot_count_mismatch_is_caught():
    d = QwenImage2Driver({}, CPU)
    bad = {"ref": cond(9, slots=3)}  # references need 7 slots
    with pytest.raises(ValueError, match="slots"):
        d.training_loss(Recorder(), [{"latent": torch.randn(64, 4, 6), "refs": [torch.randn(64, 4, 6),
                                                                                  torch.randn(64, 2, 2)]}],
                        [bad], {}, sigma=0.5)


class Oracle(torch.nn.Module):
    """Predicts the exact velocity (noise - clean) toward fixed clean latents, reading sigma = t."""

    def __init__(self, x0):
        super().__init__()
        self.x0 = x0

    def forward(self, hidden, embeds, t, img_shapes, img_mask, mask):
        n = self.x0.shape[1]
        x = hidden[:, -n:].float()
        v = (x - self.x0) / t.float().view(-1, 1, 1)  # x = (1-s) x0 + s n  =>  n - x0 = (x - x0) / s
        return torch.cat([torch.zeros(1, img_mask.shape[1] + 3 * int(img_mask[0].sum()) - n, x.shape[2]), v], 1)


def test_sampler_lands_on_the_clean_latent_given_the_true_velocity():
    d = QwenImage2Driver({"sample_steps": 12}, CPU)
    d.dtype = torch.float32
    x0 = torch.randn(1, 4 * 6, 64)
    out = d.sample(Oracle(x0), cond(5), None, 96, 64, 12, 1, 3)
    assert torch.allclose(out.reshape(1, 64, -1).transpose(1, 2), x0, atol=1e-4)


def test_masked_loss_noise_range_and_weighting():
    """Zero predictions: the loss is the target's power, over the masked half with masked loss on; loss weighting
    scales it at a drawn noise level, not a pinned one."""
    import math

    import torch.nn.functional as F

    d = QwenImage2Driver({}, CPU)
    lat = torch.randn(64, 4, 6)
    mask = torch.zeros(4, 6, dtype=torch.float16)
    mask[:, :3] = 1
    payload = {"latent": lat, "mask": mask}

    def loss(settings, **kw):
        return d.training_loss(Recorder(), [payload], [{"text": cond(5)}], settings,
                               generator=torch.Generator().manual_seed(0), **kw)

    target = torch.randn((1, 64, 4, 6), generator=torch.Generator().manual_seed(0)) - lat[None]
    assert torch.equal(loss({}, sigma=0.5), F.mse_loss(torch.zeros_like(target), target))
    masked = {"masked_loss": True, "mask_min_weight": 0.0}
    assert torch.allclose(loss(masked, sigma=0.5), (target[..., :3] ** 2).mean())
    assert torch.equal(loss({**masked, "loss_weighting": "min_snr"}, sigma=0.5), loss(masked, sigma=0.5))
    d.sample_timesteps = lambda b, hw, settings: torch.full((b,), 0.5)
    assert torch.allclose(loss({"loss_weighting": "cosmap"}), 4 / math.pi * loss({}))
    # the noise range moves the drawn level: 0.5 of the family's range lands mid-way in 0.2..0.4
    seen = []
    d._forward = lambda model, noisy, t, conds, refs: seen.append(t.clone()) or torch.zeros_like(noisy)
    loss({"timestep_min": 0.2, "timestep_max": 0.4})
    assert torch.allclose(seen[0], torch.tensor([0.3]))

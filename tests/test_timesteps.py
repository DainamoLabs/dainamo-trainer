"""The noise range training steps are drawn from (timestep_min / timestep_max) and the loss weighting by noise level
(loss_weighting: none, min_snr, cosmap), shared by every family through ``FamilyDriver``.

    .venv/bin/python -m pytest tests/test_timesteps.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

from dainamo.engine.families import FAMILIES
from dainamo.engine.families.base import FamilyDriver, FamilyInfo

CPU = torch.device("cpu")


class Recording(FamilyDriver):
    """The default flow-matching loss with a 1x1 conv for the model, recording the noise levels it's given."""
    info = FamilyInfo(id="rec", label="Rec", media=["image"], profile_fields=[])

    def __init__(self):
        super().__init__({}, CPU)
        self.seen = []

    def sample_timesteps(self, n, hw, settings):
        return torch.rand(n)

    def predict(self, model, noisy, t, text):
        self.seen.append(t.clone())
        return model(noisy)


def test_no_weighting_is_ones():
    d = Recording()
    t = torch.rand(7)
    assert torch.equal(d.loss_weights(t, {}), torch.ones(7))
    assert torch.equal(d.loss_weights(t, {"loss_weighting": "none"}), torch.ones(7))
    assert d.step_weights(t, {}) is None


def test_cosmap():
    d = Recording()
    w = d.loss_weights(torch.tensor([0.5, 0.1, 0.9]), {"loss_weighting": "cosmap"})
    assert w[0].item() == pytest.approx(4 / math.pi)
    assert w[1].item() == pytest.approx(w[2].item()) and w[1] < w[0]  # symmetric, highest mid-way


def test_min_snr():
    d = Recording()
    t = torch.tensor([0.05, 0.2, 0.5, 0.7, 0.95])
    for gamma in (5.0, 1.0):
        w = d.loss_weights(t, {"loss_weighting": "min_snr", "min_snr_gamma": gamma})
        for ti, wi in zip(t.tolist(), w.tolist()):
            snr = ((1 - ti) / ti) ** 2
            # the velocity error is the clean-image error over t^2; Min-SNR weights that error by min(snr, gamma)
            assert wi == pytest.approx(min(snr, gamma) * ti * ti, rel=1e-5)
    assert d.loss_weights(torch.tensor([0.5]), {"loss_weighting": "min_snr", "min_snr_gamma": 5})[0].item() == 0.25
    # the clean end weighs nothing, without NaN
    assert d.loss_weights(torch.tensor([0.0]), {"loss_weighting": "min_snr"}).tolist() == [0.0]
    with pytest.raises(ValueError, match="loss_weighting"):
        d.loss_weights(t, {"loss_weighting": "p2"})


def test_weighting_scales_each_items_loss_and_is_skipped_at_a_pinned_sigma():
    d = Recording()
    pred, target = torch.randn(3, 4, 6, 8), torch.randn(3, 4, 6, 8)
    t = torch.tensor([0.2, 0.5, 0.8])
    s = {"loss_weighting": "min_snr", "min_snr_gamma": 5}
    w = d.loss_weights(t, s)
    per = ((pred - target) ** 2).flatten(1).mean(1)
    assert torch.allclose(d.weighted_loss(pred, target, t, s, [{}] * 3), (per * w).mean())
    # validation pins sigma: the weights would only rescale its score, so they're left out
    assert d.step_weights(t, s, sigma_pinned=True) is None
    assert torch.equal(d.weighted_loss(pred, target, t, s, [{}] * 3, sigma_pinned=True), F.mse_loss(pred, target))
    model = torch.nn.Conv2d(3, 3, 1)
    lat = {"latent": torch.randn(3, 4, 6)}

    def val(settings):
        return d.training_loss(model, [lat], [{"text": None}], settings, sigma=0.3,
                               generator=torch.Generator().manual_seed(0)).item()
    assert val(s) == val({}) == val({"loss_weighting": "cosmap"})
    # training at a drawn sigma is weighted
    d.timesteps = lambda b, hw, settings: torch.full((b,), 0.3)

    def train(settings):
        return d.training_loss(model, [lat], [{"text": None}], settings, generator=torch.Generator().manual_seed(0))
    cos = 2 / (math.pi * (1 - 0.6 + 2 * 0.09))
    assert train({"loss_weighting": "cosmap"}).item() == pytest.approx(cos * train({}).item(), rel=1e-5)


def _modes(cls):
    return [m for m, _ in cls.info.timestep_modes]


def _driver(cls):
    return cls({}, CPU)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_familys_timesteps_land_in_the_noise_range(name):
    cls = FAMILIES[name]
    d = _driver(cls)
    for mode in _modes(cls):
        base = {"timestep_sampling": mode, "num_frames": 17}
        # the full range is the family's own distribution, untouched
        torch.manual_seed(0)
        want = d.sample_timesteps(512, (32, 32), base)
        torch.manual_seed(0)
        assert torch.equal(d.timesteps(512, (32, 32), base), want), mode
        torch.manual_seed(0)
        assert torch.equal(d.timesteps(512, (32, 32), {**base, "timestep_min": 0, "timestep_max": 1}), want), mode
        for lo, hi in ((0.2, 0.6), (0.0, 0.4), (0.5, 1.0)):
            t = d.timesteps(4096, (32, 32), {**base, "timestep_min": lo, "timestep_max": hi})
            eps = 1e-6  # float32 rounding of the range's ends
            assert t.shape == (4096,) and float(t.min()) >= max(lo, d.timestep_floor) - eps, (mode, lo, hi)
            assert float(t.max()) <= hi + eps, (mode, lo, hi)
            # the distribution keeps its shape: a linear map of the family's own
            torch.manual_seed(1)
            own = d.sample_timesteps(64, (32, 32), base)
            torch.manual_seed(1)
            got = d.timesteps(64, (32, 32), {**base, "timestep_min": lo, "timestep_max": hi})
            assert torch.allclose(got, (lo + (hi - lo) * own).clamp_min(d.timestep_floor)), (mode, lo, hi)


def test_an_empty_noise_range_is_refused():
    d = Recording()
    for lo, hi in ((0.5, 0.5), (0.7, 0.3), (-0.1, 0.5), (0.2, 1.5)):
        with pytest.raises(ValueError, match="noise range"):
            d.timesteps(4, (8, 8), {"timestep_min": lo, "timestep_max": hi})


def test_training_steps_and_sliders_draw_from_the_range():
    d = Recording()
    model = torch.nn.Conv2d(3, 3, 1)
    s = {"timestep_min": 0.25, "timestep_max": 0.5}
    lat = [{"latent": torch.randn(3, 4, 6)} for _ in range(64)]
    d.training_loss(model, lat, [{"text": None}] * 64, s)
    t = d.seen[-1]
    assert t.shape == (64,) and float(t.min()) >= 0.25 - 1e-6 and float(t.max()) <= 0.5 + 1e-6
    batch = d.slider_batch(lat, s)
    assert float(batch["t"].min()) >= 0.25 - 1e-6 and float(batch["t"].max()) <= 0.5 + 1e-6

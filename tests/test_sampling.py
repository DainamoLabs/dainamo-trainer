"""Generic flow-matching samplers and schedules.

On Gaussian data the exact velocity is known in closed form, so a many-step solution of the ODE is
the ground truth: second-order samplers must beat Euler at equal step counts.

    .venv/bin/python -m pytest tests/test_sampling.py -v
"""

import math

import pytest
import torch

from dainamo.engine import sampling

MU, C = 1.5, 0.4  # data ~ N(MU, C^2) per element


def gaussian_velocity(x, s):
    """v = (x - E[x0 | x]) / sigma for x = (1 - s) x0 + s n, x0 ~ N(MU, C^2)."""
    a = 1.0 - s
    var = a * a * C * C + s * s
    x0 = MU + a * C * C / var * (x - a * MU)
    return (x - x0) / s


def grid(start, end, n):
    return [start + (end - start) * i / n for i in range(n + 1)]


def ode_reference(x, end, steps=20000):
    return sampling.run("euler", x, grid(1.0, end, steps), gaussian_velocity)


@pytest.mark.parametrize("sampler", list(sampling.SAMPLERS))
def test_all_samplers_land_on_the_data_given_the_true_velocity(sampler):
    x0 = torch.randn(64)
    noise = torch.randn(64)
    v = lambda x, s: (x - x0) / s  # noqa: E731  (the oracle: denoised is always x0)
    sig = sampling.make_sigmas(None, "beta", 7, 64, 64, shift=3.0)
    out = sampling.run(sampler, noise, sig, v)
    assert torch.allclose(out, x0, atol=1e-5)


def test_second_order_samplers_beat_euler():
    """Compared at sigma = 0.05, before the final jump to 0 (a single first-order step for every sampler,
    as usual), the second-order methods are much closer to the exact ODE solution."""
    torch.manual_seed(0)
    noise = torch.randn(4096)
    ref = ode_reference(noise, 0.05)
    err = {s: (sampling.run(s, noise, grid(1.0, 0.05, 8), gaussian_velocity) - ref).abs().mean().item()
           for s in sampling.SAMPLERS}
    assert err["heun"] < err["euler"] / 5, err
    assert err["dpmpp_2m"] < err["euler"] / 2, err
    for s in sampling.SAMPLERS:  # and each converges with more steps
        assert (sampling.run(s, noise, grid(1.0, 0.05, 16), gaussian_velocity) - ref).abs().mean().item() < err[s]


def test_model_call_counts_and_progress():
    calls = []
    sig = sampling.spacing("simple", 5)
    for name, expected in (("euler", 5), ("heun", 9), ("dpmpp_2m", 5)):
        calls.clear()
        sampling.run(name, torch.randn(4), sig, lambda x, s: x, callback=lambda done, total: calls.append((done, total)))
        assert len(calls) == expected == sampling.model_calls(name, 5) and calls[-1] == (expected, expected)


def test_cancel_stops_the_sampler():
    def cb(done, total):
        if done == 2:
            raise sampling.Cancelled()
    with pytest.raises(sampling.Cancelled):
        sampling.run("euler", torch.randn(4), sampling.spacing("simple", 10), lambda x, s: x, callback=cb)


def test_incomplete_beta_and_beta_spacing():
    for x in (0.1, 0.37, 0.8):
        assert math.isclose(sampling._betainc(1, 1, x), x, abs_tol=1e-9)
        assert math.isclose(sampling._betainc(2, 1, x), x * x, abs_tol=1e-9)
        assert math.isclose(sampling._betainc(1, 2, x), 1 - (1 - x) ** 2, abs_tol=1e-9)
        assert math.isclose(sampling._betainc(0.6, 0.6, x) + sampling._betainc(0.6, 0.6, 1 - x), 1.0, abs_tol=1e-9)
    assert math.isclose(sampling.beta_ppf(0.5), 0.5, abs_tol=1e-9)
    u = sampling.spacing("beta", 10)
    assert u[0] == pytest.approx(1.0) and u[-1] == 0.0 and all(a > b for a, b in zip(u, u[1:]))
    # denser at the ends than evenly spaced
    assert u[0] - u[1] < 0.1 and u[4] - u[5] > 0.1


def test_schedules_and_shift():
    class D:
        def default_sigmas(self, steps, w, h):
            return ["model"]

        def default_shift(self, w, h):
            return 3.0

    assert sampling.make_sigmas(D(), "default", 4, 64, 64) == ["model"]
    simple = sampling.make_sigmas(D(), "simple", 4, 64, 64)  # the model's shift on even spacing
    assert simple == [sampling.shift_sigma(u, 3.0) for u in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert sampling.make_sigmas(D(), "default", 4, 64, 64, shift=1.0) == [1.0, 0.75, 0.5, 0.25, 0.0]
    with pytest.raises(ValueError):
        sampling.make_sigmas(D(), "karras", 4, 64, 64)


def kdiffusion_dpmpp_2m(x, sigmas, denoise):
    """k-diffusion's sample_dpmpp_2m (Katherine Crowson, MIT), for x = x0 + sigma * n."""
    t_fn = lambda s: -math.log(s)  # noqa: E731
    old = None
    for i in range(len(sigmas) - 1):
        d = denoise(x, sigmas[i])
        if sigmas[i + 1] == 0:
            x = d
        else:
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old is None:
                x = (sigmas[i + 1] / sigmas[i]) * x - math.expm1(-h) * d
            else:
                r = (t - t_fn(sigmas[i - 1])) / h
                x = (sigmas[i + 1] / sigmas[i]) * x - math.expm1(-h) * ((1 + 1 / (2 * r)) * d - (1 / (2 * r)) * old)
        old = d
    return x


def test_dpmpp_2m_matches_kdiffusion_in_ve_coordinates():
    """DPM-Solver++ depends only on log-SNR, which flow (x = (1-s) x0 + s n) and VE (x/(1-s), s/(1-s))
    coordinates share, so k-diffusion's VE implementation must give the same trajectory."""
    torch.manual_seed(0)
    x = torch.randn(4096)
    sig = [sampling.shift_sigma(u, 3.0) for u in sampling.spacing("beta", 12)][1:]  # start below 1 (VE: finite)
    ours = sampling.run("dpmpp_2m", x, sig, gaussian_velocity)
    ve = [s / (1 - s) for s in sig]
    denoise = lambda xv, sv: (lambda s: xv * (1 - s) - s * gaussian_velocity(xv * (1 - s), s))(sv / (1 + sv))  # noqa: E731
    theirs = kdiffusion_dpmpp_2m(x / (1 - sig[0]), ve, denoise)
    assert torch.allclose(ours, theirs, atol=1e-5), (ours - theirs).abs().max()


def test_saved_clips_keep_every_frame(tmp_path):
    """An MP4 has every frame, whether the soundtrack is as long as the video, shorter or longer."""
    import subprocess

    import numpy as np

    from dainamo.engine.media import save_media

    frames = np.random.default_rng(0).integers(0, 255, (49, 64, 96, 3), dtype=np.uint8)
    for i, seconds in enumerate((49 / 24, 1.5, 3.0)):
        audio = np.zeros((2, round(seconds * 48000)), dtype=np.float32)
        saved = save_media({"frames": frames, "fps": 24, "audio": audio, "sample_rate": 48000}, str(tmp_path), f"c{i}")
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", str(tmp_path / saved["file"])],
                             capture_output=True, text=True, check=True).stdout.strip()
        assert int(out) == 49, (seconds, out)

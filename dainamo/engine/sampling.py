"""Samplers and noise schedules for flow-matching models.

Every image family here uses the same parameterisation: x_sigma = (1 - sigma) * x0 + sigma * noise,
with sigma running from 1 (noise) to 0, and the model (with any CFG) gives the velocity
v = dx/dsigma = noise - x0. A driver supplies the starting noise and a ``velocity(x, sigma)``
function; the sampler and the sigma schedule are chosen here.

Samplers:
  euler      first order, one model call per step (what training samples use)
  heun       second order (trapezoidal), two calls per step except the last
  dpmpp_2m   DPM-Solver++(2M) (Lu et al., 2022) for this path: alpha = 1 - sigma, data prediction
             x0 = x - sigma * v, multistep second order, one call per step

Schedulers give the spacing of the steps; a shift then warps them the way each model was trained
(sigma' = s * u / (1 + (s - 1) * u)):
  default    the model's own schedule (its shift rule, any resolution dependence or snapping)
  simple     evenly spaced
  beta       Beta(0.6, 0.6) quantiles: denser at both ends
"""

from __future__ import annotations

import math
from typing import Callable

import torch

SAMPLERS = {"euler": "Euler", "heun": "Heun", "dpmpp_2m": "DPM++ 2M"}
SCHEDULERS = {"default": "Model default", "simple": "Simple", "beta": "Beta"}
BETA_AB = 0.6


class Cancelled(Exception):
    pass


def shift_sigma(u: float, shift: float) -> float:
    return shift * u / (1.0 + (shift - 1.0) * u)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by its continued fraction (modified Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + num / c
        c = c if abs(c) > 1e-30 else 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def beta_ppf(q: float, a: float = BETA_AB, b: float = BETA_AB) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def spacing(scheduler: str, steps: int) -> list[float]:
    """steps + 1 positions from 1 down to 0, before any shift."""
    if scheduler == "beta":
        return [beta_ppf(1.0 - i / steps) for i in range(steps)] + [0.0]
    return [1.0 - i / steps for i in range(steps)] + [0.0]


def make_sigmas(driver, scheduler: str, steps: int, width: int, height: int, shift: float | None = None) -> list[float]:
    """The sigma schedule (steps + 1 values ending at 0). ``shift`` None means the model's own."""
    if scheduler not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler {scheduler!r}")
    if scheduler == "default" and shift is None:
        return driver.default_sigmas(steps, width, height)
    s = float(shift) if shift is not None else driver.default_shift(width, height)
    return [shift_sigma(u, s) for u in spacing("simple" if scheduler == "default" else scheduler, steps)]


def model_calls(sampler: str, steps: int) -> int:
    return 2 * steps - 1 if sampler == "heun" else steps


Velocity = Callable[[torch.Tensor, float], torch.Tensor]


@torch.no_grad()
def run(sampler: str, x: torch.Tensor, sigmas: list[float], velocity: Velocity,
        callback: Callable[[int, int], None] | None = None) -> torch.Tensor:
    """Integrate from sigmas[0] to sigmas[-1] (= 0). ``callback(done, total)`` runs after each model
    call and may raise ``Cancelled``."""
    if sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler {sampler!r}")
    total = model_calls(sampler, len(sigmas) - 1)
    done = 0

    def v_at(xx, s):
        nonlocal done
        v = velocity(xx, s).float()
        done += 1
        if callback is not None:
            callback(done, total)
        return v

    x = x.float()
    old_x0, old_h = None, None
    for s, sn in zip(sigmas[:-1], sigmas[1:]):
        v = v_at(x, s)
        if sampler == "euler":
            x = x + (sn - s) * v
        elif sampler == "heun":
            xe = x + (sn - s) * v
            x = xe if sn == 0 else x + (sn - s) * (v + v_at(xe, sn)) / 2
        else:  # dpmpp_2m
            x0 = x - s * v
            if sn == 0:
                x = x0
                continue
            if s >= 1.0:  # alpha = 0, lambda = -inf: the step is the exact interpolation
                x, old_x0, old_h = sn * x + (1 - sn) * x0, x0, None
                continue
            lam, lam_n = math.log((1 - s) / s), math.log((1 - sn) / sn)
            h = lam_n - lam
            if old_x0 is not None and old_h is not None:
                r = old_h / h
                d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * old_x0
            else:
                d = x0
            x = (sn / s) * x - (1 - sn) * math.expm1(-h) * d
            old_x0, old_h = x0, h
    return x

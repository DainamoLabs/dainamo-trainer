"""The EMA of a LoRA's weights (``engine.ema``): the warmup ramp, swapping the average in and back out, and
carrying on from a saved state. CPU only.

    .venv/bin/python -m pytest tests/test_ema.py -v
"""

import pytest
import torch
import torch.nn as nn

from dainamo.engine.ema import EMA


def _params():
    torch.manual_seed(0)
    return [nn.Parameter(torch.randn(3, 4)), nn.Parameter(torch.randn(5))]


def test_the_average_starts_at_the_weights_and_lags_behind_them_with_the_warmup_decay():
    params = _params()
    ema = EMA(params, 0.99)
    assert ema.num_updates == 0
    assert all(torch.equal(s, p) for s, p in zip(ema.shadow, params))
    assert all(s.data_ptr() != p.data_ptr() for s, p in zip(ema.shadow, params))  # copies, not the weights
    # the ramp (1 + n) / (10 + n) for the first updates, capped at the decay later
    assert [ema.decay_at(n) for n in range(3)] == pytest.approx([1 / 10, 2 / 11, 3 / 12])
    assert ema.decay_at(10_000) == 0.99

    expected = [p.detach().clone() for p in params]
    for n, d in enumerate((1 / 10, 2 / 11, 3 / 12)):
        with torch.no_grad():
            for p in params:
                p.add_(1.0)  # an optimizer step
        ema.update()
        expected = [d * e + (1 - d) * p.detach() for e, p in zip(expected, params)]
        assert ema.num_updates == n + 1
        assert all(torch.allclose(s, e) for s, e in zip(ema.shadow, expected))
    # it trails the live weights, which moved 3 from where the average started
    start = _params()
    lag = [(s - p0) / 3 for s, p0 in zip(ema.shadow, start)]
    assert all(((l > 0) & (l < 1)).all() for l in lag)

    ema = EMA(_params(), 0.5)  # past the ramp, the decay itself
    assert ema.decay_at(0) == 0.1 and ema.decay_at(8) == 0.5


def test_applied_swaps_the_average_in_and_the_live_weights_back():
    params = _params()
    ema = EMA(params, 0.9)
    with torch.no_grad():
        for p in params:
            p.mul_(2.0)
    ema.update()
    live = [p.detach().clone() for p in params]
    objects = [id(p) for p in params]
    with ema.applied():
        assert all(torch.equal(p, s) for p, s in zip(params, ema.shadow))
        assert not torch.equal(params[0], live[0])
    assert all(torch.equal(p, l) for p, l in zip(params, live))
    assert [id(p) for p in params] == objects and all(p.requires_grad for p in params)

    with pytest.raises(RuntimeError, match="sampling failed"):
        with ema.applied():
            raise RuntimeError("sampling failed")
    assert all(torch.equal(p, l) for p, l in zip(params, live))  # put back on an error too

    # the optimizer still steps the same parameters afterwards
    opt = torch.optim.SGD(params, lr=0.1)
    with ema.applied():
        pass
    sum(p.sum() for p in params).backward()
    opt.step()
    assert torch.allclose(params[0], live[0] - 0.1)


def test_the_state_round_trips_and_carries_on_counting():
    params = _params()
    ema = EMA(params, 0.99)
    for _ in range(4):
        with torch.no_grad():
            params[0].add_(0.5)
        ema.update()
    sd = ema.state_dict()
    assert sd["num_updates"] == 4 and all(t.device.type == "cpu" for t in sd["shadow"])

    # a resumed job: the live weights it carries on with, and the average from its saved state
    resumed = [nn.Parameter(p.detach().clone()) for p in params]
    again = EMA(resumed, 0.99)
    again.load_state_dict(sd)
    assert again.num_updates == 4
    assert all(torch.equal(a, b) for a, b in zip(again.shadow, ema.shadow))
    ema.update()
    again.update()  # the fifth update, decay 5/14, the same on both
    assert again.num_updates == 5 and again.decay_at(4) == pytest.approx(5 / 14)
    assert all(torch.equal(a, b) for a, b in zip(again.shadow, ema.shadow))

    with pytest.raises(ValueError):
        EMA([nn.Parameter(torch.zeros(2))], 0.99).load_state_dict(sd)
    with pytest.raises(ValueError):
        EMA(params, 1.0)

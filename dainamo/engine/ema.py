"""An exponential moving average (EMA) of a LoRA's or LoKr's weights while it trains.

Training steps the live weights as usual; after each optimizer step the average moves a little toward them. The
average is what a job saves, samples and evaluates with (``EMA.applied``), since it smooths out the step-to-step
noise of the last few batches. The decay ramps up from 0 at the start (``decay_t = min(decay, (1 + n) / (10 + n))``
after n updates), so the first saves aren't stuck near the LoRA's initial weights.

The average is kept in fp32 on the weights' own device, one copy of the trainable weights.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


class EMA:
    def __init__(self, params: list[nn.Parameter], decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"The EMA decay must be at least 0 and below 1, not {decay}")
        self.params = list(params)
        self.decay = float(decay)
        self.num_updates = 0
        with torch.no_grad():
            self.shadow = [p.detach().to(torch.float32, copy=True) for p in self.params]

    def decay_at(self, n: int) -> float:
        """The decay of the update after ``n`` earlier ones: the warmup ramp, capped at ``decay``."""
        return min(self.decay, (1 + n) / (10 + n))

    @torch.no_grad()
    def update(self):
        """Move the average toward the live weights; called after each optimizer step."""
        d = self.decay_at(self.num_updates)
        live = [p.detach().to(torch.float32) for p in self.params]  # no copy when the weights are fp32
        torch._foreach_lerp_(self.shadow, live, 1.0 - d)  # shadow = d * shadow + (1 - d) * live
        self.num_updates += 1

    @contextlib.contextmanager
    def applied(self):
        """The average in place of the live weights while the block runs, and the live ones back after (also if
        it raises). The weights are copied in place, so the optimizer and the network's layers keep the same
        parameters."""
        with torch.no_grad():
            live = [p.detach().clone() for p in self.params]
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, t in zip(self.params, live):
                    p.copy_(t)

    def state_dict(self) -> dict:
        """The average (on the CPU) and how many updates it has had; the decay is the job's setting, not saved."""
        return {"shadow": [s.detach().cpu().clone() for s in self.shadow], "num_updates": self.num_updates}

    def load_state_dict(self, sd: dict):
        """Carry on from a saved ``state_dict`` of an EMA of the same weights."""
        shadow = sd["shadow"]
        if len(shadow) != len(self.shadow) or any(a.shape != b.shape for a, b in zip(shadow, self.shadow)):
            raise ValueError("The saved EMA is of other weights than these")
        with torch.no_grad():
            for s, t in zip(self.shadow, shadow):
                s.copy_(t)
        self.num_updates = int(sd["num_updates"])

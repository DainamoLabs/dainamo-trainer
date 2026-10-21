"""LoRA and LoKr networks for Dainamo.

Wraps target Linear layers (plain or ``QuantLinear``) in ``LoRALinear`` (``LoKrLinear`` for a LoKr).
Each wrapper holds the trainable adapter and, optionally, frozen "assistant" adapters (e.g. a turbo
training adapter) that are active while training and switched off when sampling or saving.

Two kinds of trainable adapter:
  * LoRA: the weight change is ``B @ A * alpha / rank``. Saved files use the PEFT-style key layout
    ComfyUI's LoRA loader understands: ``<prefix>.<module>.lora_A.weight`` / ``lora_B.weight`` /
    ``alpha``.
  * LoKr (LyCORIS's Kronecker-product adapter): the weight change is ``kron(w1, w2)``, the layer's
    output and input sizes each split in two (``factorize``); ``w1`` is the small matrix and ``w2``
    the large one, both trained in full. Saved as ``<prefix>.<module>.lokr_w1`` / ``lokr_w2``, the
    LyCORIS key layout ComfyUI loads. The product is never built at full size: the forward pass
    applies ``w2`` and ``w1`` to the input in turn (``kron_delta``).
"""

from __future__ import annotations

import fnmatch
import math

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

KINDS = ("lora", "lokr")


def factorize(dim: int, factor: int = -1) -> tuple[int, int]:
    """``dim`` as (m, dim // m), m the largest divisor of ``dim`` no bigger than ``factor`` or its
    square root (-1: the square root alone, the most even split). This is LyCORIS's split for
    factors up to the square root."""
    limit = math.isqrt(dim) if factor <= 0 else min(factor, math.isqrt(dim))
    m = max(d for d in range(1, limit + 1) if dim % d == 0)
    return m, dim // m


def kron_delta(x: torch.Tensor, w1: torch.Tensor, w2) -> torch.Tensor:
    """``x @ kron(w1, w2).T`` without building the product: ``x``'s features, as blocks (one per w1
    input) of w2's inputs, go through ``w2`` within each block and ``w1`` across the blocks, in
    whichever order keeps the intermediate smaller. ``w2`` may be a low-rank pair (w2_a, w2_b), as
    LyCORIS files can hold."""
    low_rank = isinstance(w2, tuple)
    out_n, in_n = (w2[0].shape[0], w2[1].shape[1]) if low_rank else w2.shape
    out_m, in_m = w1.shape

    def by_w2(h):
        return h @ w2[1].t() @ w2[0].t() if low_rank else h @ w2.t()

    xs = x.unflatten(-1, (in_m, in_n))
    if in_m * out_n <= out_m * in_n:
        y = torch.einsum("pi,...iq->...pq", w1, by_w2(xs))
    else:
        y = by_w2((xs.transpose(-1, -2) @ w1.t()).transpose(-1, -2))
    return y.flatten(-2)  # (..., w1 outputs, w2 outputs) -> the layer's outputs


class Frozen:
    """A frozen adapter's weights on one layer, applied at ``scale``: a LoRA (``down``, ``up``), or a
    LoKr (``w1``, and ``w2`` full or as a low-rank pair)."""

    def __init__(self, scale: float, down=None, up=None, w1=None, w2=None):
        self.scale, self.down, self.up, self.w1, self.w2 = scale, down, up, w1, w2

    def to(self, device, dtype, strength: float = 1.0) -> Frozen:
        """A copy on ``device`` in ``dtype``, its scale times ``strength``."""
        def mv(t):
            if t is None:
                return None
            return tuple(mv(x) for x in t) if isinstance(t, tuple) else t.to(device, dtype)
        return Frozen(self.scale * strength, mv(self.down), mv(self.up), mv(self.w1), mv(self.w2))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.w1 is None:
            x = x.to(self.down.dtype)
            return x @ self.down.t() @ self.up.t() * self.scale
        return kron_delta(x.to(self.w1.dtype), self.w1, self.w2) * self.scale


class LoRALinear(nn.Module):
    """A Linear with a trainable LoRA (rank 0: none, the wrapper only carries frozen adapters)."""

    kind = "lora"

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.in_features, self.out_features = in_f, out_f
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Linear(in_f, rank, bias=False) if rank > 0 else None
        self.lora_B = nn.Linear(rank, out_f, bias=False) if rank > 0 else None
        if rank > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.multiplier = 1.0
        self.enabled = True
        self.assistants: list[Frozen] = []
        self.assistants_enabled = True

    @property
    def scale(self) -> float:
        return self.alpha / self.rank if self.rank else 0.0

    @property
    def trains(self) -> bool:
        return self.lora_A is not None

    def trainable(self) -> list[nn.Parameter]:
        return [self.lora_A.weight, self.lora_B.weight] if self.trains else []

    def delta(self, x: torch.Tensor, strength: float) -> torch.Tensor:
        """The trainable adapter's change to the layer's output at ``strength``."""
        return self.lora_B(self.lora_A(x)) * (self.scale * strength)

    def adapter_to(self, device, dtype):
        self.lora_A.to(device, dtype)
        self.lora_B.to(device, dtype)

    def adapter_state(self, dtype) -> dict[str, torch.Tensor]:
        """The trainable adapter's tensors as saved, by key suffix."""
        return {"lora_A.weight": _saved(self.lora_A.weight, dtype), "lora_B.weight": _saved(self.lora_B.weight, dtype),
                "alpha": torch.tensor(float(self.alpha))}

    def load_adapter(self, sd: dict[str, torch.Tensor], key: str):
        self.lora_A.weight.data.copy_(sd[f"{key}.lora_A.weight"])
        self.lora_B.weight.data.copy_(sd[f"{key}.lora_B.weight"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.assistants_enabled:
            for adapter in self.assistants:
                out = out + adapter(x).to(out.dtype)
        if self.trains and self.enabled and self.multiplier != 0:
            h = self.dropout(x.to(self.trainable()[0].dtype))
            out = out + self.delta(h, self.multiplier).to(out.dtype)
        return out


class LoKrLinear(LoRALinear):
    """A Linear with a trainable LoKr, split by ``factor`` (``factorize``). ``w1`` starts random and
    ``w2`` at zero, as LyCORIS starts them, so the layer starts as it was. Both are full matrices, so
    there's no rank or alpha: loaders apply the product at scale 1."""

    kind = "lokr"

    def __init__(self, base: nn.Module, factor: int = -1, dropout: float = 0.0):
        super().__init__(base, 0, 0.0, dropout)
        self.factor = factor
        out_m, out_n = factorize(self.out_features, factor)
        in_m, in_n = factorize(self.in_features, factor)
        self.lokr_w1 = nn.Parameter(torch.empty(out_m, in_m))
        self.lokr_w2 = nn.Parameter(torch.zeros(out_n, in_n))
        nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))

    @property
    def trains(self) -> bool:
        return True

    def trainable(self) -> list[nn.Parameter]:
        return [self.lokr_w1, self.lokr_w2]

    def delta(self, x: torch.Tensor, strength: float) -> torch.Tensor:
        return kron_delta(x, self.lokr_w1, self.lokr_w2) * strength

    def adapter_to(self, device, dtype):
        for p in self.trainable():
            p.data = p.data.to(device, dtype)

    def adapter_state(self, dtype) -> dict[str, torch.Tensor]:
        return {"lokr_w1": _saved(self.lokr_w1, dtype), "lokr_w2": _saved(self.lokr_w2, dtype)}

    def load_adapter(self, sd: dict[str, torch.Tensor], key: str):
        self.lokr_w1.data.copy_(sd[f"{key}.lokr_w1"])
        self.lokr_w2.data.copy_(sd[f"{key}.lokr_w2"])


def _saved(t: torch.Tensor, dtype) -> torch.Tensor:
    return t.detach().to("cpu", dtype).contiguous()


class LoRANetwork:
    """The trainable adapters on a model's target layers: LoRAs (``kind="lora"``), or LoKrs split by
    ``factor`` (``kind="lokr"``; ``rank`` and ``alpha`` don't apply to them)."""

    def __init__(self, model: nn.Module, target_patterns: list[str], rank: int, alpha: float,
                 dropout: float = 0.0, save_prefix: str = "diffusion_model.", kind: str = "lora", factor: int = -1):
        if kind not in KINDS:
            raise ValueError(f"Unknown network type {kind!r}")
        self.model = model
        self.save_prefix = save_prefix
        self.kind, self.factor = kind, factor
        self.modules: dict[str, LoRALinear] = {}
        targets = []
        for name, module in model.named_modules():
            if not hasattr(module, "in_features") or isinstance(module, LoRALinear):
                continue
            if any(fnmatch.fnmatchcase(name, p) for p in target_patterns):
                targets.append((name, module))
        if not targets:
            raise ValueError(f"No layers matched LoRA targets {target_patterns}")
        self.wrappers: dict[str, LoRALinear] = {}  # every wrapped layer, incl. assistant-only
        for name, _ in targets:
            self.modules[name] = self._wrap(name, rank, alpha, dropout)

    def _wrap(self, name: str, rank: int, alpha: float, dropout: float = 0.0) -> LoRALinear:
        parent_name, _, attr = name.rpartition(".")
        parent = self.model.get_submodule(parent_name) if parent_name else self.model
        base = getattr(parent, attr)
        if self.kind == "lokr":
            wrapper = LoKrLinear(base, self.factor, dropout)
        else:
            wrapper = LoRALinear(base, rank, alpha, dropout)
        setattr(parent, attr, wrapper)
        self.wrappers[name] = wrapper
        return wrapper

    def to(self, device, dtype=torch.float32):
        for m in self.modules.values():
            m.adapter_to(device, dtype)
        return self

    def parameters(self):
        for m in self.modules.values():
            yield from m.trainable()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_multiplier(self, multiplier: float):
        """LoRA strength (1.0 = as trained, 0.0 = base model)."""
        for m in self.modules.values():
            m.multiplier = float(multiplier)

    def set_enabled(self, enabled: bool):
        for m in self.modules.values():
            m.enabled = enabled

    def set_assistants_enabled(self, enabled: bool):
        for m in self.wrappers.values():
            m.assistants_enabled = enabled

    def state_dict(self, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        sd = {}
        for name, m in self.modules.items():
            k = f"{self.save_prefix}{name}"
            sd.update({f"{k}.{suffix}": t for suffix, t in m.adapter_state(dtype).items()})
        return sd

    def save(self, path: str, metadata: dict[str, str] | None = None, dtype=torch.bfloat16):
        save_file(self.state_dict(dtype), path, metadata=metadata or {})

    def load(self, path: str):
        self.load_state_dict(load_file(path))

    def load_state_dict(self, sd: dict[str, torch.Tensor]):
        """Copy the adapters' weights in (from a saved file or ``state_dict()``), keeping dtype/device."""
        for name, m in self.modules.items():
            m.load_adapter(sd, f"{self.save_prefix}{name}")

    def attach_assistant(self, path: str, device, dtype=torch.bfloat16, multiplier: float = 1.0) -> int:
        """Attach a frozen LoRA or LoKr (e.g. a turbo training adapter). Returns number of layers matched.

        Layers the adapter touches that aren't LoRA targets get a rank-0 wrapper
        so the adapter still applies there.
        """
        matched = 0
        for name, adapter in adapter_layers(load_file(path)):
            wrapper = self.wrappers.get(name) or _wrap_existing(self.model, name, self.wrappers)
            if wrapper is None:
                continue
            wrapper.assistants.append(adapter.to(device, dtype, multiplier))
            matched += 1
        if matched == 0:
            raise ValueError(f"Training adapter {path} matched no layers of this model")
        return matched


LORA_PREFIXES = ("diffusion_model.", "transformer.")
# kohya-style files name layers lora_unet_<path_with_underscores>; LyCORIS's own trainers use lycoris_
KOHYA_PREFIXES = ("lora_unet_", "lycoris_")


def adapter_layers(sd: dict[str, torch.Tensor]) -> list[tuple[str, Frozen]]:
    """(module name, frozen adapter) for every LoRA (``lora_A``/``lora_B``) and LoKr (``lokr_w1``/``lokr_w2``,
    either may be a low-rank ``_a``/``_b`` pair) in a PEFT/ComfyUI-style state dict. As loaders do,
    alpha scales by the rank: a LoRA's, or a LoKr's low-rank pair's (a LoKr of full matrices: 1)."""
    if any(k.endswith(".dora_scale") for k in sd):
        raise ValueError("DoRA adapters aren't supported")
    out = []
    for key in sd:
        base, _, suffix = key.rpartition(".")
        if key.endswith(".lora_A.weight"):
            base = key[: -len(".lora_A.weight")]
            a, b = sd[key], sd[f"{base}.lora_B.weight"]
            alpha = sd.get(f"{base}.alpha")
            out.append((base, Frozen((float(alpha) / a.shape[0]) if alpha is not None else 1.0, down=a, up=b)))
        elif suffix in ("lokr_w1", "lokr_w1_a"):
            if suffix == "lokr_w1_a" and f"{base}.lokr_w1" in sd:
                continue
            if f"{base}.lokr_t2" in sd:
                raise ValueError("LoKr adapters for convolutions (lokr_t2) aren't supported")
            w1 = sd.get(f"{base}.lokr_w1")
            if w1 is None:
                w1 = sd[f"{base}.lokr_w1_a"].float() @ sd[f"{base}.lokr_w1_b"].float()
            w2 = sd.get(f"{base}.lokr_w2")
            if w2 is None:
                w2 = (sd[f"{base}.lokr_w2_a"], sd[f"{base}.lokr_w2_b"])
            rank = (w2[1].shape[0] if isinstance(w2, tuple)
                    else sd[f"{base}.lokr_w1_b"].shape[0] if f"{base}.lokr_w1_b" in sd else None)
            alpha = sd.get(f"{base}.alpha")
            out.append((base, Frozen(float(alpha) / rank if alpha is not None and rank else 1.0, w1=w1, w2=w2)))
    return [(next((n[len(p):] for p in LORA_PREFIXES if n.startswith(p)), n), f) for n, f in out]


def _wrap_existing(model: nn.Module, name: str, wrappers: dict) -> LoRALinear | None:
    """A rank-0 wrapper (adapters only) around ``name``, or None if it isn't a Linear of this model."""
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return None
    if isinstance(module, LoRALinear):
        wrappers[name] = module
        return module
    if not hasattr(module, "in_features"):
        return None
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    wrapper = LoRALinear(module, 0, 0.0)
    setattr(parent, attr, wrapper)
    wrappers[name] = wrapper
    return wrapper


class AdapterStack:
    """Frozen LoRAs and LoKrs applied at inference, each at its own strength (the Sandbox). Wrapping is
    permanent; ``set`` replaces what the wrappers apply."""

    def __init__(self, model: nn.Module, device, dtype=torch.bfloat16):
        self.model, self.device, self.dtype = model, device, dtype
        self.wrappers: dict[str, LoRALinear] = {}

    def clear(self):
        for w in self.wrappers.values():
            w.assistants = []

    def set_enabled(self, enabled: bool):
        for w in self.wrappers.values():
            w.assistants_enabled = enabled

    def _from_kohya(self, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """kohya-style ``lora_unet_<path_with_underscores>.lora_down/lora_up`` (or ``.lokr_w1`` ...) -> PEFT
        keys, resolving the underscored path against this model's module names."""
        names = {n.replace(".", "_"): n for n, m in self.model.named_modules() if hasattr(m, "in_features")}
        out = {}
        for key, v in sd.items():
            stem, _, rest = key.partition(".")
            prefix = next((p for p in KOHYA_PREFIXES if stem.startswith(p)), None)
            if prefix is None:
                continue
            name = names.get(stem[len(prefix):])
            if name is None:
                continue
            rest = {"lora_down.weight": "lora_A.weight", "lora_up.weight": "lora_B.weight"}.get(rest, rest)
            out[f"{name}.{rest}"] = v
        return out

    def set(self, loras: list[tuple[str, dict[str, torch.Tensor], float]]) -> list[dict]:
        """Apply [(label, state dict, strength)]; returns [{label, matched, total}] per LoRA. A file that
        isn't a LoRA or LoKr, or one that matches no layer of the model, raises."""
        self.clear()
        report = []
        for label, sd, strength in loras:
            if any(k.startswith(KOHYA_PREFIXES) for k in sd):
                sd = self._from_kohya(sd)
            try:
                layers = adapter_layers(sd)
            except ValueError as e:
                self.clear()
                raise ValueError(f"{label}: {e}") from None
            if not layers:
                self.clear()
                raise ValueError(f"{label} has no LoRA weights this model can use (expects lora_A/lora_B, "
                                 "kohya lora_down/lora_up or LoKr lokr_w1/lokr_w2 keys)")
            matched = 0
            for name, adapter in layers:
                wrapper = self.wrappers.get(name) or _wrap_existing(self.model, name, self.wrappers)
                if wrapper is None:
                    continue
                wrapper.assistants.append(adapter.to(self.device, self.dtype, float(strength)))
                matched += 1
            if matched == 0:
                self.clear()
                raise ValueError(f"LoRA {label} matches no layers of this model (wrong model family?)")
            report.append({"label": label, "matched": matched, "total": len(layers)})
        self.set_enabled(True)
        return report

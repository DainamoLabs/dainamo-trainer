"""Full finetuning: every weight of the diffusion model trained, instead of an adapter beside it.

``FullModel`` stands in for a ``LoRANetwork`` in the training loop: its parameters are the model's own, its
strength can't change (``set_multiplier`` does nothing), and ``save`` writes the whole model as a new
diffusion-model file in the layout of the one it was loaded from, which ComfyUI loads as it loads that one:
the same key prefix and the same non-weight tensors (MOSS-Image's ``rope.bands``), with every layer in bf16
(a quantized source's layers are expanded once at load, and their scales and ``comfy_quant`` markers are
left out).

The whole model's gradients and an AdamW's fp32 state wouldn't fit beside it on a 24 GB GPU, so
``FusedOptimizer`` steps each weight as soon as backward has its gradient (PyTorch's "optimizer step in
backward" recipe) and drops the gradient at once: only a few gradients exist at a time. Each step runs on an
fp32 copy of the weight, which is written back to bf16 with stochastic rounding: a finetune's updates are
far smaller than bf16's spacing near the weights, so rounding to nearest would lose nearly all of them,
while stochastic rounding keeps each one on average. Because the updates happen during backward, there's no
gradient accumulation and no global gradient-norm clipping; Adafactor clips its own updates.
"""

from __future__ import annotations

import json
import os
import struct
from functools import partial

import torch
import torch.nn as nn

from ..fsutil import replace
from . import comfy_io

# the optimizers a finetune can use: their state fits beside the model (Adafactor's is factored into a row and
# a column per matrix; AdamW 8-bit's is 2 bytes a weight). AdamW and Prodigy keep 8 or more bytes a weight.
OPTIMIZERS = ("adafactor", "adamw8bit")

SAFETENSORS_DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
                      torch.int8: "I8", torch.uint8: "U8", torch.int16: "I16", torch.int32: "I32",
                      torch.int64: "I64", torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3",
                      torch.float8_e5m2: "F8_E5M2"}


def stochastic_round_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Write fp32 ``src`` into bf16 ``dst``, rounding each value up or down at random in proportion to how near it
    is to each (so the rounding is unbiased): random bits are added below bf16's 16 kept bits, then truncated."""
    bits = src.contiguous().view(torch.int32)
    bits = (bits + torch.randint_like(bits, 0, 1 << 16)) & -65536
    return dst.copy_(bits.view(torch.float32))


def _make(name: str, params: list, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, weight_decay=weight_decay, relative_step=False, scale_parameter=False,
                         warmup_init=False)
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    raise ValueError(f"A finetune can't use the {name} optimizer (only {', '.join(OPTIMIZERS)})")


class FusedOptimizer(torch.optim.Optimizer):
    """One optimizer per parameter, stepped from the parameter's post-accumulate-grad hook (``FullModel``'s
    docstring says why). It has one param group of every parameter, so a learning-rate scheduler drives it as
    it drives any optimizer: each step reads that group's lr and weight decay. ``step`` and ``zero_grad`` have
    nothing left to do.

    Each inner optimizer steps an fp32 proxy of its parameter, which holds a copy of the weight and its gradient
    only for that step. Their state lives in one dict, ``state``, keyed by proxy (so ``offload.park_optimizer``
    sees all of it)."""

    def __init__(self, params, name: str, lr: float, weight_decay: float = 0.0):
        params = list(params)
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
        self.name = name
        self.proxies: list[nn.Parameter] = []
        self.inner: list[torch.optim.Optimizer] = []
        self._empty: dict[torch.device, torch.Tensor] = {}
        self.hooks = []
        for i, p in enumerate(params):
            proxy = nn.Parameter(self._placeholder(p.device))
            opt = _make(name, [proxy], lr, weight_decay)
            opt.state = self.state
            self.proxies.append(proxy)
            self.inner.append(opt)
            self.hooks.append(p.register_post_accumulate_grad_hook(partial(self._step_one, i)))

    def _placeholder(self, device) -> torch.Tensor:
        """What a proxy holds between steps: nothing, on its parameter's device (where loaded state is moved)."""
        if device not in self._empty:
            self._empty[device] = torch.empty(0, device=device, dtype=torch.float32)
        return self._empty[device]

    @torch.no_grad()
    def _step_one(self, i: int, p: torch.Tensor):
        proxy, opt = self.proxies[i], self.inner[i]
        group = self.param_groups[0]
        opt.param_groups[0].update(lr=group["lr"], weight_decay=group["weight_decay"])
        proxy.data = p.data.float()
        proxy.grad = p.grad.float()
        p.grad = None
        opt.step()
        if p.dtype == torch.bfloat16:
            stochastic_round_(p.data, proxy.data)
        elif proxy.data.data_ptr() != p.data.data_ptr():  # an fp32 weight is its own proxy
            p.data.copy_(proxy.data)
        proxy.grad = None
        proxy.data = self._placeholder(p.device)

    def step(self, closure=None):
        """Each parameter was stepped as backward reached it."""
        return None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def state_dict(self) -> dict:
        return {"optimizer": self.name, "state": [dict(self.state.get(px, {})) for px in self.proxies],
                "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups]}

    def load_state_dict(self, sd: dict):
        if sd.get("optimizer") != self.name or len(sd.get("state", [])) != len(self.proxies):
            raise ValueError("The saved optimizer state is for another optimizer or another model")
        for px, st in zip(self.proxies, sd["state"]):
            self.state[px] = {k: v.to(px.device) if torch.is_tensor(v) else v for k, v in st.items()}
        for g, saved in zip(self.param_groups, sd["param_groups"]):
            g.update(saved)


# a source's other tensors up to this size in all are read once, when training starts (its file may be one of
# the job's own checkpoints, which a continued job renames); larger ones (a full checkpoint's text encoder or
# VAE) are read from it at each save
KEEP_EXTRA_BYTES = 256 * 2**20


class Layout:
    """How a source diffusion-model file is laid out, for ``save_model``: its key prefix, its tensors that aren't
    the model's weights nor a model layer's quantization scales or markers (``extra``, by file key), and its header
    metadata without quantization info on the model's layers. ComfyUI's runtime hints for a quantized file (such as
    ``comfy_attention.config``'s int8 attention) and legacy fp8 markers are left out wherever they are."""

    def __init__(self, source: str, model: nn.Module):
        from safetensors import safe_open

        self.source = source
        keys, meta = comfy_io.read_header(source)
        self.prefix = "model.diffusion_model." if any(k.startswith("model.diffusion_model.") for k in keys) else ""
        state = set(model.state_dict())
        layers = {k[: -len(".weight")] for k in state if k.endswith(".weight")}

        def belongs(name: str) -> bool:
            """A model layer's quantization scale or marker, dropped with the quantization."""
            for suffix in comfy_io.SIDECAR_SUFFIXES:
                if name.endswith(suffix):
                    return name[: -len(suffix)] in layers
            return False

        self.extra: list[str] = []
        for k in keys:
            if self.prefix and not k.startswith(self.prefix):
                self.extra.append(k)  # e.g. a full checkpoint's text encoder or VAE
                continue
            name = k[len(self.prefix):]
            if not (name in state or name in comfy_io.LEGACY_MARKERS or name.endswith(comfy_io.METADATA_SUFFIXES)
                    or belongs(name)):
                self.extra.append(k)
        if "_quantization_metadata" in meta:
            try:
                q = json.loads(meta["_quantization_metadata"])
                q["layers"] = {n: c for n, c in (q.get("layers") or {}).items() if n not in layers}
                meta["_quantization_metadata"] = json.dumps(q)
                if not q["layers"]:
                    del meta["_quantization_metadata"]
            except (json.JSONDecodeError, AttributeError):
                del meta["_quantization_metadata"]
        self.metadata = meta
        self.kept: dict[str, torch.Tensor] | None = None
        with safe_open(source, "pt") as f:
            size = 0
            for k in self.extra:
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                size += n * DTYPE_SIZES.get(sl.get_dtype(), 4)
            if size <= KEEP_EXTRA_BYTES:
                self.kept = {k: f.get_tensor(k) for k in self.extra}

    def extras(self) -> list[tuple[str, torch.Tensor]]:
        if self.kept is not None:
            return list(self.kept.items())
        from safetensors import safe_open

        with safe_open(self.source, "pt") as f:
            return [(k, f.get_tensor(k)) for k in self.extra]


DTYPE_SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1}


class FullModel:
    """The diffusion model itself as what a job trains, in a ``LoRANetwork``'s place (``kind`` "full"). ``source``
    is a file of it (the one it was loaded from, or the profile's), whose layout ``save`` copies."""

    kind = "full"

    def __init__(self, model: nn.Module, source: str):
        self.model = model
        self.layout = Layout(source, model)
        self.modules: dict = {}  # no adapter layers
        self.params = [p for p in model.parameters()]

    def parameters(self):
        return iter(self.params)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.params)

    def set_multiplier(self, multiplier: float):
        """The trained weights are the model's own, always at full strength."""

    def set_enabled(self, enabled: bool):
        pass

    def set_assistants_enabled(self, enabled: bool):
        pass

    def save(self, path: str, metadata: dict[str, str] | None = None):
        save_model(self.model, path, self.layout, metadata)


def save_model(model: nn.Module, path: str, layout: Layout, metadata: dict[str, str] | None = None) -> None:
    """Write ``model``'s weights as a diffusion-model file in ``layout``: under its prefix, beside its other tensors
    as they are, with its metadata and ``metadata``."""
    tensors = [(layout.prefix + k, v) for k, v in model.state_dict().items()] + layout.extras()
    write_safetensors(path, tensors, {**layout.metadata, **(metadata or {})})


def write_safetensors(path: str, tensors: list[tuple[str, torch.Tensor]], metadata: dict[str, str]) -> None:
    """A .safetensors file written one tensor at a time (``safetensors.torch.save_file`` builds the whole file in
    memory first, twice the size of a model), to ``path`` through a temporary file."""
    tensors = sorted(tensors, key=lambda kv: kv[0])
    header, offset = {}, 0
    for k, t in tensors:
        if t.dtype not in SAFETENSORS_DTYPES:
            raise ValueError(f"{k}: can't save {t.dtype}")
        n = t.numel() * t.element_size()
        header[k] = {"dtype": SAFETENSORS_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + n]}
        offset += n
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for _, t in tensors:
                if t.numel():
                    f.write(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().data)
        replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


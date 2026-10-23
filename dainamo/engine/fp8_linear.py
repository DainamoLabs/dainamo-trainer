"""Linear layers that keep frozen base weights quantized and dequantize on the fly.

Handles fp8 (per-tensor or per-row scale) and int8 per-row weights, optionally
in the convrot basis. RTX 30-series cards have no fp8 matmul, so the weight
codes are expanded to the compute dtype just for the matmul (fp8 and int8 values
are exact in bf16) and the per-row scale is applied to the output in fp32. A
custom autograd function stores only the quantized weight for backward (never
the expanded copy), so a 12B model stays ~12 GB.

For convrot layers the stored weight is ``W @ H`` (H = block Hadamard, self-
inverse), so ``x @ W.T == rotate(x) @ (W @ H).T``: the input is rotated instead
of un-rotating the weight, which is cheaper and keeps the weight int8.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .comfy_io import LoadedWeights, QuantLayer, dequantize, nvfp4_dequantize, rotate


class _QuantMatmul(torch.autograd.Function):
    """y = (x @ codes.T) * scale + bias, with ``scale`` per output channel (or scalar)."""

    @staticmethod
    def forward(ctx, x, weight, scale, bias, dtype):
        cuda_autocast = x.is_cuda and torch.is_autocast_enabled("cuda")
        mm_dtype = torch.get_autocast_dtype("cuda") if cuda_autocast else dtype
        if mm_dtype == torch.float16:
            # fp16 can't hold the unscaled products (codes up to 127/448 times activations pass 65504
            # quickly), so fold the scale into the weight first, as an fp32 weight under fp16 would be
            s = scale.reshape(-1, 1) if scale.numel() > 1 else scale
            y = F.linear(x.to(mm_dtype), (weight.float() * s).to(mm_dtype)).float()
        else:
            y = F.linear(x, weight.to(dtype)).float() * scale
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(weight, scale)
        ctx.dtype = dtype
        return y.to(dtype)

    @staticmethod
    def backward(ctx, grad_out):
        weight, scale = ctx.saved_tensors
        g = (grad_out.float() * scale).to(ctx.dtype)
        return g @ weight.to(ctx.dtype), None, None, None, None


class _Nvfp4Matmul(torch.autograd.Function):
    """y = x @ dequant(packed).T + bias. Only the packed codes and scales are saved for backward."""

    @staticmethod
    def forward(ctx, x, packed, block_scales, pts, bias, dtype):
        w = nvfp4_dequantize(packed, block_scales, pts, dtype)
        ctx.save_for_backward(packed, block_scales, pts)
        ctx.dtype = dtype
        return F.linear(x.to(dtype), w, None if bias is None else bias.to(dtype))

    @staticmethod
    def backward(ctx, grad_out):
        packed, block_scales, pts = ctx.saved_tensors
        w = nvfp4_dequantize(packed, block_scales, pts, ctx.dtype)
        return grad_out.to(ctx.dtype) @ w, None, None, None, None, None


class Nvfp4Linear(nn.Module):
    """Frozen Linear stored as NVFP4 (~4.5 bits/weight), expanded per call. An AWQ
    pre_quant_scale multiplies the input first."""

    def __init__(self, in_features: int, out_features: int, packed: torch.Tensor, q: QuantLayer,
                 bias: torch.Tensor | None, compute_dtype: torch.dtype):
        super().__init__()
        self.in_features, self.out_features, self.compute_dtype = in_features, out_features, compute_dtype
        self.register_buffer("weight", packed, persistent=False)
        self.register_buffer("block_scale", q.weight_scale.to(packed.device), persistent=False)
        self.register_buffer("scale", q.weight_scale_2.to(packed.device).float().reshape(()), persistent=False)
        pre = q.pre_quant_scale
        self.register_buffer("pre_scale", pre.to(packed.device).float() if pre is not None else None, persistent=False)
        self.register_buffer("bias", bias.float() if bias is not None else None, persistent=False)

    def dequantized_weight(self) -> torch.Tensor:
        """The weight for raw (un-pre-scaled) activations."""
        w = nvfp4_dequantize(self.weight, self.block_scale, self.scale, torch.float32)
        if self.pre_scale is not None:
            w = w * self.pre_scale.unsqueeze(0)
        return w.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.compute_dtype)
        if self.pre_scale is not None:
            x = x * self.pre_scale.to(x.dtype)
        return _Nvfp4Matmul.apply(x, self.weight, self.block_scale, self.scale, self.bias, self.compute_dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, nvfp4" + (", awq" if self.pre_scale is not None else "")


class QuantLinear(nn.Module):
    """Frozen Linear whose weight lives in fp8 or int8 with a scale (and optional convrot basis)."""

    def __init__(self, in_features: int, out_features: int, weight: torch.Tensor, scale: torch.Tensor | None,
                 bias: torch.Tensor | None, compute_dtype: torch.dtype, rot_size: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.rot_size = rot_size
        if rot_size > 1 and in_features % rot_size:
            raise ValueError(f"in_features {in_features} not divisible by convrot group size {rot_size}")
        self.register_buffer("weight", weight, persistent=False)
        if scale is None:
            scale = torch.ones((), dtype=torch.float32, device=weight.device)
        # (out,) per output channel, or a scalar; broadcasts over the output's last dim
        scale = scale.float().reshape(-1) if scale.numel() > 1 else scale.float().reshape(())
        if scale.numel() not in (1, out_features):
            raise ValueError(f"scale with {scale.numel()} values doesn't match out_features {out_features}")
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("bias", bias.float() if bias is not None else None, persistent=False)

    def dequantized_weight(self) -> torch.Tensor:
        """The real weight in the original basis."""
        scale = self.scale.reshape(-1, 1) if self.scale.numel() > 1 else self.scale
        w = self.weight.float() * scale
        return rotate(w, self.rot_size).to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rotate(x.to(self.compute_dtype), self.rot_size)
        return _QuantMatmul.apply(x, self.weight, self.scale, self.bias, self.compute_dtype)

    def extra_repr(self) -> str:
        kind = "int8" if self.weight.dtype == torch.int8 else "fp8"
        rot = f", convrot={self.rot_size}" if self.rot_size > 1 else ""
        return f"in={self.in_features}, out={self.out_features}, {kind}{rot}"


Fp8Linear = QuantLinear  # backwards-compatible name


def _get_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def materialize(model: nn.Module, loaded: LoadedWeights, dtype: torch.dtype, keep_fp8: bool = True,
                device: torch.device | str | None = "cpu", dtype_for=None) -> None:
    """Load ``loaded`` into a meta-device ``model``.

    Quantized Linears become ``QuantLinear`` (when ``keep_fp8``) or are dequantized
    to ``dtype``. Embeddings are always expanded to ``dtype``. ``dtype_for(key)`` may return a
    different dtype for specific unquantized tensors (e.g. a model's deliberate fp32 layers). Everything else is cast to ``dtype``. Fails loudly on missing or
    unexpected keys so a wrong file/architecture is caught immediately. ``device=None`` leaves every
    tensor where it was loaded (e.g. some blocks on the GPU and the rest in system RAM).
    """
    def put(t: torch.Tensor, dt: torch.dtype | None = None) -> torch.Tensor:
        if device is None:
            return t if dt is None else t.to(dtype=dt)
        return t.to(device) if dt is None else t.to(device, dt)

    state = loaded.tensors
    expected = set(model.state_dict().keys())
    consumed = set()

    for prefix, q in loaded.quant.items():
        module = model.get_submodule(prefix)
        weight = state[f"{prefix}.weight"]
        if isinstance(module, nn.Embedding):
            # lookups only; just expand to the compute dtype
            parent, attr = _get_parent(model, prefix)
            emb = nn.Embedding(module.num_embeddings, module.embedding_dim, padding_idx=module.padding_idx, device="meta")
            emb.weight = nn.Parameter(put(dequantize(weight, q, dtype)), requires_grad=False)
            setattr(parent, attr, emb)
            consumed.add(f"{prefix}.weight")
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Quantized tensor {prefix} is not a Linear ({type(module).__name__})")
        bias = state.get(f"{prefix}.bias")
        consumed.add(f"{prefix}.weight")
        if bias is not None:
            consumed.add(f"{prefix}.bias")
        parent, attr = _get_parent(model, prefix)
        if keep_fp8 and q.format == "nvfp4":
            new = Nvfp4Linear(module.in_features, module.out_features, put(weight), q,
                              put(bias) if bias is not None else None, dtype)
        elif keep_fp8:
            new = QuantLinear(module.in_features, module.out_features, put(weight),
                              put(q.weight_scale) if q.weight_scale is not None else None,
                              put(bias) if bias is not None else None, dtype, rot_size=q.rot_size)
        else:
            new = nn.Linear(module.in_features, module.out_features, bias=bias is not None, device="meta")
            new.weight = nn.Parameter(put(dequantize(weight, q, dtype)), requires_grad=False)
            del state[f"{prefix}.weight"], weight  # its expanded copy replaces it, so the two never all exist at once
            if bias is not None:
                new.bias = nn.Parameter(put(bias, dtype), requires_grad=False)
        setattr(parent, attr, new)

    def target(k):
        return (dtype_for(k) if dtype_for is not None else None) or dtype

    rest = {k: (put(v, target(k)) if v.is_floating_point() else put(v))
            for k, v in state.items() if k not in consumed}
    missing = (expected - consumed) - set(rest.keys())
    unexpected = set(rest.keys()) - expected
    if missing or unexpected:
        raise ValueError(
            f"Weights don't match the architecture. missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}"
        )
    model.load_state_dict(rest, strict=False, assign=True)
    model.requires_grad_(False)


def assert_materialized(model: nn.Module) -> None:
    """Fail if any parameter or buffer is still a meta-device placeholder."""
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"Model has uninitialised tensors after loading: {left[:8]}")

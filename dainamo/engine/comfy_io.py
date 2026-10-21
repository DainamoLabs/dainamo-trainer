"""Native reader for ComfyUI-style model files.

ComfyUI stores models as plain ``.safetensors`` files. The quantized ones use a
few simple conventions, which are handled here without any ComfyUI code:

* ``fp8_scaled``: a Linear's ``weight`` is float8 with a sibling ``weight_scale``
  tensor (scalar or per-row). The real weight is ``weight.float() * weight_scale``.
  Older files name the scale ``scale_weight`` and carry a root ``scaled_fp8`` marker.
* ``comfy_quant``: a uint8 tensor next to the weight holding a JSON blob that
  names the format (e.g. ``{"format": "float8_e4m3fn"}``).
* ``int8_tensorwise``: int8 weight with a per-output-row ``weight_scale``. With
  ``"convrot": true`` the weight was first rotated along its input dimension by
  a block regular-Hadamard matrix (``convrot_groupsize`` wide), so the real
  weight is ``rotate(weight * weight_scale)`` (the rotation is its own inverse).
* ``nvfp4``: FP4 (E2M1) codes packed two per byte, high nibble first, with one FP8 (E4M3)
  scale per 16 values (stored in cuBLAS's swizzled 128x4 tile layout), one FP32 per-tensor
  ``weight_scale_2``, and optionally an AWQ ``pre_quant_scale`` that multiplies the input
  activations before the matmul.
* ``_quantization_metadata`` in the file header: the same per-layer info as JSON.

``load_state_dict`` returns plain tensors plus a ``QuantLayer`` map so callers can
either dequantize everything or keep weights quantized (see ``fp8_linear``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
QUANT_DTYPES = FP8_DTYPES + (torch.int8,)
WIDE_DTYPES = (torch.float32, torch.float16, torch.float64)
SUPPORTED_FORMATS = ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "nvfp4")
SIDECAR_SUFFIXES = (".weight_scale", ".input_scale", ".comfy_quant", ".weight_scale_2", ".scale_weight",
                    ".scale_input", ".pre_quant_scale")
NVFP4_BLOCK = 16
# E2M1 code -> value; bit 3 is the sign
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
LEGACY_MARKERS = ("scaled_fp8",)  # root-level marker tensor in older fp8_scaled files
# per-layer ComfyUI runtime hints (e.g. {"attention": "comfy_kitchen_int8"}); not weights
METADATA_SUFFIXES = (".comfy_attention.config",)


@dataclass
class QuantLayer:
    format: str
    weight_scale: torch.Tensor | None = None
    input_scale: torch.Tensor | None = None
    rot_size: int = 1  # block width of the convrot Hadamard rotation (1 = none)
    weight_scale_2: torch.Tensor | None = None   # nvfp4 per-tensor scale
    pre_quant_scale: torch.Tensor | None = None  # AWQ: input activations are multiplied by this

    @property
    def label(self) -> str:
        if self.format == "int8_tensorwise":
            return "int8 convrot" if self.rot_size > 1 else "int8"
        if self.format == "nvfp4":
            return "nvfp4"
        return "fp8 scaled"


_hadamard_cache: dict[tuple, torch.Tensor] = {}


def regular_hadamard(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal Kronecker power of the 4x4 regular Hadamard matrix (symmetric, self-inverse)."""
    key = (n, str(device), dtype)
    if key not in _hadamard_cache:
        r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]])
        h = r4
        while h.shape[0] < n:
            h = torch.kron(h, r4)
        if h.shape[0] != n:
            raise ValueError(f"convrot group size {n} is not a power of 4")
        _hadamard_cache[key] = (h / n**0.5).to(device=device, dtype=dtype)
    return _hadamard_cache[key]


def unswizzle_nvfp4_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo the cuBLAS 128x4-tile block-scale layout back to row-major (rows, cols).
    Layout as documented by ai-toolkit's ``unswizzle_nvfp4_scales`` (MIT)."""
    n_row_blocks, n_col_blocks = (rows + 127) // 128, (cols + 3) // 4
    x = scales.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(n_row_blocks, n_col_blocks, 4, 32, 4).reshape(n_row_blocks, n_col_blocks, 128, 4)
    x = x.permute(0, 2, 1, 3).reshape(n_row_blocks * 128, n_col_blocks * 4)
    return x[:rows, :cols].contiguous()


def nvfp4_dequantize(packed: torch.Tensor, block_scales: torch.Tensor, pts: torch.Tensor, dtype: torch.dtype,
                     chunk_rows: int = 4096) -> torch.Tensor:
    """packed (out, in/2) uint8, block_scales (out, in/16) e4m3, pts scalar -> (out, in) ``dtype``.
    Works through row chunks so the temporaries stay small."""
    rows, half = packed.shape
    cols = half * 2
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    out = torch.empty(rows, cols, dtype=dtype, device=packed.device)
    gscale = pts.to(device=packed.device, dtype=torch.float32).reshape(())
    for i in range(0, rows, chunk_rows):
        p = packed[i : i + chunk_rows]
        codes = torch.stack((p >> 4, p & 15), dim=-1).reshape(p.shape[0], cols)  # high nibble = even column
        vals = lut[codes.long()].view(p.shape[0], cols // NVFP4_BLOCK, NVFP4_BLOCK)
        scale = block_scales[i : i + chunk_rows].float() * gscale
        out[i : i + chunk_rows] = (vals * scale.unsqueeze(-1)).view(p.shape[0], cols).to(dtype)
    return out


def rotate(x: torch.Tensor, n: int) -> torch.Tensor:
    """Block regular-Hadamard rotation along the last dim. Applying it twice is the identity."""
    if n <= 1:
        return x
    if x.shape[-1] % n:
        raise ValueError(f"last dim {x.shape[-1]} is not divisible by convrot group size {n}")
    h = regular_hadamard(n, x.device, x.dtype)
    return (x.reshape(*x.shape[:-1], x.shape[-1] // n, n) @ h).reshape(x.shape)


@dataclass
class LoadedWeights:
    tensors: dict[str, torch.Tensor]
    quant: dict[str, QuantLayer] = field(default_factory=dict)  # keyed by layer prefix
    metadata: dict[str, str] = field(default_factory=dict)


def read_header(path: str) -> tuple[list[str], dict[str, str]]:
    with safe_open(path, "pt") as f:
        return list(f.keys()), dict(f.metadata() or {})


def describe_quant(path: str) -> tuple[set[str], set[str]]:
    """Cheaply list the quantization formats in a file: (supported labels, unsupported formats)."""
    labels, unsupported = set(), set()

    def note(conf: dict):
        fmt = conf.get("format", "?")
        if fmt not in SUPPORTED_FORMATS:
            unsupported.add(fmt)
        else:
            labels.add(QuantLayer(fmt, rot_size=256 if conf.get("convrot") else 1).label)

    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        markers = [k for k in keys if k.endswith(".comfy_quant")]
        for k in markers:
            note(_parse_quant_blob(f.get_tensor(k)))
        for conf in _header_quant(dict(f.metadata() or {})).values():
            note(conf)
        if not markers:
            for k in keys:
                if k.endswith((".weight_scale", ".scale_weight")):
                    w = k.rsplit(".", 1)[0] + ".weight"
                    dt = f.get_slice(w).get_dtype() if w in keys else ""
                    labels.add("int8" if dt == "I8" else "fp8 scaled")
                    break
    return labels, unsupported


def _header_quant(metadata: dict[str, str]) -> dict[str, dict]:
    try:
        return json.loads(metadata.get("_quantization_metadata", "{}")).get("layers", {})
    except (json.JSONDecodeError, AttributeError):
        return {}


def _parse_quant_blob(blob: torch.Tensor) -> dict:
    return json.loads(bytes(blob.to(torch.uint8).tolist()).decode("utf-8"))


def load_state_dict(path: str, strip_prefix: str | None = None, skip_prefixes: tuple[str, ...] = (),
                    include: tuple[str, ...] | None = None, only_prefixed: bool = False,
                    device: torch.device | str = "cpu", cast_to=None) -> LoadedWeights:
    """Load a ComfyUI/plain safetensors file, separating quantization sidecars.

    ``strip_prefix`` removes a leading key prefix (e.g. ``"model."``); with
    ``only_prefixed`` keys without it are skipped. ``skip_prefixes`` drops keys
    (after stripping) that the caller doesn't need, without ever reading their
    data. ``include`` keeps only keys starting with one of the given prefixes.
    ``device`` is where tensors are read to (reading straight to the GPU avoids
    holding the whole model in system RAM). ``cast_to(name)`` gives the dtype that fp32/fp16 weights
    are cast to as they're read (biases are left alone), so a full-precision file never sits on the
    device at its stored size.
    """
    if isinstance(device, torch.device):
        device = f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}" \
            if device.type == "cuda" else str(device)
    tensors: dict[str, torch.Tensor] = {}
    sidecars: dict[str, torch.Tensor] = {}
    with safe_open(path, "pt", device=device) as f:
        metadata = dict(f.metadata() or {})
        for key in f.keys():
            name = key
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix):]
            elif strip_prefix and only_prefixed:
                continue
            if skip_prefixes and name.startswith(skip_prefixes):
                continue
            if include is not None and not name.startswith(include):
                continue
            if name in LEGACY_MARKERS or name.endswith(METADATA_SUFFIXES):
                continue
            if name.endswith(SIDECAR_SUFFIXES):
                sidecars[name] = f.get_tensor(key)
            else:
                t = f.get_tensor(key)
                if cast_to is not None and t.dtype in WIDE_DTYPES and not name.endswith(".bias"):
                    t = t.to(cast_to(name))
                tensors[name] = t

    header_quant = _header_quant(metadata)

    quant: dict[str, QuantLayer] = {}
    for name, w in list(tensors.items()):
        prefix = name[: -len(".weight")] if name.endswith(".weight") else None
        marker = sidecars.get(f"{prefix}.comfy_quant") if prefix else None
        declared = prefix in header_quant
        if prefix is None or (w.dtype not in QUANT_DTYPES and marker is None and not declared):
            continue
        conf = _parse_quant_blob(marker) if marker is not None else dict(header_quant.get(prefix, {}))
        default = "int8_tensorwise" if w.dtype == torch.int8 else str(w.dtype).replace("torch.", "")
        fmt = conf.get("format", default)
        if fmt not in SUPPORTED_FORMATS:
            raise NotImplementedError(
                f"{path}: layer {prefix} uses quantization format {fmt!r}, which Dainamo can't read yet "
                f"(supported: {', '.join(SUPPORTED_FORMATS)})"
            )
        scale = sidecars.get(f"{prefix}.weight_scale", sidecars.get(f"{prefix}.scale_weight"))
        if fmt == "int8_tensorwise" and scale is None:
            raise ValueError(f"{path}: int8 layer {prefix} has no weight_scale")
        if fmt == "nvfp4":
            pts = sidecars.get(f"{prefix}.weight_scale_2")
            if scale is None or pts is None or w.dtype != torch.uint8:
                raise ValueError(f"{path}: nvfp4 layer {prefix} needs packed uint8 weights, weight_scale and weight_scale_2")
            rows, cols = w.shape[0], w.shape[1] * 2 // NVFP4_BLOCK
            quant[prefix] = QuantLayer(
                format=fmt,
                weight_scale=unswizzle_nvfp4_scales(scale.view(torch.float8_e4m3fn), rows, cols),
                weight_scale_2=pts.float(),
                pre_quant_scale=sidecars.get(f"{prefix}.pre_quant_scale"),
            )
            continue
        if scale is not None and scale.numel() not in (1, w.shape[0]):
            raise NotImplementedError(
                f"{path}: layer {prefix} has a {tuple(scale.shape)} scale (block-wise scaling isn't supported yet)"
            )
        quant[prefix] = QuantLayer(
            format=fmt,
            weight_scale=scale,
            input_scale=sidecars.get(f"{prefix}.input_scale", sidecars.get(f"{prefix}.scale_input")),
            rot_size=int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1,
        )
    return LoadedWeights(tensors=tensors, quant=quant, metadata=metadata)


def dequantize(weight: torch.Tensor, q: QuantLayer | None, dtype: torch.dtype,
               chunk_rows: int = 8192) -> torch.Tensor:
    """Real weight in ``dtype``. Scaling and the un-rotation run in fp32 for accuracy, in row chunks
    so a large tensor (e.g. a 248k-token embedding) never has a full-size fp32 copy. An AWQ
    pre_quant_scale is folded into the columns, so the result works on raw activations."""
    if q is None:
        return weight.to(dtype)
    if q.format == "nvfp4":
        w = nvfp4_dequantize(weight, q.weight_scale.to(weight.device), q.weight_scale_2, torch.float32)
        if q.pre_quant_scale is not None:
            w = w * q.pre_quant_scale.to(w.device, torch.float32).unsqueeze(0)
        return w.to(dtype)
    if weight.dim() >= 2 and weight.shape[0] > chunk_rows:
        out = torch.empty(weight.shape, dtype=dtype, device=weight.device)
        per_row = q.weight_scale is not None and q.weight_scale.numel() > 1
        for i in range(0, weight.shape[0], chunk_rows):
            part = QuantLayer(q.format, q.weight_scale[i : i + chunk_rows] if per_row else q.weight_scale,
                              rot_size=q.rot_size)
            out[i : i + chunk_rows] = dequantize(weight[i : i + chunk_rows], part, dtype, chunk_rows)
        return out
    w = weight.float()
    if q.weight_scale is not None:
        scale = q.weight_scale.to(device=w.device, dtype=torch.float32)
        scale = scale.reshape(-1, *([1] * (w.dim() - 1))) if scale.numel() > 1 else scale.reshape(())
        w = w * scale
    return rotate(w, q.rot_size).to(dtype)


def dequantize_all(loaded: LoadedWeights, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Return a plain state dict with every quantized weight expanded to ``dtype``."""
    out = {}
    for name, t in loaded.tensors.items():
        prefix = name[: -len(".weight")] if name.endswith(".weight") else None
        if prefix in loaded.quant:
            out[name] = dequantize(t, loaded.quant[prefix], dtype)
        elif t.is_floating_point():
            out[name] = t.to(dtype)
        else:
            out[name] = t
    return out

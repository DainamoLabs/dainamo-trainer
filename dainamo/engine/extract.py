"""A LoRA extracted from a finetune: the difference between a finetuned diffusion model and the model it was
finetuned from, saved as a file that applies that difference the way a trained LoRA applies its own.

Each weight matrix's change becomes a LoRA of the chosen rank, its best approximation of that rank: the change's
top singular directions, found as the eigenvectors of its smaller Gram matrix (the same factors an SVD gives, in a
fraction of the time on a GPU). A matrix too small for a LoRA of that rank to be any smaller is kept whole, as is
every other tensor that changed: norm scales (``.diff``) and biases (``.diff_b``), which take little room. Those
are key types ComfyUI's LoRA loader applies beside ``lora_A`` / ``lora_B``, so the file loads there as any LoRA
does, and the Sandbox applies all of them (``lora.AdapterStack``).

The two files are read a block at a time to the GPU, so neither model is ever in memory whole. The finetuned
file's tensors are compared with the original's as training started from them: a quantized original (int8, fp8)
expanded to the finetuned tensor's dtype, as ``load_transformer(trainable=True)`` expanded it, so the difference
is what training changed and not the quantization's rounding. Keys are compared as the files name them, so the
finetuned file has to be laid out like the original (``finetune.save_model`` lays it out so).
"""

from __future__ import annotations

import math
import os

import torch

from . import comfy_io
from .families.base import block_index, transformer_shapes
from .finetune import write_safetensors

LORA_PREFIX = "diffusion_model."  # as trained LoRAs are saved (``LoRANetwork``)
RANKS = (16, 32, 64, 128, 256, 512)  # what the form offers
MAX_RANK = 1024
SMALL = 1 << 20  # a whole change with up to this many values is kept in fp32, a larger one in bf16
# far more than a finetune moves the weights (about 1% after thousands of steps at 1e-5): another model
WRONG_BASE_CHANGE = 0.2


def low_rank(d: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(up, down, kept) for a change ``d`` (out, in) in fp32: ``up`` (out, r) @ ``down`` (r, in) is d's best
    approximation of rank r (``rank``, or less for a smaller matrix), each factor carrying the square root of the
    singular values, and ``kept`` the share of d's squared size it holds."""
    out_f, in_f = d.shape
    r = min(rank, out_f, in_f)
    tall = out_f > in_f
    vals, vecs = torch.linalg.eigh(d.T @ d if tall else d @ d.T)  # ascending
    vals, vecs = vals[-r:].flip(0).clamp_min(0), vecs[:, -r:].flip(1)
    s = vals.sqrt()
    # directions with no change in them (a change of lower rank than r) get zero factors
    live = s > s[0] * 1e-4 if s[0] > 0 else torch.zeros_like(s, dtype=torch.bool)
    root = torch.where(live, s.sqrt(), torch.zeros_like(s))
    inv = torch.where(live, 1 / root.clamp_min(1e-30), torch.zeros_like(s))
    if tall:  # vecs are its right singular vectors
        up, down = (d @ vecs) * inv, (vecs * root).T
    else:  # its left ones
        up, down = vecs * root, (vecs.T @ d) * inv[:, None]
    total = float((d * d).sum())
    kept = min(1.0, float(vals.sum()) / total) if total > 0 else 1.0
    return up, down, kept


def lora_bytes(out_f: int, in_f: int, rank: int) -> int:
    """What a weight matrix's change takes in the file at ``rank``: a LoRA, or the whole change if that's smaller."""
    r = min(rank, out_f, in_f)
    return 2 * min(r * (out_f + in_f), out_f * in_f)


def sizes(path: str, ranks=RANKS) -> dict[int, int]:
    """About how big a LoRA extracted from the finetuned model ``path`` is at each rank, by its header: every
    weight matrix as a LoRA (or whole, if smaller), every other tensor whole in fp32, as if all of it changed."""
    _, shapes = transformer_shapes(path)
    matrices, other = [], 0
    for k, shape in shapes.items():
        if k.endswith(comfy_io.SIDECAR_SUFFIXES) or k.endswith(comfy_io.METADATA_SUFFIXES):
            continue
        if k.endswith(".weight") and len(shape) == 2:
            matrices.append(shape)
        else:
            other += math.prod(shape)
    return {r: sum(lora_bytes(o, i, r) for o, i in matrices) + 4 * other for r in ranks}


def _read(path: str, prefix: str, chunk: str | None, block_list: str, device) -> comfy_io.LoadedWeights:
    """One block's tensors of a file (``chunk``: "<block_list>.<i>."), or (None) all but the blocks'."""
    opts = {"strip_prefix": prefix or None, "only_prefixed": bool(prefix), "device": device}
    if chunk is None:
        return comfy_io.load_state_dict(path, skip_prefixes=(block_list + ".",), **opts)
    return comfy_io.load_state_dict(path, include=(chunk,), **opts)


def _real(loaded: comfy_io.LoadedWeights, name: str, dtype: torch.dtype) -> torch.Tensor:
    """A tensor of ``loaded`` as the model holds it in ``dtype`` (a quantized weight expanded)."""
    q = loaded.quant.get(name[: -len(".weight")]) if name.endswith(".weight") else None
    t = loaded.tensors[name]
    return comfy_io.dequantize(t, q, dtype) if q is not None else t.to(dtype)


def extract(base: str, tuned: str, out: str, rank: int, block_list: str, device, metadata: dict | None = None,
            log=print, progress=None, check=None) -> dict:
    """Save the change from ``base`` to ``tuned`` as a LoRA of ``rank`` at ``out``, comparing the files a block of
    ``block_list`` at a time on ``device``. ``progress(done, total)`` follows the parts compared; ``check()`` may
    raise to stop. Returns what it saved: how many matrices became LoRAs (``layers``) and how many tensors were
    kept whole (``whole``) or hadn't changed, the share of the change's squared size the file keeps (``kept``),
    the layer that keeps least of its change (``worst``), the change's size relative to the weights' (``change``),
    and the file's size."""
    if not 1 <= int(rank) <= MAX_RANK:
        raise ValueError(f"The rank must be 1–{MAX_RANK}")
    bp, bshapes = transformer_shapes(base)
    tp, tshapes = transformer_shapes(tuned)
    if not any(k in bshapes for k in tshapes):
        raise ValueError(f"{os.path.basename(tuned)} and {os.path.basename(base)} have no tensors in common: "
                         "they aren't the same kind of model")
    blocks = sorted({block_index(k, block_list) for k in tshapes if k.startswith(block_list + ".")})
    chunks: list[str | None] = [f"{block_list}.{i}." for i in blocks] + [None]
    tensors: dict[str, torch.Tensor] = {}
    made = {"rank": int(rank), "layers": 0, "whole": 0, "unchanged": 0, "missing": 0}
    total = kept = changed = weights = 0.0
    worst: tuple[float, str] | None = None
    for n, chunk in enumerate(chunks):
        if check:
            check()
        b = _read(base, bp, chunk, block_list, device)
        t = _read(tuned, tp, chunk, block_list, device)
        for name in sorted(t.tensors):
            quantized = name.endswith(".weight") and name[: -len(".weight")] in t.quant
            if not (t.tensors[name].is_floating_point() or quantized):
                continue
            if name not in b.tensors:
                made["missing"] += 1
                continue
            dtype = torch.bfloat16 if quantized else t.tensors[name].dtype
            new, old = _real(t, name, dtype), _real(b, name, dtype)
            if old.shape != new.shape:
                raise ValueError(f"{name} is {tuple(old.shape)} in {os.path.basename(base)} but {tuple(new.shape)} "
                                 f"in {os.path.basename(tuned)}: they aren't the same model")
            if torch.equal(old, new):
                made["unchanged"] += 1
                continue
            d = new.float() - old.float()
            size = float((d * d).sum())
            total += size
            stem = name[: -len(".weight")] if name.endswith(".weight") else name
            if name.endswith(".weight") and d.dim() == 2:
                changed += size
                weights += float(old.float().pow(2).sum())
                out_f, in_f = d.shape
                r = min(int(rank), out_f, in_f)
                key = LORA_PREFIX + stem
                if r * (out_f + in_f) >= out_f * in_f:  # a LoRA wouldn't be smaller: the whole change, exactly
                    tensors[key + ".diff"] = d.to("cpu", torch.bfloat16)
                    made["whole"] += 1
                    kept += size
                    continue
                up, down, share = low_rank(d, int(rank))
                tensors[key + ".lora_A.weight"] = down.to("cpu", torch.bfloat16).contiguous()
                tensors[key + ".lora_B.weight"] = up.to("cpu", torch.bfloat16).contiguous()
                tensors[key + ".alpha"] = torch.tensor(float(r))  # alpha / rank = 1: the change as it is
                made["layers"] += 1
                kept += share * size
                worst = min(worst, (share, stem)) if worst else (share, stem)
                continue
            whole = d.to("cpu", torch.float32 if d.numel() <= SMALL else torch.bfloat16)
            if name.endswith(".bias") and f"{name[: -len('.bias')]}.weight" in tshapes:
                tensors[LORA_PREFIX + name[: -len(".bias")] + ".diff_b"] = whole
            else:  # a norm's scale, or a tensor named as itself (ComfyUI maps those keys as they are)
                tensors[LORA_PREFIX + stem + ".diff"] = whole
            made["whole"] += 1
            kept += size
        del b, t
        if progress:
            progress(n + 1, len(chunks))
    if not total:
        raise ValueError(f"{os.path.basename(tuned)} is the same as {os.path.basename(base)}: there's no change to "
                         "extract")
    made.update(kept=round(kept / total, 4), change=round(math.sqrt(changed / weights), 5) if weights else 0.0,
                worst={"name": worst[1], "kept": round(worst[0], 4)} if worst else None)
    if made["change"] > WRONG_BASE_CHANGE:
        log(f"Warning: the weights differ by {made['change']:.0%}, far more than a finetune changes them. Check that "
            f"{os.path.basename(base)} is the model {os.path.basename(tuned)} was finetuned from.")
    if made["missing"]:
        log(f"Note: {made['missing']} tensor(s) of the finetuned model aren't in the original, so they're left out")
    write_safetensors(out, list(tensors.items()), {str(k): str(v) for k, v in (metadata or {}).items()}
                      | {"dainamo.kept": str(made["kept"])})
    made["bytes"] = os.path.getsize(out)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return made

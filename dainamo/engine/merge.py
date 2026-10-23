"""LoRAs merged into one: what several LoRAs (or LoKrs) do together, each at its own strength, as one LoRA file that
ComfyUI and the Sandbox load like any other. With one input, it's that LoRA cut to a smaller rank.

Each layer's change is the sum of the inputs' changes there, times their strengths. Where every input is a LoRA, the
sum is itself a LoRA, of their ranks added up: their down matrices stacked, and their up matrices side by side, each
scaled by its own alpha / rank and strength. That's exact, and the file is about as big as the inputs together. With
a rank given, a layer whose stacked rank is higher is cut to that rank, to its best approximation of that rank: from
a QR factor of each side and the SVD of the small matrix between them, so the whole change is never built. A LoKr's
change, or a whole one (``.diff``) on a weight, has no rank to stack: those layers are built whole and cut to the
rank (``extract.low_rank``), so merging a LoKr needs one. A layer whose LoRA wouldn't be smaller than its whole
change is saved whole, as an extracted LoRA's are, and changes to norms and biases are added up whole.

Layer names are checked against the family's model, built from the profile's diffusion-model file without its
weights (``model_layers``); kohya-style inputs (``lora_unet_...``) are resolved against it too. The inputs are read a
layer at a time, so they're never in memory whole, and the file is written in the layout trained LoRAs are
(``diffusion_model.<module>``).
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

from .extract import LORA_PREFIX, MAX_RANK, RANKS, SMALL, low_rank
from .families.base import transformer_shapes
from .finetune import write_safetensors
from .lora import KOHYA_PREFIXES, LORA_PREFIXES, adapter_layers, from_kohya

MAX_INPUTS = 8
NO_MATCH = "matches no layers of this model: it may be for another model, or name its layers another way (diffusers')"
RANKS = (4, 8, *RANKS)  # what the form offers: an extract's, and the small ones LoRAs are often trained at


@dataclass
class Layers:
    """A family's model as merging sees it: its Linears' (out, in) sizes, every tensor's shape, and the Linears'
    names as kohya-style files spell them (``lora.linear_names``)."""
    linears: dict[str, tuple[int, int]]
    tensors: dict[str, tuple[int, ...]]
    kohya: dict[str, str]


def model_layers(driver_cls, profile: dict) -> Layers:
    """The model of ``profile``'s diffusion-model file, built on the meta device (no weights are read)."""
    _, shapes = transformer_shapes(profile["diffusion_model"])
    model = driver_cls(dict(profile), torch.device("cpu"), log=lambda *_: None).build_transformer(shapes)
    linears = {n: (m.out_features, m.in_features) for n, m in model.named_modules() if hasattr(m, "in_features")}
    tensors = {n: tuple(t.shape) for n, t in model.state_dict().items()}
    return Layers(linears, tensors, {n.replace(".", "_"): n for n in linears})


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _slot(name: str, kind: str, shape, layers: Layers) -> str | None:
    """Where an input's change on ``name`` goes: a Linear's weight ("weight"), a bias ("bias"), or another tensor
    ("tensor": a norm's scale, a modulation), or None if the model has no such layer of that size. ``shape`` is
    (out, in) for a LoRA, the change's shape for a whole one, None for a LoKr (checked once it's built)."""
    if kind == "diff_b":
        bias = layers.tensors.get(f"{name}.bias")
        return "bias" if bias is not None and _numel(bias) == _numel(shape) else None
    if name in layers.linears and (kind != "diff" or len(shape) == 2):
        return "weight" if shape is None or tuple(shape) == layers.linears[name] else None
    if kind == "diff":
        t = layers.tensors.get(f"{name}.weight", layers.tensors.get(name))
        return "tensor" if t is not None and _numel(t) == _numel(shape) else None
    return None


def _stem(key: str) -> str:
    """The layer a key of a LoRA file is for, as the file names it."""
    for s in (".lora_A.weight", ".lora_B.weight"):
        if key.endswith(s):
            return key[: -len(s)]
    return key.rpartition(".")[0]


def _index(f, layers: Layers) -> dict[str, dict[str, str]]:
    """{layer: {key as merging reads it: key in the file}} for an open file, kohya-style keys resolved."""
    keys = {k: k for k in f.keys()}
    if any(k.startswith(KOHYA_PREFIXES) for k in keys):
        keys = from_kohya(keys, layers.kohya)
    out: dict[str, dict[str, str]] = {}
    for k, src in keys.items():
        out.setdefault(_stem(k), {})[k] = src
    return out


def _bare(stem: str) -> str:
    return next((stem[len(p):] for p in LORA_PREFIXES if stem.startswith(p)), stem)


def describe(path: str, layers: Layers) -> dict:
    """What a LoRA file holds, from its header: how many layers it changes and how many of them this model has (at
    that size), the kinds of change and LoRA ranks among those, and its trigger words. ``changes`` is
    {(name, slot): (kind, rank of a LoRA or values of a whole change)}."""
    with safe_open(path, "pt") as f:
        index = _index(f, layers)
        shape = {k: tuple(f.get_slice(src).get_shape()) for keys in index.values() for k, src in keys.items()}
        meta = f.metadata() or {}
    if any(k.endswith(".dora_scale") for k in shape):
        raise ValueError(f"{os.path.basename(path)} is a DoRA, which can't be merged")
    found = []  # (name, kind, shape for _slot, rank or values)
    for stem, keys in index.items():
        name = _bare(stem)
        if f"{stem}.lora_A.weight" in keys and f"{stem}.lora_B.weight" in keys:
            down, up = shape[f"{stem}.lora_A.weight"], shape[f"{stem}.lora_B.weight"]
            found.append((name, "lora", (up[0], down[1]), down[0]))
        if f"{stem}.lokr_w1" in keys or f"{stem}.lokr_w1_a" in keys:
            found.append((name, "lokr", None, None))
        for kind in ("diff", "diff_b"):
            if f"{stem}.{kind}" in keys:
                s = shape[f"{stem}.{kind}"]
                found.append((name, kind, s, _numel(s)))
    changes = {}
    for name, kind, s, n in found:
        slot = _slot(name, kind, s, layers)
        if slot is not None:
            changes[(name, slot)] = (kind, n)
    ranks = [n for kind, n in changes.values() if kind == "lora"]
    return {"path": path, "file": os.path.basename(path), "bytes": os.path.getsize(path), "layers": len(found),
            "matched": len(changes), "kinds": sorted({kind for kind, _ in changes.values()}),
            "ranks": [min(ranks), max(ranks)] if ranks else [], "trigger": meta.get("dainamo.trigger", ""),
            "changes": changes}


def _whole_bytes(n: int, weight: bool) -> int:
    return n * (2 if weight or n > SMALL else 4)  # as ``merge`` saves a whole change


def plan(paths: list[str], layers: Layers, ranks=RANKS) -> dict:
    """``describe`` of each input (without its changes), and about how big the merge is: kept exact
    (``sizes["0"]``, None if a LoKr makes that impossible) and cut to each rank."""
    inputs = [describe(p, layers) for p in paths]
    weights: dict[str, list] = {}
    whole: dict[tuple[str, str], int] = {}
    for d in inputs:
        for (name, slot), (kind, n) in d["changes"].items():
            if slot == "weight":
                weights.setdefault(name, []).append((kind, n))
            else:
                whole[(name, slot)] = n
    lokr = any("lokr" in d["kinds"] for d in inputs)

    def size(rank: int) -> int | None:
        if not rank and lokr:
            return None
        total = sum(_whole_bytes(n, False) for n in whole.values())
        for name, changes in weights.items():
            o, i = layers.linears[name]
            if all(kind == "lora" for kind, _ in changes):
                stacked = sum(n for _, n in changes)
                r = min(stacked, rank) if rank else stacked
            else:
                r = rank or o * i
            total += 2 * min(r * (o + i), o * i)
        return total

    return {"inputs": [{k: v for k, v in d.items() if k != "changes"} for d in inputs],
            "sizes": {str(r): size(r) for r in (0, *ranks)}, "lokr": lokr}


def merge(loras: list[tuple[str, float]], out: str, rank: int, layers: Layers, device, metadata: dict | None = None,
          progress=None, check=None) -> dict:
    """Save ``loras`` [(path, strength)] merged at ``out``: kept exact (``rank`` 0) or each layer cut to at most
    ``rank``. ``progress(done, total)`` follows the layers; ``check()`` may raise to stop. Returns what it saved:
    layers as LoRAs (``layers``, ``cut`` of them to the rank) and kept whole (``whole``), the share of the merged
    change's squared size the file keeps (``kept``), the layer that keeps least (``worst``), each input's matched
    layers, and the file's size."""
    rank = int(rank)
    if not 0 <= rank <= MAX_RANK:
        raise ValueError(f"The rank must be 1–{MAX_RANK}, or 0 to keep every rank")
    if not 1 <= len(loras) <= MAX_INPUTS:
        raise ValueError(f"Merge 1–{MAX_INPUTS} LoRAs")
    inputs = []
    for path, strength in loras:
        d = describe(path, layers)
        if not d["matched"]:
            raise ValueError(f"{d['file']} {NO_MATCH}")
        if not rank and "lokr" in d["kinds"]:
            raise ValueError(f"{d['file']} is a LoKr, whose change has no rank to stack, so it can't be merged "
                             "exactly: choose a rank")
        inputs.append({"file": d["file"], "strength": float(strength), "matched": d["matched"],
                       "layers": d["layers"]})

    tensors: dict[str, torch.Tensor] = {}
    made = {"rank": rank, "layers": 0, "cut": 0, "whole": 0}
    total = kept = 0.0
    worst: tuple[float, str] | None = None
    with contextlib.ExitStack() as stack:
        files = [(stack.enter_context(safe_open(p, "pt", device="cpu")), float(s)) for p, s in loras]
        indexes = [_index(f, layers) for f, _ in files]
        stems = sorted({stem for index in indexes for stem in index})
        for n, stem in enumerate(stems):
            if check and n % 16 == 0:
                check()
            parts: dict[tuple[str, str], list] = {}
            for (f, strength), index in zip(files, indexes):
                keys = index.get(stem)
                if not keys:
                    continue
                for name, a in adapter_layers({k: f.get_tensor(src) for k, src in keys.items()}):
                    kind = ("diff_b" if a.bias is not None else "diff" if a.full is not None
                            else "lokr" if a.w1 is not None else "lora")
                    shape = ((a.up.shape[0], a.down.shape[1]) if kind == "lora" else None if kind == "lokr"
                             else tuple((a.bias if kind == "diff_b" else a.full).shape))
                    slot = _slot(name, kind, shape, layers)
                    if slot is not None:
                        parts.setdefault((name, slot), []).append((a.to(device, torch.float32), strength))
            for (name, slot), fs in parts.items():
                key = LORA_PREFIX + name
                if slot != "weight":  # a bias's or a norm's change, added up whole
                    v = sum(s * a.dense() for a, s in fs)
                    tensors[key + (".diff_b" if slot == "bias" else ".diff")] = \
                        v.to("cpu", torch.float32 if v.numel() <= SMALL else torch.bfloat16)
                    size = float((v * v).sum())
                    total, kept = total + size, kept + size
                    made["whole"] += 1
                    continue
                o, i = layers.linears[name]
                d = up = down = None
                share = 1.0
                if all(a.w1 is None and a.full is None for a, _ in fs):  # LoRAs: stacked
                    up = torch.cat([a.up * (a.scale * s) for a, s in fs], 1)
                    down = torch.cat([a.down for a, _ in fs], 0)
                    qu, ru = torch.linalg.qr(up)
                    qd, rd = torch.linalg.qr(down.T)
                    core = ru @ rd.T  # up @ down == qu @ core @ qd.T
                    size = float((core * core).sum())
                    r = min(rank, up.shape[1]) if rank else up.shape[1]
                    if r * (o + i) >= o * i:  # no smaller as a LoRA: the whole change, exactly
                        d = up @ down
                    elif r < up.shape[1]:
                        w, sv, vh = torch.linalg.svd(core, full_matrices=False)
                        root = sv[:r].sqrt()
                        up, down = qu @ (w[:, :r] * root), (root[:, None] * vh[:r]) @ qd.T
                        share = float((sv[:r] ** 2).sum()) / size if size > 0 else 1.0
                        made["cut"] += 1
                else:  # a LoKr's or a whole change: built whole, then cut to the rank
                    d = sum(s * a.dense() for a, s in fs)
                    if tuple(d.shape) != (o, i):
                        raise ValueError(f"A change to {name} is {tuple(d.shape)}, but the model's layer is "
                                         f"{(o, i)}: is a LoRA for another model?")
                    size = float((d * d).sum())
                    if rank and rank * (o + i) < o * i:
                        up, down, share = low_rank(d, rank)
                        d = None
                        made["cut"] += 1
                total += size
                kept += share * size
                if d is not None:
                    tensors[key + ".diff"] = d.to("cpu", torch.bfloat16)
                    made["whole"] += 1
                    continue
                tensors[key + ".lora_A.weight"] = down.to("cpu", torch.bfloat16).contiguous()
                tensors[key + ".lora_B.weight"] = up.to("cpu", torch.bfloat16).contiguous()
                tensors[key + ".alpha"] = torch.tensor(float(down.shape[0]))  # alpha / rank = 1: the sum as it is
                made["layers"] += 1
                if share < 1.0 and (worst is None or share < worst[0]):
                    worst = (share, name)
            if progress:
                progress(n + 1, len(stems))
    made.update(kept=round(kept / total, 4) if total else 1.0, inputs=inputs,
                worst={"name": worst[1], "kept": round(worst[0], 4)} if worst else None)
    write_safetensors(out, list(tensors.items()), {str(k): str(v) for k, v in (metadata or {}).items()}
                      | {"dainamo.kept": str(made["kept"])})
    made["bytes"] = os.path.getsize(out)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return made

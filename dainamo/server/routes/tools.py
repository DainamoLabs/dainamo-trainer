"""Merge & extract (a tab of the Profiles page): a LoRA extracted from any finetuned model, or LoRAs merged into one.
Each is queued as a job of its own, as a LoRA extracted from a finetune job's checkpoint is (``jobs.extract_lora``),
so it gets the same progress, its own folder, and a place in the Sandbox's LoRA list."""

from __future__ import annotations

import math
import os
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...engine.families import get_family
from .common import profile_or_404
from .jobs import queue_made

router = APIRouter(prefix="/api/tools")


def _file(path: str, what: str) -> str:
    p = os.path.expanduser(str(path or "").strip())
    if not (os.path.isfile(p) and p.endswith(".safetensors")):
        raise HTTPException(400, f"{what} not found (or not .safetensors): {p or 'none chosen'}")
    return os.path.abspath(p)


def _base(profile: dict, base: str | None) -> str:
    """The model a LoRA is extracted as the change from: ``base``, else the profile's diffusion model."""
    if base and base.strip():
        return _file(base, "The original model")
    p = profile["data"].get("diffusion_model") or ""
    if not os.path.isfile(p):
        raise HTTPException(400, f"The profile's diffusion model, {p or 'not set'}, isn't there: choose the model "
                                 "it was finetuned from")
    return p


def _name(name: str, default: str) -> str:
    return name.strip() or default[:80]


@router.get("/extract/sizes")
def extract_sizes(profile_id: str, path: str, base: str = ""):
    """For the Extract form: about how big a LoRA extracted from the finetuned model ``path`` is at each rank, and
    the original it's the change from (``base``, else the profile's diffusion model)."""
    from ...engine.extract import sizes

    profile = profile_or_404(profile_id)
    p = _file(path, "The finetuned model")
    try:
        b = _base(profile, base)
    except HTTPException as e:
        b, found, why = profile["data"].get("diffusion_model") or "", False, e.detail
    else:
        found, why = True, ""
    try:
        made = sizes(p)
    except Exception as e:  # not a model's file
        raise HTTPException(400, f"Can't read {os.path.basename(p)}: {e}")
    return {"sizes": {str(r): n for r, n in made.items()}, "base": b, "base_found": found, "base_error": why}


class ExtractIn(BaseModel):
    profile_id: str
    tuned: str  # the finetuned model
    base: str = ""  # the model it was finetuned from (default: the profile's diffusion model)
    rank: int = 128
    name: str = ""
    trigger: str = ""
    gpu: str = "auto"


@router.post("/extract")
def extract(body: ExtractIn):
    """Queue a job that extracts a LoRA from a finetuned model (``Trainer.run_extract``): its change from the model
    it was finetuned from."""
    from ...engine.extract import MAX_RANK

    profile = profile_or_404(body.profile_id)
    tuned, base = _file(body.tuned, "The finetuned model"), _base(profile, body.base)
    if os.path.samefile(tuned, base):
        raise HTTPException(400, "The finetuned model and the original are the same file")
    if not 1 <= body.rank <= MAX_RANK:
        raise HTTPException(400, f"The rank must be 1–{MAX_RANK}")
    file = os.path.basename(tuned)
    name = _name(body.name, f"{os.path.splitext(file)[0]}_r{body.rank}")
    settings = {"steps": 0, "extract_enabled": True, "extract_rank": int(body.rank), "extract_source": tuned,
                "extract_base": base, "extract_from": {"file": file, "step": 0, "trigger": body.trigger.strip()}}
    return queue_made(name, profile["family"], profile["name"], profile["data"], settings, body.gpu)


# the profile's model as merging checks layers against, built from its file's header: by (family, path, mtime)
_layers: dict[tuple, object] = {}
_layers_lock = threading.Lock()


def _model_layers(profile: dict):
    from ...engine import merge

    path = profile["data"].get("diffusion_model") or ""
    if not os.path.isfile(path):
        raise HTTPException(400, f"The profile's diffusion model, {path or 'not set'}, isn't there; merging checks "
                                 "the LoRAs' layers against it")
    key = (profile["family"], path, os.path.getmtime(path))
    with _layers_lock:
        if key not in _layers:
            try:
                _layers.clear()
                _layers[key] = merge.model_layers(get_family(profile["family"]), profile["data"])
            except Exception as e:
                raise HTTPException(400, f"Can't read the profile's diffusion model: {e}")
        return _layers[key]


def _plan(profile: dict, paths: list[str]) -> dict:
    from ...engine import merge

    if not 1 <= len(paths) <= merge.MAX_INPUTS:
        raise HTTPException(400, f"Choose 1–{merge.MAX_INPUTS} LoRAs")
    if len(set(paths)) < len(paths):
        raise HTTPException(400, "A LoRA is in the list twice: give it one strength instead")
    layers = _model_layers(profile)
    try:
        return merge.plan(paths, layers)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # not a LoRA's file
        raise HTTPException(400, f"Can't read the LoRAs: {e}")


class MergeCheckIn(BaseModel):
    profile_id: str
    loras: list[str]


@router.post("/merge/check")
def merge_check(body: MergeCheckIn):
    """For the Merge form: what each LoRA holds and how much of it this profile's model has, and about how big the
    merge is kept exact (``sizes["0"]``; None with a LoKr, which needs a rank) and at each rank."""
    profile = profile_or_404(body.profile_id)
    return _plan(profile, [_file(p, "LoRA") for p in body.loras])


class MergeLora(BaseModel):
    path: str
    strength: float = 1.0


class MergeIn(BaseModel):
    profile_id: str
    loras: list[MergeLora]
    rank: int = 0  # 0: exact, every rank kept
    name: str = ""
    gpu: str = "auto"


@router.post("/merge")
def merge(body: MergeIn):
    """Queue a job that merges LoRAs into one (``Trainer.run_merge``), each at its strength: kept exact, or cut to
    ``rank``."""
    from ...engine.merge import MAX_RANK, NO_MATCH

    profile = profile_or_404(body.profile_id)
    paths = [_file(e.path, "LoRA") for e in body.loras]
    for e in body.loras:
        if not math.isfinite(e.strength) or not -10 <= e.strength <= 10 or e.strength == 0:
            raise HTTPException(400, "Each strength must be a number from -10 to 10, and not 0")
    if not 0 <= body.rank <= MAX_RANK:
        raise HTTPException(400, f"The rank must be 1–{MAX_RANK}, or 0 to keep every rank")
    plan = _plan(profile, paths)
    for i in plan["inputs"]:
        if not i["matched"]:
            raise HTTPException(400, f"{i['file']} {NO_MATCH}")
    if plan["lokr"] and not body.rank:
        raise HTTPException(400, "A LoKr's change has no rank to stack, so it can't be merged exactly: choose a rank")
    stems = [os.path.splitext(i["file"])[0] for i in plan["inputs"]]
    name = _name(body.name, ("+".join(stems) if len(stems) > 1 else stems[0])
                 + (f"_r{body.rank}" if body.rank else "" if len(stems) > 1 else "_copy"))
    settings = {"steps": 0, "merge_enabled": True, "merge_rank": int(body.rank),
                "merge_loras": [{"path": p, "strength": float(e.strength), "trigger": i["trigger"]}
                                for p, e, i in zip(paths, body.loras, plan["inputs"])]}
    return queue_made(name, profile["family"], profile["name"], profile["data"], settings, body.gpu)

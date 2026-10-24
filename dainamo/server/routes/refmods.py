"""Making RefMods in the Sandbox, of chosen images and clips or a sheet of images (``engine.refmod``). The Sandbox's
worker makes them with the VAE it has loaded, after the generations queued before, so trying one costs no model
load. Each is kept in data/sandbox/refmods/<id>/: the file, its preview, the sheet, and made.json."""

from __future__ import annotations

import io
import json
import os
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ...engine.families import get_family
from .. import hardware
from .. import sandbox as sbx
from ..validation import REFMOD_DEFAULTS, check_refmod
from .common import not_found, profile_or_404

router = APIRouter(prefix="/api/sandbox/refmod")

REFMODS_DIR = os.path.join(sbx.SANDBOX_DIR, "refmods")
MAX_REFERENCES = 64
SHEET_PREVIEW = 960  # the sheet preview's long edge, in pixels
GB = 1024**3


class RefModIn(BaseModel):
    profile_id: str
    paths: list[str] = []
    sheet: bool = False  # the images on one sheet, as one reference
    name: str = ""
    settings: dict = {}  # refmod_mode, _size, _frames, _grid, _refine, _merge, _max_tokens, _concept, _description, _copy_to
    gpu: int | None = None  # where to make it if the Sandbox isn't running (else its GPU)


def made_dir(mid: str) -> str:
    return os.path.join(REFMODS_DIR, os.path.basename(mid))


def made_refmods(family: str) -> list[dict]:
    """The RefMods the Sandbox made for ``family``: each one's made.json, with its id and where it is now."""
    out = []
    for mid in os.listdir(REFMODS_DIR) if os.path.isdir(REFMODS_DIR) else []:
        try:
            with open(os.path.join(made_dir(mid), "made.json"), encoding="utf-8") as f:
                made = json.load(f)
        except (OSError, ValueError):
            continue
        path = os.path.join(made_dir(mid), made.get("file") or "")
        if made.get("family") == family and made.get("file") and os.path.exists(path):
            out.append({**made, "id": mid, "path": path})
    return out


def _name(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("._-")[:80]


def check(body: RefModIn) -> tuple[dict, type, list[str], dict, str]:
    """-> (profile, its family's driver class, the files, the settings checked, the file name); 400 if any is
    wrong."""
    from ...engine import dataset as ds
    from ...engine.refmod import SHEET_MAX

    profile = profile_or_404(body.profile_id)
    F = get_family(profile["family"])
    if not F.info.refmods:
        raise HTTPException(400, f"{F.info.label} doesn't take RefMods")
    paths = [os.path.expanduser(p.strip()) for p in body.paths if p and p.strip()]
    if not paths:
        raise HTTPException(400, "Add the images or clips to make it of")
    if len(paths) > MAX_REFERENCES:
        raise HTTPException(400, f"Up to {MAX_REFERENCES} images and clips")
    for p in paths:
        if ds.media_kind(p) not in ("image", "video"):
            raise HTTPException(400, f"{os.path.basename(p)} isn't an image or a clip")
        if not os.path.isfile(p):
            raise HTTPException(400, f"Not found: {p}")
    if body.sheet:
        if any(ds.media_kind(p) != "image" for p in paths):
            raise HTTPException(400, "Only images go on a sheet: remove the clips, or turn the sheet off")
        if not 2 <= len(paths) <= SHEET_MAX:
            raise HTTPException(400, f"A sheet takes 2–{SHEET_MAX} images")
    s = {**REFMOD_DEFAULTS, **{k: v for k, v in body.settings.items() if k in REFMOD_DEFAULTS},
         "refmod_enabled": True, "refmod_kind": "visual"}
    check_refmod(s)
    stem = os.path.splitext(os.path.basename(paths[0]))[0]
    name = _name(body.name.strip().removesuffix(".safetensors")) or _name(stem + ("_sheet" if body.sheet else "_refmod"))
    return profile, F, paths, s, name or "refmod"


def _gpu(body: RefModIn) -> int:
    """The Sandbox's GPU, or (it isn't running) the one asked for, as a generation picks it."""
    try:
        return sbx.pick_gpu(body.gpu)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/check")
def refmod_check(body: RefModIn):
    """What making it will do, without doing it: each reference's encode size and frames, the tokens (and how many
    the cap keeps), the sheet's layout, the VRAM encoding the largest takes beside what's free, and a warning when
    a clip may not fit (an error when it can't be made as set)."""
    from ...engine import memory, refmod

    profile, F, paths, s, name = check(body)
    plan = refmod.plan(paths, s, F.fps, F.snap_frames, F.latent_frames or (lambda f: f), body.sheet)
    cap, per_frame, tokens = s["refmod_max_tokens"], plan["per_frame"], plan["tokens"]
    error = None
    if cap and per_frame > cap:
        error = (f"One {'frame' if s['refmod_mode'] == 'full' else 'grid'} takes {per_frame:,} tokens, more than the "
                 f"cap of {cap:,}: lower the {'size' if s['refmod_mode'] == 'full' else 'grid'}, or raise the cap")
    vram, warning = None, None
    try:
        gpu = _gpu(body)
    except HTTPException:
        gpu = None
    stats = next((g for g in hardware.gpus() if g["index"] == gpu), None)
    if stats:
        loaded = sbx.sandbox.alive() and sbx.sandbox.gpu == gpu
        need = memory.encode_bytes(F, profile["data"], plan["pixels"], loaded=loaded)
        free = (stats["mem_total"] - stats["mem_used"]) * 1024**2
        vram = {"gpu": gpu, "need_gb": round(need / GB, 1), "free_gb": round(free / GB, 1), "loaded": loaded}
        clips = [r for r in plan["refs"] if r["kind"] == "video"]
        if clips and need > free:
            big = max(clips, key=lambda r: r["width"] * r["height"] * r["frames"])
            warning = (f"Encoding the largest clip ({big['width']}×{big['height']}, {big['frames']} frames) takes about "
                       f"{need / GB:.1f} GB, and GPU {gpu} has {free / GB:.1f} GB free"
                       + (" beside the Sandbox's model" if loaded else "") + ". It may run out of memory: lower the "
                       "size or the frames per clip" + (", or release the Sandbox first" if loaded else "") + ".")
    return {"name": name, "refs": plan["refs"], "tokens": tokens, "per_frame": per_frame, "cap": cap,
            "kept": min(tokens, cap // per_frame * per_frame) if cap and per_frame <= cap else tokens,
            "sheet": plan["sheet"], "vram": vram, "warning": warning, "error": error}


@router.get("/sheet")
def refmod_sheet(path: list[str] = Query(...), size: int = 1024):
    """A preview of the sheet ``path`` (the images) make, as a JPEG."""
    from ...engine import dataset as ds
    from ...engine.refmod import SHEET_MAX, make_sheet

    paths = [os.path.expanduser(p) for p in path if p.strip()]
    if not 2 <= len(paths) <= SHEET_MAX or any(ds.media_kind(p) != "image" or not os.path.isfile(p) for p in paths):
        raise HTTPException(400, f"A sheet takes 2–{SHEET_MAX} images")
    sheet = make_sheet(paths, max(256, min(2048, size)))
    sheet.thumbnail((SHEET_PREVIEW, SHEET_PREVIEW))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg")


@router.post("")
def refmod_make(body: RefModIn):
    """Queue making it in the Sandbox's worker (after the generations queued before it): its progress and result
    are the Sandbox status's ``session.refmod``."""
    from .. import db

    profile, _, paths, s, name = check(body)
    gpu = _gpu(body)
    reason = sbx.sandbox.check_gpu(gpu)
    if reason:
        raise HTTPException(409, reason)
    mid = db.new_id()
    if not sbx.sandbox.submit_refmod({"id": mid, "out_dir": made_dir(mid), "name": name, "paths": paths,
                                      "sheet": body.sheet, "settings": s,
                                      "profile": {"id": profile["id"], "family": profile["family"],
                                                  "data": profile["data"]}}, gpu):
        raise HTTPException(409, "A RefMod is being made: wait for it, or cancel it")
    return {"id": mid, "name": name}


@router.post("/cancel")
def refmod_cancel():
    made = sbx.sandbox.refmod
    if made and made["state"] in ("queued", "running"):
        sbx.sandbox.cancel(made["id"])
    return {"ok": True}


@router.get("/file/{mid}/{name}")
def refmod_file(mid: str, name: str):
    path = os.path.join(made_dir(mid), os.path.basename(name))
    if not name.endswith(".png") or not os.path.exists(path):
        not_found()
    return FileResponse(path)

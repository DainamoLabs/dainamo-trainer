"""Upscaling (``engine.upscale``): the upscalers in the models folder's upscale_models, a Sandbox run's output or
a file upscaled as a run of its own, and a dataset's small images upscaled in place. The Sandbox's worker does the
work, after the generations queued before it, so the models it holds stay loaded."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...engine import upscale
from ...engine.dataset import list_media, media_kind, probe_video
from .. import dataset_check, db, model_files
from .. import sandbox as sbx
from .common import dataset_or_404, not_found
from .sandbox import RUN_COLUMNS

router = APIRouter(prefix="/api")

MAX_FACTOR = 8
MAX_IMAGE_SIDE = 8192  # an upscaled image's longest side
MAX_VIDEO_SIDE = 4096
MIN_RESOLUTION, MAX_RESOLUTION = 128, 4096


@router.get("/upscalers")
def list_upscalers():
    """{folder, models: [{path, kind ("pixel", "seedvr2" or "unknown"), name, scale (0: any size), video, companion
    (SeedVR2's VAE), error, size}], error}: what's in the models folder's upscale_models."""
    return model_files.upscalers()


def check_model(raw: str) -> dict:
    """An upscaler file as ``upscale.Info`` JSON, or a 400 saying why it can't upscale."""
    path = os.path.abspath(os.path.expanduser(raw.strip())) if raw.strip() else ""
    if not path:
        raise HTTPException(400, "Choose an upscaler")
    if not os.path.isfile(path):
        raise HTTPException(400, f"Upscaler not found: {path}")
    info = next((m for m in model_files.upscalers()["models"] if m["path"] == path), None)
    if info is None:  # not in upscale_models
        info = upscale.identify(path).to_json()
    if info["kind"] not in ("pixel", "seedvr2") or info["error"]:
        raise HTTPException(400, f"{os.path.basename(path)}: {info['error'] or 'not an upscaler Dainamo reads'}")
    return info


# ------------------------------------------------------------------------------------------------ Sandbox ---
@router.post("/sandbox/upload")
def sandbox_upload(file: UploadFile):  # sync: the copy runs in the threadpool
    """An image or video sent from the browser to upscale: {path, name, kind, url}. Uploads are kept until the server
    restarts."""
    name = os.path.basename(file.filename or "")
    kind = media_kind(name) if name else None
    if kind not in ("image", "video"):
        raise HTTPException(400, f"{name or 'That'} isn't an image or a video")
    uid = db.new_id()
    folder = os.path.join(sbx.UPLOADS_DIR, uid)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, name)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"path": dest, "name": name, "kind": kind, "url": f"/api/sandbox/upload/{uid}/{quote(name)}"}


@router.get("/sandbox/upload/{uid}/{name}")
def sandbox_uploaded(uid: str, name: str):
    path = os.path.join(sbx.UPLOADS_DIR, os.path.basename(uid), os.path.basename(name))
    if not os.path.isfile(path):
        not_found()
    return FileResponse(path)


class UpscaleIn(BaseModel):
    model: str  # an upscaler file (GET /api/upscalers)
    factor: float = 2  # how many times bigger each side gets
    source: dict | None = None  # {run_id, file}: a Sandbox run's output
    upload: str | None = None  # or a file on this machine: one sent with POST /api/sandbox/upload, or any other
    options: dict = {}  # passed to the upscaler as they are (SeedVR2's settings)
    gpu: int | None = None  # where to run it if the Sandbox isn't running (else its GPU)


def _media_size(path: str, kind: str) -> dict:
    """{width, height} as shown (EXIF-turned), and a clip's {frames, fps}; 400 when it can't be read."""
    if kind == "image":
        info = dataset_check.file_info(path)
        if "error" in info:
            raise HTTPException(400, f"{os.path.basename(path)} can't be read: {info['error']}")
        return {"width": info["width"], "height": info["height"]}
    try:
        v = probe_video(path)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise HTTPException(400, f"{os.path.basename(path)} can't be read: {e}")
    if not v.get("width") or not v.get("height"):
        raise HTTPException(400, f"{os.path.basename(path)} has no picture")
    fps = float(v.get("fps") or 24)
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": fps,
            "frames": max(1, int(round(float(v.get("duration") or 0) * fps)))}


def check_input(body: UpscaleIn) -> tuple[dict, dict | None]:
    """-> ({path, name, kind, width, height[, frames, fps], url, poster}, the source run {run_id, file, n} or None)."""
    upload = (body.upload or "").strip()
    if bool(body.source) == bool(upload):
        raise HTTPException(400, "Choose one thing to upscale: a run's output, or a file")
    source, url, poster = None, None, None
    if body.source:
        rid = os.path.basename(str(body.source.get("run_id") or ""))
        file = os.path.basename(str(body.source.get("file") or ""))
        run = db.query_one(f"SELECT {RUN_COLUMNS} FROM sandbox_runs WHERE id=?", (rid,)) or not_found("Run not found")
        out = next((o for o in run["outputs"] or [] if o.get("file") == file), None)
        if out is None:
            not_found(f"Run {run['n']} has no output called {file}")
        path = os.path.join(sbx.run_dir(rid), file)
        url = f"/api/sandbox/file/{rid}/{quote(file)}"
        poster = f"/api/sandbox/file/{rid}/{quote(out['poster'])}" if out.get("poster") else None
        source = {"run_id": rid, "file": file, "n": run["n"]}
    else:
        path = os.path.abspath(os.path.expanduser(upload))
        uploads = os.path.abspath(sbx.UPLOADS_DIR) + os.sep
        if path.startswith(uploads) and len(os.path.relpath(path, uploads).split(os.sep)) == 2:
            uid, name = os.path.relpath(path, uploads).split(os.sep)
            url = f"/api/sandbox/upload/{uid}/{quote(name)}"
    if not os.path.isfile(path):
        raise HTTPException(400, f"Not found: {path}")
    kind = media_kind(path)
    if kind not in ("image", "video"):
        raise HTTPException(400, f"{os.path.basename(path)} isn't an image or a video")
    return {"path": path, "name": os.path.basename(path), "kind": kind, **_media_size(path, kind), "url": url,
            "poster": poster}, source


def check_upscale(body: UpscaleIn) -> dict:
    """-> the run's params: {model, model_name, kind, scale, factor, source, upload, options, input, width, height}
    (``width`` and ``height``: the size asked for); 400 on anything wrong."""
    info = check_model(body.model)
    if not 1 < body.factor <= MAX_FACTOR:
        raise HTTPException(400, f"The factor must be above 1 and at most {MAX_FACTOR}")
    if any(not isinstance(v, (str, int, float, bool)) or v is None for v in body.options.values()):
        raise HTTPException(400, "Upscaler options must be numbers, text or on/off")
    inp, source = check_input(body)
    w, h = upscale.target_size(inp["width"], inp["height"], body.factor)
    video = inp["kind"] == "video"
    side = MAX_VIDEO_SIDE if video else MAX_IMAGE_SIDE
    if max(w, h) > side:
        raise HTTPException(400, f"{body.factor:g}× makes {w}×{h}; a{' video' if video else 'n image'}'s side can be at "
                                 f"most {side} pixels. Choose a smaller factor.")
    return {"model": info["path"], "model_name": info["name"], "kind": info["kind"], "scale": info["scale"],
            "factor": float(body.factor), "source": source, "upload": inp["path"] if not source else None,
            "options": dict(body.options), "input": inp, "width": w, "height": h}


def _gpu(asked: int | None) -> int:
    try:
        gpu = sbx.pick_gpu(asked)
    except ValueError as e:
        raise HTTPException(400, str(e))
    reason = sbx.sandbox.check_gpu(gpu)
    if reason:
        raise HTTPException(409, reason)
    return gpu


@router.post("/sandbox/upscale")
def sandbox_upscale(body: UpscaleIn):
    """Queue upscaling a run's output, or a file, as a Sandbox run (workflow "upscale"): it's in the history with the
    generations, and its output is the upscaled image or video."""
    params = check_upscale(body)
    gpu = _gpu(body.gpu)
    run = {"id": db.new_id(), "params": params}
    db.execute("INSERT INTO sandbox_runs (id, created, status, profile_id, profile_name, family, workflow, gpu, "
               "params, outputs) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (run["id"], db.now(), "queued", None, params["model_name"], "upscale", "upscale", gpu,
                json.dumps(params), "[]"))
    sbx.sandbox.submit_upscale(run, gpu)
    return db.query_one(f"SELECT {RUN_COLUMNS} FROM sandbox_runs WHERE id=?", (run["id"],))


# ----------------------------------------------------------------------------------------------- datasets ---
def plan(root: str, resolution: int, max_factor: float) -> dict:
    """Which of a dataset's images an upscale to ``resolution`` takes: those with fewer pixels than ``resolution``
    squared (``upscale.factor_to_reach``). {images, small: [paths], capped (they need more than ``max_factor``, so
    stay smaller), clash (taken: another file has the name the result gets, <stem>.png), unreadable, clips, smallest:
    {name, width, height}}."""
    files = list_media(root) if os.path.isdir(root) else []
    images = [f for f in files if media_kind(f) == "image"]
    small, capped, clash, unreadable, smallest = [], 0, [], 0, None
    for f in images:
        info = dataset_check.file_info(f)
        if "error" in info:
            unreadable += 1
            continue
        w, h = info["width"], info["height"]
        if math.sqrt(w * h) >= resolution:
            continue
        png = os.path.splitext(f)[0] + ".png"
        if png != f and os.path.exists(png):  # a.jpg beside a.png: its result would replace a.png
            clash.append(f)
            continue
        small.append(f)
        capped += resolution / math.sqrt(w * h) > max_factor
        if smallest is None or w * h < smallest["width"] * smallest["height"]:
            smallest = {"name": os.path.relpath(f, root), "width": w, "height": h}
    return {"images": len(images), "small": small, "capped": capped, "clash": [os.path.relpath(f, root) for f in clash],
            "unreadable": unreadable, "clips": sum(1 for f in files if media_kind(f) == "video"), "smallest": smallest}


def _resolution(resolution: int, max_factor: float):
    if not MIN_RESOLUTION <= resolution <= MAX_RESOLUTION:
        raise HTTPException(400, f"The resolution must be {MIN_RESOLUTION}–{MAX_RESOLUTION}")
    if not 1 < max_factor <= MAX_FACTOR:
        raise HTTPException(400, f"The largest factor must be above 1 and at most {MAX_FACTOR}")


@router.get("/datasets/{did}/upscale/plan")
def dataset_upscale_plan(did: str, resolution: int = 1024, max_factor: float = 4):
    """What Upscale small images would do at ``resolution``, without doing it: ``plan`` with ``small`` counted."""
    d = dataset_or_404(did)
    _resolution(resolution, max_factor)
    p = plan(d["path"], resolution, max_factor)
    return {**p, "small": len(p["small"])}


class DatasetUpscaleIn(BaseModel):
    model: str
    resolution: int = 1024  # the images with fewer pixels than this squared are upscaled to it
    max_factor: float = 4  # but by at most this
    gpu: int | None = None


def _status(did: str) -> dict:
    """This dataset's upscale (the last one, or the one going): {state ("none" | queued | running | done | failed
    | cancelled), message, done, total, made, model, resolution, max_factor...}, with ``elsewhere`` ({id, name}: the
    dataset being upscaled now, when it's another) and ``last`` (the settings of the last upscale of any dataset)."""
    j = sbx.sandbox.upscale
    mine = dict(j) if j and j["dataset_id"] == did else None
    busy = j and j["state"] in ("queued", "running") and not mine
    return {**(mine or {"state": "none", "message": "", "done": 0, "total": 0, "made": None}),
            "elsewhere": {"id": j["dataset_id"], "name": j["dataset"]} if busy else None,
            "last": {k: j[k] for k in ("model", "resolution", "max_factor")} if j else None}


@router.post("/datasets/{did}/upscale")
def dataset_upscale(did: str, body: DatasetUpscaleIn):
    """Upscale the dataset's images smaller than the resolution, in the Sandbox's worker: each becomes <stem>.png at
    about the resolution (by at most ``max_factor``), its original moved to the _originals folder beside it; captions
    stay as they are. Linked folders are written to as well. One dataset at a time."""
    d = dataset_or_404(did)
    if not os.path.isdir(d["path"]):
        raise HTTPException(400, f"The dataset's folder is missing: {d['path']}")
    _resolution(body.resolution, body.max_factor)
    info = check_model(body.model)
    p = plan(d["path"], body.resolution, body.max_factor)
    if not p["small"]:
        raise HTTPException(400, f"No image is smaller than {body.resolution}×{body.resolution} pixels"
                                 + (f" ({len(p['clash'])} skipped: another file has the name it would get)"
                                    if p["clash"] else ""))
    gpu = _gpu(body.gpu)
    job = {"id": db.new_id(), "dataset_id": did, "dataset": d["name"], "model": info["path"],
           "model_name": info["name"], "paths": p["small"], "resolution": int(body.resolution),
           "max_factor": float(body.max_factor)}
    if not sbx.sandbox.submit_dataset_upscale(job, gpu):
        j = sbx.sandbox.upscale
        raise HTTPException(409, f"{j['dataset']} is being upscaled: wait for it, or cancel it")
    return _status(did)


@router.get("/datasets/{did}/upscale")
def dataset_upscale_status(did: str):
    dataset_or_404(did)
    return _status(did)


@router.post("/datasets/{did}/upscale/cancel")
def dataset_upscale_cancel(did: str):
    j = sbx.sandbox.upscale
    if j and j["dataset_id"] == did and j["state"] in ("queued", "running"):
        sbx.sandbox.cancel(j["id"])
    return {"ok": True}

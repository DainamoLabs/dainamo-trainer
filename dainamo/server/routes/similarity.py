"""Likeness (``engine.similarity``): how much an image looks like a dataset, as a percentage, with the training
images it's closest to; and a dataset's duplicates. The measuring runs in a short-lived process on a free GPU (or the
CPU), one request at a time; a dataset's image vectors are kept in data/similarity/<dataset id>.pt and brought up to
date when its files change, so comparing against a dataset again costs only the one image."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...engine.dataset import list_media, media_kind
from .. import db, procs
from .. import sandbox as sbx
from ..queue import job_dir, read_status
from .common import dataset_or_404, job_or_404, not_found
from .sandbox import RUN_COLUMNS

router = APIRouter(prefix="/api")

SIMILARITY_DIR = os.path.join(db.DATA_DIR, "similarity")
LOG = os.path.join(db.DATA_DIR, "similarity.log")
TIMEOUT = 1800  # seconds: a big dataset's first embedding, on the CPU
TOP = 5  # nearest training images shown
MIN_THRESHOLD, MAX_THRESHOLD = 0.8, 0.999
_lock = threading.Lock()


def _run(request: dict) -> dict:
    """``python -m dainamo.engine.similarity`` on ``request`` (see ``similarity.run``): its result, or a 500/503 saying
    why not. One at a time: a second request waits for the first."""
    os.makedirs(SIMILARITY_DIR, exist_ok=True)
    with _lock:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="request-", dir=SIMILARITY_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(request, f)
        try:
            with open(LOG, "a", encoding="utf-8") as log:
                p = subprocess.run([sys.executable, "-m", "dainamo.engine.similarity", path], cwd=sbx.ROOT,
                                   env=procs.worker_env(sbx.free_gpu()), stdout=subprocess.PIPE, stderr=log,
                                   text=True, encoding="utf-8", timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            raise HTTPException(503, "Measuring took too long and was stopped (see data/similarity.log)")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    try:
        last = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        last = {}
    if p.returncode or "error" in last or "results" not in last:
        raise HTTPException(500, last.get("error") or f"Measuring failed (exit {p.returncode}; see data/similarity.log)")
    return last


def dataset_images(d: dict) -> list[str]:
    """The dataset's pictures: its images and clips (sound has no likeness)."""
    if not os.path.isdir(d["path"]):
        raise HTTPException(400, f"The dataset's folder is missing: {d['path']}")
    return [f for f in list_media(d["path"]) if media_kind(f) in ("image", "video")]


def concept_of(d: dict) -> dict:
    paths = dataset_images(d)
    if not paths:
        raise HTTPException(400, f"{d['name']} has no images or clips to compare with")
    return {"paths": paths, "cache": os.path.join(SIMILARITY_DIR, f"{os.path.basename(d['id'])}.pt")}


def _file(path: str, root: str) -> dict:
    return {"path": path, "name": os.path.relpath(path, root), "url": f"/api/thumb?path={quote(path)}"}


class CompareIn(BaseModel):
    dataset_id: str
    source: dict | None = None  # {run_id, file}: a Sandbox run's output
    sample: dict | None = None  # {job_id, file}: one of a job's training samples
    upload: str | None = None  # or a file on this machine (one sent with POST /api/sandbox/upload, or any other)


def check_image(body: CompareIn) -> tuple[str, str | None, dict]:
    """-> (the image's path, the base model's render of it when there is one (its floor), what it is: {kind, label,
    url, floor_url})."""
    given = [k for k in ("source", "sample", "upload") if getattr(body, k)]
    if len(given) != 1:
        raise HTTPException(400, "Choose one image to compare: a run's output, a training sample, or a file")
    if body.source:
        rid = os.path.basename(str(body.source.get("run_id") or ""))
        file = os.path.basename(str(body.source.get("file") or ""))
        run = db.query_one(f"SELECT {RUN_COLUMNS} FROM sandbox_runs WHERE id=?", (rid,)) or not_found("Run not found")
        outs = run["outputs"] or []
        out = next((o for o in outs if o.get("file") == file), None)
        if out is None:
            not_found(f"Run {run['n']} has no output called {file}")
        if out.get("kind") not in ("image", "video"):
            raise HTTPException(400, "Only an image or a clip has a likeness")
        # the same seed rendered without LoRAs, when the run compared them: the floor
        floor = None
        if out.get("variant") == "lora":
            base = next((o for o in outs if o.get("variant") == "base" and o.get("seed") == out.get("seed")), None)
            floor = os.path.join(sbx.run_dir(rid), os.path.basename(base["file"])) if base else None
        url = lambda f: f"/api/sandbox/file/{rid}/{quote(f)}"  # noqa: E731
        return (os.path.join(sbx.run_dir(rid), file), floor,
                {"kind": "run", "label": f"Run {run['n']} · {file}", "url": url(out.get("poster") or file),
                 "floor_url": url(os.path.basename(floor)) if floor else None})
    if body.sample:
        jid = os.path.basename(str(body.sample.get("job_id") or ""))
        file = os.path.basename(str(body.sample.get("file") or ""))
        j = job_or_404(jid)
        samples = read_status(jid).get("samples") or []
        s = next((x for x in samples if x.get("file") == file), None)
        if s is None:
            not_found(f"{j['name']} has no sample called {file}")
        if s.get("kind", "image") not in ("image", "video"):
            raise HTTPException(400, "Only an image or a clip has a likeness")
        floor = None
        if s.get("step"):
            base = next((x for x in samples if x.get("step") == 0 and x.get("index") == s.get("index")
                         and x.get("prompt") == s.get("prompt") and x.get("seed") == s.get("seed")), None)
            floor = os.path.join(job_dir(jid), "samples", os.path.basename(base["file"])) if base else None
        url = lambda f: f"/api/jobs/{jid}/file/samples/{quote(f)}"  # noqa: E731
        return (os.path.join(job_dir(jid), "samples", file), floor,
                {"kind": "sample", "label": f"{j['name']} · step {s.get('step')} · {file}",
                 "url": url(s.get("poster") or file), "floor_url": url(os.path.basename(floor)) if floor else None})
    path = os.path.abspath(os.path.expanduser(body.upload.strip()))
    if not os.path.isfile(path):
        raise HTTPException(400, f"Not found: {path}")
    if media_kind(path) not in ("image", "video"):
        raise HTTPException(400, f"{os.path.basename(path)} isn't an image or a video")
    uploads = os.path.abspath(sbx.UPLOADS_DIR) + os.sep
    url = None
    if path.startswith(uploads) and len(os.path.relpath(path, uploads).split(os.sep)) == 2:
        uid, name = os.path.relpath(path, uploads).split(os.sep)
        url = f"/api/sandbox/upload/{uid}/{quote(name)}"
    return path, None, {"kind": "file", "label": os.path.basename(path), "url": url, "floor_url": None}


@router.post("/similarity/compare")
def compare(body: CompareIn):
    """How much an image looks like a dataset: {dataset: {id, name, count, self_sim}, image: {kind, label, url},
    likeness (%), similarity (cosine to the dataset's centre), floor ({url, similarity}: the base model's render of
    the same prompt and seed, when there is one; else null, and the floor is 0), nearest: [{path, name, url, sim}] (the
    ``TOP`` closest training images), near_copy (the closest is at least ``NEAR_COPY`` alike: a near copy of it)}."""
    from ...engine.similarity import FLOOR, NEAR_COPY

    d = dataset_or_404(body.dataset_id)
    path, floor, image = check_image(body)
    if not os.path.isfile(path):
        not_found("The image is gone")
    if floor and not os.path.isfile(floor):
        floor = None
    out = _run({"concept": concept_of(d), "images": [path], "floors": [floor], "top": TOP})
    r = out["results"][0]
    return {"dataset": {"id": d["id"], "name": d["name"], "count": out["count"], "self_sim": out["self_sim"]},
            "image": {k: image[k] for k in ("kind", "label", "url")},
            "likeness": r["likeness"], "similarity": r["sim"],
            "floor": {"url": image["floor_url"], "similarity": r["floor"]} if r.get("floor") is not None else None,
            "floor_default": FLOOR, "near_copy_at": NEAR_COPY,
            "nearest": [{**_file(t["path"], d["path"]), "sim": t["sim"]} for t in r.get("top") or []],
            "near_copy": bool(r.get("near_copy"))}


@router.get("/datasets/{did}/duplicates")
def duplicates(did: str, threshold: float = 0.98):
    """Pairs of the dataset's images at least ``threshold`` alike (cosine of their DINOv2 features: 0.98 catches
    copies and re-encodes, 0.95 small crops too, a mirrored copy scores about 0.92), the most alike first: {threshold,
    count (images compared), pairs: [{a: {path, name, url}, b, sim}]}."""
    d = dataset_or_404(did)
    if not MIN_THRESHOLD <= threshold <= MAX_THRESHOLD:
        raise HTTPException(400, f"The threshold must be {MIN_THRESHOLD}–{MAX_THRESHOLD}")
    paths = dataset_images(d)
    if len(paths) < 2:
        return {"threshold": threshold, "count": len(paths), "pairs": []}
    out = _run({"concept": {"paths": paths, "cache": os.path.join(SIMILARITY_DIR, f"{os.path.basename(did)}.pt")},
                "duplicates": threshold})
    return {"threshold": threshold, "count": out["count"],
            "pairs": [{"a": _file(p["a"], d["path"]), "b": _file(p["b"], d["path"]), "sim": p["sim"]}
                      for p in out["duplicates"]]}

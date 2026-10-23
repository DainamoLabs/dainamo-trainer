"""The clip editor: the plan of a dataset Split into clips made (``clip_split``), which clips each video is cut
into, seen and changed, then cut again. ``clip_split.SplitJob`` does the looking and the cutting; the frames shown
are taken from the videos as the cut takes them."""

from __future__ import annotations

import hashlib
import math
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...engine.dataset import list_media, media_kind, probe_video
from .. import captioner, clip_split, db
from ..validation import MAX_FRAMES
from .common import dataset_or_404, family_or_400, not_found
from .datasets import THUMBS, _in_dataset

router = APIRouter(prefix="/api")

MIN_CROP = 64  # px a side


def _split(tid: str):
    """(target, source, family driver, manifest) of a dataset Split into clips made, or a 400 saying why its clips
    can't be edited."""
    t = dataset_or_404(tid)
    m = clip_split.load_manifest(t["path"]) if os.path.isdir(t["path"]) else None
    src = (m or {}).get("source") or {}
    if not src.get("id") or not t["managed"]:
        raise HTTPException(400, "Only a dataset made by Split into clips has clips to edit")
    s = db.query_one("SELECT * FROM datasets WHERE id=?", (src["id"],))
    if not s or not os.path.isdir(s["path"]):
        raise HTTPException(400, f"The dataset these clips were cut from, {src.get('name') or 'their source'}, is "
                                 "gone, so they can't be changed")
    return t, s, family_or_400(m.get("family", "")), m


def _video(rel: str, s: dict, m: dict) -> dict:
    """A video as the editor shows it: its plan, and how each clip was cut (``cut``, by number), where it was cut
    at the size cutting now gives."""
    v = m["videos"].get(rel) or {}
    path = os.path.join(s["path"], *rel.split("/"))
    side = m["settings"]["max_side"]
    cut = {c["id"]: {"start": c["start"], "frames": c["frames"], "crop": c.get("crop")}
           for c in m["clips"].values() if c.get("source") == rel and "id" in c and c.get("max_side") == side}
    return {"rel": rel, "path": path if os.path.isfile(path) else None, "looked": "duration" in v, "stem": v.get("stem"),
            "duration": v.get("duration"), "fps": v.get("fps"), "width": v.get("width"), "height": v.get("height"),
            "has_audio": bool(v.get("has_audio")), "cuts": v.get("cuts") or [], "clips": v.get("clips") or [],
            "next": v.get("next", 1), "note": v.get("note") or "", "cut": cut}


def _cutting(t: dict) -> bool:
    j = clip_split.job_for(t["id"])
    return bool(j and j.running and j.cut)


def _idle(s: dict, t: dict):
    j = clip_split.split_jobs.get(s["id"])
    if j and j.running:
        raise HTTPException(409, "Wait for the clips being looked at or cut" if j.target["id"] == t["id"] else
                            f"{s['name']} is being split into another dataset; wait for it to finish")


@router.get("/datasets/{tid}/clips")
def clip_plan(tid: str):
    """The clip editor's page: every video of the source (and any the plan has that's gone), each with its clips."""
    t, s, F, m = _split(tid)
    rels = {os.path.relpath(f, s["path"]).replace(os.sep, "/") for f in list_media(s["path"]) if media_kind(f) == "video"}
    job = clip_split.job_for(tid)
    return {"id": t["id"], "name": t["name"], "source": {"id": s["id"], "name": s["name"]}, "family": F.info.id,
            "family_label": F.info.label, "fps": F.fps, "frames": int(m.get("frames") or 0), "edge": clip_split.EDGE,
            "lengths": clip_split.clip_lengths(F), "settings": m["settings"],
            "videos": [_video(rel, s, m) for rel in sorted(rels | set(m["videos"]))],
            "changes": clip_split.changes(m), "job": job.to_json() if job else None}


class ClipIn(BaseModel):
    id: int  # its number in its video, kept when it changes (its file is <stem>_<id>.mp4)
    start: float  # s
    frames: int
    crop: list[int] | None = None  # [x, y, w, h] in the video's pixels; None: the whole frame


class ClipsIn(BaseModel):
    rel: str  # the video, in the source dataset
    clips: list[ClipIn]


def _looked(m: dict, rel: str) -> dict:
    v = m["videos"].get(rel)
    if not v or "duration" not in v:
        raise HTTPException(400, "This video hasn't been looked at yet")
    return v


def _saved(t: dict, s: dict, rel: str, change) -> dict:
    m = clip_split.update_manifest(t["path"], change)
    return {"video": _video(rel, s, m), "changes": clip_split.changes(m)}


@router.put("/datasets/{tid}/clips")
def save_clips(tid: str, body: ClipsIn):
    """A video's clips as the editor has them now; they're cut when the dataset is cut again."""
    t, s, F, m = _split(tid)
    if _cutting(t):
        raise HTTPException(409, "Wait for the clips being cut")
    v = _looked(m, body.rel)
    lengths = set(clip_split.clip_lengths(F))
    ids = [c.id for c in body.clips]
    if len(ids) > clip_split.MAX_PLANNED:
        raise HTTPException(400, f"A video can have up to {clip_split.MAX_PLANNED} clips")
    if len(set(ids)) < len(ids) or min(ids, default=1) < 1:
        raise HTTPException(400, "Each clip needs a number of its own")
    clips = []
    for c in body.clips:
        if c.frames not in lengths:
            raise HTTPException(400, f"{F.info.label} takes clips of {F.info.frame_rule or 'other'} frames, up to "
                                     f"{MAX_FRAMES}")
        latest = clip_split.latest_start(v["duration"], c.frames, F.fps)
        if not math.isfinite(c.start) or c.start < 0 or c.start > latest + 1e-3:
            raise HTTPException(400, f"Clip {c.id} runs past the end of the video")
        crop = None
        if c.crop is not None:
            if len(c.crop) != 4:
                raise HTTPException(400, "A crop is [x, y, width, height]")
            x, y, w, h = c.crop
            if x < 0 or y < 0 or w < MIN_CROP or h < MIN_CROP or x + w > v["width"] or y + h > v["height"]:
                raise HTTPException(400, f"Clip {c.id}'s crop must be inside the frame and at least {MIN_CROP} px a side")
            crop = None if (w, h) == (v["width"], v["height"]) else [x, y, w, h]  # the whole frame is no crop
        clips.append({"id": c.id, "start": round(min(c.start, max(0.0, latest)), 3), "frames": c.frames, "crop": crop})
    clips.sort(key=lambda c: (c["start"], c["id"]))

    def put(m: dict):
        v = m["videos"][body.rel]
        v["clips"] = clips
        v["next"] = max(v.get("next", 1), max(ids, default=0) + 1)
    return _saved(t, s, body.rel, put)


class ReplanIn(BaseModel):
    rel: str
    frames: int


@router.post("/datasets/{tid}/clips/replan")
def replan(tid: str, body: ReplanIn):
    """A video's clips planned again as Split into clips plans them, at ``frames``, in place of the ones it has."""
    t, s, F, m = _split(tid)
    if _cutting(t):
        raise HTTPException(409, "Wait for the clips being cut")
    _looked(m, body.rel)
    if body.frames not in clip_split.clip_lengths(F):
        raise HTTPException(400, f"{F.info.label} takes clips of {F.info.frame_rule or 'other'} frames, up to {MAX_FRAMES}")

    def put(m: dict):
        v = m["videos"][body.rel]
        v["clips"] = clip_split.plan_video(v, m["settings"], body.frames, F)
        v["note"] = "" if v["clips"] else "no shot is long enough for a clip of this length"
    return _saved(t, s, body.rel, put)


@router.post("/datasets/{tid}/clips/look")
def look(tid: str):
    """Look at the videos the plan doesn't know yet (new to the source, or from a split made before the editor):
    their length and shots, and for a new one its clips."""
    t, s, F, _ = _split(tid)
    _idle(s, t)
    clip_split.split_jobs[s["id"]] = clip_split.SplitJob(s, t, F, cut=False)
    return clip_split.split_jobs[s["id"]].to_json()


@router.post("/datasets/{tid}/clips/cut")
def cut(tid: str):
    """Cut the clips that are new or changed, and remove those the plan no longer has."""
    t, s, F, _ = _split(tid)
    _idle(s, t)
    cap = captioner.caption_jobs.get(t["id"])
    if cap and cap.running:
        raise HTTPException(409, "Wait for Auto-Caption to finish on this dataset")
    clip_split.split_jobs[s["id"]] = clip_split.SplitJob(s, t, F, cut=True)
    return clip_split.split_jobs[s["id"]].to_json()


@router.delete("/datasets/{tid}/clips/job")
def cancel(tid: str):
    j = clip_split.job_for(tid)
    if j:
        j.cancel()
    return {"ok": True}


def _cached(path: str, key: str, make) -> FileResponse:
    """A JPEG made by ``make()`` once per version of the video, kept with the thumbnails."""
    out = os.path.join(THUMBS, hashlib.sha1(f"{key}|{path}|{os.stat(path).st_mtime_ns}".encode()).hexdigest() + ".jpg")
    if not os.path.exists(out):
        try:
            data = make()
        except RuntimeError as e:
            not_found(f"Couldn't read the video there: {e}")
        os.makedirs(THUMBS, exist_ok=True)
        with open(out + ".tmp", "wb") as f:
            f.write(data)
        os.replace(out + ".tmp", out)
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def _a_video(path: str) -> str:
    _in_dataset(path)
    if media_kind(path) != "video" or not os.path.isfile(path):
        not_found("No such video")
    return path


@router.get("/clips/frame")
def clip_frame(path: str, start: float, n: int = 0, fps: int = 24, size: int = 480):
    """Frame ``n`` of a clip cut from ``start`` s at ``fps`` (``clip_split.frame_at``)."""
    _a_video(path)
    if not (math.isfinite(start) and start >= 0 and 0 <= n <= MAX_FRAMES and 1 <= fps <= 120 and 64 <= size <= 2048):
        raise HTTPException(400, "A frame is at 0 s or later, up to the longest clip, at 1–120 fps and 64–2048 px")
    return _cached(path, f"frame|{start:.3f}|{n}|{fps}|{size}", lambda: clip_split.frame_at(path, start, n, fps, size))


@router.get("/clips/strip")
def clip_strip(path: str, count: int = 16, height: int = 64):
    """``count`` frames spread over a video, side by side (``clip_split.filmstrip``)."""
    _a_video(path)
    if not (1 <= count <= 64 and 16 <= height <= 160):
        raise HTTPException(400, "A strip has 1–64 frames, 16–160 px tall")
    return _cached(path, f"strip|{count}|{height}",
                   lambda: clip_split.filmstrip(path, probe_video(path)["duration"], count, height))

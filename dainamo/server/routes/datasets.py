"""Datasets: folders of images and videos with captions, thumbnails and auto-captioning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageOps
from pydantic import BaseModel

from ...engine.dataset import caption_path, list_media, media_kind, read_caption, resolutions, split_clips
from ...engine.families import get_family
from ...engine.media import spectrogram
from ...engine.transcribe import MODELS as WHISPER_MODELS
from .. import captioner, clip_split, dataset_check, db, export, media_download
from ..llama_server import describe, provider
from ..validation import MAX_FRAMES, check_resolution
from .common import dataset_or_404, family_or_400, not_found, profile_or_404

router = APIRouter(prefix="/api")

THUMBS = os.path.join(db.DATA_DIR, "thumbs")
MANAGED_DATASETS = os.path.join(db.DATA_DIR, "datasets")
CAPTION_BACKUPS = os.path.join(db.DATA_DIR, "caption_backups")  # <dataset id>.json: the captions before the last bulk edit


class DatasetIn(BaseModel):
    name: str
    path: str = ""  # empty -> managed folder for uploads
    trigger: str = ""


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "dataset"


@router.get("/datasets")
def list_datasets():
    out = []
    for d in db.query("SELECT * FROM datasets ORDER BY name COLLATE NOCASE"):
        files = list_media(d["path"]) if os.path.isdir(d["path"]) else []
        d["count"] = len(files)
        d["videos"] = sum(1 for f in files if media_kind(f) == "video")
        d["audio"] = sum(1 for f in files if media_kind(f) == "audio")
        d["images"] = d["count"] - d["videos"] - d["audio"]
        d["captioned"] = sum(1 for f in files if os.path.exists(caption_path(f)))
        d["bytes"] = sum(_size(f) for f in files)  # the media files'
        # what a video model trains on per epoch: clips split from one video count as one
        cut = split_clips(d["path"]) if d["videos"] else {}
        videos = [os.path.normpath(f) for f in files if media_kind(f) == "video"]
        d["video_sources"] = len({cut[f]["source"] for f in videos if f in cut}) + sum(1 for f in videos if f not in cut)
        d["exists"] = os.path.isdir(d["path"])
        d["cover"] = files[0] if files else None
        d["split"] = clip_split.summary(d["path"]) if d["exists"] else None  # made by Split into clips
        out.append(d)
    return out


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


@router.get("/export/datasets")
def export_datasets(ids: str, trigger: bool = False):
    """The datasets ``ids`` (comma-separated) as one .zip (``export.zip_datasets``), with each caption as training
    reads it if ``trigger`` (the dataset's trigger word in it, for another trainer)."""
    rows = [dataset_or_404(i) for i in dict.fromkeys(x.strip() for x in ids.split(",") if x.strip())]
    if not rows:
        raise HTTPException(400, "Choose the datasets to export")
    missing = [d["name"] for d in rows if not os.path.isdir(d["path"])]
    if missing:
        raise HTTPException(400, f"The folder of {', '.join(missing)} is missing")
    name = export.folder_name(rows[0]["name"] if len(rows) == 1 else f"datasets_{time.strftime('%Y-%m-%d')}", set()) + ".zip"
    ascii_name = re.sub(r'[^A-Za-z0-9._ ()-]+', "_", name)
    return StreamingResponse(export.zip_datasets(rows, trigger), media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name, safe='')}"})


@router.post("/datasets")
def create_dataset(d: DatasetIn):
    did = db.new_id()
    managed = not d.path.strip()
    path = os.path.join(MANAGED_DATASETS, f"{_slug(d.name)}_{did[:4]}") if managed else \
        os.path.abspath(os.path.expanduser(d.path.strip()))
    if managed:
        os.makedirs(path, exist_ok=True)
    elif not os.path.isdir(path):
        raise HTTPException(400, f"Folder not found: {path}")
    db.execute("INSERT INTO datasets (id, name, path, managed, trigger, created) VALUES (?,?,?,?,?,?)",
               (did, d.name.strip(), path, int(managed), d.trigger.strip(), db.now()))
    return dataset_or_404(did)


@router.put("/datasets/{did}")
def update_dataset(did: str, d: DatasetIn):
    dataset_or_404(did)
    db.execute("UPDATE datasets SET name=?, trigger=? WHERE id=?", (d.name.strip(), d.trigger.strip(), did))
    return dataset_or_404(did)


@router.delete("/datasets/{did}")
def delete_dataset(did: str, delete_files: bool = False):
    d = dataset_or_404(did)
    if delete_files and d["managed"] and d["path"].startswith(MANAGED_DATASETS):
        shutil.rmtree(d["path"], ignore_errors=True)
    db.execute("DELETE FROM datasets WHERE id=?", (did,))
    return {"ok": True}


@router.get("/datasets/{did}/items")
def dataset_items(did: str):
    d = dataset_or_404(did)
    if not os.path.isdir(d["path"]):
        return []
    items = []
    cut = split_clips(d["path"])
    for f in list_media(d["path"]):
        c = cut.get(os.path.normpath(f))
        try:
            size = os.path.getsize(f)
        except OSError:
            size = None
        items.append({"path": f, "name": os.path.relpath(f, d["path"]), "kind": media_kind(f), "size": size,
                      "caption": read_caption(f), "has_caption": os.path.exists(caption_path(f)),
                      "cut_from": {"source": c["source"], "start": c.get("start", 0)} if c else None})
    return items


@router.post("/datasets/{did}/upload")
def upload(did: str, files: list[UploadFile]):  # sync: file copies run in the threadpool
    d = dataset_or_404(did)
    saved = 0
    for uf in files:
        name = os.path.basename(uf.filename or "")
        if not name or not (media_kind(name) or name.lower().endswith(".txt")):
            continue
        dest = os.path.join(d["path"], name)
        with open(dest, "wb") as out:
            shutil.copyfileobj(uf.file, out)
        saved += 1
    return {"saved": saved}


def _in_dataset(path: str) -> dict:
    path = os.path.abspath(path)
    for d in db.query("SELECT * FROM datasets"):
        if path.startswith(os.path.abspath(d["path"]) + os.sep):
            return d
    raise HTTPException(403, "File is not inside a dataset")


class CaptionIn(BaseModel):
    path: str
    caption: str


@router.put("/caption")
def put_caption(c: CaptionIn):
    _in_dataset(c.path)
    with open(caption_path(c.path), "w", encoding="utf-8") as f:
        f.write(c.caption.strip())
    return {"ok": True}


@router.delete("/media")
def delete_media(path: str):
    d = _in_dataset(path)
    if not d["managed"]:
        raise HTTPException(400, "Files in linked folders are not deleted from disk; remove them yourself")
    for p in (path, caption_path(path)):
        if os.path.exists(p):
            os.remove(p)
    return {"ok": True}


@router.get("/media")
def media(path: str):
    _in_dataset(path)
    return FileResponse(path)


@router.get("/thumb")
def thumb(path: str, size: int = 384):
    _in_dataset(path)
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{size}".encode()).hexdigest()
    out = os.path.join(THUMBS, key + ".jpg")
    if not os.path.exists(out):
        if media_kind(path) == "video":
            frames, _ = captioner.video_frames(path, 1)
            if not frames:
                not_found("No frame")
            img = frames[0]
        elif media_kind(path) == "audio":
            poster = out + ".png"
            try:
                spectrogram(path, poster, max(size, 64), max(size // 2, 32))
            except subprocess.SubprocessError:
                not_found("No audio")
            img = Image.open(poster)
            img.load()
            os.remove(poster)
        else:
            img = ImageOps.exif_transpose(Image.open(path))
        img = img.convert("RGB")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        img.save(out, "JPEG", quality=85)
    return FileResponse(out, headers={"Cache-Control": "max-age=86400"})


@router.get("/datasets/{did}/check")
def check_dataset(did: str, resolution: str = "", family: str = "", frames: int = 0, seconds: int = -1,
                  auto: bool = False):
    """Health check (``dataset_check``) for training at ``resolution`` (comma-separated sizes) with a
    model ``family``; ``frames`` is the clip length for video models (blank: the family's default; ``auto``:
    each clip at its own length, up to the longest a clip trains at), and ``seconds`` a music model's (0: whole
    songs; -1: the family's default)."""
    d = dataset_or_404(did)
    F = family_or_400(family) if family else None
    if F and "audio" in F.info.media:
        clip = seconds if seconds >= 0 else int(F.info.training_defaults.get("clip_seconds", 30))
        return dataset_check.check(d["path"], d["trigger"], token_limit=F.caption_tokens, clip_seconds=clip)
    res = resolutions(check_resolution(resolution)) if resolution.strip() else \
        resolutions(F.info.training_defaults.get("resolution", 1024) if F else 1024)
    video_frames = clip_length = None
    if F and "video" in F.info.media:
        clip_length = MAX_FRAMES if auto else frames or int(F.info.training_defaults.get("num_frames", 39))
        video_frames = F.snap_frames(clip_length) if F.snap_frames else clip_length
    return dataset_check.check(d["path"], d["trigger"], res, F.bucket_divisibility if F else 16, video_frames,
                               F.caption_tokens if F else None, clip_length=clip_length, auto_frames=auto)


class BulkCaptionIn(BaseModel):
    """One edit to every caption of a dataset; without ``apply`` it's only counted and previewed."""
    op: str  # replace | add | remove
    find: str = ""  # replace: what to find (plain text, or a regular expression with ``regex``)
    replace: str = ""
    text: str = ""  # add / remove: the words (whole words, ignoring case unless ``match_case``)
    where: str = "start"  # add: at the "start" or "end", joined with a comma
    regex: bool = False
    match_case: bool = False
    uncaptioned: bool = False  # add: files without a caption get one with just ``text``
    apply: bool = False


def _words(text: str, match_case: bool) -> re.Pattern:
    """``text`` as whole words: "ohwx" doesn't match inside "ohwx2"."""
    left = r"(?<!\w)" if text[:1].isalnum() or text[:1] == "_" else ""
    right = r"(?!\w)" if text[-1:].isalnum() or text[-1:] == "_" else ""
    return re.compile(left + re.escape(text) + right, 0 if match_case else re.IGNORECASE)


def _tidy(caption: str) -> str:
    """What removing words leaves: no doubled commas or spaces, no separator at either end."""
    caption = re.sub(r"\s*,(\s*,)+", ",", caption)
    caption = re.sub(r"\s+([,.;:])", r"\1", caption)
    caption = re.sub(r"[ \t]{2,}", " ", caption)
    return caption.strip(" ,;\t")


def bulk_caption_edit(body: BulkCaptionIn):
    """-> a function from a caption (None: the file has none) to the edited one (None: still none)."""
    if body.op == "replace":
        if not body.find:
            raise HTTPException(400, "Enter what to find")
        try:
            pattern = re.compile(body.find if body.regex else re.escape(body.find), 0 if body.match_case else re.IGNORECASE)
        except re.error as e:
            raise HTTPException(400, f"That regular expression doesn't work: {e}")
        # plain text goes in as it is; a regular expression's replacement may use \1 and the like
        repl = body.replace if body.regex else (lambda m: body.replace)
        return lambda c: None if c is None else pattern.sub(repl, c)
    text = body.text.strip()
    if body.op not in ("add", "remove") or not text:
        raise HTTPException(400, "Enter the words to " + ("add" if body.op == "add" else "remove"))
    words = _words(text, body.match_case)
    if body.op == "remove":
        return lambda c: None if c is None else _tidy(words.sub("", c))

    def add(c):
        if c is None and not body.uncaptioned:
            return None
        c = (c or "").strip()
        if words.search(c):
            return c  # already there
        if not c:
            return text
        return f"{text}, {c}" if body.where == "start" else f"{c.rstrip(' ,')}, {text}"
    return add


def _describe(body: BulkCaptionIn) -> str:
    if body.op == "replace":
        return f"Replaced “{body.find}” with “{body.replace}”"
    return f"Added “{body.text.strip()}” at the {body.where}" if body.op == "add" else f"Removed “{body.text.strip()}”"


def _caption_or_none(path: str) -> str | None:
    p = caption_path(path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read().strip()


@router.post("/datasets/{did}/captions/bulk")
def bulk_captions(did: str, body: BulkCaptionIn):
    """Find and replace, add words to or remove words from every caption. Applying keeps a copy of the
    captions it changed, for one undo."""
    d = dataset_or_404(did)
    edit = bulk_caption_edit(body)
    files = list_media(d["path"]) if os.path.isdir(d["path"]) else []
    changes = []
    try:
        for f in files:
            before = _caption_or_none(f)
            after = edit(before)
            if after is not None and after != before:
                changes.append((f, before, after))
    except (re.error, IndexError) as e:  # a replacement naming a group the pattern doesn't have
        raise HTTPException(400, f"That replacement doesn't work with the pattern: {e}")
    if body.apply and changes:
        os.makedirs(CAPTION_BACKUPS, exist_ok=True)
        with open(os.path.join(CAPTION_BACKUPS, f"{did}.json"), "w", encoding="utf-8") as fh:
            json.dump({"at": time.time(), "what": _describe(body), "captions": {f: before for f, before, _ in changes}}, fh)
        for f, _, after in changes:
            with open(caption_path(f), "w", encoding="utf-8") as fh:
                fh.write(after)
    return {"changed": len(changes), "total": len(files), "applied": body.apply and bool(changes),
            "examples": [{"name": os.path.relpath(f, d["path"]), "before": before, "after": after}
                         for f, before, after in changes[:6]]}


def _caption_backup(did: str) -> dict | None:
    try:
        with open(os.path.join(CAPTION_BACKUPS, f"{did}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@router.get("/datasets/{did}/captions/undo")
def caption_undo_info(did: str):
    dataset_or_404(did)
    b = _caption_backup(did)
    return {"available": False} if not b else {"available": True, "at": b["at"], "what": b["what"], "count": len(b["captions"])}


@router.post("/datasets/{did}/captions/undo")
def caption_undo(did: str):
    """Put back the captions the last bulk edit changed (a file that had none loses the one it was given)."""
    d = dataset_or_404(did)
    b = _caption_backup(did)
    if not b:
        raise HTTPException(400, "There's no bulk edit to undo")
    root = os.path.abspath(d["path"]) + os.sep
    restored = 0
    for path, before in b["captions"].items():
        if not os.path.abspath(path).startswith(root):
            continue
        if before is None:
            if os.path.exists(caption_path(path)):
                os.remove(caption_path(path))
        else:
            with open(caption_path(path), "w", encoding="utf-8") as f:
                f.write(before)
        restored += 1
    os.remove(os.path.join(CAPTION_BACKUPS, f"{did}.json"))
    return {"restored": restored}


class AutoCaptionIn(BaseModel):
    """Choices left out (None) are the ones this dataset used last time, else the defaults."""
    overwrite: bool = False
    trigger: str | None = None
    style: str | None = None  # a caption style id
    clip_frames: int | None = None  # clips: describe the first that many frames (at 24 fps); 0 for all of it
    speech: str | None = None  # clips with sound: a Whisper model to transcribe them with; "" for none


def caption_style(settings: dict, wanted: str | None) -> dict | None:
    styles = settings["caption_styles"]
    by_id = {st["id"]: st for st in styles["styles"]}
    return by_id.get(wanted or "") or by_id.get(styles["default"]) or (styles["styles"] or [None])[0]


@router.post("/datasets/{did}/autocaption")
def autocaption(did: str, body: AutoCaptionIn):
    d = dataset_or_404(did)
    cur = captioner.caption_jobs.get(did)
    if cur and cur.running:
        raise HTTPException(409, "Already captioning")
    files = list_media(d["path"]) if os.path.isdir(d["path"]) else []
    if files and all(media_kind(f) == "audio" for f in files):
        raise HTTPException(400, "Auto-caption describes images and clips, so it can't caption music; write each "
                                 "song's style tags (and any lyrics after a [Lyrics] line) yourself")
    trigger = d["trigger"] if body.trigger is None else body.trigger
    settings = db.get_settings()
    last = d.get("caption_options") or {}
    known = {st["id"] for st in settings["caption_styles"]["styles"]}
    if body.style and body.style not in known:
        raise HTTPException(400, "That caption style no longer exists; pick another")
    style = caption_style(settings, body.style or last.get("style"))
    if style is None:
        raise HTTPException(400, "Add a caption style in Settings first")
    clip_frames = int(last.get("clip_frames", 0) if body.clip_frames is None else body.clip_frames)
    if not 0 <= clip_frames <= MAX_FRAMES:
        raise HTTPException(400, f"Clip frames must be 1–{MAX_FRAMES}, or 0 for whole clips")
    speech = str(last.get("speech", "") if body.speech is None else body.speech)
    if speech and speech not in WHISPER_MODELS:
        raise HTTPException(400, f"Unknown Whisper model {speech!r}")
    options = {"style": style["id"], "clip_frames": clip_frames, "speech": speech}
    db.execute("UPDATE datasets SET caption_options=? WHERE id=?", (json.dumps(options), did))  # for next time
    cfg = {**settings["captioner"], "prompt": style["prompt"]}
    captioner.caption_jobs[did] = captioner.CaptionJob(d, cfg, body.overwrite, trigger, clip_frames, speech,
                                                       settings["speech"]["device"], settings)
    return {**captioner.caption_jobs[did].to_json(), **options}


@router.get("/datasets/{did}/autocaption")
def autocaption_status(did: str):
    j = captioner.caption_jobs.get(did)
    return j.to_json() if j else {"running": False, "total": 0, "done": 0, "errors": []}


@router.delete("/datasets/{did}/autocaption")
def autocaption_cancel(did: str):
    j = captioner.caption_jobs.get(did)
    if j:
        j.cancel()
    return {"ok": True}


@router.post("/captioner/test")
def captioner_test(cfg: dict):
    """Whether the caption provider can answer: a server you run is asked; one Dainamo runs is checked (it starts
    when captioning does)."""
    settings = db.get_settings()
    settings = {**settings, "captioner": {**settings["captioner"], **cfg}}
    if provider(settings) != "server":
        return describe(settings)
    try:
        return captioner.test_endpoint(settings["captioner"])
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)


class SplitIn(BaseModel):
    """Split into clips (``clip_split.SplitJob``): a new dataset of ``name`` with clips for ``family``."""
    name: str
    family: str
    frames: int  # the clip length, one the family takes
    max_clips: int = 8  # per video; 0 for no cap
    scenes: bool = True  # keep each clip inside one shot
    keep_short: bool = True  # a shot too short for a clip gives a shorter one
    max_side: int = 1024  # shrink to this shorter side; 0 keeps the size
    captions: str = "copy"  # "copy": each clip gets its video's caption; "none"
    review: bool = False  # only plan the clips, to change them in the clip editor before they're cut


@router.post("/datasets/{did}/split")
def split_dataset(did: str, body: SplitIn):
    d = dataset_or_404(did)
    cur = clip_split.split_jobs.get(did)
    if cur and cur.running:
        raise HTTPException(409, "Already splitting this dataset")
    F = family_or_400(body.family)
    if "video" not in F.info.media:
        raise HTTPException(400, f"{F.info.label} doesn't train on clips")
    if body.frames not in clip_split.clip_lengths(F):
        raise HTTPException(400, f"{F.info.label} takes clips of {F.info.frame_rule or 'other'} frames, up to {MAX_FRAMES}")
    if not 0 <= body.max_clips <= 1000:
        raise HTTPException(400, "Clips per video must be 0–1000 (0 for no limit)")
    if body.max_side and not 64 <= body.max_side <= 8192:
        raise HTTPException(400, "The size to scale down to must be 64–8192 px, or 0 to keep it")
    if body.captions not in ("copy", "none"):
        raise HTTPException(400, f"Unknown caption choice {body.captions!r}")
    if not body.name.strip():
        raise HTTPException(400, "Name the new dataset")
    files = list_media(d["path"]) if os.path.isdir(d["path"]) else []
    if not any(media_kind(f) == "video" for f in files):
        raise HTTPException(400, "This dataset has no videos to split")
    target = create_dataset(DatasetIn(name=body.name, trigger=d["trigger"]))
    if d.get("caption_options"):
        db.execute("UPDATE datasets SET caption_options=? WHERE id=?", (json.dumps(d["caption_options"]), target["id"]))
    clip_split.split_jobs[did] = clip_split.SplitJob(d, target, F, body.model_dump(), cut=not body.review,
                                                     fresh=not body.review)
    return clip_split.split_jobs[did].to_json()


@router.get("/datasets/{did}/split")
def split_status(did: str):
    j = clip_split.split_jobs.get(did)
    return j.to_json() if j else {"running": False, "total": 0, "done": 0, "clips": 0, "errors": [], "target": None}


@router.delete("/datasets/{did}/split")
def split_cancel(did: str):
    j = clip_split.split_jobs.get(did)
    if j:
        j.cancel()
    return {"ok": True}


class DownloadIn(BaseModel):
    """Add from URL (``media_download.DownloadJob``), experimental."""
    url: str
    kind: str = "video"  # "video", or "audio": only the sound
    max_height: int = 1080  # one of media_download.HEIGHTS; 0: the best there is
    start: float | None = None  # seconds: only the part from here...
    end: float | None = None  # ...to here
    playlist: int = 0  # a playlist's first this many items; 0: just the one the URL names
    format: str = ""  # one of media_download.VIDEO_FORMATS or AUDIO_FORMATS, for the kind; "": its default


@router.post("/datasets/{did}/download")
def download_media(did: str, body: DownloadIn):
    d = dataset_or_404(did)
    cur = media_download.download_jobs.get(did)
    if cur and cur.running:
        raise HTTPException(409, "Already downloading into this dataset")
    url = body.url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise HTTPException(400, "Paste a web address starting with http:// or https://")
    if body.kind not in ("video", "audio"):
        raise HTTPException(400, f"Unknown kind {body.kind!r}")
    formats = media_download.AUDIO_FORMATS if body.kind == "audio" else media_download.VIDEO_FORMATS
    if body.format and body.format not in formats:
        raise HTTPException(400, f"Unknown {'sound' if body.kind == 'audio' else 'video'} format {body.format!r}")
    if body.max_height not in media_download.HEIGHTS:
        raise HTTPException(400, f"The largest size must be one of {', '.join(map(str, media_download.HEIGHTS[1:]))} (or 0: any)")
    if not 0 <= body.playlist <= media_download.MAX_ITEMS:
        raise HTTPException(400, f"A playlist download takes 1–{media_download.MAX_ITEMS} items")
    if (body.start is not None and body.start < 0) or (body.end is not None and body.end <= (body.start or 0)):
        raise HTTPException(400, "The part to download must end after it starts")
    if body.playlist and (body.start is not None or body.end is not None):
        raise HTTPException(400, "A part of a video can't be chosen for a whole playlist")
    if not os.path.isdir(d["path"]):
        raise HTTPException(400, f"The dataset's folder is missing: {d['path']}")
    tool = media_download.tool()
    if not tool["installed"]:
        raise HTTPException(400, "yt-dlp isn't installed: install it in the Add from URL window, or run the launcher again")
    if not tool["current"]:
        raise HTTPException(400, f"yt-dlp {tool['version']} is older than Dainamo needs ({tool['minimum']} or newer): "
                                 "update it in the Add from URL window")
    opts = {**body.model_dump(exclude={"url", "format"}), "fmt": body.format}
    media_download.download_jobs[did] = media_download.DownloadJob(d, url, opts)
    return media_download.download_jobs[did].to_json()


@router.get("/datasets/{did}/download")
def download_status(did: str):
    j = media_download.download_jobs.get(did)
    return j.to_json() if j else {"running": False, "count": 0, "saved": [], "errors": [], "cancelled": False}


@router.delete("/datasets/{did}/download")
def download_cancel(did: str):
    j = media_download.download_jobs.get(did)
    if j:
        j.cancel()
    return {"ok": True}


@router.get("/ytdlp")
def ytdlp_info():
    """The yt-dlp Add from URL runs: {installed, version, current, minimum, bundled, network}."""
    return media_download.tool()


@router.post("/ytdlp/update")
def ytdlp_update():
    if any(j.running for j in media_download.download_jobs.values()):
        raise HTTPException(409, "Wait for the download to finish first")
    try:
        return media_download.update()
    except Exception as e:
        raise HTTPException(500, f"Couldn't update yt-dlp: {e}")


class ClipFitIn(BaseModel):
    profile_id: str
    resolution: int
    offload: float | None = None  # % of the blocks streamed from RAM (None: the family's default)
    low_vram: bool = False  # the text encoder's layers streamed too
    gpu_gb: float  # the GPU's memory


@router.post("/datasets/{did}/split/fit")
def split_fit(did: str, body: ClipFitIn):
    """The longest clip a job with the profile fits on the GPU at that resolution and offload
    (``clip_split.fit_frames``); {"error"} when the model files can't be read."""
    dataset_or_404(did)
    profile = profile_or_404(body.profile_id)
    F = get_family(profile["family"])
    if "video" not in F.info.media:
        raise HTTPException(400, f"{F.info.label} doesn't train on clips")
    res = check_resolution(body.resolution)
    if isinstance(res, list):
        res = max(res)
    offload = F.info.training_defaults.get("offload", 0) if body.offload is None else body.offload
    if not 0 <= offload <= 100:
        raise HTTPException(400, "Offload must be 0–100%")
    data = dict(profile["data"])
    path = os.path.expanduser(str(data.get("diffusion_model") or "").strip())
    if not os.path.isfile(path):
        return {"error": "The profile's diffusion model file wasn't found, so VRAM can't be projected."}
    try:
        return clip_split.fit_frames(F, {**data, "diffusion_model": path}, int(res), float(offload), body.gpu_gb,
                                     body.low_vram)
    except Exception as e:  # an unreadable or unexpected file: no projection, but no failure either
        return {"error": f"Couldn't read the model files to project VRAM: {e}"}

"""Split into clips: a copy of a dataset with each video cut into clips a video model trains on whole. Clips stay
inside one shot, spread over the video, scaled down, with their sound. The copy's manifest (``CLIPS_MANIFEST``)
records the video each clip came from, so training counts each video once (``dataset.epoch_items``), and the plan
the clips were cut from, which the clip editor changes: each video's clips, where they start, how long they are, and
what part of the frame they keep. Cutting again cuts only the clips that changed."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from ..engine.dataset import (CLIPS_MANIFEST, REFS_DIR, caption_path, list_media, media_kind, probe_video,
                              reference_images)
from . import db

SCENE_THRESHOLD = 0.3  # ffmpeg's scene score (0-1) from which a frame starts a new shot
MIN_SECONDS = 1.0  # a shot too short for a clip still gives a shorter one from this long (``keep_short``)
EDGE = 1  # frames left out beside each cut, so a clip never shows the frame before or after it
FIT_MARGIN_GB = 0.75  # headroom a clip length leaves on the GPU, as the VRAM panel's Fit to GPU does


def plan_clips(duration: float, cuts: list[float], fps: int, frames: int, snap=None, cap: int = 0,
               keep_short: bool = True) -> list[tuple[float, int]]:
    """(start in seconds, frames) of the clips to cut from a video ``duration`` s long with shot cuts at
    ``cuts``. Each shot gives as many clips of ``frames`` as fit, with equal gaps around them. A shot too
    short for one gives one shorter clip (snapped with ``snap``) if ``keep_short`` and it's at least
    ``MIN_SECONDS``. ``cap`` (0: none) keeps that many, spread over the video: full-length clips first,
    then the longest short ones."""
    edges = [0.0] + sorted(c for c in cuts if 0 < c < duration) + [duration]
    clip_s, margin = frames / fps, EDGE / fps
    full, short = [], []
    for a, b in zip(edges, edges[1:]):
        start, end = a + (margin if a > 0 else 0.0), b - margin  # the last frames of a file can be missing
        avail = int((end - start) * fps + 1e-3)
        n = avail // frames
        if n:
            gap = (end - start - n * clip_s) / (n + 1)
            full += [(round(start + gap + i * (clip_s + gap), 3), frames) for i in range(n)]
        elif keep_short and avail >= MIN_SECONDS * fps:
            f = snap(avail) if snap else avail
            if 2 <= f <= avail:
                short.append((round(start + (end - start - f / fps) / 2, 3), f))
    if cap and len(full) >= cap:
        return [full[round(i * (len(full) - 1) / (cap - 1))] for i in range(cap)] if cap > 1 else [full[len(full) // 2]]
    if cap:
        short = sorted(short, key=lambda c: -c[1])[: cap - len(full)]
    return sorted(full + short)


def clip_lengths(F) -> list[int]:
    """Every clip length (frames) family ``F`` trains on, up to the longest a job may ask for."""
    from .validation import MAX_FRAMES
    snap = F.snap_frames or (lambda n: n)
    return sorted({f for f in (snap(n) for n in range(2, MAX_FRAMES + 1)) if 2 <= f <= MAX_FRAMES})


def fit_frames(F, profile: dict, resolution: int, offload: float, gpu_gb: float, low_vram: bool = False) -> dict:
    """The VRAM projection of a job on clips of each length (``clip_lengths``) at ``resolution`` with ``offload``
    percent of the blocks streamed (and with ``low_vram``, the text encoder's layers), and the longest that fits
    ``gpu_gb``. -> {frames (None: none fits), fps, lengths: [{frames, seconds, peak_gb}]}; lengths stop at the
    first that doesn't fit at all."""
    from ..engine import memory
    settings = {**F.info.training_defaults, "resolution": resolution, "offload": offload, "low_vram": low_vram,
                "datasets": [{"resolution": resolution, "video": True}]}
    rows = []
    for n in clip_lengths(F):
        p = memory.project(F, profile, {**settings, "num_frames": n})
        peak = memory.peak_gb(p, memory.streamed_for(p, offload))
        rows.append({"frames": n, "seconds": round(n / F.fps, 2), "peak_gb": round(peak, 2)})
        if peak > gpu_gb:
            break
    fits = [r["frames"] for r in rows if r["peak_gb"] <= gpu_gb - FIT_MARGIN_GB]
    return {"frames": fits[-1] if fits else None, "fps": F.fps, "lengths": rows}


def find_cuts(path: str, on_proc=None, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Seconds at which a new shot starts in a video (ffmpeg's scene score on a small copy of each frame)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "json", path],
                         capture_output=True, text=True, timeout=60).stdout
    offset = float((json.loads(out or "{}").get("format") or {}).get("start_time") or 0.0)
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:v:0", "-an", "-vf",
                             f"scale=-2:144,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    if on_proc:
        on_proc(proc)
    _, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"couldn't look for shots: {err.strip().splitlines()[-1] if err.strip() else proc.returncode}")
    return [float(t) - offset for t in re.findall(r"Parsed_showinfo.*?pts_time:\s*([-0-9.]+)", err)]


def scale_filter(max_side: int) -> str:
    """Shrink so the shorter side is at most ``max_side`` (0: keep the size); x264 needs even sides."""
    if not max_side:
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    return (f"scale=w='if(gte(iw,ih),-2,trunc(min(iw,{max_side})/2)*2)':"
            f"h='if(gte(iw,ih),trunc(min(ih,{max_side})/2)*2,-2)':flags=lanczos")


def cut_clip(src: str, dst: str, start: float, frames: int, fps: int, max_side: int, audio: bool, on_proc=None,
             crop: list[int] | None = None):
    """``frames`` frames of ``src`` from ``start`` s at ``fps``, cropped to ``crop`` ([x, y, w, h] in the video's
    pixels, as it's shown) if given, scaled (``scale_filter``), with the sound."""
    # the frame count bounds the video and atrim the sound: -t would count from the first frame after the seek,
    # which can be a frame in, and lose the last one
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", src, "-map", "0:v:0"]
    cmd += ["-map", "0:a:0", "-af", f"atrim=duration={frames / fps:.6f}", "-c:a", "aac", "-b:a", "192k"] if audio else ["-an"]
    cut = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else ""
    cmd += ["-vf", f"{cut}fps={fps},{scale_filter(max_side)},setsar=1", "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    if on_proc:
        on_proc(proc)
    _, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError(err.strip().splitlines()[-1] if err.strip() else f"ffmpeg exited with {proc.returncode}")




def frame_at(path: str, start: float, n: int, fps: int, size: int) -> bytes:
    """Frame ``n`` of a clip cut from ``start`` s at ``fps`` (``cut_clip``), as a JPEG with its longer side at most
    ``size``: seeked and resampled as the cut is, so it's the clip's frame there, not the nearest one a browser shows."""
    scale = f"scale='if(gte(iw,ih),min(iw,{size}),-2)':'if(gte(iw,ih),-2,min(ih,{size}))'"
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-map", "0:v:0", "-an",
                          "-vf", f"fps={fps},select=gte(n\\,{n}),{scale}", "-frames:v", "1", "-f", "image2pipe",
                          "-c:v", "mjpeg", "-q:v", "3", "-"], capture_output=True, timeout=120)
    if out.returncode or not out.stdout:
        err = out.stderr.decode(errors="replace").strip()
        raise RuntimeError(err.splitlines()[-1] if err else "no frame there")
    return out.stdout


def filmstrip(path: str, duration: float, count: int, height: int) -> bytes:
    """``count`` frames spread over a video, side by side in one JPEG ``height`` px tall: the clip editor's timeline.
    Each is seeked to on its own, a few at a time, so a long video isn't decoded whole."""
    def grab(t: float) -> Image.Image | None:
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-map", "0:v:0", "-an",
                            "-vf", f"scale=-2:{height}", "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg",
                            "-q:v", "5", "-"], capture_output=True, timeout=120)
        if r.returncode or not r.stdout:
            return None
        with Image.open(io.BytesIO(r.stdout)) as im:
            return im.convert("RGB")

    with ThreadPoolExecutor(4) as pool:
        frames = list(pool.map(grab, [duration * (i + 0.5) / count for i in range(count)]))
    w = next((f.width for f in frames if f), None)
    if w is None:
        raise RuntimeError("no frames could be read")
    sheet = Image.new("RGB", (w * count, height))
    for i, f in enumerate(frames):
        if f:
            sheet.paste(f if f.size == (w, height) else f.resize((w, height)), (i * w, 0))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=80)
    return buf.getvalue()


# ---- the plan: what the clip editor changes, and cutting again cuts ----

MANIFEST_VERSION = 2  # 1: only the clips cut (before the clip editor)
SETTINGS = {"max_clips": 8, "scenes": True, "keep_short": True, "max_side": 1024, "captions": "copy"}
SIZES = (512, 768, 1024, 1280)  # Scale down to's sizes
MAX_PLANNED = 1000  # clips a video may have
_CLIP = re.compile(r"^(.+)_(\d+)\.mp4$")
_lock = threading.Lock()  # one change to a manifest at a time: the clip editor's saves, and a running split's


def clip_name(v: dict, clip: dict) -> str:
    """A clip's file in the dataset (with /): its video's stem and its number, which it keeps when it's changed."""
    return f"{v['stem']}_{clip['id']:03d}.mp4"


def latest_start(duration: float, frames: int, fps: int) -> float:
    """Where a clip of ``frames`` starts at the latest: a frame clear of the file's end, which can be missing."""
    return duration - (frames + EDGE) / fps


def same_cut(cut: dict, clip: dict, max_side: int) -> bool:
    """Whether a clip as it was cut (a manifest's ``clips`` entry) is ``clip`` as it's planned now."""
    return (abs(float(cut.get("start", -1)) - clip["start"]) < 5e-4 and cut.get("frames") == clip["frames"]
            and (cut.get("crop") or None) == (clip.get("crop") or None) and cut.get("max_side") == max_side)


def changes(m: dict) -> dict:
    """{cut: planned clips not cut as they're planned (new or changed), remove: cut clips the plan no longer has}."""
    planned = {clip_name(v, c): c for v in m["videos"].values() for c in v["clips"]}
    side = m["settings"]["max_side"]
    return {"cut": sum(1 for n, c in planned.items() if n not in m["clips"] or not same_cut(m["clips"][n], c, side)),
            "remove": sum(1 for n in m["clips"] if n not in planned)}


def _read(root: str) -> dict | None:
    try:
        with open(os.path.join(root, CLIPS_MANIFEST), encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def _write(root: str, m: dict):
    tmp = os.path.join(root, CLIPS_MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1)
    os.replace(tmp, os.path.join(root, CLIPS_MANIFEST))


def _scaled_to(root: str, clips: dict) -> int:
    """What a split from before the clip editor scaled to, from a clip's shorter side: one of Scale down to's sizes,
    else the size was kept. Either way a clip cut with it now comes out the same size as the old ones."""
    for name in clips:
        try:
            info = probe_video(os.path.join(root, *name.split("/")))
        except Exception:
            continue
        short = min(info["width"], info["height"])
        return short if short in SIZES else 0
    return SETTINGS["max_side"]


def _upgrade(m: dict, root: str) -> dict:
    """A manifest as the clip editor reads it. One from before it (version 1) has only the clips cut, so the plan is
    taken from them; each video is looked at again for its length and shots (``SplitJob._look``) when it's next
    needed."""
    m.setdefault("clips", {})
    m.setdefault("videos", {})
    m["settings"] = {**SETTINGS, **(m.get("settings") or {})}
    if m.get("version", 1) < MANIFEST_VERSION:
        if m["clips"]:
            m["settings"]["max_side"] = _scaled_to(root, m["clips"])
        for name, c in m["clips"].items():
            match = _CLIP.match(name)
            if not isinstance(c, dict) or not c.get("source") or not match:
                continue
            v = m["videos"].setdefault(c["source"], {"stem": match[1], "clips": [], "next": 1})
            cid = int(match[2])
            v["clips"].append({"id": cid, "start": round(float(c.get("start") or 0), 3),
                               "frames": int(c.get("frames") or 0), "crop": None})
            v["next"] = max(v["next"], cid + 1)
            c.update(id=cid, crop=None, max_side=m["settings"]["max_side"])
        m["version"] = MANIFEST_VERSION
    return m


def _drop_deleted(m: dict, root: str) -> bool:
    """Clips deleted from the dataset since they were cut (on its page, or by hand) leave the plan too, so cutting
    again doesn't bring them back. Whether there were any."""
    gone = [n for n in m["clips"] if not os.path.exists(os.path.join(root, *n.split("/")))]
    for n in gone:
        v = m["videos"].get(m["clips"].pop(n).get("source"))
        if v:
            v["clips"] = [c for c in v["clips"] if clip_name(v, c) != n]
    return bool(gone)


def load_manifest(root: str) -> dict | None:
    """A split dataset's manifest as the clip editor reads it (``_upgrade``, ``_drop_deleted``), or None."""
    with _lock:
        m = _read(root)
        if m is None:
            return None
        old = m.get("version", 1)
        m = _upgrade(m, root)
        if _drop_deleted(m, root) or old < MANIFEST_VERSION:
            _write(root, m)
        return m


def update_manifest(root: str, change) -> dict:
    """A split dataset's manifest, changed by ``change(manifest)`` and saved."""
    with _lock:
        m = _upgrade(_read(root) or {}, root)
        change(m)
        _write(root, m)
        return m


def summary(root: str) -> dict | None:
    """For the datasets list: a split dataset's source ({id, name}), and how many changes to its clips aren't cut."""
    m = _read(root)
    if not m or not (m.get("source") or {}).get("id"):
        return None
    if m.get("version", 1) < MANIFEST_VERSION:
        return {"source": m["source"], "changes": 0}
    c = changes(_upgrade(m, root))
    return {"source": m["source"], "changes": c["cut"] + c["remove"]}


def plan_video(v: dict, settings: dict, frames: int, F) -> list[dict]:
    """A video's clips as Split into clips plans them (``plan_clips``), numbered on from its last clip."""
    plan = plan_clips(v["duration"], v["cuts"], F.fps, frames, F.snap_frames, int(settings["max_clips"]),
                      bool(settings["keep_short"]))
    clips = [{"id": v["next"] + i, "start": start, "frames": n, "crop": None} for i, (start, n) in enumerate(plan)]
    v["next"] += len(clips)
    return clips


def _stem(rel: str, taken: set[str]) -> str:
    """The name a video's clips are numbered from: its path without the extension, or with it if another file's name
    starts the same (a.mp4 beside a.mov, or an image named a_001.png)."""
    stem = os.path.splitext(rel)[0]
    if any(t == stem or t.startswith(stem + "_") for t in taken):
        stem += "_" + os.path.splitext(rel)[1].lstrip(".").lower()
    taken.add(stem + "_")
    return stem


class SplitJob:
    """Fills ``target`` from ``source`` in a thread, as the target's manifest plans it. A video not looked at yet is
    first probed, its shots found (with the ``scenes`` setting) and its clips planned (``plan_video``). Then, with
    ``cut``, each video's new and changed clips are cut, the ones the plan no longer has are removed, and the rest are
    left as they are, with their captions; images and sound files the target doesn't have yet are copied, with their
    captions and reference images. Without ``cut`` the videos are only planned, for the clip editor. ``opts`` (a new
    split's): frames, max_clips (0: no cap), scenes, keep_short, max_side (0: keep the size), and captions ("copy": a
    new clip gets its video's caption; "none"); None keeps the manifest's. Cancelling a ``fresh`` split (a new
    dataset's first) removes the copy; cancelling any other stops it where it is."""

    def __init__(self, source: dict, target: dict, F, opts: dict | None = None, cut: bool = True, fresh: bool = False):
        self.source, self.target, self.F, self.opts, self.cut, self.fresh = source, target, F, opts, cut, fresh
        self.total = self.done = self.clips = self.removed = 0
        self.current = self.message = ""
        self.errors: list[str] = []
        self.running, self.cancelled = True, False
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def to_json(self) -> dict:
        return {"running": self.running, "mode": "cut" if self.cut else "plan", "total": self.total, "done": self.done,
                "clips": self.clips, "removed": self.removed, "current": os.path.basename(self.current),
                "message": self.message, "errors": self.errors[-20:], "cancelled": self.cancelled, "fresh": self.fresh,
                "target": {"id": self.target["id"], "name": self.target["name"]}}

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.proc and self.proc.poll() is None:
                self.proc.kill()

    def _track(self, proc: subprocess.Popen):
        with self.lock:
            self.proc = proc
            if self.cancelled:
                proc.kill()

    def _setup(self, m: dict):
        m.update(version=MANIFEST_VERSION, source={"id": self.source["id"], "name": self.source["name"]},
                 family=self.F.info.id, fps=self.F.fps)
        if self.opts:
            m["frames"] = int(self.opts["frames"])
            m["settings"] = {k: self.opts[k] for k in SETTINGS}
        _drop_deleted(m, self.target["path"])

    def _run(self):
        src_root, dst_root = self.source["path"], self.target["path"]
        try:
            m = update_manifest(dst_root, self._setup)
            files = list_media(src_root)
            self.total = len(files)
            # names a video's clips mustn't start with: every copied file's, so no two files share a caption
            taken = {os.path.splitext(os.path.relpath(f, src_root))[0].replace(os.sep, "/")
                     for f in files if media_kind(f) != "video"}
            taken |= {v["stem"] + "_" for v in m["videos"].values()}
            for f in files:
                if self.cancelled:
                    break
                self.current = f
                rel = os.path.relpath(f, src_root).replace(os.sep, "/")
                dst = os.path.join(dst_root, *rel.split("/"))
                try:
                    if media_kind(f) == "video":
                        self._video(f, rel, taken)
                    elif self.cut and not os.path.exists(dst):
                        self._copy(f, dst)
                except Exception as e:
                    if not self.cancelled:
                        self.errors.append(f"{rel}: {e}")
                self.done += 1
        except Exception as e:
            self.errors.append(str(e))
        finally:
            self.current = self.message = ""
            if self.cancelled and self.fresh:
                shutil.rmtree(dst_root, ignore_errors=True)
                db.execute("DELETE FROM datasets WHERE id=?", (self.target["id"],))
            self.running = False

    def _copy(self, src: str, dst: str, caption: bool = True):
        """A file with its caption (unless ``caption`` is False) and reference images, under ``dst``'s name."""
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        self._extras(src, dst, caption)

    def _extras(self, src: str, dst: str, caption: bool = True):
        if caption and os.path.exists(caption_path(src)):
            shutil.copy2(caption_path(src), caption_path(dst))
        src_stem = os.path.splitext(os.path.basename(src))[0]
        dst_stem = os.path.splitext(os.path.basename(dst))[0]
        for ref in reference_images(src):  # <stem>.png, <stem>_1.png, ... -> the same for the new name
            base, ext = os.path.splitext(os.path.basename(ref))
            folder = os.path.join(os.path.dirname(dst), REFS_DIR)
            os.makedirs(folder, exist_ok=True)
            shutil.copy2(ref, os.path.join(folder, dst_stem + base[len(src_stem):] + ext))

    def _video(self, path: str, rel: str, taken: set[str]):
        with _lock:
            v = _upgrade(_read(self.target["path"]) or {}, self.target["path"])["videos"].get(rel)
        if v is None or "duration" not in v or v.get("size") != os.path.getsize(path):
            self._look(path, rel, taken)
        if self.cut and not self.cancelled:
            self._sync(path, rel)

    def _look(self, path: str, rel: str, taken: set[str]):
        """Probe a video and find its shots; plan its clips if it's new, or keep a changed video's clips inside it."""
        with _lock:
            scenes = _upgrade(_read(self.target["path"]) or {}, self.target["path"])["settings"]["scenes"]
        self.message = "Finding shots" if scenes else "Reading the video"
        size, info = os.path.getsize(path), probe_video(path)
        cuts = find_cuts(path, self._track) if scenes else []
        if self.cancelled:
            return
        F = self.F

        def look(m: dict):
            v = m["videos"].get(rel)
            if v is None:
                v = m["videos"][rel] = {"stem": _stem(rel, taken), "clips": [], "next": 1}
            planned = "duration" in v
            v.update(size=size, duration=info["duration"], fps=info["fps"], width=info["width"],
                     height=info["height"], has_audio=info["has_audio"], cuts=[round(c, 3) for c in cuts])
            if not planned and not v["clips"]:
                v["clips"] = plan_video(v, m["settings"], int(m["frames"]), F)
                v["note"] = "" if v["clips"] else (f"no shot is long enough for a clip ({info['duration']:.1f} s"
                                                   + (f", {len(cuts) + 1} shots" if cuts else "") + ")")
            for c in v["clips"]:
                c["start"] = round(max(0.0, min(c["start"], latest_start(v["duration"], c["frames"], F.fps))), 3)

        v = update_manifest(self.target["path"], look)["videos"][rel]
        if v.get("note"):
            raise ValueError(v["note"])

    def _sync(self, path: str, rel: str):
        """Cut a video's clips as they're planned: the new and changed ones; the ones no longer planned removed."""
        root = self.target["path"]
        with _lock:
            m = _upgrade(_read(root) or {}, root)
        v, settings = m["videos"][rel], m["settings"]
        want = {clip_name(v, c): c for c in v["clips"]}
        have = {n: c for n, c in m["clips"].items() if c.get("source") == rel}
        for n in sorted(set(have) - set(want)):
            dst = os.path.join(root, *n.split("/"))
            for p in (dst, caption_path(dst), *reference_images(dst)):
                if os.path.exists(p):
                    os.remove(p)
            update_manifest(root, lambda m, n=n: m["clips"].pop(n, None))
            self.removed += 1
        todo = [(n, c) for n, c in want.items() if n not in have or not same_cut(have[n], c, settings["max_side"])
                or not os.path.exists(os.path.join(root, *n.split("/")))]
        for i, (n, c) in enumerate(todo, 1):
            if self.cancelled:
                return
            self.message = f"Cutting clip {i} of {len(todo)}"
            dst = os.path.join(root, *n.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            part = os.path.join(os.path.dirname(dst), "." + os.path.basename(dst) + ".part.mp4")  # hidden: not listed
            try:
                cut_clip(path, part, c["start"], c["frames"], self.F.fps, int(settings["max_side"]), v["has_audio"],
                         self._track, c.get("crop"))
                os.replace(part, dst)
            finally:
                if os.path.exists(part):
                    os.remove(part)
            if n not in have:  # a changed clip keeps its caption
                self._extras(path, dst, settings["captions"] == "copy")
            entry = {"source": rel, "id": c["id"], "start": c["start"], "frames": c["frames"], "crop": c.get("crop"),
                     "max_side": settings["max_side"]}
            update_manifest(root, lambda m, n=n, entry=entry: m["clips"].__setitem__(n, entry))
            self.clips += 1


split_jobs: dict[str, SplitJob] = {}  # by the id of the dataset being split


def job_for(target_id: str) -> SplitJob | None:
    """The last split into the dataset ``target_id``, if it's the last of its source's."""
    return next((j for j in split_jobs.values() if j.target["id"] == target_id), None)

"""Split into clips: a copy of a dataset with each video cut into clips a video model trains on whole. Clips stay
inside one shot, spread over the video, scaled down, with their sound. The copy's manifest (``CLIPS_MANIFEST``)
records the video each clip came from, so training counts each video once (``dataset.epoch_items``)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading

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


def cut_clip(src: str, dst: str, start: float, frames: int, fps: int, max_side: int, audio: bool, on_proc=None):
    """``frames`` frames of ``src`` from ``start`` s at ``fps``, scaled (``scale_filter``), with the sound."""
    # the frame count bounds the video and atrim the sound: -t would count from the first frame after the seek,
    # which can be a frame in, and lose the last one
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", src, "-map", "0:v:0"]
    cmd += ["-map", "0:a:0", "-af", f"atrim=duration={frames / fps:.6f}", "-c:a", "aac", "-b:a", "192k"] if audio else ["-an"]
    cmd += ["-vf", f"fps={fps},{scale_filter(max_side)},setsar=1", "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    if on_proc:
        on_proc(proc)
    _, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError(err.strip().splitlines()[-1] if err.strip() else f"ffmpeg exited with {proc.returncode}")


class SplitJob:
    """Fills ``target`` (a new, empty dataset) from ``source`` in a thread: each video is cut into clips
    (``plan_clips``), and images and sound files are copied as they are, with their captions and reference
    images. ``opts``: frames, max_clips (0: no cap), scenes, keep_short, max_side (0: keep the size), and
    captions ("copy": each clip gets its video's caption; "none"). A cancelled split removes the copy."""

    def __init__(self, source: dict, target: dict, F, opts: dict):
        self.source, self.target, self.F, self.opts = source, target, F, opts
        self.total = self.done = self.clips = 0
        self.current = self.message = ""
        self.errors: list[str] = []
        self.running, self.cancelled = True, False
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def to_json(self) -> dict:
        return {"running": self.running, "total": self.total, "done": self.done, "clips": self.clips,
                "current": os.path.basename(self.current), "message": self.message, "errors": self.errors[-20:],
                "cancelled": self.cancelled, "target": {"id": self.target["id"], "name": self.target["name"]}}

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

    def _run(self):
        src_root, dst_root = self.source["path"], self.target["path"]
        manifest = {"version": 1, "source": {"id": self.source["id"], "name": self.source["name"]},
                    "family": self.F.info.id, "fps": self.F.fps, "frames": int(self.opts["frames"]), "clips": {}}
        try:
            files = list_media(src_root)
            self.total = len(files)
            # names a clip mustn't take: every copied file's, so no two files share a caption
            taken = {os.path.splitext(os.path.relpath(f, src_root))[0] for f in files if media_kind(f) != "video"}
            for f in files:
                if self.cancelled:
                    break
                self.current = f
                rel = os.path.relpath(f, src_root)
                try:
                    if media_kind(f) == "video":
                        self._split(f, rel, taken, manifest)
                    else:
                        self._copy(f, os.path.join(dst_root, rel))
                except Exception as e:
                    if not self.cancelled:
                        self.errors.append(f"{rel}: {e}")
                self.done += 1
        finally:
            self.current = self.message = ""
            if self.cancelled:
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

    def _split(self, path: str, rel: str, taken: set[str], manifest: dict):
        o, F = self.opts, self.F
        info = probe_video(path)
        cuts = []
        if o["scenes"]:
            self.message = "Finding shots"
            cuts = find_cuts(path, self._track)
        plan = plan_clips(info["duration"], cuts, F.fps, int(o["frames"]), F.snap_frames, int(o["max_clips"]),
                          bool(o["keep_short"]))
        if not plan:
            raise ValueError(f"no shot is long enough for a clip ({info['duration']:.1f} s"
                             + (f", {len(cuts) + 1} shots" if cuts else "") + ")")
        stem = os.path.splitext(rel)[0]
        if any(t.startswith(stem + "_") for t in taken):  # a.mp4 beside a.mov, or an image named a_001.png
            stem += "_" + os.path.splitext(rel)[1].lstrip(".").lower()
        caption = o["captions"] == "copy"
        try:
            for i, (start, frames) in enumerate(plan, 1):
                if self.cancelled:
                    return
                self.message = f"Cutting clip {i} of {len(plan)}"
                name = f"{stem}_{i:03d}"
                taken.add(name)
                dst = os.path.join(self.target["path"], name + ".mp4")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                cut_clip(path, dst, start, frames, F.fps, int(o["max_side"]), info["has_audio"], self._track)
                self._extras(path, dst, caption)
                manifest["clips"][name.replace(os.sep, "/") + ".mp4"] = {
                    "source": rel.replace(os.sep, "/"), "start": start, "frames": frames}
                self.clips += 1
        finally:  # the clips cut so far count as the video's, even if a later one failed
            tmp = os.path.join(self.target["path"], CLIPS_MANIFEST + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1)
            os.replace(tmp, os.path.join(self.target["path"], CLIPS_MANIFEST))


split_jobs: dict[str, SplitJob] = {}  # by the id of the dataset being split

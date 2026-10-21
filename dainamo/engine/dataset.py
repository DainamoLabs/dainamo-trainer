"""Training dataset: discovery, aspect-ratio bucketing, and image/video/audio loading."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
REFS_DIR = "_refs"  # per-clip reference images for reference-conditioned families; never trained on directly
# written by Split into clips: which video each clip was cut from ({"clips": {relative clip path: {source, ...}}})
CLIPS_MANIFEST = ".clips.json"


def media_kind(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def list_media(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != REFS_DIR)
        for f in sorted(files):
            if not f.startswith(".") and media_kind(f):
                out.append(os.path.join(dirpath, f))
    return out


def split_clips(root: str) -> dict[str, dict]:
    """{clip path: {source, start, frames}} from a split dataset's manifest ({} for any other dataset): the
    video each clip was cut from (its path in the dataset it was split from), and where."""
    try:
        with open(os.path.join(root, CLIPS_MANIFEST), encoding="utf-8") as f:
            clips = json.load(f).get("clips") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    return {os.path.normpath(os.path.join(root, *rel.split("/"))): c for rel, c in clips.items()
            if isinstance(c, dict) and c.get("source")}


def clip_sources(root: str) -> dict[str, str]:
    """{clip path: the video it was cut from} (``split_clips``)."""
    return {path: str(c["source"]) for path, c in split_clips(root).items()}


def caption_path(media_path: str) -> str:
    return os.path.splitext(media_path)[0] + ".txt"


def read_caption(media_path: str) -> str:
    p = caption_path(media_path)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return ""


LYRICS_MARK = "[lyrics]"


def split_lyrics(caption: str) -> tuple[str, str]:
    """A song's caption -> (style, lyrics): lyrics follow a "[Lyrics]" line, any case (none: all style)."""
    lines = caption.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == LYRICS_MARK:
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i + 1:]).strip()
    return caption.strip(), ""


def apply_trigger(caption: str, trigger: str) -> str:
    trigger = (trigger or "").strip()
    if not trigger or trigger.lower() in caption.lower():
        return caption
    return f"{trigger}, {caption}" if caption else trigger


def resolutions(value) -> list[int]:
    """The training resolutions in a job's ``resolution`` setting: one number, or a list of them."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return sorted({int(v) for v in values})


SMALL = 0.75  # a file whose bucket has under this share of a large enough file's pixels "trains small"


def full_bucket(w: int, h: int, resolution: int, div: int) -> tuple[int, int]:
    """Largest (W, H), multiples of ``div``, with area <= resolution^2 and the source aspect: the bucket
    of a file of this shape that's large enough."""
    ar = w / h
    area = resolution * resolution
    return int(math.sqrt(area * ar) // div) * div, int(math.sqrt(area / ar) // div) * div


def bucket_size(w: int, h: int, resolution: int, div: int) -> tuple[int, int]:
    """The bucket a w x h file trains in: ``full_bucket``, shrunk to fit a file smaller than that."""
    bw, bh = full_bucket(w, h, resolution, div)
    # don't upscale: shrink the bucket to fit images smaller than the target area
    if w < bw or h < bh:
        s = min(w / bw, h / bh)
        bw, bh = max(div, int(bw * s // div) * div), max(div, int(bh * s // div) * div)
    return max(bw, div), max(bh, div)


def trains_small(w: int, h: int, resolution: int, div: int) -> bool:
    """Whether a w x h file trains well below ``resolution`` because it's smaller (it isn't upscaled): its
    bucket has under ``SMALL`` of the pixels of ``full_bucket``. Measured against that rather than
    resolution^2, since rounding to ``div`` alone can lose over 10% (a 3:2 photo at 384 with div 32 is
    448x288). The trainer's log and the dataset health check both use this, so they agree."""
    fw, fh = full_bucket(w, h, resolution, div)
    bw, bh = bucket_size(w, h, resolution, div)
    return bw * bh < SMALL * max(fw, div) * max(fh, div)


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB, with any transparency composited onto white (dropping alpha would expose junk colours)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, img).convert("RGB")
    return img.convert("RGB")


def load_and_crop(path: str, size: tuple[int, int], flip: bool = False) -> torch.Tensor:
    img = to_rgb(ImageOps.exif_transpose(Image.open(path)))
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    if flip:
        img = ImageOps.mirror(img)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# ---- video / audio ---------------------------------------------------------
def probe_video(path: str) -> dict:
    """{width, height, duration, fps, has_audio} via ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate:stream_tags=rotate:format=duration",
                          "-of", "json", path], capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    num, _, den = (video.get("avg_frame_rate") or "24/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 24.0
    w, h = int(video["width"]), int(video["height"])
    if str((video.get("tags") or {}).get("rotate", "0")) in ("90", "270", "-90"):
        w, h = h, w
    return {"width": w, "height": h, "duration": float(info.get("format", {}).get("duration") or 0.0),
            "fps": fps, "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def read_video_frames(path: str, size: tuple[int, int], frames: int, fps: int, flip: bool = False,
                      start: float = 0.0) -> torch.Tensor:
    """``frames`` frames resampled to ``fps``, cover-cropped to ``size`` -> (1, 3, T, H, W) in [-1, 1]."""
    w, h = size
    vf = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}"
    if flip:
        vf += ",hflip"
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-vf", vf, "-frames:v",
                          str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=600)
    data = np.frombuffer(out.stdout, dtype=np.uint8)
    got = data.size // (w * h * 3)
    if got < frames:
        raise ValueError(f"{path}: decoded {got} of {frames} frames")
    arr = data[: frames * w * h * 3].reshape(frames, h, w, 3).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)


def read_audio(path: str, seconds: float, sample_rate: int, channels: int = 2, start: float = 0.0) -> torch.Tensor:
    """(channels, samples) float32 waveform, zero-padded to ``seconds``."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-t", f"{seconds:.4f}",
                          "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-"],
                         capture_output=True, timeout=600)
    data = np.frombuffer(out.stdout, dtype=np.float32)
    n = int(round(seconds * sample_rate))
    wav = np.zeros((n, channels), dtype=np.float32)
    got = min(n, data.size // channels)
    wav[:got] = data[: got * channels].reshape(got, channels)
    return torch.from_numpy(wav.T.copy())


def probe_audio(path: str) -> dict:
    """{duration, sample_rate, channels} of a file's first audio stream via ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate,channels:format=duration", "-of", "json", path],
                         capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    stream = (info.get("streams") or [None])[0]
    if stream is None:
        raise ValueError(f"{path}: no audio stream")
    return {"duration": float(info.get("format", {}).get("duration") or 0.0),
            "sample_rate": int(stream.get("sample_rate") or 0), "channels": int(stream.get("channels") or 0)}


def first_frame(path: str, size: tuple[int, int], fps: int, flip: bool = False) -> Image.Image:
    px = read_video_frames(path, size, 1, fps, flip)[0, :, 0]
    return Image.fromarray(((px.permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8"))


def reference_images(media_path: str) -> list[str]:
    """``<dataset>/_refs/<stem>.png``, ``<stem>_1.png``, ... for a clip or image."""
    folder = os.path.join(os.path.dirname(media_path), REFS_DIR)
    if not os.path.isdir(folder):
        return []
    stem = os.path.splitext(os.path.basename(media_path))[0]
    out = []
    for f in sorted(os.listdir(folder)):
        base, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS and (base == stem or (base.startswith(stem + "_") and base[len(stem) + 1:].isdigit())):
            out.append(os.path.join(folder, f))
    return out


@dataclass
class Item:
    path: str
    caption: str
    size: tuple[int, int]
    flip: bool = False
    frames: int = 1          # >1 for video clips
    has_audio: bool = False
    resolution: int = 0      # the training resolution this item's bucket is for
    caption_dropout: float = 0.0  # its dataset's chance of training with an empty caption instead
    small: bool = False      # the file trains well below ``resolution``, being smaller (``trains_small``)
    seconds: float = 0.0     # audio: the clip's length, from ``start``
    start: float = 0.0
    lyrics: str = ""         # audio: lyrics the item trains with (whole songs only)
    source: str = ""         # a clip cut from a longer video (``clip_sources``): that video
    group: str = ""          # items of which an epoch trains one, picked at random: a source's clips at one size

    @property
    def is_video(self) -> bool:
        return self.frames > 1

    @property
    def is_audio(self) -> bool:
        return self.seconds > 0

    def key(self) -> str:
        st = os.stat(self.path)
        raw = f"{self.path}|{st.st_mtime_ns}|{self.size}|{self.flip}|{self.resolution}"
        if self.frames > 1:
            raw += f"|{self.frames}"
        if self.seconds:
            raw += f"|{self.start}|{self.seconds}"
        for ref in reference_images(self.path):  # editing a reference re-encodes the item
            raw += f"|{ref}|{os.stat(ref).st_mtime_ns}"
        return hashlib.sha1(raw.encode()).hexdigest()


def build_items(datasets: list[dict], resolution: int | list[int], div: int, flip: bool, log=print,
                video_frames: int | None = None, snap_frames=None, fps: int = 24,
                caption_dropout: float = 0.0) -> list[Item]:
    """``datasets``: [{path, trigger, repeats, resolution?, flip?, caption_dropout?, name?}] -> list of Items
    (repeats expanded). A dataset's own ``resolution`` (one or several), ``flip`` and ``caption_dropout`` are
    used instead of the arguments of those names.
    With several resolutions, each file is an item at each of them, except where a small image gives
    the same bucket at two (it isn't upscaled), which would only count it twice.
    With ``video_frames`` set (video families), clips are included at up to that many frames,
    snapped with ``snap_frames``; otherwise videos are skipped. Clips cut from one video (``clip_sources``)
    share a ``group`` per size, flip and repeat, so the video counts once (``epoch_items``)."""
    items = []
    for ds in datasets:
        res_list = resolutions(ds.get("resolution") or resolution)
        files = [f for f in list_media(ds["path"]) if media_kind(f) == "image" or (video_frames and media_kind(f) == "video")]
        if not files:
            log(f"Warning: no usable images{' or videos' if video_frames else ''} found in {ds['path']}")
            continue
        start, used, small = len(items), {"image": 0, "clip": 0}, 0
        sources, cut_from, n_cut = clip_sources(ds["path"]), set(), 0
        for f in files:
            frames, has_audio = 1, False
            if media_kind(f) == "video":
                try:
                    info = probe_video(f)
                except (ValueError, subprocess.SubprocessError, json.JSONDecodeError) as e:
                    log(f"Warning: skipping {os.path.basename(f)}: {e}")
                    continue
                available = int(info["duration"] * fps + 1e-3)  # ffprobe's 6 decimals: 107/24 s reads 4.458333
                frames = snap_frames(min(video_frames, available)) if snap_frames else min(video_frames, available)
                if frames < 2 or frames > available:
                    log(f"Warning: skipping {os.path.basename(f)}: too short ({available} frames at {fps} fps)")
                    continue
                w, h, has_audio = info["width"], info["height"], info["has_audio"]
            else:
                with Image.open(f) as im:
                    w, h = ImageOps.exif_transpose(im).size
            sizes = {}
            for res in res_list:
                sizes.setdefault(bucket_size(w, h, res, div), res)
            used["clip" if frames > 1 else "image"] += 1
            cut = sources.get(os.path.normpath(f)) if frames > 1 else None
            source = f"{ds['path']}|{cut}" if cut else ""
            if source:
                cut_from.add(source)
                n_cut += 1
            small += len(sizes) < len(res_list)
            cap = apply_trigger(read_caption(f), ds.get("trigger", ""))
            variants = [False, True] if ds.get("flip", flip) else [False]
            drop = float(ds.get("caption_dropout", caption_dropout))
            for size, res in sizes.items():
                small_at = trains_small(w, h, res, div)
                for fl in variants:
                    for rep in range(max(1, int(ds.get("repeats", 1)))):
                        group = f"{source}|{res}|{fl}|{rep}" if source else ""
                        items.append(Item(f, cap, size, fl, frames, has_audio, res, drop, small_at,
                                          source=source, group=group))
        repeats = max(1, int(ds.get("repeats", 1)))
        counts = " and ".join(f"{n} {kind}{'' if n == 1 else 's'}" for kind, n in used.items() if n)
        mine = items[start:]
        if cut_from:
            n = len(cut_from)
            counts += (f" ({'' if n_cut == used['clip'] else f'{n_cut} of them '}cut from {n} video"
                       f"{'' if n == 1 else 's'}, each training once per epoch)")
        per_epoch = sum(1 for it in mine if not it.group) + len({it.group for it in mine if it.group})
        log(f"{ds.get('name') or os.path.basename(os.path.normpath(ds['path']))}: {counts or 'nothing usable'} at "
            + ", ".join(map(str, res_list)) + (f", {repeats} repeats" if repeats > 1 else "")
            + f" -> {len(mine)} items" + (f", {per_epoch} per epoch" if per_epoch < len(mine) else "")
            + (f" ({small} too small for the larger sizes train at fewer of them)" if small else ""))
    return items


def song_clips(duration: float, clip_seconds: float) -> list[tuple[float, float]]:
    """(start, length) of the clips a song trains as: as many as fit to the nearest whole one, spread evenly
    from its start to its end, so they overlap a little or leave small gaps rather than drop its last part."""
    if duration <= clip_seconds:
        return [(0.0, float(duration))]
    n = max(1, math.floor(duration / clip_seconds + 0.5))
    step = (duration - clip_seconds) / (n - 1) if n > 1 else 0.0
    return [(round(i * step, 2), float(clip_seconds)) for i in range(n)]


def build_audio_items(datasets: list[dict], clip_seconds: float, flip: bool, log=print, caption_dropout: float = 0.0,
                      min_seconds: float = 5, max_seconds: float = 360) -> list[Item]:
    """Audio datasets -> Items (repeats expanded), in whole seconds. With ``clip_seconds``, each song is cut
    into clips of that length (``song_clips``; a song shorter than one clip but at least ``min_seconds`` long
    is one shorter clip); 0 trains whole songs, up to ``max_seconds`` of each. A caption's lyrics
    (``split_lyrics``) are only trained with whole songs, as where a clip's words fall isn't known.
    ``flip`` adds each item again with its stereo channels swapped."""
    items = []
    for ds in datasets:
        files = [f for f in list_media(ds["path"]) if media_kind(f) == "audio"]
        if not files:
            log(f"Warning: no audio files found in {ds['path']}")
            continue
        start, used = len(items), 0
        variants = [False, True] if ds.get("flip", flip) else [False]
        drop = float(ds.get("caption_dropout", caption_dropout))
        repeats = max(1, int(ds.get("repeats", 1)))
        for f in files:
            try:
                duration = math.floor(min(probe_audio(f)["duration"], max_seconds))
            except (ValueError, subprocess.SubprocessError, json.JSONDecodeError) as e:
                log(f"Warning: skipping {os.path.basename(f)}: {e}")
                continue
            if duration < min_seconds:
                log(f"Warning: skipping {os.path.basename(f)}: too short ({duration} s)")
                continue
            style, lyrics = split_lyrics(read_caption(f))
            style = apply_trigger(style, ds.get("trigger", ""))
            if clip_seconds:
                clips = song_clips(duration, clip_seconds)
                lyrics = ""
            else:
                clips = [(0.0, duration)]
            used += 1
            for at, length in clips:
                for fl in variants:
                    for _ in range(repeats):
                        items.append(Item(f, style, (0, 0), fl, caption_dropout=drop, seconds=float(length),
                                          start=float(at), lyrics=lyrics))
        log(f"{ds.get('name') or os.path.basename(os.path.normpath(ds['path']))}: {used} song"
            f"{'' if used == 1 else 's'}" + (f", {repeats} repeats" if repeats > 1 else "")
            + f" -> {len(items) - start} items")
    return items


def epoch_items(items: list[Item], rng: random.Random) -> list[int]:
    """The indices an epoch trains: every item, except that of each ``group`` (clips cut from one video) it
    trains one, picked at random, so a long video counts no more than a short one."""
    out, groups = [], {}
    for i, it in enumerate(items):
        if it.group:
            groups.setdefault(it.group, []).append(i)
        else:
            out.append(i)
    return out + [rng.choice(g) for g in groups.values()]


def make_batches(items: list[Item], batch_size: int, rng: random.Random) -> list[list[int]]:
    """One epoch (``epoch_items``): item indices grouped by bucket (size and frame count, or audio length),
    shuffled, as same-shape batches."""
    buckets: dict[tuple, list[int]] = {}
    for i in epoch_items(items, rng):
        it = items[i]
        buckets.setdefault((it.size, it.frames, it.seconds), []).append(i)
    batches = []
    for idxs in buckets.values():
        rng.shuffle(idxs)
        for j in range(0, len(idxs), batch_size):
            batches.append(idxs[j : j + batch_size])
    rng.shuffle(batches)
    return batches

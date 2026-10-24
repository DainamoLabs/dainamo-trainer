"""Training dataset: discovery, aspect-ratio bucketing, and image/video/audio loading."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
# per-item reference images (an edit's source images, a clip's references) kept inside the dataset, for when no
# source datasets are chosen; never trained on directly
REFS_DIR = "_refs"
# per-image loss masks (masked loss): ``<dataset>/_masks/<stem>.png`` beside ``<dataset>/<stem>.jpg``, greyscale,
# white where the loss counts and black where it doesn't; never trained on directly
MASKS_DIR = "_masks"
# where Upscale keeps the images it replaced (``engine.upscale``); never trained on
ORIGINALS_DIR = "_originals"
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
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in (REFS_DIR, MASKS_DIR, ORIGINALS_DIR))
        for f in sorted(files):
            if not f.startswith(".") and media_kind(f):
                out.append(os.path.join(dirpath, f))
    return out


def mask_path(media_path: str) -> str | None:
    """The loss mask of an image, if it has one: ``<folder>/_masks/<stem>.<image ext>`` beside it (PNG first)."""
    folder = os.path.join(os.path.dirname(media_path), MASKS_DIR)
    if not os.path.isdir(folder):
        return None
    stem = os.path.splitext(os.path.basename(media_path))[0]
    for ext in [".png"] + sorted(IMAGE_EXTS - {".png"}):
        p = os.path.join(folder, stem + ext)
        if os.path.isfile(p):
            return p
    return None


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


def image_size(path: str) -> tuple[int, int]:
    """An image's (width, height) the way it's shown, turned by its EXIF orientation (as ``build_items`` sizes it)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).size


def load_mask(path: str, image_size: tuple[int, int], size: tuple[int, int], flip: bool = False) -> torch.Tensor:
    """An image's loss mask (``mask_path``) cropped like the image: read as greyscale, stretched to the image's own
    size (``image_size``, so a mask of another size or aspect still lines up with it), then fitted to ``size`` and
    mirrored exactly as ``load_and_crop`` does the image -> (1, H, W) in [0, 1], 1 where the loss counts."""
    with Image.open(path) as im:
        mask = ImageOps.exif_transpose(im).convert("L")
    if mask.size != tuple(image_size):
        mask = mask.resize(tuple(image_size), Image.Resampling.BILINEAR)
    mask = ImageOps.fit(mask, size, Image.Resampling.LANCZOS)
    if flip:
        mask = ImageOps.mirror(mask)
    return torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)


# ---- video / audio ---------------------------------------------------------
def probe_video(path: str) -> dict:
    """{width, height, duration, fps, has_audio} via ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate:stream_tags=rotate:stream_side_data=rotation"
                          ":format=duration", "-of", "json", path], capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    num, _, den = (video.get("avg_frame_rate") or "24/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 24.0
    w, h = int(video["width"]), int(video["height"])
    # a phone clip shot upright: the rotation is a tag in older files and side data in ones ffmpeg 5+ wrote
    turns = [(video.get("tags") or {}).get("rotate")] + [d.get("rotation") for d in video.get("side_data_list") or []]
    if any(t is not None and abs(int(float(t))) % 180 == 90 for t in turns):
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
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if refers_to(f, stem)]


def refers_to(name: str, stem: str) -> bool:
    """Whether a file in a _refs folder is a reference of the file named ``stem``: ``<stem>.png``, ``<stem>_1.png``."""
    base, ext = os.path.splitext(name)
    return ext.lower() in IMAGE_EXTS and (base == stem or (base.startswith(stem + "_") and base[len(stem) + 1:].isdigit()))


def _name(path: str, root: str) -> str:
    """A file's path in ``root`` without its extension, "/"-separated: what it pairs by."""
    return os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, "/")


def source_index(root: str, dataset: str | None = None) -> dict[str, str]:
    """A source dataset's images by the name they pair with: their path in ``root`` without the extension
    ("sub/photo"), and their file name alone ("photo") where no other image has it, so sources kept in one folder
    pair with images in subfolders. ``dataset``: the folder of the dataset that pairs with it; when that's ``root``
    or inside it, its files aren't sources (a file doesn't pair with itself)."""
    holds = dataset is not None and (os.path.abspath(dataset) == os.path.abspath(root) or inside(dataset, root))
    by_path, by_name = {}, {}
    for f in list_media(root) if os.path.isdir(root) else []:
        if media_kind(f) == "image" and not (holds and inside(f, dataset)):
            name = _name(f, root)
            by_path[name] = f
            by_name.setdefault(name.rsplit("/", 1)[-1], []).append(f)
    return {**{k: v[0] for k, v in by_name.items() if len(v) == 1}, **by_path}


def paired_source(path: str, root: str, index: dict[str, str]) -> str | None:
    """The image of a source dataset (its ``source_index``) that a file of the dataset at ``root`` pairs with: the
    one of the same name, of any image type, in the same subfolder if it has subfolders."""
    name = _name(path, root)
    return index.get(name) or index.get(name.rsplit("/", 1)[-1])


def item_references(path: str, root: str, indexes: list[dict[str, str]]) -> list[str]:
    """The reference images a file of the dataset at ``root`` trains with: the image of its name in each source
    dataset (``source_index``) that has one, in order; with no source datasets, those in its _refs folder."""
    if indexes:
        return [s for s in (paired_source(path, root, ix) for ix in indexes) if s]
    return reference_images(path)


def inside(path: str, folder: str) -> bool:
    return os.path.abspath(path).startswith(os.path.abspath(folder) + os.sep)


def nested_sources(root: str, sources: list[dict]) -> list[str]:
    """The folders of ``sources`` (source datasets) inside the dataset at ``root``: their files are sources only,
    never trained on. (A source dataset whose folder holds the dataset's is the other way round: ``source_index``
    leaves the dataset's files out of it, and they train.)"""
    return [src["path"] for src in sources if inside(src["path"], root)]


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
    refs: list[str] = field(default_factory=list)  # reference images it trains with (``item_references``)
    mask: str = ""           # an image's loss mask (``mask_path``), for masked loss

    @property
    def is_video(self) -> bool:
        return self.frames > 1

    @property
    def is_audio(self) -> bool:
        return self.seconds > 0

    def key(self) -> str:
        """Its encoding's name in the cache: the file (path, size, modified time) and how it's prepared, so an
        edited or replaced file is encoded again."""
        st = os.stat(self.path)
        raw = f"{self.path}|{st.st_size}|{st.st_mtime_ns}|{self.size}|{self.flip}|{self.resolution}"
        if self.frames > 1:
            raw += f"|{self.frames}"
        if self.seconds:
            raw += f"|{self.start}|{self.seconds}"
        for ref in self.refs:  # editing or pairing another reference re-encodes the item
            raw += f"|{ref}|{os.stat(ref).st_mtime_ns}"
        if self.mask:  # so does adding or editing its mask (an item without one keeps the key it had)
            raw += f"|mask|{self.mask}|{os.stat(self.mask).st_mtime_ns}"
        return hashlib.sha1(raw.encode()).hexdigest()


def build_items(datasets: list[dict], resolution: int | list[int], div: int, flip: bool, log=print,
                video_frames: int | None = None, snap_frames=None, fps: int = 24,
                caption_dropout: float = 0.0, references: bool = False, masks: bool = True) -> list[Item]:
    """``datasets``: [{path, trigger, repeats, resolution?, flip?, caption_dropout?, name?, sources?}] -> list of
    Items (repeats expanded). A dataset's own ``resolution`` (one or several), ``flip`` and ``caption_dropout`` are
    used instead of the arguments of those names.
    With ``references`` (the model trains with per-item references: an edit's source images), each item has the
    reference images it pairs with (``item_references``): from the dataset's ``sources`` ([{path, name?}], source
    datasets, whose files never train as items themselves; ``nested_sources``, ``source_index``), or else its _refs
    folder.
    With ``masks``, each image has the loss mask in its folder's _masks folder, if it has one (``mask_path``; clips
    don't take them).
    With several resolutions, each file is an item at each of them, except where a small image gives
    the same bucket at two (it isn't upscaled), which would only count it twice.
    With ``video_frames`` set (video families), clips are included at up to that many frames,
    snapped with ``snap_frames``; otherwise videos are skipped. Clips cut from one video (``clip_sources``)
    share a ``group`` per size, flip and repeat, so the video counts once (``epoch_items``)."""
    items = []
    for ds in datasets:
        res_list = resolutions(ds.get("resolution") or resolution)
        sources = (ds.get("sources") or []) if references else []
        for src in sources:
            if not os.path.isdir(src["path"]):
                log(f"Warning: the folder of source dataset {src.get('name') or src['path']} wasn't found: {src['path']}")
        indexes = [source_index(src["path"], ds["path"]) for src in sources]
        inner = nested_sources(ds["path"], sources)  # source datasets in a subfolder
        files = [f for f in list_media(ds["path"]) if (media_kind(f) == "image" or (video_frames and media_kind(f) == "video"))
                 and not any(inside(f, p) for p in inner)]
        if not files:
            log(f"Warning: no usable images{' or videos' if video_frames else ''} found in {ds['path']}")
            continue
        start, used, small, paired, masked = len(items), {"image": 0, "clip": 0}, 0, 0, 0
        cuts, cut_from, n_cut = clip_sources(ds["path"]), set(), 0
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
            cut = cuts.get(os.path.normpath(f)) if frames > 1 else None
            source = f"{ds['path']}|{cut}" if cut else ""
            if source:
                cut_from.add(source)
                n_cut += 1
            small += len(sizes) < len(res_list)
            cap = apply_trigger(read_caption(f), ds.get("trigger", ""))
            refs = item_references(f, ds["path"], indexes) if references else []
            paired += bool(refs)
            mask = (mask_path(f) or "") if masks and frames == 1 else ""
            masked += bool(mask)
            variants = [False, True] if ds.get("flip", flip) else [False]
            drop = float(ds.get("caption_dropout", caption_dropout))
            for size, res in sizes.items():
                small_at = trains_small(w, h, res, div)
                for fl in variants:
                    for rep in range(max(1, int(ds.get("repeats", 1)))):
                        group = f"{source}|{res}|{fl}|{rep}" if source else ""
                        items.append(Item(f, cap, size, fl, frames, has_audio, res, drop, small_at,
                                          source=source, group=group, refs=refs, mask=mask))
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
        if sources or paired:
            n = sum(used.values())
            log(f"  {paired} of {n} with references"
                + (f" from {', '.join(src.get('name') or os.path.basename(os.path.normpath(src['path'])) for src in sources)}"
                   if sources else f" in {REFS_DIR}") + (f"; {n - paired} without" if paired < n else ""))
        if masked:
            log(f"  {masked} of {used['image']}{' images' if used['clip'] else ''} with masks in {MASKS_DIR}")
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
    is one shorter clip); 0 trains whole songs, up to ``max_seconds`` of each. A song's clips share a
    ``group`` per flip and repeat, so the song counts once per epoch however long it is (``epoch_items``).
    A caption's lyrics (``split_lyrics``) are only trained with whole songs, as where a clip's words fall
    isn't known. ``flip`` adds each item again with its stereo channels swapped."""
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
                    for rep in range(repeats):
                        items.append(Item(f, style, (0, 0), fl, caption_dropout=drop, seconds=float(length),
                                          start=float(at), lyrics=lyrics,
                                          group=f"{f}|{fl}|{rep}" if clip_seconds else ""))
        mine = items[start:]
        per_epoch = sum(1 for it in mine if not it.group) + len({it.group for it in mine if it.group})
        log(f"{ds.get('name') or os.path.basename(os.path.normpath(ds['path']))}: {used} song"
            f"{'' if used == 1 else 's'}" + (f", {repeats} repeats" if repeats > 1 else "")
            + f" -> {len(mine)} items" + (f", {per_epoch} per epoch (a clip of each song)"
                                          if per_epoch < len(mine) else ""))
    return items


def epoch_items(items: list[Item], rng: random.Random) -> list[int]:
    """The indices an epoch trains: every item, except that of each ``group`` (clips cut from one video, or
    a song's clips) it trains one, picked at random, so a long video or song counts no more than a short one."""
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

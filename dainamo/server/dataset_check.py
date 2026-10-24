"""Dataset health check: problems that quietly make training worse, found before a job runs.

Errors are files training can't use. Warnings are things that probably aren't what was meant (images that
train smaller than asked, near-duplicates, missing or cut-off captions, two files sharing one caption).
Notes are worth knowing but often intended.

Reading every file is the slow part, so what's learned about a file (size, a perceptual hash, or why it
can't be read) is kept per path, size and modification time for as long as the server runs."""

from __future__ import annotations

import os
import subprocess
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

from ..engine.dataset import (IMAGE_EXTS, MASKS_DIR, REFS_DIR, SMALL, caption_path, inside, list_media, mask_path,
                              media_kind, nested_sources, paired_source, probe_audio, probe_video, reference_images,
                              refers_to, source_index, split_clips, split_lyrics, trains_small)

DUPLICATE_BITS = 4  # difference hashes this close (of 64 bits) are near-duplicates: resized, re-saved, lightly edited
FPS = 24
MAX_LISTED = 60
SONG_RATE = 44100  # music: files below this sample rate sound duller than what the model makes
MIN_SONG, MAX_SONG = 5, 360  # seconds a music model trains on: shorter files are skipped, longer ones cut
MASK_ASPECT = 0.02  # a mask whose shape is further than this from its image's is noticeably stretched to fit it

_files: dict[tuple, dict] = {}
_lock = threading.Lock()


def _dhash(im: Image.Image) -> int:
    """64-bit difference hash: whether each pixel of a 9x8 greyscale thumbnail is brighter than its neighbour."""
    a = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), "big")


def file_info(path: str) -> dict:
    """{kind, width, height, hash (images), seconds (clips)}, {kind, seconds, sample_rate, channels} (audio)
    or {kind, error}."""
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    with _lock:
        if key in _files:
            return _files[key]
    kind = media_kind(path)
    info: dict = {"kind": kind}
    try:
        if kind == "image":
            with Image.open(path) as im:
                w, h = im.size
                if im.getexif().get(0x0112) in (5, 6, 7, 8):  # stored on its side
                    w, h = h, w
                im.draft("RGB", (128, 128))  # JPEGs decode small; every file is still read through
                im.load()
                info.update(width=w, height=h, hash=_dhash(im))
        elif kind == "audio":
            a = probe_audio(path)
            info.update(seconds=a["duration"], sample_rate=a["sample_rate"], channels=a["channels"])
        else:
            v = probe_video(path)
            info.update(width=v["width"], height=v["height"], seconds=v["duration"])
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as e:
        info = {"kind": kind, "error": str(e) or type(e).__name__}
    with _lock:
        _files[key] = info
    return info


def mask_size(path: str) -> tuple[int, int] | str:
    """A mask's (width, height) as training reads it (turned by its EXIF orientation), or why it can't be read; kept
    like ``file_info``."""
    st = os.stat(path)
    key = ("mask", path, st.st_size, st.st_mtime_ns)
    with _lock:
        if key in _files:
            return _files[key]["size"]
    try:
        with Image.open(path) as im:
            w, h = im.size
            size = (h, w) if im.getexif().get(0x0112) in (5, 6, 7, 8) else (w, h)
            im.draft("L", (128, 128))  # JPEGs decode small; every file is still read through
            im.load()
    except (OSError, ValueError, SyntaxError) as e:
        size = str(e) or type(e).__name__
    with _lock:
        _files[key] = {"size": size}
    return size


def _duplicate_groups(hashes: list[tuple[str, int]]) -> list[list[str]]:
    """Paths whose hashes are within DUPLICATE_BITS of each other, grouped (transitively)."""
    if len(hashes) < 2:
        return []
    h = np.array([v for _, v in hashes], dtype=np.uint64)
    parent = list(range(len(h)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h) - 1):
        close = np.nonzero(np.bitwise_count(h[i + 1:] ^ h[i]) <= DUPLICATE_BITS)[0]
        for j in close + i + 1:
            parent[root(int(j))] = root(i)
    groups: dict[int, list[str]] = {}
    for i, (path, _) in enumerate(hashes):
        groups.setdefault(root(i), []).append(path)
    return [g for g in groups.values() if len(g) > 1]


@lru_cache(maxsize=1)
def _tokenizer():
    """The Qwen3-VL tokenizer, if it's been downloaded already (the families with a caption limit read
    captions with Qwen tokenizers); else None, and lengths are estimated."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from ..engine.hf_assets import _TOKENIZER_FILES, QWEN3VL_REPO, REVISIONS
        return AutoTokenizer.from_pretrained(snapshot_download(
            QWEN3VL_REPO, revision=REVISIONS[QWEN3VL_REPO], allow_patterns=_TOKENIZER_FILES, local_files_only=True))
    except Exception:
        return None


def caption_tokens(texts: list[str]) -> tuple[list[int], bool]:
    """Token counts, and whether they're exact (else about four characters a token)."""
    tok = _tokenizer()
    if tok is None:
        return [len(t) // 4 for t in texts], False
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]], True


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else many or word + 's'}"


def pairing(root: str, files: list[str], sources: list[dict]) -> dict:
    """How ``files`` of the dataset at ``root`` pair with reference images, as training pairs them
    (``dataset.item_references``). With ``sources`` ([{name, path}], source datasets): {"from": "sources", "refs":
    {file: [its image in each source dataset, or None]}, "unused": [the images of each that no file pairs with]}.
    Without, from its _refs folders: {"from": "refs", "refs": {file: [references]}, "unused": [[files in _refs that
    pair with none]]}, or {"from": None} when it has none."""
    if sources:
        indexes = [source_index(src["path"], root) for src in sources]
        refs = {f: [paired_source(f, root, ix) for ix in indexes] for f in files}
        unused = []
        for i, ix in enumerate(indexes):
            taken = {r[i] for r in refs.values()}
            unused.append(sorted({p for p in ix.values() if p not in taken}))
        return {"from": "sources", "refs": refs, "unused": unused}
    folders = []
    for dirpath, dirnames, _ in os.walk(root) if os.path.isdir(root) else []:
        if REFS_DIR in dirnames:
            folders.append(os.path.join(dirpath, REFS_DIR))
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in (REFS_DIR, MASKS_DIR)]
    if not folders:
        return {"from": None}
    stems: dict[str, set[str]] = {}
    for f in files:
        stems.setdefault(os.path.dirname(f), set()).add(os.path.splitext(os.path.basename(f))[0])
    unused = []
    for folder in folders:
        mine = stems.get(os.path.dirname(folder), set())
        unused += [os.path.join(folder, n) for n in sorted(os.listdir(folder))
                   if os.path.splitext(n)[1].lower() in IMAGE_EXTS and not any(refers_to(n, st) for st in mine)]
    return {"from": "refs", "refs": {f: reference_images(f) for f in files}, "unused": [unused]}


def _check_pairs(add, root: str, used: list[str], sources: list[dict], info: dict) -> dict | None:
    """Findings on how a model that trains with reference images (``info``: its ``FamilyInfo.sources``) pairs the
    files ``used`` with them; returns {with, of, from, most (the most a file has)} for the summary, or None when nothing pairs (text-to-image)."""
    noun, label = info["noun"], info["label"].lower()
    for src in sources:
        if not os.path.isdir(src["path"]):
            add("source_missing", "error", f"The folder of {label} “{src['name']}” wasn't found",
                f"Nothing pairs from it: {src['path']}")
        elif inside(src["path"], root):
            add("source_inside", "info", f"{info['label']} “{src['name']}” is inside this dataset's folder",
                f"Its images are {noun}s only; they don't train as results.")
        elif os.path.abspath(src["path"]) == os.path.abspath(root):
            add("source_same", "warning", f"{info['label']} “{src['name']}” is this dataset's own folder",
                f"A file isn't its own {noun}, so nothing pairs from it. Choose the folder its {noun}s are in.")
        elif inside(root, src["path"]):
            add("source_contains", "info", f"{info['label']} “{src['name']}” holds this dataset's folder",
                f"Its images outside this dataset's folder are the {noun}s; this dataset's own files train as results.")
    p = pairing(root, used, sources)
    if not p["from"]:
        return None
    refs = p["refs"]
    paired = [f for f in used if any(refs[f])]
    without = [f for f in used if not any(refs[f])]
    where = (" or ".join(f"“{src['name']}”" for src in sources) if sources else f"its {REFS_DIR} folder")
    if without:
        add("no_source", "warning",
            f"No file pairs with a {noun}" if not paired and used else
            f"{_plural(len(without), 'file')} without a {noun}",
            f"They {info['without']}. Files pair by name: photo.png with photo.jpg in {where}."
            + (" Check the names match." if not paired else ""), without)
    if sources and len(sources) > 1:
        for i, src in enumerate(sources):
            partly = [f for f in paired if refs[f][i] is None]
            if partly:
                add(f"no_source_{i}", "warning", f"{_plural(len(partly), 'file')} without a {noun} in “{src['name']}”",
                    f"They train with their other {noun}s only.", partly)
    for i, unused in enumerate(p["unused"]):
        if unused:
            name = f"“{sources[i]['name']}”" if sources else REFS_DIR
            add(f"unused_source_{i}", "warning" if without else "info",
                f"{_plural(len(unused), noun)} in {name} pair{'s' if len(unused) == 1 else ''} with no file",
                "Nothing trains with them; maybe a name differs. They pair by name: photo.jpg with photo.png.",
                unused, under=sources[i]["path"] if sources else root)
    if sources and any(os.path.isdir(os.path.join(dp, REFS_DIR)) for dp, _, _ in os.walk(root)):
        add("refs_unused", "info", f"Its {REFS_DIR} folder isn't used",
            f"{'A ' + label + ' is' if len(sources) == 1 else info['label'] + 's are'} chosen, so files pair with "
            f"those instead.")
    return {"with": len(paired), "of": len(used), "from": p["from"],
            "most": max((sum(1 for r in refs[f] if r) for f in used), default=0)}


def check(root: str, trigger: str = "", resolution: list[int] | None = None, div: int = 16,
          video_frames: int | None = None, token_limit: int | None = None, clip_seconds: int | None = None,
          clip_length: int | None = None, auto_frames: bool = False, references: dict | None = None,
          sources: list[dict] | None = None) -> dict:
    """``resolution``: the sizes it will train at; ``video_frames``: the clip length when the model trains
    on clips (None: an image model), snapped to one it takes, and ``clip_length`` the one asked for (default
    ``video_frames``), or with ``auto_frames`` the longest a clip trains at, each training at its own length;
    ``token_limit``: caption tokens the model reads; ``clip_seconds``: a music model's clip length (0: whole
    songs; None: not a music model, which ``resolution`` etc. are then for). The result's ``longest_clip`` is
    the longest clip's frames at 24 fps. ``references``: the model's ``FamilyInfo.sources`` when it trains with
    reference images, which come from ``sources`` ([{name, path}], source datasets) or else the _refs folder; the
    result's ``pairs`` is then {with, of, from, most} (``_check_pairs``)."""
    sources = (sources or []) if references else []
    inner = nested_sources(root, sources)  # their files are sources, not this dataset's (as training reads them)
    files = [f for f in (list_media(root) if os.path.isdir(root) else []) if not any(inside(f, p) for p in inner)]
    rel = {f: os.path.relpath(f, root) for f in files}
    issues: list[dict] = []

    def add(id_, level, title, detail, paths=None, groups=None, under=None):
        """``under``: the folder ``paths`` are named from, when it isn't this dataset's (a source dataset's)."""
        entry = {"id": id_, "level": level, "title": title, "detail": detail}
        if paths is not None:
            entry.update(count=len(paths), files=[{"path": p, "name": rel.get(p) or os.path.relpath(p, under or root)}
                                                  for p in paths[:MAX_LISTED]])
        if groups is not None:
            entry.update(count=len(groups), groups=[[{"path": p, "name": rel[p]} for p in g] for g in groups[:20]])
        issues.append(entry)

    infos = {f: file_info(f) for f in files}
    bad = [f for f in files if "error" in infos[f]]
    if bad:
        add("unreadable", "error", f"{_plural(len(bad), 'file')} can't be read",
            "They're skipped in training. Re-save or remove them: " + infos[bad[0]]["error"][:160], bad)
    ok = [f for f in files if "error" not in infos[f]]
    clips = [f for f in ok if infos[f]["kind"] == "video"]
    songs = [f for f in ok if infos[f]["kind"] == "audio"]
    music = clip_seconds is not None
    if music:
        visual = [f for f in ok if infos[f]["kind"] != "audio"]
        if visual:
            add("visual_skipped", "info", f"{_plural(len(visual), 'image or clip', 'images and clips')} won't be used",
                "This model trains on audio only, so they're skipped.", visual)
        used = songs
        resolution = None
    else:
        if clips and video_frames is None:
            add("clips_skipped", "info", f"{_plural(len(clips), 'clip')} won't be used",
                "This model trains on images only, so clips are skipped.", clips)
        if songs:
            add("audio_skipped", "info", f"{_plural(len(songs), 'audio file')} won't be used",
                "This model trains on images" + (" and clips" if video_frames is not None else "")
                + ", so audio files are skipped (a music model trains on them).", songs)
        used = [f for f in ok if infos[f]["kind"] == "image" or (video_frames is not None and infos[f]["kind"] == "video")]

    if resolution:
        # at each size, as the trainer's log counts them: "2 files train below 1024, 1 below 512"
        small = {r: {f for f in used if trains_small(infos[f]["width"], infos[f]["height"], r, div)}
                 for r in sorted(resolution, reverse=True)}
        small = {r: fs for r, fs in small.items() if fs}
        if small:
            (top, first), *rest = small.items()
            title = (f"{_plural(len(first), 'file')} train{'s' if len(first) == 1 else ''} below {top}"
                     + "".join(f", {len(fs)} below {r}" for r, fs in rest))
            add("small", "warning", title,
                f"They have fewer pixels than {f'{top}×{top}' if not rest else 'those sizes'} at their shape, and files "
                f"are never upscaled, so they train at a smaller size (under {SMALL:.0%} of the pixels). Fine for a few; "
                "for many, pick a smaller resolution or use larger sources.",
                [f for f in used if any(f in fs for fs in small.values())])

    frames = {f: int(infos[f]["seconds"] * FPS + 1e-3) for f in clips}  # as the trainer counts them
    if video_frames:
        need = video_frames / FPS
        short = [f for f in clips if frames[f] < video_frames]
        too_short = [f for f in short if frames[f] < 2]
        if too_short:
            add("clip_too_short", "error", f"{_plural(len(too_short), 'clip')} too short to use",
                "Under two frames at 24 fps; they're skipped.", too_short)
        rest = [f for f in short if f not in too_short]
        if rest and not auto_frames:  # with auto, every clip trains at its own length
            add("short_clips", "info", f"{_plural(len(rest), 'clip')} shorter than {video_frames} frames ({need:.1f} s)",
                "They train at their own, shorter length.", rest)
        limit = clip_length or video_frames
        long = [f for f in clips if frames[f] > limit]
        if long:
            add("long_clips", "info", f"{_plural(len(long), 'clip')} longer than {limit} frames ({limit / FPS:.1f} s)",
                f"Only their first {limit / FPS:.1f} s train" + (", the longest a clip trains at" if auto_frames else
                                                                  "; an Auto clip length trains each whole, up to 10 s")
                + ". Split into clips (on the dataset's page) cuts them into clips that train whole.", long)
        cut = split_clips(root)
        grouped = [f for f in clips if os.path.normpath(f) in cut]
        if grouped:
            videos = len({cut[os.path.normpath(f)]["source"] for f in grouped})
            add("split_clips", "info", f"{_plural(len(grouped), 'clip')} cut from {_plural(videos, 'video')}",
                "Each video trains once per epoch, as one of its clips picked at random, so a long video counts "
                "no more than a short one.")

    if music:
        _check_songs(add, used, infos, clip_seconds)
    pairs = _check_pairs(add, root, used, sources, references) if references and not music else None

    groups = _duplicate_groups([(f, infos[f]["hash"]) for f in used if "hash" in infos[f]])
    if groups:
        n = sum(len(g) for g in groups)
        add("duplicates", "warning", f"{_plural(len(groups), 'group')} of near-duplicate images ({n} images)",
            "The same picture resized, re-saved or lightly edited counts twice as much in training. Keep one of each "
            "unless that's what you want.", groups=groups)

    # captions
    stems: dict[str, list[str]] = {}
    for f in files:
        stems.setdefault(os.path.splitext(f)[0], []).append(f)
    shared = [g for g in stems.values() if len(g) > 1]
    if shared:
        add("shared_caption", "warning", f"{_plural(len(shared), 'caption file')} shared by two files",
            "Files with the same name and different extensions (a.png and a.jpg) both use a.txt.", groups=shared)
    captions = {f: open(caption_path(f), encoding="utf-8").read().strip() if os.path.exists(caption_path(f)) else None
                for f in used}
    missing = [f for f in used if captions[f] is None]
    if missing:
        add("no_caption", "warning", f"{_plural(len(missing), 'file')} without a caption",
            f"They train on the trigger word “{trigger}” alone." if trigger else "They train with an empty caption.",
            missing)
    empty = [f for f in used if captions[f] == ""]
    if empty:
        add("empty_caption", "warning", f"{_plural(len(empty), 'caption file')} empty",
            f"They train on the trigger word “{trigger}” alone." if trigger else "They train with an empty caption.",
            empty)
    written = [f for f in used if captions[f]]
    if token_limit and written:
        counts, exact = caption_tokens([captions[f] for f in written])
        long = [f for f, n in zip(written, counts) if n > token_limit]
        if long:
            add("long_caption", "warning", f"{_plural(len(long), 'caption')} longer than the model reads",
                f"This model reads the first {token_limit} tokens of a caption{'' if exact else ' (counted roughly)'}; "
                "the rest is cut off.", long)
    if music and clip_seconds:
        sung = [f for f in written if split_lyrics(captions[f])[1]]
        if sung:
            add("lyrics_unused", "info", f"{_plural(len(sung), 'caption')} with lyrics",
                f"Training on {clip_seconds} s clips leaves lyrics out, since where a clip's words fall isn't known; "
                "set the clip length to 0 (whole songs) to train with them.", sung)
    if trigger:
        low = trigger.lower()
        without = [f for f in written if low not in captions[f].lower()]
        if without:
            add("no_trigger", "info", f"{_plural(len(without), 'caption')} without the trigger word",
                f"“{trigger}” is added at the start of these when training.", without)
    media_stems = set(stems)
    orphans, mask_dirs = [], []
    skip = {os.path.abspath(p) for p in inner}  # a source dataset's captions (and masks) are its own
    for dirpath, dirnames, names in os.walk(root):
        if MASKS_DIR in dirnames:
            mask_dirs.append(os.path.join(dirpath, MASKS_DIR))
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in (REFS_DIR, MASKS_DIR)
                       and os.path.abspath(os.path.join(dirpath, d)) not in skip]
        orphans += [os.path.join(dirpath, n) for n in sorted(names)
                    if n.lower().endswith(".txt") and os.path.join(dirpath, n)[:-4] not in media_stems]
    if orphans:
        add("orphan_caption", "info", f"{_plural(len(orphans), 'caption file')} without a media file",
            "Nothing trains with them; maybe the file they belonged to was renamed or removed.", orphans)
    if mask_dirs and not music:
        _check_masks(add, files, [f for f in used if infos[f]["kind"] == "image"], infos, mask_dirs,
                     video_frames is not None)

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: order[i["level"]])
    return {"files": len(files), "images": sum(1 for f in ok if infos[f]["kind"] == "image"), "videos": len(clips),
            "audio": len(songs), "issues": issues, "longest_clip": max(frames.values(), default=0), "pairs": pairs}


def _check_masks(add, files: list[str], images: list[str], infos: dict, folders: list[str], video: bool):
    """Masked loss's masks (``_masks/<name>.png`` beside an image, ``dataset.mask_path``) in ``folders``: how many of
    the ``images`` trained have one, masks no image of the dataset's ``files`` is named like, and masks of another
    shape than their image, which are stretched to fit it. A ``video`` model trains without them."""
    masks = {f: m for f in images if (m := mask_path(f))}
    if video:
        if masks:
            add("masks_unused", "info", f"{_plural(len(masks), 'mask')} won't be used",
                "Masked loss is for image models; this model trains on the whole picture.", list(masks.values()))
        return
    if masks:
        n = len(masks)
        add("masks", "info", f"{n} of {_plural(len(images), 'image')} {'has a mask' if n == 1 else 'have masks'}",
            "With Masked loss on (Create Job), the loss counts where each mask is white, and elsewhere at the "
            "Outside the mask weight.")
    named = {(os.path.dirname(f), os.path.splitext(os.path.basename(f))[0]) for f in files
             if media_kind(f) == "image"}
    lone = [os.path.join(folder, n) for folder in folders for n in sorted(os.listdir(folder))
            if not n.startswith(".") and os.path.splitext(n)[1].lower() in IMAGE_EXTS
            and (os.path.dirname(folder), os.path.splitext(n)[0]) not in named]
    if lone:
        add("mask_without_image", "info", f"{_plural(len(lone), 'mask')} without an image",
            f"Nothing uses them; maybe the image was renamed or removed. A mask has its image's name, in a "
            f"{MASKS_DIR} folder beside it: {MASKS_DIR}/photo.png is the mask of photo.jpg.", lone)
    sizes = {f: mask_size(m) for f, m in masks.items()}
    unreadable = {f: why for f, why in sizes.items() if isinstance(why, str)}
    if unreadable:
        add("mask_unreadable", "warning", f"{_plural(len(unreadable), 'mask')} can't be read",
            "Re-save them as greyscale PNGs: " + next(iter(unreadable.values()))[:160], [masks[f] for f in unreadable])

    def stretched(f):
        (w, h), (iw, ih) = sizes[f], (infos[f]["width"], infos[f]["height"])
        return not (w and h and iw and ih) or abs(w * ih / (h * iw) - 1) > MASK_ASPECT

    other = [masks[f] for f in masks if f not in unreadable and stretched(f)]
    if other:
        add("mask_shape", "warning", f"{_plural(len(other), 'mask')} of another shape than "
                                     f"{'its image' if len(other) == 1 else 'their images'}",
            "Each is stretched to its image's size, so it may not line up with it. Make each mask the same size as "
            "its image.", other)


def _check_songs(add, songs: list[str], infos: dict, clip_seconds: int):
    """A music model's files: too short to use, cut at the longest it trains, shorter than a clip, duller
    than it makes (low sample rate), or mono."""
    short = [f for f in songs if infos[f]["seconds"] < MIN_SONG]
    if short:
        add("song_too_short", "error", f"{_plural(len(short), 'file')} too short to use",
            f"Under {MIN_SONG} seconds; they're skipped.", short)
    ok = [f for f in songs if f not in short]
    long = [f for f in ok if infos[f]["seconds"] > MAX_SONG]
    if long:
        add("song_too_long", "info", f"{_plural(len(long), 'file')} longer than {MAX_SONG // 60} minutes",
            f"Only their first {MAX_SONG // 60} minutes train, the longest song the model writes.", long)
    if clip_seconds:
        under = [f for f in ok if infos[f]["seconds"] < clip_seconds]
        if under:
            add("song_short_clip", "info", f"{_plural(len(under), 'file')} shorter than a clip ({clip_seconds} s)",
                "Each trains as one shorter clip.", under)
    dull = [f for f in ok if 0 < infos[f]["sample_rate"] < SONG_RATE]
    if dull:
        rates = sorted({infos[f]["sample_rate"] for f in dull})
        add("low_sample_rate", "warning", f"{_plural(len(dull), 'file')} below {SONG_RATE / 1000:g} kHz",
            f"Recorded at {', '.join(f'{r / 1000:g} kHz' for r in rates)}, so they lack the highs, and the LoRA can "
            "learn to leave them out too.", dull)
    mono = [f for f in ok if infos[f]["channels"] == 1]
    if mono:
        add("mono", "info", f"{_plural(len(mono), 'mono file')}",
            "They train as stereo with both channels the same, which can make the LoRA's songs narrower.", mono)

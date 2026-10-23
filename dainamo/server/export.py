"""Export: datasets as one .zip, a folder each, streamed as it's made. Files are stored, not compressed (images,
videos and sound are compressed already), so the download starts at once and a large dataset is never held in
memory or written to disk whole."""

from __future__ import annotations

import os
import re
import zipfile

from ..engine.dataset import (CLIPS_MANIFEST, LYRICS_MARK, apply_trigger, caption_path, list_media, media_kind,
                              read_caption, reference_images, split_lyrics)

CHUNK = 1 << 20


class _Sink:
    """Where the zip is written: held until the response takes it."""

    def __init__(self):
        self.buf = bytearray()

    def write(self, b) -> int:
        self.buf += b
        return len(b)

    def flush(self):
        pass

    def take(self) -> bytes:
        out = bytes(self.buf)
        self.buf.clear()
        return out


def trained_caption(path: str, trigger: str) -> str:
    """A file's caption as training reads it: with the dataset's trigger word first if it isn't in it already (a
    song's on its style tags, before the lyrics). A file without a caption gets just the trigger word."""
    caption = read_caption(path)
    if media_kind(path) == "audio":
        style, lyrics = split_lyrics(caption)
        style = apply_trigger(style, trigger)
        return f"{style}\n{LYRICS_MARK.title()}\n{lyrics}" if lyrics else style
    return apply_trigger(caption, trigger)


def folder_name(name: str, used: set[str]) -> str:
    """A dataset's folder in the zip: its name, made safe for a file system, and unique."""
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip(" .") or "dataset"
    out, n = base, 2
    while out.lower() in used:
        out, n = f"{base} ({n})", n + 1
    used.add(out.lower())
    return out


def _file(zf: zipfile.ZipFile, sink: _Sink, path: str, arc: str):
    try:
        src = open(path, "rb")
    except OSError:  # gone since it was listed
        return
    with src:
        info = zipfile.ZipInfo.from_file(path, arc, strict_timestamps=False)
        info.compress_type = zipfile.ZIP_STORED
        with zf.open(info, "w") as dst:
            while chunk := src.read(CHUNK):
                dst.write(chunk)
                yield sink.take()
    yield sink.take()


def _text(zf: zipfile.ZipFile, sink: _Sink, text: str, arc: str, like: str):
    info = zipfile.ZipInfo.from_file(like, arc, strict_timestamps=False)
    info.compress_type = zipfile.ZIP_STORED
    zf.writestr(info, text.encode("utf-8"))
    yield sink.take()


def zip_datasets(datasets: list[dict], with_trigger: bool):
    """The .zip's bytes, a piece at a time: per dataset, a folder with its images, videos and sound files where they
    are in it, each with its caption (as training reads it, with ``with_trigger``; else as it is) and reference
    images, and a split's manifest, so each video still trains once per epoch."""
    sink, used = _Sink(), set()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True, strict_timestamps=False) as zf:
        for d in datasets:
            root, top = d["path"], folder_name(d["name"], used)
            arc = lambda p: f"{top}/{os.path.relpath(p, root).replace(os.sep, '/')}"  # noqa: E731
            for f in list_media(root):
                yield from _file(zf, sink, f, arc(f))
                cap = caption_path(f)
                if with_trigger and (d.get("trigger") or os.path.exists(cap)):
                    yield from _text(zf, sink, trained_caption(f, d.get("trigger", "")), arc(cap), f)
                elif os.path.exists(cap):
                    yield from _file(zf, sink, cap, arc(cap))
                for ref in reference_images(f):
                    yield from _file(zf, sink, ref, arc(ref))
            manifest = os.path.join(root, CLIPS_MANIFEST)
            if os.path.exists(manifest):
                yield from _file(zf, sink, manifest, arc(manifest))
    yield sink.take()

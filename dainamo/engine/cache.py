"""The cache of encoded training data that jobs share: VAE payloads ("latents") and text conditions ("text").

Each kind is kept under ``<root>/<kind>/<signature>/``. The signature names what made the encodings: the model
family, the files whose weights encode them (by fingerprint, so a copy of a file elsewhere still matches), and
any settings or profile values that change them (``FamilyDriver.cache_inputs``). Jobs with the same signature
share the folder, so a job using the same VAE as another reuses its latents whatever diffusion model each
trains. Inside, a latent file is named by its item's key (the source file's path, size and modified time, and
how it's cropped) and a text file by its caption, so a new or edited file, or a changed caption, is encoded
and the rest are reused.
"""

from __future__ import annotations

import hashlib
import json
import os

from ..fsutil import replace

HEAD_BYTES = 1024 * 1024
MAX_HEADER = 100 * 1024 * 1024  # the largest safetensors header the format allows
SLICES, SLICE_BYTES = 64, 16 * 1024


def fingerprint(path: str | None) -> str:
    """A model file's identity from its size, its first MB (and the rest of a safetensors header, which lists
    every tensor's name, dtype and shape) and 64 slices of 16 KB spread through it: a couple of MB read, not
    the whole file. A copy of the file matches; another precision or a fine-tune of it doesn't. A folder is
    the fingerprints of its files."""
    if not path:
        return "none"
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        h = hashlib.sha1()
        for root, _, files in sorted(os.walk(path)):
            for name in sorted(files):
                p = os.path.join(root, name)
                h.update(f"{os.path.relpath(p, path)}:{fingerprint(p)}".encode())
        return h.hexdigest()
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        head = f.read(HEAD_BYTES)
        h.update(head)
        n = int.from_bytes(head[:8], "little") if len(head) >= 8 else 0
        if HEAD_BYTES < 8 + n <= min(size, MAX_HEADER):
            h.update(f.read(8 + n - HEAD_BYTES))
        for i in range(SLICES):
            f.seek(size * i // SLICES)
            h.update(f.read(SLICE_BYTES))
    return h.hexdigest()


def signature(driver, kind: str, settings: dict) -> dict:
    """What ``driver``'s family makes the cached ``kind`` ("latents" or "text") from."""
    inputs = dict(driver.cache_inputs(kind, settings))
    files = inputs.pop("files")
    return {"family": driver.info.id, "kind": kind, "version": driver.cache_version, "dtype": str(driver.dtype),
            "files": [fingerprint(p) for p in files], **inputs}


def folder(root: str, driver, kind: str, settings: dict) -> str:
    """The folder under ``root`` for ``kind`` encoded as ``driver`` and ``settings`` encode it, made if new, with
    a ``signature.json`` saying what made it (and the files' paths, for people reading it)."""
    sig = signature(driver, kind, settings)
    d = os.path.join(root, kind, hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16])
    path = os.path.join(d, "signature.json")
    if not os.path.exists(path):
        os.makedirs(d, exist_ok=True)
        paths = [os.path.expanduser(p) if p else None for p in driver.cache_inputs(kind, settings)["files"]]
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({**sig, "paths": paths}, f, indent=2)
        replace(tmp, path)
    return d

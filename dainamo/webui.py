"""Whether the built web UI (web/dist) is up to date with its source, for the launchers:

    python -m dainamo.webui check    exit 0 when web/dist was built from the source as it is now, else 1
    python -m dainamo.webui built    record that web/dist was just built from the source as it is now

web/dist isn't in git, so a pull changes the source without the build. The record is a hash of the source's
contents (not file times, which a checkout resets), kept inside web/dist so a deleted build counts as stale."""

from __future__ import annotations

import hashlib
import os
import sys

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
SOURCES = ["src", "public", "index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts"]
STAMP = os.path.join("dist", ".source-hash")


def source_hash(web: str = WEB) -> str:
    files = []
    for name in SOURCES:
        path = os.path.join(web, name)
        if os.path.isfile(path):
            files.append(path)
        for root, _, names in os.walk(path):
            files += [os.path.join(root, n) for n in names]
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(os.path.relpath(path, web).replace(os.sep, "/").encode() + b"\0")
        with open(path, "rb") as f:
            h.update(f.read() + b"\0")
    return h.hexdigest()


def is_current(web: str = WEB) -> bool:
    try:
        with open(os.path.join(web, STAMP)) as f:
            stamp = f.read().strip()
    except OSError:
        return False
    return os.path.isfile(os.path.join(web, "dist", "index.html")) and stamp == source_hash(web)


def mark_built(web: str = WEB) -> None:
    with open(os.path.join(web, STAMP), "w") as f:
        f.write(source_hash(web) + "\n")


def main():
    if sys.argv[1:] == ["check"]:
        sys.exit(0 if is_current() else 1)
    if sys.argv[1:] == ["built"]:
        mark_built()
        return
    sys.exit("usage: python -m dainamo.webui check|built")


if __name__ == "__main__":
    main()

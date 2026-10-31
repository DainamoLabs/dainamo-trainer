"""Finished files where people look for them: when a job completes, its final LoRA (or finetuned model) is put in the
output folder (settings ``output``: ``loras`` and ``models``; blank: ``db.OUTPUT_DIR``) under its own name, and any
checkpoint can be sent there too.

It's hardlinked when it can be, so it takes no more space and outlives the job's folder being deleted, else copied.
What a job put where is in its exports.json: when the job finishes again (trained on, or made again) the file it put
there is replaced, while a file of that name that isn't its own (another job's, or one you put there) never is: its
file is named "name (2).safetensors" instead."""

from __future__ import annotations

import errno
import json
import os
import shutil
import sys
import threading

from ..fsutil import replace
from . import db
from .queue import job_dir, read_status

SPARE = 1 << 30  # left free on the disk by a copy (a hardlink takes no space)
KINDS = ("loras", "models")


def kind(cfg: dict) -> str | None:
    """The output folder a job's files go in: "loras", "models" (a finetune's), or None: a RefMod (its job copies
    it where ``refmod_copy_to`` says)."""
    s = (cfg or {}).get("settings") or {}
    if s.get("refmod_enabled"):
        return None
    return "models" if s.get("finetune_enabled") else "loras"


def folder(which: str) -> str:
    chosen = str(db.get_settings()["output"].get(which) or "").strip()
    return os.path.abspath(os.path.expanduser(chosen)) if chosen else db.OUTPUT_DIR


def exports(job_id: str) -> dict[str, str]:
    """The job's files that are in an output folder: {its file: where}."""
    try:
        with open(os.path.join(job_dir(job_id), "exports.json")) as f:
            made = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in made.items() if isinstance(v, str) and os.path.isfile(v)}


def _free_name(out: str, file: str, src: str) -> str:
    """Where in ``out`` to put ``file``: under its name, or "name (2)" and on if that's taken by another file
    (a hardlink to ``src`` already there is it)."""
    stem, ext = os.path.splitext(file)
    n = 1
    while os.path.exists(path := os.path.join(out, file if n == 1 else f"{stem} ({n}){ext}")):
        if os.path.samefile(path, src):
            break
        n += 1
    return path


def export(job_id: str, file: str) -> str:
    """Put the job's output/``file`` in its output folder; returns where it is. Raises ValueError (a RefMod),
    FileNotFoundError, or OSError saying what went wrong."""
    row = db.query_one("SELECT config FROM jobs WHERE id=?", (job_id,))
    which = kind(row["config"]) if row else None
    if which is None:
        raise ValueError("A RefMod goes where its job's Copy to says")
    file = os.path.basename(file)
    src = os.path.join(job_dir(job_id), "output", file)
    if not os.path.isfile(src):
        raise FileNotFoundError(errno.ENOENT, f"{file} isn't in this job's output")
    out = folder(which)
    made = exports(job_id)
    dest = made.get(file)
    if not dest or os.path.dirname(dest) != out:  # its first time there, or the folder has changed since
        dest = _free_name(out, file, src)
    elif os.path.samefile(src, dest):
        return dest
    tmp = os.path.join(out, f".{os.path.basename(dest)}.partial")
    try:
        os.makedirs(out, exist_ok=True)
        if os.path.exists(tmp):  # what a copy cut short by a restart left
            os.remove(tmp)
        try:
            os.link(src, tmp)
        except OSError:  # another disk, or a file system without hardlinks: a copy, if there's room
            size = os.path.getsize(src)
            if shutil.disk_usage(out).free < size + SPARE:
                raise OSError(errno.ENOSPC, f"there isn't room to copy it ({size / 1e9:.1f} GB)")
            shutil.copyfile(src, tmp)
        replace(tmp, dest)
        made[file] = dest
        path = os.path.join(job_dir(job_id), "exports.json")
        with open(path + ".tmp", "w") as f:
            json.dump(made, f)
        replace(path + ".tmp", path)
    except OSError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise OSError(e.errno, f"Couldn't put {file} in {out}: {e.strerror or e}") from e
    return dest


def export_final(job_id: str) -> None:
    """A job that completed: its final file put in the output folder, said in its log. Never raises."""
    try:
        row = db.query_one("SELECT config FROM jobs WHERE id=?", (job_id,))
        if not row or kind(row["config"]) is None:
            return
        final = next((c for c in read_status(job_id).get("checkpoints") or [] if c.get("final")), None)
        if final is None:
            return
        msg = f"Put in the output folder: {export(job_id, final['file'])}"
    except Exception as e:
        msg = f"Warning: {e}"
    try:
        with open(os.path.join(job_dir(job_id), "log.txt"), "a", encoding="utf-8") as log:
            log.write(msg + "\n")
    except OSError:
        print(f"[outputs] job {job_id}: {msg}", file=sys.stderr)


def export_final_later(job_id: str) -> None:
    """``export_final`` in the background: a copy to another disk can take minutes."""
    threading.Thread(target=export_final, args=(job_id,), daemon=True).start()

"""The local LLM server Dainamo runs for the caption provider (Settings → Caption provider): its status, starting
and stopping it, its log, and which text encoders can write text."""

from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, HTTPException

from ...engine.textgen import EncoderError, detect
from .. import db
from ..llama_server import LOG_LINES, LlamaError, auto_context, llama

router = APIRouter(prefix="/api/llm")


@router.get("")
def status():
    """How it is, the context llama-server gets when sized automatically, and the llama-server on the PATH ("" if
    none)."""
    return {**llama.snapshot(), "auto_ctx": auto_context(db.get_settings()),
            "on_path": shutil.which("llama-server") or ""}


@router.post("/start")
def start():
    """Start it with the saved settings; the model loads in the background (GET /api/llm shows how it goes)."""
    try:
        llama.start(db.get_settings())
    except LlamaError as e:
        raise HTTPException(400, str(e))
    return status()


@router.post("/stop")
def stop():
    llama.stop()
    return status()


@router.get("/log")
def log(lines: int = 200):
    """The last lines it wrote (the whole log is in data/local-llm.log)."""
    return {"lines": list(llama.log)[-max(1, min(lines, LOG_LINES)):]}


def _encoder(path: str) -> dict:
    try:
        return {"path": path, "ok": True, **detect(path).to_json()}
    except EncoderError as e:
        return {"path": path, "ok": False, "error": str(e)}


@router.get("/encoder")
def encoder(path: str):
    """Which text encoder ``path`` is ({ok, name, whose, vision}), or why it can't write text ({ok: false, error})."""
    return _encoder(os.path.expanduser(path.strip()))


@router.get("/encoders")
def encoders():
    """The text encoder files the profiles use, each with the profiles using it and whether it can write text."""
    used: dict[str, list[str]] = {}
    for p in db.query("SELECT name, data FROM profiles ORDER BY name COLLATE NOCASE"):
        path = str((p["data"] or {}).get("text_encoder") or "").strip()
        if path and os.path.isfile(os.path.expanduser(path)):  # local files only: a hub path isn't downloaded here
            used.setdefault(os.path.abspath(os.path.expanduser(path)), []).append(p["name"])
    return [{**_encoder(path), "profiles": names} for path, names in used.items()]

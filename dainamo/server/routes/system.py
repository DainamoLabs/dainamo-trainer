"""GPUs and model families, app settings, notifications, and the file browser."""

from __future__ import annotations

import os
import re
import string

from fastapi import APIRouter, HTTPException

from ...engine.families import load as load_families
from ...engine.transcribe import MODELS as WHISPER_MODELS
from .. import db, hardware, notify
from .common import not_found

router = APIRouter(prefix="/api")


@router.get("/system")
def system():
    from ...engine.families.base import WORKFLOWS

    return {"gpus": hardware.gpus(), "ram": hardware.ram(),
            "families": [F.info.to_json() for F in load_families().values()],
            "workflows": [{"id": k, **v} for k, v in WORKFLOWS.items()],
            "whisper_models": [{"id": k, "size": size} for k, (_, size) in WHISPER_MODELS.items()],
            "output_dir": db.OUTPUT_DIR}


@router.get("/hardware")
def hardware_status():
    """The GPUs, the CPU and system RAM now, for the Queue's hardware cards."""
    return {"gpus": hardware.gpus(), "cpu": hardware.cpu(), "ram": hardware.ram()}


@router.get("/settings")
def get_settings():
    return db.get_settings()


def _web_url(value, what: str) -> str:
    url = str(value or "").strip()
    if url and not url.startswith(("http://", "https://")):
        raise HTTPException(400, f"{what} must start with http:// or https://")
    return url


def _check_notifications(v: dict) -> dict:
    if v.get("kind") not in notify.KINDS:
        raise HTTPException(400, f"Unknown notification service {v.get('kind')!r}")
    v["url"] = _web_url(v.get("url"), "The webhook URL")
    v["link"] = _web_url(v.get("link"), "The app address").rstrip("/")
    for key in notify.EVENTS.values():
        v[key] = bool(v.get(key))
    return v


def _check_enhancer(v: dict) -> dict:
    defaults = db.DEFAULT_SETTINGS["enhancer"]
    for kind in ("image", "edit", "video"):
        v[kind] = str(v.get(kind) or "").strip() or defaults[kind]  # blank goes back to the default
    try:
        v["max_tokens"] = int(v.get("max_tokens", defaults["max_tokens"]))
        v["temperature"] = float(v.get("temperature", defaults["temperature"]))
    except (TypeError, ValueError):
        raise HTTPException(400, "Prompt enhancer max tokens and temperature must be numbers")
    if not 16 <= v["max_tokens"] <= 8192 or not 0 <= v["temperature"] <= 2:
        raise HTTPException(400, "Prompt enhancer: max tokens 16–8192, temperature 0–2")
    return v


def _gpu_choice(v: dict, what: str, cpu: bool = True) -> str:
    choice = "auto" if v.get("gpu") in (None, "") else str(v["gpu"])
    if not cpu and choice != "auto" and not choice.isdigit():
        raise HTTPException(400, f"{what} runs on a GPU: 'auto' or a GPU's number")
    if choice not in ("auto", "cpu") and not choice.isdigit():
        raise HTTPException(400, f"{what} runs on 'auto', 'cpu' or a GPU's number")
    return choice


def _idle_minutes(v: dict, what: str) -> float:
    try:
        return max(0.0, float(v.get("idle_minutes", 10)))
    except (TypeError, ValueError):
        raise HTTPException(400, f"{what}'s idle minutes must be a number")


def _check_llama(v: dict) -> dict:
    from ..llama_server import check_extra_args

    v.pop("managed", None)  # before the caption provider setting
    for key in ("binary", "model", "mmproj", "extra_args"):
        v[key] = str(v.get(key) or "").strip()
    for key, what, exts in (("binary", "llama-server", ()), ("model", "The model", (".gguf",)),
                            ("mmproj", "The vision projector", (".gguf", ".mmproj"))):
        if not v[key]:
            continue
        if not os.path.isfile(os.path.expanduser(v[key])):
            raise HTTPException(400, f"{what} wasn't found: {v[key]}")
        if exts and not v[key].lower().endswith(exts):
            raise HTTPException(400, f"{what} must be a {' or '.join(exts)} file")
    v["gpu"] = _gpu_choice(v, "llama-server")
    v["idle_minutes"] = _idle_minutes(v, "llama-server")
    try:
        v["ctx_size"] = int(v.get("ctx_size") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "llama-server's context size must be a number")
    if v["ctx_size"] and not 2048 <= v["ctx_size"] <= 1048576:
        raise HTTPException(400, "llama-server's context size must be 2048–1048576 tokens, or 0 to size it to the requests")
    try:
        check_extra_args(v["extra_args"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return v


def _check_text_encoder(v: dict) -> dict:
    from ...engine.textgen import EncoderError, detect

    v["path"] = str(v.get("path") or "").strip()
    if v["path"]:
        try:
            detect(v["path"])
        except EncoderError as e:
            raise HTTPException(400, str(e))
    v["gpu"] = _gpu_choice(v, "The text encoder", cpu=False)
    v["idle_minutes"] = _idle_minutes(v, "The text encoder")
    return v


def _check_models(v: dict) -> dict:
    from ..model_files import resolve_models_folder

    folder = str(v.get("folder") or "").strip()
    if folder:
        folder = resolve_models_folder(folder)
        if not os.path.isdir(folder):
            raise HTTPException(400, f"The models folder wasn't found: {folder}")
    return {"folder": folder}


def _check_output(v: dict) -> dict:
    """The output folders: blank (``db.OUTPUT_DIR``), or a folder that's there."""
    out = {}
    for which, what in (("loras", "LoRAs"), ("models", "finetuned models")):
        folder = str(v.get(which) or "").strip()
        if folder:
            folder = os.path.abspath(os.path.expanduser(folder))
            if not os.path.isdir(folder):
                raise HTTPException(400, f"The folder for {what} wasn't found: {folder}")
        out[which] = folder
    return out


def _style_number(st: dict, key: str, what: str, kind, low: float, high: float):
    """A caption style's own temperature or max_tokens, checked; None (left out) when it's blank."""
    val = st.get(key)
    if val is None or val == "":
        return None
    try:
        val = kind(val)
    except (TypeError, ValueError):
        raise HTTPException(400, f"Caption style {st.get('name')!r}: {what} must be a number")
    if not low <= val <= high:
        raise HTTPException(400, f"Caption style {st.get('name')!r}: {what} must be {low:g}–{high:g}, or blank for "
                                 "the caption provider's")
    return val


def _check_caption_styles(v: dict) -> dict:
    from ..captioner import MAX_TOKENS

    styles, seen = [], set()
    for st in v.get("styles") or []:
        sid, name, prompt, system = (str(st.get(k) or "").strip() for k in ("id", "name", "prompt", "system"))
        if not re.fullmatch(r"[a-z0-9-]{1,40}", sid) or sid in seen:
            raise HTTPException(400, f"Caption style {name or sid!r} has a bad or repeated id")
        if not name:
            raise HTTPException(400, "Every caption style needs a name")
        if not prompt:
            raise HTTPException(400, f"Caption style {name!r} needs a prompt")
        seen.add(sid)
        # the optional fields only when set: a style without them uses the caption provider's settings
        out = {"id": sid, "name": name, "prompt": prompt, **({"system": system} if system else {})}
        for key, what, kind, low, high in (("temperature", "Temperature", float, 0, 2),
                                           ("max_tokens", "Max tokens", int, 16, MAX_TOKENS)):
            val = _style_number(st, key, what, kind, low, high)
            if val is not None:
                out[key] = val
        styles.append(out)
    if not 1 <= len(styles) <= 30:
        raise HTTPException(400, "Keep between 1 and 30 caption styles")
    default = v.get("default") if v.get("default") in seen else styles[0]["id"]
    return {"default": default, "styles": styles}


@router.put("/settings")
def put_settings(body: dict):
    """Save the settings in ``body`` (whole sections or some of their fields), all or none."""
    from ..llama_server import PROVIDERS, llama

    current = db.get_settings()
    updates = {}
    for k, v in body.items():
        if k not in db.DEFAULT_SETTINGS:
            continue
        if isinstance(v, dict):
            v = {**current[k], **v}  # partial updates keep the other fields
        if k == "sandbox":
            v["idle_minutes"] = _idle_minutes({"idle_minutes": v.get("idle_minutes", 15)}, "The Sandbox")
        if k == "queue":
            if v.get("mode") not in ("sequential", "parallel_single", "parallel_multi"):
                raise HTTPException(400, f"Unknown queue mode {v.get('mode')!r}")
            v["gpus"] = sorted({int(g) for g in v.get("gpus") or [0]})
            v["max_per_gpu"] = max(1, int(v.get("max_per_gpu", 1)))
        if k == "notifications":
            v = _check_notifications(v)
        if k == "enhancer":
            v = _check_enhancer(v)
        if k == "caption_styles":
            v = _check_caption_styles(v)
        if k == "models":
            v = _check_models(v)
        if k == "output":
            v = _check_output(v)
        if k == "llama":
            v = _check_llama(v)
        if k == "text_encoder":
            v = _check_text_encoder(v)
        if k == "captioner":
            if v.get("provider") not in PROVIDERS:
                raise HTTPException(400, f"Unknown caption provider {v.get('provider')!r}")
            if v.get("video_mode") not in ("frames", "sheet"):
                raise HTTPException(400, "Clips are sent as 'frames' or 'sheet'")
            try:
                v["video_frames"] = int(v.get("video_frames", 6))
            except (TypeError, ValueError):
                raise HTTPException(400, "Frames per clip must be a number")
            if not 1 <= v["video_frames"] <= 16:
                raise HTTPException(400, "Frames per clip must be 1–16")
        if k == "speech":
            if v.get("model") not in WHISPER_MODELS:
                raise HTTPException(400, f"Unknown Whisper model {v.get('model')!r}")
            if v.get("device") not in ("auto", "cpu"):
                raise HTTPException(400, "Whisper runs on 'auto' (a free GPU, else the CPU) or 'cpu'")
        updates[k] = v
    after = {**current, **updates}
    chosen = after["captioner"]["provider"]
    if chosen == "llama" and not after["llama"]["model"]:
        raise HTTPException(400, "Choose a model (.gguf) for llama-server, or another caption provider")
    if chosen == "encoder" and not after["text_encoder"]["path"]:
        raise HTTPException(400, "Choose a text encoder file, or another caption provider")
    for k, v in updates.items():
        db.put_setting(k, v)
    if chosen != current["captioner"]["provider"]:  # the old provider's server isn't needed any more
        llama.stop_unused("Stopped: the caption provider changed")
    return db.get_settings()


@router.get("/notifications")
def notifications_status():
    """The latest delivery: {at, ok, error, title}, or {} when nothing has been sent since the server started."""
    return notify.last


@router.post("/notifications/test")
def notifications_test(cfg: dict):
    """Send a sample message with these (unsaved) settings."""
    merged = _check_notifications({**db.get_settings()["notifications"], **cfg})
    if not merged["url"]:
        raise HTTPException(400, "Enter a webhook URL first")
    msg = {"title": "Dainamo Trainer test", "body": "Notifications work. You'll get one like this when a job ends.",
           "url": f"{merged['link']}/#/queue" if merged["link"] else "", "status": "completed", "job_id": ""}
    try:
        notify.send(merged, msg)
    except Exception as e:
        return {"ok": False, "error": str(e) or type(e).__name__}
    return {"ok": True}


def _drives() -> list[str]:
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


@router.get("/fs")
def browse(path: str = "~", exts: str = ""):
    path = os.path.abspath(os.path.expanduser(path or "~"))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        not_found("Folder not found")
    allowed = {e.strip().lower() for e in exts.split(",") if e.strip()}
    dirs, files = [], []
    try:
        for e in sorted(os.scandir(path), key=lambda e: e.name.lower()):
            if e.name.startswith("."):
                continue
            if e.is_dir():
                dirs.append(e.name)
            elif not allowed or os.path.splitext(e.name)[1].lower() in allowed:
                files.append({"name": e.name, "size": e.stat().st_size})
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    # Windows: the drive letters, since "up" stops at a drive's root
    return {"path": path, "parent": os.path.dirname(path), "dirs": dirs, "files": files, "sep": os.sep,
            "roots": _drives() if os.name == "nt" else []}

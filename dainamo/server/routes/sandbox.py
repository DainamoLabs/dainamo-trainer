"""Sandbox: generating with a profile and LoRAs in a worker that keeps the models loaded."""

from __future__ import annotations

import json
import os
import re
import secrets

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel

from ...engine.families import get_family
from .. import captioner, db
from ..llama_server import LlamaError, llama, local_llm, provider
from .. import sandbox as sbx
from ..queue import job_dir, read_status
from ..validation import MAX_FRAMES, MAX_LYRICS, MAX_SONG_SECONDS, MIN_SONG_SECONDS
from .common import not_found, profile_or_404

router = APIRouter(prefix="/api/sandbox")


class SandboxIn(BaseModel):
    profile_id: str
    workflow: str
    prompt: str = ""
    negative: str = ""
    width: int = 1024
    height: int = 1024
    frames: int = 1
    seconds: float = 30  # music: the song's length
    lyrics: str = ""  # music: the lyrics, with [Verse] / [Chorus] section tags
    steps: int = 20
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "default"
    shift: float | None = None
    seed: int = -1
    count: int = 1
    loras: list[dict] = []
    model: str = ""  # a finetuned diffusion model (a finetune job's checkpoint) in place of the profile's
    images: list[str] = []
    refmods: list[dict] = []  # [{path, strength 0-1}]: RefMods as references (workflows with ``refmods``)
    audio: list[str] = []  # reference sound files (workflows with ``audio``, for a video)
    compare_base: bool = False
    gpu: int | None = None
    offload: float | None = None  # % of the diffusion model's blocks streamed from RAM (None: family default)
    low_vram: bool = False  # stream the text encoder's layers from RAM too
    source_prompt: str = ""  # what the user typed, when ``prompt`` is its enhanced version
    # compare checkpoints: {job_id, checkpoints: [paths], strengths: [numbers], base: bool}; every image of the
    # grid also has ``loras``
    grid: dict | None = None


MAX_GRID = 64  # images in one run of a grid (all seeds)
MAX_REFMODS = 4


def check_refmods(raw: list[dict], audio_ok: bool) -> tuple[list[dict], list[str]]:
    """RefMods for a generation: each a RefMod file at a strength 0–1 (an audio-only one if ``audio_ok``) ->
    (the ones used, the kinds of their references in order: "image", "video" or "audio")."""
    from ...engine import refmod

    if len(raw) > MAX_REFMODS:
        raise HTTPException(400, f"Up to {MAX_REFMODS} RefMods at once")
    out, kinds = [], []
    for r in raw:
        path = os.path.expanduser(str(r.get("path") or "").strip())
        if not path:
            continue
        name = os.path.basename(path)
        if not (os.path.isfile(path) and path.endswith(".safetensors")):
            raise HTTPException(400, f"RefMod not found (or not .safetensors): {path}")
        meta = refmod.read_meta(path)
        if meta is None:
            raise HTTPException(400, f"{name} isn't a RefMod")
        held = refmod.member_kinds(meta)
        if not held:
            raise HTTPException(400, f"{name} holds no references")
        if not audio_ok and all(k == "audio" for k in held):
            raise HTTPException(400, f"{name} only holds audio, which this workflow doesn't take")
        try:
            strength = float(r.get("strength", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "RefMod strength must be a number")
        if not 0 <= strength <= 1:
            raise HTTPException(400, "RefMod strength must be between 0 and 1")
        if strength > 0:  # at 0 it's left out
            out.append({"path": path, "strength": strength})
            kinds += held
    return out, kinds


def check_audio_refs(wf, raw: list[str], ref_kinds: list[str], visual: int, frames: int) -> list[str]:
    """A generation's reference sound files: each a file with sound in it, for a video, at most ``wf.audio["max"]``
    with the RefMods' audio references, beside at least one visual reference (``visual``: images and visual
    RefMod references), as the released ref2va pipeline asks."""
    from ...engine import dataset as ds
    from ...engine.families.base import WORKFLOWS

    paths = [os.path.expanduser(p.strip()) for p in raw if p and p.strip()]
    if paths and wf.audio is None:
        raise HTTPException(400, f"{WORKFLOWS[wf.id]['label']} takes no reference audio")
    if frames <= 1:
        if paths or (ref_kinds and all(k == "audio" for k in ref_kinds)):
            raise HTTPException(400, "Audio references need a video: set more than one frame")
        return []  # a bundle's audio sits out an image
    for p in paths:
        if not os.path.isfile(p):
            raise HTTPException(400, f"Audio not found: {p}")
        try:
            ds.probe_audio(p)
        except ValueError:
            raise HTTPException(400, f"{os.path.basename(p)} has no sound in it")
    n = len(paths) + ref_kinds.count("audio")
    if wf.audio and n > int(wf.audio["max"]):
        raise HTTPException(400, f"Up to {wf.audio['max']} audio references at once (audio RefMods count too)")
    if n and not visual:
        raise HTTPException(400, "An audio reference needs a reference image or a visual RefMod beside it")
    return paths


def check_grid(raw: dict, count: int) -> dict:
    """A checkpoint grid's request -> {job_id, job_name, checkpoints: [{path, label, step}], strengths, base}."""
    paths = [os.path.expanduser(str(p).strip()) for p in raw.get("checkpoints") or [] if str(p).strip()]
    if not 1 <= len(paths) <= 12:
        raise HTTPException(400, "Choose 1–12 checkpoints to compare")
    for p in paths:
        if not (os.path.isfile(p) and p.endswith(".safetensors")):
            raise HTTPException(400, f"Checkpoint not found: {p}")
    s = raw.get("strengths")
    try:
        strengths = [float(x) for x in (s.replace(",", " ").split() if isinstance(s, str) else s or [])]
    except (TypeError, ValueError):
        raise HTTPException(400, "Strengths must be numbers, e.g. 0.6, 0.8, 1.0")
    if not 1 <= len(strengths) <= 6 or any(not -10 <= x <= 10 for x in strengths):
        raise HTTPException(400, "Compare at 1–6 strengths between -10 and 10")
    base = bool(raw.get("base"))
    images = (len(paths) * len(strengths) + base) * count
    if images > MAX_GRID:
        raise HTTPException(400, f"That's {images} images; keep a grid to {MAX_GRID} (fewer checkpoints, strengths "
                                 "or images per run)")
    job_id = str(raw.get("job_id") or "")
    job = db.query_one("SELECT name FROM jobs WHERE id=?", (job_id,)) if job_id else None
    steps = {c["file"]: c for c in read_status(job_id).get("checkpoints") or []} if job else {}
    checkpoints = []
    for p in paths:
        c = steps.get(os.path.basename(p))
        label = os.path.basename(p) if c is None else "final" if c.get("final") else f"step {c['step']}"
        checkpoints.append({"path": p, "label": label, "step": c["step"] if c else None})
    return {"job_id": job_id, "job_name": job["name"] if job else "", "checkpoints": checkpoints,
            "strengths": strengths, "base": base}


def check_model(F, profile: dict, raw: str) -> str | None:
    """A finetuned diffusion model to generate with in place of the profile's: a file of the profile's model, such as
    a finetune job's checkpoint (None: the profile's own)."""
    path = os.path.expanduser(raw.strip())
    if not path:
        return None
    if not (os.path.isfile(path) and path.endswith(".safetensors")):
        raise HTTPException(400, f"Finetuned model not found (or not .safetensors): {path}")
    check = next(r for r in F.validate_profile({**profile["data"], "diffusion_model": path})
                 if r["field"] == "diffusion_model")
    if check["level"] == "error":
        raise HTTPException(400, f"{os.path.basename(path)}: {check['message']}")
    return path


def _workflow_for(F, profile: dict, workflow_id: str):
    variant = str(profile["data"].get("variant") or "")
    return next((w for w in F.info.workflows if w.id == workflow_id and (not w.variants or variant in w.variants)),
                None)


def check_sandbox(body: SandboxIn) -> tuple[dict, dict, int]:
    """-> (profile, params for the worker, gpu); 400 with a readable message on anything invalid."""
    from ...engine.families.base import WORKFLOWS
    from ...engine.sampling import SAMPLERS, SCHEDULERS

    profile = profile_or_404(body.profile_id)
    F = get_family(profile["family"])
    wf = _workflow_for(F, profile, body.workflow)
    if wf is None:
        label = WORKFLOWS.get(body.workflow, {}).get("label", body.workflow)
        raise HTTPException(400, f"{label} isn't available for {F.info.label}"
                                 + (f" ({profile['data'].get('variant')})" if profile["data"].get("variant") else ""))
    video = WORKFLOWS[wf.id]["output"] == "video"
    audio = WORKFLOWS[wf.id]["output"] == "audio"
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Enter a prompt")
    images = [os.path.expanduser(p.strip()) for p in body.images if p and p.strip()]
    if wf.images is None and images:
        raise HTTPException(400, f"{WORKFLOWS[wf.id]['label']} takes no images")
    if body.refmods and not wf.refmods:
        raise HTTPException(400, f"{WORKFLOWS[wf.id]['label']} takes no RefMods")
    refmods, ref_kinds = check_refmods(body.refmods, wf.audio is not None) if wf.refmods else ([], [])
    visual = sum(k != "audio" for k in ref_kinds)
    if wf.images is not None:
        lo, hi = int(wf.images["min"]), int(wf.images["max"])
        if visual:  # visual RefMods are references too
            lo = 0
        if not lo <= len(images) <= hi:
            want = f"{lo}" if lo == hi else f"{lo}–{hi}"
            raise HTTPException(400, f"{WORKFLOWS[wf.id]['label']} needs {want} {wf.images['label'].lower()}")
        for p in images:
            if not os.path.isfile(p):
                raise HTTPException(400, f"Image not found: {p}")
    div = F.bucket_divisibility
    width, height = body.width // div * div, body.height // div * div
    if audio:  # a song has no size
        width = height = 0
        if not MIN_SONG_SECONDS <= body.seconds <= MAX_SONG_SECONDS:
            raise HTTPException(400, f"Length must be {MIN_SONG_SECONDS}–{MAX_SONG_SECONDS} seconds")
        if len(body.lyrics) > MAX_LYRICS:
            raise HTTPException(400, f"Lyrics are limited to {MAX_LYRICS} characters")
    elif not (128 <= width <= 4096 and 128 <= height <= 4096):
        raise HTTPException(400, f"Width and height must be 128–4096 (multiples of {div})")
    frames = int(body.frames) if video else 1
    if video and not 1 <= frames <= MAX_FRAMES:
        raise HTTPException(400, f"Frames must be 1–{MAX_FRAMES}")
    audio_refs = check_audio_refs(wf, body.audio, ref_kinds, len(images) + visual, frames)
    if not 1 <= body.steps <= 200:
        raise HTTPException(400, "Steps must be 1–200")
    if not 0 <= body.cfg <= 30:
        raise HTTPException(400, "CFG must be 0–30")
    sampler, scheduler = body.sampler, body.scheduler
    if not F.supports_samplers:
        sampler, scheduler = "euler", "default"
    if sampler not in SAMPLERS or scheduler not in SCHEDULERS:
        raise HTTPException(400, "Unknown sampler or scheduler")
    if body.shift is not None and not 0.05 <= body.shift <= 100:
        raise HTTPException(400, "Shift must be 0.05–100 (blank for the model's own)")
    if not 1 <= body.count <= 8:
        raise HTTPException(400, "Count must be 1–8")
    if len(body.loras) > 8:
        raise HTTPException(400, "Up to 8 LoRAs at once")
    loras = []
    for lr in body.loras:
        path = os.path.expanduser(str(lr.get("path") or "").strip())
        if not path:
            continue
        if not (os.path.isfile(path) and path.endswith(".safetensors")):
            raise HTTPException(400, f"LoRA not found (or not .safetensors): {path}")
        try:
            strength = float(lr.get("strength", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "LoRA strength must be a number")
        if not -10 <= strength <= 10:
            raise HTTPException(400, "LoRA strength must be between -10 and 10")
        loras.append({"path": path, "strength": strength})
    offload = float(F.info.training_defaults.get("offload", 0) if body.offload is None else body.offload)
    if not 0 <= offload <= 100:
        raise HTTPException(400, "Offload must be 0–100%")
    seed = body.seed if body.seed >= 0 else secrets.randbelow(2**31)
    n = sbx.gpu_count()
    if sbx.mode() == "reserve":
        gpu = body.gpu if body.gpu is not None else n - 1
        if not 0 <= gpu < n:
            raise HTTPException(400, f"No GPU {gpu}")
    else:
        gpu = 0
    grid = check_grid(body.grid, int(body.count)) if body.grid else None
    model = check_model(F, profile, body.model)
    params = {"workflow": wf.id, "prompt": prompt, "negative": body.negative, "width": width, "height": height,
              "frames": frames, "steps": int(body.steps), "cfg": float(body.cfg), "sampler": sampler,
              "scheduler": scheduler, "shift": body.shift, "seed": int(seed), "count": int(body.count),
              "loras": loras, "model": model, "images": images, "refmods": refmods, "audio": audio_refs,
              "compare_base": bool(body.compare_base and loras and not grid),
              "offload": offload, "low_vram": bool(body.low_vram and F.info.streams_text_encoder), "grid": grid}
    if audio:
        params.update(seconds=int(body.seconds), lyrics=body.lyrics.strip())
    if body.source_prompt.strip() and body.source_prompt.strip() != prompt:
        params["source_prompt"] = body.source_prompt.strip()
    return profile, params, gpu


class EnhanceIn(BaseModel):
    profile_id: str
    workflow: str
    prompt: str
    seed: int | None = None
    images: list[str] = []


@router.post("/enhance")
def enhance(body: EnhanceIn):
    """The prompt rewritten by the local LLM (Settings: Caption provider), with the instruction for this kind of
    workflow; input images go along so the rewrite can describe them."""
    from ...engine.families.base import WORKFLOWS

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Enter a prompt")
    profile = profile_or_404(body.profile_id)
    wf = _workflow_for(get_family(profile["family"]), profile, body.workflow)
    if wf is None:
        raise HTTPException(400, f"Unknown workflow {body.workflow!r} for this profile")
    settings = db.get_settings()
    llm, cfg = settings["captioner"], settings["enhancer"]
    kind = "edit" if wf.id == "edit" else WORKFLOWS[wf.id]["output"]
    if kind not in cfg:
        raise HTTPException(400, f"Prompt enhancement isn't available for {WORKFLOWS[wf.id]['label']}")
    instruction = cfg[kind]
    images = [p for p in (os.path.expanduser(i.strip()) for i in body.images if i and i.strip()) if os.path.isfile(p)]
    if images and wf.images:
        instruction += (f" The attached image{'s are' if len(images) > 1 else ' is'} the "
                        f"{re.sub(r' *[(].*?[)]', '', wf.images['label']).lower()} for this generation; stay consistent with "
                        f"{'them' if len(images) > 1 else 'it'}.")
    try:
        with local_llm(settings) as llm:
            return {"prompt": captioner.enhance_prompt(prompt, instruction, images[:4], llm, cfg, body.seed)}
    except LlamaError as e:
        raise HTTPException(502, str(e))
    except httpx.ConnectError:
        if provider(settings) != "server":
            raise HTTPException(502, f"{llama.lost()}; its log is in Settings → Caption provider")
        raise HTTPException(502, f"Couldn't reach the local LLM at {llm['endpoint']}. Start it (Settings → Caption "
                                 "provider), or turn off Enhance prompt.")
    except httpx.TimeoutException:
        raise HTTPException(504, "The local LLM didn't answer in time")
    except (httpx.HTTPStatusError, RuntimeError, KeyError, IndexError, ValueError) as e:
        msg = str(e) or "the local LLM's answer couldn't be read"
        raise HTTPException(502, msg[:1].upper() + msg[1:])


@router.get("/image-size")
def image_size(path: str):
    """Width and height of an input image (as displayed, after EXIF rotation), for "Match input"."""
    path = os.path.expanduser(path.strip())
    if not os.path.isfile(path):
        not_found("Image not found")
    try:
        with Image.open(path) as im:
            w, h = ImageOps.exif_transpose(im).size
    except (OSError, ValueError):
        raise HTTPException(400, "Not an image")
    return {"width": w, "height": h}


@router.get("")
def sandbox_status():
    from ...engine.sampling import SAMPLERS, SCHEDULERS

    m = sbx.mode()
    return {"mode": m, "gpu_count": sbx.gpu_count(), "training_gpus": sorted(sbx.training_gpus()),
            "blocked": sbx.sandbox.check_gpu(0) if m == "idle" else None, "session": sbx.sandbox.snapshot(),
            "samplers": SAMPLERS, "schedulers": SCHEDULERS,
            "idle_minutes": db.get_settings().get("sandbox", {}).get("idle_minutes", 15),
            "runs": db.query("SELECT * FROM sandbox_runs ORDER BY created DESC LIMIT 60")}


@router.get("/defaults")
def sandbox_defaults(profile_id: str):
    import torch

    profile = profile_or_404(profile_id)
    F = get_family(profile["family"])
    d, td = F(profile["data"], torch.device("cpu")), F.info.training_defaults
    variant = str(profile["data"].get("variant") or "")
    return {"steps": d.sample_steps(), "cfg": d.cfg_scale(F.profile_value(profile["data"], "sample_cfg")),
            "negative": str(profile["data"].get("sample_negative") or ""),
            "width": int(td.get("sample_width") or 1024), "height": int(td.get("sample_height") or 1024),
            "frames": int(td.get("sample_frames") or 1), "seconds": int(td.get("sample_seconds") or 30),
            "supports_samplers": F.supports_samplers,
            "divisibility": F.bucket_divisibility, "offload": float(td.get("offload", 0)),
            "workflows": [w.to_json() for w in F.info.workflows if not w.variants or variant in w.variants]}


@router.get("/loras")
def sandbox_loras(family: str):
    """Checkpoints of this family's jobs, newest job first; the picker's recommendation is marked."""
    out = []
    for j in db.query("SELECT id, name, status, created, config FROM jobs WHERE family=? ORDER BY created DESC",
                      (family,)):
        s = (j["config"] or {}).get("settings", {})
        if s.get("refmod_enabled") or s.get("finetune_enabled"):
            continue  # its file is a RefMod (``sandbox_refmods``) or a whole model (``sandbox_models``)
        st = read_status(j["id"])
        # a checkpoint evaluation's recommendation, else training's
        rec = ((st.get("evaluation") or {}).get("recommended") or st.get("recommended") or {}).get("file")
        for c in sorted(st.get("checkpoints") or [], key=lambda c: (c.get("final", False), c.get("step", 0)), reverse=True):
            path = os.path.join(job_dir(j["id"]), "output", c["file"])
            if os.path.exists(path):
                out.append({"job_id": j["id"], "job_name": j["name"], "job_status": j["status"], "file": c["file"],
                            "path": path, "step": c.get("step"), "final": bool(c.get("final")),
                            "recommended": c["file"] == rec})
    return out


@router.get("/models")
def sandbox_models(family: str):
    """The finetuned models this family's finetune jobs saved, newest job first, its last checkpoint first."""
    out = []
    for j in db.query("SELECT id, name, status, config FROM jobs WHERE family=? ORDER BY created DESC", (family,)):
        if not (j["config"] or {}).get("settings", {}).get("finetune_enabled"):
            continue
        for c in sorted(read_status(j["id"]).get("checkpoints") or [],
                        key=lambda c: (c.get("final", False), c.get("step", 0)), reverse=True):
            path = os.path.join(job_dir(j["id"]), "output", c["file"])
            if os.path.exists(path):
                out.append({"job_id": j["id"], "job_name": j["name"], "job_status": j["status"], "file": c["file"],
                            "path": path, "step": c.get("step"), "final": bool(c.get("final"))})
    return out


@router.get("/refmods")
def sandbox_refmods(family: str):
    """The RefMods this family's jobs made, newest first."""
    out = []
    for j in db.query("SELECT id, name, config FROM jobs WHERE family=? ORDER BY created DESC", (family,)):
        if not (j["config"] or {}).get("settings", {}).get("refmod_enabled"):
            continue
        made = read_status(j["id"]).get("refmod") or {}
        path = os.path.join(job_dir(j["id"]), "output", made.get("file") or "")
        if made.get("file") and os.path.exists(path):
            out.append({"job_id": j["id"], "job_name": j["name"], "file": made["file"], "path": path,
                        **{k: made.get(k) for k in ("kind", "mode", "frames", "seconds", "tokens", "references")}})
    return out


@router.post("/generate")
def sandbox_generate(body: SandboxIn):
    profile, params, gpu = check_sandbox(body)
    reason = sbx.sandbox.check_gpu(gpu)
    if reason:
        raise HTTPException(409, reason)
    run = {"id": db.new_id(), "params": params}
    db.execute("INSERT INTO sandbox_runs (id, created, status, profile_id, profile_name, family, workflow, gpu, "
               "params, outputs) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (run["id"], db.now(), "queued", profile["id"], profile["name"], profile["family"], params["workflow"],
                gpu, json.dumps(params), "[]"))
    sbx.sandbox.submit(run, gpu, profile)
    return db.query_one("SELECT * FROM sandbox_runs WHERE id=?", (run["id"],))


@router.post("/runs/{rid}/cancel")
def sandbox_cancel(rid: str):
    sbx.sandbox.cancel(rid)
    return {"ok": True}


@router.delete("/runs/{rid}")
def sandbox_delete(rid: str):
    run = db.query_one("SELECT status FROM sandbox_runs WHERE id=?", (rid,)) or not_found("Run not found")
    if run["status"] in ("queued", "running"):
        raise HTTPException(409, "Cancel it first")
    sbx.sandbox.delete_run(rid)
    return {"ok": True}


@router.post("/release")
def sandbox_release():
    sbx.sandbox.release()
    return {"ok": True}


@router.get("/file/{rid}/{name}")
def sandbox_file(rid: str, name: str):
    path = os.path.join(sbx.run_dir(os.path.basename(rid)), os.path.basename(name))
    if not os.path.exists(path):
        not_found()
    return FileResponse(path)

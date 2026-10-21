"""Training jobs: creating and controlling them, and their progress, logs and files."""

from __future__ import annotations

import json
import os
import re
import shutil

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from ...engine.dataset import resolutions
from ...engine.families import get_family
from .. import db
from ..queue import job_dir, read_status, scheduler, send_control
from ..validation import check_evaluation, check_resolution, check_settings
from .common import dataset_or_404, job_or_404, not_found, profile_or_404

router = APIRouter(prefix="/api")


class JobDataset(BaseModel):
    id: str
    repeats: int = 1
    resolution: int | list[int] | str | None = None  # its training resolution(s); None: the job's ``resolution``
    flip: bool | None = None  # None: the job's ``flip``
    caption_dropout: float | None = None  # None: the job's ``caption_dropout``


class JobIn(BaseModel):
    name: str
    profile_id: str
    datasets: list[JobDataset]
    settings: dict
    gpu: str = "auto"
    output_name: str = ""


def _job_summary(j: dict) -> dict:
    st = read_status(j["id"])
    cfg = j.pop("config", {}) or {}
    j["progress"] = {k: st.get(k) for k in ("phase", "step", "total_steps", "loss", "avg_loss", "lr",
                                            "it_per_sec", "eta_seconds", "message", "vram_gb")}
    # the job's own step count (a job given more steps shows them before it starts again)
    j["progress"]["total_steps"] = cfg.get("settings", {}).get("steps") or j["progress"]["total_steps"]
    j["last_sample"] = st.get("samples", [])[-1] if st.get("samples") else None
    # a checkpoint evaluation's recommendation, else the one made in training
    j["recommended"] = (st.get("evaluation") or {}).get("recommended") or st.get("recommended")
    j["evaluating"] = "evaluate" in cfg
    j["datasets"] = [d.get("name") for d in cfg.get("datasets", [])]
    j["slider"] = bool(cfg.get("settings", {}).get("slider_enabled"))
    j["network"] = cfg.get("settings", {}).get("network_type") or "lora"
    try:
        j["media"] = get_family(j["family"]).info.media  # what its samples are: images, clips or songs
    except KeyError:
        j["media"] = []
    return j


@router.get("/jobs")
def list_jobs():
    return [_job_summary(j) for j in db.query("SELECT * FROM jobs ORDER BY position, created")]


@router.post("/jobs")
def create_job(body: JobIn):
    prof = profile_or_404(body.profile_id)
    Family = get_family(prof["family"])
    audio = "audio" in Family.info.media
    settings = check_settings({**Family.info.training_defaults, **body.settings}, Family.bucket_divisibility, audio)
    if settings["slider_enabled"] and not Family.info.sliders:
        raise HTTPException(400, f"{Family.info.label} can't train sliders yet")
    if not body.datasets and not settings["slider_enabled"]:  # a slider renders its own images without one
        raise HTTPException(400, "Select at least one dataset")
    dsets = []
    for jd in body.datasets:
        d = dataset_or_404(jd.id)
        try:
            res = None if audio else settings["resolution"] if jd.resolution is None else check_resolution(jd.resolution)
        except HTTPException as e:
            raise HTTPException(400, f"{d['name']}: {e.detail}")
        drop = float(settings.get("caption_dropout", 0.0) if jd.caption_dropout is None else jd.caption_dropout)
        if not 0 <= drop <= 1:
            raise HTTPException(400, f"{d['name']}: caption dropout must be 0–1")
        dsets.append({"id": d["id"], "name": d["name"], "path": d["path"], "trigger": d["trigger"],
                      "repeats": max(1, jd.repeats), "resolution": res,
                      "flip": bool(settings.get("flip", False) if jd.flip is None else jd.flip), "caption_dropout": drop})
    # the job's resolution is then all of its datasets' together (validation uses the largest)
    if dsets and not audio:
        every = sorted({r for d in dsets for r in resolutions(d["resolution"])})
        settings["resolution"] = every[0] if len(every) == 1 else every
    modes = [m[0] for m in Family.info.timestep_modes]
    if settings.get("timestep_sampling") not in modes:  # e.g. a Krea-only mode on a MOSS job
        settings["timestep_sampling"] = Family.info.training_defaults.get("timestep_sampling", modes[0])
    if body.gpu not in ("auto", "") and not body.gpu.isdigit():
        raise HTTPException(400, f"Invalid GPU {body.gpu!r}")
    val_dsets = []
    if settings.get("picker_enabled") and settings.get("val_dataset"):
        d = dataset_or_404(settings["val_dataset"])
        val_dsets.append({"id": d["id"], "name": d["name"], "path": d["path"], "trigger": d["trigger"], "repeats": 1})
    jid = db.new_id()
    config = {"id": jid, "name": body.name.strip() or "job", "output_name": body.output_name.strip(),
              "family": prof["family"], "profile_name": prof["name"], "profile": prof["data"],
              "datasets": dsets, "validation_datasets": val_dsets, "settings": settings}
    db.execute("INSERT INTO jobs (id, name, status, family, profile_name, gpu, position, created, config) "
               "VALUES (?,?,?,?,?,?,?,?,?)", (jid, config["name"], "queued", prof["family"], prof["name"],
                                              body.gpu, db.now(), db.now(), json.dumps(config)))
    return job_or_404(jid)


class EstimateIn(BaseModel):
    profile_id: str
    settings: dict = {}
    kind: str = "train"  # train: a job's settings; sample: a Sandbox generation (width, height, frames, loras)


@router.post("/estimate")
def estimate(body: EstimateIn):
    """VRAM projection per offload setting (``engine.memory``); {"error"} when the files can't be read."""
    from ...engine import memory

    profile = profile_or_404(body.profile_id)
    F = get_family(profile["family"])
    data = dict(profile["data"])
    path = os.path.expanduser(str(data.get("diffusion_model") or "").strip())
    if not os.path.isfile(path):
        return {"error": "The profile's diffusion model file wasn't found, so VRAM can't be projected."}
    try:
        return memory.project(F, {**data, "diffusion_model": path}, body.settings,
                              "sample" if body.kind == "sample" else "train")
    except Exception as e:  # an unreadable or unexpected file: no projection, but no failure either
        return {"error": f"Couldn't read the model files to project VRAM: {e}"}


@router.get("/jobs/{jid}")
def get_job(jid: str):
    j = job_or_404(jid)
    j["status_detail"] = read_status(jid)
    j["dir"] = job_dir(jid)
    j["resumable"] = os.path.exists(os.path.join(job_dir(jid), "state", "last.pt"))  # can train on from its last step
    return j


@router.get("/jobs/{jid}/metrics")
def job_metrics(jid: str, after: int = 0):
    """Points with step > ``after``, plus ``last`` (highest step logged) so the UI can tell
    when a restart has rewritten the file from step 1."""
    path = os.path.join(job_dir(jid), "metrics.jsonl")
    out, last = [], 0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last = max(last, m["step"])
                if m["step"] > after:
                    out.append(m)
    # a resumed run may re-log steps; keep the latest value per step
    return {"points": sorted({m["step"]: m for m in out}.values(), key=lambda m: m["step"]), "last": last}


@router.get("/jobs/{jid}/log", response_class=PlainTextResponse)
def job_log(jid: str, tail: int = 200):
    path = os.path.join(job_dir(jid), "log.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-tail:])


LOG_CHUNK = 256 * 1024


def decode_complete(data: bytes) -> tuple[str, int]:
    """Decode UTF-8, leaving a character cut off at the end for the next read: (text, bytes used)."""
    cut = len(data)
    for i in range(1, min(4, len(data)) + 1):
        b = data[-i]
        if b & 0xC0 == 0x80:  # continuation byte: look further back for the lead byte
            continue
        if b >= 0xC0 and i < (2 if b < 0xE0 else 3 if b < 0xF0 else 4):
            cut = len(data) - i
        break
    return data[:cut].decode("utf-8", errors="replace"), cut


@router.get("/jobs/{jid}/log/chunk")
def job_log_chunk(jid: str, offset: int = -1, file: str = ""):
    """The job's terminal output from byte ``offset`` on, for a live view. Pass back ``file`` and
    ``offset`` from the previous answer; the first call (or a replaced log) gets the last 256 KB,
    starting at a line, with ``reset`` set. ``more`` means another call right away will return more."""
    job_or_404(jid)
    path = os.path.join(job_dir(jid), "log.txt")
    try:
        st = os.stat(path)
    except OSError:
        return {"file": "", "start": 0, "offset": 0, "text": "", "reset": offset != 0, "more": False}
    fid = str(st.st_ino)
    reset = fid != file or not 0 <= offset <= st.st_size
    start = max(0, st.st_size - LOG_CHUNK) if reset else offset
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(LOG_CHUNK)
    if reset and start > 0:
        nl = data.find(b"\n")
        if nl >= 0:
            data, start = data[nl + 1:], start + nl + 1
    text, used = decode_complete(data)
    return {"file": fid, "start": start, "offset": start + used, "text": text, "reset": reset,
            "more": start + used < st.st_size and used > 0}


@router.get("/jobs/{jid}/log.txt")
def job_log_file(jid: str):
    j = job_or_404(jid)
    path = os.path.join(job_dir(jid), "log.txt")
    if not os.path.exists(path):
        not_found("No log yet")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", j["name"]).strip("_") or jid
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"{name}.log.txt")


@router.get("/jobs/{jid}/file/{kind}/{name}")
def job_file(jid: str, kind: str, name: str):
    if kind not in ("samples", "output"):
        not_found()
    path = os.path.join(job_dir(jid), kind, os.path.basename(name))
    if not os.path.exists(path):
        not_found()
    return FileResponse(path, filename=name if kind == "output" else None)


class ActionIn(BaseModel):
    action: str
    steps: int | None = None  # extend: the new total step count


@router.post("/jobs/{jid}/action")
def job_action(jid: str, body: ActionIn):
    j = job_or_404(jid)
    a = body.action
    if a == "stop":
        scheduler.stop(jid)
    elif a in ("sample", "save"):
        if j["status"] != "running":
            raise HTTPException(400, "Job is not running")
        send_control(jid, a)
    elif a == "resume":
        scheduler.requeue(jid)
    elif a == "extend":  # train a finished (or stopped) job on to a new total, from where it got to
        if j["status"] in ("queued", "running", "stopping"):
            raise HTTPException(400, "Wait for the job to finish, or stop it, first")
        if "evaluate" in j["config"]:
            raise HTTPException(400, "Cancel its checkpoint evaluation first")
        if not os.path.exists(os.path.join(job_dir(jid), "state", "last.pt")):
            raise HTTPException(400, "This job's training state wasn't kept, so it can't continue; duplicate it instead")
        done = int(read_status(jid).get("step") or 0)
        if body.steps is None or not done < body.steps <= 1_000_000:
            raise HTTPException(400, f"Train to more than step {done}")
        cfg = j["config"]
        cfg["settings"]["steps"] = int(body.steps)
        db.execute("UPDATE jobs SET config=? WHERE id=?", (json.dumps(cfg), jid))
        scheduler.requeue(jid)
    elif a == "restart":
        if j["status"] in ("running", "stopping"):
            raise HTTPException(400, "Stop the job first")
        if "evaluate" in j["config"]:
            raise HTTPException(400, "Cancel its checkpoint evaluation first")
        d = job_dir(jid)
        for sub in ("state", "samples", "output"):
            shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
        for f in ("metrics.jsonl", "status.json"):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
        scheduler.requeue(jid)
    elif a in ("up", "down"):
        queued = db.query("SELECT id, position FROM jobs WHERE status='queued' ORDER BY position, created")
        ids = [q["id"] for q in queued]
        if jid in ids:
            i = ids.index(jid)
            k = i - 1 if a == "up" else i + 1
            if 0 <= k < len(ids):
                ids[i], ids[k] = ids[k], ids[i]
                for n, q in enumerate(ids):
                    db.execute("UPDATE jobs SET position=? WHERE id=?", (float(n), q))
    else:
        raise HTTPException(400, f"Unknown action {a}")
    return job_or_404(jid)


@router.get("/jobs/{jid}/sandbox")
def sample_in_sandbox(jid: str, step: int, index: int):
    """Sandbox settings that re-create one of the job's training samples: its profile, prompt, seed, size and
    LoRA strength, the sampler training samples use (the profile's steps and CFG, Euler, the default
    schedule), and the checkpoint saved at that step, or the nearest one. ``notes`` says what differs."""
    import torch

    j = job_or_404(jid)
    cfg = j["config"]
    st = read_status(jid)
    sample = next((x for x in reversed(st.get("samples") or []) if x.get("step") == step and x.get("index") == index), None)
    if sample is None:
        not_found("No such sample")
    if sample.get("width") is None:
        raise HTTPException(400, "This sample is from before samples recorded their settings")
    F = get_family(cfg["family"])
    snapshot = cfg["profile"]
    profiles = db.query("SELECT * FROM profiles WHERE family=? ORDER BY updated DESC", (cfg["family"],))
    model = os.path.expanduser(str(snapshot.get("diffusion_model") or ""))
    profile = next((p for p in profiles if p["name"] == cfg.get("profile_name")), None) or \
        next((p for p in profiles if os.path.expanduser(str(p["data"].get("diffusion_model") or "")) == model), None)
    if profile is None:
        raise HTTPException(400, f"The {cfg.get('profile_name')} profile this job trained with is gone; create a "
                                 f"{F.info.label} profile with its model files to try it in the Sandbox")
    notes = []
    if os.path.expanduser(str(profile["data"].get("diffusion_model") or "")) != model:
        notes.append(f"The {profile['name']} profile now uses a different model file than this job trained with.")
    variant = str(profile["data"].get("variant") or "")
    workflows = [w for w in F.info.workflows if not w.variants or variant in w.variants]
    images = list(sample.get("images") or [])
    wf = next((w for w in workflows if bool(w.images) == bool(images)), workflows[0])
    loras = []
    if step == 0:
        notes.append("Step 0 is the base model before training, so there's no LoRA.")
    else:
        out = os.path.join(job_dir(jid), "output")
        saved = [c for c in st.get("checkpoints") or [] if os.path.exists(os.path.join(out, c["file"]))]
        best = min(saved, key=lambda c: (abs(c["step"] - step), -c["step"]), default=None)
        if best is None:
            notes.append("None of this job's checkpoints are left on disk, so there's no LoRA.")
        else:
            if best["step"] != step:
                notes.append(f"No checkpoint was saved at step {step}; this uses step {best['step']}, the nearest.")
            loras.append({"path": os.path.join(out, best["file"]), "strength": float(sample.get("lora_scale", 1.0))})
    d = F(snapshot, torch.device("cpu"))
    return {"profile_id": profile["id"], "workflow": wf.id, "prompt": sample["prompt"],
            "negative": str(snapshot.get("sample_negative") or ""), "width": sample["width"], "height": sample["height"],
            "frames": int(sample.get("frames") or 1), "steps": d.sample_steps(),
            "cfg": d.cfg_scale(F.profile_value(snapshot, "sample_cfg")), "seed": int(sample["seed"]),
            "loras": loras, "images": images, "offload": cfg["settings"].get("offload"),
            "low_vram": bool(cfg["settings"].get("low_vram")), "notes": notes,
            "job_name": j["name"], "step": step,
            **({"seconds": sample["seconds"], "lyrics": sample.get("lyrics") or ""} if "seconds" in sample else {})}


# a job folder's parts, as the disk-use view shows them; the rest (log, metrics, status) is "other"
DISK_PARTS = ("output", "samples", "cache", "state")
FINISHED = ("completed", "stopped", "failed", "interrupted")


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def job_disk(jid: str) -> dict:
    d = job_dir(jid)
    sizes = {part: _dir_size(os.path.join(d, part)) for part in DISK_PARTS}
    sizes["other"] = max(0, _dir_size(d) - sum(sizes.values()))
    sizes["total"] = sum(sizes.values())
    return sizes


@router.get("/disk")
def disk():
    """Bytes each job's folder uses, by part (checkpoints, samples, cache, training state, other), and in all."""
    jobs = {j["id"]: job_disk(j["id"]) for j in db.query("SELECT id FROM jobs")}
    keys = (*DISK_PARTS, "other", "total")
    return {"jobs": jobs, "total": {k: sum(j[k] for j in jobs.values()) for k in keys}}


@router.get("/jobs/{jid}/disk")
def disk_of_job(jid: str):
    job_or_404(jid)
    return job_disk(jid)


class CleanIn(BaseModel):
    parts: list[str]  # "cache" (re-made when the job runs again) and/or "state" (needed to resume or continue)
    jobs: list[str] | None = None  # None: every finished job


@router.post("/disk/clean")
def clean_disk(body: CleanIn):
    """Delete the cached captions and latents, and/or the training state, of finished jobs. A stopped or failed
    job's state is kept, since resuming needs it; so is anything of a job that's queued or running."""
    if not body.parts or any(p not in ("cache", "state") for p in body.parts):
        raise HTTPException(400, "Choose what to clear: cache and/or state")
    rows = db.query("SELECT id, name, status, config FROM jobs")
    if body.jobs is not None:
        rows = [r for r in rows if r["id"] in body.jobs]
    freed, cleaned, kept = 0, 0, []
    for r in rows:
        if r["status"] not in FINISHED or "evaluate" in (r["config"] or {}):
            if body.jobs is not None:
                kept.append(f"{r['name']} is queued or running")
            continue
        parts = [p for p in body.parts if p != "state" or r["status"] == "completed"]
        if "state" in body.parts and "state" not in parts and body.jobs is not None:
            kept.append(f"{r['name']} keeps its training state to resume")
        before = freed
        for part in parts:
            path = os.path.join(job_dir(r["id"]), part)
            freed += _dir_size(path)
            shutil.rmtree(path, ignore_errors=True)
        cleaned += freed > before
    return {"freed": freed, "jobs": cleaned, "kept": kept}


EVALUATION_SETTINGS = ("picker_enabled", "val_dataset", "val_timesteps", "sweep_enabled", "sweep_strengths",
                       "bleed_enabled", "bleed_prompts", "bleed_width", "bleed_height", "bleed_seed", "bleed_max")


class EvaluateIn(BaseModel):
    settings: dict  # EVALUATION_SETTINGS; a blank val_dataset means the images the job held out in training


@router.post("/jobs/{jid}/evaluate")
def evaluate_job(jid: str, body: EvaluateIn):
    """Queue a checkpoint evaluation of a finished job: the base model and each saved checkpoint scored
    with the checkpoint picker and/or the concept bleed detector (``Trainer.run_evaluation``)."""
    j = job_or_404(jid)
    if j["status"] in ("queued", "running", "stopping"):
        raise HTTPException(400, "Wait for the job to finish, or stop it, first")
    st = read_status(jid)
    out = os.path.join(job_dir(jid), "output")
    if not any(os.path.exists(os.path.join(out, c["file"])) for c in st.get("checkpoints", [])):
        raise HTTPException(400, "This job has no saved checkpoints to evaluate")
    cfg = j["config"]
    s = {k: body.settings[k] for k in EVALUATION_SETTINGS if k in body.settings}
    F = get_family(cfg["family"])
    check_evaluation(s, F.bucket_divisibility, "audio" in F.info.media)
    if cfg["settings"].get("slider_enabled") and s["picker_enabled"]:
        raise HTTPException(400, "The checkpoint picker doesn't apply to sliders; concept bleed does")
    if not (s["picker_enabled"] or s["bleed_enabled"]):
        raise HTTPException(400, "Turn on the checkpoint picker, the concept bleed detector, or both")
    val_dsets = []
    if s["picker_enabled"]:
        trained = cfg["settings"]
        if s["val_dataset"]:
            d = dataset_or_404(s["val_dataset"])
            val_dsets.append({"id": d["id"], "name": d["name"], "path": d["path"], "trigger": d["trigger"], "repeats": 1})
        elif trained.get("picker_enabled") and not trained.get("val_dataset"):
            s["val_holdout"] = trained.get("val_holdout", 2)  # the images it held out in training
        else:
            raise HTTPException(400, "This job trained on all of its images, so the checkpoint picker needs a "
                                     "validation dataset of images it didn't train on")
    cfg["evaluate"] = {"settings": s, "validation_datasets": val_dsets, "status": j["status"],
                       "phase": st.get("phase"), "message": st.get("message", ""), "requested": db.now()}
    db.execute("UPDATE jobs SET status='queued', position=?, config=? WHERE id=?", (db.now(), json.dumps(cfg), jid))
    return job_or_404(jid)


@router.delete("/jobs/{jid}")
def delete_job(jid: str, delete_files: bool = False):
    j = job_or_404(jid)
    if j["status"] in ("running", "stopping"):
        raise HTTPException(400, "Stop the job first")
    if delete_files:
        shutil.rmtree(job_dir(jid), ignore_errors=True)
    db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    return {"ok": True}


@router.get("/jobs/{jid}/config")
def job_config(jid: str):
    return job_or_404(jid)["config"]

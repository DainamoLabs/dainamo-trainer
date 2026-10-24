"""Training jobs: creating and controlling them, and their progress, logs and files."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from ...engine.dataset import resolutions
from ...engine.families import get_family
from ...engine.families.base import takes_sources
from .. import db
from ..queue import CACHE_DIR, job_dir, job_folder, read_status, scheduler, send_control
from ..validation import LOSS_AND_EMA_DEFAULTS, check_evaluation, check_resolution, check_settings
from .common import dataset_or_404, job_or_404, not_found, profile_or_404

router = APIRouter(prefix="/api")


class JobDataset(BaseModel):
    id: str
    repeats: int = 1
    resolution: int | list[int] | str | None = None  # its training resolution(s); None: the job's ``resolution``
    flip: bool | None = None  # None: the job's ``flip``
    caption_dropout: float | None = None  # None: the job's ``caption_dropout``
    # source datasets (ids, in order) whose images its files pair with by name, for a model that trains with them
    # (``FamilyInfo.sources``: an edit's source images); none: its _refs folder, if it has one
    sources: list[str] = []


class JobIn(BaseModel):
    name: str
    profile_id: str
    datasets: list[JobDataset]
    settings: dict
    gpu: str = "auto"
    output_name: str = ""


CURVE_POINTS = 60  # the most values in a job's loss curve (the Queue's cards, the top bar)
_curves: dict[str, dict] = {}  # job id -> how much of its metrics.jsonl has been read, its losses, and the curve
_curves_lock = threading.Lock()


def _loss_curve(jid: str) -> list[float]:
    """The job's loss so far, smoothed as the Loss tab smooths it, in at most CURVE_POINTS values evenly spaced over
    its steps. Only the lines added to its metrics.jsonl since the last call are read."""
    path = os.path.join(job_dir(jid), "metrics.jsonl")
    with _curves_lock:
        try:
            size = os.path.getsize(path)
        except OSError:
            _curves.pop(jid, None)
            return []
        c = _curves.get(jid)
        if c and size == c["size"]:
            return c["curve"]
        with open(path, "rb") as f:
            head = f.readline()
            if not c or size < c["size"] or head != c["head"]:  # new, or written again (the job started over)
                c = _curves[jid] = {"head": head, "size": 0, "losses": {}, "curve": []}
            f.seek(c["size"])
            chunk = f.read(size - c["size"])
        end = chunk.rfind(b"\n") + 1  # a line still being written is read next time
        for line in chunk[:end].splitlines():
            try:
                m = json.loads(line)
                c["losses"][int(m["step"])] = float(m["loss"])  # a resumed run may log a step again: the last counts
            except (ValueError, KeyError, TypeError):
                continue
        c["size"] += end
        ema, smooth = None, []
        for _, v in sorted(c["losses"].items()):
            if math.isfinite(v):
                ema = v if ema is None else 0.95 * ema + 0.05 * v
                smooth.append(ema)
        n = min(CURVE_POINTS, len(smooth))
        c["curve"] = [round(smooth[round((len(smooth) - 1) * i / (n - 1))], 5) for i in range(n)] if n > 1 else smooth
        return c["curve"]


def _job_summary(j: dict) -> dict:
    st = read_status(j["id"])
    cfg = j.pop("config", {}) or {}
    j["progress"] = {k: st.get(k) for k in ("phase", "step", "total_steps", "loss", "avg_loss", "lr",
                                            "it_per_sec", "eta_seconds", "message", "vram_gb")}
    # the job's own step count (a job given more steps shows them before it starts again)
    j["progress"]["total_steps"] = cfg.get("settings", {}).get("steps") or j["progress"]["total_steps"]
    j["last_sample"] = st.get("samples", [])[-1] if st.get("samples") else None
    j["loss_curve"] = _loss_curve(j["id"])
    # a checkpoint evaluation's recommendation, else the one made in training
    j["recommended"] = (st.get("evaluation") or {}).get("recommended") or st.get("recommended")
    j["evaluating"] = "evaluate" in cfg
    j["datasets"] = [d.get("name") for d in cfg.get("datasets", [])]
    j["slider"] = bool(cfg.get("settings", {}).get("slider_enabled"))
    j["refmod"] = bool(cfg.get("settings", {}).get("refmod_enabled"))
    j["extract"] = bool(cfg.get("settings", {}).get("extract_enabled"))  # a LoRA extracted from a finetune
    j["merge"] = bool(cfg.get("settings", {}).get("merge_enabled"))  # LoRAs merged into one
    j["network"] = "full" if cfg.get("settings", {}).get("finetune_enabled") else \
        cfg.get("settings", {}).get("network_type") or "lora"
    # its training state is kept: Resume trains on from it (a job that saved none can only restart)
    j["resumable"] = os.path.exists(os.path.join(job_dir(j["id"]), "state", "last.pt"))
    made = st.get("refmod") or {}
    if made.get("preview") and not j["last_sample"]:  # a RefMod's thumbnail: its first frame, decoded
        j["last_sample"] = {"file": made["preview"][0]["file"], "kind": "image"}
    elif made.get("audio") and not j["last_sample"]:  # an audio one's: its sound's spectrogram
        j["last_sample"] = {"file": made["audio"]["poster"], "kind": "image"}
    try:
        j["media"] = get_family(j["family"]).info.media  # what its samples are: images, clips or songs
    except KeyError:
        j["media"] = []
    return j


@router.get("/jobs")
def list_jobs():
    jobs = [_job_summary(j) for j in db.query("SELECT * FROM jobs ORDER BY position, created")]
    with _curves_lock:
        for jid in set(_curves) - {j["id"] for j in jobs}:  # deleted jobs
            del _curves[jid]
    return jobs


def _job_config(body: JobIn, old: dict | None = None) -> dict:
    """A job's config (all but its id) from the Create Job form, checked and filled in. ``old``: the config of the
    job being edited, whose datasets (and source and validation datasets) it keeps as they were if they've been
    removed from Datasets since."""
    kept = _kept_datasets(old or {})
    prof = profile_or_404(body.profile_id)
    Family = get_family(prof["family"])
    audio = "audio" in Family.info.media
    settings = check_settings({**Family.info.training_defaults, **body.settings}, Family.bucket_divisibility, audio)
    if settings["slider_enabled"] and not Family.info.sliders:
        raise HTTPException(400, f"{Family.info.label} can't train sliders yet")
    if settings["refmod_enabled"] and not Family.info.refmods:
        raise HTTPException(400, f"{Family.info.label} doesn't make RefMods (MiniMax H3 does)")
    if settings["finetune_enabled"] and not Family.info.finetune:
        raise HTTPException(400, f"{Family.info.label} can't be finetuned yet")
    if not body.datasets and not settings["slider_enabled"]:  # a slider renders its own images without one
        raise HTTPException(400, "Select at least one dataset")
    # a slider learns from its prompts, and a RefMod encodes the files themselves
    paired = takes_sources(Family.info, prof["data"]) and not settings["slider_enabled"] and not settings["refmod_enabled"]
    dsets = []
    for jd in body.datasets:
        d = _dataset(jd.id, kept)
        sources = _sources(d, jd.sources, Family.info, kept) if paired else []
        try:
            res = None if audio else settings["resolution"] if jd.resolution is None else check_resolution(jd.resolution)
        except HTTPException as e:
            raise HTTPException(400, f"{d['name']}: {e.detail}")
        drop = float(settings.get("caption_dropout", 0.0) if jd.caption_dropout is None else jd.caption_dropout)
        if not 0 <= drop <= 1:
            raise HTTPException(400, f"{d['name']}: caption dropout must be 0–1")
        dsets.append({"id": d["id"], "name": d["name"], "path": d["path"], "trigger": d["trigger"],
                      "repeats": max(1, jd.repeats), "resolution": res,
                      "flip": bool(settings.get("flip", False) if jd.flip is None else jd.flip), "caption_dropout": drop,
                      **({"sources": sources} if sources else {})})
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
        d = _dataset(settings["val_dataset"], kept)
        val_dsets.append({"id": d["id"], "name": d["name"], "path": d["path"], "trigger": d["trigger"], "repeats": 1})
    return {"name": body.name.strip() or "job", "output_name": body.output_name.strip(),
            "family": prof["family"], "profile_name": prof["name"], "profile": prof["data"],
            "datasets": dsets, "validation_datasets": val_dsets, "settings": settings}


def _kept_datasets(cfg: dict) -> dict[str, dict]:
    """The datasets a job's config names, by id: its datasets, their source datasets and its validation datasets, as
    it saved them."""
    datasets = cfg.get("datasets") or []
    every = [s for d in datasets for s in d.get("sources") or []] + (cfg.get("validation_datasets") or []) + datasets
    return {d["id"]: d for d in every if d.get("id")}


def _dataset(did: str, kept: dict[str, dict]) -> dict:
    """The dataset ``did``; one removed from Datasets since an edited job saved it is the job's own entry of it in
    ``kept`` (``_kept_datasets``), with the id, name, path and trigger it had. 404 if it's in neither."""
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (did,))
    if row is None and did in kept:
        d = kept[did]
        return {"id": did, "name": d.get("name") or did, "path": d.get("path") or "", "trigger": d.get("trigger") or ""}
    return row or not_found("Dataset not found")


def _sources(d: dict, ids: list[str], info, kept: dict[str, dict]) -> list[dict]:
    """A training dataset's source datasets, checked: {id, name, path} of each, in order (``kept``: ``_dataset``)."""
    if len(ids) > info.sources["max"]:
        raise HTTPException(400, f"{d['name']}: at most {info.sources['max']} {info.sources['label'].lower()}s")
    if len(set(ids)) < len(ids):
        raise HTTPException(400, f"{d['name']}: a {info.sources['label'].lower()} is chosen twice")
    out = []
    for sid in ids:
        if sid == d["id"]:
            raise HTTPException(400, f"{d['name']}: a dataset can't be its own {info.sources['label'].lower()}")
        s = _dataset(sid, kept)
        out.append({"id": s["id"], "name": s["name"], "path": s["path"]})
    return out


@router.post("/jobs")
def create_job(body: JobIn):
    jid = db.new_id()
    config = {"id": jid, **_job_config(body)}
    db.execute("INSERT INTO jobs (id, name, status, family, profile_name, gpu, position, created, config, folder) "
               "VALUES (?,?,?,?,?,?,?,?,?,?)", (jid, config["name"], "queued", config["family"], config["profile_name"],
                                                body.gpu, db.now(), db.now(), json.dumps(config),
                                                job_folder(config["name"], jid)))
    return job_or_404(jid)


class JobEdit(JobIn):
    restart: bool = False  # the changes make a job that has trained start again from step 0


def _restart_reason(key: str) -> str | None:
    """Why a job that has trained can't change setting ``key`` and carry on from its last save, if it can't: its
    saved LoRA and optimizer state wouldn't fit another network or optimizer, and a slider would learn something
    else. (What the checkpoint picker and concept bleed score: ``_restart_reasons``.)"""
    if key in ("network_type", "rank", "alpha", "lokr_factor"):
        return "the network (LoRA or LoKr, rank, alpha)"
    if key == "optimizer":
        return "the optimizer"
    if key.startswith("slider_") and key != "slider_sample_strengths":
        return "the slider"
    return None


# while the checkpoint picker is on, what its validation items (and how they're scored) come from, along with the
# datasets: changed mid-run, checkpoints scored before and after wouldn't compare. Likewise the concept bleed check's
# renders. (The strength sweep and bleed limit only apply to the scores at the end, so they can change; so can EMA,
# since each checkpoint is scored with the weights it saves, and the loss weighting, which validation leaves out.)
PICKER_INPUTS = ("val_dataset", "val_holdout", "val_timesteps", "seed", "resolution", "num_frames", "auto_frames",
                 "clip_seconds", "timestep_sampling", "timestep_min", "timestep_max", "masked_loss", "mask_min_weight")
BLEED_INPUTS = ("bleed_prompts", "bleed_width", "bleed_height", "bleed_seed")


def _restart_reasons(old: dict, new: dict) -> list[str]:
    """What in config ``new`` a job that has trained as ``old`` can't carry on with: another model, settings
    ``_restart_reason`` names, or what the checkpoint picker or concept bleed scores while it's on (settings
    ``old`` doesn't have, from before they existed, aren't compared; it trained without masked loss, EMA, a noise
    range or loss weighting, so those are their defaults)."""
    o, n = {**LOSS_AND_EMA_DEFAULTS, **old.get("settings", {})}, new["settings"]
    reasons = ["the model"] if old.get("profile_name") != new["profile_name"] else []
    if bool(o.get("refmod_enabled")) != bool(n.get("refmod_enabled")):
        reasons.append("what it makes (a LoRA or a RefMod)")
    if bool(o.get("finetune_enabled")) != bool(n.get("finetune_enabled")):
        reasons.append("what it trains (a LoRA or the whole model)")
    full = bool(n.get("finetune_enabled"))  # a finetune has no LoRA settings to keep
    for k, v in o.items():
        why = _restart_reason(k)
        if why and v != n.get(k) and why not in reasons and not (full and why.startswith("the network")):
            reasons.append(why)

    def changed(keys):
        return any(k in o and o[k] != n.get(k) for k in keys)

    if n.get("picker_enabled") and (not o.get("picker_enabled") or changed(PICKER_INPUTS)
                                    or _dataset_choices(old) != _dataset_choices(new)):
        reasons.append("what the checkpoint picker scores (made from the datasets, resolution, clip length, seed, "
                       "noise levels and masks)")
    if n.get("bleed_enabled") and (not o.get("bleed_enabled") or changed(BLEED_INPUTS)):
        reasons.append("the concept bleed check")
    return reasons


def _dataset_choices(c: dict) -> list[tuple]:
    s = c.get("settings", {})  # a dataset from before per-dataset settings has the job's
    return [(d["id"], d.get("repeats"), *(d.get(k, s.get(k)) for k in ("resolution", "flip", "caption_dropout")),
             tuple(src["id"] for src in d.get("sources") or []))
            for d in c.get("datasets", [])]


def _changes(old: dict, new: dict) -> list[str]:
    """What an edit changed, for the job's log: "learning_rate 0.0001 → 5e-05", "datasets"."""
    def short(v):
        t = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
        return t if len(t) <= 40 else t[:37] + "..."

    s = {**LOSS_AND_EMA_DEFAULTS, **old.get("settings", {})}  # a job from before them trained with their defaults
    out = [f"{k} {short(s[k])} → {short(v)}" for k, v in sorted(new["settings"].items()) if k in s and s[k] != v]
    return out + (["datasets"] if _dataset_choices(old) != _dataset_choices(new) else [])


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _clear_progress(jid: str):
    """Delete what a job has trained, so it starts again from step 0."""
    d = job_dir(jid)
    for sub in ("state", "samples", "output"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    for f in ("metrics.jsonl", "status.json"):
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


@router.put("/jobs/{jid}")
def edit_job(jid: str, body: JobEdit):
    """Change a job that isn't running. One that hasn't trained yet takes any change. One that has carries on
    from its last save with the changes (noted in its log), unless one is something it can't carry on with
    (``_restart_reasons``): then it starts again from step 0, which ``restart`` has to confirm (else 409). A RefMod
    job that's been made is queued to be made again with the changes."""
    saved = job_or_404(jid)["config"]
    settings = saved["settings"]
    if settings.get("extract_enabled"):
        raise HTTPException(400, "An extracted LoRA isn't edited: extract it again")
    if settings.get("merge_enabled"):
        raise HTTPException(400, "A merged LoRA isn't edited: merge again (Profiles, Merge & extract)")
    new = _job_config(body, saved)  # a dataset removed from Datasets since stays as the job has it
    with scheduler.paused():  # so the queue doesn't start the job meanwhile
        j = job_or_404(jid)
        if j["status"] in ("running", "stopping"):
            raise HTTPException(400, "Stop the job first")
        old = j["config"]
        if "evaluate" in old:
            raise HTTPException(400, "Cancel its checkpoint evaluation first")
        done = int(read_status(jid).get("step") or 0)
        trained = done > 0 or os.path.exists(os.path.join(job_dir(jid), "state", "last.pt"))
        restart = _restart_reasons(old, new) if trained else []
        if restart and not body.restart:
            raise HTTPException(409, f"Changing {_and(restart)} starts the job again from step 0, and what it trained "
                                     f"in {done} steps (its checkpoints and samples) is deleted.")
        changed = []
        if trained and not restart:
            steps = int(new["settings"]["steps"])
            if steps != int(old["settings"].get("steps") or 0):
                if j["status"] == "completed":
                    raise HTTPException(400, "To train a finished job for more steps, use Continue training")
                if steps <= done:
                    raise HTTPException(400, f"Train to more than step {done}, where it got to")
            new["profile"] = old["profile"]  # the model as it trained with it, though the profile may have changed
            changed = _changes(old, new)
        config = {"id": jid, **new}
        db.execute("UPDATE jobs SET name=?, family=?, profile_name=?, gpu=?, config=? WHERE id=?",
                   (config["name"], config["family"], config["profile_name"], body.gpu, json.dumps(config), jid))
        # a RefMod is quick to make, so one that's been made (or failed) is made again with any change
        remake = (bool(old["settings"].get("refmod_enabled")) and j["status"] != "queued"
                  and {k: v for k, v in old.items() if k != "id"} != new)
        if restart or remake:
            _clear_progress(jid)
            db.execute("UPDATE jobs SET status='queued', position=?, exit_code=NULL WHERE id=?", (db.now(), jid))
        elif changed:  # its log says where the run changed
            with open(os.path.join(job_dir(jid), "log.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n===== edited {time.strftime('%Y-%m-%d %H:%M:%S')}: it carries on from step {done} with "
                        f"{'; '.join(changed)} =====\n")
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
    model = os.path.expanduser(str(body.settings.get("model") or "").strip())  # a Sandbox's finetuned model
    if body.kind == "sample" and model and os.path.isfile(model):
        path = model
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
        if any(j["config"]["settings"].get(k) for k in ("refmod_enabled", "extract_enabled", "merge_enabled")):
            raise HTTPException(400, "This job trains nothing to continue")
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
        _clear_progress(jid)
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
    schedule), and the checkpoint saved at that step, or the nearest one (a finetune's as the ``model``, in place
    of the profile's diffusion model). ``notes`` says what differs."""
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
    loras, finetuned = [], None
    full = bool(cfg["settings"].get("finetune_enabled"))
    if step == 0:
        notes.append("Step 0 is the base model before training" + ("." if full else ", so there's no LoRA."))
    else:
        out = os.path.join(job_dir(jid), "output")
        saved = [c for c in st.get("checkpoints") or [] if os.path.exists(os.path.join(out, c["file"]))]
        best = min(saved, key=lambda c: (abs(c["step"] - step), -c["step"]), default=None)
        if best is None:
            notes.append("None of this job's checkpoints are left on disk, so " + (
                "this uses the base model." if full else "there's no LoRA."))
        else:
            if best["step"] != step:
                notes.append(f"No checkpoint was saved at step {step}; this uses step {best['step']}, the nearest.")
            if full:
                finetuned = os.path.join(out, best["file"])
            else:
                loras.append({"path": os.path.join(out, best["file"]), "strength": float(sample.get("lora_scale", 1.0))})
    d = F(snapshot, torch.device("cpu"))
    return {"profile_id": profile["id"], "workflow": wf.id, "prompt": sample["prompt"],
            "negative": str(snapshot.get("sample_negative") or ""), "width": sample["width"], "height": sample["height"],
            "frames": int(sample.get("frames") or 1), "steps": d.sample_steps(),
            "cfg": d.cfg_scale(F.profile_value(snapshot, "sample_cfg")), "seed": int(sample["seed"]),
            "loras": loras, "model": finetuned, "images": images, "offload": cfg["settings"].get("offload"),
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


def _cache_files(jid: str) -> set[str]:
    """The shared cache's files a job used when it last ran (as its trainer lists them in cache.json)."""
    try:
        with open(os.path.join(job_dir(jid), "cache.json")) as f:
            return {os.path.join(CACHE_DIR, p) for p in json.load(f)["files"]}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def _cache_in_use() -> set[str]:
    """The shared cache's files that queued and running jobs use."""
    return set().union(*(_cache_files(r["id"])
                         for r in db.query("SELECT id FROM jobs WHERE status IN ('queued','running','stopping')")))


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _clean_shared_cache() -> int:
    """Delete the shared cache's encodings that no queued or running job uses, and the folders left empty;
    -> bytes freed. Files still being written (``.tmp``) are left alone."""
    freed = 0
    with scheduler.paused():  # no job starts meanwhile
        used = _cache_in_use()
        for kind in ("latents", "text"):
            base = os.path.join(CACHE_DIR, kind)
            for sig in os.listdir(base) if os.path.isdir(base) else []:
                d = os.path.join(base, sig)
                for name in os.listdir(d):
                    p = os.path.join(d, name)
                    if name == "signature.json" or name.endswith(".tmp") or p in used:
                        continue
                    size = _size(p)
                    try:
                        os.remove(p)
                        freed += size
                    except OSError:
                        pass
                if os.listdir(d) == ["signature.json"]:
                    shutil.rmtree(d, ignore_errors=True)
    return freed


@router.get("/disk")
def disk():
    """Bytes each job's folder uses, by part (checkpoints, samples, cache, training state, other), and in all; and
    the shared cache's size, with how much of it queued and running jobs use."""
    jobs = {j["id"]: job_disk(j["id"]) for j in db.query("SELECT id FROM jobs")}
    keys = (*DISK_PARTS, "other", "total")
    return {"jobs": jobs, "total": {k: sum(j[k] for j in jobs.values()) for k in keys},
            "shared_cache": {"total": _dir_size(CACHE_DIR), "in_use": sum(_size(p) for p in _cache_in_use())}}


@router.get("/jobs/{jid}/disk")
def disk_of_job(jid: str):
    """``job_disk``, and ``shared``: the bytes of the shared cache's files it uses (not counted in its total)."""
    job_or_404(jid)
    return {**job_disk(jid), "shared": sum(_size(p) for p in _cache_files(jid))}


class CleanIn(BaseModel):
    parts: list[str]  # "cache" (re-made when the job runs again) and/or "state" (needed to resume or continue)
    jobs: list[str] | None = None  # None: every finished job, and the shared cache


@router.post("/disk/clean")
def clean_disk(body: CleanIn):
    """Delete the cached captions and latents, and/or the training state, of finished jobs. A stopped or failed
    job's state is kept, since resuming needs it; so is anything of a job that's queued or running. Clearing
    every job's cache also clears what the shared cache holds that no queued or running job uses. (A job's own
    cache is from before caches were shared.)"""
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
    if body.jobs is None and "cache" in body.parts:
        freed += _clean_shared_cache()
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
    if j["config"]["settings"].get("refmod_enabled"):
        raise HTTPException(400, "A RefMod has no checkpoints to evaluate")
    if j["config"]["settings"].get("finetune_enabled"):
        raise HTTPException(400, "A finetune's checkpoints are whole models; a checkpoint evaluation scores LoRAs")
    if j["config"]["settings"].get("extract_enabled") or j["config"]["settings"].get("merge_enabled"):
        raise HTTPException(400, f"An {'extracted' if j['config']['settings'].get('extract_enabled') else 'merged'} "
                                 "LoRA has one file and no datasets to score it on")
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
        raise HTTPException(400, "Turn on the checkpoint picker, concept bleed, or both")
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


def _finetune_checkpoint(j: dict, file: str) -> tuple[str, dict]:
    """(path, checkpoint entry) of one of a finetune job's saved checkpoints."""
    if not j["config"]["settings"].get("finetune_enabled"):
        raise HTTPException(400, "Only a finetune's checkpoints are whole models to extract a LoRA from")
    c = next((c for c in read_status(j["id"]).get("checkpoints") or [] if c.get("file") == file), None)
    path = os.path.join(job_dir(j["id"]), "output", os.path.basename(file))
    if c is None or not os.path.isfile(path):
        not_found(f"{file} isn't one of this job's saved checkpoints")
    return path, c


@router.get("/jobs/{jid}/extract")
def extract_sizes(jid: str, file: str):
    """For the Extract LoRA form: about how big a LoRA extracted from one of a finetune's checkpoints is at each rank
    it offers, and the model it was finetuned from (the job's profile's), which the LoRA is the change from."""
    from ...engine.extract import sizes

    j = job_or_404(jid)
    path, _ = _finetune_checkpoint(j, file)
    base = j["config"]["profile"].get("diffusion_model") or ""
    return {"sizes": {str(r): n for r, n in sizes(path).items()}, "base": base, "base_found": os.path.isfile(base)}


class ExtractIn(BaseModel):
    file: str  # the checkpoint, in the finetune job's output
    rank: int = 128
    name: str = ""
    gpu: str = "auto"


@router.post("/jobs/{jid}/extract")
def extract_lora(jid: str, body: ExtractIn):
    """Queue a job that extracts a LoRA from one of a finetune's checkpoints (``Trainer.run_extract``): what the
    finetune changed from the model it started from, the job's profile's. It's a job of its own, so its LoRA is
    listed with it (in the Sandbox too) and the finetune's checkpoints are left as they are."""
    from ...engine.extract import MAX_RANK

    j = job_or_404(jid)
    path, c = _finetune_checkpoint(j, body.file)
    cfg = j["config"]
    base = cfg["profile"].get("diffusion_model") or ""
    if not os.path.isfile(base):
        raise HTTPException(400, f"The model it was finetuned from, {base or 'its profile’s diffusion model'}, "
                                 "isn't there any more, and the LoRA is the change from it")
    if not 1 <= body.rank <= MAX_RANK:
        raise HTTPException(400, f"The rank must be 1–{MAX_RANK}")
    name = body.name.strip() or f"{os.path.splitext(body.file)[0]}_r{body.rank}"
    trigger = ", ".join(d["trigger"] for d in cfg.get("datasets") or [] if d.get("trigger"))
    settings = {"steps": 0, "extract_enabled": True, "extract_rank": int(body.rank), "extract_source": path,
                "extract_base": base, "extract_from": {"job_id": jid, "job_name": j["name"], "file": body.file,
                                                       "folder": os.path.basename(job_dir(jid)),
                                                       "step": int(c.get("step") or 0), "trigger": trigger}}
    return queue_made(name, cfg["family"], cfg["profile_name"], cfg["profile"], settings, body.gpu)


def queue_made(name: str, family: str, profile_name: str, profile: dict, settings: dict, gpu: str) -> dict:
    """Queue a job that trains nothing but makes a file (an extracted or merged LoRA), with no datasets."""
    if gpu not in ("auto", "") and not gpu.isdigit():
        raise HTTPException(400, f"Invalid GPU {gpu!r}")
    new = db.new_id()
    config = {"id": new, "name": name, "output_name": name, "family": family, "profile_name": profile_name,
              "profile": profile, "datasets": [], "validation_datasets": [], "settings": settings}
    db.execute("INSERT INTO jobs (id, name, status, family, profile_name, gpu, position, created, config, folder) "
               "VALUES (?,?,?,?,?,?,?,?,?,?)", (new, name, "queued", family, profile_name, gpu or "auto", db.now(),
                                                db.now(), json.dumps(config), job_folder(name, new)))
    return job_or_404(new)


@router.delete("/jobs/{jid}")
def delete_job(jid: str, delete_files: bool = False):
    with scheduler.paused():  # so the queue doesn't start the job between the check and the delete
        j = job_or_404(jid)
        if j["status"] in ("running", "stopping"):
            raise HTTPException(400, "Stop the job first")
        folder = job_dir(jid)  # named from the row, so before it goes
        db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    if delete_files:  # after the lock, so a big folder doesn't hold the queue
        shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True}


@router.get("/jobs/{jid}/config")
def job_config(jid: str):
    return job_or_404(jid)["config"]

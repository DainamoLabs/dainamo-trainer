"""Jobs moved between machines: Export writes a job to a folder (its own files, its datasets' and the encodings it used),
and Import adds a job from such a folder, on another machine or this one.

The folder holds ``dainamo_job.json`` (what's in it; written last, so a folder without it is an export cut short) and:
  job/        the job's folder: its config, status, log, metrics, checkpoints, samples and training state
  datasets/   each dataset's folder as it is (captions, masks, reference images), unless they were left out
  cache/      the shared cache's encodings the job last used, by the same paths as in ``data/cache``

The model files aren't copied: they're big, and the other machine has its own. The export records each one's
fingerprint (``engine.cache.fingerprint``: of its contents, not its path), and the import is given a profile of the
same family there, compared by those. A job that has trained carries on only with the same diffusion model (and
training adapter and variant), as an edit can (``routes.jobs._same_model``); with another it starts again from step 0.
The text encoder and VAE may be other files: the cached encodings then don't match theirs, and are made again.

The cache names a dataset's files by the dataset's id and their sizes and modified times (``Item.key``), so an
imported dataset keeps its id (unless this machine has another by it) and its files their modified times, which the
export records (a USB stick's file system may round them).

One export or import runs at a time, in a thread, with its progress in ``state`` (``GET /api/transfer``). One that's
cancelled or fails removes what it wrote. An import adds its datasets and job only once everything is copied.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time

from ..engine.cache import fingerprint
from ..engine.families import get_family
from . import db
from .queue import CACHE_DIR, JOBS_DIR, job_dir, job_folder, read_status

MANIFEST = "dainamo_job.json"
FORMAT, VERSION = "dainamo-job", 1
MANAGED_DATASETS = os.path.join(db.DATA_DIR, "datasets")  # (``routes.datasets``)
CHUNK = 8 << 20
# a profile's fields that make the model a job trains (``routes.jobs.MODEL_FIELDS``)
MODEL_FIELDS = ("diffusion_model", "training_adapter", "variant")
PROGRESS = ("state", "samples", "output", "metrics.jsonl", "status.json")  # what a job has trained (``_clear_progress``)
BUSY = ("queued", "running", "stopping")
SPARE = 256 << 20  # room left on the disk beyond what's copied


class Busy(Exception):
    """Another export or import is running."""


class Cancelled(Exception):
    pass


_lock = threading.Lock()
_cancel = threading.Event()
state: dict = {"state": "idle"}


def status() -> dict:
    return dict(state)


def cancel() -> None:
    if state.get("state") == "running":
        _cancel.set()


def exporting(jid: str) -> bool:
    """Whether ``jid`` is being exported (it mustn't run or be deleted meanwhile)."""
    return state.get("state") == "running" and state.get("kind") == "export" and state.get("job_id") == jid


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:48].strip("_")


def _unique(path: str) -> str:
    out, n = path, 2
    while os.path.exists(out):
        out, n = f"{path}_{n}", n + 1
    return out


def _inside(path: str, folder: str) -> bool:
    path, folder = os.path.abspath(path), os.path.abspath(folder)
    return path == folder or path.startswith(folder + os.sep)


def _files(folder: str, skip: tuple[str, ...] = ()) -> list[tuple[str, str, int]]:
    """(path, path in ``folder``, size) of every file in ``folder``, but those in its parts named in ``skip`` and
    files being written (``.tmp``)."""
    out = []
    for root, dirs, names in os.walk(folder):
        dirs.sort()
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, folder)
            if rel.split(os.sep)[0] in skip or n.endswith(".tmp"):
                continue
            try:
                out.append((p, rel, os.path.getsize(p)))
            except OSError:
                pass
    return out


def _copy(src: str, dst: str, mtime_ns: int | None = None) -> None:
    """``src`` to ``dst``, counting its bytes in ``state``, with its modified time (or ``mtime_ns``)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as f, open(dst, "wb") as g:
        while chunk := f.read(CHUNK):
            if _cancel.is_set():
                raise Cancelled
            g.write(chunk)
            state["done"] += len(chunk)
    st = os.stat(src)
    try:
        os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns if mtime_ns is None else mtime_ns))
    except OSError:  # a file system that keeps no times
        pass


def _start(kind: str, label: str, total: int, work, job_id: str | None = None, path: str = "") -> dict:
    """Run ``work(undo)`` in a thread as the one transfer; ``undo`` collects paths to remove if it doesn't finish."""
    with _lock:
        if state.get("state") == "running":
            raise Busy("Another export or import is running: wait for it, or cancel it")
        _cancel.clear()
        state.clear()
        state.update(kind=kind, state="running", label=label, done=0, total=total, job_id=job_id, path=path,
                     message="", started=time.time())

    def run():
        undo: list[str] = []
        try:
            work(undo)
            state.update(state="done", finished=time.time())
        except BaseException as e:  # (Cancelled, an OSError such as a full disk, or a bug)
            for p in reversed(undo):
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            cancelled = isinstance(e, Cancelled)
            msg = "Cancelled" if cancelled else (e.strerror or str(e)) if isinstance(e, OSError) else str(e) or repr(e)
            state.update(state="cancelled" if cancelled else "failed", message=msg, finished=time.time())

    threading.Thread(target=run, daemon=True).start()
    return status()


# --- export -----------------------------------------------------------------------------------------------------------

def _job(jid: str) -> dict:
    j = db.query_one("SELECT * FROM jobs WHERE id=?", (jid,))
    if j is None:
        raise LookupError("Job not found")
    return j


def job_datasets(cfg: dict) -> list[dict]:
    """The datasets a job's config names, once each, in order: its datasets, their source datasets, its validation
    datasets ({id, name, path, trigger, roles})."""
    out: dict[str, dict] = {}

    def add(d: dict, role: str):
        if not d.get("id"):
            return
        e = out.setdefault(d["id"], {"id": d["id"], "name": d.get("name") or d["id"], "path": d.get("path") or "",
                                     "trigger": d.get("trigger") or "", "roles": []})
        if role not in e["roles"]:
            e["roles"].append(role)

    for d in cfg.get("datasets") or []:
        add(d, "training")
    for d in cfg.get("datasets") or []:
        for s in d.get("sources") or []:
            add(s, "source")
    for d in cfg.get("validation_datasets") or []:
        add(d, "validation")
    return list(out.values())


def _cache_used(jd: str) -> list[tuple[str, str, int]]:
    """The shared cache's files the job last used (its ``cache.json``), and their folders' ``signature.json``."""
    try:
        with open(os.path.join(jd, "cache.json")) as f:
            rels = json.load(f)["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    rels = sorted(set(rels) | {os.path.join(os.path.dirname(r), "signature.json") for r in rels})
    out = []
    for rel in rels:
        p = os.path.join(CACHE_DIR, rel)
        if os.path.isfile(p):
            out.append((p, rel, os.path.getsize(p)))
    return out


def export_plan(jid: str, with_datasets: bool = True) -> dict:
    """What exporting ``jid`` copies: the job's files, each dataset's (with ``with_datasets``), the cache's."""
    j = _job(jid)
    cfg = j["config"]
    jd = job_dir(jid)
    job_files = _files(jd, skip=("control.json", "cache")) if os.path.isdir(jd) else []
    datasets = []
    for d in job_datasets(cfg):
        row = db.query_one("SELECT * FROM datasets WHERE id=?", (d["id"],))
        path = (row or d)["path"]
        there = bool(path) and os.path.isdir(path)
        files = _files(path) if there and with_datasets else []
        datasets.append({**d, "path": path, "row": row, "exists": there, "files": files,
                         "bytes": sum(f[2] for f in files) if with_datasets else sum(f[2] for f in _files(path)) if there else 0})
    cache = _cache_used(jd)
    return {"job": j, "job_dir": jd, "job_files": job_files, "datasets": datasets, "cache": cache}


def export_summary(jid: str, with_datasets: bool = True) -> dict:
    """The sizes Export shows before it starts."""
    plan = export_plan(jid, with_datasets)
    job_b = sum(f[2] for f in plan["job_files"])
    cache_b = sum(f[2] for f in plan["cache"])
    ds = [{"id": d["id"], "name": d["name"], "roles": d["roles"], "exists": d["exists"], "bytes": d["bytes"],
           "files": len(d["files"])} for d in plan["datasets"]]
    data_b = sum(d["bytes"] for d in ds if d["exists"]) if with_datasets else 0
    return {"job": job_b, "cache": cache_b, "datasets": ds, "total": job_b + cache_b + data_b,
            "status": plan["job"]["status"]}


def _models(cfg: dict) -> dict:
    """The job's model files: {field: {label, path, name, fingerprint}} (fingerprint None: not there)."""
    F = get_family(cfg["family"])
    data = cfg.get("profile") or {}
    out = {}
    for f in F.info.profile_fields:
        if f.kind != "path":
            continue
        p = os.path.expanduser(str(data.get(f.key) or "").strip())
        if not p:
            continue
        out[f.key] = {"label": f.label, "path": p, "name": os.path.basename(p.rstrip("/\\")),
                      "fingerprint": _fingerprint(p)}
    return out


_prints: dict[tuple, str] = {}


def _fingerprint(path: str) -> str | None:
    """``fingerprint``, remembered while the file stays as it is (None: there's no such file)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _prints:
        _prints[key] = fingerprint(path)
    return _prints[key]


def start_export(jid: str, parent: str, with_datasets: bool = True) -> dict:
    """Export job ``jid`` into a new folder in ``parent``, named after it."""
    j = _job(jid)
    if j["status"] in BUSY:
        raise ValueError("Wait for the job to finish, or stop it, before exporting it")
    parent = os.path.abspath(os.path.expanduser(parent.strip()))
    if not parent.strip() or not os.path.isdir(parent):
        raise ValueError(f"Folder not found: {parent}")
    plan = export_plan(jid, with_datasets)
    cfg = plan["job"]["config"]
    dest = _unique(os.path.join(parent, _slug(cfg.get("name") or "") or "job"))
    sources = [plan["job_dir"], CACHE_DIR, *(d["path"] for d in plan["datasets"] if with_datasets and d["exists"])]
    if any(_inside(dest, s) for s in sources):
        raise ValueError("Choose a folder outside the job's, its datasets' and the cache's folders")
    total = sum(f[2] for f in plan["job_files"]) + sum(f[2] for f in plan["cache"]) + \
        sum(f[2] for d in plan["datasets"] for f in d["files"])
    free = shutil.disk_usage(parent).free
    if free < total + SPARE:
        raise ValueError(f"Not enough space there: it needs {total / 1e9:.1f} GB, and {free / 1e9:.1f} GB is free")

    def work(undo: list[str]):
        os.makedirs(dest)
        undo.append(dest)
        for src, rel, _ in plan["job_files"]:
            _copy(src, os.path.join(dest, "job", rel))
        datasets, used = [], set()
        for n, d in enumerate(plan["datasets"]):
            row = d["row"] or {}
            entry = {"id": d["id"], "name": row.get("name") or d["name"], "trigger": row.get("trigger", d["trigger"]),
                     "path": d["path"], "roles": d["roles"], "managed": bool(row.get("managed")),
                     "caption_options": row.get("caption_options"), "folder": None, "files": 0, "bytes": d["bytes"],
                     "mtimes": {}}
            if d["files"]:
                name = _slug(entry["name"]) or "dataset"
                folder = name if name.lower() not in used else f"{name}_{n + 1}"
                used.add(folder.lower())
                entry["folder"] = f"datasets/{folder}"
                for src, rel, _ in d["files"]:
                    entry["mtimes"][rel.replace(os.sep, "/")] = os.stat(src).st_mtime_ns
                    _copy(src, os.path.join(dest, "datasets", folder, rel))
                entry["files"] = len(d["files"])
            datasets.append(entry)
        for src, rel, _ in plan["cache"]:
            _copy(src, os.path.join(dest, "cache", rel))
        st = read_status(jid)
        job = plan["job"]
        manifest = {
            "format": FORMAT, "version": VERSION, "exported": time.time(),
            "job": {"config": {k: v for k, v in cfg.items() if k != "evaluate"}, "status": job["status"],
                    "created": job["created"], "started": job["started"], "finished": job["finished"],
                    "exit_code": job["exit_code"], "step": int(st.get("step") or 0)},
            "models": _models(cfg), "variant": (cfg.get("profile") or {}).get("variant"),
            "datasets": datasets, "cache": {"files": len(plan["cache"]), "bytes": sum(f[2] for f in plan["cache"])},
        }
        with open(os.path.join(dest, MANIFEST), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)

    return _start("export", f"Exporting {cfg.get('name')}", total, work, job_id=jid, path=dest)


# --- import -----------------------------------------------------------------------------------------------------------

def find_export(path: str) -> str:
    """The exported job's folder: ``path``, or the one folder in it that is one."""
    p = os.path.abspath(os.path.expanduser(path.strip()))
    if os.path.basename(p) == MANIFEST:
        p = os.path.dirname(p)
    if os.path.isfile(os.path.join(p, MANIFEST)):
        return p
    try:
        subs = [os.path.join(p, d) for d in sorted(os.listdir(p)) if os.path.isfile(os.path.join(p, d, MANIFEST))]
    except OSError:
        raise ValueError(f"Folder not found: {p}") from None
    if len(subs) == 1:
        return subs[0]
    if subs:
        raise ValueError("That folder has several exported jobs: choose one of them")
    raise ValueError("No exported job there: choose the folder Export made (it has dainamo_job.json in it)")


def read_manifest(folder: str) -> dict:
    try:
        with open(os.path.join(folder, MANIFEST), encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"Can't read the export's {MANIFEST}: {e}") from None
    if m.get("format") != FORMAT:
        raise ValueError(f"{MANIFEST} isn't a Dainamo job export")
    if int(m.get("version") or 0) > VERSION:
        raise ValueError("It was exported by a newer Dainamo: update this one to import it")
    return m


def _family(m: dict):
    family = m["job"]["config"].get("family")
    try:
        return get_family(family)
    except KeyError:
        raise ValueError(f"It's a job for a model family this Dainamo doesn't have ({family}): update Dainamo") from None


def profile_match(profile: dict, m: dict) -> dict:
    """How a profile here compares with the export's models: per model field, whether it's the same file (by
    fingerprint; None: one side has none, or the file isn't here); ``same_model``: a trained job can carry on with
    it (``MODEL_FIELDS``)."""
    F = get_family(profile["family"])
    data = profile["data"] or {}
    fields = []
    for f in F.info.profile_fields:
        if f.kind != "path":
            continue
        theirs = (m.get("models") or {}).get(f.key)
        mine = os.path.expanduser(str(data.get(f.key) or "").strip())
        if not theirs and not mine:
            continue
        here = _fingerprint(mine) if mine else None
        same = bool(theirs and here and theirs.get("fingerprint") == here)
        fields.append({"key": f.key, "label": f.label, "exported": (theirs or {}).get("name", ""),
                       "here": os.path.basename(mine.rstrip("/\\")) if mine else "", "missing": bool(mine) and here is None,
                       "same": same})
    same_variant = (data.get("variant") or None) == (m.get("variant") or None)
    model = [x for x in fields if x["key"] in MODEL_FIELDS]
    return {"id": profile["id"], "name": profile["name"], "fields": fields, "same_variant": same_variant,
            "same_model": same_variant and all(x["same"] for x in model) and any(x["key"] == "diffusion_model" for x in model),
            "identical": same_variant and bool(fields) and all(x["same"] for x in fields)}


def _trained(folder: str, m: dict) -> bool:
    return int(m["job"].get("step") or 0) > 0 or os.path.isdir(os.path.join(folder, "job", "state"))


def inspect(path: str) -> dict:
    """What Import shows of an export: the job, the profiles here it could use, its datasets and those here."""
    folder = find_export(path)
    m = read_manifest(folder)
    F = _family(m)
    cfg = m["job"]["config"]
    profiles = [profile_match(p, m) for p in db.query("SELECT * FROM profiles WHERE family=? ORDER BY name", (F.info.id,))]
    best = next((p for p in profiles if p["identical"]), None) or next((p for p in profiles if p["same_model"]), None) \
        or (profiles[0] if profiles else None)
    datasets = []
    for d in m["datasets"]:
        here = db.query_one("SELECT id, name, path FROM datasets WHERE id=?", (d["id"],))
        datasets.append({"id": d["id"], "name": d["name"], "trigger": d.get("trigger") or "", "roles": d["roles"],
                         "included": bool(d.get("folder")), "files": d.get("files", 0), "bytes": d.get("bytes", 0),
                         "here": here})
    job_b = sum(f[2] for f in _files(os.path.join(folder, "job")))
    return {"path": folder, "name": cfg.get("name"), "family": F.info.id, "family_label": F.info.label,
            "status": m["job"]["status"], "step": m["job"].get("step") or 0, "steps": (cfg.get("settings") or {}).get("steps"),
            "trained": _trained(folder, m), "exported": m.get("exported"), "profiles": profiles,
            "recommended": best["id"] if best else None, "datasets": datasets,
            "sizes": {"job": job_b, "cache": (m.get("cache") or {}).get("bytes", 0)}}


def _imported_status(exported: str, restart: bool, trained: bool) -> str:
    """The status an imported job takes: a trained one as it was (one waiting to run, ready), else ready."""
    if restart or not trained or exported in ("ready", *BUSY):
        return "ready"
    return exported if exported in ("completed", "stopped", "failed", "interrupted") else "stopped"


def start_import(path: str, name: str, profile_id: str, choices: dict[str, str], restart: bool = False) -> dict:
    """Import the job exported to ``path`` as ``name``, on profile ``profile_id``. ``choices``: per dataset of the
    export (by its id), "export" (add the export's copy as a dataset here) or the id of a dataset here to use.
    ``restart``: it starts again from step 0 (its progress isn't imported), as it must on another diffusion model."""
    folder = find_export(path)
    m = read_manifest(folder)
    F = _family(m)
    name = name.strip()
    if not name:
        raise ValueError("Give the job a name")
    profile = db.query_one("SELECT * FROM profiles WHERE id=?", (profile_id,))
    if profile is None:
        raise ValueError("Choose a profile")
    if profile["family"] != F.info.id:
        raise ValueError(f"Choose a {F.info.label} profile: the job is for {F.info.label}")
    trained = _trained(folder, m)
    if trained and not restart and not profile_match(profile, m)["same_model"]:
        raise ValueError(f"{profile['name']}'s diffusion model isn't the one the job trained on, so it can only "
                         "start again from step 0 with it")
    plan = []  # (exported dataset, "export" or a dataset here)
    for d in m["datasets"]:
        choice = choices.get(d["id"]) or ""
        if choice == "export":
            if not d.get("folder"):
                raise ValueError(f"The export doesn't include {d['name']}'s files: choose a dataset here for it")
            plan.append((d, None))
        else:
            row = db.query_one("SELECT * FROM datasets WHERE id=?", (choice,)) if choice else None
            if row is None:
                raise ValueError(f"Choose a dataset for {d['name']}")
            plan.append((d, row))
    progress = PROGRESS if restart or not trained else ()
    job_files = _files(os.path.join(folder, "job"), skip=progress)
    cache_files = _files(os.path.join(folder, "cache"))
    data_files = {d["id"]: _files(os.path.join(folder, d["folder"])) for d, row in plan if row is None}
    total = sum(f[2] for f in job_files) + sum(f[2] for f in cache_files) + sum(f[2] for fs in data_files.values() for f in fs)
    os.makedirs(db.DATA_DIR, exist_ok=True)
    free = shutil.disk_usage(db.DATA_DIR).free
    if free < total + SPARE:
        raise ValueError(f"Not enough space here: it needs {total / 1e9:.1f} GB, and {free / 1e9:.1f} GB is free")
    jid = db.new_id()

    def work(undo: list[str]):
        added, remap = [], {}  # dataset rows to add; exported dataset id -> {id, name, path} here
        for d, row in plan:
            if row is not None:
                remap[d["id"]] = {"id": row["id"], "name": row["name"], "path": row["path"]}
                continue
            # its own id, so its cached encodings are still its (``Item.key``), unless a dataset here has it
            did = d["id"] if not db.query_one("SELECT id FROM datasets WHERE id=?", (d["id"],)) \
                and d["id"] not in (a["id"] for a in added) else db.new_id()
            dest = _unique(os.path.join(MANAGED_DATASETS, f"{_slug(d['name']) or 'dataset'}_{did[:4]}"))
            os.makedirs(dest)
            undo.append(dest)
            mtimes = d.get("mtimes") or {}
            for src, rel, _ in data_files[d["id"]]:
                _copy(src, os.path.join(dest, rel), mtimes.get(rel.replace(os.sep, "/")))
            added.append({"id": did, "name": d["name"], "path": dest, "trigger": d.get("trigger") or "",
                          "caption_options": d.get("caption_options")})
            remap[d["id"]] = {"id": did, "name": d["name"], "path": dest}
        for src, rel, size in cache_files:  # into the shared cache, but what it has already
            dst = os.path.join(CACHE_DIR, rel)
            if os.path.exists(dst):
                state["done"] += size
                continue
            undo.append(dst)
            _copy(src, dst)
        jd = os.path.join(JOBS_DIR, job_folder(name, jid))
        os.makedirs(jd)
        undo.append(jd)
        for src, rel, _ in job_files:
            _copy(src, os.path.join(jd, rel))

        cfg = copy.deepcopy(m["job"]["config"])
        cfg.pop("evaluate", None)
        cfg.update(id=jid, name=name, profile_id=profile["id"], profile_name=profile["name"], profile=profile["data"])

        def moved(d: dict):
            here = remap.get(d.get("id"))
            if here:
                d.update(id=here["id"], name=here["name"], path=here["path"])
                d.pop("moved_from", None)

        for d in cfg.get("datasets") or []:
            moved(d)
            for s in d.get("sources") or []:
                moved(s)
        for d in cfg.get("validation_datasets") or []:
            moved(d)
        settings = cfg.get("settings") or {}
        if settings.get("val_dataset") in remap:
            settings["val_dataset"] = remap[settings["val_dataset"]]["id"]
        with open(os.path.join(jd, "job.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        status = _imported_status(m["job"]["status"], restart, trained)
        with open(os.path.join(jd, "log.txt"), "a", encoding="utf-8") as log:
            log.write(f"\n===== imported {time.strftime('%Y-%m-%d %H:%M:%S')} from {folder} =====\n")
            if trained and status == "ready":
                log.write("It starts again from step 0 here\n")
        if _cancel.is_set():
            raise Cancelled
        job = m["job"]
        kept = status != "ready"  # a trained job keeps when it ran
        try:
            for a in added:
                db.execute("INSERT INTO datasets (id, name, path, managed, trigger, created, caption_options) "
                           "VALUES (?,?,?,?,?,?,?)", (a["id"], a["name"], a["path"], 1, a["trigger"], db.now(),
                                                      json.dumps(a["caption_options"]) if a["caption_options"] else None))
            db.execute("INSERT INTO jobs (id, name, status, family, profile_name, gpu, position, created, started, "
                       "finished, exit_code, config, folder) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (jid, name, status, cfg["family"], profile["name"], "auto", db.now(),
                        (job.get("created") or db.now()) if kept else db.now(), job.get("started") if kept else None,
                        job.get("finished") if kept else None, job.get("exit_code") if kept else None, json.dumps(cfg),
                        os.path.basename(jd)))
        except BaseException:
            for a in added:
                db.execute("DELETE FROM datasets WHERE id=?", (a["id"],))
            db.execute("DELETE FROM jobs WHERE id=?", (jid,))
            raise
        state["job_id"] = jid

    return _start("import", f"Importing {name}", total, work, path=folder)

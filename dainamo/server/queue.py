"""Job queue: launches training subprocesses onto GPUs according to the queue mode.

Modes (settings.queue.mode):
  sequential       one job at a time, on the first enabled GPU
  parallel_single  up to ``max_per_gpu`` jobs at once, all on the first enabled GPU
  parallel_multi   up to ``max_per_gpu`` jobs on *each* enabled GPU

A job may be pinned to a GPU (``gpu`` = "0", "1", ...) or left on "auto".

A finished job can be queued again for a checkpoint evaluation (``config["evaluate"]``): it runs like a
job, and afterwards the job is put back the way it was (status, times), with the results in its status file.

Training processes run in their own session, so they survive a server restart;
on startup the scheduler reattaches to any that are still running.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

from ..fsutil import replace
from . import db, procs

JOBS_DIR = os.path.join(db.DATA_DIR, "jobs")
CACHE_DIR = os.path.join(db.DATA_DIR, "cache")  # encoded items and captions, shared by jobs (``engine.cache``)
EXIT_CODES = {"completed": 0, "stopped": 3}
# seconds a job asked to stop may go without any sign of work (``last_activity``) before it's killed: a save or a
# sample in progress keeps writing, a hung process doesn't
STOP_GRACE = 180


_folders: dict[str, str] = {}  # job id -> its folder's name, which never changes


def job_folder(name: str, job_id: str) -> str:
    """A new job's folder name: its name, made safe for any file system, then its id (names can repeat)."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:48].strip("_")
    return f"{slug}_{job_id}" if slug else job_id


def job_dir(job_id: str) -> str:
    """A job's folder: "<name>_<id>", named when the job was added (renaming the job doesn't move its files), or
    "<id>" for jobs from before folders were named."""
    folder = _folders.get(job_id)
    if folder is None:
        row = db.query_one("SELECT folder FROM jobs WHERE id=?", (job_id,))
        if row is None:  # not a job (any more)
            return os.path.join(JOBS_DIR, job_id)
        folder = _folders[job_id] = row["folder"] or job_id
    return os.path.join(JOBS_DIR, folder)


def read_status(job_id: str) -> dict:
    try:
        with open(os.path.join(job_dir(job_id), "status.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def last_activity(job_id: str) -> float:
    """When the job last wrote anything: its status file and log, and the newest file of its state and output
    folders (a save in progress)."""
    d = job_dir(job_id)
    paths = [os.path.join(d, "status.json"), os.path.join(d, "log.txt")]
    for sub in ("state", "output", "samples"):
        try:
            paths += [e.path for e in os.scandir(os.path.join(d, sub))]
        except OSError:
            pass
    latest = 0.0
    for p in paths:
        try:
            latest = max(latest, os.path.getmtime(p))
        except OSError:
            pass
    return latest


def end_evaluation(job_id: str, cfg: dict, unrecorded: str = "failed") -> str | None:
    """Put a job back as it was before its checkpoint evaluation (``cfg["evaluate"]``) ran or was cancelled.
    If the evaluation process ended without recording how (killed, or crashed), it's recorded as
    ``unrecorded``. Returns how the evaluation ended (None: cancelled before it started)."""
    ev = cfg.pop("evaluate")
    db.execute("UPDATE jobs SET status=?, pid=NULL, config=? WHERE id=?", (ev["status"], json.dumps(cfg), job_id))
    st = read_status(job_id)
    record = st.get("evaluation")
    if not record or record.get("started_at", 0) < ev.get("requested", 0):
        return None  # it never started; any evaluation in the status file is an earlier one
    if record.get("state") == "running":
        record.update(state=unrecorded, finished_at=time.time(),
                      message="Stopped" if unrecorded == "stopped" else "The process ended without finishing; see the log")
        st.update(phase=ev.get("phase"), message=ev.get("message", ""))
        path = os.path.join(job_dir(job_id), "status.json")
        with open(path + ".tmp", "w") as f:
            json.dump(st, f)
        replace(path + ".tmp", path)
    return record["state"]


def send_control(job_id: str, action: str) -> None:
    path = os.path.join(job_dir(job_id), "control.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"action": action, "at": time.time()}, f)
    replace(path + ".tmp", path)


class Scheduler:
    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}
        self.stop_deadline: dict[str, float] = {}
        self._unrecorded: set[str] = set()  # jobs whose process ended, its end not yet recorded (``_reap``)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if self._thread.is_alive():  # already running (e.g. a second app lifespan in one process)
            return
        self._reattach()
        self._thread.start()

    def _reattach(self):
        """Reattach to training processes that outlived the previous server; a job that ended while the
        server was down gets the outcome it wrote to its status file."""
        from . import notify

        for j in db.query("SELECT * FROM jobs WHERE status IN ('running','stopping')"):
            if j["pid"] and procs.is_trainer(j["pid"], j["id"]):
                self.procs[j["id"]] = _Detached(j["pid"], j["id"])
                if j["status"] == "stopping":  # asked before the restart, when isn't kept: it gets the time from now
                    self.stop_deadline[j["id"]] = time.time() + STOP_GRACE
            elif "evaluate" in (j["config"] or {}):
                notify.evaluation_ended(j["id"], end_evaluation(j["id"], j["config"]))
            else:
                phase = read_status(j["id"]).get("phase")
                status = phase if phase in ("completed", "stopped", "failed") else "interrupted"
                db.execute("UPDATE jobs SET status=?, finished=?, pid=NULL WHERE id=?", (status, db.now(), j["id"]))
                notify.job_ended(j["id"], status)

    def _loop(self):
        from .llama_server import llama
        from .sandbox import sandbox

        while True:
            try:
                self.tick()
                sandbox.tick()
                llama.tick()
            except Exception as e:  # keep the scheduler alive
                print(f"[queue] error: {e}", file=sys.stderr)
            time.sleep(1.0)

    def tick(self):
        with self._lock:
            self._reap()
            self._launch()

    def _reap(self):
        for job_id, proc in list(self.procs.items()):
            code = proc.poll()
            if code is None:
                deadline = self.stop_deadline.get(job_id)
                # the grace runs from when it was asked or, if later, from the last thing it wrote
                if deadline and time.time() > max(deadline, last_activity(job_id) + STOP_GRACE):
                    procs.kill_tree(proc.pid)
                continue
            asked_to_stop = job_id in self.stop_deadline
            # the process is let go of only once how it ended is recorded: on a full disk (which may be what ended
            # it) the database can't take the write, and a job left "running" with no process to stop could only
            # be cleared by restarting the server. It's tried again each tick until there's room.
            try:
                ended = self._record_end(job_id, code, asked_to_stop)
            except Exception as e:
                if job_id not in self._unrecorded:
                    self._unrecorded.add(job_id)
                    print(f"[queue] job {job_id} ended (exit code {code}), but that couldn't be recorded: {e}. "
                          "Trying again until it can be.", file=sys.stderr)
                continue
            self._unrecorded.discard(job_id)
            del self.procs[job_id]
            self.stop_deadline.pop(job_id, None)
            ended()

    def _record_end(self, job_id: str, code: int, asked_to_stop: bool):
        """Record in the database how a job's process ended; returns the notification to send for it."""
        from . import notify

        cfg = (db.query_one("SELECT config FROM jobs WHERE id=?", (job_id,)) or {}).get("config") or {}
        if "evaluate" in cfg:
            killed = asked_to_stop  # (stopped, or forced to; killed unasked, it failed)
            state = end_evaluation(job_id, cfg, "stopped" if killed else "failed")
            return lambda: notify.evaluation_ended(job_id, state)
        status = {0: "completed", 3: "stopped"}.get(code, "failed")
        # killed because it was asked to stop (by the grace running out, or Force stop): stopped. Killed
        # unasked (the out-of-memory killer) it failed, and the notification says so
        if asked_to_stop and code != 0:
            status = "stopped"
        db.execute("UPDATE jobs SET status=?, exit_code=?, finished=?, pid=NULL WHERE id=?",
                   (status, code, db.now(), job_id))
        return lambda: notify.job_ended(job_id, status, code)

    def _launch(self):
        from . import sandbox as sbx
        from .llama_server import llama

        q = db.get_settings()["queue"]
        mode = q.get("mode", "sequential")
        gpus = [int(g) for g in q.get("gpus", [0])] or [0]
        # with 2+ GPUs a running Sandbox holds its GPU; training uses the others meanwhile
        reserved = sbx.sandbox.reserved_gpu()
        gpus = [g for g in gpus if g != reserved]
        if not gpus:
            return
        per_gpu = 1 if mode == "sequential" else max(1, int(q.get("max_per_gpu", 1)))
        usable = gpus if mode == "parallel_multi" else gpus[:1]
        total_cap = 1 if mode == "sequential" else per_gpu * len(usable)

        running = db.query("SELECT id, assigned_gpu FROM jobs WHERE status IN ('running','stopping')")
        if len(running) >= total_cap:
            return
        load = {g: 0 for g in usable}
        for r in running:
            if r["assigned_gpu"] in load:
                load[r["assigned_gpu"]] += 1
            elif r["assigned_gpu"] is not None:
                load[r["assigned_gpu"]] = load.get(r["assigned_gpu"], 0) + 1

        for job in db.query("SELECT * FROM jobs WHERE status='queued' ORDER BY position, created"):
            if sum(load.values()) >= total_cap:
                break
            pin = job["gpu"]
            if pin not in (None, "", "auto"):
                candidates = [int(pin)]
            else:
                candidates = sorted(usable, key=lambda g: load.get(g, 0))
            candidates = [g for g in candidates if g != reserved]
            gpu = next((g for g in candidates if load.get(g, 0) < per_gpu), None)
            if gpu is None:  # pinned to a GPU that's busy or reserved: the next job in line goes first
                continue
            if sbx.sandbox.occupies(gpu):  # one GPU: training comes first, the Sandbox unloads
                sbx.sandbox.yield_gpu()
                break
            if llama.occupies(gpu):  # so does llama-server, once the Auto-Caption using it is done
                llama.yield_gpu()
                break
            self._spawn(job, gpu)
            load[gpu] = load.get(gpu, 0) + 1

    def _spawn(self, job: dict, gpu: int):
        evaluating = "evaluate" in job["config"]
        d = job_dir(job["id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "job.json"), "w") as f:
            json.dump({**job["config"], "cache_dir": CACHE_DIR}, f, indent=2)
        try:
            os.remove(os.path.join(d, "control.json"))
        except OSError:
            pass
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(d, "log.txt"), "a", encoding="utf-8") as log:
            what = "checkpoint evaluation started" if evaluating else "started"
            log.write(f"\n===== {what} {time.strftime('%Y-%m-%d %H:%M:%S')} on GPU {gpu} =====\n")
            log.flush()
            proc = subprocess.Popen([sys.executable, "-m", "dainamo.engine.run", d], cwd=root,
                                    env=procs.worker_env(gpu), stdout=log, stderr=subprocess.STDOUT,
                                    **procs.detached())
        self.procs[job["id"]] = proc
        if evaluating:  # the job keeps its training times and exit code
            db.execute("UPDATE jobs SET status='running', assigned_gpu=?, pid=? WHERE id=?", (gpu, proc.pid, job["id"]))
        else:
            db.execute("UPDATE jobs SET status='running', assigned_gpu=?, pid=?, started=?, finished=NULL, "
                       "exit_code=NULL WHERE id=?", (gpu, proc.pid, db.now(), job["id"]))

    # --- actions -----------------------------------------------------------
    def stop(self, job_id: str):
        """Stop a job: one waiting is cancelled; one running is asked to (it saves its state and exits), and killed
        if it hasn't within ``STOP_GRACE``; one stopping is killed at once (Force stop). A job whose process has
        gone without its end being recorded (``_reap`` retrying) is stopped there and then."""
        with self._lock:
            job = db.query_one("SELECT status, config, pid FROM jobs WHERE id=?", (job_id,))
            if not job:
                return
            proc = self.procs.get(job_id)
            if job["status"] == "queued" and "evaluate" in (job["config"] or {}):
                end_evaluation(job_id, job["config"])  # cancelled before it started
            elif job["status"] in ("queued", "ready"):  # cancelled before it ran
                db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (job_id,))
            elif job["status"] in ("running", "stopping") and (proc is None or proc.poll() is not None):
                self._settle(job_id, job, proc)
            elif job["status"] == "running":
                # give it time to save state, then force
                self.stop_deadline[job_id] = time.time() + STOP_GRACE
                try:
                    send_control(job_id, "stop")
                except OSError:  # e.g. a full disk: SIGTERM asks it the same (``engine.run``)
                    procs.terminate(proc.pid)
                db.execute("UPDATE jobs SET status='stopping' WHERE id=?", (job_id,))
            elif job["status"] == "stopping":
                procs.kill_tree(proc.pid)

    def _settle(self, job_id: str, job: dict, proc):
        """Stop a job whose process has ended (or that has none: a server that lost it) as ``_reap`` would, its end
        recorded now. A process of its still about (not this server's) is killed first."""
        if proc is None and job["pid"] and procs.is_trainer(job["pid"], job_id):
            procs.kill_tree(job["pid"])
        code = proc.poll() if proc is not None else None
        ended = self._record_end(job_id, 1 if code is None else code, asked_to_stop=True)
        self._unrecorded.discard(job_id)
        self.procs.pop(job_id, None)
        self.stop_deadline.pop(job_id, None)
        ended()

    def paused(self) -> threading.Lock:
        """``with scheduler.paused():`` keeps the queue from starting or reaping jobs meanwhile."""
        return self._lock

    def requeue(self, job_id: str):
        with self._lock:
            db.execute("UPDATE jobs SET status='queued', position=?, exit_code=NULL WHERE id=? AND status NOT IN "
                       "('running','stopping')", (db.now(), job_id))


class _Detached:
    """A training process started by an earlier server instance (not our child)."""

    def __init__(self, pid: int, job_id: str):
        self.pid = pid
        self.job_id = job_id

    def poll(self) -> int | None:
        if procs.pid_alive(self.pid):
            return None
        return EXIT_CODES.get(read_status(self.job_id).get("phase"), 1)


scheduler = Scheduler()

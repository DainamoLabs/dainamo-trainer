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
import subprocess
import sys
import threading
import time

from ..fsutil import replace
from . import db, procs

JOBS_DIR = os.path.join(db.DATA_DIR, "jobs")
CACHE_DIR = os.path.join(db.DATA_DIR, "cache")  # encoded items and captions, shared by jobs (``engine.cache``)
EXIT_CODES = {"completed": 0, "stopped": 3}


def job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)


def read_status(job_id: str) -> dict:
    try:
        with open(os.path.join(job_dir(job_id), "status.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


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
        from . import notify

        for job_id, proc in list(self.procs.items()):
            code = proc.poll()
            if code is None:
                deadline = self.stop_deadline.get(job_id)
                if deadline and time.time() > deadline:
                    procs.kill_tree(proc.pid)
                continue
            del self.procs[job_id]
            asked_to_stop = self.stop_deadline.pop(job_id, None) is not None
            cfg = (db.query_one("SELECT config FROM jobs WHERE id=?", (job_id,)) or {}).get("config") or {}
            if "evaluate" in cfg:
                killed = code in procs.KILLED or asked_to_stop
                notify.evaluation_ended(job_id, end_evaluation(job_id, cfg, "stopped" if killed else "failed"))
                continue
            status = {0: "completed", 3: "stopped"}.get(code, "failed")
            if code in procs.KILLED or (asked_to_stop and code != 0):
                status = "stopped"
            db.execute("UPDATE jobs SET status=?, exit_code=?, finished=?, pid=NULL WHERE id=?",
                       (status, code, db.now(), job_id))
            notify.job_ended(job_id, status, code)

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
            if gpu is None:
                if mode == "sequential":
                    break  # strict FIFO
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
        with self._lock:
            job = db.query_one("SELECT status, config FROM jobs WHERE id=?", (job_id,))
            if not job:
                return
            if job["status"] == "queued" and "evaluate" in (job["config"] or {}):
                end_evaluation(job_id, job["config"])  # cancelled before it started
            elif job["status"] == "queued":
                db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (job_id,))
            elif job["status"] == "running":
                send_control(job_id, "stop")
                # give it time to save state, then force
                self.stop_deadline[job_id] = time.time() + 180
                db.execute("UPDATE jobs SET status='stopping' WHERE id=?", (job_id,))
            elif job["status"] == "stopping" and job_id in self.procs:
                procs.kill_tree(self.procs[job_id].pid)

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

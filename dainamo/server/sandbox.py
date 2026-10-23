"""Sandbox sessions: one worker process (``dainamo.engine.sandbox``) that keeps a profile's models
loaded on one GPU between generations.

GPU rules:
  * two or more GPUs ("reserve"): the Sandbox holds the GPU it runs on, and the training queue uses
    the others until the Sandbox is released (by hand, or after ``idle_minutes`` without use);
  * one GPU ("idle"): the Sandbox only runs while no training job needs the GPU. A job that is ready
    to start makes it finish the generation in progress and unload.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time

from . import db, procs

SANDBOX_DIR = os.path.join(db.DATA_DIR, "sandbox")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_dir(run_id: str) -> str:
    return os.path.join(SANDBOX_DIR, "runs", run_id)


_gpu_cache = {"at": 0.0, "count": 0}


def gpu_count() -> int:
    if time.time() - _gpu_cache["at"] > 60:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5).stdout
            _gpu_cache["count"] = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        except (OSError, subprocess.SubprocessError):
            _gpu_cache["count"] = 0
        _gpu_cache["at"] = time.time()
    return _gpu_cache["count"]


def mode() -> str:
    return "reserve" if gpu_count() >= 2 else "idle"


def training_gpus() -> set[int]:
    return {r["assigned_gpu"] for r in db.query("SELECT assigned_gpu FROM jobs WHERE status IN ('running','stopping')")
            if r["assigned_gpu"] is not None}


def free_gpu() -> int | None:
    """A GPU no training job or Sandbox is on (the last one first, since training starts from GPU 0), for
    short work such as transcribing speech; None when there's none."""
    busy = training_gpus() | ({sandbox.gpu} if sandbox.alive() else set())
    return next((g for g in reversed(range(gpu_count())) if g not in busy), None)


class SandboxManager:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.gpu: int | None = None
        self.state = "off"  # off | starting | ready | busy | stopping
        self.message = ""
        self.progress: dict | None = None
        self.current: str | None = None
        self.pending: list[str] = []
        self.last_used = 0.0
        self.yielding = False
        # the RefMod being made, or the last one: {id, name, state (queued | running | done | failed | cancelled),
        # message, made (its made.json, when done)}; it's queued with the generations, but isn't one
        self.refmod: dict | None = None
        self.lock = threading.RLock()

    # --- status ------------------------------------------------------------
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def occupies(self, gpu: int) -> bool:
        return self.alive() and self.gpu == gpu

    def reserved_gpu(self) -> int | None:
        return self.gpu if self.alive() and mode() == "reserve" else None

    def _progress_now(self) -> dict | None:
        """The last progress event, with the time elapsed and left brought up to now."""
        p = self.progress
        if not p:
            return None
        now = time.time()
        out = {k: v for k, v in p.items() if k not in ("at", "started")}
        if p.get("started"):
            out["elapsed"] = now - p["started"]
        if p.get("eta") is not None:
            out["eta"] = max(0.0, p["eta"] - (now - p["at"]))
        return out

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state if self.alive() else "off", "gpu": self.gpu if self.alive() else None,
                    "message": self.message, "progress": self._progress_now(), "current": self.current,
                    "pending": list(self.pending), "yielding": self.yielding,
                    "refmod": dict(self.refmod) if self.refmod else None,
                    "idle_seconds": int(time.time() - self.last_used) if self.alive() and self.last_used else None}

    # --- worker process ---------------------------------------------------------
    def _start(self, gpu: int):
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        with open(os.path.join(SANDBOX_DIR, "worker.log"), "a", encoding="utf-8") as log:
            log.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} on GPU {gpu} =====\n")
            log.flush()
            self.proc = subprocess.Popen([sys.executable, "-m", "dainamo.engine.sandbox"], cwd=ROOT,
                                         env=procs.worker_env(gpu), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=log, text=True, encoding="utf-8", bufsize=1)
        self.gpu, self.state, self.message, self.yielding = gpu, "starting", "Starting", False
        self.last_used = time.time()
        threading.Thread(target=self._read_events, args=(self.proc,), daemon=True).start()

    def _send(self, msg: dict):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError, AttributeError):
            pass

    def _stop(self):
        """Ask the worker to exit; it is killed if it hasn't within a minute."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self.state, self.message = "stopping", "Unloading"
        self._send({"cmd": "shutdown"})
        try:
            proc.stdin.close()
        except OSError:
            pass

        def reap():
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        threading.Thread(target=reap, daemon=True).start()

    def _read_events(self, proc: subprocess.Popen):
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.lock:
                self._on_event(ev)
        proc.wait()
        with self.lock:
            if proc is not self.proc:
                return
            why = ("The Sandbox stopped before this finished (see data/sandbox/worker.log)" if not self.yielding
                   else "Stopped: a training job needed the GPU")
            for rid in ([self.current] if self.current else []) + self.pending:
                db.execute("UPDATE sandbox_runs SET status='failed', message=? WHERE id=? AND status IN "
                           "('queued','running')", (why, rid))
            if self.refmod and self.refmod["state"] in ("queued", "running"):
                self.refmod.update(state="failed", message=why)
            self.current, self.pending, self.progress = None, [], None
            self.state, self.message = "off", ""

    def _on_event(self, ev: dict):
        kind, rid = ev.get("event"), ev.get("id")
        made = self.refmod if self.refmod and rid == self.refmod["id"] else None  # a RefMod's, not a run's
        if kind == "ready":
            self.state, self.message = "ready", "Ready"
        elif kind == "status":
            self.message = ev.get("message", "")
        elif kind == "started":
            self.current = rid
            if rid in self.pending:
                self.pending.remove(rid)
            self.state, self.progress = "busy", None
            if made:
                made["state"] = "running"
            else:
                db.execute("UPDATE sandbox_runs SET status='running', started=? WHERE id=?", (db.now(), rid))
        elif kind == "progress" and made:
            made["message"] = self.message = ev.get("message") or ""
        elif kind == "progress":
            self.progress = {**{k: v for k, v in ev.items() if k not in ("event", "id", "message")}, "at": time.time()}
            if ev.get("message"):
                self.message = ev["message"]
        elif kind == "output":
            row = db.query_one("SELECT outputs FROM sandbox_runs WHERE id=?", (rid,))
            outputs = (row or {}).get("outputs") or []
            outputs.append(ev["item"])
            db.execute("UPDATE sandbox_runs SET outputs=? WHERE id=?", (json.dumps(outputs), rid))
        elif kind in ("done", "error", "cancelled"):
            status = {"done": "done", "error": "failed", "cancelled": "cancelled"}[kind]
            if made:
                made.update(state=status, message=ev.get("message", ""), made=ev.get("made"))
            else:
                extra = json.dumps({"loras": ev.get("loras")}) if kind == "done" else None
                db.execute("UPDATE sandbox_runs SET status=?, message=?, elapsed=?, report=? WHERE id=?",
                           (status, ev.get("message", ""), ev.get("elapsed"), extra, rid))
            if rid in self.pending:
                self.pending.remove(rid)
            if self.current == rid:
                self.current = None
            self.progress = None
            self.last_used = time.time()
            if not self.pending:
                self.state, self.message = "ready", ""
                if self.yielding:
                    self._stop()

    # --- API ---------------------------------------------------------------
    def check_gpu(self, gpu: int) -> str | None:
        """Why the Sandbox can't run on ``gpu`` right now, or None."""
        if mode() == "idle":
            if db.query("SELECT id FROM jobs WHERE status IN ('running','stopping','queued') LIMIT 1"):
                return "The GPU is busy with training. With one GPU, the Sandbox runs while no job is running or queued."
            return None
        if gpu in training_gpus():
            return f"GPU {gpu} is training. Pick another GPU, or wait for the job to finish."
        if self.alive() and self.gpu != gpu and (self.current or self.pending):
            return f"The Sandbox is busy on GPU {self.gpu}. Wait for it, or cancel its generations first."
        return None

    def submit(self, run: dict, gpu: int, profile: dict):
        self._enqueue(gpu, {"cmd": "generate", "id": run["id"], "out_dir": run_dir(run["id"]),
                            "profile": {"id": profile["id"], "family": profile["family"], "data": profile["data"]},
                            "params": run["params"]})

    def submit_refmod(self, make: dict, gpu: int) -> bool:
        """Make a RefMod after the generations queued before it: ``make`` is the worker's "refmod" command, less
        its "cmd". False (nothing queued) while another is being made: one at a time, checked as it's queued."""
        with self.lock:
            if self.refmod and self.refmod["state"] in ("queued", "running"):
                return False
            self.refmod = {"id": make["id"], "name": make["name"], "state": "queued", "message": "", "made": None}
        self._enqueue(gpu, {"cmd": "refmod", **make})
        return True

    def _enqueue(self, gpu: int, msg: dict):
        """Send the worker on ``gpu`` a command (started there if needed). One on another GPU is stopped first, and
        one unloading (its stdin already closed) is let finish: either way a new one starts once it has exited."""
        with self.lock:
            old = self.proc if self.alive() and (self.gpu != gpu or self.state == "stopping") else None
            if old is not None and self.state != "stopping":
                self._stop()
        if old is not None:  # wait outside the lock (the event reader needs it)
            try:
                old.wait(timeout=120)
            except subprocess.TimeoutExpired:
                old.kill()
                old.wait()
        from .llama_server import llama
        if llama.occupies(gpu):  # generating needs its VRAM; enhancing a prompt afterwards uses another GPU or the CPU
            llama.yield_gpu("the Sandbox")
        with self.lock:
            if not self.alive():
                self._start(gpu)
            self.pending.append(msg["id"])
            self.last_used = time.time()
            self._send(msg)

    def cancel(self, run_id: str):
        with self.lock:
            if run_id == self.current or run_id in self.pending:
                self._send({"cmd": "cancel", "id": run_id})
            else:
                db.execute("UPDATE sandbox_runs SET status='cancelled' WHERE id=? AND status='queued'", (run_id,))

    def release(self):
        with self.lock:
            for rid in ([self.current] if self.current else []) + self.pending:
                self._send({"cmd": "cancel", "id": rid})
            self._stop()

    def yield_gpu(self):
        """A training job wants the (only) GPU: finish the current generation, drop the rest, unload."""
        with self.lock:
            if self.yielding or not self.alive():
                return
            self.yielding = True
            self.message = "A training job needs the GPU; unloading after this generation"
            for rid in list(self.pending):
                self._send({"cmd": "cancel", "id": rid})
            if not self.current:
                self._stop()

    def tick(self):
        """Release an idle Sandbox after ``idle_minutes``."""
        with self.lock:
            if not self.alive() or self.current or self.pending or self.state != "ready":
                return
            minutes = float(db.get_settings().get("sandbox", {}).get("idle_minutes", 15) or 0)
            if minutes > 0 and time.time() - self.last_used > minutes * 60:
                self.message = "Released after being idle"
                self._stop()

    def startup(self):
        os.makedirs(os.path.join(SANDBOX_DIR, "runs"), exist_ok=True)
        db.execute("UPDATE sandbox_runs SET status='failed', message='The server restarted' "
                   "WHERE status IN ('queued','running')")

    @staticmethod
    def delete_run(run_id: str):
        shutil.rmtree(run_dir(run_id), ignore_errors=True)
        db.execute("DELETE FROM sandbox_runs WHERE id=?", (run_id,))


sandbox = SandboxManager()

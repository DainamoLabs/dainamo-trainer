"""Worker processes on Linux and Windows: how they're started, found again after a server restart,
and stopped."""

from __future__ import annotations

import os
import signal
import subprocess

import psutil

WINDOWS = os.name == "nt"


def detached() -> dict:
    """Popen arguments for a training process that outlives the server (a restarted server reattaches
    to it) and doesn't receive the server console's Ctrl+C."""
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def worker_env(gpu: int | None) -> dict:
    """The environment for a worker on ``gpu`` (None: no GPU, the CPU only). ``gpu`` is nvidia-smi's index, which
    is in PCI bus order: CUDA numbers GPUs fastest first unless told otherwise, so on mixed GPUs the worker would
    run on another card than the one the UI shows."""
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
               PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.pop("DAINAMO_PASSWORD", None)
    if not WINDOWS:  # PyTorch doesn't support expandable segments on Windows and warns about it
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def is_trainer(pid: int, job_id: str) -> bool:
    """Whether ``pid`` is still this job's training process (and not a reused PID)."""
    try:
        cmd = " ".join(psutil.Process(pid).cmdline())
    except psutil.Error:
        return False
    return "dainamo.engine.run" in cmd and job_id in cmd


def pid_alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE  # a zombie has exited
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True


def kill_tree(pid: int) -> None:
    """Kill a worker and anything it started (e.g. ffmpeg)."""
    if not WINDOWS:
        try:
            os.killpg(pid, signal.SIGKILL)  # the worker leads its own session
        except OSError:
            pass
        return
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass


# exit codes of a process ended by a signal (POSIX only; on Windows a killed process exits with 1)
KILLED = {-getattr(signal, name) for name in ("SIGKILL", "SIGTERM") if hasattr(signal, name)}


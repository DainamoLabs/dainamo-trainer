"""The local LLM server Dainamo runs for Auto-Caption and Enhance prompt (Settings → Caption provider), so
they work without starting a server by hand: llama-server with a GGUF model ("llama.cpp"), or Dainamo's own
server for a model's text encoder ("Text encoder", dainamo/engine/textgen.py), which answers the same requests.

It starts when captioning or an enhancement needs it, loads the model once, and stops after
``idle_minutes`` without use. GPU rules, with training and the Sandbox first:
  * "auto" puts it on a GPU no training job or Sandbox is on (the last one first), else llama-server on the
    CPU; the text encoder needs a GPU, so it waits for one (the request fails, saying why);
  * a training job or Sandbox that's about to start on its GPU makes it stop, once the Auto-Caption or
    enhancement using it is done.

The command line, the readiness check (``/health``: 503 while the model loads) and the log hints follow
LLMHub (https://github.com/William-Kruta/LLMHub).
"""

from __future__ import annotations

import collections
import contextlib
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time

import httpx
import psutil

from . import db, procs

LOG_PATH = os.path.join(db.DATA_DIR, "local-llm.log")
PID_PATH = os.path.join(db.DATA_DIR, "local-llm.pid")  # "pid port", to stop one a crashed server left behind
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENCODER_SERVER = [sys.executable, "-m", "dainamo.engine.textgen"]
PROVIDERS = ("server", "llama", "encoder")  # a server the user runs, llama.cpp, a model's text encoder
HOST = "127.0.0.1"
FIRST_PORT = 8090  # not 8080: llama-server's own default, likely taken by a server the user runs
READY_TIMEOUT = 1800  # seconds to load a model; a big one from a slow disk takes minutes
LOG_LINES = 400
LOG_MAX_BYTES = 10 * 2**20  # the log file starts again (keeping one old copy) past this
# flags Dainamo sets, or that would stop it reaching the server, which extra arguments mustn't change
OWN_FLAGS = {"-m", "--model", "--host", "--port", "-mm", "--mmproj", "--api-key", "--api-key-file"}
# (what a failed start's log says, what to do about it); the first that matches is shown
HINTS = [
    (("out of memory", "cudamalloc failed", "failed to allocate"),
     "The model doesn't fit: pick a smaller quantization, a smaller context or the CPU, or keep part of it in "
     "system RAM with an extra argument such as -ngl 20."),
    (("address already in use", "couldn't bind", "failed to bind"), "Another program took the port; start it again."),
    (("unknown model architecture", "unknown architecture"), "This llama.cpp is too old for the model: update it."),
    (("failed to load multimodal", "clip_init", "clip_model_load", "mmproj"),
     "The vision projector (mmproj) didn't load: it must be the one made for this model."),
    (("error loading model", "failed to load model", "no such file", "failed to open"),
     "llama-server couldn't read the model: check that it's a complete .gguf file."),
    (("invalid argument", "unknown argument", "unrecognized"),
     "llama-server doesn't know one of the extra arguments (or this llama.cpp is too old for it)."),
]
ENCODER_HINTS = [
    (("out of memory",),
     "The text encoder doesn't fit in this GPU's free memory: pick a GPU with more free, or a smaller encoder "
     "(Krea 2's Qwen3-VL-4B), or use llama.cpp with a small GGUF model."),
    (("address already in use", "couldn't bind"), "Another program took the port; start it again."),
]


class LlamaError(RuntimeError):
    """The local LLM server couldn't start or stopped; the message says why, for the user."""


def provider(settings: dict) -> str:
    """Who answers Auto-Caption and Enhance prompt: "server" (one the user runs), "llama" or "encoder"."""
    choice = settings["captioner"].get("provider")
    return choice if choice in PROVIDERS else "server"


def find_binary(path: str = "") -> str:
    """The llama-server program: ``path``, or the one on the PATH."""
    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise LlamaError(f"llama-server wasn't found at {path}")
        if not os.access(path, os.X_OK):
            raise LlamaError(f"{path} isn't a program (it can't be run)")
        return path
    found = shutil.which("llama-server")
    if not found:
        raise LlamaError("llama-server wasn't found on the PATH. Install llama.cpp (github.com/ggml-org/llama.cpp/"
                         "releases) and choose its llama-server in Settings → Caption provider.")
    return found


def check_extra_args(text: str) -> list[str]:
    """The extra arguments as a list; ValueError when they can't be split or change a flag Dainamo sets."""
    try:
        args = shlex.split(text or "", posix=not procs.WINDOWS)
    except ValueError as e:
        raise ValueError(f"Extra arguments: {e}")
    taken = sorted({a.split("=", 1)[0] for a in args} & OWN_FLAGS)
    if taken:
        raise ValueError(f"Extra arguments can't set {', '.join(taken)}: Dainamo sets the model, the vision "
                         "projector and where the server listens")
    return args


def auto_context(settings: dict) -> int:
    """Context for one request: 8192 tokens for its images (up to four at 1024 px when enhancing, a clip's frames
    or a contact sheet) and the prompt, plus the longest reply with its retry (``captioner.RETRY_ROOM``)."""
    from .captioner import MAX_TOKENS, RETRY_ROOM

    longest = max(int(settings["captioner"].get("max_tokens") or 600), int(settings["enhancer"].get("max_tokens") or 400))
    return -(-(8192 + min(MAX_TOKENS, longest * RETRY_ROOM)) // 1024) * 1024


def free_port(start: int = FIRST_PORT) -> int:
    for port in range(start, start + 100):
        with socket.socket() as s:
            try:
                s.bind((HOST, port))
            except OSError:
                continue
            return port
    raise LlamaError(f"No free port for the local LLM server between {start} and {start + 99}")


def gpu_choice(cfg: dict) -> str:
    """The GPU setting: "auto", "cpu" or a GPU's index."""
    choice = cfg.get("gpu")
    return "auto" if choice in (None, "") else str(choice)


def _device(choice: str) -> tuple[int | None, str]:
    """(GPU, why on the CPU) for the GPU setting. None is the CPU, also when the chosen GPU is training or has
    the Sandbox: they need its VRAM, and starting next to them could run them out of memory."""
    if choice == "cpu":
        return None, ""
    from .sandbox import free_gpu, gpu_count, sandbox, training_gpus

    if choice.isdigit():
        gpu = int(choice)
        if gpu in training_gpus():
            return None, f"GPU {gpu} is training"
        if sandbox.occupies(gpu):
            return None, f"the Sandbox is on GPU {gpu}"
        return gpu, ""
    gpu = free_gpu()
    return gpu, "" if gpu is not None else "every GPU is training or running the Sandbox" if gpu_count() else "no GPU"


def has_gpu(binary: str, gpu: int) -> bool:
    """Whether this llama-server can use ``gpu``: a CPU-only build (like llama.cpp's plain Linux release) lists no
    devices. True when it can't tell, e.g. a llama.cpp too old to list them."""
    try:
        out = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=60,
                             stdin=subprocess.DEVNULL, env=procs.worker_env(gpu)).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    if "Available devices:" not in out:
        return True
    devices = [ln.strip() for ln in out.split("Available devices:", 1)[1].splitlines() if ln.strip()]
    return bool(devices) and devices != ["(none)"]


def llama_files(cfg: dict) -> tuple[str, str, str, list[str]]:
    """(llama-server, the model, the vision projector or "", the extra arguments) from the llama settings, checked.
    Raises LlamaError."""
    binary = find_binary(cfg.get("binary") or "")
    if not cfg.get("model"):
        raise LlamaError("Choose a model (.gguf) for llama-server in Settings → Caption provider")
    model = os.path.abspath(os.path.expanduser(cfg["model"]))
    if not os.path.isfile(model):
        raise LlamaError(f"The model wasn't found: {model}")
    mmproj = os.path.abspath(os.path.expanduser(cfg["mmproj"])) if cfg.get("mmproj") else ""
    if mmproj and not os.path.isfile(mmproj):
        raise LlamaError(f"The vision projector wasn't found: {mmproj}")
    try:
        extra = check_extra_args(cfg.get("extra_args") or "")
    except ValueError as e:
        raise LlamaError(str(e))
    return binary, model, mmproj, extra


def encoder_file(cfg: dict):
    """(the path, which encoder it is) from the text_encoder settings, checked. Raises LlamaError."""
    from ..engine.textgen import EncoderError, detect

    if not cfg.get("path"):
        raise LlamaError("Choose a text encoder file in Settings → Caption provider")
    path = os.path.abspath(os.path.expanduser(cfg["path"]))
    try:
        return path, detect(path)
    except EncoderError as e:
        raise LlamaError(str(e))


def describe(settings: dict) -> dict:
    """The caption provider Dainamo runs, checked without starting it, for the Auto-Caption dialog: {ok, managed,
    models, message} or {ok: false, managed, error}."""
    try:
        if provider(settings) == "encoder":
            _, enc = encoder_file(settings["text_encoder"])
            name, what = enc.name, f"{enc.name} ({enc.whose}'s text encoder)"
        else:
            _, model, mmproj, _ = llama_files(settings["llama"])
            name = os.path.basename(model)
            what = f"llama-server with {name}" + ("" if mmproj else ", which can't see images without its mmproj")
    except LlamaError as e:
        return {"ok": False, "managed": True, "error": str(e)}
    ready = llama.alive() and llama.state == "ready" and llama.key == LlamaServer._key(settings)
    return {"ok": True, "managed": True, "models": [name], "message": f"Dainamo runs {what}: "
            + (f"ready on {llama.device()}" if ready else "it starts when captioning starts")}


def where(gpu: int | None) -> str:
    return "the CPU" if gpu is None else f"GPU {gpu}"


def _cap(text: str) -> str:
    """``text`` to start a sentence; llama-server, a program's name, stays lowercase."""
    return text if text.startswith("llama-server") else text[:1].upper() + text[1:]


def failure(lines: list[str], code: int | None, hints=HINTS) -> tuple[str, str]:
    """(what went wrong, what to do about it) from the end of a failed server's log."""
    tail = [ln.strip() for ln in lines[-40:] if ln.strip()]
    bad = [ln for ln in tail if any(w in ln.lower() for w in ("error", "fail", "out of memory", "unknown", "invalid",
                                                                "unrecognized", "abort", "terminate"))]
    # a server's own "error: …" line says it best (the text encoder's follows the traceback it logs)
    said = " · ".join(dict.fromkeys(([ln for ln in bad if ln.startswith("error: ")] or bad)[-3:])) or \
        (tail[-1] if tail else "")
    said = said[:600] or f"it exited with code {code}"
    low = " ".join(bad or tail[-3:]).lower()
    hint = next((h for words, h in hints if any(w in low for w in words)), "")
    return said, hint


class LlamaServer:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.gpu: int | None = None
        self.port: int | None = None
        self.key: tuple | None = None  # the settings it was started with: other settings restart it
        self.command: list[str] = []
        self.model = ""
        self.why = ""  # why it's on the CPU, when "auto" or a GPU was asked for
        self.provider = "llama"  # what it runs: "llama" (llama-server) or "encoder" (a model's text encoder)
        self.what = "llama-server"
        self.hints = HINTS
        self.state = "off"  # off | starting | ready | stopping | failed
        self.message = ""
        self.error = ""
        self.hint = ""
        self.started = 0.0
        self.load_seconds: float | None = None
        self.last_used = 0.0
        self.users = 0  # Auto-Captions and enhancements using it now: it isn't stopped under them
        self.yielding = ""  # who's waiting for its GPU
        self.log: collections.deque[str] = collections.deque(maxlen=LOG_LINES)
        self.lock = threading.RLock()

    # --- status ------------------------------------------------------------
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def occupies(self, gpu: int) -> bool:
        return self.alive() and self.gpu == gpu

    def endpoint(self) -> str:
        return f"http://{HOST}:{self.port}/v1"

    def device(self) -> str:
        """Where it runs, and why on the CPU when that wasn't asked for."""
        return where(self.gpu) + (f" ({self.why})" if self.why else "")

    def snapshot(self) -> dict:
        with self.lock:
            alive = self.alive()
            state = self.state if alive or self.state == "failed" else "off"
            return {"state": state, "message": self.message, "error": self.error if state == "failed" else "",
                    "hint": self.hint if state == "failed" else "", "gpu": self.gpu if alive else None,
                    "device": self.device() if alive else None, "endpoint": self.endpoint() if alive else None,
                    "pid": self.proc.pid if alive else None, "model": self.model if alive else "",
                    "command": self.command, "users": self.users, "load_seconds": self.load_seconds,
                    "loading_seconds": int(time.time() - self.started) if state == "starting" else None,
                    "idle_seconds": int(time.time() - self.last_used) if state == "ready" and not self.users else None,
                    "provider": self.provider, "what": self.what}

    @staticmethod
    def _key(settings: dict) -> tuple:
        if provider(settings) == "encoder":
            cfg = settings["text_encoder"]
            return ("encoder", cfg.get("path") or "", gpu_choice(cfg))
        cfg = settings["llama"]
        return ("llama", cfg.get("binary") or "", cfg.get("model") or "", cfg.get("mmproj") or "", gpu_choice(cfg),
                int(cfg.get("ctx_size") or 0) or auto_context(settings), cfg.get("extra_args") or "")

    # --- the process --------------------------------------------------------------
    @staticmethod
    def _llama(settings: dict) -> tuple[list[str], int | None, str, str]:
        """(llama-server's command line, GPU, why on the CPU, model name)."""
        cfg = settings["llama"]
        binary, model, mmproj, extra = llama_files(cfg)
        gpu, why = _device(gpu_choice(cfg))
        if gpu is not None and not has_gpu(binary, gpu):
            gpu, why = None, "this llama-server has no GPU support: a CUDA build of llama.cpp runs on the GPU"
        ctx = int(cfg.get("ctx_size") or 0) or auto_context(settings)
        cmd = [binary, "--model", model, "--host", HOST, "--port", str(free_port()), "--ctx-size", str(ctx),
               "--parallel", "1", "--n-gpu-layers", "0" if gpu is None else "999"]
        return cmd + (["--mmproj", mmproj] if mmproj else []) + extra, gpu, why, os.path.basename(model)

    @staticmethod
    def _encoder(settings: dict) -> tuple[list[str], int | None, str, str]:
        """(the text-encoder server's command line, GPU, "", the encoder's name). It runs on a GPU only: on the
        CPU an 8B model in PyTorch would take minutes a caption."""
        cfg = settings["text_encoder"]
        path, enc = encoder_file(cfg)
        gpu, why = _device(gpu_choice(cfg))
        if gpu is None:
            raise LlamaError("The text encoder runs on a GPU: choose one in Settings → Caption provider" if not why else
                             "The text encoder runs on a GPU, and none was found: use llama.cpp, which can run on "
                             "the CPU" if why == "no GPU" else
                             f"The text encoder runs on a GPU, but {why} right now: try again later, or use "
                             "llama.cpp, which can run on the CPU")
        return ENCODER_SERVER + ["--model", path, "--host", HOST, "--port", str(free_port())], gpu, "", enc.name

    def _start(self, settings: dict):
        """Start the server for ``settings`` (not waiting for the model to load). Raises LlamaError."""
        encoder = provider(settings) == "encoder"
        what = "the text encoder" if encoder else "llama-server"
        try:
            cmd, gpu, why, model = self._encoder(settings) if encoder else self._llama(settings)
        except LlamaError as e:
            self.state, self.error, self.hint, self.message = "failed", str(e), "", ""
            raise
        port = int(cmd[cmd.index("--port") + 1])
        self.log.clear()
        os.makedirs(db.DATA_DIR, exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
            os.replace(LOG_PATH, LOG_PATH + ".old")
        log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        log.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')}: {what} on {where(gpu)} =====\n"
                  f"{shlex.join(cmd)}\n")
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    cwd=ROOT, env=procs.worker_env(gpu), text=True, encoding="utf-8",
                                    errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if procs.WINDOWS else 0)
        except OSError as e:
            log.close()
            self.state, self.error, self.hint, self.message = "failed", f"{_cap(what)} couldn't start: {e}", "", ""
            raise LlamaError(self.error)
        self.proc, self.gpu, self.port, self.key, self.command = proc, gpu, port, self._key(settings), cmd
        self.provider, self.what, self.hints = ("encoder", what, ENCODER_HINTS) if encoder else ("llama", what, HINTS)
        self.model, self.why = model, why
        self.state, self.error, self.hint, self.yielding = "starting", "", "", ""
        self.started, self.last_used, self.load_seconds = time.time(), time.time(), None
        self.message = f"Loading {self.model} on {self.device()}"
        with contextlib.suppress(OSError), open(PID_PATH, "w") as f:
            f.write(f"{proc.pid} {port}")
        threading.Thread(target=self._read, args=(proc, log), daemon=True).start()
        threading.Thread(target=self._watch, args=(proc, port), daemon=True).start()

    def _read(self, proc: subprocess.Popen, log):
        """Keep the output (in the log file and the last lines in memory) until it exits."""
        with log:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                self.log.append(line)
                log.write(line + "\n")
        code = proc.wait()
        with self.lock:
            if proc is not self.proc:
                return
            with contextlib.suppress(OSError):
                os.remove(PID_PATH)
            if self.state == "stopping":
                self.state = "off"
            elif self.state != "failed":
                loading = self.state == "starting"
                said, self.hint = failure(list(self.log), code, self.hints)
                self.error = (f"{_cap(self.what)} stopped while loading the model: " if loading else
                              f"{_cap(self.what)} stopped (exit code {code}): ") + said
                self.state, self.message = "failed", ""

    def _watch(self, proc: subprocess.Popen, port: int):
        """Ready once ``/health`` answers 200 (it answers 503 while the model loads)."""
        with httpx.Client(timeout=5) as client:
            while proc.poll() is None:
                try:
                    if client.get(f"http://{HOST}:{port}/health").status_code == 200:
                        with self.lock:
                            if self.proc is proc and self.state == "starting":
                                self.state, self.last_used = "ready", time.time()
                                self.load_seconds = round(time.time() - self.started, 1)
                                self.message = f"Ready on {self.device()}: loaded in {self.load_seconds:g} s"
                        return
                except httpx.HTTPError:
                    pass
                if time.time() - self.started > READY_TIMEOUT:
                    with self.lock:
                        if self.proc is proc and self.state == "starting":
                            self.state, self.hint = "failed", ""
                            self.error = (f"{_cap(self.what)} didn't finish loading the model within "
                                          f"{READY_TIMEOUT // 60} minutes")
                            proc.kill()
                    return
                time.sleep(0.5)

    def _stop(self, reason: str):
        """Ask the server to exit; it's killed if it hasn't within 10 seconds."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self.state, self.message, self.yielding = "stopping", reason, ""
        proc.terminate()

        def reap():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        threading.Thread(target=reap, daemon=True).start()

    # --- API ---------------------------------------------------------------
    def start(self, settings: dict, using: bool = False) -> bool:
        """Start the server for ``settings`` unless it runs with them already (or is in use with others); a
        running one with other settings is restarted. ``using``: the caller is one of its users. Returns whether
        it's running with ``settings``."""
        with self.lock:
            if self.alive() and self.state != "stopping":
                if self.key == self._key(settings) or self.users > using:
                    return self.key == self._key(settings)
                old = self.proc
                self._stop("Restarting with the new settings")
            else:
                old = self.proc if self.alive() else None
        if old is not None:  # wait for it outside the lock: its output reader needs the lock to finish
            try:
                old.wait(timeout=15)
            except subprocess.TimeoutExpired:
                old.kill()
                old.wait()
        with self.lock:
            if not self.alive():
                self._start(settings)
            return self.key == self._key(settings)

    def wait_ready(self, status=None, cancelled=None) -> str:
        """The endpoint (…/v1) once the model has loaded. ``status(message)`` hears how loading goes; a true
        ``cancelled()`` stops the wait. Raises LlamaError."""
        with self.lock:
            proc = self.proc
        while True:
            with self.lock:
                if self.proc is not proc or (self.state == "off" and not self.alive()):
                    raise LlamaError(f"{_cap(self.what)} was stopped before the model loaded")
                if self.state == "failed":
                    raise LlamaError(self.error + (f" {self.hint}" if self.hint else ""))
                if self.state == "ready":
                    return self.endpoint()
                if self.state == "stopping":
                    raise LlamaError(f"{_cap(self.what)} is stopping: " + self.message)
                message = (f"Starting {self.what}: {self.message[:1].lower()}{self.message[1:]} "
                           f"({int(time.time() - self.started)} s)")
            if cancelled and cancelled():
                raise LlamaError("Cancelled")
            if status:
                status(message)
            time.sleep(0.25)

    @contextlib.contextmanager
    def serving(self, settings: dict, status=None, cancelled=None):
        """The server's endpoint, started (and its model loaded) if needed. It isn't stopped for being idle, or
        for training, until the block ends."""
        with self.lock:
            self.users += 1
        try:
            self.start(settings, using=True)
            yield self.wait_ready(status, cancelled)
        finally:
            with self.lock:
                self.users -= 1
                self.last_used = time.time()
                if self.yielding and not self.users:
                    self._stop(f"Stopped: {self.yielding} needed {where(self.gpu)}")

    def lost(self) -> str:
        """What to say when the server stopped answering in the middle of a request."""
        return f"{_cap(self.what)} stopped answering" + (f": {self.error}" if self.error else "")

    def stop_unused(self, reason: str):
        """Stop it unless an Auto-Caption or enhancement is using it (then it stops when idle)."""
        with self.lock:
            if not self.users:
                self._stop(reason)

    def stop(self, wait: bool = False, reason: str = "Stopped"):
        with self.lock:
            proc = self.proc
            self._stop(reason)
        if wait and proc is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=15)

    def yield_gpu(self, who: str = "a training job"):
        """``who`` is about to start on the server's GPU: stop it now, or once the Auto-Caption or enhancement
        using it is done."""
        with self.lock:
            if not self.alive() or self.state == "stopping":
                return
            if self.users:
                self.yielding = who
                self.message = f"{who[:1].upper()}{who[1:]} is waiting for {where(self.gpu)}: stopping once it's done here"
                return
            self._stop(f"Stopped: {who} needed {where(self.gpu)}")

    def tick(self):
        """Stop the server after its ``idle_minutes`` without use."""
        with self.lock:
            if not self.alive() or self.users or self.state != "ready":
                return
            cfg = db.get_settings()["text_encoder" if self.provider == "encoder" else "llama"]
            minutes = float(cfg.get("idle_minutes") or 0)
            if minutes > 0 and time.time() - self.last_used > minutes * 60:
                self._stop(f"Stopped after {minutes:g} minutes idle")

    def startup(self):
        """Stop a server left running by a Dainamo server that didn't shut down: nothing can use it now."""
        try:
            with open(PID_PATH) as f:
                pid, port = (int(x) for x in f.read().split()[:2])
        except (OSError, ValueError):
            return
        with contextlib.suppress(psutil.Error):
            p = psutil.Process(pid)
            cmd = p.cmdline()
            if "--port" in cmd[:-1] and cmd[cmd.index("--port") + 1] == str(port):
                p.kill()
        with contextlib.suppress(OSError):
            os.remove(PID_PATH)


llama = LlamaServer()


@contextlib.contextmanager
def local_llm(settings: dict, cfg: dict | None = None, status=None, cancelled=None):
    """The captioner settings (``cfg``, by default those in ``settings``) for talking to the local LLM: as they
    are for a server the user runs, or with the address of the server Dainamo runs (llama-server or the text
    encoder), started if needed and kept until the block ends."""
    cfg = settings["captioner"] if cfg is None else cfg
    if provider(settings) == "server":
        yield cfg
        return
    with llama.serving(settings, status, cancelled) as endpoint:
        yield {**cfg, "endpoint": endpoint, "model": "", "api_key": ""}

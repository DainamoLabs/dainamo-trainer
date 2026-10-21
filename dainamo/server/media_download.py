"""Add from URL (experimental): yt-dlp downloads a video, or only its sound, into a dataset. Sites change often and
can break yt-dlp until it's updated, so the UI marks this experimental and can update it (``update``)."""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import threading

from ..engine.dataset import media_kind
from . import db
from .procs import WINDOWS, detached, kill_tree

DOWNLOADS = os.path.join(db.DATA_DIR, "downloads")  # a folder per download until its files are moved in
HEIGHTS = (0, 360, 480, 720, 1080, 1440, 2160)  # the largest a video may be; 0: the best there is
MAX_ITEMS = 200  # the most items one playlist download may add
JS_RUNTIMES = ("node", "bun")  # YouTube needs one; yt-dlp looks for deno itself, these only when told
MIN_VERSION = "2026.08.19"  # the oldest yt-dlp Dainamo runs, as in requirements.txt
# a line on stdout as each item starts downloading, for each progress report (a part cut by ffmpeg has none), each
# post-processing step and each finished file (``_line``)
TEMPLATES = [
    "--print", "before_dl:ITEM|%(playlist_index)s|%(n_entries)s|%(title)s",
    "--progress-template", "download:DL|%(info.playlist_index)s|%(info.n_entries)s|%(progress.downloaded_bytes)s|"
                           "%(progress.total_bytes,progress.total_bytes_estimate)s|%(info.title)s",
    "--progress-template", "postprocess:PP|%(progress.postprocessor)s|%(info.title)s",
    "--print", "after_move:FILE|%(filepath)s",
]
STEPS = {"Merger": "Joining video and sound", "ExtractAudio": "Extracting the sound",
         "VideoRemuxer": "Converting to mp4"}  # post-processors worth naming (yt-dlp adds an "FFmpeg" prefix)


def on_network() -> bool:
    """Whether this server answers other machines (``python -m dainamo --host 0.0.0.0`` or an address)."""
    return os.environ.get("DAINAMO_LISTEN", "127.0.0.1") not in ("127.0.0.1", "localhost", "::1")


def ytdlp() -> list[str] | None:
    """How to run yt-dlp: the one in Dainamo's environment (requirements.txt), else one on the PATH; None: neither."""
    importlib.invalidate_caches()  # ``update`` may have just installed it
    if importlib.util.find_spec("yt_dlp"):
        return [sys.executable, "-m", "yt_dlp"]
    exe = shutil.which("yt-dlp")
    return [exe] if exe else None


def _env() -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.pop("DAINAMO_PASSWORD", None)
    return env


_version: dict = {}  # the last ``tool`` answer, until an update


def release(version: str | None) -> tuple[int, ...]:
    """A yt-dlp version ("2026.08.19", or a nightly's "2026.08.19.232857") as (year, month, day); () if it isn't one."""
    parts = re.findall(r"\d+", version or "")[:3]
    return tuple(map(int, parts)) if len(parts) == 3 else ()


def tool() -> dict:
    """{installed, version, current (``MIN_VERSION`` or newer), minimum, bundled (in Dainamo's environment, which
    ``update`` can update), network}."""
    if not _version:
        cmd = ytdlp()
        version = None
        if cmd:
            try:
                r = subprocess.run([*cmd, "--version"], capture_output=True, text=True, timeout=60, env=_env(),
                                   creationflags=subprocess.CREATE_NO_WINDOW if WINDOWS else 0)
                version = r.stdout.strip() or None
            except (OSError, subprocess.TimeoutExpired):
                pass
        _version.update(installed=bool(version), version=version, current=release(version) >= release(MIN_VERSION),
                        minimum=MIN_VERSION, bundled=cmd == [sys.executable, "-m", "yt_dlp"])
    return {**_version, "network": on_network()}


def update() -> dict:
    """Install or update yt-dlp, with its recommended extras (YouTube's challenge solver among them), in Dainamo's
    environment, leaving the other packages as they are. -> ``tool()``."""
    uv = shutil.which("uv")
    if uv:  # environments made by uv have no pip
        cmd = [uv, "pip", "install", "--python", sys.executable, "--upgrade-package", "yt-dlp", "yt-dlp[default]"]
    elif importlib.util.find_spec("pip"):
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp[default]"]
    else:
        raise RuntimeError("Neither uv nor pip was found to install yt-dlp with")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=_env(),
                       creationflags=subprocess.CREATE_NO_WINDOW if WINDOWS else 0)
    if r.returncode:
        lines = (r.stderr or r.stdout).strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"the installer exited with {r.returncode}")
    _version.clear()
    return tool()


def command(url: str, out_dir: str, kind: str = "video", max_height: int = 1080, start: float | None = None,
            end: float | None = None, playlist: int = 0) -> list[str]:
    """The yt-dlp command that downloads ``url`` into ``out_dir``: a video of at most ``max_height`` (0: any), H.264
    in an mp4 where there's a choice, or with ``kind`` "audio" only the sound, as the site has it; with ``start``
    and/or ``end`` (seconds) only that part; ``playlist`` (0: just the one the URL names) a playlist's first that
    many items. Needs yt-dlp ``MIN_VERSION`` or newer."""
    cmd = [*ytdlp(), "--ignore-config", "--newline", "--no-colors", "--progress", "--no-mtime", "--windows-filenames",
           "-P", out_dir, "-o", "%(title).100B [%(id)s].%(ext)s", *TEMPLATES]
    cmd += [arg for rt in JS_RUNTIMES if shutil.which(rt) for arg in ("--js-runtimes", rt)]
    # a playlist's URL downloads the whole playlist even with --no-playlist; -I keeps it to its first item
    cmd += ["--yes-playlist", "--ignore-errors", "-I", f"1:{playlist}"] if playlist else ["--no-playlist", "-I", "1:1"]
    if kind == "audio":
        cmd += ["-f", "ba/b", "-x", "--audio-format", "best"]
    else:
        cmd += ["-S", f"res{f':{max_height}' if max_height else ''},vcodec:h264,ext", "--merge-output-format", "mp4",
                "--remux-video", "webm>webm/mov>mov/mp4"]
    if start is not None or end is not None:
        # cut where asked: without it a video starts at the keyframe before, and sound can run on 10 s past the end
        cmd += ["--download-sections", f"*{start or 0:g}-{'inf' if end is None else f'{end:g}'}",
                "--force-keyframes-at-cuts"]
    return cmd + ["--", url]


def free_name(folder: str, name: str) -> str:
    """``name``, or with " 2", " 3"... after its stem, so that no file in ``folder`` shares its stem (files that
    share a stem share a caption)."""
    stem, ext = os.path.splitext(name)
    stems = {os.path.splitext(f)[0].lower() for f in os.listdir(folder)} if os.path.isdir(folder) else set()
    cand, n = stem, 1
    while cand.lower() in stems:
        n += 1
        cand = f"{stem} {n}"
    return cand + ext


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


class DownloadJob:
    """Downloads ``url`` (``command``, with ``opts``) into ``dataset``'s folder in a thread: first into a folder of
    its own, so a half-done file never shows in the dataset, then each finished file is moved in under a name no
    other file has (``free_name``). Cancelling keeps the files already moved in."""

    def __init__(self, dataset: dict, url: str, opts: dict):
        self.dataset, self.url, self.opts = dataset, url, opts
        self.phase, self.title, self.step = "starting", "", ""
        self.item = self.items = 0
        self.done_bytes = self.total_bytes = 0.0
        self.saved: list[str] = []
        self.errors: list[str] = []
        self.running, self.cancelled = True, False
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.folder = os.path.join(DOWNLOADS, db.new_id())
        threading.Thread(target=self._run, daemon=True).start()

    def to_json(self) -> dict:
        return {"running": self.running, "phase": self.phase, "title": self.title, "step": self.step,
                "item": self.item, "items": self.items, "done_bytes": self.done_bytes, "total_bytes": self.total_bytes,
                "saved": self.saved[-50:], "count": len(self.saved), "errors": self.errors[-10:],
                "cancelled": self.cancelled, "url": self.url}

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.proc and self.proc.poll() is None:
                kill_tree(self.proc.pid)  # yt-dlp and the ffmpeg it runs

    def _run(self):
        stderr: list[str] = []
        try:
            os.makedirs(self.folder, exist_ok=True)
            cmd = command(self.url, self.folder, **self.opts)
            with self.lock:
                if self.cancelled:
                    return
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                             encoding="utf-8", errors="replace", env=_env(), **detached())
            reader = threading.Thread(target=lambda: stderr.extend(self.proc.stderr), daemon=True)
            reader.start()
            for line in self.proc.stdout:
                self._line(line.rstrip("\r\n"))
            self.proc.wait()
            reader.join(timeout=5)
            if not self.cancelled:
                self.errors += [ln[6:].strip()[:300] for ln in stderr if ln.startswith("ERROR:")]
                if self.proc.returncode and not self.errors:
                    last = [ln.strip() for ln in stderr if ln.strip()]
                    self.errors.append(last[-1][:300] if last else f"yt-dlp exited with {self.proc.returncode}")
        except Exception as e:
            if not self.cancelled:
                self.errors.append(str(e))
        finally:
            shutil.rmtree(self.folder, ignore_errors=True)
            self.phase = self.step = ""
            self.running = False

    def _line(self, line: str):
        kind, _, rest = line.partition("|")
        if kind == "ITEM":
            index, count, title = (rest.split("|", 2) + [""] * 3)[:3]
            self.phase, self.step, self.title = "downloading", "", title
            self.item = int(index) if index.isdigit() else max(self.item, 1)
            self.items = int(count) if count.isdigit() else max(self.items, 1)
            self.done_bytes = self.total_bytes = 0.0
        elif kind == "DL":
            index, count, done, total, title = (rest.split("|", 4) + [""] * 5)[:5]
            self.phase, self.step, self.title = "downloading", "", title
            self.item = int(index) if index.isdigit() else max(self.item, 1)
            self.items = int(count) if count.isdigit() else max(self.items, 1)
            self.done_bytes, self.total_bytes = _num(done), _num(total)
        elif kind == "PP":
            name, _, title = rest.partition("|")
            step = STEPS.get(name.removeprefix("FFmpeg"))
            if step:
                self.phase, self.step, self.title = "processing", step, title or self.title
        elif kind == "FILE":
            self._keep(rest)

    def _keep(self, path: str):
        name = os.path.basename(path)
        if not media_kind(name):
            self.errors.append(f"{name}: not a kind of file datasets take")
            return
        folder = self.dataset["path"]
        dest = os.path.join(folder, free_name(folder, name))
        shutil.move(path, dest)
        self.saved.append(os.path.basename(dest))


download_jobs: dict[str, DownloadJob] = {}  # by dataset id

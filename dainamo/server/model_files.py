"""The model manager: the files each family's profiles need (each driver's ``downloads``), which of them are in the
ComfyUI models folder (Settings), and downloading the missing ones from Hugging Face into the folder ComfyUI reads
each kind from (``diffusion_models``, ``text_encoders``…) instead of the Hugging Face cache, where ComfyUI never
looks.

Downloads run one at a time in a thread, into ``<name>.part`` beside where the file goes, so a half-done file never
looks finished. One cut short (cancelled, a dropped connection, the server stopping) carries on from where it
stopped. A finished file is checked against the size and SHA-256 Hugging Face lists, and its header read, before
it's renamed into place."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections import deque
from urllib.parse import quote

import httpx

from ..engine.families import FAMILIES
from . import db

# ComfyUI's folder for each kind of file, then the older names it still reads
KIND_FOLDERS = {"diffusion_models": ("diffusion_models", "unet"), "text_encoders": ("text_encoders", "clip"),
                "vae": ("vae",), "loras": ("loras",), "checkpoints": ("checkpoints",)}
STATE_FILE = os.path.join(db.DATA_DIR, "model_downloads.json")
SPARE_BYTES = 2 * 1024**3  # left free on the disk after a download
CHUNK = 1 << 20
MAX_FILES = 5000  # the most model files a scan looks at
ACTIVE = ("queued", "checking", "downloading", "verifying")
GB = 1e9  # as Hugging Face shows sizes
HF_URL = "https://huggingface.co"


def _gb(n: float) -> str:
    return f"{n / GB:.1f} GB" if n >= GB else f"{n / 1e6:.0f} MB"


def models_folder() -> str:
    """The ComfyUI models folder from Settings ("" when it isn't set)."""
    return (db.get_settings().get("models") or {}).get("folder", "")


def resolve_models_folder(path: str) -> str:
    """``path`` as a models folder: ComfyUI's own folder means the ``models`` folder in it."""
    path = os.path.abspath(os.path.expanduser(path))
    inner = os.path.join(path, "models")
    if os.path.isdir(inner) and not any(os.path.isdir(os.path.join(path, k)) for k in KIND_FOLDERS):
        return inner
    return path


def hf_token() -> str | None:
    """The Hugging Face token: HF_TOKEN, or the one ``hf auth login`` saved."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or None


def find_download(download_id: str):
    """(family driver, ``Download``) for a catalog id ("<repo>/<path>"); KeyError for one no family lists."""
    for F in FAMILIES.values():
        for d in F.downloads:
            if d.id == download_id:
                return F, d
    raise KeyError(f"No model family downloads {download_id!r}")


# ------------------------------------------------------------------------------------------------- scanning ---
_scanned: dict[tuple[str, int, int], dict] = {}  # (path, size, mtime) -> what the file is
_scan_lock = threading.Lock()


def _identify(path: str) -> dict:
    """{fields: [[family, field]...] whose signature the file has, quant: labels, unsupported: formats, error}."""
    from ..engine.comfy_io import describe_quant
    from ..engine.memory import tensors

    try:
        t = tensors(path)
    except Exception as e:
        return {"fields": [], "quant": [], "unsupported": [], "error": str(e)}
    fields = [[F.info.id, f] for F in FAMILIES.values() for f in F.signature_keys if F.matches(f, t)]
    quant, unsupported = set(), set()
    if fields:
        try:
            quant, unsupported = describe_quant(path)
        except Exception:
            pass
    return {"fields": fields, "quant": sorted(quant), "unsupported": sorted(unsupported), "error": ""}


def _profile_files() -> list[str]:
    out = []
    for p in db.query("SELECT data FROM profiles"):
        for v in (p["data"] or {}).values():
            if isinstance(v, str) and v.lower().endswith(".safetensors"):
                path = os.path.abspath(os.path.expanduser(v))
                if os.path.isfile(path):
                    out.append(path)
    return out


def _candidates(base: str) -> list[str]:
    """The .safetensors files in the models folder's kind folders (and theirs), then the ones profiles use."""
    paths = []
    for names in KIND_FOLDERS.values() if base else ():
        for name in names:
            for top, dirs, files in os.walk(os.path.join(base, name), followlinks=True):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                paths += [os.path.join(top, f) for f in sorted(files) if f.lower().endswith(".safetensors")]
                if len(paths) > MAX_FILES:
                    break
    return list(dict.fromkeys(paths + _profile_files()))[:MAX_FILES]


def scan() -> list[dict]:
    """Model files on this machine: {path, name, size, fields, quant, unsupported, error}. Only headers are read,
    once per file (until it changes)."""
    out = []
    with _scan_lock:
        seen = set()
        for path in _candidates(models_folder()):
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = (path, st.st_size, st.st_mtime_ns)
            seen.add(key)
            if key not in _scanned:
                _scanned[key] = _identify(path)
            out.append({"path": path, "name": os.path.basename(path), "size": st.st_size, **_scanned[key]})
        for key in set(_scanned) - seen:
            del _scanned[key]
    return out


def overview() -> dict:
    """What the Model files page shows: the models folder, each family's path fields with the files found for them
    and the ones to download, and the downloads."""
    base = models_folder()
    files = scan()
    have = {}  # (name, size) -> path: a catalog file is the file of that name and size, wherever it is
    for f in files:
        have.setdefault((f["name"], f["size"]), f["path"])
    tasks = downloads.list()
    pending = {t["download"]: t["id"] for t in tasks if t["state"] != "done"}

    def where(path: str) -> str:
        rel = os.path.relpath(path, base) if base else ""
        return rel if base and not rel.startswith("..") else path

    families = []
    for F in FAMILIES.values():
        fields = []
        for pf in F.info.profile_fields:
            if pf.kind != "path":
                continue
            options = [{**d.to_json(), "have": have.get((d.name, d.size)), "task": pending.get(d.id)}
                       for d in F.downloads if d.field == pf.key]
            known = {o["have"]: o["id"] for o in options if o["have"]}
            found = [{"path": f["path"], "name": f["name"], "size": f["size"], "where": where(f["path"]),
                      "quant": f["quant"], "unsupported": f["unsupported"], "download": known.get(f["path"])}
                     for f in files if [F.info.id, pf.key] in f["fields"] or f["path"] in known]
            found.sort(key=lambda f: (bool(f["unsupported"]), f["download"] is None, f["name"].lower()))
            fields.append({"key": pf.key, "label": pf.label, "hint": pf.hint, "required": pf.required,
                           "downloads": options, "found": found})
        families.append({"id": F.info.id, "label": F.info.label, "fields": fields})
    return {"folder": base, "folder_ok": bool(base) and os.path.isdir(base),
            "folders": {k: os.path.join(base, k) if base else "" for k in KIND_FOLDERS},
            "token": bool(hf_token()), "families": families, "tasks": tasks}


def suggested_folders() -> list[str]:
    """Models folders the profiles' files are in (a file in <x>/diffusion_models suggests <x>), most used first."""
    names = {n for ns in KIND_FOLDERS.values() for n in ns}
    counts: dict[str, int] = {}
    for path in _profile_files():
        d = os.path.dirname(path)
        while os.path.basename(d) not in names and os.path.dirname(d) != d:
            d = os.path.dirname(d)
        if os.path.basename(d) in names:
            counts[os.path.dirname(d)] = counts.get(os.path.dirname(d), 0) + 1
    return sorted(counts, key=lambda k: -counts[k])


def _existing(folder: str) -> str:
    while folder and not os.path.isdir(folder) and os.path.dirname(folder) != folder:
        folder = os.path.dirname(folder)
    return folder


def space(folder: str) -> dict:
    """{free bytes on the folder's disk, whether it exists, whether it's in the models folder (ComfyUI sees it)}."""
    folder = os.path.abspath(os.path.expanduser(folder))
    base = models_folder()
    inside = bool(base) and (folder + os.sep).startswith(os.path.abspath(base) + os.sep)
    try:
        free = shutil.disk_usage(_existing(folder)).free
    except OSError:
        free = None
    return {"free": free, "exists": os.path.isdir(folder), "in_models": inside}


def note_downloads(checks: list[dict], data: dict) -> list[dict]:
    """Profile checks with "File not found" replaced, for a file being downloaded, by how far it's got."""
    tasks = {os.path.abspath(t["dest"]): t for t in downloads.list() if t["state"] != "done"}
    for c in checks:
        val = data.get(c["field"])
        if c["level"] != "error" or not isinstance(val, str) or not val.strip():
            continue
        path = os.path.abspath(os.path.expanduser(val.strip()))
        t = tasks.get(path)
        if t and not os.path.exists(path):
            of = f"{int(100 * t['done'] / t['size']) if t['size'] else 0}% of {_gb(t['size'])}"
            say = {"queued": f"Waiting to download ({_gb(t['size'])})",
                   "failed": "The download failed (see Model files)",
                   "cancelled": f"The download is paused at {of}", "interrupted": f"The download is paused at {of}"}
            c.update(level="warning", message=say.get(t["state"], f"Downloading: {of}") + ". You can save the profile now")
    return checks


# ----------------------------------------------------------------------------------------------- downloads ---
class Cancelled(Exception):
    pass


def _refused(t: dict, token: str | None, status: int) -> str:
    page = f"https://huggingface.co/{t['repo']}"
    if not token:
        return (f"{t['repo']} needs a Hugging Face account: accept its terms at {page}, sign in on this machine "
                "with `hf auth login` (or set HF_TOKEN), then press Resume")
    return (f"Hugging Face refused the download ({status}): accept the terms at {page} with the account your token "
            "is for (a fine-grained token also needs read access to public gated repos), then press Resume")


class Downloads:
    """The download queue, kept in ``STATE_FILE`` so it outlives the server (what was running is then paused)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tasks: list[dict] = []
        self.thread: threading.Thread | None = None
        self.cancelling: set[str] = set()
        self.response: httpx.Response | None = None
        self.samples: deque = deque(maxlen=64)  # (time, bytes) of the running download, for its speed
        self.saved = 0.0
        try:
            with open(STATE_FILE) as f:
                self.tasks = json.load(f)
        except (OSError, ValueError):
            pass
        for t in self.tasks:
            if t["state"] in ACTIVE:  # the server stopped during it
                t.update(state="interrupted", speed=0, done=self._part_size(t))

    @staticmethod
    def _part_size(t: dict) -> int:
        try:
            return os.path.getsize(t["dest"] + ".part")
        except OSError:
            return 0

    def _save(self):
        """With the lock held."""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.tasks, f)
        os.replace(tmp, STATE_FILE)
        self.saved = time.monotonic()

    def _set(self, t: dict, **kw):
        with self.lock:
            t.update(kw)
            self._save()

    def list(self) -> list[dict]:
        with self.lock:
            return [dict(t) for t in reversed(self.tasks)]

    def _get(self, tid: str) -> dict:
        t = next((t for t in self.tasks if t["id"] == tid), None)
        if t is None:
            raise KeyError("No such download")
        return t

    def add(self, download_id: str, folder: str) -> dict:
        """Queue a catalog file to download into ``folder``. ValueError when it can't be."""
        F, d = find_download(download_id)
        folder = os.path.abspath(os.path.expanduser(folder.strip())) if folder.strip() else ""
        if not folder:
            raise ValueError("Choose the folder to download it into")
        dest = os.path.join(folder, d.name)
        with self.lock:
            same = next((t for t in self.tasks if t["dest"] == dest and t["state"] in ACTIVE), None)
            if same:
                return dict(same)
        if os.path.exists(dest):
            if os.path.getsize(dest) == d.size:
                raise ValueError(f"It's already there: {dest}")
            raise ValueError(f"A different {d.name} is already in {folder}")
        if os.path.exists(folder) and not os.path.isdir(folder):
            raise ValueError(f"{folder} is a file, not a folder")
        part = dest + ".part"
        need = d.size - (os.path.getsize(part) if os.path.exists(part) else 0)
        info = space(folder)
        if info["free"] is not None and info["free"] - need < SPARE_BYTES:
            raise ValueError(f"Not enough space: it needs {_gb(need)} and that disk has {_gb(info['free'])} free "
                             f"(Dainamo leaves {_gb(SPARE_BYTES)} spare)")
        t = {"id": db.new_id(), "download": d.id, "family": F.info.id, "family_label": F.info.label,
             "field": d.field, "repo": d.repo, "path": d.path, "revision": d.revision, "sha256": d.sha256,
             "name": d.name, "label": d.label, "dest": dest, "size": d.size, "done": self._part_size({"dest": dest}),
             "state": "queued", "error": "", "speed": 0, "added": time.time(), "finished": None, "gated": d.gated}
        with self.lock:
            self.tasks = [x for x in self.tasks if x["dest"] != dest]  # an earlier try at the same file
            self.tasks.append(t)
            self._save()
        self._start()
        return dict(t)

    def cancel(self, tid: str) -> dict:
        with self.lock:
            t = self._get(tid)
            if t["state"] == "queued":
                t.update(state="cancelled")
                self._save()
            elif t["state"] in ACTIVE:
                self.cancelling.add(tid)
                if self.response is not None:
                    try:
                        self.response.close()  # a stalled read gives up now, not at its timeout
                    except Exception:
                        pass
            return dict(t)

    def resume(self, tid: str) -> dict:
        with self.lock:
            t = self._get(tid)
            if t["state"] in ("cancelled", "failed", "interrupted"):
                t.update(state="queued", error="")
                self._save()
        self._start()
        return dict(t)

    def remove(self, tid: str):
        """Take a download off the list; one that isn't finished loses its part file too."""
        with self.lock:
            t = self._get(tid)
            if t["state"] in ACTIVE:
                raise ValueError("Cancel the download first")
            if t["state"] != "done":
                try:
                    os.remove(t["dest"] + ".part")
                except OSError:
                    pass
            self.tasks.remove(t)
            self._save()

    def _start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._work, daemon=True, name="model-downloads")
                self.thread.start()

    def _work(self):
        while True:
            with self.lock:
                t = next((t for t in self.tasks if t["state"] == "queued"), None)
                if t is None:
                    self.thread = None
                    return
                t.update(state="checking", error="", speed=0)
                self._save()
            try:
                self._download(t)
            except Exception as e:
                cancelled = isinstance(e, Cancelled) or t["id"] in self.cancelling
                self._set(t, state="cancelled" if cancelled else "failed", speed=0,
                          error="" if cancelled else (str(e) or type(e).__name__), done=self._part_size(t))
            finally:
                with self.lock:
                    self.cancelling.discard(t["id"])
                    self.response = None

    def _progress(self, t: dict, done: int):
        now = time.monotonic()
        self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[0][0] > 8:
            self.samples.popleft()
        t0, d0 = self.samples[0]
        t["done"] = done
        t["speed"] = (done - d0) / (now - t0) if now - t0 > 0.5 else t.get("speed", 0)
        if t["id"] in self.cancelling:
            raise Cancelled()

    def _download(self, t: dict):
        from ..engine.memory import tensors

        dest, part = t["dest"], t["dest"] + ".part"
        folder = os.path.dirname(dest)
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(dest):
            if os.path.getsize(dest) != t["size"]:
                raise RuntimeError(f"A different {t['name']} is already in {folder}")
            self._set(t, state="done", done=t["size"], finished=time.time())
            return
        have = self._part_size(t)
        if have > t["size"]:
            os.remove(part)
            have = 0
        free = shutil.disk_usage(folder).free
        if free - (t["size"] - have) < SPARE_BYTES:
            raise RuntimeError(f"Not enough space: {_gb(t['size'] - have)} to go and {_gb(free)} free")
        sha = hashlib.sha256() if t["sha256"] else None
        self.samples.clear()
        if have and sha:  # the part already downloaded counts towards the checksum
            with open(part, "rb") as f:
                read = 0
                while chunk := f.read(16 << 20):
                    sha.update(chunk)
                    read += len(chunk)
                    self._progress(t, read)
        token = hf_token()
        headers = {"User-Agent": "dainamo-trainer"}
        if token:
            headers["Authorization"] = f"Bearer {token}"  # httpx drops it on the redirect to the file's CDN
        if have:
            headers["Range"] = f"bytes={have}-"
        url = f"{HF_URL}/{t['repo']}/resolve/{t['revision']}/{quote(t['path'])}"
        self._set(t, state="downloading", done=have, speed=0)
        self.samples.clear()
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=httpx.Timeout(30.0, read=120.0)) as r:
                with self.lock:
                    self.response = r
                if r.status_code in (401, 403):
                    raise RuntimeError(_refused(t, token, r.status_code))
                if r.status_code == 404:
                    raise RuntimeError(f"Hugging Face doesn't have {t['path']} in {t['repo']} any more")
                if r.status_code == 200 and have:  # the whole file came back: start over
                    have = 0
                    sha = hashlib.sha256() if t["sha256"] else None
                elif r.status_code not in (200, 206):
                    raise RuntimeError(f"Hugging Face answered {r.status_code}")
                done = have
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_bytes(CHUNK):
                        f.write(chunk)
                        if sha:
                            sha.update(chunk)
                        done += len(chunk)
                        self._progress(t, done)
                        if time.monotonic() - self.saved > 10:
                            with self.lock:
                                self._save()
        except httpx.HTTPError as e:
            if t["id"] in self.cancelling:
                raise Cancelled() from e
            raise RuntimeError(f"The connection failed ({e or type(e).__name__}). Press Resume to carry on") from e
        size = self._part_size(t)
        if size != t["size"]:
            raise RuntimeError(f"The download stopped at {_gb(size)} of {_gb(t['size'])}. Press Resume to carry on")
        self._set(t, state="verifying", speed=0)
        if sha and sha.hexdigest() != t["sha256"]:
            os.remove(part)
            raise RuntimeError("The file's checksum doesn't match Hugging Face's, so it was deleted. Download it "
                               "again")
        try:
            tensors(part)
        except Exception as e:
            raise RuntimeError(f"The downloaded file can't be read: {e}") from e
        os.replace(part, dest)
        self._set(t, state="done", done=t["size"], finished=time.time())


downloads = Downloads()

"""The model manager: the files each family's profiles need (each driver's ``downloads``), which of them are on this
machine (in the ComfyUI models folder set on Model files, or in the Hugging Face cache), and downloading the missing
ones from Hugging Face into the folder ComfyUI reads each kind from (``diffusion_models``, ``text_encoders``…)
instead of the Hugging Face cache, where ComfyUI never looks.

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

from ..engine import upscale
from ..engine.families import FAMILIES
from ..engine.families.base import Download
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
    """The ComfyUI models folder from the settings ("" when it isn't set)."""
    return (db.get_settings().get("models") or {}).get("folder", "")


def hf_cache_folder() -> str:
    """The Hugging Face cache huggingface_hub reads and downloads into: HF_HUB_CACHE, else HF_HOME's ``hub``, else
    ~/.cache/huggingface/hub (as read when the server started)."""
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


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


# The upscalers Model files offers (``engine.upscale``), into the models folder's upscale_models, where a SeedVR2
# model looks for its VAE. They belong to no family: their "family" is UPSCALERS. The files are ComfyUI's own repacks.
UPSCALERS = ("upscalers", "Upscalers")
UPSCALER_FIELD = "upscaler"
UPSCALER_DOWNLOADS = [
    Download(UPSCALER_FIELD, "Comfy-Org/Real-ESRGAN", "RealESRGAN_x4plus.safetensors", 66857836,
             "Real-ESRGAN x4plus · 4×, photos", "5fd49b7b278836f48af63ecd314d0f98ab336105",
             "37f9a931c215f040aa6d50f711f2cb115f713c46df1d0d6469a8bd7bfe9a60bb", folder=upscale.FOLDER,
             recommended=True, note="A small, fast upscaler for images and video frames (BSD-3-Clause)."),
    Download(UPSCALER_FIELD, "Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_fp16.safetensors", 6784268336,
             "SeedVR2 3B · any size, video", "df48879708206a403d2a61acd55578c2e80fd233",
             "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660", folder=upscale.FOLDER,
             note="ByteDance's one-step video restorer (Apache-2.0). It needs the SeedVR2 VAE in the same folder: "
                  "download that too."),
    Download(UPSCALER_FIELD, "Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", 501324814,
             "SeedVR2 VAE · for SeedVR2 3B", "df48879708206a403d2a61acd55578c2e80fd233",
             "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1", folder=upscale.FOLDER,
             note="SeedVR2 3B reads and writes frames with it; it goes beside the model."),
]


def find_download(download_id: str) -> tuple[str, str, Download]:
    """(family id, its label, ``Download``) for a catalog id ("<repo>/<path>"); KeyError for one no family (nor
    the upscalers) lists."""
    for F in FAMILIES.values():
        for d in F.downloads:
            if d.id == download_id:
                return F.info.id, F.info.label, d
    for d in UPSCALER_DOWNLOADS:
        if d.id == download_id:
            return *UPSCALERS, d
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


# What each repo's main snapshot in the Hugging Face cache holds: repo folder -> (its folders and when each last
# changed, its models). A download adds files to a folder, which changes its mtime, so an unchanged cache is listed
# without reading anything but its folders.
_hf_repos: dict[str, tuple[tuple, list[dict]]] = {}


def _repo_id(folder: str) -> str:
    """"owner/repo" of a cache folder named ``models--owner--repo``."""
    return "/".join(folder.split("--")[1:])


def _hf_repo_id(path: str, cache: str) -> str | None:
    """"owner/repo" of a path in the Hugging Face cache (in its ``models--owner--repo`` folder), else None."""
    if not cache:
        return None
    rel = os.path.relpath(path, cache)
    top = rel.split(os.sep, 1)[0]
    return _repo_id(top) if not rel.startswith("..") and top.startswith("models--") else None


def _hf_models(repo_dir: str, repo: str) -> list[dict]:
    """The models in a cached repo's main snapshot (the one ``refs/main`` names): {path, name, repo, size, mtime,
    file: a single file's (device, inode)}, named as on the hub ("owner/repo", "owner/repo/text_encoder",
    "owner/repo/vae/ae.safetensors"). A folder whose shards an index lists, all of them downloaded, is one model: at
    the top of the repo its path is the repo folder, which ``comfy_io.shard_files`` reads at refs/main, so it follows
    the repo; in a subfolder (a diffusers pipeline's ``text_encoder/``) the subfolder. Any other .safetensors file is
    a model by itself, by its path in the snapshot, since some loaders read one file only (a diffusion model's)."""
    from ..engine.comfy_io import INDEX_NAME

    try:
        with open(os.path.join(repo_dir, "refs", "main")) as f:
            snap = os.path.join(repo_dir, "snapshots", f.read().strip())
        if not os.path.isdir(snap):
            return []
        tree, key = [], []
        for top, dirs, files in os.walk(snap):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            tree.append((top, sorted(files)))
            key.append((top, os.stat(top).st_mtime_ns))
    except (OSError, ValueError):
        return []
    key = tuple(key)
    if repo_dir in _hf_repos and _hf_repos[repo_dir][0] == key:
        return _hf_repos[repo_dir][1]
    models = []
    for top, names in tree:
        sub = os.path.relpath(top, snap).replace(os.sep, "/")
        sub = "" if sub == "." else sub
        weights = [n for n in names if n.endswith(".safetensors")]
        units, claimed = [], set()
        for index in (n for n in names if n.endswith(".safetensors.index.json")):
            try:
                with open(os.path.join(top, index)) as f:
                    shards = sorted(set(json.load(f)["weight_map"].values()))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
            claimed |= set(shards)  # a shard is never a model by itself
            if not shards or not set(shards) <= set(weights):  # still downloading, or only partly
                continue
            path = (top if sub else repo_dir) if index == INDEX_NAME else os.path.join(top, index)
            units.append((path, "/".join(filter(None, (repo, sub))), shards))
        units += [(os.path.join(top, n), "/".join(filter(None, (repo, sub, n))), [n])
                  for n in weights if n not in claimed]
        for path, name, files in units:
            try:
                stats = [os.stat(os.path.join(top, n)) for n in files]  # the blobs the snapshot links to
            except OSError:
                continue
            models.append({"path": path, "name": name, "repo": repo, "size": sum(s.st_size for s in stats),
                           "mtime": max(s.st_mtime_ns for s in stats),
                           "file": (stats[0].st_dev, stats[0].st_ino) if len(files) == 1 else None})
    _hf_repos[repo_dir] = (key, models)
    return models


def _hf_candidates(cache: str) -> list[dict]:
    """The models in the Hugging Face cache (``_hf_models`` of each ``models--owner--repo`` folder)."""
    out, seen = [], set()
    try:
        names = sorted(n for n in os.listdir(cache) if n.startswith("models--")) if cache else []
    except OSError:
        names = []
    for name in names:
        repo_dir = os.path.join(cache, name)
        seen.add(repo_dir)
        out += _hf_models(repo_dir, _repo_id(name))
        if len(out) >= MAX_FILES:
            break
    for gone in set(_hf_repos) - seen:
        del _hf_repos[gone]
    return out[:MAX_FILES]


def scan() -> list[dict]:
    """Model files on this machine: {path, name, size, source, repo, fields, quant, unsupported, error}. ``source``
    is "models" for the models folder's files and the ones profiles use, listed first, then "hf_cache" for the
    Hugging Face cache's models (``_hf_models``: a folder of shards is one, named "owner/repo"), with their
    "owner/repo". Only headers are read, once per file (until it changes)."""
    out = []
    with _scan_lock:
        seen = set()
        cache = hf_cache_folder()
        hf = _hf_candidates(cache)
        in_cache = {m["path"] for m in hf}

        def add(entry: dict, mtime: int):
            key = (entry["path"], entry["size"], mtime)
            seen.add(key)
            if key not in _scanned:
                _scanned[key] = _identify(entry["path"])
            out.append({**entry, **_scanned[key]})

        listed = set()  # (device, inode) of the files listed: one linked into the models folder is listed once
        for path in _candidates(models_folder()):
            if path in in_cache:  # a profile's file in the cache: listed with the cache's
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            listed.add((st.st_dev, st.st_ino))
            repo = _hf_repo_id(path, cache)  # a profile's file in another snapshot
            add({"path": path, "name": os.path.basename(path), "size": st.st_size,
                 "source": "hf_cache" if repo else "models", "repo": repo or ""}, st.st_mtime_ns)
        for m in hf:
            if m["file"] not in listed:
                add({"path": m["path"], "name": m["name"], "size": m["size"], "source": "hf_cache",
                     "repo": m["repo"]}, m["mtime"])
        for key in set(_scanned) - seen:
            del _scanned[key]
    return out


def overview() -> dict:
    """What the Model files page shows: the models folder and the Hugging Face cache, each family's path fields with
    the files found for them (models folder's first) and the ones to download, and the downloads. A family's
    ``extra_fields`` come last, optional and marked ``profile: false``: no profile uses them."""
    base = models_folder()
    files = scan()
    # (file name, size) -> path: a catalog file is the file of that name and size, wherever it is; the models
    # folder's copy, which ComfyUI sees too, before the cache's
    have = {}
    for f in files:
        have.setdefault((os.path.basename(f["path"]), f["size"]), f["path"])
    tasks = downloads.list()
    pending = {t["download"]: t["id"] for t in tasks if t["state"] != "done"}

    def where(path: str) -> str:
        rel = os.path.relpath(path, base) if base else ""
        return rel if base and not rel.startswith("..") else path

    # the catalog files of rows that are no profile's field (FLUX 3's prompt upsampler, a Qwen3.5-9B with MOSS
    # Image's encoder's tensors): a profile shouldn't pick one up for a field of its own
    extra = {have.get((d.name, d.size)) for F in FAMILIES.values() for d in F.downloads if d.field in F.extra_fields}
    families = []
    for F in FAMILIES.values():
        fields = []
        rows = [(pf.key, pf.label, pf.hint, pf.required) for pf in F.info.profile_fields if pf.kind == "path"]
        rows += [(key, label, hint, False) for key, (label, hint) in F.extra_fields.items()]
        for key, label, hint, required in rows:
            options = [{**d.to_json(), "have": have.get((d.name, d.size)), "task": pending.get(d.id)}
                       for d in F.downloads if d.field == key]
            known = {o["have"]: o["id"] for o in options if o["have"]}
            found = [{"path": f["path"], "name": f["name"], "size": f["size"], "where": where(f["path"]),
                      "source": f["source"], "repo": f["repo"], "quant": f["quant"], "unsupported": f["unsupported"],
                      "download": known.get(f["path"])}
                     for f in files
                     if f["path"] in known or ([F.info.id, key] in f["fields"] and f["path"] not in extra)]
            # the first usable one is what Create profile picks: the models folder's before the cache's
            found.sort(key=lambda f: (bool(f["unsupported"]), f["source"] != "models", f["download"] is None,
                                      f["name"].lower()))
            fields.append({"key": key, "label": label, "hint": hint, "required": required, "downloads": options,
                           "found": found, **({"profile": False} if key in F.extra_fields else {})})
        families.append({"id": F.info.id, "label": F.info.label, "fields": fields})
    cache = hf_cache_folder()
    return {"folder": base, "folder_ok": bool(base) and os.path.isdir(base),
            "folders": {k: os.path.join(base, k) if base else "" for k in (*KIND_FOLDERS, upscale.FOLDER)},
            "hf_cache": cache, "hf_cache_ok": bool(cache) and os.path.isdir(cache),
            "token": bool(hf_token()), "families": families, "upscalers": _upscalers_overview(base, pending, where),
            "tasks": tasks}


def _upscalers_overview(base: str, pending: dict, where) -> dict:
    """The Upscalers card: like a family with one field (any upscaler, and SeedVR2's VAE, to download), and the
    upscalers in upscale_models with what each is (``upscalers``)."""
    folder = os.path.join(base, upscale.FOLDER) if base else ""
    files = []
    for name in sorted(os.listdir(folder)) if folder and os.path.isdir(folder) else []:
        path = os.path.join(folder, name)
        if name.lower().endswith(upscale.EXTS) and os.path.isfile(path):
            files.append({"path": path, "name": name, "size": os.path.getsize(path)})
    have = {(f["name"], f["size"]): f["path"] for f in files}
    options = [{**d.to_json(), "have": have.get((d.name, d.size)), "task": pending.get(d.id)} for d in UPSCALER_DOWNLOADS]
    known = {o["have"]: o["id"] for o in options if o["have"]}
    found = [{**f, "where": where(f["path"]), "quant": [], "unsupported": [], "download": known.get(f["path"])}
             for f in files]
    field = {"key": UPSCALER_FIELD, "label": "Upscaler", "hint": "Any file in upscale_models", "required": False,
             "downloads": options, "found": found}
    listed = upscalers(base) if folder else {"models": [], "error": ""}
    return {"id": UPSCALERS[0], "label": UPSCALERS[1], "folder": folder, "folder_ok": bool(folder) and os.path.isdir(folder),
            "fields": [field], "models": listed["models"], "error": listed["error"]}


_upscalers_seen: dict = {"key": None, "models": [], "error": ""}
_upscalers_lock = threading.Lock()


def upscalers(base: str | None = None) -> dict:
    """{folder, models: [``upscale.Info`` as JSON, with its size]}: the upscalers in the models folder's
    upscale_models (``upscale.list_models``). Telling a .pth apart loads it, so the list is kept until a file in the
    folder changes."""
    base = models_folder() if base is None else base
    folder = os.path.join(base, upscale.FOLDER) if base else ""
    key = [folder]
    for name in sorted(os.listdir(folder)) if folder and os.path.isdir(folder) else []:
        try:
            st = os.stat(os.path.join(folder, name))
        except OSError:
            continue
        key.append((name, st.st_size, st.st_mtime_ns))
    with _upscalers_lock:
        if _upscalers_seen["key"] != key:
            models, error = [], ""
            try:
                found = upscale.list_models(base) if folder and os.path.isdir(folder) else []
            except Exception as e:  # a file that breaks the reader shouldn't break the page
                found, error = [], f"Couldn't read the upscale_models folder: {e}"
            for info in found:
                try:
                    size = os.path.getsize(info.path)
                except OSError:
                    size = 0
                models.append({**info.to_json(), "size": size})
            _upscalers_seen.update(key=key, models=models, error=error)
        return {"folder": folder, "models": list(_upscalers_seen["models"]), "error": _upscalers_seen.get("error", "")}


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
        family, family_label, d = find_download(download_id)
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
        t = {"id": db.new_id(), "download": d.id, "family": family, "family_label": family_label,
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

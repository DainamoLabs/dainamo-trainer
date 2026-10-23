"""SQLite persistence: profiles, datasets, jobs, settings and Sandbox runs, with schema migrations."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

DATA_DIR = os.environ.get("DAINAMO_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
DATA_DIR = os.path.abspath(DATA_DIR)
DB_PATH = os.path.join(DATA_DIR, "dainamo.db")

# Auto-Caption styles: the prompt sent with each image. Datasets remember the style they were captioned with
# (in caption_options, with the other Auto-Caption choices).
DEFAULT_CAPTION_PROMPT = ("Describe this image in one detailed paragraph for training an image generation model. "
                          "Describe the subject, setting, lighting, composition and style. Do not start with 'This image'.")
# the "long" caption of MOSS-Image's own recaptioning prompt (its USERGUIDE.md), as plain text instead of JSON
MOSS_CAPTION_STYLE = {
    "id": "moss", "name": "MOSS-Image: detailed, with text",
    "prompt": "Describe this image in one detailed paragraph for training an image generation model: the "
              "composition, key subjects, actions, setting, colors, lighting and artistic style. Include only what "
              "is visible. Quote any visible text exactly as written (don't translate, paraphrase or correct it) and "
              "describe its layout: line breaks, order, relative position, alignment, size, font, color and spacing. "
              "Don't mention watermarks. Reply with the paragraph only, without Markdown, and don't start with "
              "'This image'.",
}

CAPTION_STYLES = [
    {"id": "detailed", "name": "Detailed paragraph", "prompt": DEFAULT_CAPTION_PROMPT},
    {"id": "short", "name": "One sentence",
     "prompt": "Describe this image in one sentence for training an image generation model: the main subject, what "
               "they're doing, the setting and the style. Do not start with 'This image'."},
    {"id": "tags", "name": "Tags",
     "prompt": "List what's in this image as comma-separated tags for training an image generation model: the "
               "subject, clothing, pose, expression, setting, lighting, style and camera framing. Lowercase tags "
               "only, no sentences."},
    {"id": "video", "name": "Video: action and camera",
     "prompt": "Describe this clip in one detailed paragraph for training a video model: the subject and setting, "
               "what happens over the clip in order, the camera movement, and the lighting and style. You only see "
               "frames, so don't describe sounds. Do not start with 'This video'."},
    MOSS_CAPTION_STYLE,
]

DEFAULT_SETTINGS = {
    "queue": {"mode": "sequential", "gpus": [0], "max_per_gpu": 2},
    "sandbox": {"idle_minutes": 15},
    # Sandbox "Enhance prompt": the local LLM above rewrites the prompt, with an instruction per kind of workflow
    "enhancer": {
        "image": "You rewrite prompts for a text-to-image model. Expand the user's prompt into one detailed paragraph "
                 "describing the image: the subject, their appearance and pose, the setting, lighting, colours, "
                 "composition, and the camera or art style. Keep everything the user asked for, and keep names and "
                 "unusual words (they may be trigger words) exactly as written. Don't add text, logos or watermarks "
                 "unless asked. Reply with the prompt only.",
        "edit": "You rewrite instructions for an image-editing model. Make the user's edit instruction clear and "
                "specific: say exactly what to change and what to keep the same, in one or two sentences. Keep names "
                "and unusual words (they may be trigger words) exactly as written. Reply with the instruction only.",
        "video": "You rewrite prompts for a video model that also makes sound. Expand the user's prompt into one "
                 "detailed paragraph: the subject and setting, what happens over the clip in order, the camera "
                 "movement, the lighting and style, and the sounds or speech. Keep everything the user asked for, and "
                 "keep names and unusual words (they may be trigger words) exactly as written. Reply with the prompt "
                 "only.",
        "max_tokens": 400,
        "temperature": 0.7,
    },
    "notifications": {"url": "", "kind": "ntfy", "link": "", "on_completed": True, "on_failed": True,
                      "on_stopped": False},
    "captioner": {
        "endpoint": "http://127.0.0.1:8080/v1",
        "model": "",
        "api_key": "",
        "max_tokens": 600,  # a detailed paragraph, or a clip's events in order, can take 300-500
        "temperature": 0.2,
        "video_frames": 6,
        "video_mode": "frames",  # clips go as separate frames, or "sheet": one numbered grid (one-image models)
        # who answers: "server" (the endpoint above, a server you run), "llama" or "encoder" (run by Dainamo)
        "provider": "server",
    },
    # llama-server run by Dainamo for the "llama" provider (llama_server.py). gpu "auto" (a free GPU, else the
    # CPU), "cpu" or an index; ctx_size 0 sizes it for the requests
    "llama": {"binary": "", "model": "", "mmproj": "", "gpu": "auto", "ctx_size": 0, "extra_args": "",
              "idle_minutes": 10},
    # a model's text encoder writing the text, for the "encoder" provider (engine/textgen.py); gpu "auto" or an index
    "text_encoder": {"path": "", "gpu": "auto", "idle_minutes": 10},
    "speech": {"model": "large-v3-turbo", "device": "auto"},  # Whisper, for clips with sound; device "auto" or "cpu"
    # ComfyUI's models folder: the model manager looks for files in it and downloads into its subfolders
    "models": {"folder": ""},
    "caption_styles": {"default": "detailed", "styles": CAPTION_STYLES},
}

_lock = threading.RLock()


def _conn() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


# Schema steps, applied in order; PRAGMA user_version counts the ones a database has. Add a new step at
# the end to change the schema (never edit a released one): a SQL script, or a function taking the
# connection for changes that need Python. Name columns in INSERTs, so added columns don't break them.
MIGRATIONS: list = [
    # 1: the original tables (databases from before versioning have them already)
    """
    CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, name TEXT, family TEXT, data TEXT,
        created REAL, updated REAL);
    CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, name TEXT, path TEXT, managed INTEGER,
        trigger TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT, status TEXT, family TEXT,
        profile_name TEXT, gpu TEXT, assigned_gpu INTEGER, pid INTEGER, position REAL, created REAL,
        started REAL, finished REAL, exit_code INTEGER, config TEXT);
    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS sandbox_runs (id TEXT PRIMARY KEY, created REAL, started REAL, status TEXT,
        profile_id TEXT, profile_name TEXT, family TEXT, workflow TEXT, gpu INTEGER, params TEXT,
        outputs TEXT, report TEXT, message TEXT, elapsed REAL);
    """,
    # 2: caption styles and per-dataset Auto-Caption choices (see _caption_options below)
    lambda c: _caption_options(c),
    # 3: the MOSS-Image caption style, for caption styles saved before it was a default
    lambda c: _add_caption_style(c, MOSS_CAPTION_STYLE),
    # 4: each job's folder in data/jobs, named after the job (older jobs have none: their folder is their id)
    "ALTER TABLE jobs ADD COLUMN folder TEXT;",
]


def _caption_options(c: sqlite3.Connection) -> None:
    """Datasets remember their Auto-Caption choices (JSON: style, clip_frames, speech); a caption prompt changed
    from the old default becomes a caption style of its own, used by default."""
    c.execute("ALTER TABLE datasets ADD COLUMN caption_options TEXT")
    row = c.execute("SELECT value FROM settings WHERE key='captioner'").fetchone()
    old = (json.loads(row[0]).get("prompt") or "").strip() if row else ""
    if old and old != DEFAULT_CAPTION_PROMPT:
        styles = {"default": "mine", "styles": [{"id": "mine", "name": "My caption prompt", "prompt": old},
                                                 *CAPTION_STYLES]}
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('caption_styles', ?)", (json.dumps(styles),))


def _add_caption_style(c: sqlite3.Connection, style: dict) -> None:
    """Caption styles saved in Settings replace the defaults, so a new default style is added to them here
    (once: a style deleted later stays deleted)."""
    row = c.execute("SELECT value FROM settings WHERE key='caption_styles'").fetchone()
    if not row:
        return  # the defaults apply, and they have it
    styles = json.loads(row[0])
    if not any(st.get("id") == style["id"] for st in styles.get("styles", [])):
        styles["styles"].append(style)
        c.execute("UPDATE settings SET value=? WHERE key='caption_styles'", (json.dumps(styles),))


def schema_version(c: sqlite3.Connection) -> int:
    return c.execute("PRAGMA user_version").fetchone()[0]


def migrate(c: sqlite3.Connection, steps: list | None = None, backup_to: str | None = None) -> int:
    """Bring the database up to date, one transaction per step; returns the version reached.
    With ``backup_to``, an existing database is copied there before its first step runs."""
    steps = MIGRATIONS if steps is None else steps
    version = schema_version(c)
    if version > len(steps):
        raise RuntimeError(f"The database is from a newer Dainamo Trainer (schema {version}, this version knows "
                           f"{len(steps)}). Update the app, or point DAINAMO_DATA at another folder.")
    existing = c.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
    if version < len(steps) and existing and backup_to:
        with sqlite3.connect(backup_to) as copy:
            c.backup(copy)
    for n in range(version + 1, len(steps) + 1):
        step = steps[n - 1]
        c.commit()
        c.isolation_level = None  # explicit BEGIN/COMMIT, so a step and its version bump land together
        try:
            c.execute("BEGIN")
            if callable(step):
                step(c)
            else:
                for statement in _statements(step):
                    c.execute(statement)
            c.execute(f"PRAGMA user_version = {n}")
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        finally:
            c.isolation_level = ""
    return len(steps)


def _statements(script: str) -> list[str]:
    out, buf = [], ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


_db = None


def db() -> sqlite3.Connection:
    global _db
    if _db is None:
        c = _conn()
        migrate(c, backup_to=f"{DB_PATH}.v{schema_version(c)}.bak")
        _db = c
    return _db


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("data", "config", "params", "outputs", "report", "caption_options"):
        if k in d and isinstance(d[k], str):
            d[k] = json.loads(d[k])
    return d


def query(sql: str, args: tuple = ()) -> list[dict]:
    with _lock:
        return [row(r) for r in db().execute(sql, args).fetchall()]


def query_one(sql: str, args: tuple = ()) -> dict | None:
    with _lock:
        return row(db().execute(sql, args).fetchone())


def execute(sql: str, args: tuple = ()) -> None:
    with _lock:
        db().execute(sql, args)
        db().commit()


def get_settings() -> dict:
    out = json.loads(json.dumps(DEFAULT_SETTINGS))
    for r in query("SELECT key, value FROM settings"):
        val = json.loads(r["value"])
        if isinstance(out.get(r["key"]), dict) and isinstance(val, dict):
            out[r["key"]].update(val)
        else:
            out[r["key"]] = val
    return out


def put_setting(key: str, value) -> None:
    execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))


def now() -> float:
    return time.time()

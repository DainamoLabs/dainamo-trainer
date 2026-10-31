"""Dainamo Trainer web server: the FastAPI app, its middleware, and the built web UI.

The API routes are in ``routes/``, one module per area of the UI."""

from __future__ import annotations

import errno
import os
import shutil
import sqlite3
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..engine import families
from . import auth, db, media_download
from .llama_server import llama
from .queue import scheduler
from .routes import clips, datasets, jobs, llm, models, profiles, refmods, sandbox, similarity, system, tools, upscale
from .routes.common import not_found

WEB_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web", "dist"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.db()
    os.makedirs(datasets.THUMBS, exist_ok=True)
    os.makedirs(datasets.MANAGED_DATASETS, exist_ok=True)
    shutil.rmtree(media_download.DOWNLOADS, ignore_errors=True)  # what downloads cut short by a restart left
    scheduler.start()
    from .sandbox import sandbox as sandbox_manager
    sandbox_manager.startup()
    llama.startup()
    # the model families' code (about 2 s of imports) loads now, in the background, rather than on the first
    # request that needs it
    threading.Thread(target=families.load, daemon=True).start()
    yield
    sandbox_manager.release()
    llama.stop(wait=True)


app = FastAPI(title="Dainamo Trainer", lifespan=lifespan)

app.add_middleware(auth.AuthMiddleware)
app.add_middleware(auth.CrossSiteGuard)
# The API can browse the filesystem, so only answer requests addressed to this machine by name;
# this blocks DNS-rebinding from web pages. ``--host 0.0.0.0`` sets this to "*" (and turns on sign-in).
_hosts = [h.strip() for h in os.environ.get("DAINAMO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h.strip()]
if "*" not in _hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_hosts)

for module in (auth, system, profiles, models, datasets, clips, jobs, sandbox, refmods, upscale, similarity, llm, tools):
    app.include_router(module.router)


def _disk_full(e: Exception) -> bool:
    """A write that failed for want of space: the OS's ENOSPC, or SQLite's "database or disk is full"."""
    return (isinstance(e, OSError) and e.errno == errno.ENOSPC) or (
        isinstance(e, sqlite3.OperationalError) and "full" in str(e))


@app.exception_handler(OSError)
@app.exception_handler(sqlite3.OperationalError)
async def disk_full(request, e: Exception):
    """A full disk said as much (the data folder's, where the database, jobs and cache are), not as a bare 500."""
    if not _disk_full(e):
        raise e
    return JSONResponse({"detail": f"The disk is full ({db.DATA_DIR}): free some space and try again"}, status_code=507)


# ---------------------------------------------------------------- static ---
if os.path.isdir(WEB_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIST, "assets")), name="assets")


@app.get("/{full_path:path}")
def spa(full_path: str):
    if full_path.startswith("api/"):
        not_found()
    candidate = os.path.join(WEB_DIST, full_path)
    if full_path and os.path.isfile(candidate) and os.path.abspath(candidate).startswith(WEB_DIST):
        return FileResponse(candidate)
    index = os.path.join(WEB_DIST, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return PlainTextResponse("Web UI not built. Run: cd web && npm install && npm run build", status_code=503)

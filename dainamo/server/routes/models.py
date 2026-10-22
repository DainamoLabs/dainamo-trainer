"""Model files: what each family needs, what's in the ComfyUI models folder, and downloads from Hugging Face."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import model_files
from .common import not_found

router = APIRouter(prefix="/api/models")


@router.get("")
def overview():
    return model_files.overview()


@router.get("/suggest")
def suggest():
    """Models folders the profiles' files are in, for Settings to offer."""
    return {"folders": model_files.suggested_folders()}


@router.get("/space")
def space(folder: str):
    return model_files.space(folder)


@router.get("/downloads")
def list_downloads():
    return model_files.downloads.list()


class DownloadIn(BaseModel):
    id: str  # the catalog id: "<repo>/<path>"
    folder: str


@router.post("/downloads")
def start_download(d: DownloadIn):
    try:
        return model_files.downloads.add(d.id, d.folder)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e.args[0] if e.args else e))


@router.post("/downloads/{tid}/{action}")
def download_action(tid: str, action: str):
    if action not in ("cancel", "resume"):
        not_found()
    try:
        return getattr(model_files.downloads, action)(tid)
    except KeyError:
        not_found("Download not found")


@router.delete("/downloads/{tid}")
def remove_download(tid: str):
    try:
        model_files.downloads.remove(tid)
    except KeyError:
        not_found("Download not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}

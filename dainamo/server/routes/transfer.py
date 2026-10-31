"""Exporting a job to a folder and importing one from it (``job_transfer``), and the one running transfer's progress."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import job_transfer
from .common import job_or_404

router = APIRouter(prefix="/api")


def _call(fn, *args):
    try:
        return fn(*args)
    except job_transfer.Busy as e:
        raise HTTPException(409, str(e))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OSError as e:
        raise HTTPException(400, e.strerror or str(e))


@router.get("/jobs/{jid}/export-job")
def export_size(jid: str, datasets: bool = True):
    """What exporting the job would copy, and how big it is."""
    job_or_404(jid)
    return _call(job_transfer.export_summary, jid, datasets)


class ExportJobIn(BaseModel):
    path: str  # the folder the export's folder is made in
    datasets: bool = True  # copy the datasets' files too


@router.post("/jobs/{jid}/export-job")
def export_job(jid: str, body: ExportJobIn):
    job_or_404(jid)
    return _call(job_transfer.start_export, jid, body.path, body.datasets)


class InspectIn(BaseModel):
    path: str


@router.post("/import-job/inspect")
def inspect_export(body: InspectIn):
    return _call(job_transfer.inspect, body.path)


class ImportJobIn(BaseModel):
    path: str
    name: str
    profile_id: str
    datasets: dict[str, str]  # per exported dataset id: "export", or the id of a dataset here
    restart: bool = False


@router.post("/import-job")
def import_job(body: ImportJobIn):
    return _call(job_transfer.start_import, body.path, body.name, body.profile_id, body.datasets, body.restart)


@router.get("/transfer")
def transfer_status():
    return job_transfer.status()


@router.post("/transfer/cancel")
def transfer_cancel():
    job_transfer.cancel()
    return job_transfer.status()

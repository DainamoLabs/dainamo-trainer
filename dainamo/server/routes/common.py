"""Lookups shared by the API routes: a 404 with a readable message when the row is missing."""

from __future__ import annotations

from fastapi import HTTPException

from ...engine.families import get_family
from .. import db


def not_found(what: str = "Not found"):
    raise HTTPException(404, what)


def profile_or_404(pid: str) -> dict:
    return db.query_one("SELECT * FROM profiles WHERE id=?", (pid,)) or not_found("Profile not found")


def dataset_or_404(did: str) -> dict:
    return db.query_one("SELECT * FROM datasets WHERE id=?", (did,)) or not_found("Dataset not found")


def family_or_400(family_id: str):
    """The model family's driver, or a 400 for one Dainamo doesn't know."""
    try:
        return get_family(family_id)
    except KeyError:
        raise HTTPException(400, f"Unknown model family {family_id!r}")


def job_or_404(jid: str) -> dict:
    return db.query_one("SELECT * FROM jobs WHERE id=?", (jid,)) or not_found("Job not found")

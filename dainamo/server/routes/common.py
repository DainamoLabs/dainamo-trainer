"""Lookups shared by the API routes: a 404 with a readable message when the row is missing."""

from __future__ import annotations

from fastapi import HTTPException

from .. import db


def not_found(what: str = "Not found"):
    raise HTTPException(404, what)


def profile_or_404(pid: str) -> dict:
    return db.query_one("SELECT * FROM profiles WHERE id=?", (pid,)) or not_found("Profile not found")


def dataset_or_404(did: str) -> dict:
    return db.query_one("SELECT * FROM datasets WHERE id=?", (did,)) or not_found("Dataset not found")


def job_or_404(jid: str) -> dict:
    return db.query_one("SELECT * FROM jobs WHERE id=?", (jid,)) or not_found("Job not found")

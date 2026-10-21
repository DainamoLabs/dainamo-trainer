"""Model profiles: which files a model family loads from."""

from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import BaseModel

from .. import db
from .common import family_or_400, profile_or_404

router = APIRouter(prefix="/api/profiles")


class ProfileIn(BaseModel):
    name: str
    family: str
    data: dict = {}


@router.get("")
def list_profiles():
    return db.query("SELECT * FROM profiles ORDER BY name COLLATE NOCASE")


@router.post("")
def create_profile(p: ProfileIn):
    family_or_400(p.family)
    pid = db.new_id()
    db.execute("INSERT INTO profiles (id, name, family, data, created, updated) VALUES (?,?,?,?,?,?)",
               (pid, p.name.strip(), p.family, json.dumps(p.data), db.now(), db.now()))
    return profile_or_404(pid)


@router.put("/{pid}")
def update_profile(pid: str, p: ProfileIn):
    family_or_400(p.family)
    db.execute("UPDATE profiles SET name=?, family=?, data=?, updated=? WHERE id=?",
               (p.name.strip(), p.family, json.dumps(p.data), db.now(), pid))
    return profile_or_404(pid)


@router.delete("/{pid}")
def delete_profile(pid: str):
    db.execute("DELETE FROM profiles WHERE id=?", (pid,))
    return {"ok": True}


@router.post("/validate")
def validate_profile(p: ProfileIn):
    return family_or_400(p.family).validate_profile(p.data)

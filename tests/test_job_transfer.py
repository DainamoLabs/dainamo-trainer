"""Export and Import of a job (``server.job_transfer``): what the export folder holds, and a job imported from it as on
another machine (its datasets added from the export with their ids and modified times, so its cached encodings
still match), linked to a dataset already here, or started again on another diffusion model; a cancelled export
leaves nothing.

    .venv/bin/python -m pytest tests/test_job_transfer.py -v
"""

import json
import os
import shutil
import tempfile
import time

os.environ.setdefault("DAINAMO_DATA", tempfile.mkdtemp(prefix="dainamo-test-"))
os.environ["DAINAMO_ALLOWED_HOSTS"] = "127.0.0.1,localhost,testserver"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from dainamo.server import db, job_transfer  # noqa: E402
from dainamo.server.app import app  # noqa: E402
from dainamo.server.queue import CACHE_DIR, job_dir  # noqa: E402

MTIME = 1_600_000_000_123_456_789  # a dataset file's modified time, to the nanosecond


def _cache_files() -> set[str]:
    return {os.path.join(r, n) for r, _, names in os.walk(CACHE_DIR) for n in names}


@pytest.fixture(scope="module")
def client():
    """The app, and afterwards the data folder (which other test modules share) as it was: without the jobs,
    datasets and profiles these tests add, or the cache files. (The jobs are never queued, so none runs.)"""
    tables = ("jobs", "datasets", "profiles")
    rows = {t: {r["id"] for r in db.query(f"SELECT id FROM {t}")} for t in tables}
    cached = _cache_files()
    with TestClient(app) as c:
        yield c
    for t in tables:
        for r in db.query(f"SELECT id FROM {t}"):
            if r["id"] not in rows[t]:
                db.execute(f"DELETE FROM {t} WHERE id=?", (r["id"],))
    for p in _cache_files() - cached:
        os.remove(p)
    for r, dirs, names in sorted(os.walk(CACHE_DIR), reverse=True):
        if not dirs and not names and r != CACHE_DIR:
            os.rmdir(r)


def _file(path, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _wait(client) -> dict:
    for _ in range(200):
        st = client.get("/api/transfer").json()
        if st["state"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("the transfer didn't finish")


def _profile(client, tmp_path, name, model=b"model A"):
    data = {k: _file(tmp_path / "models" / name / f"{k}.safetensors", v)
            for k, v in (("diffusion_model", model), ("text_encoder", b"te"), ("vae", b"vae"))}
    return client.post("/api/profiles", json={"name": name, "family": "krea2", "data": data}).json()


def _trained_job(client, tmp_path, profile):
    """A Krea 2 job stopped at step 500 of 1000, with a dataset of two images and the cache files it used."""
    data = tmp_path / "photos"
    for n in ("a", "b"):
        os.utime(_file(data / f"{n}.png", n.encode() * 100), ns=(MTIME, MTIME))
        _file(data / f"{n}.txt", b"a photo")
    ds = client.post("/api/datasets", json={"name": "photos", "path": str(data), "trigger": "ph0to"}).json()
    jid = db.new_id()
    cfg = {"id": jid, "name": "my job", "output_name": "", "family": "krea2", "profile_id": profile["id"],
           "profile_name": profile["name"], "profile": profile["data"],
           "datasets": [{"id": ds["id"], "name": ds["name"], "path": ds["path"], "trigger": "ph0to", "repeats": 1}],
           "validation_datasets": [], "settings": {"steps": 1000, "rank": 16, "val_dataset": ds["id"]}}
    db.execute("INSERT INTO jobs (id, name, status, family, profile_name, gpu, position, created, started, finished, "
               "config, folder) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
               (jid, "my job", "stopped", "krea2", profile["name"], "1", db.now(), 1000.0, 1001.0, 1002.0,
                json.dumps(cfg), f"my_job_{jid}"))
    jd = job_dir(jid)
    _file(os.path.join(jd, "status.json"), json.dumps({"step": 500, "phase": "stopped"}).encode())
    _file(os.path.join(jd, "state", "last.pt"), b"state")
    _file(os.path.join(jd, "output", "my_job_000500.safetensors"), b"lora")
    _file(os.path.join(jd, "samples", "step000500_0.png"), b"sample")
    _file(os.path.join(jd, "log.txt"), b"trained\n")
    _file(os.path.join(jd, "control.json"), b"{}")  # (not exported)
    cached = ["latents/sig1/k1.pt", "text/sig2/k2.pt"]
    for rel in cached:
        _file(os.path.join(CACHE_DIR, rel), b"encoded " + rel.encode())
        _file(os.path.join(CACHE_DIR, os.path.dirname(rel), "signature.json"), b"{}")
    _file(os.path.join(jd, "cache.json"), json.dumps({"root": "/elsewhere", "files": cached}).encode())
    return jid, ds, cached


def _export(client, jid, tmp_path) -> str:
    out = tmp_path / "usb"
    out.mkdir(exist_ok=True)
    r = client.post(f"/api/jobs/{jid}/export-job", json={"path": str(out), "datasets": True})
    assert r.status_code == 200, r.text
    st = _wait(client)
    assert st["state"] == "done", st
    return st["path"]


def _forget(jid, ds, cached):
    """This machine as another: without the job, its dataset and the cache's files."""
    shutil.rmtree(job_dir(jid))
    db.execute("DELETE FROM jobs WHERE id=?", (jid,))
    db.execute("DELETE FROM datasets WHERE id=?", (ds["id"],))
    shutil.rmtree(ds["path"])
    for rel in cached:
        os.remove(os.path.join(CACHE_DIR, rel))


def test_an_exported_job_imports_as_it_was_on_another_machine(client, tmp_path):
    profile = _profile(client, tmp_path, "krea here")
    jid, ds, cached = _trained_job(client, tmp_path, profile)
    size = client.get(f"/api/jobs/{jid}/export-job").json()
    assert size["datasets"][0]["files"] == 4 and size["cache"] > 0 and size["total"] > size["job"]

    folder = _export(client, jid, tmp_path)
    assert os.path.basename(folder) == "my_job"
    m = json.load(open(os.path.join(folder, "dainamo_job.json")))
    assert m["job"]["step"] == 500 and m["job"]["status"] == "stopped"
    assert m["models"]["diffusion_model"]["fingerprint"] and m["datasets"][0]["folder"] == "datasets/photos"
    assert os.path.exists(os.path.join(folder, "job", "state", "last.pt"))
    assert not os.path.exists(os.path.join(folder, "job", "control.json"))
    assert all(os.path.exists(os.path.join(folder, "cache", rel)) for rel in cached)
    assert os.path.exists(os.path.join(folder, "cache", "latents", "sig1", "signature.json"))

    _forget(jid, ds, cached)
    # the profile here has the same files under other names: it's the same model
    other = _profile(client, tmp_path / "elsewhere", "krea there")
    seen = client.post("/api/import-job/inspect", json={"path": str(tmp_path / "usb")}).json()  # (the folder above it)
    assert seen["path"] == folder and seen["trained"] and seen["step"] == 500
    mine = {p["id"]: p for p in seen["profiles"]}
    assert mine[other["id"]]["identical"] and mine[other["id"]]["same_model"]
    assert seen["recommended"] in (profile["id"], other["id"])
    assert seen["datasets"][0]["included"] and seen["datasets"][0]["here"] is None

    r = client.post("/api/import-job", json={"path": folder, "name": "moved job", "profile_id": other["id"],
                                             "datasets": {ds["id"]: "export"}})
    assert r.status_code == 200, r.text
    st = _wait(client)
    assert st["state"] == "done", st
    job = client.get(f"/api/jobs/{st['job_id']}").json()
    assert job["status"] == "stopped" and job["resumable"] and job["gpu"] == "auto" and job["name"] == "moved job"
    cfg = job["config"]
    assert cfg["profile"] == other["data"] and cfg["profile_name"] == "krea there"
    added = db.query_one("SELECT * FROM datasets WHERE id=?", (ds["id"],))  # it kept its id
    assert added and added["managed"] and added["trigger"] == "ph0to"
    assert cfg["datasets"][0]["path"] == added["path"] and cfg["settings"]["val_dataset"] == ds["id"]
    assert os.stat(os.path.join(added["path"], "a.png")).st_mtime_ns == MTIME  # so the cache's names still match
    assert all(os.path.exists(os.path.join(CACHE_DIR, rel)) for rel in cached)
    jd = job_dir(job["id"])
    assert os.path.exists(os.path.join(jd, "output", "my_job_000500.safetensors"))
    assert "imported" in open(os.path.join(jd, "log.txt")).read()


def test_an_import_can_use_a_dataset_already_here(client, tmp_path):
    profile = _profile(client, tmp_path, "k1")
    jid, ds, _ = _trained_job(client, tmp_path, profile)
    folder = _export(client, jid, tmp_path)
    seen = client.post("/api/import-job/inspect", json={"path": folder}).json()
    assert seen["datasets"][0]["here"]["id"] == ds["id"]
    before = len(db.query("SELECT id FROM datasets"))
    client.post("/api/import-job", json={"path": folder, "name": "again", "profile_id": profile["id"],
                                         "datasets": {ds["id"]: ds["id"]}})
    st = _wait(client)
    assert st["state"] == "done", st
    cfg = client.get(f"/api/jobs/{st['job_id']}").json()["config"]
    assert cfg["datasets"][0]["id"] == ds["id"] and cfg["datasets"][0]["path"] == ds["path"]
    assert len(db.query("SELECT id FROM datasets")) == before


def test_on_another_diffusion_model_a_trained_job_starts_again(client, tmp_path):
    profile = _profile(client, tmp_path, "k2")
    jid, ds, _ = _trained_job(client, tmp_path, profile)
    folder = _export(client, jid, tmp_path)
    other = _profile(client, tmp_path, "k2 fp8", model=b"another model")
    seen = client.post("/api/import-job/inspect", json={"path": folder}).json()
    match = next(p for p in seen["profiles"] if p["id"] == other["id"])
    assert not match["same_model"] and next(f for f in match["fields"] if f["key"] == "diffusion_model")["same"] is False
    body = {"path": folder, "name": "new model", "profile_id": other["id"], "datasets": {ds["id"]: "export"}}
    r = client.post("/api/import-job", json=body)
    assert r.status_code == 400 and "step 0" in r.json()["detail"]
    assert client.post("/api/import-job", json={**body, "restart": True}).status_code == 200
    st = _wait(client)
    job = client.get(f"/api/jobs/{st['job_id']}").json()
    assert job["status"] == "ready" and not job["resumable"]
    jd = job_dir(job["id"])
    assert not any(os.path.exists(os.path.join(jd, p)) for p in ("state", "output", "samples", "status.json"))


def test_a_cancelled_export_leaves_nothing_and_a_busy_job_isnt_exported(client, tmp_path, monkeypatch):
    profile = _profile(client, tmp_path, "k3")
    jid, _, _ = _trained_job(client, tmp_path, profile)
    copy = job_transfer._copy

    def then_cancel(*a, **k):
        copy(*a, **k)
        job_transfer.cancel()
    monkeypatch.setattr(job_transfer, "_copy", then_cancel)
    (tmp_path / "usb").mkdir()
    client.post(f"/api/jobs/{jid}/export-job", json={"path": str(tmp_path / "usb")})
    st = _wait(client)
    assert st["state"] == "cancelled" and os.listdir(tmp_path / "usb") == []

    db.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    r = client.post(f"/api/jobs/{jid}/export-job", json={"path": str(tmp_path / "usb")})
    assert r.status_code == 400 and "stop it" in r.json()["detail"]
    db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (jid,))
    assert client.post("/api/import-job/inspect", json={"path": str(tmp_path)}).status_code == 400  # no export there

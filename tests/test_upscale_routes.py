"""Upscaling's routes: the upscalers listed from upscale_models, a run's output or a file upscaled as a Sandbox run,
and a dataset's small images upscaled one dataset at a time. The Sandbox's worker is a stand-in: what the manager
sends it is recorded, and its events are fed to the manager by hand.

    .venv/bin/python -m pytest tests/test_upscale_routes.py -v
"""

import io
import json
import os
import subprocess
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import header_only_safetensors
from dainamo.engine import upscale
from dainamo.server import db
from dainamo.server import sandbox as sbx


@pytest.fixture(scope="module")
def client():
    from dainamo.server.app import app

    with TestClient(app) as c:
        c.put("/api/settings", json={"queue": {"gpus": [99]}})  # no GPU to reserve or occupy
        yield c


class FakeProc:
    stdin = io.StringIO()

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def sent(monkeypatch):
    """A running Sandbox on GPU 1 of two, whose worker gets nothing: what the manager sends it is recorded. It was
    just used, so the queue's idle check doesn't release it mid-test."""
    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    for k, v in (("proc", FakeProc()), ("gpu", 1), ("state", "ready"), ("message", ""), ("progress", None),
                 ("current", None), ("pending", []), ("refmod", None), ("upscale", None), ("last_used", time.time()),
                 ("yielding", False)):
        monkeypatch.setattr(sbx.sandbox, k, v)
    out = []
    monkeypatch.setattr(sbx.sandbox, "_send", out.append)
    return out


@pytest.fixture
def models(client, tmp_path, monkeypatch):
    """A models folder whose upscale_models holds a pixel upscaler, a SeedVR2 model with its VAE, one without, and a
    file that's neither; what each is comes from a stand-in ``upscale.list_models``. -> {name: path}"""
    folder = tmp_path / "models" / "upscale_models"
    folder.mkdir(parents=True)
    paths = {}
    for name in ("RealESRGAN_x4plus", "seedvr2_3b_fp16", "seedvr2_ema_vae_fp16", "lonely_seedvr2", "notes"):
        paths[name] = header_only_safetensors(folder / f"{name}.safetensors", {"w": ("BF16", (4,))})
    infos = [upscale.Info(paths["RealESRGAN_x4plus"], "pixel", "RealESRGAN_x4plus", scale=4),
             upscale.Info(paths["seedvr2_3b_fp16"], "seedvr2", "seedvr2_3b_fp16", video=True,
                          companion=paths["seedvr2_ema_vae_fp16"]),
             upscale.Info(paths["lonely_seedvr2"], "seedvr2", "lonely_seedvr2", video=True,
                          error="SeedVR2 needs its VAE (seedvr2_ema_vae) in the same folder"),
             upscale.Info(paths["notes"], "unknown", "notes", error="not an upscaler")]
    seen = []
    monkeypatch.setattr(upscale, "list_models", lambda base: seen.append(base) or infos)
    assert client.put("/api/settings", json={"models": {"folder": str(tmp_path / "models")}}).status_code == 200
    yield paths
    client.put("/api/settings", json={"models": {"folder": ""}})


def test_upscalers_are_listed_from_the_models_folder(client, models, tmp_path):
    r = client.get("/api/upscalers").json()
    assert r["folder"] == str(tmp_path / "models" / "upscale_models") and r["error"] == ""
    by = {m["name"]: m for m in r["models"]}
    assert by["RealESRGAN_x4plus"]["kind"] == "pixel" and by["RealESRGAN_x4plus"]["scale"] == 4
    assert by["seedvr2_3b_fp16"]["video"] and by["seedvr2_3b_fp16"]["companion"] == models["seedvr2_ema_vae_fp16"]
    assert by["RealESRGAN_x4plus"]["size"] == os.path.getsize(models["RealESRGAN_x4plus"])
    client.put("/api/settings", json={"models": {"folder": ""}})
    assert client.get("/api/upscalers").json() == {"folder": "", "models": [], "error": ""}


def _image(path, size=(320, 240)):
    Image.new("RGB", size, (90, 140, 200)).save(path)
    return str(path)


def _run_with_output(name="img_0.png", size=(320, 240)):
    """A finished generation run with one image output: (run id, the image's path)."""
    rid = db.new_id()
    os.makedirs(sbx.run_dir(rid), exist_ok=True)
    path = _image(os.path.join(sbx.run_dir(rid), name), size)
    db.execute("INSERT INTO sandbox_runs (id, created, status, profile_id, profile_name, family, workflow, gpu, params, "
               "outputs) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (rid, db.now(), "done", "p", "Krea", "krea2", "t2i", 1, json.dumps({"prompt": "a cat"}),
                json.dumps([{"file": name, "kind": "image", "seed": 1, "variant": "lora"}])))
    return rid, path


def test_a_runs_output_is_upscaled_as_a_run(client, sent, models):
    src, path = _run_with_output()
    body = {"model": models["RealESRGAN_x4plus"], "factor": 2, "source": {"run_id": src, "file": "img_0.png"}}
    run = client.post("/api/sandbox/upscale", json=body).json()
    assert (run["workflow"], run["family"], run["profile_name"], run["status"], run["gpu"]) == \
        ("upscale", "upscale", "RealESRGAN_x4plus", "queued", 1)  # the Sandbox's GPU
    p = run["params"]
    assert p["input"] == {"path": path, "name": "img_0.png", "kind": "image", "width": 320, "height": 240,
                          "url": f"/api/sandbox/file/{src}/img_0.png", "poster": None}
    assert (p["width"], p["height"], p["kind"], p["scale"], p["upload"]) == (640, 480, "pixel", 4, None)
    assert p["source"]["run_id"] == src and p["source"]["file"] == "img_0.png" and p["source"]["n"]
    assert sent == [{"cmd": "upscale", "id": run["id"], "out_dir": sbx.run_dir(run["id"]),
                     "model": models["RealESRGAN_x4plus"], "inputs": [{"path": path, "kind": "image"}],
                     "factor": 2.0, "options": {}}]

    # the worker's events update the run like a generation's
    m = sbx.sandbox
    m._on_event({"event": "started", "id": run["id"]})
    m._on_event({"event": "progress", "id": run["id"], "message": "Upscaling img_0.png", "done": 0, "total": 1})
    assert m.snapshot()["progress"] == {"done": 0, "total": 1} and m.message == "Upscaling img_0.png"
    item = {"file": "img_0_x2.png", "kind": "image", "source": path, "width": 640, "height": 480}
    m._on_event({"event": "output", "id": run["id"], "item": item})
    m._on_event({"event": "done", "id": run["id"], "elapsed": 1.5})
    row = next(r for r in client.get("/api/sandbox").json()["runs"] if r["id"] == run["id"])
    assert row["status"] == "done" and row["outputs"] == [item] and row["elapsed"] == 1.5
    assert m.state == "ready" and m.current is None

    for bad, code in (({"source": {"run_id": src, "file": "img_0_poster.png"}}, 404),  # not one of its outputs
                      ({"source": {"run_id": "nope", "file": "img_0.png"}}, 404),
                      ({"upload": path}, 400),  # both
                      ({"source": None}, 400),  # neither
                      ({"factor": 1}, 400), ({"factor": 9}, 400),
                      ({"model": models["lonely_seedvr2"]}, 400), ({"model": models["notes"]}, 400),
                      ({"model": "/no/such.safetensors"}, 400), ({"options": {"tiles": [1, 2]}}, 400)):
        r = client.post("/api/sandbox/upscale", json={**body, **bad})
        assert r.status_code == code, (bad, r.json())
    r = client.post("/api/sandbox/upscale", json={**body, "model": models["lonely_seedvr2"]})
    assert "needs its VAE" in r.json()["detail"]
    assert len(sent) == 1  # nothing else was queued


def test_files_are_uploaded_to_upscale(client, sent, models, tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (200, 100)).save(buf, "JPEG")
    up = client.post("/api/sandbox/upload", files={"file": ("photo.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert up["kind"] == "image" and up["path"].startswith(sbx.UPLOADS_DIR) and up["name"] == "photo.jpg"
    assert client.get(up["url"]).status_code == 200
    r = client.post("/api/sandbox/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400

    body = {"model": models["seedvr2_3b_fp16"], "factor": 3, "upload": up["path"], "options": {"tiles": 2}}
    run = client.post("/api/sandbox/upscale", json=body).json()
    p = run["params"]
    assert p["input"]["url"] == up["url"] and p["upload"] == up["path"] and p["source"] is None
    assert (p["width"], p["height"], p["kind"]) == (600, 300, "seedvr2")
    assert sent[-1]["inputs"] == [{"path": up["path"], "kind": "image"}] and sent[-1]["options"] == {"tiles": 2}

    # any image or clip on this machine: a clip keeps its kind, frames and rate
    clip = tmp_path / "walk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=24:duration=2",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    run = client.post("/api/sandbox/upscale", json={**body, "upload": str(clip), "factor": 4}).json()
    assert {k: run["params"]["input"][k] for k in ("kind", "width", "height", "frames", "fps", "url")} == \
        {"kind": "video", "width": 160, "height": 96, "frames": 48, "fps": 24.0, "url": None}
    assert sent[-1]["inputs"] == [{"path": str(clip), "kind": "video"}] and sent[-1]["factor"] == 4.0
    r = client.post("/api/sandbox/upscale", json={**body, "upload": _image(tmp_path / "wide.png", (2100, 1000)),
                                                  "factor": 4})
    assert r.status_code == 400 and "smaller factor" in r.json()["detail"]  # 8400 px wide
    assert client.post("/api/sandbox/upscale", json={**body, "upload": str(tmp_path / "gone.png")}).status_code == 400


def _dataset(client, folder, name):
    return client.post("/api/datasets", json={"name": name, "path": str(folder)}).json()


def test_a_datasets_small_images_are_upscaled_one_dataset_at_a_time(client, sent, models, tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    _image(folder / "big.png", (1024, 1024))
    _image(folder / "small.jpg", (400, 300))
    _image(folder / "tiny.png", (100, 100))  # needs 10×: upscaled 4×, so still small
    _image(folder / "c.jpg", (300, 300))  # its result, c.png, would replace this one:
    _image(folder / "c.png", (2000, 2000))
    (folder / "small.txt").write_text("a small photo")
    d = _dataset(client, folder, "photos")
    other = _dataset(client, tmp_path, "other")

    plan = client.get(f"/api/datasets/{d['id']}/upscale/plan", params={"resolution": 1024, "max_factor": 4}).json()
    assert {k: plan[k] for k in ("images", "small", "capped", "clash", "unreadable", "clips")} == \
        {"images": 5, "small": 2, "capped": 1, "clash": ["c.jpg"], "unreadable": 0, "clips": 0}
    assert plan["smallest"] == {"name": "tiny.png", "width": 100, "height": 100}

    body = {"model": models["RealESRGAN_x4plus"], "resolution": 1024, "max_factor": 4}
    st = client.post(f"/api/datasets/{d['id']}/upscale", json=body).json()
    assert (st["state"], st["done"], st["total"], st["dataset"], st["model_name"]) == \
        ("queued", 0, 2, "photos", "RealESRGAN_x4plus")
    assert sent == [{"cmd": "upscale_dataset", "id": st["id"], "model": models["RealESRGAN_x4plus"],
                     "paths": [str(folder / "small.jpg"), str(folder / "tiny.png")], "resolution": 1024,
                     "max_factor": 4.0}]
    # one at a time, whichever dataset
    assert client.post(f"/api/datasets/{d['id']}/upscale", json=body).status_code == 409
    r = client.post(f"/api/datasets/{other['id']}/upscale", json=body)
    assert r.status_code == 409 and "photos is being upscaled" in r.json()["detail"]
    elsewhere = client.get(f"/api/datasets/{other['id']}/upscale").json()
    assert elsewhere["state"] == "none" and elsewhere["elsewhere"] == {"id": d["id"], "name": "photos"}
    assert elsewhere["last"] == {"model": models["RealESRGAN_x4plus"], "resolution": 1024, "max_factor": 4.0}

    m = sbx.sandbox
    m._on_event({"event": "started", "id": st["id"]})
    m._on_event({"event": "progress", "id": st["id"], "done": 1, "total": 2, "message": "small.jpg: 400×300 → 1182×887"})
    st = client.get(f"/api/datasets/{d['id']}/upscale").json()
    assert (st["state"], st["done"], st["total"], st["message"]) == ("running", 1, 2, "small.jpg: 400×300 → 1182×887")
    assert m.snapshot()["upscale"]["done"] == 1 and m.progress is None  # not a run's progress
    client.post(f"/api/datasets/{other['id']}/upscale/cancel")  # not that dataset's
    assert sent[-1]["cmd"] == "upscale_dataset"
    client.post(f"/api/datasets/{d['id']}/upscale/cancel")
    assert sent[-1] == {"cmd": "cancel", "id": st["id"]}
    m._on_event({"event": "cancelled", "id": st["id"]})
    assert client.get(f"/api/datasets/{d['id']}/upscale").json()["state"] == "cancelled" and m.state == "ready"

    # once it has ended, another can go
    st = client.post(f"/api/datasets/{d['id']}/upscale", json={**body, "max_factor": 3}).json()
    m._on_event({"event": "started", "id": st["id"]})
    made = {"upscaled": 2, "skipped": 0, "errors": []}
    m._on_event({"event": "done", "id": st["id"], "made": made})
    st = client.get(f"/api/datasets/{d['id']}/upscale").json()
    assert (st["state"], st["made"], st["max_factor"]) == ("done", made, 3.0)
    assert db.query_one("SELECT id FROM sandbox_runs WHERE id=?", (st["id"],)) is None  # not a run

    for bad in ({"resolution": 64}, {"max_factor": 1}, {"model": models["notes"]}):
        assert client.post(f"/api/datasets/{d['id']}/upscale", json={**body, **bad}).status_code == 400, bad
    big = tmp_path / "big"
    big.mkdir()
    _image(big / "a.png", (1024, 1024))
    r = client.post(f"/api/datasets/{_dataset(client, big, 'big')['id']}/upscale", json=body)
    assert r.status_code == 400 and "No image is smaller than 1024×1024" in r.json()["detail"]


def test_a_dataset_upscale_fails_when_the_worker_stops(client, sent, models, tmp_path, monkeypatch):
    folder = tmp_path / "few"
    folder.mkdir()
    _image(folder / "small.png", (200, 200))
    d = _dataset(client, folder, "few")
    st = client.post(f"/api/datasets/{d['id']}/upscale",
                     json={"model": models["RealESRGAN_x4plus"], "resolution": 512, "max_factor": 4}).json()

    class Ended(FakeProc):
        stdout = iter(())

        def wait(self):
            return 1
    proc = Ended()
    monkeypatch.setattr(sbx.sandbox, "proc", proc)
    sbx.sandbox._read_events(proc)
    st = client.get(f"/api/datasets/{d['id']}/upscale").json()
    assert st["state"] == "failed" and "stopped before this finished" in st["message"]

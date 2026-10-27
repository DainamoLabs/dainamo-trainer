"""Likeness routes: an image compared with a dataset (a Sandbox run's output, with the run's render without LoRAs as
its floor; a job's training sample, with its step-0 sample; a file), and a dataset's duplicates. The measuring
process is a stand-in: what the server would ask it is recorded.

    .venv/bin/python -m pytest tests/test_similarity_routes.py -v
"""

import json
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dainamo.server import db
from dainamo.server import sandbox as sbx
from dainamo.server.queue import job_dir
from dainamo.server.routes import similarity as routes


@pytest.fixture(scope="module")
def client():
    from dainamo.server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def engine(monkeypatch):
    """The measuring process, answering as if the image were 80% alike with one nearest image."""
    asked = []

    def run(request):
        asked.append(request)
        paths = request["concept"]["paths"]
        results = [{"sim": 0.8, "target": 1.0, "likeness": 80.0, "floor": 0.0 if f else None,
                    "base_sim": 0.0 if f else None, "nearest": {"path": paths[0], "sim": 0.8},
                    "top": [{"path": p, "sim": 0.8 - 0.1 * i} for i, p in enumerate(paths[:5])], "near_copy": False}
                   for f in (request.get("floors") or [None] * len(request.get("images") or []))]
        dups = [{"a": paths[0], "b": paths[1], "sim": 0.99}] if request.get("duplicates") is not None and len(paths) > 1 else []
        return {"count": len(paths), "self_sim": 0.9, "results": results, "duplicates": dups}

    monkeypatch.setattr(routes, "_run", run)
    return asked


@pytest.fixture
def dataset(client, tmp_path):
    folder = tmp_path / "cats"
    folder.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8), (255, 255, 0)).save(folder / f"cat_{i}.png")
    (folder / "cat_0.txt").write_text("a cat")
    (folder / "song.mp3").write_bytes(b"ID3")
    d = client.post("/api/datasets", json={"name": "cats", "path": str(folder)}).json()
    yield d
    client.delete(f"/api/datasets/{d['id']}")


def _run_with_outputs(outputs: list[dict]) -> str:
    rid = db.new_id()
    os.makedirs(sbx.run_dir(rid), exist_ok=True)
    for o in outputs:
        Image.new("RGB", (8, 8)).save(os.path.join(sbx.run_dir(rid), o["file"]))
    db.execute("INSERT INTO sandbox_runs (id, created, status, profile_id, profile_name, family, workflow, gpu, params, "
               "outputs) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (rid, db.now(), "done", "p", "P", "krea2", "t2i", 0, json.dumps({"prompt": "x", "compare_base": True}),
                json.dumps(outputs)))
    return rid


def test_a_runs_output_is_compared_with_its_render_without_loras_as_the_floor(client, engine, dataset):
    rid = _run_with_outputs([{"file": "0_lora.png", "kind": "image", "seed": 5, "variant": "lora"},
                             {"file": "0_base.png", "kind": "image", "seed": 5, "variant": "base"},
                             {"file": "1_lora.png", "kind": "image", "seed": 6, "variant": "lora"}])
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "source": {"run_id": rid, "file": "0_lora.png"}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["likeness"] == 80.0 and out["dataset"]["count"] == 3 and out["dataset"]["self_sim"] == 0.9
    assert out["floor"] == {"url": f"/api/sandbox/file/{rid}/0_base.png", "similarity": 0.0, "used": True}
    assert out["target"] == 1.0 and out["neighbours"] == 3
    assert out["image"]["kind"] == "run" and out["image"]["url"] == f"/api/sandbox/file/{rid}/0_lora.png"
    assert [n["name"] for n in out["nearest"]] == ["cat_0.png", "cat_1.png", "cat_2.png"]
    assert out["nearest"][0]["url"].startswith("/api/thumb?path=")
    req = engine[-1]
    assert req["images"] == [os.path.join(sbx.run_dir(rid), "0_lora.png")]
    assert req["floors"] == [os.path.join(sbx.run_dir(rid), "0_base.png")] and req["top"] == routes.TOP
    # the dataset's pictures only (not the song), and a cache file named after it
    assert all(p.endswith(".png") for p in req["concept"]["paths"]) and len(req["concept"]["paths"]) == 3
    assert req["concept"]["cache"] == os.path.join(routes.SIMILARITY_DIR, f"{dataset['id']}.pt")
    # a seed without a base render has no floor; the base render itself neither
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "source": {"run_id": rid, "file": "1_lora.png"}}).json()
    assert r["floor"] is None and engine[-1]["floors"] == [None]
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "source": {"run_id": rid, "file": "0_base.png"}}).json()
    assert r["floor"] is None
    sbx.sandbox.delete_run(rid)


def test_a_training_sample_is_compared_with_its_step_0_sample(client, engine, dataset):
    r = client.post("/api/jobs", json={"name": "cats", "profile_id": "none", "datasets": [], "settings": {}})
    jid = None
    if r.status_code == 200:
        jid = r.json()["id"]
    else:  # no profile: a job row by hand
        jid = db.new_id()
        db.execute("INSERT INTO jobs (id, name, status, family, profile_name, gpu, position, created, config, folder) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?)", (jid, "cats", "completed", "krea2", "P", "auto", 1, db.now(),
                                                   json.dumps({"settings": {}}), f"cats_{jid}"))
    samples_dir = os.path.join(job_dir(jid), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    samples = [{"step": 0, "index": 0, "prompt": "a cat", "seed": 1, "file": "step000000_0.png", "kind": "image"},
               {"step": 500, "index": 0, "prompt": "a cat", "seed": 1, "file": "step000500_0.png", "kind": "image"},
               {"step": 500, "index": 1, "prompt": "a dog", "seed": 2, "file": "step000500_1.png", "kind": "image"}]
    for s in samples:
        Image.new("RGB", (8, 8)).save(os.path.join(samples_dir, s["file"]))
    with open(os.path.join(job_dir(jid), "status.json"), "w") as f:
        json.dump({"samples": samples}, f)
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "sample": {"job_id": jid, "file": "step000500_0.png"}})
    assert r.status_code == 200, r.text
    assert r.json()["floor"]["url"] == f"/api/jobs/{jid}/file/samples/step000000_0.png"
    assert engine[-1]["floors"] == [os.path.join(samples_dir, "step000000_0.png")]
    # a card with no step-0 sample, and the step-0 sample itself: no floor
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "sample": {"job_id": jid, "file": "step000500_1.png"}}).json()
    assert r["floor"] is None
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "sample": {"job_id": jid, "file": "step000000_0.png"}}).json()
    assert r["floor"] is None and r["image"]["kind"] == "sample"
    assert client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "sample": {"job_id": jid, "file": "nope.png"}}).status_code == 404
    client.delete(f"/api/jobs/{jid}?delete_files=true")


def test_a_file_is_compared_too_and_the_request_is_checked(client, engine, dataset, tmp_path):
    f = tmp_path / "pic.jpg"
    Image.new("RGB", (8, 8)).save(f)
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "upload": str(f)})
    assert r.status_code == 200 and r.json()["image"] == {"kind": "file", "label": "pic.jpg", "url": None}
    assert r.json()["floor"] is None and r.json()["floor_default"] == 0.0
    # one image at a time, and only pictures
    assert client.post("/api/similarity/compare", json={"dataset_id": dataset["id"]}).status_code == 400
    assert client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "upload": str(f), "source": {"run_id": "x", "file": "y"}}).status_code == 400
    song = tmp_path / "song.mp3"
    song.write_bytes(b"ID3")
    assert "isn't an image" in client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "upload": str(song)}).json()["detail"]
    assert client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "upload": str(tmp_path / "gone.png")}).status_code == 400
    assert client.post("/api/similarity/compare", json={"dataset_id": "nope", "upload": str(f)}).status_code == 404
    # an uploaded file (POST /api/sandbox/upload) is shown from its upload URL
    up = client.post("/api/sandbox/upload", files={"file": ("mine.png", open(f, "rb"), "image/png")}).json()
    r = client.post("/api/similarity/compare", json={"dataset_id": dataset["id"], "upload": up["path"]}).json()
    assert r["image"]["url"] == up["url"]


def test_a_dataset_without_pictures_cant_be_compared_with(client, engine, tmp_path):
    folder = tmp_path / "songs"
    folder.mkdir()
    (folder / "a.mp3").write_bytes(b"ID3")
    d = client.post("/api/datasets", json={"name": "songs", "path": str(folder)}).json()
    pic = tmp_path / "pic.png"
    Image.new("RGB", (8, 8)).save(pic)
    r = client.post("/api/similarity/compare", json={"dataset_id": d["id"], "upload": str(pic)})
    assert r.status_code == 400 and "no images" in r.json()["detail"]
    client.delete(f"/api/datasets/{d['id']}")


def test_duplicates_are_listed_with_thumbnails(client, engine, dataset):
    r = client.get(f"/api/datasets/{dataset['id']}/duplicates?threshold=0.95")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["count"] == 3 and out["threshold"] == 0.95
    assert len(out["pairs"]) == 1 and out["pairs"][0]["a"]["name"] == "cat_0.png" and out["pairs"][0]["sim"] == 0.99
    assert out["pairs"][0]["b"]["url"].startswith("/api/thumb?path=")
    assert engine[-1]["duplicates"] == 0.95 and "images" not in engine[-1]
    assert client.get(f"/api/datasets/{dataset['id']}/duplicates?threshold=0.5").status_code == 400
    assert client.get("/api/datasets/nope/duplicates").status_code == 404


def test_the_measuring_process_is_run_with_the_request_in_a_file(monkeypatch, tmp_path):
    """``_run`` writes the request to a file, runs the engine on it, and reads the last JSON line it prints."""
    import subprocess

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        with open(cmd[-1]) as f:
            seen["request"] = json.load(f)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"progress": 1, "total": 2}\n{"count": 2, "self_sim": 0.9, "results": []}\n')

    monkeypatch.setattr(routes.subprocess, "run", fake_run)
    monkeypatch.setattr(routes.sbx, "free_gpu", lambda: None)
    monkeypatch.setattr(routes, "SIMILARITY_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "LOG", str(tmp_path / "log"))
    out = routes._run({"concept": {"paths": ["a"]}, "images": []})
    assert out["count"] == 2 and seen["request"]["images"] == [] and seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert seen["cmd"][1:3] == ["-m", "dainamo.engine.similarity"] and not os.path.exists(seen["cmd"][-1])
    # a failure is a 500 with the engine's message
    monkeypatch.setattr(routes.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout='{"error": "RuntimeError: no"}\n'))
    with pytest.raises(Exception) as e:
        routes._run({"concept": {"paths": []}})
    assert e.value.status_code == 500 and "RuntimeError: no" in e.value.detail

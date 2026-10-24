"""The web UI: when the launcher rebuilds it, the conventions its source settled on, and the API it reads for them."""

import os
import re
from types import SimpleNamespace

import pytest

from dainamo import webui


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_rebuilds_only_when_the_source_changes(tmp_path):
    web = str(tmp_path)
    _write(os.path.join(web, "src", "pages", "App.tsx"), "one")
    _write(os.path.join(web, "package.json"), "{}")
    assert not webui.is_current(web)  # never built

    _write(os.path.join(web, "dist", "index.html"), "<html>")
    assert not webui.is_current(web)  # built before the launcher kept a record (an older checkout)
    webui.mark_built(web)
    assert webui.is_current(web)

    os.utime(os.path.join(web, "src", "pages", "App.tsx"))  # a checkout touches files without changing them
    _write(os.path.join(web, "node_modules", "x.js"), "not source")
    assert webui.is_current(web)

    _write(os.path.join(web, "src", "pages", "App.tsx"), "two")  # a pull changed it
    assert not webui.is_current(web)
    webui.mark_built(web)
    _write(os.path.join(web, "src", "pages", "New.tsx"), "")  # or added a file
    assert not webui.is_current(web)
    webui.mark_built(web)
    os.remove(os.path.join(web, "dist", "index.html"))  # the build is gone
    assert not webui.is_current(web)


# ---- the web UI's source: the conventions the cleanup settled on stay settled -------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "web", "src")


def _sources(exts=(".ts", ".tsx")):
    for folder, _, files in os.walk(SRC):
        for name in files:
            if name.endswith(exts):
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as f:
                    yield os.path.relpath(path, SRC), f.read()


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_no_browser_dialogs():
    # a destructive action asks with ConfirmDialog (red, its consequence said), never the browser's confirm()
    for name, text in _sources():
        assert not re.search(r"(?<![\w.])(confirm|alert|prompt)\(|window\.(confirm|alert|prompt)\(", text), name


def test_no_em_dashes_in_ui_copy():
    # a lone "—" stands for no value in a table; anywhere else it's prose, which uses other punctuation
    for name, text in _sources((".ts", ".tsx", ".css")):
        prose = re.sub(r"\"—\"|>—<", "", text)
        assert "—" not in prose, name


def test_settled_names():
    text = "\n".join(t for n, t in _sources() if not n.endswith("ModelFiles.tsx"))
    for old in ("Best checkpoint picker", "Concept bleed detector", "concept bleed detector", "Save events", "Keep latest",
                "Queue merge", "Queue extraction", "Queue evaluation", "Cancel making", "<h1>Create job</h1>",
                "Profiles page", "Choose LoRA…", "Training jobs", "No training jobs yet"):
        assert old not in text, old
    # the nav tab and page are Models; the route stays #/profiles
    top = _read("web", "src", "components", "TopBar.tsx")
    assert '{ id: "profiles", label: "Models"' in top
    assert "<h1>Models</h1>" in _read("web", "src", "pages", "Profiles.tsx")
    assert "<h1>New job</h1>" in _read("web", "src", "pages", "CreateJob.tsx")


def test_two_page_widths():
    css = _read("web", "src", "styles.css")
    widths = re.findall(r"^\.page(\.[\w-]+)? \{[^}]*max-width", css, re.M)
    assert sorted(widths) == ["", ".wide"]
    for name, text in _sources():
        for cls in re.findall(r'className="(page(?: [\w-]+)*)"', text):
            assert not set(cls.split()) & {"narrow", "dataset", "profiles"}, (name, cls)


def test_docs_name_the_models_page():
    for name in sorted(os.listdir(os.path.join(ROOT, "docs"))) + ["README.md"]:
        if not name.endswith(".md") or name == "development.md":
            continue
        text = _read(name) if name == "README.md" else _read("docs", name)
        assert "Profiles page" not in text and "Create Job" not in text, name


# ---- the server side of it: dataset tasks for the top bar, and whether a job can resume ---------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from dainamo.server.app import app

    with TestClient(app) as c:
        yield c


def _job(state: dict):
    return SimpleNamespace(to_json=lambda: state)


def test_background_tasks_for_the_top_bar(client, tmp_path, monkeypatch):
    from dainamo.server import captioner, clip_split, media_download
    from dainamo.server import sandbox as sbx

    did = client.post("/api/datasets", json={"name": "tasks", "path": str(tmp_path)}).json()["id"]
    assert client.get("/api/tasks").json() == []

    monkeypatch.setitem(captioner.caption_jobs, did, _job({"running": True, "phase": "captioning", "done": 3, "total": 10,
                                                           "heard": 0, "to_hear": 0}))
    monkeypatch.setitem(clip_split.split_jobs, did, _job({"running": True, "mode": "plan", "done": 1, "total": 4}))
    # a single video's download isn't counted; a playlist's is, by item
    monkeypatch.setitem(media_download.download_jobs, did, _job({"running": True, "item": 1, "items": 1}))
    monkeypatch.setattr(sbx.sandbox, "upscale", {"state": "queued", "dataset_id": did, "done": 0, "total": 12})
    tasks = client.get("/api/tasks").json()
    assert tasks == [
        {"kind": "caption", "label": "Captioning", "dataset_id": did, "dataset": "tasks", "done": 3, "total": 10},
        {"kind": "split", "label": "Planning clips", "dataset_id": did, "dataset": "tasks", "done": 1, "total": 4},
        {"kind": "download", "label": "Downloading", "dataset_id": did, "dataset": "tasks", "done": 0, "total": 0},
        {"kind": "upscale", "label": "Upscale waiting", "dataset_id": did, "dataset": "tasks", "done": 0, "total": 0},
    ]

    # what's over isn't shown: the dataset's page says how it went
    monkeypatch.setitem(captioner.caption_jobs, did, _job({"running": False, "phase": "done", "done": 10, "total": 10}))
    monkeypatch.setitem(clip_split.split_jobs, did, _job({"running": False, "mode": "cut", "done": 4, "total": 4}))
    monkeypatch.setitem(media_download.download_jobs, did, _job({"running": True, "item": 2, "items": 5}))
    monkeypatch.setattr(sbx.sandbox, "upscale", {"state": "running", "dataset_id": did, "done": 5, "total": 12})
    assert [(t["kind"], t["label"], t["done"], t["total"]) for t in client.get("/api/tasks").json()] == \
        [("download", "Downloading", 2, 5), ("upscale", "Upscaling", 5, 12)]


def test_a_job_is_resumable_only_with_a_saved_state(client, tmp_path):
    from dainamo.server.queue import job_dir

    client.put("/api/settings", json={"queue": {"gpus": [99]}})  # a GPU that isn't there: nothing starts
    pid = client.post("/api/profiles", json={"name": "rs", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "rs", "path": str(tmp_path)}).json()["id"]
    jid = client.post("/api/jobs", json={"name": "rs", "profile_id": pid, "datasets": [{"id": did}],
                                         "settings": {}}).json()["id"]
    summary = lambda: next(j for j in client.get("/api/jobs").json() if j["id"] == jid)  # noqa: E731
    assert summary()["resumable"] is False  # never saved: only Restart
    state = os.path.join(job_dir(jid), "state")
    os.makedirs(state, exist_ok=True)
    open(os.path.join(state, "last.pt"), "wb").close()
    assert summary()["resumable"] is True
    client.post(f"/api/jobs/{jid}/action", json={"action": "stop"})

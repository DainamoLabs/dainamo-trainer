"""Add from URL (dainamo/server/media_download.py): the yt-dlp command, a download moved into a dataset, errors,
cancelling, the API's checks, updating yt-dlp and the launch warning. A stand-in yt-dlp (``FAKE``) does the
"downloading", so nothing goes to the network.

    .venv/bin/python -m pytest tests/test_media_download.py -v
"""

import os
import subprocess
import sys
import time

import pytest

from dainamo.server import media_download as md

# answers like yt-dlp with the templates in md.TEMPLATES; what it does depends on the URL's path
FAKE = r'''
import os, sys, time
args = sys.argv[1:]
if args == ["--version"]:
    print("2099.01.01"); sys.exit(0)
out, url = args[args.index("-P") + 1], args[args.index("--") + 1]
def item(i, n, name, title):
    print(f"ITEM|{i}|{n}|{title}", flush=True)
    print(f"DL|{i}|{n}|512|2048|{title}", flush=True)
    print(f"DL|{i}|{n}|2048|2048|{title}", flush=True)
    print(f"PP|FFmpegMerger|{title}", flush=True)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, name + ".part"), "w").close()  # left behind, as an interrupted download leaves it
    path = os.path.join(out, name)
    open(path, "wb").write(b"media")
    print(f"FILE|{path}", flush=True)
if url.endswith("/ok"):
    item("NA", "NA", "Clip [abc].mp4", "Clip")
elif url.endswith("/list"):
    item(1, 2, "First [a1].mp4", "First")
    print("ERROR: [generic] b2: Video unavailable", file=sys.stderr)
    sys.exit(1)
elif url.endswith("/page"):
    item("NA", "NA", "page [p].html", "page")
elif url.endswith("/slow"):
    print("ITEM|NA|NA|Slow", flush=True)
    time.sleep(60)
else:
    print("ERROR: Unsupported URL: " + url, file=sys.stderr)
    sys.exit(1)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_ytdlp.py"
    script.write_text(FAKE)
    monkeypatch.setattr(md, "ytdlp", lambda: [sys.executable, str(script)])
    md._version.clear()
    yield
    md._version.clear()


def run(dataset, url, **opts):
    j = md.DownloadJob(dataset, url, {"kind": "video", "max_height": 1080, **opts})
    end = time.time() + 30
    while j.running and time.time() < end:
        time.sleep(0.05)
    assert not j.running
    return j


def test_the_command_asks_for_what_the_form_chose(monkeypatch):
    monkeypatch.setattr(md, "ytdlp", lambda: ["yt-dlp"])
    monkeypatch.setattr(md.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    video = md.command("https://e.com/v", "/out", "video", 720, 20, 24.5)
    assert video[-2:] == ["--", "https://e.com/v"]  # a URL can never be read as an option
    s = " ".join(video)
    assert "-S res:720,vcodec:h264,ext" in s and "--merge-output-format mp4" in s
    assert "--download-sections *20-24.5 --force-keyframes-at-cuts" in s
    assert "--no-playlist -I 1:1" in s  # a playlist's own URL gives just its first item
    assert "--ignore-config" in s and "--js-runtimes node" in s and "--js-runtimes bun" not in s
    assert "-S res,vcodec:h264,ext" in " ".join(md.command("https://e.com/v", "/out", "video", 0))
    sound = " ".join(md.command("https://e.com/v", "/out", "audio", 1080, None, 90))
    assert "-x --audio-format best" in sound and "-S" not in sound
    assert "--download-sections *0-90" in sound  # sound is cut exactly too
    whole = " ".join(md.command("https://e.com/l", "/out", "audio", playlist=5))
    assert "--yes-playlist --ignore-errors -I 1:5" in whole and "--download-sections" not in whole


def test_new_files_never_share_a_stem_and_so_a_caption(tmp_path):
    for f in ("Clip [abc].mp4", "Clip [abc] 2.txt", "Other.png"):
        (tmp_path / f).write_text("x")
    assert md.free_name(str(tmp_path), "New.mp4") == "New.mp4"
    assert md.free_name(str(tmp_path), "clip [ABC].opus") == "clip [ABC] 3.opus"  # case-blind, as Windows is
    assert md.free_name(str(tmp_path / "missing"), "a.mp4") == "a.mp4"


def test_a_download_lands_in_the_dataset_with_nothing_half_done(fake, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "Clip [abc].mp4").write_text("already here")
    j = run({"id": "d", "path": str(ds)}, "https://e.com/ok")
    assert j.errors == [] and j.saved == ["Clip [abc] 2.mp4"]
    assert (ds / "Clip [abc] 2.mp4").read_bytes() == b"media"
    assert sorted(os.listdir(ds)) == ["Clip [abc] 2.mp4", "Clip [abc].mp4"]  # no .part
    assert not os.path.exists(j.folder)
    s = j.to_json()
    assert (s["title"], s["item"], s["items"], s["count"]) == ("Clip", 1, 1, 1)


def test_a_playlist_keeps_what_worked_and_reports_what_didnt(fake, tmp_path):
    j = run({"id": "d", "path": str(tmp_path)}, "https://e.com/list", playlist=2)
    assert j.saved == ["First [a1].mp4"] and j.errors == ["[generic] b2: Video unavailable"]
    assert (j.item, j.items) == (1, 2)
    j = run({"id": "d", "path": str(tmp_path)}, "https://e.com/page")
    assert j.saved == [] and "not a kind of file datasets take" in j.errors[0]
    j = run({"id": "d", "path": str(tmp_path)}, "https://e.com/nothing")
    assert j.errors == ["Unsupported URL: https://e.com/nothing"]


def test_cancelling_stops_yt_dlp_and_cleans_up(fake, tmp_path):
    j = md.DownloadJob({"id": "d", "path": str(tmp_path)}, "https://e.com/slow", {})
    end = time.time() + 15
    while j.phase != "downloading" and time.time() < end:
        time.sleep(0.05)
    j.cancel()
    end = time.time() + 15
    while j.running and time.time() < end:
        time.sleep(0.05)
    assert not j.running and j.cancelled and j.errors == []
    assert j.proc.poll() is not None and not os.path.exists(j.folder)


def test_the_api_checks_the_request(fake, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dainamo.server.app import app

    with TestClient(app) as client:
        did = client.post("/api/datasets", json={"name": "web", "path": str(tmp_path)}).json()["id"]
        url = f"/api/datasets/{did}/download"
        for body, words in (({"url": "file:///etc/passwd"}, "http"), ({"url": "-o x"}, "http"),
                            ({"url": "https://e.com/ok", "kind": "pdf"}, "Unknown kind"),
                            ({"url": "https://e.com/ok", "max_height": 999}, "largest size"),
                            ({"url": "https://e.com/ok", "start": 30, "end": 10}, "end after"),
                            ({"url": "https://e.com/ok", "playlist": 3, "end": 10}, "whole playlist"),
                            ({"url": "https://e.com/ok", "playlist": 500}, "1–200")):
            r = client.post(url, json=body)
            assert r.status_code == 400 and words in r.json()["detail"], (body, r.json())
        assert client.post(url, json={"url": "https://e.com/ok", "start": 5}).status_code == 200
        end = time.time() + 30
        while client.get(url).json()["running"] and time.time() < end:
            time.sleep(0.05)
        assert client.get(url).json()["saved"] == ["Clip [abc].mp4"]
        names = [i["name"] for i in client.get(f"/api/datasets/{did}/items").json()]
        assert names == ["Clip [abc].mp4"]
        info = client.get("/api/ytdlp").json()
        assert info == {"installed": True, "version": "2099.01.01", "bundled": False, "network": False}
        monkeypatch.setattr(md, "ytdlp", lambda: None)
        r = client.post(url, json={"url": "https://e.com/ok"})
        assert r.status_code == 400 and "isn't installed" in r.json()["detail"]


def test_updating_upgrades_only_yt_dlp(fake, monkeypatch):
    calls = []

    def run_(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="2099.02.02\n" if "--version" in cmd else "", stderr="")
    monkeypatch.setattr(md.subprocess, "run", run_)
    monkeypatch.setattr(md.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    assert md.update()["version"] == "2099.02.02"
    assert calls[0] == ["/bin/uv", "pip", "install", "--python", sys.executable, "--upgrade-package", "yt-dlp",
                        "yt-dlp[default]"]  # not --upgrade, which would upgrade every package it touches

    def fail(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error: no network\n")
    monkeypatch.setattr(md.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="no network"):
        md.update()


def test_listening_on_the_network_warns_about_downloads(monkeypatch, capsys):
    import dainamo.__main__ as main

    monkeypatch.setenv("DAINAMO_LISTEN", "127.0.0.1")
    monkeypatch.setattr(main.uvicorn, "run", lambda *a, **k: None)
    for host, warned in (("127.0.0.1", False), ("0.0.0.0", True)):
        monkeypatch.setattr(sys, "argv", ["dainamo", "--host", host, "--no-browser", "--no-auth"])
        main.main()
        assert ("Add from URL (experimental) works from other machines" in capsys.readouterr().out) == warned
        assert md.on_network() == warned

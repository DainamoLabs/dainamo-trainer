"""The clip editor: a split's plan reviewed before it's cut, changed, and cut again (only what changed), splits
from before the editor read as plans, and the frames it shows being the clips' own.

    .venv/bin/python -m pytest tests/test_clip_editor.py -v
"""

import io
import json
import os
import subprocess
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dainamo.engine import dataset as ds
from dainamo.server import clip_split
from dainamo.server.app import app
from test_clip_split import frame_count, shots_video


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait(client, tid, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        st = client.get(f"/api/datasets/{tid}/clips").json()
        if not (st["job"] or {}).get("running"):
            return st
        time.sleep(0.1)
    raise TimeoutError


def source(client, tmp_path, name="src"):
    src = tmp_path / name
    src.mkdir()
    shots_video(src / "long.mp4")  # shots of 4, 7 and 2.5 s, 320x180
    (src / "long.txt").write_text("three shots")
    return str(src), client.post("/api/datasets", json={"name": name, "path": str(src)}).json()["id"]


def video_of(st, rel="long.mp4"):
    return next(v for v in st["videos"] if v["rel"] == rel)


def put(client, tid, clips, rel="long.mp4"):
    return client.put(f"/api/datasets/{tid}/clips", json={"rel": rel, "clips": clips})


def files(root):
    return sorted(f for f in os.listdir(root) if f.endswith(".mp4"))


def test_reviewed_clips_are_cut_as_planned_and_again_only_where_they_changed(client, tmp_path):
    _, did = source(client, tmp_path)
    r = client.post(f"/api/datasets/{did}/split", json={"name": "long clips", "family": "minimax_h3", "frames": 73,
                                                         "max_clips": 0, "max_side": 0, "review": True})
    assert r.status_code == 200, r.text
    tid = r.json()["target"]["id"]
    st = wait(client, tid)
    root = next(d for d in client.get("/api/datasets").json() if d["id"] == tid)["path"]
    assert files(root) == []  # planned, not cut
    v = video_of(st)
    assert v["looked"] and v["width"] == 320 and v["duration"] == pytest.approx(13.5, abs=0.05)
    assert [round(c) for c in v["cuts"]] == [4, 11]
    assert [(c["id"], c["frames"]) for c in v["clips"]] == [(1, 73), (2, 73), (3, 73), (4, 56)] and v["cut"] == {}
    assert st["changes"] == {"cut": 4, "remove": 0} and 73 in st["lengths"] and st["fps"] == 24
    listed = next(d for d in client.get("/api/datasets").json() if d["id"] == tid)
    assert listed["split"] == {"source": {"id": did, "name": "src"}, "changes": 4}

    # move 1, shorten 2, drop 3, keep 4, add 5 with a crop
    c1, c2, _, c4 = v["clips"]
    clips = [{**c1, "start": 0.5}, {**c2, "frames": 56}, c4,
             {"id": 5, "start": 1.0, "frames": 22, "crop": [40, 20, 160, 120]}]
    r = put(client, tid, clips)
    assert r.status_code == 200, r.text
    assert r.json()["changes"] == {"cut": 4, "remove": 0}
    st = client.post(f"/api/datasets/{tid}/clips/cut")
    assert st.status_code == 200, st.text
    st = wait(client, tid)
    assert st["job"]["errors"] == [] and st["changes"] == {"cut": 0, "remove": 0}
    assert files(root) == ["long_001.mp4", "long_002.mp4", "long_004.mp4", "long_005.mp4"]
    assert frame_count(os.path.join(root, "long_002.mp4")) == 56
    info = ds.probe_video(os.path.join(root, "long_005.mp4"))
    assert (info["width"], info["height"]) == (160, 120) and frame_count(os.path.join(root, "long_005.mp4")) == 22
    assert open(os.path.join(root, "long_005.txt")).read() == "three shots"  # a new clip gets its video's caption
    assert set(video_of(st)["cut"]) == {"1", "2", "4", "5"}
    assert set(ds.clip_sources(root).values()) == {"long.mp4"}  # trained as one video

    # a caption written for clip 1 stays when it's moved; clip 2 goes with its caption; 4 isn't cut again
    with open(os.path.join(root, "long_001.txt"), "w") as f:
        f.write("my own words")
    kept = os.path.getmtime(os.path.join(root, "long_004.mp4"))
    c = {x["id"]: x for x in video_of(st)["clips"]}
    assert put(client, tid, [{**c[1], "start": 0.25}, c[4], c[5]]).status_code == 200
    client.post(f"/api/datasets/{tid}/clips/cut")
    st = wait(client, tid)
    assert st["job"]["clips"] == 1 and st["job"]["removed"] == 1
    assert files(root) == ["long_001.mp4", "long_004.mp4", "long_005.mp4"]
    assert not os.path.exists(os.path.join(root, "long_002.txt"))
    assert open(os.path.join(root, "long_001.txt")).read() == "my own words"
    assert os.path.getmtime(os.path.join(root, "long_004.mp4")) == kept

    # a clip deleted from the dataset leaves the plan, so cutting again doesn't bring it back
    client.delete("/api/media", params={"path": os.path.join(root, "long_004.mp4")})
    st = client.get(f"/api/datasets/{tid}/clips").json()
    assert [c["id"] for c in video_of(st)["clips"]] == [1, 5] and st["changes"] == {"cut": 0, "remove": 0}


def test_a_split_from_before_the_editor_reads_as_its_plan(client, tmp_path):
    src, did = source(client, tmp_path)
    root = tmp_path / "old"
    root.mkdir()
    for i, start in ((1, 0.5), (2, 5.0)):  # as a version 1 split cut them: the size kept
        clip_split.cut_clip(os.path.join(src, "long.mp4"), str(root / f"long_{i:03d}.mp4"), start, 22, 24, 0, True)
    (root / ds.CLIPS_MANIFEST).write_text(json.dumps({
        "version": 1, "source": {"id": did, "name": "src"}, "family": "minimax_h3", "fps": 24, "frames": 22,
        "clips": {"long_001.mp4": {"source": "long.mp4", "start": 0.5, "frames": 22},
                  "long_002.mp4": {"source": "long.mp4", "start": 5.0, "frames": 22}}}))
    tid = client.post("/api/datasets", json={"name": "old"}).json()["id"]
    target = next(d for d in client.get("/api/datasets").json() if d["id"] == tid)["path"]
    os.rmdir(target)
    os.rename(root, target)
    st = client.get(f"/api/datasets/{tid}/clips").json()
    v = video_of(st)
    assert not v["looked"] and [c["id"] for c in v["clips"]] == [1, 2] and st["settings"]["max_side"] == 0
    client.post(f"/api/datasets/{tid}/clips/look")
    st = wait(client, tid)
    v = video_of(st)
    assert v["looked"] and len(v["cuts"]) == 2 and set(v["cut"]) == {"1", "2"} and st["changes"] == {"cut": 0, "remove": 0}
    # a clip added now numbers on, and comes out the size the old ones are
    assert put(client, tid, [*v["clips"], {"id": 3, "start": 8.0, "frames": 22}]).status_code == 200
    client.post(f"/api/datasets/{tid}/clips/cut")
    st = wait(client, tid)
    assert st["job"]["clips"] == 1 and files(target) == ["long_001.mp4", "long_002.mp4", "long_003.mp4"]
    assert ds.probe_video(os.path.join(target, "long_003.mp4"))["height"] == 180


def test_replanning_a_video_numbers_its_clips_on(client, tmp_path):
    _, did = source(client, tmp_path)
    tid = client.post(f"/api/datasets/{did}/split", json={"name": "r", "family": "minimax_h3", "frames": 73,
                                                           "max_clips": 0, "review": True}).json()["target"]["id"]
    wait(client, tid)
    r = client.post(f"/api/datasets/{tid}/clips/replan", json={"rel": "long.mp4", "frames": 141})
    assert r.status_code == 200, r.text
    clips = r.json()["video"]["clips"]
    assert [c["frames"] for c in clips] == [90, 141, 56] and [c["id"] for c in clips] == [5, 6, 7]


def test_clip_edits_are_checked(client, tmp_path):
    _, did = source(client, tmp_path)
    tid = client.post(f"/api/datasets/{did}/split", json={"name": "c", "family": "minimax_h3", "frames": 73,
                                                           "review": True}).json()["target"]["id"]
    wait(client, tid)
    ok = {"id": 1, "start": 0.0, "frames": 22}
    for clips, says in (([{**ok, "frames": 23}], "17n + 5"), ([{**ok, "start": 13.0}], "past the end"),
                        ([ok, ok], "number of its own"), ([{**ok, "crop": [200, 0, 160, 120]}], "inside the frame"),
                        ([{**ok, "crop": [0, 0, 32, 120]}], "64 px")):
        r = put(client, tid, clips)
        assert r.status_code == 400 and says in r.json()["detail"], (clips, r.text)
    assert put(client, tid, [ok], rel="nope.mp4").json()["detail"] == "This video hasn't been looked at yet"
    whole = put(client, tid, [{**ok, "crop": [0, 0, 320, 180]}]).json()
    assert whole["video"]["clips"][0]["crop"] is None  # the whole frame is no crop
    assert "made by Split into clips" in client.get(f"/api/datasets/{did}/clips").json()["detail"]


def bar_video(path, seconds=4, fps=30):
    """Frame N has a bar at x = 2 * (N % 64): each frame of a clip from it is told apart from the next."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"nullsrc=s=128x32:r={fps}:d={seconds}", "-vf",
                    "geq=lum='if(eq(floor(X/2),mod(N,64)),235,16)':cb=128:cr=128", "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def gray_frames(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=128:32", "-f", "rawvideo", "-pix_fmt",
                          "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 32, 128).astype(np.float32)


def test_the_frames_shown_are_the_clips_own(client, tmp_path):
    _, did = source(client, tmp_path)
    video = bar_video(tmp_path / "src" / "bars.mp4")
    out = str(tmp_path / "clip.mp4")
    clip_split.cut_clip(video, out, 1.234, 22, 24, 0, False)
    frames = gray_frames(out)
    for n in (0, 9, 21):
        r = client.get("/api/clips/frame", params={"path": video, "start": 1.234, "n": n, "fps": 24, "size": 128})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        (tmp_path / "shown.jpg").write_bytes(r.content)
        shown = gray_frames(str(tmp_path / "shown.jpg"))[0]
        err = [np.abs(shown - f).mean() for f in frames]
        assert int(np.argmin(err)) == n and err[n] < 0.5 * min(e for i, e in enumerate(err) if i != n), err
    strip = client.get("/api/clips/strip", params={"path": video, "count": 8, "height": 32})
    assert strip.status_code == 200 and Image.open(io.BytesIO(strip.content)).size == (128 * 8, 32)
    assert client.get("/api/clips/frame", params={"path": "/etc/passwd", "start": 0}).status_code == 403

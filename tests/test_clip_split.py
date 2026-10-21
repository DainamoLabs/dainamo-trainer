"""Split into clips: planning clips inside shots, cutting them exactly, and training each video once per epoch.

    .venv/bin/python -m pytest tests/test_clip_split.py -v
"""

import json
import os
import random
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from dainamo.engine import dataset as ds
from dainamo.engine import evaluation
from dainamo.engine.families import get_family
from dainamo.server import clip_split
from dainamo.server.app import app

H3 = get_family("minimax_h3")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def shots_video(path, shots=((4, "testsrc2"), (7, "smptebars"), (2.5, "rgbtestsrc")), fps=30, audio=True):
    """A video of hard cuts between lavfi sources, ``shots`` of (seconds, source)."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for seconds, src in shots:
        cmd += ["-f", "lavfi", "-i", f"{src}=size=320x180:rate={fps}:duration={seconds}"]
    total = sum(s for s, _ in shots)
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={total}"]
    joined = "".join(f"[{i}:v]" for i in range(len(shots))) + f"concat=n={len(shots)}:v=1:a=0[v]"
    cmd += ["-filter_complex", joined, "-map", "[v]"] + (["-map", f"{len(shots)}:a", "-c:a", "aac"] if audio else [])
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True)
    return int(out.stdout.strip())


def test_clips_stay_inside_shots_and_spread_out():
    cuts, fps = [4.0, 11.0], 24
    plan = clip_split.plan_clips(13.5, cuts, fps, 73, H3.snap_frames, cap=0)
    assert [f for _, f in plan] == [73, 73, 73, 56]  # the 2.5 s shot gives a shorter clip, snapped to 17n + 5
    edges = [0.0] + cuts + [13.5]
    for start, frames in plan:
        shot = next(i for i in range(3) if edges[i] <= start < edges[i + 1])
        # a frame clear of each cut, and of the file's end
        assert start + frames / fps <= edges[shot + 1] - 1 / fps + 1e-3
        assert shot == 0 or start >= edges[shot] + 1 / fps - 1e-3
    # the 7 s shot holds two clips, with equal gaps around them
    (a, _), (b, _) = plan[1], plan[2]
    gap = a - (4 + 1 / fps)
    assert b - (a + 73 / fps) == pytest.approx(gap, abs=2e-3)
    assert (11 - 1 / fps) - (b + 73 / fps) == pytest.approx(gap, abs=2e-3)
    assert len(clip_split.plan_clips(13.5, cuts, fps, 73, H3.snap_frames, keep_short=False)) == 3
    # no shot cuts: back to back over the whole video
    assert len(clip_split.plan_clips(30, [], fps, 73)) == 9


def test_a_cap_keeps_clips_spread_over_the_video():
    plan = clip_split.plan_clips(120, [], 24, 39, cap=4)
    starts = [s for s, _ in plan]
    assert len(plan) == 4 and starts[0] < 2 and starts[-1] > 115  # first and last of the 73 that fit
    assert clip_split.plan_clips(120, [], 24, 39, cap=1)[0][0] == pytest.approx(60, abs=2)  # the middle one
    # full-length clips come first; the longest short shots fill what's left
    plan = clip_split.plan_clips(10, [3.0, 4.5, 6.5], 24, 72, cap=3)
    # shots of 3, 1.5, 2 and 3.5 s: one clip fits the last; then the 3 s shot (71 frames clear of the cut) and the 2 s one
    assert sorted(f for _, f in plan) == [46, 71, 72]


def test_clip_lengths_follow_the_frame_rule():
    lengths = clip_split.clip_lengths(H3)
    assert lengths[:3] == [5, 22, 39] and all((n - 5) % 17 == 0 for n in lengths) and lengths[-1] <= 241
    assert clip_split.clip_lengths(get_family("ltx2_5"))[:3] == [9, 17, 25]  # 8n + 1 (the 1 isn't a clip)


def test_split_cuts_exact_clips_with_sound_and_copies_the_rest(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "_refs").mkdir(parents=True)
    out.mkdir()
    shots_video(src / "long.mp4")
    (src / "long.txt").write_text("three shots")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:size=64x64", "-frames:v", "1",
                    str(src / "_refs" / "long.png")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:size=64x64", "-frames:v", "1",
                    str(src / "long_001.png")], check=True)  # an image with the name a clip would take
    (src / "long_001.txt").write_text("a blue square")
    job = clip_split.SplitJob({"id": "s", "name": "src", "path": str(src)}, {"id": "t", "name": "out", "path": str(out)},
                              H3, {"frames": 73, "max_clips": 8, "scenes": True, "keep_short": True, "max_side": 128,
                                   "captions": "copy"})
    end = time.time() + 120
    while job.running and time.time() < end:
        time.sleep(0.1)
    assert not job.running and job.errors == [] and job.clips == 4
    clips = sorted(f for f in os.listdir(out) if f.endswith(".mp4"))
    assert clips == [f"long_mp4_{i:03d}.mp4" for i in range(1, 5)]  # renamed away from the image
    assert (out / "long_001.png").exists() and (out / "long_001.txt").read_text() == "a blue square"
    manifest = json.loads((out / ds.CLIPS_MANIFEST).read_text())
    for name, want in zip(clips, [73, 73, 73, 56]):
        path = str(out / name)
        info = ds.probe_video(path)
        assert frame_count(path) == want and manifest["clips"][name]["frames"] == want
        assert int(info["duration"] * 24 + 1e-3) >= want  # what the trainer reads it as
        assert info["has_audio"] and info["height"] == 128 and info["fps"] == 24
        assert (out / name.replace(".mp4", ".txt")).read_text() == "three shots"
        assert (out / "_refs" / name.replace(".mp4", ".png")).exists()
        assert manifest["clips"][name]["source"] == "long.mp4"
    assert set(ds.clip_sources(str(out)).values()) == {"long.mp4"}


def test_cancelling_removes_the_copy(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    shots_video(src / "a.mp4", shots=((20, "testsrc2"),))
    did = client.post("/api/datasets", json={"name": "a", "path": str(src)}).json()["id"]
    r = client.post(f"/api/datasets/{did}/split", json={"name": "a clips", "family": "minimax_h3", "frames": 22,
                                                         "max_clips": 0, "scenes": False})
    assert r.status_code == 200, r.text
    target = r.json()["target"]
    client.delete(f"/api/datasets/{did}/split")
    end = time.time() + 60
    while client.get(f"/api/datasets/{did}/split").json()["running"] and time.time() < end:
        time.sleep(0.05)
    assert client.get(f"/api/datasets/{did}/split").json()["cancelled"]
    assert target["id"] not in {d["id"] for d in client.get("/api/datasets").json()}


def test_split_requests_are_checked(client, tmp_path):
    (tmp_path / "img").mkdir()
    did = client.post("/api/datasets", json={"name": "v", "path": str(tmp_path)}).json()["id"]
    body = {"name": "x", "family": "minimax_h3", "frames": 73}
    assert "no videos" in client.post(f"/api/datasets/{did}/split", json=body).json()["detail"]
    shots_video(tmp_path / "v.mp4", shots=((1, "testsrc2"),), audio=False)
    for bad, says in (({"frames": 70}, "17n + 5"), ({"family": "krea2"}, "doesn't train on clips"),
                      ({"captions": "rewrite"}, "caption choice"), ({"max_side": 10}, "64–8192"), ({"name": " "}, "Name")):
        r = client.post(f"/api/datasets/{did}/split", json={**body, **bad})
        assert r.status_code == 400 and says in r.json()["detail"], (bad, r.text)
    pid = client.post("/api/profiles", json={"name": "h", "family": "minimax_h3",
                                             "data": {"diffusion_model": "/nope.safetensors"}}).json()["id"]
    r = client.post(f"/api/datasets/{did}/split/fit", json={"profile_id": pid, "resolution": 384, "gpu_gb": 24})
    assert "wasn't found" in r.json()["error"]


def write_split(root, clips: dict[str, str], images=()):
    """A split dataset: clip name -> its video, as tiny real clips, plus images."""
    os.makedirs(root, exist_ok=True)
    for name in clips:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=1",
                        "-pix_fmt", "yuv420p", os.path.join(root, name)], check=True)
    for name in images:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:size=64x64", "-frames:v", "1",
                        os.path.join(root, name)], check=True)
    with open(os.path.join(root, ds.CLIPS_MANIFEST), "w") as f:
        json.dump({"clips": {n: {"source": v, "start": 0, "frames": 24} for n, v in clips.items()}}, f)


def test_each_video_trains_once_per_epoch(tmp_path):
    root = str(tmp_path / "split")
    write_split(root, {"a_001.mp4": "a.mp4", "a_002.mp4": "a.mp4", "a_003.mp4": "a.mp4", "b_001.mp4": "b.mp4"},
                images=["still.png"])
    log = []
    items = ds.build_items([{"path": root, "name": "split", "repeats": 2, "flip": True}], 64, 16, False, log.append,
                           video_frames=22, fps=24)
    assert len(items) == 5 * 2 * 2  # files x flip x repeats
    assert "4 clips (cut from 2 videos, each training once per epoch)" in log[0] and "12 per epoch" in log[0]
    assert {it.source for it in items if it.is_video} == {f"{root}|a.mp4", f"{root}|b.mp4"}
    rng = random.Random(0)
    seen = set()
    for _ in range(30):
        picks = ds.epoch_items(items, rng)
        assert len(picks) == 2 * 2 + 2 * 2 * 2  # the image's 4 items, and one per video, flip and repeat
        assert sum(items[i].path.endswith("still.png") for i in picks) == 4
        seen |= {items[i].path for i in picks}
    assert {os.path.basename(p) for p in seen} >= {"a_001.mp4", "a_002.mp4", "a_003.mp4"}  # each clip gets its turn
    assert sum(len(b) for b in ds.make_batches(items, 2, random.Random(1))) == 12
    # a clip added by hand, not in the manifest, counts on its own
    write_split(str(tmp_path / "x"), {"c_001.mp4": "c.mp4"})
    os.replace(str(tmp_path / "x" / "c_001.mp4"), os.path.join(root, "extra.mp4"))
    items = ds.build_items([{"path": root}], 64, 16, False, lambda *_: None, video_frames=22, fps=24)
    assert len(ds.epoch_items(items, random.Random(0))) == 1 + 2 + 1


def test_datasets_without_clips_from_a_split_train_as_before(tmp_path):
    items = [ds.Item(f"/x/{i}.png", "", (64, 64)) for i in range(5)]
    rng = random.Random(3)
    assert ds.epoch_items(items, rng) == list(range(5))
    assert rng.random() == random.Random(3).random()  # nothing drawn from the shuffle's generator


def test_the_picker_holds_out_a_whole_video(tmp_path):
    root = str(tmp_path / "split")
    write_split(root, {"a_001.mp4": "a.mp4", "a_002.mp4": "a.mp4", "b_001.mp4": "b.mp4", "b_002.mp4": "b.mp4"})
    items = ds.build_items([{"path": root, "resolution": [32, 64]}], 64, 16, False, lambda *_: None,
                           video_frames=22, fps=24)
    for seed in range(6):
        train, val = evaluation.split_holdout(items, 1, seed)
        held = {it.source for it in val}
        assert len(val) == 1 and not held & {it.source for it in train}  # none of the held video trains
        assert os.path.basename(val[0].path).endswith("_001.mp4") and val[0].resolution == 64


def test_dataset_list_and_check_count_videos(client, tmp_path):
    root = str(tmp_path / "split")
    write_split(root, {"a_001.mp4": "a.mp4", "a_002.mp4": "a.mp4", "b_001.mp4": "b.mp4"}, images=["still.png"])
    did = client.post("/api/datasets", json={"name": "s", "path": root}).json()["id"]
    d = next(x for x in client.get("/api/datasets").json() if x["id"] == did)
    assert d["videos"] == 3 and d["video_sources"] == 2 and d["images"] == 1
    items = {it["name"]: it for it in client.get(f"/api/datasets/{did}/items").json()}
    assert items["a_002.mp4"]["cut_from"] == {"source": "a.mp4", "start": 0} and items["still.png"]["cut_from"] is None
    check = client.get(f"/api/datasets/{did}/check", params={"family": "minimax_h3", "resolution": "256", "frames": 22}).json()
    note = next(i for i in check["issues"] if i["id"] == "split_clips")
    assert note["title"] == "3 clips cut from 2 videos"

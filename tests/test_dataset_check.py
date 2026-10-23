"""The dataset health check: each kind of problem, on a small made-up dataset.

    .venv/bin/python -m pytest tests/test_dataset_check.py -v
"""

import numpy as np
from PIL import Image

from dainamo.server import dataset_check


def _photo(seed, size=(1024, 1024)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)  # big smooth blocks, so resizing keeps the hash
    return Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)


def test_each_problem_is_found(tmp_path, monkeypatch):
    _photo(1).save(tmp_path / "a.png")
    _photo(1, (700, 700)).save(tmp_path / "a_copy.jpg", quality=80)  # the same picture, smaller and re-saved
    _photo(2).save(tmp_path / "b.png")
    _photo(2).save(tmp_path / "b.jpg")  # b.png and b.jpg share b.txt
    _photo(3, (400, 300)).save(tmp_path / "small.png")
    (tmp_path / "broken.png").write_bytes(b"\x89PNG not really")
    (tmp_path / "a.txt").write_text("ohwx, a pattern")
    (tmp_path / "b.txt").write_text("a pattern " + "word " * 600)
    (tmp_path / "small.txt").write_text("")
    (tmp_path / "gone.txt").write_text("caption of a file that was removed")
    monkeypatch.setattr(dataset_check, "_tokenizer", lambda: None)  # counted roughly, whatever is downloaded

    r = dataset_check.check(str(tmp_path), "ohwx", [1024], 16, None, 512)
    issues = {i["id"]: i for i in r["issues"]}
    names = lambda i: sorted(f["name"] for f in issues[i]["files"])
    assert (r["files"], r["images"], r["videos"]) == (6, 5, 0)
    assert [i["level"] for i in r["issues"]] == sorted((i["level"] for i in r["issues"]), key=["error", "warning", "info"].index)
    assert names("unreadable") == ["broken.png"] and issues["unreadable"]["level"] == "error"
    assert names("small") == ["a_copy.jpg", "small.png"]
    # the resized copy, and b saved twice
    assert sorted(sorted(f["name"] for f in g) for g in issues["duplicates"]["groups"]) == [["a.png", "a_copy.jpg"],
                                                                                         ["b.jpg", "b.png"]]
    assert [sorted(f["name"] for f in g) for g in issues["shared_caption"]["groups"]] == [["b.jpg", "b.png"]]
    assert names("no_caption") == ["a_copy.jpg"] and names("empty_caption") == ["small.png"]
    assert names("long_caption") == ["b.jpg", "b.png"] and "roughly" in issues["long_caption"]["detail"]
    assert names("no_trigger") == ["b.jpg", "b.png"]
    assert names("orphan_caption") == ["gone.txt"]

    # at 512 only the 400x300 image is short of pixels; with no caption limit, nothing is too long
    r = dataset_check.check(str(tmp_path), "", [512], 16, None, None)
    issues = {i["id"]: i for i in r["issues"]}
    assert names("small") == ["small.png"] and "long_caption" not in issues and "no_trigger" not in issues
    # several sizes: each is checked, as the trainer's log counts them
    r = dataset_check.check(str(tmp_path), "", [512, 1024], 16, None, None)
    issues = {i["id"]: i for i in r["issues"]}
    assert names("small") == ["a_copy.jpg", "small.png"]
    assert issues["small"]["title"] == "2 files train below 1024, 1 below 512"
    assert dataset_check._duplicate_groups([("x", 0), ("y", 0b111), ("z", 0xFF00FF00FF)]) == [["x", "y"]]


def test_source_datasets_nested_either_way(tmp_path):
    from dainamo.engine.families import get_family

    info = get_family("qwen_image_2").info.sources
    for i, n in enumerate(("photos/cat.jpg", "photos/dog.png", "photos/edited/cat.png", "photos/edited/dog.png",
                           "photos/edited/owl.png", "results/cat.png", "results/dog.png", "results/before/cat.jpg")):
        (tmp_path / n).parent.mkdir(parents=True, exist_ok=True)
        _photo(i, (64, 64)).save(tmp_path / n)
    (tmp_path / "results/before/cat.txt").write_text("a source image's own caption")

    # the sources in photos/, the results in photos/edited/: the results are checked, and pair with the sources
    r = dataset_check.check(str(tmp_path / "photos/edited"), "", [64], 16, references=info,
                            sources=[{"name": "Photos", "path": str(tmp_path / "photos")}])
    issues = {i["id"]: i for i in r["issues"]}
    assert r["files"] == 3 and r["pairs"] == {"with": 2, "of": 3, "from": "sources", "most": 1}
    assert issues["source_contains"]["level"] == "info"
    assert [f["name"] for f in issues["no_source"]["files"]] == ["owl.png"]  # not its own source
    assert "unused_source_0" not in issues  # the results aren't counted as sources
    # the other way round: sources in results/before/, which are only sources, captions and all
    r = dataset_check.check(str(tmp_path / "results"), "", [64], 16, references=info,
                            sources=[{"name": "Before", "path": str(tmp_path / "results/before")}])
    issues = {i["id"]: i for i in r["issues"]}
    assert r["files"] == 2 and r["pairs"]["with"] == 1 and "source_inside" in issues
    assert "orphan_caption" not in issues
    # its own folder: nothing pairs
    r = dataset_check.check(str(tmp_path / "results"), "", [64], 16, references=info,
                            sources=[{"name": "Results", "path": str(tmp_path / "results")}])
    assert r["pairs"]["with"] == 0 and next(i for i in r["issues"] if i["id"] == "source_same")["level"] == "warning"


def test_files_are_read_once(tmp_path, monkeypatch):
    _photo(5).save(tmp_path / "a.png")
    dataset_check.check(str(tmp_path))
    monkeypatch.setattr(dataset_check, "_dhash", lambda im: (_ for _ in ()).throw(AssertionError("read again")))
    dataset_check.check(str(tmp_path))  # same size and time: from the cache


def test_clips(tmp_path):
    import subprocess

    for name, seconds in (("short.mp4", 0.5), ("blip.mp4", 0.04), ("long.mp4", 3)):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=128x128:rate=24",
                        "-pix_fmt", "yuv420p", str(tmp_path / name)], check=True)
    image_model = {i["id"]: i for i in dataset_check.check(str(tmp_path), "", [512], 16, None)["issues"]}
    assert image_model["clips_skipped"]["count"] == 3 and "small" not in image_model
    result = dataset_check.check(str(tmp_path), "", [256], 32, 39)
    video_model = {i["id"]: i for i in result["issues"]}
    assert [f["name"] for f in video_model["clip_too_short"]["files"]] == ["blip.mp4"]
    assert [f["name"] for f in video_model["short_clips"]["files"]] == ["short.mp4"]
    assert video_model["short_clips"]["title"] == "1 clip shorter than 39 frames (1.6 s)"
    # a clip longer than the clip length trains only its start
    assert [f["name"] for f in video_model["long_clips"]["files"]] == ["long.mp4"]
    assert video_model["long_clips"]["title"] == "1 clip longer than 39 frames (1.6 s)"
    assert video_model["small"]["count"] == 3  # 128x128 clips at 256
    assert result["longest_clip"] == 72  # 3 s at 24 fps
    # an Auto clip length trains each clip at its own length (the longest a clip trains at being 241 frames)
    auto = dataset_check.check(str(tmp_path), "", [256], 32, 226, clip_length=241, auto_frames=True)
    assert {i["id"] for i in auto["issues"]} & {"short_clips", "long_clips"} == set()
    assert "clip_too_short" in {i["id"] for i in auto["issues"]} and auto["longest_clip"] == 72

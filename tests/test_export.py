"""Exporting datasets: one .zip, a folder each, with captions as they are or as training reads them.

    .venv/bin/python -m pytest tests/test_export.py -v
"""

import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dainamo.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_export_zips_each_dataset_with_its_captions(client, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for folder in ("_refs", "_masks", "sub/_masks"):
        (a / folder).mkdir(parents=True)
    b.mkdir()
    Image.new("RGB", (8, 8), "red").save(a / "one.png")
    Image.new("RGB", (8, 8), "red").save(a / "one.jpg")  # shares its mask (and reference) with one.png
    (a / "one.txt").write_text("a red cat")
    Image.new("RGB", (8, 8)).save(a / "sub" / "two.png")  # no caption
    Image.new("RGB", (8, 8)).save(a / "_refs" / "one.png")
    Image.new("L", (8, 8), 255).save(a / "_masks" / "one.png")
    Image.new("L", (8, 8), 255).save(a / "sub" / "_masks" / "two.png")
    Image.new("L", (8, 8), 255).save(a / "_masks" / "song.png")  # a song has no mask
    (a / "song.mp3").write_bytes(b"ID3 not really a song")
    (a / "song.txt").write_text("pop, upbeat\n[Lyrics]\nla la la")
    (a / "notes.md").write_text("not a dataset file")
    big = os.urandom(3 * 1024 * 1024 + 7)  # over a few of the stream's pieces
    (b / "clip.mp4").write_bytes(big)
    (b / ".clips.json").write_text('{"clips": {}}')
    da = client.post("/api/datasets", json={"name": "Cats/dogs", "path": str(a), "trigger": "ohwx"}).json()["id"]
    db = client.post("/api/datasets", json={"name": "Cats/dogs", "path": str(b)}).json()["id"]
    listed = {d["id"]: d for d in client.get("/api/datasets").json()}
    assert listed[db]["bytes"] == len(big)
    # masks aren't files of the dataset; three of its images have one
    assert (listed[da]["count"], listed[da]["images"], listed[da]["masks"], listed[db]["masks"]) == (4, 3, 3, 0)

    r = client.get("/api/export/datasets", params={"ids": f"{da},{db}"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert 'filename="datasets_' in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert z.testzip() is None
    assert sorted(z.namelist()) == sorted([
        "Cats_dogs/one.jpg", "Cats_dogs/one.png", "Cats_dogs/one.txt", "Cats_dogs/_refs/one.png", "Cats_dogs/_masks/one.png",
        "Cats_dogs/song.mp3", "Cats_dogs/song.txt", "Cats_dogs/sub/two.png", "Cats_dogs/sub/_masks/two.png",
        "Cats_dogs (2)/clip.mp4", "Cats_dogs (2)/.clips.json"])  # each file once
    assert z.read("Cats_dogs/one.txt") == b"a red cat" and z.read("Cats_dogs (2)/clip.mp4") == big
    assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())

    # as training reads them: the trigger word in each caption, a song's on its style tags
    r = client.get("/api/export/datasets", params={"ids": da, "trigger": True})
    assert r.headers["content-disposition"] == "attachment; filename=\"Cats_dogs.zip\"; filename*=UTF-8''Cats_dogs.zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert z.read("Cats_dogs/one.txt").decode() == "ohwx, a red cat"
    assert z.read("Cats_dogs/sub/two.txt").decode() == "ohwx"
    assert z.read("Cats_dogs/song.txt").decode() == "ohwx, pop, upbeat\n[Lyrics]\nla la la"

    assert client.get("/api/export/datasets", params={"ids": ""}).status_code == 400
    assert client.get("/api/export/datasets", params={"ids": "nope"}).status_code == 404

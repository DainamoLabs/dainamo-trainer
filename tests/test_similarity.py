"""Likeness (engine.similarity): the concept of a dataset, the score of an image against it, the duplicate finder, the
cache that embeds only new files, the meter that scores training samples, and the program the server runs. DINOv2 is
stood in for by a fake that reads a feature vector off each image's colour.

    .venv/bin/python -m pytest tests/test_similarity.py -v
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from dainamo.engine import similarity as sim

# an image's top-left pixel -> its (unnormalised) feature vector
VECS = {
    (255, 0, 0): [1.0, 0.0, 0.0],      # "red": the base render
    (0, 255, 0): [0.0, 1.0, 0.0],      # "green": unrelated
    (255, 255, 0): [0.0, 0.0, 1.0],    # "yellow": the dataset
    (255, 250, 0): [0.1, 0.0, 1.0],    # "nearly yellow": a near copy of a dataset image
    (0, 0, 255): [0.6, 0.0, 0.8],      # "blue": a sample part way from red to yellow
    (255, 255, 255): [0.0, 0.0, 0.0],  # white: nothing
}


class FakeEmbedder:
    def __init__(self, device=None, what=""):
        self.calls = []

    def __call__(self, images):
        px = [im.convert("RGB").getpixel((0, 0)) for im in images]
        self.calls.append(px)
        return F.normalize(torch.tensor([VECS[p] for p in px]), dim=-1) if px else torch.zeros(0, 3)

    def unload(self):
        pass


def square(path, color, size=8):
    Image.new("RGB", (size, size), color).save(path)
    return str(path)


@pytest.fixture
def dataset(tmp_path):
    """Three yellow images (and a caption, which isn't media)."""
    paths = [square(tmp_path / f"img_{i}.png", (255, 255, 0)) for i in range(3)]
    (tmp_path / "img_0.txt").write_text("a caption")
    return paths


def test_a_concept_scores_images_between_the_floor_and_its_own_images(dataset):
    emb = FakeEmbedder()
    c = sim.Concept.build(emb, dataset)
    assert c.self_sim == pytest.approx(1.0) and len(c.paths) == 3
    blue = emb([Image.new("RGB", (4, 4), (0, 0, 255))])[0]
    # without a base render the floor is FLOOR (0): a cosine of 0.8 to a dataset whose images score 1 is 80%
    r = c.score(blue)
    assert r["sim"] == pytest.approx(0.8, abs=1e-3) and r["likeness"] == pytest.approx(80.0, abs=0.1)
    assert r["nearest"]["path"] in dataset and r["nearest"]["sim"] == pytest.approx(0.8, abs=1e-3)
    assert not r["near_copy"]
    # with the base render (red, cosine 0) as the floor the number is the same here; a floor half way halves it
    red = emb([Image.new("RGB", (4, 4), (255, 0, 0))])[0]
    assert c.score(blue, red)["likeness"] == pytest.approx(80.0, abs=0.1) and c.score(blue, red)["floor"] == 0.0
    assert c.score(blue, torch.tensor([0.8, 0.0, 0.6]))["likeness"] == pytest.approx(50.0, abs=0.1)
    # clamped to 0–100
    green = emb([Image.new("RGB", (4, 4), (0, 255, 0))])[0]
    assert c.score(green, torch.tensor([0.866, 0.0, 0.5]))["likeness"] == 0.0
    yellow = emb([Image.new("RGB", (4, 4), (255, 255, 0))])[0]
    assert c.score(yellow)["likeness"] == 100.0
    # a base render within MIN_GAP of the target leaves too narrow a scale to read: the floor is FLOOR instead
    close = F.normalize(torch.tensor([0.1, 0.0, 1.0]), dim=0)
    r = c.score(blue, close)
    assert r["floor"] is None and r["base_sim"] == pytest.approx(0.995, abs=1e-3)
    assert r["likeness"] == pytest.approx(80.0, abs=0.1)


def test_an_image_is_scored_against_the_part_of_a_varied_dataset_it_looks_like():
    """A dataset of two unlike kinds of image has a centre neither is near; an image of one kind is measured against
    its own kind, so the LoRA moving a sample towards it reads as rising likeness, not falling."""
    kind_a = [[1.0, 0.1 * i, 0.0] for i in range(3)]
    kind_b = [[0.0, 0.1 * i, 1.0] for i in range(3)]
    emb = F.normalize(torch.tensor(kind_a + kind_b), dim=-1)
    c = sim.Concept.of([f"{i}.png" for i in range(6)], emb)
    assert sim.neighbours(6) == 3 and c.self_sim < 0.75
    base = F.normalize(torch.tensor([1.0, 0.0, 1.0]), dim=0)  # the plain model: between the two
    early = F.normalize(torch.tensor([1.0, 0.1, 0.6]), dim=0)
    late = F.normalize(torch.tensor([1.0, 0.1, 0.1]), dim=0)  # trained: well into kind A
    # against the dataset's centre the trained sample scores lower than the base render did
    assert float(late @ c.centroid) < float(base @ c.centroid)
    r_early, r_late = c.score(early, base), c.score(late, base)
    assert {Path(t["path"]).stem for t in c.score(late, top=3)["top"]} == {"0", "1", "2"}
    assert 0 < r_early["likeness"] < r_late["likeness"] and r_late["likeness"] > 80


def test_a_near_copy_of_a_training_image_is_flagged(dataset):
    emb = FakeEmbedder()
    c = sim.Concept.build(emb, dataset)
    near = emb([Image.new("RGB", (4, 4), (255, 250, 0))])[0]
    r = c.score(near, top=5)
    assert r["near_copy"] and r["nearest"]["sim"] >= sim.NEAR_COPY
    assert len(r["top"]) == 3 and r["top"][0] == r["nearest"]  # top-k is capped by the dataset's size


def test_duplicates_are_pairs_above_the_threshold(tmp_path):
    emb = FakeEmbedder()
    paths = [square(tmp_path / "a.png", (255, 255, 0)), square(tmp_path / "b.png", (255, 250, 0)),
             square(tmp_path / "c.png", (0, 255, 0))]
    c = sim.Concept.build(emb, paths)
    pairs = c.duplicates(0.95)
    assert len(pairs) == 1 and {pairs[0]["a"], pairs[0]["b"]} == {paths[0], paths[1]} and pairs[0]["sim"] > 0.99
    assert c.duplicates(0.0) and len(c.duplicates(0.0)) == 3  # every pair
    assert sim.Concept.build(emb, paths[:1]).duplicates() == []


def test_unreadable_files_are_skipped_and_an_empty_concept_scores_nothing(tmp_path, dataset):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    c = sim.Concept.build(FakeEmbedder(), dataset + [str(bad)])
    assert len(c.paths) == 3
    empty = sim.Concept.build(FakeEmbedder(), [str(bad)])
    assert empty.score(torch.tensor([1.0, 0, 0]))["likeness"] is None and empty.duplicates() == []


def test_the_cache_embeds_only_new_or_changed_files(tmp_path, dataset):
    emb = FakeEmbedder()
    cache = tmp_path / "cache" / "d.pt"
    c = sim.cached_concept(emb, dataset, str(cache))
    assert len(c.paths) == 3 and cache.exists() and sum(len(x) for x in emb.calls) == 3
    # the same files: nothing is embedded
    emb.calls.clear()
    c = sim.cached_concept(emb, dataset, str(cache))
    assert len(c.paths) == 3 and emb.calls == []
    # one added, one changed, one removed: two embedded, and the removed one drops out
    new = square(tmp_path / "img_9.png", (255, 255, 0))
    os.utime(dataset[0], (1, 1))
    os.remove(dataset[2])
    c = sim.cached_concept(emb, [dataset[0], dataset[1], new], str(cache))
    assert sorted(c.paths) == sorted([dataset[0], dataset[1], new])
    assert sorted(p for call in emb.calls for p in call) == [(255, 255, 0)] * 2
    # a cache from another model is ignored
    torch.save({"repo": "other", "paths": c.paths, "stamps": [], "emb": c.emb}, cache)
    emb.calls.clear()
    sim.cached_concept(emb, c.paths, str(cache))
    assert sum(len(x) for x in emb.calls) == 3


def test_big_datasets_are_sampled(tmp_path):
    paths = [square(tmp_path / f"i{i}.png", (255, 255, 0)) for i in range(6)]
    c = sim.Concept.build(FakeEmbedder(), paths, limit=4)
    assert len(c.paths) == 4 and set(c.paths) <= set(paths)
    c2 = sim.Concept.build(FakeEmbedder(), paths, limit=4)
    assert c.paths == c2.paths  # the same sample each time


def test_the_meter_scores_samples_against_the_step_0_render(tmp_path, dataset, monkeypatch):
    """The step-0 sample of a card is the base model's render, so it's the floor of the card's later samples; a
    card first sampled later is rendered at strength 0 for its floor, once. The state survives a resume."""
    monkeypatch.setattr(sim, "Embedder", FakeEmbedder)
    renders = []

    def render(spec):
        renders.append(spec)
        return Image.new("RGB", (4, 4), (255, 0, 0))  # the base model: red

    lines = []
    state = tmp_path / "state" / "likeness.pt"
    m = sim.LikenessMeter("cpu", str(state), render, lines.append)
    m.setup(dataset)
    assert m.concept.self_sim == pytest.approx(1.0) and any("3 training image" in ln for ln in lines)
    card = {"prompt": "a yellow thing", "seed": 1, "width": 64, "height": 64, "lora_scale": 1.0}
    # step 0: the sample is red (the base model); its own vector becomes the floor, and it scores 0%
    r0 = m.score(card, Image.new("RGB", (4, 4), (255, 0, 0)), 0)
    assert r0["likeness"] == 0.0 and r0["floor"] == pytest.approx(0.0) and renders == []
    # step 500: blue is 0.8 of the way from red (0) to yellow (1)
    r = m.score(card, Image.new("RGB", (4, 4), (0, 0, 255)), 500)
    assert r["likeness"] == pytest.approx(80.0, abs=0.1) and r["nearest"]["file"].startswith("img_") and not r["near_copy"]
    assert renders == []
    # a new card at step 500 has no floor yet: rendered at strength 0 (red), once
    other = {"prompt": "another", "seed": 2, "width": 64, "height": 64, "lora_scale": 1.0}
    m.score(other, Image.new("RGB", (4, 4), (0, 0, 255)), 500)
    m.score(other, Image.new("RGB", (4, 4), (0, 0, 255)), 750)
    assert len(renders) == 1 and renders[0]["lora_scale"] == 0.0 and renders[0]["prompt"] == "another"
    # the strength isn't part of a card's identity: a slider's cards at several strengths share one floor
    assert sim.spec_key(dict(card, lora_scale=0.5)) == sim.spec_key(card)
    # resumed: the concept and the floors come from the state file, nothing is embedded or rendered again
    renders.clear()
    m2 = sim.LikenessMeter("cpu", str(state), render, lines.append)
    m2.setup(dataset)
    assert m2.embed.calls == [] and set(m2.bases) == {sim.spec_key(card), sim.spec_key(other)}
    assert m2.score(card, Image.new("RGB", (4, 4), (0, 0, 255)), 1000)["likeness"] == pytest.approx(80.0, abs=0.1)
    assert renders == []
    # the dataset changed: the concept is rebuilt, the floors kept
    m3 = sim.LikenessMeter("cpu", str(state), render, lines.append)
    m3.setup(dataset[:2])
    assert len(m3.concept.paths) == 2 and set(m3.bases) == {sim.spec_key(card), sim.spec_key(other)}


def test_a_finetune_renders_its_floors_at_step_0_only(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(sim, "Embedder", FakeEmbedder)
    renders = []
    m = sim.LikenessMeter("cpu", str(tmp_path / "likeness.pt"), lambda sp: renders.append(sp) or Image.new("RGB", (4, 4), (255, 0, 0)),
                          lambda *a: None, can_render_base=False)
    m.setup(dataset)
    card = {"prompt": "p", "seed": 1, "width": 64, "height": 64}
    # mid-training with no floor: scored from FLOOR, nothing rendered
    r = m.score(card, Image.new("RGB", (4, 4), (0, 0, 255)), 500)
    assert r["floor"] is None and r["likeness"] == pytest.approx(80.0, abs=0.1) and renders == []
    # prime at step 0 renders the cards without a floor
    m.prime([card, dict(card, seed=2)])
    assert len(renders) == 2 and all(sp["lora_scale"] == 0.0 for sp in renders)
    m.prime([card])
    assert len(renders) == 2
    assert m.score(card, Image.new("RGB", (4, 4), (0, 0, 255)), 600)["floor"] == pytest.approx(0.0)


def test_sound_and_clips(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(sim, "Embedder", FakeEmbedder)
    m = sim.LikenessMeter("cpu", str(tmp_path / "likeness.pt"), lambda sp: None, lambda *a: None)
    m.setup(dataset)
    assert m.score({"prompt": "p"}, {"audio": torch.zeros(2, 10), "sample_rate": 48000}, 10) == {}
    # a clip stands in as its first frame
    frames = torch.zeros(3, 4, 4, 3, dtype=torch.uint8)
    frames[..., 2] = 255  # blue
    r = m.score({"prompt": "p", "seed": 1}, {"frames": frames, "fps": 24}, 0)
    assert r["likeness"] == 0.0  # step 0: its own floor
    assert sim.summarise([r, {"likeness": 40.0, "nearest": {"sim": 0.5}, "near_copy": False}, {}]) == \
        {"likeness": 20.0, "nearest": max(r["nearest"]["sim"], 0.5), "near_copy": False}
    assert sim.summarise([{}]) is None


def test_the_program_answers_the_servers_request(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(sim, "Embedder", FakeEmbedder)
    blue = square(tmp_path / "blue.png", (0, 0, 255))
    red = square(tmp_path / "red.png", (255, 0, 0))
    near = square(tmp_path / "near.png", (255, 250, 0))
    cache = str(tmp_path / "c.pt")
    out = sim.run({"concept": {"paths": dataset, "cache": cache}, "images": [blue, near], "floors": [red, None], "top": 2})
    assert out["count"] == 3 and out["self_sim"] == pytest.approx(1.0)
    a, b = out["results"]
    assert a["likeness"] == pytest.approx(80.0, abs=0.1) and a["floor"] == pytest.approx(0.0) and len(a["top"]) == 2
    assert b["near_copy"] and b["floor"] is None
    assert os.path.exists(cache)
    # duplicates, through the file-based entry point the server uses
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"concept": {"paths": dataset + [near], "cache": cache}, "duplicates": 0.95}))
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = sim.main(["similarity", str(req)])
    lines = [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.strip()]
    assert code == 0 and lines[-1]["count"] == 4 and len(lines[-1]["duplicates"]) == 6  # yellow×3 + near: all alike
    assert any("progress" in ln for ln in lines[:-1])
    # an error is reported as JSON too
    req.write_text(json.dumps({"concept": {"paths": dataset}, "images": [str(tmp_path / "missing.png")]}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = sim.main(["similarity", str(req)])
    assert code == 1 and "error" in json.loads(buf.getvalue().splitlines()[-1])

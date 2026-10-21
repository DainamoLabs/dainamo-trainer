"""Checkpoint picker and concept-bleed logic (no GPU or model downloads needed)."""

import os

import pytest
import torch
from PIL import Image

from dainamo.engine import evaluation
from dainamo.engine.dataset import Item
from dainamo.engine.families.base import FamilyDriver


def items(paths, repeats=2, flip=True):
    out = []
    for p in paths:
        for fl in ([False, True] if flip else [False]):
            out += [Item(p, f"cap {p}", (512, 512), fl) for _ in range(repeats)]
    return out


def test_holdout_is_reproducible_and_excludes_every_copy():
    data = items([f"/d/{i}.png" for i in range(10)])
    train, val = evaluation.split_holdout(data, 3, seed=42)
    assert len(val) == 3 and all(not v.flip for v in val)
    held = {v.path for v in val}
    assert not any(t.path in held for t in train)  # no repeat or flipped copy leaks into training
    assert len(train) == len(data) - 3 * 4
    assert {v.path for v in evaluation.split_holdout(data, 3, seed=42)[1]} == held
    assert {v.path for v in evaluation.split_holdout(data, 3, seed=7)[1]} != held
    with pytest.raises(ValueError, match="only have 10"):
        evaluation.split_holdout(data, 10, seed=0)


EVALS = [
    {"step": 0, "file": None, "val_loss": 0.30, "bleed": 0.0},
    {"step": 100, "file": "a_100", "val_loss": 0.20, "bleed": 10.0},
    {"step": 200, "file": "a_200", "val_loss": 0.15, "bleed": 70.0},
    {"step": 300, "file": "a_300", "val_loss": 0.18, "bleed": 40.0},
]
ALL = {"a_100", "a_200", "a_300"}


def test_recommend_picker_only():
    r = evaluation.recommend(EVALS, picker=True, bleed=False, max_bleed=50, available=ALL)
    assert r["file"] == "a_200" and "Lowest validation loss" in r["reason"]


def test_recommend_picker_respects_bleed_limit():
    r = evaluation.recommend(EVALS, picker=True, bleed=True, max_bleed=50, available=ALL)
    assert r["file"] == "a_300"  # a_200 has lower loss but 70% bleed
    r = evaluation.recommend(EVALS, picker=True, bleed=True, max_bleed=5, available=ALL)
    assert r["file"] == "a_200" and "over the 5% bleed limit" in r["reason"]


def test_recommend_bleed_only():
    r = evaluation.recommend(EVALS, picker=False, bleed=True, max_bleed=50, available=ALL)
    assert r["file"] == "a_300"  # the most-trained checkpoint under the limit
    r = evaluation.recommend(EVALS, picker=False, bleed=True, max_bleed=5, available=ALL)
    assert r["file"] == "a_100" and "Lowest bleed" in r["reason"]


def test_recommend_ignores_deleted_and_baseline():
    r = evaluation.recommend(EVALS, picker=True, bleed=False, max_bleed=50, available={"a_100", "a_300"})
    assert r["file"] == "a_300"
    assert evaluation.recommend(EVALS[:1], picker=True, bleed=False, max_bleed=50, available=ALL) is None


class FakeDriver(FamilyDriver):
    """The default flow-matching loss with a tiny conv standing in for the model."""

    def __init__(self):
        super().__init__({}, torch.device("cpu"))

    def sample_timesteps(self, n, hw, settings):
        return torch.rand(n)

    def predict(self, model, noisy, t, text):
        return model(noisy)


def test_validator_is_deterministic(tmp_path):
    files = []
    for i in range(2):
        f = str(tmp_path / f"{i}.pt")
        torch.save({"latent": torch.randn(4, 8, 8)}, f)
        files.append(f)
    model = torch.nn.Conv2d(4, 4, 1)
    assert evaluation.latent_hw(files[0]) == (8, 8)
    v = evaluation.Validator(FakeDriver(), files, lambda i: {"text": torch.zeros(1)}, {}, 6, seed=3, latent_hw=(8, 8))
    assert v.sigmas == sorted(v.sigmas) and len(v.sigmas) == 6 and 0 < v.sigmas[0] < v.sigmas[-1] <= 1
    # quantiles of the family's distribution: uniform here, so roughly evenly spaced
    assert abs(v.sigmas[0] - 1 / 12) < 0.02 and abs(v.sigmas[-1] - 11 / 12) < 0.02
    torch.manual_seed(0)
    a = v.loss(model)
    torch.manual_seed(99)  # global RNG state must not matter
    assert v.loss(model) == a
    with torch.no_grad():
        model.weight.add_(0.1)
    assert v.loss(model) != a


def test_bleed_measures_movement_toward_dataset_and_drift_measures_change(tmp_path, monkeypatch):
    vecs = {  # image colour -> feature vector
        (255, 0, 0): torch.tensor([1.0, 0.0, 0.0]),    # base render
        (0, 255, 0): torch.tensor([0.0, 1.0, 0.0]),    # base, other seed: distance 1.0
        (255, 255, 0): torch.tensor([0.0, 0.0, 1.0]),  # training images: the concept
        (0, 0, 255): torch.tensor([0.6, 0.0, 0.8]),    # checkpoint render
    }

    class FakeEmbed:
        def __init__(self, device):
            pass

        def __call__(self, images):
            return torch.stack([vecs[im.convert("RGB").getpixel((0, 0))] for im in images])

    monkeypatch.setattr(evaluation, "DinoEmbedder", FakeEmbed)
    train = []
    for i in range(3):
        f = tmp_path / f"train_{i}.png"
        Image.new("RGB", (8, 8), (255, 255, 0)).save(f)
        train.append(str(f))
    out = tmp_path / "samples"
    calls = []

    def render(model, network, specs):
        calls.append(specs)
        if all(sp["lora_scale"] == 0.0 for sp in specs):  # base references
            n = len(specs) // 2
            return [Image.new("RGB", (8, 8), (255, 0, 0))] * n + [Image.new("RGB", (8, 8), (0, 255, 0))] * n
        return [Image.new("RGB", (8, 8), (0, 0, 255))] * len(specs)

    probe = evaluation.BleedProbe(["fruit bowl"], 64, 64, 1234, str(out), render, log=lambda *a: None)
    probe.setup(None, None, "cpu", train)
    assert calls[0][0]["seed"] == 1234 and calls[0][1]["seed"] == 1234 + 100_000
    result = probe.measure(None, None, 500)
    # concept similarity 0 -> 0.8 of a 0 -> 1 gap; changed 0.4 against a seed change of 1.0
    assert result["bleed"] == pytest.approx(80.0, abs=0.1)
    assert result["drift"] == pytest.approx(40.0, abs=0.1)
    assert os.path.exists(out / "bleed_step000500_0.png")
    # a resumed job reuses the base references from disk instead of re-rendering them
    calls.clear()
    evaluation.BleedProbe(["fruit bowl"], 64, 64, 1234, str(out), render, log=lambda *a: None).setup(None, None, "cpu", train)
    assert calls == []


def test_multi_resolution_items_cache_keys_and_holdout(tmp_path):
    import hashlib

    from dainamo.engine import dataset as ds

    Image.new("RGB", (1600, 1200)).save(tmp_path / "big.png")
    Image.new("RGB", (500, 500)).save(tmp_path / "small.png")  # too small to fill either size
    got = ds.build_items([{"path": str(tmp_path), "trigger": "", "repeats": 2}], [1024, 512], 16, False)
    sizes = sorted((os.path.basename(it.path), it.resolution, it.size) for it in got)
    assert sorted(set(sizes)) == [("big.png", 512, (576, 432)), ("big.png", 1024, (1168, 880)),
                                  ("small.png", 512, (496, 496))]  # the small one isn't counted twice
    assert len(got) == 6  # repeats apply at each size
    assert ds.resolutions(1024) == [1024] and ds.resolutions([1024, 512, 512]) == [512, 1024]
    # a single-resolution job keeps the cache keys it had before multi-resolution training
    one = next(it for it in ds.build_items([{"path": str(tmp_path)}], 1024, 16, False) if it.size == (1168, 880))
    st = os.stat(one.path)
    assert one.key() == hashlib.sha1(f"{one.path}|{st.st_mtime_ns}|{one.size}|False|1024".encode()).hexdigest()
    train, val = evaluation.split_holdout(got, 1, seed=0)
    assert len(val) == 1 and val[0].resolution == max(it.resolution for it in got if it.path == val[0].path)
    assert not any(t.path == val[0].path for t in train)


def test_each_dataset_trains_at_its_own_resolutions(tmp_path):
    from dainamo.engine import dataset as ds

    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        Image.new("RGB", (1600, 1200)).save(tmp_path / name / "x.png")
    Image.new("RGB", (500, 500)).save(tmp_path / "b" / "small.png")
    lines = []
    got = ds.build_items([{"path": str(tmp_path / "a"), "name": "first", "resolution": [512, 1024], "repeats": 3},
                          {"path": str(tmp_path / "b")}], 768, 16, False, log=lines.append)
    by = sorted({(os.path.basename(os.path.dirname(it.path)), os.path.basename(it.path), it.resolution) for it in got})
    assert by == [("a", "x.png", 512), ("a", "x.png", 1024), ("b", "small.png", 768), ("b", "x.png", 768)]
    assert len(got) == 2 * 3 + 2
    assert lines == ["first: 1 image at 512, 1024, 3 repeats -> 6 items",
                     "b: 2 images at 768 -> 2 items"]
    lines.clear()
    ds.build_items([{"path": str(tmp_path / "b"), "name": "b", "resolution": [512, 1024]}], 768, 16, False, log=lines.append)
    assert lines == ["b: 2 images at 512, 1024 -> 3 items (1 too small for the larger sizes train at fewer of them)"]

    # flip and caption dropout per dataset; the arguments are for datasets that don't say
    got = ds.build_items([{"path": str(tmp_path / "a"), "flip": True, "caption_dropout": 0.5}, {"path": str(tmp_path / "b")}],
                         768, 16, False, log=lambda *a: None, caption_dropout=0.05)
    a = [it for it in got if "/a/" in it.path]
    b = [it for it in got if "/b/" in it.path]
    assert sorted(it.flip for it in a) == [False, True] and {it.caption_dropout for it in a} == {0.5}
    assert not any(it.flip for it in b) and {it.caption_dropout for it in b} == {0.05}


def test_the_bucket_summary_gives_the_range_the_most_used_and_what_trains_small():
    from dainamo.engine.dataset import Item
    from dainamo.engine.trainer import bucket_summary

    def item(path, size, res=1024, frames=1, small=False):
        return Item(path, "", size, frames=frames, resolution=res, small=small)

    items = [item(f"big{i}.png", (1168, 880)) for i in range(12)] + [item(f"tall{i}.png", (880, 1168)) for i in range(9)] + \
        [item("sq.png", (1024, 1024)), item("tiny.png", (144, 208), small=True), item("mid.png", (400, 688), small=True)]
    assert bucket_summary(items) == [
        "24 training items in 5 bucket sizes, from 144x208 to 1024x1024; the most used: 1168x880 (12), 880x1168 (9), "
        "144x208 (1)",
        "2 of 24 files train below 1024: they're smaller than that, and files are never upscaled (the dataset's "
        "Health check lists them)"]
    # several resolutions and clips: each counted against its own resolution
    items = [item("a.png", (512, 512), 512), item("a.png", (880, 1168), 1024), item("c.mp4", (320, 192), 384, 39, True)]
    assert bucket_summary(items) == [
        "3 training items in 3 bucket sizes, from 320x192x39f to 880x1168; the most used: 320x192x39f (1), "
        "512x512 (1), 880x1168 (1)",
        "1 of 2 files train below 384: they're smaller than that, and files are never upscaled (the dataset's "
        "Health check lists them)"]
    assert bucket_summary([item("x.png", (1024, 1024))]) == ["1 training item in 1 bucket size: 1024x1024"]


def test_what_trains_small_is_measured_against_the_bucket_for_its_shape(tmp_path):
    from dainamo.engine import dataset as ds

    # large files whose buckets lose over 10% of resolution^2 to rounding (div 32) aren't small
    assert ds.bucket_size(6000, 4000, 384, 32) == (448, 288) and not ds.trains_small(6000, 4000, 384, 32)
    assert not ds.trains_small(2560, 1080, 384, 32) and not ds.trains_small(4000, 3000, 256, 32)
    # a little short of the bucket trains a little smaller, which isn't worth a warning
    assert ds.bucket_size(1000, 1000, 1024, 16) == (992, 992) and not ds.trains_small(1000, 1000, 1024, 16)
    assert ds.trains_small(700, 700, 1024, 16) and ds.trains_small(400, 300, 512, 16)
    # build_items marks each item at its own resolution
    Image.new("RGB", (700, 700)).save(tmp_path / "mid.png")
    got = ds.build_items([{"path": str(tmp_path)}], [512, 1024], 16, False, log=lambda *a: None)
    assert sorted((it.resolution, it.small) for it in got) == [(512, False), (1024, True)]

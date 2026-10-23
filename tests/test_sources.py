"""Source datasets: an edit's source images (or a clip's references) kept as datasets of their own, paired with a
training dataset's files by name; the _refs folder when there are none.

    .venv/bin/python -m pytest tests/test_sources.py -v
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dainamo.engine import dataset as ds
from dainamo.engine.families import get_family
from dainamo.engine.families.base import takes_sources
from dainamo.server import dataset_check
from dainamo.server.app import app


def _images(folder, *names):
    for n in names:
        (folder / n).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(folder / n)


@pytest.fixture
def edit(tmp_path):
    """results/ (cat, dog, sub/owl, fish) with sources in before/ (cat, dog, owl, and stray) and second/ (cat)."""
    results, before, second = tmp_path / "results", tmp_path / "before", tmp_path / "second"
    _images(results, "cat.png", "dog.png", "sub/owl.png", "fish.png")
    _images(before, "cat.jpg", "dog.webp", "owl.png", "stray.png")
    _images(second, "cat.png")
    (results / "cat.txt").write_text("make it night")
    return results, before, second


def test_files_pair_with_the_image_of_their_name_in_each_source_dataset(edit):
    results, before, second = edit
    sources = [{"path": str(before), "name": "Before"}, {"path": str(second), "name": "Second"}]
    lines = []
    items = ds.build_items([{"path": str(results), "sources": sources}], 64, 16, False, log=lines.append, references=True)
    refs = {it.path.removeprefix(str(results) + "/"): [r.removeprefix(str(results.parent) + "/") for r in it.refs]
            for it in items}
    # any image type; sources in one folder pair with files in subfolders; in the source datasets' order
    assert refs == {"cat.png": ["before/cat.jpg", "second/cat.png"], "dog.png": ["before/dog.webp"],
                    "sub/owl.png": ["before/owl.png"], "fish.png": []}
    assert "  3 of 4 with references from Before, Second; 1 without" in lines
    # a model that doesn't train with references pairs nothing, and the files' cache keys don't change with them
    plain = ds.build_items([{"path": str(results), "sources": sources}], 64, 16, False, log=lambda *_: None)
    assert all(it.refs == [] for it in plain)
    cat = next(it for it in items if it.path.endswith("cat.png"))
    assert cat.key() != next(it for it in plain if it.path.endswith("cat.png")).key()


def test_without_source_datasets_the_refs_folder_is_used(tmp_path):
    _images(tmp_path, "a.png", "b.png", "_refs/a.png", "_refs/a_1.png", "_refs/b_x.png")
    items = {it.path: it.refs for it in ds.build_items([{"path": str(tmp_path)}], 64, 16, False,
                                                       log=lambda *_: None, references=True)}
    assert items == {str(tmp_path / "a.png"): [str(tmp_path / "_refs/a.png"), str(tmp_path / "_refs/a_1.png")],
                     str(tmp_path / "b.png"): []}
    # with a source dataset, _refs isn't used; one inside the dataset's folder is sources only
    _images(tmp_path, "src/a.png")
    items = ds.build_items([{"path": str(tmp_path), "sources": [{"path": str(tmp_path / "src")}]}], 64, 16, False,
                           log=lambda *_: None, references=True)
    assert {it.path: it.refs for it in items} == {str(tmp_path / "a.png"): [str(tmp_path / "src/a.png")],
                                                  str(tmp_path / "b.png"): []}


def test_a_source_dataset_may_hold_the_datasets_folder(tmp_path):
    # the sources in tmp/, the results in tmp/edited/: the results train, and pair with the sources outside their folder
    _images(tmp_path, "cat.jpg", "dog.png", "edited/cat.png", "edited/dog.png", "edited/owl.png")
    lines = []
    items = ds.build_items([{"path": str(tmp_path / "edited"), "sources": [{"path": str(tmp_path), "name": "Photos"}]}],
                           64, 16, False, log=lines.append, references=True)
    assert {it.path: it.refs for it in items} == {str(tmp_path / "edited/cat.png"): [str(tmp_path / "cat.jpg")],
                                                  str(tmp_path / "edited/dog.png"): [str(tmp_path / "dog.png")],
                                                  str(tmp_path / "edited/owl.png"): []}  # not its own source
    assert "  2 of 3 with references from Photos; 1 without" in lines
    # a dataset chosen as its own source still trains, but nothing pairs
    items = ds.build_items([{"path": str(tmp_path / "edited"), "sources": [{"path": str(tmp_path / "edited")}]}],
                           64, 16, False, log=lambda *_: None, references=True)
    assert len(items) == 3 and all(it.refs == [] for it in items)


def test_which_models_train_with_sources():
    qwen, minimax, krea = (get_family(f).info for f in ("qwen_image_2", "minimax_h3", "krea2"))
    assert takes_sources(qwen, {}) and not takes_sources(krea, {})
    assert takes_sources(minimax, {"variant": "ref2va"}) and not takes_sources(minimax, {"variant": "fl2va"})
    assert not takes_sources(minimax, {})  # fl2va, the variant's default


def test_the_vram_projection_counts_source_images(monkeypatch):
    from dainamo.engine import memory

    Qwen, MiniMax = get_family("qwen_image_2"), get_family("minimax_h3")
    assert Qwen.reference_tokens(1024, 1024) == 1280  # its latents, and its vision tokens in the prompt
    monkeypatch.setattr(memory, "model_weights", lambda *a, **k: (0, [0] * 4, {}))
    monkeypatch.setattr(memory, "vae_bytes", lambda *a: 0)
    monkeypatch.setattr(memory, "file_gpu_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(memory, "lora_params", lambda *a, **k: 0)

    def activations(F, refs):
        p = memory.project(F, {"diffusion_model": "x", "vae": "x", "text_encoder": ""},
                           {"datasets": [{"resolution": 512, "refs": refs}], "num_frames": 1})
        return dict(next(ph for ph in p["phases"] if ph["id"] == "training")["parts"])["Activations"]

    one, two = activations(Qwen, 0), activations(Qwen, 1)
    tokens = Qwen.sequence_tokens(512, 512)
    assert two / one == pytest.approx((tokens + Qwen.reference_tokens(512, 512)) / tokens)
    # MiniMax H3 already counts one reference
    assert activations(MiniMax, 1) == activations(MiniMax, 0) < activations(MiniMax, 2)


def test_the_health_check_says_how_files_pair(edit):
    results, before, second = edit
    info = get_family("qwen_image_2").info.sources
    sources = [{"name": "Before", "path": str(before)}, {"name": "Second", "path": str(second)}]
    r = dataset_check.check(str(results), "", [64], 16, references=info, sources=sources)
    issues = {i["id"]: i for i in r["issues"]}
    names = lambda i: sorted(f["name"] for f in issues[i]["files"])
    assert r["pairs"] == {"with": 3, "of": 4, "from": "sources", "most": 2}
    assert issues["no_source"]["title"] == "1 file without a source image" and names("no_source") == ["fish.png"]
    assert "train text-to-image" in issues["no_source"]["detail"]
    assert names("no_source_1") == ["dog.png", "sub/owl.png"]  # paired in Before, not Second
    assert issues["unused_source_0"]["title"] == "1 source image in “Before” pairs with no file"
    assert [f["path"] for f in issues["unused_source_0"]["files"]] == [str(before / "stray.png")]
    # nothing pairs: a naming problem, most likely
    r = dataset_check.check(str(second), "", [64], 16, references=info,
                            sources=[{"name": "Results", "path": str(results / "sub")}])
    assert next(i for i in r["issues"] if i["id"] == "no_source")["title"] == "No file pairs with a source image"
    # a missing folder; and with no sources and no _refs, it's text-to-image: nothing to say
    r = dataset_check.check(str(results), "", [64], 16, references=info,
                            sources=[{"name": "Gone", "path": str(results.parent / "gone")}])
    assert any(i["id"] == "source_missing" and i["level"] == "error" for i in r["issues"])
    r = dataset_check.check(str(results), "", [64], 16, references=info)
    assert r["pairs"] is None and not any("source" in i["id"] for i in r["issues"])


def test_the_health_check_reads_the_refs_folder(tmp_path):
    _images(tmp_path, "a.png", "b.png", "_refs/a.png", "_refs/c.png")
    info = get_family("qwen_image_2").info.sources
    r = dataset_check.check(str(tmp_path), "", [64], 16, references=info)
    issues = {i["id"]: i for i in r["issues"]}
    assert r["pairs"] == {"with": 1, "of": 2, "from": "refs", "most": 1}
    assert [f["name"] for f in issues["no_source"]["files"]] == ["b.png"]
    assert [f["name"] for f in issues["unused_source_0"]["files"]] == ["_refs/c.png"]
    assert dataset_check.check(str(tmp_path), "", [64], 16)["pairs"] is None  # a model without references


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_jobs_keep_their_source_datasets(client, edit):
    results, before, second = edit
    client.put("/api/settings", json={"queue": {"gpus": [99]}})
    qwen = client.post("/api/profiles", json={"name": "qe", "family": "qwen_image_2", "data": {}}).json()["id"]
    krea = client.post("/api/profiles", json={"name": "kr", "family": "krea2", "data": {}}).json()["id"]
    did, bid, sid = (client.post("/api/datasets", json={"name": n, "path": str(p)}).json()["id"]
                     for n, p in (("Results", results), ("Before", before), ("Second", second)))

    def job(profile, sources):
        return client.post("/api/jobs", json={"name": "se", "profile_id": profile, "settings": {},
                                              "datasets": [{"id": did, "sources": sources}]})

    r = job(qwen, [bid, sid])
    assert r.status_code == 200
    assert r.json()["config"]["datasets"][0]["sources"] == [
        {"id": bid, "name": "Before", "path": str(before)}, {"id": sid, "name": "Second", "path": str(second)}]
    client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})
    r = job(krea, [bid])  # Krea doesn't train with them
    assert r.status_code == 200 and "sources" not in r.json()["config"]["datasets"][0]
    client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})
    for bad in ([did], [bid, bid], [bid, sid, bid, sid]):
        assert job(qwen, bad).status_code == 400

    check = client.get(f"/api/datasets/{did}/check", params={"family": "qwen_image_2", "references": "true",
                                                              "sources": f"{bid},{sid}", "resolution": "512"}).json()
    assert check["pairs"] == {"with": 3, "of": 4, "from": "sources", "most": 2}
    pairs = client.get(f"/api/datasets/{did}/pairs", params={"sources": bid}).json()
    assert pairs["from"] == "sources" and pairs["columns"] == ["Before"]
    rows = {row["name"]: row for row in pairs["rows"]}
    assert rows["cat.png"]["refs"] == [str(before / "cat.jpg")] and rows["cat.png"]["caption"] == "make it night"
    assert rows["fish.png"]["refs"] == [None]
    assert pairs["unused"] == [{"name": "Before", "files": [{"path": str(before / "stray.png"), "name": "stray.png"}]}]
    spaced = client.get(f"/api/datasets/{did}/pairs", params={"sources": f" {bid} ,"})  # spaces around the ids
    assert spaced.status_code == 200 and spaced.json()["columns"] == ["Before"]
    # a source dataset's files are shown like any dataset's
    assert client.get("/api/thumb", params={"path": str(before / "cat.jpg"), "size": 64}).status_code == 200


def test_the_models_train_with_the_images_an_item_pairs_with(edit):
    import torch

    results, before, _ = edit
    items = {it.path.rsplit("/", 1)[-1]: it for it in ds.build_items(
        [{"path": str(results), "sources": [{"path": str(before)}]}], 512, 32, False, log=lambda *_: None, references=True)}
    qwen = get_family("qwen_image_2")({}, torch.device("cpu"))
    assert qwen.trains_references
    conds = qwen.item_conditions(items["cat.png"], {})
    assert list(conds) == ["ref"] and conds["ref"][0] == "make it night" and len(conds["ref"][1]) == 1
    assert list(qwen.item_conditions(items["fish.png"], {})) == ["text"]  # no source: text-to-image
    ref2va = get_family("minimax_h3")({"variant": "ref2va"}, torch.device("cpu"))
    assert ref2va.trains_references and len(ref2va.item_conditions(items["cat.png"], {})["ref"][1]) == 1
    assert not get_family("minimax_h3")({}, torch.device("cpu")).trains_references

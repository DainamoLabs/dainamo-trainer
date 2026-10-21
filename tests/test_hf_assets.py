"""Hugging Face files come from the local cache when they're there; the hub is only asked for what isn't.

    .venv/bin/python -m pytest tests/test_hf_assets.py -v
"""

import pytest
from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError

from dainamo.engine import hf_assets


@pytest.fixture
def hub(monkeypatch):
    """A stand-in hub: ``cached`` says what the local cache holds; ``online`` whether downloads work."""
    state = {"cached": set(), "online": True, "calls": []}

    def snapshot_download(repo, revision=None, allow_patterns=None, local_files_only=False, token=None):
        state["calls"].append(("snapshot", repo, revision, local_files_only))
        if local_files_only:
            if repo not in state["cached"]:
                raise IncompleteSnapshotError("the cached snapshot is incomplete", snapshot_path=f"/cache/{repo}")
            return f"/cache/{repo}"
        if not state["online"]:
            raise ConnectionError("Connection refused")
        state["cached"].add(repo)
        return f"/cache/{repo}"

    def hf_hub_download(repo, filename, local_files_only=False, token=None):
        state["calls"].append(("file", repo, filename, local_files_only))
        if local_files_only:
            if repo not in state["cached"]:
                raise LocalEntryNotFoundError("not cached")
            return f"/cache/{repo}/{filename}"
        if not state["online"]:
            raise ConnectionError("Connection refused")
        state["cached"].add(repo)
        return f"/cache/{repo}/{filename}"

    monkeypatch.setattr(hf_assets, "snapshot_download", snapshot_download)
    monkeypatch.setattr(hf_assets, "hf_hub_download", hf_hub_download)
    return state


def test_cached_files_need_no_network(hub):
    hub["cached"].add(hf_assets.MOSS_REPO)
    hub["online"] = False
    assert hf_assets.load_assets(hf_assets.MOSS_REPO, ["tokenizer/*"], lambda p: p, "x") == "/cache/" + hf_assets.MOSS_REPO
    assert hub["calls"] == [("snapshot", hf_assets.MOSS_REPO, hf_assets.REVISIONS[hf_assets.MOSS_REPO], True)]


def test_missing_files_are_downloaded_once_at_the_pinned_commit(hub):
    loads = []
    hf_assets.load_assets(hf_assets.MOSS_REPO, ["tokenizer/*"], loads.append, "x")
    rev = hf_assets.REVISIONS[hf_assets.MOSS_REPO]
    assert hub["calls"] == [("snapshot", hf_assets.MOSS_REPO, rev, True), ("snapshot", hf_assets.MOSS_REPO, rev, False)]
    hub["calls"].clear()
    hub["online"] = False
    hf_assets.load_assets(hf_assets.MOSS_REPO, ["tokenizer/*"], loads.append, "x")  # now from the cache
    assert len(hub["calls"]) == 1 and len(loads) == 2


def test_a_cached_folder_missing_a_file_is_completed(hub):
    hub["cached"].add(hf_assets.QWEN3VL_REPO)
    tries = []

    def load(path):  # the first folder lacks a file, as an interrupted download would leave it
        tries.append(path)
        if len(tries) == 1:
            raise OSError("tokenizer.json not found")
        return "loaded"

    assert hf_assets.load_assets(hf_assets.QWEN3VL_REPO, ["*.json"], load, "x") == "loaded"
    assert [c[3] for c in hub["calls"]] == [True, False]


def test_offline_without_the_files_explains(hub):
    hub["online"] = False
    with pytest.raises(RuntimeError) as e:
        hf_assets.load_assets(hf_assets.MOSS_REPO, ["tokenizer/*"], lambda p: p, "MOSS-Image's tokenizer")
    assert "MOSS-Image's tokenizer come from Hugging Face (OpenMOSS-Team/MOSS-Image-SFT)" in str(e.value)
    assert "Connection refused" in str(e.value) and "Connect to the internet" in str(e.value)


def test_hub_paths_and_models_come_from_the_cache_first(hub, tmp_path):
    local = tmp_path / "adapter.safetensors"
    local.write_bytes(b"")
    assert hf_assets.resolve_path(str(local)) == str(local)
    assert hf_assets.resolve_path("ostris/adapter/a.safetensors") == "/cache/ostris/adapter/a.safetensors"
    assert [c[3] for c in hub["calls"]] == [True, False]
    hub["online"] = False
    assert hf_assets.resolve_path("ostris/adapter/a.safetensors").endswith("a.safetensors")
    with pytest.raises(RuntimeError, match="The file b.safetensors come from Hugging Face"):
        hf_assets.resolve_path("ostris/other/b.safetensors")

    class Model:
        calls = []

        @classmethod
        def from_pretrained(cls, repo, local_files_only=False, token=None, **kw):
            cls.calls.append(local_files_only)
            if local_files_only and repo != "cached/model":
                raise OSError("not in the cache")
            return cls()

    hf_assets.from_pretrained(Model, "cached/model", "x")
    hf_assets.from_pretrained(Model, "new/model", "x")
    assert Model.calls == [True, True, False]

"""Hugging Face files come from the local cache when they're there; the hub is only asked for what isn't.

    .venv/bin/python -m pytest tests/test_hf_assets.py -v
"""

import pytest
from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError

from dainamo.engine import hf_assets


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A stand-in hub: ``cached`` says what the local cache holds; ``online`` whether downloads work. Nothing is
    bundled, so every load goes to the hub."""
    state = {"cached": set(), "online": True, "calls": []}
    monkeypatch.setattr(hf_assets, "ASSETS_DIR", str(tmp_path / "no-assets"))

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


def test_bundled_files_come_first_and_need_no_hub(hub, monkeypatch, tmp_path):
    folder = tmp_path / "OpenMOSS-Team--MOSS-Image-SFT"
    (folder / "tokenizer").mkdir(parents=True)
    monkeypatch.setattr(hf_assets, "ASSETS_DIR", str(tmp_path))
    hub["online"] = False
    assert hf_assets.bundled(hf_assets.MOSS_REPO) == str(folder)
    assert hf_assets.bundled(hf_assets.MINIMAX_REPO) is None  # not bundled: its license excludes territories
    assert hf_assets.load_assets(hf_assets.MOSS_REPO, ["tokenizer/*"], lambda p: p, "x") == str(folder)
    assert hub["calls"] == []
    # a bundled folder that can't be loaded (a file missing) falls back to the cache, then the hub
    hub["cached"].add(hf_assets.MOSS_REPO)
    tries = []

    def load(path):
        tries.append(path)
        if path == str(folder):
            raise OSError("tokenizer.json not found")
        return "loaded"

    assert hf_assets.load_assets(hf_assets.MOSS_REPO, ["tokenizer/*"], load, "x") == "loaded"
    assert tries == [str(folder), "/cache/" + hf_assets.MOSS_REPO]


def test_the_shipped_assets_match_the_pinned_repos():
    """The folders in dainamo/assets are there for the repos the loaders bundle, laid out as they read them."""
    import os
    for repo, files in {hf_assets.QWEN3VL_REPO: ["tokenizer.json", "config.json"],
                        hf_assets.MOSS_REPO: ["tokenizer/tokenizer.json", "text_encoder/config.json"],
                        hf_assets.QWEN_IMAGE_2_REPO: ["processor/tokenizer.json", "text_encoder/config.json",
                                                      "vae/config.json"],
                        hf_assets.QWEN3_REPOS["4B"]: ["tokenizer.json", "tokenizer_config.json", "config.json"],
                        hf_assets.QWEN3_REPOS["8B"]: ["tokenizer.json", "tokenizer_config.json", "config.json"]}.items():
        folder = hf_assets.bundled(repo)
        assert folder is not None, repo
        for f in files:
            assert os.path.getsize(os.path.join(folder, f)) > 0, f"{repo}: {f}"
    assert hf_assets.bundled(hf_assets.MINIMAX_REPO) is None


def test_the_bundled_qwen3_text_encoders_are_the_ones_flux2_klein_reads():
    """FLUX.2 Klein's two text encoders: the configs are the sizes the DiTs take (3 x 2560 and 3 x 4096 wide), and
    the chat template turns thinking off as the reference prompts it."""
    tok4, cfg4 = hf_assets.qwen3_tokenizer_and_config("4B")
    tok8, cfg8 = hf_assets.qwen3_tokenizer_and_config("8B")
    assert (cfg4.model_type, cfg4.hidden_size, cfg4.num_hidden_layers) == ("qwen3", 2560, 36)
    assert (cfg8.model_type, cfg8.hidden_size, cfg8.num_hidden_layers) == ("qwen3", 4096, 36)
    text = tok4.apply_chat_template([{"role": "user", "content": "a cat"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    assert text == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert tok8(text)["input_ids"] == tok4(text)["input_ids"]


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

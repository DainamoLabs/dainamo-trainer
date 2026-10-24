"""Small Hugging Face assets that ComfyUI single-file models don't carry (tokenizers, chat templates, configs),
hub-hosted adapters, and the models some features use (DINOv2, Whisper).

The families' tokenizers and configs whose licenses allow it ship with the app, in ``dainamo/assets/<owner>--<repo>/``
at the pinned commit (``ASSETS_DIR``; see its README), so those families run without a network from the start.
Everything else is read from the local Hugging Face cache when it's there; Hugging Face is only contacted to
download what isn't cached yet, once, so a job with its files cached runs without a network. The model
families' repos are pinned to a commit, so their files can't change under a project when a repo is updated."""

from __future__ import annotations

import os
from functools import lru_cache

from huggingface_hub import hf_hub_download, snapshot_download

# the bundled copies of some repos' small files, laid out like the repos: Qwen3-VL-4B, Qwen3-4B, Qwen3-8B and MOSS-Image
# (Apache 2.0),
# Qwen-Image 2.1 (redistributable under the Qwen Research License, with the notice in NOTICE). MiniMax H3's aren't:
# its license excludes several territories outright
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

QWEN3VL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MOSS_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MINIMAX_REPO = "MiniMaxAI/MiniMax-H3"
QWEN_IMAGE_2_REPO = "Qwen/Qwen-Image-2.1"
# FLUX.2 Klein's text encoders: Qwen3-4B for the 4B, Qwen3-8B for the 9B
QWEN3_REPOS = {"4B": "Qwen/Qwen3-4B", "8B": "Qwen/Qwen3-8B"}
# the commits the families are built against (later ones, so far, only changed READMEs)
REVISIONS = {
    QWEN3VL_REPO: "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    MOSS_REPO: "2c7954437269461767f533b006823e8f85973a60",
    MINIMAX_REPO: "42ed227ee7df40d41602854ae760620d6eb651fe",
    QWEN_IMAGE_2_REPO: "d26bb61231c349cf6b7896fa83353113880e1ba3",
    QWEN3_REPOS["4B"]: "1cfa9a7208912126459214e8b04321603b3df60c",
    QWEN3_REPOS["8B"]: "b968826d9c46dd6066d109eabc6255188de91218",
}
_TOKENIZER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "generation_config.json"]


def _download_failed(what: str, repo: str, error: Exception) -> RuntimeError:
    return RuntimeError(f"{what} come from Hugging Face ({repo}) and aren't downloaded yet, but the download "
                        f"failed: {error}. Connect to the internet (and set HF_TOKEN if the repo needs a login), "
                        "then start again; they're kept after that.")


def bundled(repo: str) -> str | None:
    """The folder of ``repo``'s files that ship with the app (``ASSETS_DIR``), or None when they don't."""
    path = os.path.join(ASSETS_DIR, repo.replace("/", "--"))
    return path if os.path.isdir(path) else None


def load_assets(repo: str, patterns: list[str], load, what: str):
    """``load(folder)`` on the files of ``repo`` matching ``patterns`` at its pinned commit: the bundled copies
    (``bundled``), else from the cache when they're all there (``load`` fails on a missing one), else downloaded
    first."""
    revision = REVISIONS[repo]
    folder = bundled(repo)
    if folder is not None:
        try:
            return load(folder)
        except OSError:  # a bundled file is missing or damaged: the hub has the same ones
            pass
    try:
        return load(snapshot_download(repo, revision=revision, allow_patterns=patterns, local_files_only=True))
    except OSError:  # not cached, or only partly
        pass
    try:
        path = snapshot_download(repo, revision=revision, allow_patterns=patterns, token=os.getenv("HF_TOKEN"))
    except Exception as e:
        raise _download_failed(what, repo, e) from e
    return load(path)


def from_pretrained(cls, repo: str, what: str, **kwargs):
    """``cls.from_pretrained(repo)`` (a transformers class) from the cache when it's there, else downloaded."""
    try:
        return cls.from_pretrained(repo, local_files_only=True, **kwargs)
    except OSError:
        pass
    try:
        return cls.from_pretrained(repo, token=os.getenv("HF_TOKEN"), **kwargs)
    except OSError as e:
        raise _download_failed(what, repo, e) from e


@lru_cache(maxsize=1)
def qwen3vl_tokenizer_and_config():
    from transformers import AutoConfig, AutoTokenizer

    return load_assets(QWEN3VL_REPO, _TOKENIZER_FILES,
                       lambda path: (AutoTokenizer.from_pretrained(path), AutoConfig.from_pretrained(path)),
                       "The Qwen3-VL tokenizer and config")


@lru_cache(maxsize=2)
def qwen3_tokenizer_and_config(size: str):
    """Qwen3 tokenizer + chat template and the model config of ``size`` ("4B" or "8B")."""
    from transformers import AutoConfig, AutoTokenizer

    repo = QWEN3_REPOS[size]
    return load_assets(repo, _TOKENIZER_FILES,
                       lambda path: (AutoTokenizer.from_pretrained(path), AutoConfig.from_pretrained(path)),
                       f"The Qwen3-{size} tokenizer and config")


@lru_cache(maxsize=1)
def moss_tokenizer_and_config():
    """Qwen3.5 tokenizer + chat template and the text-encoder config shipped with MOSS-Image."""
    from transformers import AutoConfig, AutoTokenizer

    return load_assets(MOSS_REPO, ["tokenizer/*", "text_encoder/config.json"],
                       lambda path: (AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer")),
                                     AutoConfig.from_pretrained(os.path.join(path, "text_encoder"))),
                       "MOSS-Image's tokenizer, chat template and text-encoder config")


@lru_cache(maxsize=1)
def minimax_assets():
    """Tokenizer, Qwen3-VL processor and text-encoder config from MiniMax's release (small files)."""
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    def load(path):
        return (AutoTokenizer.from_pretrained(os.path.join(path, "FL2VA", "tokenizer")),
                AutoProcessor.from_pretrained(os.path.join(path, "FL2VA", "processor")),
                AutoConfig.from_pretrained(os.path.join(path, "FL2VA", "text_encoder")))
    return load_assets(MINIMAX_REPO, ["FL2VA/tokenizer/*", "FL2VA/processor/*", "FL2VA/text_encoder/config.json"],
                       load, "MiniMax H3's tokenizer, processor and text-encoder config")


@lru_cache(maxsize=1)
def qwen_image_2_assets():
    """Qwen3-VL processor, text-encoder config and VAE config from the Qwen-Image 2.1 release."""
    import json

    from transformers import AutoConfig, AutoProcessor

    def load(path):
        with open(os.path.join(path, "vae", "config.json")) as f:
            vae_config = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        return (AutoProcessor.from_pretrained(os.path.join(path, "processor")),
                AutoConfig.from_pretrained(os.path.join(path, "text_encoder")), vae_config)
    return load_assets(QWEN_IMAGE_2_REPO, ["processor/*", "text_encoder/config.json", "vae/config.json"], load,
                       "Qwen-Image 2.1's processor and configs")


def resolve_path(path: str) -> str:
    """Local file path, or ``owner/repo/filename`` on the Hugging Face hub (cached once downloaded)."""
    path = os.path.expanduser(path.strip())
    if os.path.exists(path):
        return path
    parts = path.split("/")
    if len(parts) >= 3 and not path.startswith("/"):
        repo, filename = "/".join(parts[:2]), "/".join(parts[2:])
        try:
            return hf_hub_download(repo, filename, local_files_only=True)
        except OSError:
            pass
        try:
            return hf_hub_download(repo, filename, token=os.getenv("HF_TOKEN"))
        except Exception as e:
            raise _download_failed(f"The file {filename}", repo, e) from e
    raise FileNotFoundError(path)

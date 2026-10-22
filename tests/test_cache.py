"""The encoding cache jobs share (``engine.cache``): which model files and settings make a cache folder, and
how model files are recognised.

    .venv/bin/python -m pytest tests/test_cache.py -v
"""

import json
import os
import shutil

import torch

from dainamo.engine import cache
from dainamo.engine.families import get_family


def _model(path, tensors=None, fill=0):
    """A small safetensors file: ``tensors`` {name: dtype}, filled from ``fill``."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(fill)
    save_file({k: torch.randn(512, 512, generator=g).to(dt) for k, dt in (tensors or {"w": torch.bfloat16}).items()},
              str(path))
    return str(path)


def test_a_model_file_is_known_by_its_contents(tmp_path):
    a = _model(tmp_path / "vae.safetensors")
    (tmp_path / "elsewhere").mkdir()
    copy = shutil.copy(a, tmp_path / "elsewhere" / "renamed.safetensors")
    assert cache.fingerprint(a) == cache.fingerprint(str(copy))  # a copy anywhere matches
    assert cache.fingerprint(a) != cache.fingerprint(_model(tmp_path / "tuned.safetensors", fill=1))  # other weights
    fp8 = _model(tmp_path / "fp8.safetensors", {"w": torch.float8_e4m3fn})
    assert cache.fingerprint(a) != cache.fingerprint(fp8)  # another precision
    assert cache.fingerprint(None) == cache.fingerprint("") == "none"


def test_jobs_share_what_the_same_files_encode(tmp_path):
    vae, te, te_fp8 = (_model(tmp_path / f"{n}.safetensors", fill=i) for i, n in enumerate(("vae", "te", "te_fp8")))
    dit, other_dit = _model(tmp_path / "dit.safetensors", fill=3), _model(tmp_path / "other.safetensors", fill=4)
    root = str(tmp_path / "cache")

    def folders(family, settings=None, **profile):
        d = get_family(family)(profile, torch.device("cpu"))
        return {kind: cache.folder(root, d, kind, settings or {}) for kind in ("latents", "text")}

    base = folders("krea2", vae=vae, text_encoder=te, diffusion_model=dit)
    # another checkpoint of the model: the same VAE and text encoder, so the same encodings
    assert folders("krea2", vae=vae, text_encoder=te, diffusion_model=other_dit) == base
    # another text encoder (or its fp8 copy) encodes captions differently, but images the same
    other = folders("krea2", vae=vae, text_encoder=te_fp8, diffusion_model=dit)
    assert other["latents"] == base["latents"] and other["text"] != base["text"]
    # another family with the same files doesn't share either
    moss = folders("moss_image", vae=vae, text_encoder=te, diffusion_model=dit)
    assert moss["latents"] != base["latents"] and moss["text"] != base["text"]
    # settings that change a payload: Qwen-Image's alpha channel
    q = folders("qwen_image_2", vae=vae, text_encoder=te, diffusion_model=dit)
    q_rgb = folders("qwen_image_2", {"keep_alpha": False}, vae=vae, text_encoder=te, diffusion_model=dit)
    assert q["latents"] != q_rgb["latents"] and q["text"] == q_rgb["text"]
    # LTX's captions run through the DiT's text connectors too; its clips' sound only when trained
    ltx = dict(vae=vae, audio_vae=vae, text_encoder=te)
    a, b = folders("ltx2_5", **ltx, diffusion_model=dit), folders("ltx2_5", **ltx, diffusion_model=other_dit)
    assert a["latents"] == b["latents"] and a["text"] != b["text"]
    assert folders("ltx2_5", {"train_audio": False}, **ltx, diffusion_model=dit)["latents"] != a["latents"]
    # YuE2's checkpoint holds the composer and the VAE
    y, y2 = folders("yue2", diffusion_model=dit), folders("yue2", diffusion_model=other_dit)
    assert y["latents"] != y2["latents"] and y["text"] != y2["text"]

    with open(os.path.join(base["latents"], "signature.json")) as f:
        sig = json.load(f)
    assert sig["family"] == "krea2" and sig["paths"] == [vae]  # says what made it

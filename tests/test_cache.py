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


def test_a_moved_dataset_keeps_its_cached_latents(tmp_path):
    """An item's cache name goes by its dataset and where the file is in it, so a dataset moved to another folder
    still finds its encodings, and another dataset of the same files has its own. One encoded before names went by
    dataset (by the file's full path) is found by the folder the dataset moved from."""
    from types import SimpleNamespace

    from PIL import Image

    from dainamo.engine import dataset as ds
    from dainamo.engine.trainer import Trainer

    old, new = tmp_path / "old", tmp_path / "new"
    (old / "sub" / "_masks").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(old / "a.png")
    Image.new("RGB", (64, 64)).save(old / "sub" / "b.png")
    Image.new("L", (64, 64), 255).save(old / "sub" / "_masks" / "b.png")

    def build(**d):
        return ds.build_items([d], 64, 16, False, log=lambda *_: None)
    items = build(id="d1", path=str(old))
    assert [bool(it.mask) for it in items] == [False, True]
    before = [it.key() for it in items]
    by_path = [it.key() for it in build(path=str(old))]  # (no id: by the full path, as all were before)
    assert by_path != before
    shutil.move(old, new)
    after = build(id="d1", path=str(new), moved_from=[str(old)])
    assert [it.key() for it in after] == before
    assert [it.keys()[-1] for it in after] == by_path
    assert not {it.key() for it in build(id="d2", path=str(new))} & {it.key() for it in after}

    # the trainer takes the encoding under the name the item had (so its saved batch order still applies), else
    # names it the new way
    folder = tmp_path / "cache"
    folder.mkdir()
    (folder / f"{by_path[0]}.pt").write_bytes(b"")
    (folder / f"{after[1].key()}.pt").write_bytes(b"")
    me = SimpleNamespace(cache_dir=lambda driver, kind: str(folder), use_cache=lambda files: None)
    files = Trainer.cache_items(me, None, after)
    assert files == [str(folder / f"{by_path[0]}.pt"), str(folder / f"{after[1].key()}.pt")]

"""Masked loss: an image's loss mask in its folder's _masks folder, found for images (not clips), cropped and mirrored
like the image, carried in its payload at the latent's size, and weighting its pixels in the loss.

    .venv/bin/python -m pytest tests/test_masks.py -v
"""

import hashlib
import os
import subprocess

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from dainamo.engine import dataset as ds
from dainamo.engine.families.base import FamilyDriver, FamilyInfo
from dainamo.engine.families.qwen_image_2.driver import QwenImage2Driver

CPU = torch.device("cpu")


class PoolFamily(FamilyDriver):
    """Images encode to their 16x16-pixel averages; the model is a 1x1 conv."""
    info = FamilyInfo(id="pool", label="Pool", media=["image"], profile_fields=[])

    def __init__(self):
        super().__init__({}, CPU)

    def encode_image(self, pixels):
        return F.avg_pool2d(pixels, 16)[0]

    def sample_timesteps(self, n, hw, settings):
        return torch.rand(n)

    def predict(self, model, noisy, t, text):
        return model(noisy)


def _image(path, size=(64, 64), colour=(120, 80, 40)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def _half_mask(path, size, left=True):
    """White over the left (or right) half, black over the other."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = size
    im = Image.new("L", size, 0)
    im.paste(255, (0, 0, w // 2, h) if left else (w // 2, 0, w, h))
    im.save(path)


def _clip(path, seconds=1):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x64:rate=24:d={seconds}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def test_a_mask_is_found_for_an_image_and_isnt_media(tmp_path):
    _image(str(tmp_path / "a.png"))
    _image(str(tmp_path / "b.jpg"))
    _half_mask(str(tmp_path / "_masks" / "a.png"), (64, 64))
    _half_mask(str(tmp_path / "_masks" / "stray.png"), (64, 64))
    assert ds.list_media(str(tmp_path)) == [str(tmp_path / "a.png"), str(tmp_path / "b.jpg")]
    assert ds.mask_path(str(tmp_path / "a.png")) == str(tmp_path / "_masks" / "a.png")
    assert ds.mask_path(str(tmp_path / "b.jpg")) is None
    lines = []
    items = ds.build_items([{"path": str(tmp_path), "repeats": 2}], 64, 16, True, log=lines.append)
    assert {(os.path.basename(it.path), os.path.basename(it.mask)) for it in items} == {("a.png", "a.png"),
                                                                                       ("b.jpg", "")}
    assert "  1 of 2 with masks in _masks" in lines
    # not asked for: no masks, and no line about them
    lines = []
    assert all(not it.mask for it in ds.build_items([{"path": str(tmp_path)}], 64, 16, False, log=lines.append,
                                                    masks=False))
    assert not any("masks" in line for line in lines)


def test_a_clip_takes_no_mask(tmp_path):
    _clip(tmp_path / "clip.mp4")
    _image(str(tmp_path / "still.png"))
    _half_mask(str(tmp_path / "_masks" / "clip.png"), (64, 64))
    _half_mask(str(tmp_path / "_masks" / "still.png"), (64, 64))
    lines = []
    items = ds.build_items([{"path": str(tmp_path)}], 64, 16, False, log=lines.append, video_frames=9)
    masks = {os.path.basename(it.path): it.mask for it in items}
    assert masks == {"clip.mp4": "", "still.png": str(tmp_path / "_masks" / "still.png")}
    assert "  1 of 1 images with masks in _masks" in lines


def test_editing_a_mask_changes_the_items_key(tmp_path):
    _image(str(tmp_path / "a.png"))
    plain = ds.build_items([{"path": str(tmp_path)}], 64, 16, False, log=lambda *_: None)[0]
    mask = str(tmp_path / "_masks" / "a.png")
    _half_mask(mask, (64, 64))
    item = ds.build_items([{"path": str(tmp_path)}], 64, 16, False, log=lambda *_: None)[0]
    assert item.mask == mask and item.key() != plain.key()  # adding a mask encodes the image again
    before = item.key()
    st = os.stat(mask)
    os.utime(mask, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert item.key() != before
    # an item without a mask keeps the key it had before masks, so its encoding is still used
    st = os.stat(plain.path)
    raw = f"{plain.path}|{st.st_size}|{st.st_mtime_ns}|{plain.size}|False|64"
    assert plain.key() == hashlib.sha1(raw.encode()).hexdigest()


@pytest.mark.parametrize("mask_size", [(128, 64), (40, 20), (50, 50)])
def test_a_mask_of_another_size_or_aspect_lines_up_with_the_crop(tmp_path, mask_size):
    """A 2:1 image fitted to a square keeps its middle half; a mask over the image's left half (at any size, even
    square: it's stretched to the image) covers the crop's left half, and its right half when mirrored."""
    _half_mask(str(tmp_path / "m.png"), mask_size)
    m = ds.load_mask(str(tmp_path / "m.png"), (128, 64), (32, 32))
    assert m.shape == (1, 32, 32) and m.dtype == torch.float32
    assert (m[..., :15] > 0.99).all() and (m[..., 17:] < 0.01).all()
    flipped = ds.load_mask(str(tmp_path / "m.png"), (128, 64), (32, 32), flip=True)
    assert torch.equal(flipped, m.flip(-1))


def test_a_tall_images_mask_is_cropped_top_and_bottom(tmp_path):
    # a 1:2 image fitted to a 2:1 bucket keeps rows 48-80 of its 128; a square mask white over its top half,
    # stretched to the image, covers the crop's top half
    im = Image.new("L", (100, 100), 0)
    im.paste(255, (0, 0, 100, 50))
    im.save(tmp_path / "m.png")
    m = ds.load_mask(str(tmp_path / "m.png"), (64, 128), (32, 16))
    assert m.shape == (1, 16, 32)
    assert (m[:, :7] > 0.99).all() and (m[:, 9:] < 0.01).all()


def test_the_payload_carries_the_mask_at_the_latents_size(tmp_path):
    _image(str(tmp_path / "masked.png"), (128, 128))
    _image(str(tmp_path / "plain.png"), (128, 128))
    _half_mask(str(tmp_path / "_masks" / "masked.png"), (64, 64))  # another size: stretched to the image
    items = {os.path.basename(it.path): it for it in
             ds.build_items([{"path": str(tmp_path)}], 128, 16, False, log=lambda *_: None)}
    d = PoolFamily()
    p = d.encode_item(items["masked.png"], {})
    assert p["latent"].shape[-2:] == p["mask"].shape and p["mask"].dtype == torch.float16
    w = p["mask"].shape[1]  # (stretching the mask blurs its edge a little)
    assert (p["mask"][:, : w // 2] > 0.95).all() and (p["mask"][:, w // 2:] < 0.05).all()
    assert "mask" not in d.encode_item(items["plain.png"], {})

    # Qwen-Image 2 encodes its own items: the same mask at its latent's size
    q = QwenImage2Driver({}, CPU)
    q.encode_image = lambda px: F.avg_pool2d(px, 16)[0]
    qp = q.encode_item(items["masked.png"], {})
    assert qp["mask"].shape == qp["latent"].shape[-2:] and torch.equal(qp["mask"], p["mask"])
    assert "mask" not in q.encode_item(items["plain.png"], {})


def test_only_image_families_train_with_masks():
    from dainamo.engine.families import FAMILIES
    takes = {name: cls({}, CPU).trains_masks for name, cls in FAMILIES.items()}
    assert {n for n, t in takes.items() if t} == {n for n, cls in FAMILIES.items() if cls.info.media == ["image"]}


# --- the loss ---------------------------------------------------------------------------------------------------
def _halves(b=1):
    """pred 0, target 1 on the left half and 3 on the right: per-pixel MSE 1 and 9, 5 overall."""
    target = torch.ones(b, 4, 6, 8)
    target[..., 4:] = 3.0
    mask = torch.zeros(6, 8, dtype=torch.float16)
    mask[:, :4] = 1
    return torch.zeros(b, 4, 6, 8), target, mask


def test_without_a_mask_the_loss_is_the_mse():
    d = PoolFamily()
    pred, target = torch.randn(3, 4, 6, 8), torch.randn(3, 4, 6, 8)
    t = torch.rand(3)
    mse = F.mse_loss(pred, target)
    assert torch.equal(d.weighted_loss(pred, target, t, {}, [{}, {}, {}]), mse)
    assert torch.equal(d.weighted_loss(pred, target, t, {"masked_loss": True}, [{}, {}, {}]), mse)
    # a mask that's 1 everywhere is the same loss
    ones = {"mask": torch.ones(6, 8, dtype=torch.float16)}
    assert torch.allclose(d.weighted_loss(pred, target, t, {"masked_loss": True}, [ones] * 3), mse)
    # with masked loss off, a payload's mask is ignored
    _, _, half = _halves()
    assert torch.equal(d.weighted_loss(pred, target, t, {}, [{"mask": half}] * 3), mse)


def test_a_half_mask_counts_its_half():
    d = PoolFamily()
    pred, target, mask = _halves()
    t = torch.full((1,), 0.5)
    payload = [{"mask": mask}]
    off = d.weighted_loss(pred, target, t, {"masked_loss": True, "mask_min_weight": 0.0}, payload)
    assert torch.isclose(off, torch.tensor(1.0))  # only the left half's MSE
    low = d.weighted_loss(pred, target, t, {"masked_loss": True, "mask_min_weight": 0.1}, payload)
    assert torch.isclose(low, torch.tensor((1 + 0.1 * 9) / 1.1))  # normalised by the weights' sum
    assert 1.0 < low.item() < 5.0  # between the masked half and the whole image
    assert torch.isclose(d.weighted_loss(pred, target, t, {"masked_loss": True, "mask_min_weight": 1.0}, payload),
                         torch.tensor(5.0))
    # in a batch, an item without a mask counts its whole image; the batch is the items' mean
    pred2, target2, _ = _halves(2)
    both = d.weighted_loss(pred2, target2, t.repeat(2), {"masked_loss": True, "mask_min_weight": 0.0},
                           [{"mask": mask}, {}])
    assert torch.isclose(both, torch.tensor((1.0 + 5.0) / 2))
    # an all-black mask with nothing outside it counted adds nothing, rather than NaN
    black = d.weighted_loss(pred, target, t, {"masked_loss": True, "mask_min_weight": 0.0},
                            [{"mask": torch.zeros(6, 8)}])
    assert black.item() == 0.0


def test_masks_count_in_training_loss_and_validation():
    d = PoolFamily()
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 3, 1)
    lat = torch.randn(3, 4, 6)
    mask = torch.zeros(4, 6, dtype=torch.float16)
    mask[:, :3] = 1

    def loss(settings, payload, sigma=0.5):
        g = torch.Generator().manual_seed(1)
        return d.training_loss(model, [payload], [{"text": None}], settings, sigma=sigma, generator=g).item()

    plain = loss({}, {"latent": lat})
    assert loss({}, {"latent": lat, "mask": mask}) == plain  # masked loss off
    masked = loss({"masked_loss": True, "mask_min_weight": 0.0}, {"latent": lat, "mask": mask})
    assert masked != plain
    # by hand: the left half's MSE at the same noise
    noise = torch.randn(1, 3, 4, 6, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        pred = model(0.5 * lat[None] + 0.5 * noise)
    err = (pred - (noise - lat[None])) ** 2
    assert masked == pytest.approx(err[..., :3].mean().item(), rel=1e-5)

"""Pixel upscalers (``engine.upscalers.pixel``): tiles that add up to a whole-image pass, overlaps blended without
seams, odd sizes, greyscale and alpha, fitting to another factor, fp16 overflow, cancelling between tiles, and
telling files apart. A stand-in for spandrel's model descriptor takes the model's place; with a GPU and
RealESRGAN_x4plus in the models folder, the real thing.

    .venv/bin/python -m pytest tests/test_upscale.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F

from dainamo.engine import upscale
from dainamo.engine.upscalers import pixel

CPU = torch.device("cpu")
REAL_ESRGAN = "/home/william/AI_APPS/COMFY_CACHE/models/upscale_models/RealESRGAN_x4plus.safetensors"


class Stub:
    """Stands in for spandrel's ImageModelDescriptor: its scale and flags, ``to``, and a call on (1, C, H, W)."""

    def __init__(self, fn=None, scale=2, supports_half=False, input_channels=3):
        self.fn = fn or (lambda x: F.interpolate(x, scale_factor=scale, mode="nearest"))
        self.scale, self.supports_half = scale, supports_half
        self.input_channels = self.output_channels = input_channels
        self.calls, self.dtype = [], torch.float32

    def to(self, device=None, dtype=None):
        self.dtype = dtype or self.dtype
        return self

    def __call__(self, x):
        self.calls.append(tuple(x.shape))
        return self.fn(x).clamp(0, 1)


def make(monkeypatch, stub, tile=pixel.TILE, overlap=pixel.OVERLAP, scale=None):
    monkeypatch.setattr(pixel, "_load", lambda path: stub)
    info = upscale.Info("/models/upscale_models/x.pth", "pixel", "x", scale=scale or stub.scale)
    return pixel.PixelUpscaler(info, CPU, log=lambda *_: None, tile=tile, overlap=overlap)


def blur_up(x):
    """A 3x3 box blur with zero padding, then 2x nearest: a tile's edge pixels come out darker than in a whole-image
    pass, as a real model's edges come out off for want of what's around them."""
    k = torch.full((x.shape[1], 1, 3, 3), 1 / 9, dtype=x.dtype)
    return F.interpolate(F.conv2d(x, k, padding=1, groups=x.shape[1]), scale_factor=2, mode="nearest")


def test_spans_cover_the_side_evenly_in_as_few_tiles_as_fit():
    for n, tile, overlap in ((1024, 512, 32), (1000, 512, 32), (513, 512, 32), (37, 16, 4), (4096, 512, 32),
                             (100, 30, 29)):
        tiles = pixel.spans(n, tile, overlap)
        assert tiles[0][0] == 0 and tiles[-1][1] == n
        assert all(b - a == tiles[0][1] - tiles[0][0] <= tile for a, b in tiles)
        ov = min(overlap, tile // 2)
        assert all(a1 < b0 and b0 - a1 >= ov - 1 for (_, b0), (a1, _) in zip(tiles, tiles[1:]))
        assert len(tiles) == -(-(n - ov) // (tile - ov))  # no more than it takes
    assert pixel.spans(300, 512, 32) == [(0, 300)]
    assert pixel.spans(1024, 512, 32) == [(0, 363), (330, 693), (661, 1024)]  # three tiles of 363, not 512s
    assert all((b - a) % 16 == 0 for a, b in pixel.spans(1000, 512, 32, multiple=16))  # what the model pads to


def test_tiles_give_what_a_whole_image_pass_gives(monkeypatch):
    x = torch.rand(1, 3, 203, 301)  # odd sizes, several tiles each way
    up = make(monkeypatch, Stub(), tile=64, overlap=16)
    tiled = up.upscale(x, (602, 406))
    assert len(up.model.calls) == 4 * 6 and all(h <= 64 and w <= 64 for _, _, h, w in up.model.calls)
    up.tile = 10_000
    whole = up.upscale(x, (602, 406))
    assert len(up.model.calls) == 25 and tiled.shape == whole.shape == (1, 3, 406, 602)
    assert torch.allclose(tiled, whole, atol=1e-6)
    assert torch.equal(whole, F.interpolate(x, scale_factor=2, mode="nearest"))


def test_overlaps_blend_without_seams(monkeypatch):
    x = torch.rand(1, 3, 160, 200)
    up = make(monkeypatch, Stub(blur_up), tile=64, overlap=16)
    up.tile = 10_000
    whole = up.upscale(x, (400, 320))
    up.tile, up.overlap = 64, 0  # tiles side by side: their edges show
    seams = (up.upscale(x, (400, 320)) - whole).abs()
    assert seams.max() > 0.1
    up.overlap = 16
    blended = (up.upscale(x, (400, 320)) - whole).abs()
    # only a tile's outermost pixels are off, and they weigh next to nothing beside the neighbour's
    assert blended.max() < min(0.04, seams.max() / 5) and blended.mean() < 1e-3


def test_odd_and_tiny_sizes(monkeypatch):
    up = make(monkeypatch, Stub(blur_up), tile=16, overlap=4)
    for h, w in ((1, 1), (17, 3), (23, 37), (16, 17)):
        x = torch.rand(2, 3, h, w)
        out = up.upscale(x, (2 * w, 2 * h))
        assert out.shape == (2, 3, 2 * h, 2 * w) and torch.isfinite(out).all()
        assert 0 <= out.min() and out.max() <= 1


def test_alpha_and_greyscale_carry_through(monkeypatch):
    up = make(monkeypatch, Stub(), tile=32, overlap=8)
    rgb = torch.rand(1, 3, 40, 50)
    alpha = torch.linspace(0, 1, 50).expand(1, 1, 40, 50)
    out = up.upscale(torch.cat([rgb, alpha], 1), (100, 80))
    assert out.shape == (1, 4, 80, 100)
    assert torch.allclose(out[:, :3], up.upscale(rgb, (100, 80)), atol=1e-6)  # the model saw the colour alone
    assert torch.allclose(out[:, 3:], upscale.fit(alpha, (100, 80)))  # alpha resized beside it
    assert all(c == 3 for _, c, _, _ in up.model.calls)  # always RGB in
    grey = torch.rand(1, 1, 40, 50)
    out = up.upscale(grey, (100, 80))
    assert out.shape == (1, 1, 80, 100) and torch.allclose(out, F.interpolate(grey, scale_factor=2), atol=1e-6)
    out = up.upscale(torch.cat([grey, alpha], 1), (100, 80))  # grey with alpha
    assert out.shape == (1, 2, 80, 100) and torch.allclose(out[:, 1:], upscale.fit(alpha, (100, 80)))


def test_a_greyscale_model_takes_each_channel(monkeypatch):
    up = make(monkeypatch, Stub(input_channels=1), tile=32, overlap=8)
    x = torch.rand(1, 3, 20, 30)
    assert torch.allclose(up.upscale(x, (60, 40)), F.interpolate(x, scale_factor=2), atol=1e-6)
    assert all(c == 1 for _, c, _, _ in up.model.calls)


def test_a_fixed_scale_is_fitted_to_the_size_asked(monkeypatch):
    up = make(monkeypatch, Stub(scale=4))
    assert up.scale == 4
    x = torch.rand(3, 3, 30, 40)
    seen = []
    out = up.upscale(x, (80, 60), progress=seen.append)  # 2x of a 4x model
    assert out.shape == (3, 3, 60, 80)
    assert torch.allclose(out, upscale.fit(F.interpolate(x, scale_factor=4), (80, 60)), atol=1e-6)
    assert seen == ["Frame 1 of 3", "Frame 2 of 3", "Frame 3 of 3"]  # a clip's frames
    assert torch.equal(up.upscale(x[:1], (160, 120)), F.interpolate(x[:1], scale_factor=4))  # its own scale: as is
    assert up.upscale(x[:1], (123, 77)).shape == (1, 3, 77, 123)  # any size, aspect and all


def test_fp16_that_overflows_runs_in_fp32(monkeypatch):
    stub = Stub(lambda x: torch.full_like(F.interpolate(x, scale_factor=2), float("nan"))
                if x.dtype == torch.float16 else F.interpolate(x, scale_factor=2))
    up = make(monkeypatch, stub)
    up.dtype = torch.float16  # as on a GPU, for an architecture that says it takes fp16
    x = torch.rand(1, 3, 8, 8)
    out = up.upscale(x, (16, 16))
    assert up.dtype == stub.dtype == torch.float32 and torch.allclose(out, F.interpolate(x, scale_factor=2))


def test_check_cancels_between_tiles(monkeypatch):
    class Stop(Exception):
        pass

    up = make(monkeypatch, Stub(), tile=16, overlap=4)
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 3:
            raise Stop

    with pytest.raises(Stop):
        up.upscale(torch.rand(1, 3, 64, 64), (128, 128), check=check)
    assert len(up.model.calls) == 2  # the frame's check, then two tiles; the third tile's check stopped it


def test_identify_reads_a_file_once_while_it_is_unchanged(monkeypatch, tmp_path):
    from spandrel import UnsupportedModelError

    path = tmp_path / "4x_thing.pth"
    path.write_bytes(b"x" * 100)
    loads = []

    def load(p):
        loads.append(p)
        return Stub(scale=4)

    monkeypatch.setattr(pixel, "_load", load)
    monkeypatch.setattr(pixel, "_identified", {})
    info = pixel.identify(str(path), "4x_thing")
    assert (info.kind, info.scale, info.video, info.error) == ("pixel", 4, False, None)
    assert pixel.identify(str(path), "4x_thing") == info and len(loads) == 1
    os.utime(path, ns=(1, 1))  # changed: read again
    monkeypatch.setattr(pixel, "_load", lambda p: (_ for _ in ()).throw(UnsupportedModelError()))
    info = pixel.identify(str(path), "4x_thing")
    assert info.kind == "unknown" and "no architecture spandrel supports" in info.error
    path.write_bytes(b"y" * 101)
    monkeypatch.setattr(pixel, "_load", lambda p: (_ for _ in ()).throw(ValueError("it's a LaMa inpainting model")))
    assert "inpainting" in pixel.identify(str(path), "4x_thing").error
    monkeypatch.setattr(pixel, "MAX_BYTES", 100)  # too big to be one: not loaded to find out
    monkeypatch.setattr(pixel, "_load", lambda p: pytest.fail("loaded a file too big to be a pixel upscaler"))
    path.write_bytes(b"z" * 102)
    assert "too big" in pixel.identify(str(path), "4x_thing").error
    assert "Couldn't read it" in pixel.identify(str(tmp_path / "gone.pth"), "gone").error


def test_a_real_file_spandrel_does_not_know_is_unknown(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "not_an_upscaler.safetensors"
    save_file({"some.weight": torch.zeros(4, 4)}, str(path))
    info = upscale.identify(str(path))
    assert info.kind == "unknown" and info.error and info.scale == 0


@pytest.mark.skipif(not (torch.cuda.is_available() and os.path.exists(REAL_ESRGAN)),
                    reason="needs a GPU and RealESRGAN_x4plus in the models folder")
def test_real_esrgan_on_the_gpu():
    info = upscale.identify(REAL_ESRGAN)
    assert (info.kind, info.scale) == ("pixel", 4)
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    up = upscale.load(info, dev, log=lambda *_: None)
    assert up.dtype == torch.float16
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 96), torch.linspace(0, 1, 128), indexing="ij")
    x = torch.stack([xx, yy, ((xx * 8).floor() + (yy * 8).floor()) % 2 * 0.6 + 0.2])[None]  # gradients, checks
    out = up.upscale(x, (512, 384))
    assert out.shape == (1, 3, 384, 512) and torch.isfinite(out).all()
    assert (upscale.fit(out, (128, 96)) - x).abs().mean() < 0.05  # the same picture, sharper
    up.tile, up.overlap = 48, 16
    assert (up.upscale(x, (512, 384)) - out).abs().mean() < 0.01  # tiled, much the same
    up.unload()

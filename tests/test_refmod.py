"""RefMods: the file format (ComfyUI-MiniMaxH3Mod's), the latent maths, making one from images and clips (or their
sound) with a stand-in VAE, generating with them and with reference audio (ref2va), and a RefMod job end to end.

    .venv/bin/python -m pytest tests/test_refmod.py -v
"""

import json
import os
import subprocess

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from dainamo.engine import refmod
from dainamo.engine.families.minimax_h3 import packing as P
from dainamo.engine.families.minimax_h3.driver import MinimaxH3Driver
from dainamo.engine.families.minimax_h3.text_encoder import AudioRef, encode_minimax_h3_prompt
from dainamo.engine.trainer import cond_key

CPU = torch.device("cpu")


def _meta(**kw):
    return {"name": "x", "kind": "image", "latent_h": 4, "latent_w": 4, "latent_t": 1, "mode": "encode", **kw}


# --- files ------------------------------------------------------------------
def test_a_saved_refmod_reads_back(tmp_path):
    z = torch.randn(1, 24, 3, 4, 6)
    meta = refmod.describe("fox", z, "compressed", "stack", "1x8x12 +2x8x12", "3x4x6", 50, ["1 img, 1 vid"],
                           "a red fox", "identity")
    path = refmod.save(str(tmp_path / "fox.safetensors"), z, meta)
    assert refmod.read_meta(path) == meta
    want = {"kind": "video", "latent_t": 3, "latent_h": 4, "latent_w": 6, "mode": "training", "_format_version": 4,
            "concept_type": "identity", "optimize_steps": 50, "description": "a red fox"}
    assert {k: meta[k] for k in want} == want
    [ref] = refmod.load(path)
    assert ref["name"] == "fox" and ref["kind"] == "video" and ref["latent"].dtype == torch.float32
    assert torch.allclose(ref["latent"], z, atol=1e-2)  # stored as fp16
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_older_files_and_bundles_load(tmp_path):
    # format 2 (the pack's example files): "pooled" mode, fewer fields
    old = tmp_path / "old.safetensors"
    save_file({"latent": torch.ones(1, 24, 1, 4, 4, dtype=torch.float16)}, str(old),
              metadata={"refmod_meta": json.dumps(_meta(mode="pooled", _format_version=2))})
    assert refmod.load(str(old))[0]["latent"].shape == (1, 24, 1, 4, 4)
    # a bundle (format 5) with a voice in it: read, but only the picture is a reference here
    bundle = tmp_path / "bundle.safetensors"
    members = [_meta(name="face"), {"name": "voice", "kind": "audio"}, _meta(name="walk", kind="video", latent_t=2)]
    save_file({"ref_0": torch.zeros(1, 24, 1, 4, 4), "ref_1": torch.zeros(1, 32, 2, 50),
               "ref_2": torch.zeros(1, 24, 2, 4, 4)}, str(bundle),
              metadata={"refmod_meta": json.dumps({"kind": "bundle", "members": members, "_format_version": 5})})
    assert [r["kind"] for r in refmod.load(str(bundle))] == ["image", "audio", "video"]
    assert [r["name"] for r in refmod.visual(str(bundle))] == ["face", "walk"]
    assert refmod.member_kinds(refmod.read_meta(str(bundle))) == ["image", "audio", "video"]
    # a voice saved by the older H3AudioMod
    voice = tmp_path / "voice.safetensors"
    save_file({"latent": torch.zeros(1, 32, 2, 50)}, str(voice),
              metadata={"audio_refmod_meta": json.dumps({"name": "voice", "kind": "audio"})})
    assert [(r["kind"], tuple(r["latent"].shape)) for r in refmod.load(str(voice))] == [("audio", (1, 32, 2, 50))]


def test_an_audio_refmod_reads_back_as_the_pack_writes_one(tmp_path):
    z = torch.randn(1, 32, 2, 120)
    meta = refmod.describe_audio("her voice", z, ["1 sound(s), 3.0 s"], "a warm alto", "voice")
    path = refmod.save(str(tmp_path / "voice.safetensors"), z, meta)
    assert refmod.read_meta(path) == meta
    want = {"kind": "audio", "latent_h": 0, "latent_w": 0, "latent_t": 120, "mode": "encode", "source": "audio",
            "_format_version": 4, "sample_rate": 32000, "concept_type": "voice", "description": "a warm alto"}
    assert {k: meta[k] for k in want} == want
    [ref] = refmod.load(path)
    assert ref["kind"] == "audio" and torch.allclose(ref["latent"], z, atol=1e-2)
    assert refmod.tokens(z) == 240  # 2 per latent, as the pack counts them
    # the model's rows: channel 0's latents, then channel 1's (pack_audio_latents of (1, 2, 32, T))
    assert torch.equal(refmod.audio_rows(z), P.pack_audio_latents(z.permute(0, 2, 1, 3))[0])
    bad = tmp_path / "bad.safetensors"
    save_file({"latent": torch.zeros(1, 16, 2, 5)}, str(bad), metadata={"refmod_meta": json.dumps(meta)})
    with pytest.raises(ValueError, match="unexpected latent shape"):
        refmod.load(str(bad))


def test_files_that_arent_refmods_are_refused(tmp_path):
    lora = tmp_path / "lora.safetensors"
    save_file({"w": torch.zeros(2)}, str(lora))
    assert refmod.read_meta(str(lora)) is None
    with pytest.raises(ValueError, match="isn't a RefMod"):
        refmod.load(str(lora))
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert refmod.read_meta(str(junk)) is None
    wrong = tmp_path / "wrong.safetensors"
    save_file({"latent": torch.zeros(1, 16, 1, 4, 4)}, str(wrong), metadata={"refmod_meta": json.dumps(_meta())})
    with pytest.raises(ValueError, match="unexpected latent shape"):
        refmod.load(str(wrong))
    missing = tmp_path / "missing.safetensors"
    save_file({"other": torch.zeros(1)}, str(missing), metadata={"refmod_meta": json.dumps(_meta())})
    with pytest.raises(ValueError, match="missing the latent"):
        refmod.load(str(missing))


# --- latents ----------------------------------------------------------------
def test_sizes_and_grids():
    assert refmod.ref_size(2048, 1536, 1024) == (1376, 1024)  # the short edge to 1024, both /32
    assert refmod.ref_size(512, 384, 1024) == (512, 384)  # never enlarged
    assert refmod.ref_size(40, 20, 1024) == (32, 32)
    assert refmod.aspect_grid(16, 1024, 1024) == (16, 16)
    assert refmod.aspect_grid(16, 768, 1024) == (16, 12)  # portrait: the height is the long edge
    assert refmod.aspect_grid(16, 1200, 900) == (12, 16)
    assert refmod.aspect_grid(4, 4000, 100) == (2, 4)  # at least one 2x2 patch
    assert refmod.tokens(torch.zeros(1, 24, 3, 8, 12)) == 3 * 4 * 6


def test_refining_a_grid_matches_the_full_encode_better_than_averaging():
    torch.manual_seed(0)
    full = F.interpolate(torch.randn(1, 24, 1, 8, 8), size=(1, 32, 32), mode="trilinear") + 0.1 * torch.randn(1, 24, 1, 32, 32)
    small = refmod.pool(full, 1, 8, 8)
    assert small.shape == (1, 24, 1, 8, 8) and refmod.pool(full, 1, 32, 32) is full

    def err(z):
        return F.mse_loss(F.interpolate(z, size=(1, 32, 32), mode="trilinear", align_corners=False), full).item()

    refined = refmod.refine(small, [full], 100)
    assert refined.shape == small.shape and not refined.requires_grad
    assert err(refined) < 0.8 * err(small)
    assert refmod.refine(small, [full], 0) is small


def test_the_token_cap_drops_duplicates_then_spreads_the_rest():
    torch.manual_seed(1)
    a, b, c = (torch.randn(1, 24, 1, 4, 4) for _ in range(3))
    z = torch.cat([a, a + 1e-4, a, b, c, c], dim=2)  # 6 frames, 3 distinct, 4 tokens each
    assert refmod.distinct_frames(z) == [0, 3, 4]
    logs = []
    assert refmod.fit_tokens(z, 0) is z and refmod.fit_tokens(z, 24) is z  # no cap / under it
    capped = refmod.fit_tokens(z, 12, logs.append)
    assert torch.equal(capped, torch.cat([a, b, c], dim=2)) and "near-duplicate" in logs[0]
    two = refmod.fit_tokens(z, 8, logs.append)  # 3 distinct frames, room for 2: the first and the last
    assert torch.equal(two, torch.cat([a, c], dim=2)) and "spread evenly" in logs[-1]
    with pytest.raises(ValueError, match="more than the cap"):
        refmod.fit_tokens(z, 3)


def test_weakening_blends_toward_a_blur():
    torch.manual_seed(2)
    z = torch.randn(1, 24, 2, 16, 16)
    assert refmod.weaken(z, 1.0) is z and refmod.weaken(z, 7) is z
    blur = refmod.weaken(z, 0.0)
    half = refmod.weaken(z, 0.5)
    assert blur.shape == z.shape and blur.std() < 0.5 * z.std()  # the detail is gone
    assert torch.allclose(half, (z + blur) / 2, atol=1e-6)
    assert torch.allclose(blur.mean(), z.mean(), atol=1e-2)  # it's still the same picture on average
    # a sound: blurred along time
    a = torch.randn(1, 32, 2, 64)
    assert refmod.weaken(a, 1.0) is a and refmod.weaken(a[..., :1], 0.2).shape == (1, 32, 2, 1)
    smooth = refmod.weaken(a, 0.0)
    assert smooth.shape == a.shape and smooth.diff(dim=-1).abs().mean() < 0.2 * a.diff(dim=-1).abs().mean()
    assert torch.allclose(refmod.weaken(a, 0.25), 0.25 * a + 0.75 * smooth, atol=1e-6)


# --- making one -------------------------------------------------------------
class FakeVAE:
    """Encodes like the H3 VAE in shape: 16x16 pixels to a cell, 17n+5 frames to 5n+2, 24 channels (the RGB
    averages, repeated)."""
    fps = 24
    snap_frames = staticmethod(P.align_num_frames_down)

    def __init__(self):
        self.seen = []

    def encode_reference(self, px):
        self.seen.append(tuple(px.shape))
        t = 1 if px.shape[2] == 1 else P.video_latent_num_frames(px.shape[2])
        return F.adaptive_avg_pool3d(px, (t, px.shape[3] // 16, px.shape[4] // 16)).repeat(1, 8, 1, 1, 1)


def _media(tmp_path):
    d = tmp_path / "refs"
    d.mkdir()
    Image.new("RGB", (512, 384), (200, 40, 40)).save(d / "a.png")
    Image.new("RGB", (300, 400), (40, 200, 40)).save(d / "b.png")
    (d / "_refs").mkdir()
    Image.new("RGB", (64, 64)).save(d / "_refs" / "a.png")  # a training reference, not one of these
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x256:rate=24:duration=1.5",
                    "-pix_fmt", "yuv420p", str(d / "c.mp4")], check=True)
    return str(d)


SETTINGS = {"refmod_mode": "compressed", "refmod_size": 256, "refmod_frames": 22, "refmod_grid": 8,
            "refmod_refine": 5, "refmod_merge": False, "refmod_max_tokens": 0, "refmod_concept": "identity",
            "refmod_description": " a test "}


def test_making_one_from_images_and_a_clip(tmp_path):
    paths = refmod.sources([{"path": _media(tmp_path)}])
    assert [os.path.basename(p) for p in paths] == ["a.png", "b.png", "c.mp4"]
    vae, logs = FakeVAE(), []
    z, meta = refmod.make(vae, paths, SETTINGS, log=logs.append)
    # each encoded at its own size (short edge 256, never enlarged), the clip's first 22 frames
    assert vae.seen == [(1, 3, 1, 256, 352), (1, 3, 1, 352, 256), (1, 3, 22, 256, 320)]
    # every grid the first one's shape (512x384: 8 cells wide, 6 high), stacked: 1 + 1 + 7 latent frames
    assert z.shape == (1, 24, 9, 6, 8)
    assert meta["kind"] == "video" and meta["mode"] == "training" and meta["source"] == "stack"
    assert meta["tags"] == ["2 img, 1 vid"] and meta["pool"] == "9x6x8" and meta["optimize_steps"] == 5
    assert meta["concept_type"] == "identity" and meta["description"] == "a test"
    assert meta["source_shape"] == "1x16x22 +1x22x16 +7x16x20" and len(logs) == 4

    # full: all of them cropped to the first one's canvas, as they are
    vae = FakeVAE()
    z, meta = refmod.make(vae, paths, {**SETTINGS, "refmod_mode": "full"}, log=lambda m: None)
    assert {s[3:] for s in vae.seen} == {(256, 352)} and z.shape == (1, 24, 9, 16, 22)
    assert meta["mode"] == "encode" and meta["optimize_steps"] == 0 and meta["pool"].startswith("full-res 352x256px")

    # merged: one grid for all of them, at the clip's length
    z, meta = refmod.make(FakeVAE(), paths, {**SETTINGS, "refmod_merge": True}, log=lambda m: None)
    assert z.shape == (1, 24, 7, 6, 8) and meta["source"] == "merge 3 refs" and "merged 3 refs" in meta["tags"]

    # over the cap: the clip's frames thinned out
    z, meta = refmod.make(FakeVAE(), paths, {**SETTINGS, "refmod_max_tokens": 5 * 12}, log=lambda m: None)
    assert refmod.tokens(z) == 60 and "capped at 60 tokens" in meta["tags"]

    # one image: an image RefMod
    z, meta = refmod.make(FakeVAE(), paths[:1], SETTINGS, log=lambda m: None)
    assert z.shape == (1, 24, 1, 6, 8) and meta["kind"] == "image" and meta["source"] == "image"


class FakeAudioVAE:
    """Encodes like the H3 audio VAE in shape: (B, 1, samples) -> (B, 32, ceil(samples / 800)), each latent its
    800 samples' loudness (mean absolute value, repeated over the channels)."""
    def __init__(self):
        self.seen = []

    def encode(self, wav):
        self.seen.append(tuple(wav.shape))
        wav = F.pad(wav, (0, -wav.shape[-1] % 800))
        return wav.reshape(wav.shape[0], 1, -1, 800).abs().mean(-1).repeat(1, 32, 1)

    def decode(self, z):  # (B, 32, T) -> (B, 1, T * 800)
        return z[:, :1].repeat_interleave(800, dim=-1)


def _sounds(tmp_path):
    d = tmp_path / "sounds"
    d.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "44100",
                    str(d / "a.wav")], check=True)  # mono, and not at 32 kHz
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=1", "-f", "lavfi",
                    "-i", "sine=frequency=220:duration=1", "-pix_fmt", "yuv420p", "-shortest", str(d / "b.mp4")],
                   check=True)  # a clip with sound
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=1",
                    "-pix_fmt", "yuv420p", str(d / "c.mp4")], check=True)  # a silent one
    Image.new("RGB", (64, 64)).save(d / "d.png")
    return str(d)


def test_making_an_audio_one_from_sound_files_and_soundtracks(tmp_path, monkeypatch):
    paths = refmod.audio_sources([{"path": _sounds(tmp_path)}])
    assert [os.path.basename(p) for p in paths] == ["a.wav", "b.mp4"]  # not the silent clip or the image
    drv = type("Drv", (), {"audio_vae": FakeAudioVAE(), "device": CPU})()
    logs = []
    s = {"refmod_seconds": 2.5, "refmod_max_tokens": 0, "refmod_concept": "singing", "refmod_description": ""}
    z, meta = refmod.make_audio(drv, paths, s, log=logs.append)
    # all 2 s of the first, then half a second of the clip's soundtrack, each channel encoded on its own
    assert drv.audio_vae.seen == [(2, 1, 64000), (2, 1, 16000)] and z.shape == (1, 32, 2, 100)
    # the mono sine in both channels at its own level (ffmpeg's -ac 2 would take it down 3 dB, to 0.056)
    assert torch.allclose(z[..., :80], torch.full_like(z[..., :80], 0.125 * 2 / torch.pi), atol=3e-3)
    assert meta["kind"] == "audio" and meta["latent_t"] == 100 and meta["concept_type"] == "singing"
    assert meta["tags"] == ["2 sound(s), 2.5 s"] and "b.mp4: 0.5 s of 1.0" in logs[-1]
    # encoded 10 s at a time (here 1 s): the same latents
    monkeypatch.setattr(refmod, "AUDIO_CHUNK", 1)
    drv.audio_vae.seen = []
    chunked, _ = refmod.make_audio(drv, paths, s, log=lambda m: None)
    assert drv.audio_vae.seen[:2] == [(2, 1, 32000), (2, 1, 32000)] and torch.allclose(chunked, z)
    # over the cap: the end is cut
    z, meta = refmod.make_audio(drv, paths, {**s, "refmod_max_tokens": 120}, log=logs.append)
    assert z.shape[-1] == 60 and "capped at 120 tokens" in meta["tags"] and "kept the first 1.5 s" in logs[-1]
    with pytest.raises(RuntimeError, match="has any sound"):
        refmod.make_audio(drv, [os.path.join(os.path.dirname(paths[0]), "c.mp4")], s, log=lambda m: None)


def test_a_short_clip_is_its_first_frame(tmp_path):
    clip = tmp_path / "blink.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=0.125",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    vae = FakeVAE()
    z, _ = refmod.make(vae, [str(clip)], {**SETTINGS, "refmod_mode": "full"}, log=lambda m: None)
    assert vae.seen == [(1, 3, 1, 64, 64)] and z.shape == (1, 24, 1, 4, 4)


# --- generating with them ---------------------------------------------------
class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kw):
        self.calls.append({k: v.detach().clone() for k, v in kw.items()})
        return torch.zeros_like(kw["hidden_states"]), torch.zeros_like(kw["audio_hidden_states"])


def text_cond(n=7):
    return {"embeds": torch.randn(n, 5120).bfloat16(), "tags": torch.ones(n, dtype=torch.long)}


def test_ref2va_takes_refmods_as_reference_rows(tmp_path):
    a = refmod.save(str(tmp_path / "a.safetensors"), torch.randn(1, 24, 1, 4, 6), _meta(latent_w=6))
    b = refmod.save(str(tmp_path / "b.safetensors"), torch.randn(1, 24, 3, 2, 2), _meta(kind="video", latent_t=3))
    d = MinimaxH3Driver({"variant": "ref2va"}, CPU)
    plain, m = Recorder(), Recorder()
    d.sample(plain, text_cond(), None, 96, 64, 2, 0, 7, frames=22)
    d.sample(m, text_cond(), None, 96, 64, 2, 0, 7, frames=22,
             refmods=[{"path": a, "strength": 1.0}, {"path": b, "strength": 0.5}])
    video = plain.calls[0]["hidden_states"].shape[1]
    assert video == 7 * 2 * 3
    # the references' tokens go in ahead of the video's: a's 1x2x3 and b's 3x1x1
    assert m.calls[0]["hidden_states"].shape[1] == video + 6 + 3
    with pytest.raises(ValueError, match="ref2va"):
        MinimaxH3Driver({"variant": "fl2va"}, CPU).sample(Recorder(), text_cond(), None, 96, 64, 2, 0, 7, frames=22,
                                                         refmods=[{"path": a}])


def test_a_standalone_audio_reference_packs_as_the_released_pipeline_does():
    tags = torch.ones(5, dtype=torch.long)
    layout = P.build_packed_sequence(tags, num_latent_frames=7, latent_height=4, latent_width=6, num_audio_latents=37,
                                     ref_blocks=((1, 4, 6, 0), (0, 0, 0, 10)))
    image_rows, ref_audio, target_audio = 1 * 2 * 3, 2 * 10, 2 * 37
    assert layout.num_condition_video_rows == image_rows and layout.num_condition_audio_rows == ref_audio
    rows = layout.audio_indices[:ref_audio]
    assert torch.equal(rows, torch.arange(5 + image_rows, 5 + image_rows + ref_audio))
    pos = layout.position_ids[rows]
    # after the image's block (clock 5 -> 6), one rotary unit per latent, both channels
    assert torch.equal(pos[:, 0], (6.0 + torch.arange(10, dtype=torch.float64)).repeat(2))
    # no height; the width pinned to the TARGET grid's extremes, as the target's own audio is
    target = layout.position_ids[layout.audio_indices[ref_audio:]]
    assert torch.equal(pos[:, 1], torch.zeros(ref_audio, dtype=torch.float64))
    assert pos[0, 2] == target[0, 2] and pos[-1, 2] == target[-1, 2] and pos[0, 2] != pos[-1, 2]
    # the target starts after the audio's 10 units
    assert target[0, 0] == 16.0 and layout.audio_indices.shape[0] == ref_audio + target_audio
    assert (layout.token_tags[rows] == P.AUDIO_TAG).all()
    row_t = P.build_row_timesteps(layout, 0.3, 0.5)
    assert (row_t[rows] == 1.0).all() and (row_t[layout.audio_indices[ref_audio:]] == 0.5).all()


class _Tok:
    """A tokenizer of characters (and a processor and text encoder around it), for the presentation alone."""
    def __init__(self):
        self.texts = []

    def __call__(self, text, add_special_tokens=False):
        self.texts.append(text)
        return {"input_ids": [ord(c) for c in text]}

    def convert_tokens_to_ids(self, tok):
        return {"<|vision_start|>": 1, "<|vision_end|>": 2, "<|image_pad|>": 3, "<|video_pad|>": 4}[tok]


def test_audio_references_are_labelled_in_the_prompt():
    tok = _Tok()
    processor = type("Proc", (), {"image_processor": type("IP", (), {"merge_size": 2})(),
                                  "create_mm_token_type_ids": staticmethod(lambda ids: [[0] * len(ids[0])])})()
    seen = {}

    def model(input_ids, **kw):
        seen["ids"] = input_ids[0].tolist()
        return type("Out", (), {"hidden_states": [torch.zeros(1, input_ids.shape[1], 8)] * 51})()

    cfg = type("Cfg", (), {"text_config": type("TC", (), {"num_hidden_layers": 50})()})()
    te = type("TE", (), {"config": cfg, "device": CPU, "dtype": torch.float32, "model": staticmethod(model)})()
    embeds, tags = encode_minimax_h3_prompt(te, tok, processor, "a song", keyframes=[AudioRef(), AudioRef()])
    assert tok.texts == ["<Audio 1>: ", "<Audio 2>: ", "a song"]
    assert "".join(map(chr, seen["ids"])) == "<Audio 1>: <Audio 2>: a song"
    assert (tags == P.TEXT_TAG).all() and embeds.shape[0] == len(seen["ids"])
    # the cached encodings tell them apart
    assert len({cond_key("a song", None), cond_key("a song", [AudioRef()]), cond_key("a song", [AudioRef()] * 2)}) == 3


def test_ref2va_takes_audio_references_as_clean_rows(tmp_path):
    face = refmod.save(str(tmp_path / "face.safetensors"), torch.randn(1, 24, 1, 4, 6), _meta(latent_w=6))
    long_voice = torch.randn(1, 32, 2, 200)  # longer than the video: cut to its length
    voice = refmod.save(str(tmp_path / "voice.safetensors"), long_voice,
                        refmod.describe_audio("voice", long_voice, []))
    wav = tmp_path / "hum.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=0.5", str(wav)], check=True)
    d = MinimaxH3Driver({"variant": "ref2va"}, CPU)
    d.audio_vae = FakeAudioVAE()
    spec = {"prompt": "p", "frames": 22, "refmods": [{"path": face}, {"path": voice}], "audio": [str(wav)]}
    assert d.sample_condition(spec) == ("p", [AudioRef(), AudioRef()])  # one label per sound, no pictures
    assert d.sample_condition({**spec, "frames": 1}) == ("p", None)  # an image takes no sound
    m = Recorder()
    out = d.sample(m, text_cond(), None, 96, 64, 2, 0, 7, frames=22,
                   refmods=[{"path": face}, {"path": voice, "strength": 1.0}], audio=[str(wav)])
    a_lat = P.audio_latent_num_frames(22)  # 37, as is the longest a reference may be
    first, second = m.calls
    cond = 2 * a_lat + 2 * 20  # the voice cut to 37 latents, the hum's 0.5 s is 20
    assert first["audio_hidden_states"].shape[1] == cond + 2 * a_lat
    assert torch.allclose(first["audio_hidden_states"][0, : 2 * a_lat].float(),
                          refmod.audio_rows(long_voice[..., :a_lat]), atol=1e-2)
    # clean (t = 1) at every step, and left as they are: only the generated audio is stepped
    rows = first["audio_indices"][:cond]
    assert (first["row_timesteps"][0, rows] == 1.0).all() and (second["row_timesteps"][0, rows] == 1.0).all()
    assert torch.equal(first["audio_hidden_states"][:, :cond], second["audio_hidden_states"][:, :cond])
    assert (first["row_timesteps"][0, first["audio_indices"][cond:]] < 1.0).all()
    assert out["audio"].shape == (1, 2 * a_lat, 32)
    # a sound reference needs a picture one beside it
    with pytest.raises(ValueError, match="needs an image or a visual RefMod"):
        d.sample(Recorder(), text_cond(), None, 96, 64, 2, 0, 7, frames=22, audio=[str(wav)])
    # an image (one frame) leaves the sound out
    still = Recorder()
    d.sample(still, text_cond(), None, 96, 64, 2, 0, 7, frames=1, refmods=[{"path": face}, {"path": voice}])
    assert still.calls[0]["audio_hidden_states"].shape[1] == 2 * P.audio_latent_num_frames(1)


# --- a RefMod job -----------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer needs a GPU")
def test_a_refmod_job_saves_copies_and_previews_it(tmp_path, monkeypatch):
    from dainamo.engine import trainer as trainer_mod
    from dainamo.engine.families.base import FamilyDriver, FamilyInfo

    class TinyRef(FakeVAE, FamilyDriver):
        info = FamilyInfo(id="tinyref", label="Tiny", media=["image", "video"], profile_fields=[], refmods=True)
        loaded = None

        def __init__(self, profile, device, log=print):
            FamilyDriver.__init__(self, profile, device, log=log)
            FakeVAE.__init__(self)

        def load_vae(self, audio=True, video=True):
            TinyRef.loaded = (audio, video)

        def to_media(self, latents):
            assert latents["frames"] == 1 and latents["video"].shape[2] == 1
            return [Image.new("RGB", (16, 16))]

    monkeypatch.setattr(trainer_mod, "get_family", lambda fid: TinyRef)
    media = _media(tmp_path)
    copy_to = tmp_path / "comfy" / "models" / "refmods"
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({
        "name": "fox ref", "family": "tinyref", "profile": {}, "datasets": [{"path": media}],
        "settings": {**SETTINGS, "refmod_enabled": True, "refmod_copy_to": str(copy_to), "steps": 1}}))
    t = trainer_mod.Trainer(str(job_dir))
    assert t.run_refmod() == "completed"
    assert TinyRef.loaded == (False, True)  # only the video VAE
    made = t.status["refmod"]
    assert made["file"] == "fox_ref.safetensors" and (made["frames"], made["height"], made["width"]) == (9, 6, 8)
    assert made["references"] == 3 and made["tokens"] == 9 * 12 and made["copied_to"] == str(copy_to)
    out = job_dir / "output" / "fox_ref.safetensors"
    assert refmod.read_meta(str(out))["name"] == "fox_ref" and (copy_to / "fox_ref.safetensors").exists()
    assert len(made["preview"]) == 9 and (job_dir / "samples" / "refmod_08.png").exists()
    assert t.status["checkpoints"] == [{"step": 0, "file": "fox_ref.safetensors", "final": True}]
    assert t.status["phase"] == "completed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer needs a GPU")
def test_an_audio_refmod_job_saves_it_and_its_sound(tmp_path, monkeypatch):
    from dainamo.engine import trainer as trainer_mod
    from dainamo.engine.families.base import FamilyDriver, FamilyInfo

    class TinySound(FamilyDriver):
        info = FamilyInfo(id="tinysound", label="Tiny", media=["image", "video"], profile_fields=[], refmods=True)
        loaded = None

        def load_vae(self, audio=True, video=True):
            TinySound.loaded = (audio, video)
            self.audio_vae = FakeAudioVAE()

    monkeypatch.setattr(trainer_mod, "get_family", lambda fid: TinySound)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({
        "name": "her voice", "family": "tinysound", "profile": {}, "datasets": [{"path": _sounds(tmp_path)}],
        "settings": {**SETTINGS, "refmod_enabled": True, "refmod_kind": "audio", "refmod_seconds": 2.5,
                     "refmod_concept": "voice", "refmod_copy_to": "", "steps": 1}}))
    t = trainer_mod.Trainer(str(job_dir))
    assert t.run_refmod() == "completed"
    assert TinySound.loaded == (True, False)  # only the audio VAE
    made = t.status["refmod"]
    assert made["kind"] == "audio" and made["seconds"] == 2.5 and made["tokens"] == 200 and made["references"] == 2
    assert made["preview"] == [] and made["audio"] == {"file": "refmod_audio.mp3", "poster": "refmod_audio_poster.png",
                                                      "kind": "audio"}
    assert (job_dir / "samples" / "refmod_audio.mp3").exists() and (job_dir / "samples" / "refmod_audio_poster.png").exists()
    meta = refmod.read_meta(str(job_dir / "output" / "her_voice.safetensors"))
    assert meta["name"] == "her_voice" and meta["kind"] == "audio" and meta["latent_t"] == 100

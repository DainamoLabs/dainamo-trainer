"""YuE2 (music): song clips and captions with lyrics, the settings and sample cards of music jobs, the acoustic
model's padding and sampler on a one-layer random model, LoRA keys ComfyUI loads, saved songs, the health
check on audio, and (with the real checkpoint) the prompt format, the VAE and the VRAM projection.

    .venv/bin/python -m pytest tests/test_yue2.py -v
"""

import json
import os
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import HTTPException

from dainamo.engine import dataset as ds
from dainamo.engine import memory
from dainamo.engine.families.yue2 import model as M
from dainamo.engine.families.yue2.driver import LORA_STRENGTH, YuE2Driver, parse_condition, song_condition
from dainamo.engine.lora import LoRANetwork
from dainamo.engine.media import save_media
from dainamo.engine.trainer import Trainer, bucket_summary, id_of
from dainamo.server import dataset_check
from dainamo.server.validation import check_settings

CKPT = os.path.join(os.environ.get("DAINAMO_TEST_MODELS", os.path.expanduser("~/AI_APPS/COMFY_CACHE/models/diffusion_models")),
                    "yue2_3b_int8_convrot.safetensors")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
needs_ckpt = pytest.mark.skipif(not os.path.exists(CKPT), reason="the YuE2 checkpoint isn't on this machine")


def _tone(path, seconds, rate=48000, channels=2, freq=440):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f={freq}:d={seconds}:sample_rate={rate}",
                    "-ac", str(channels), str(path)], check=True)


# --- datasets -----------------------------------------------------------------------------------------------
def test_lyrics_split_from_captions():
    assert ds.split_lyrics("rock, male vocals\n[Lyrics]\n[Verse]\nHello\nworld") == \
        ("rock, male vocals", "[Verse]\nHello\nworld")
    assert ds.split_lyrics("jazz\n  [LYRICS]  \nla la") == ("jazz", "la la")  # any case, surrounding spaces
    assert ds.split_lyrics("ambient, [lyrics] in a tag") == ("ambient, [lyrics] in a tag", "")  # only on its own line


def test_song_clips_spread_over_the_song():
    assert ds.song_clips(29, 30) == [(0.0, 29.0)]  # shorter than a clip: one shorter clip
    assert ds.song_clips(39, 20) == [(0.0, 20.0), (19.0, 20.0)]  # 39.99 s songs keep their end
    assert ds.song_clips(50, 20) == [(0.0, 20.0), (15.0, 20.0), (30.0, 20.0)]
    starts = [s for s, _ in ds.song_clips(200, 30)]
    assert len(starts) == 7 and starts[0] == 0 and starts[-1] == 170


def test_audio_items(tmp_path):
    _tone(tmp_path / "long.wav", 65)
    _tone(tmp_path / "short.flac", 12)
    _tone(tmp_path / "tiny.mp3", 3)
    (tmp_path / "long.txt").write_text("rock, guitar\n[Lyrics]\n[Verse]\nla la")
    (tmp_path / "short.txt").write_text("jazz")
    from PIL import Image
    Image.new("RGB", (64, 64)).save(tmp_path / "cover.png")  # not audio: skipped
    logs = []
    sets = [{"path": str(tmp_path), "trigger": "ohwx", "repeats": 2}]
    items = ds.build_audio_items(sets, 30, flip=True, log=logs.append)
    long = [it for it in items if it.path.endswith("long.wav")]
    assert sorted({(it.start, it.seconds) for it in long}) == [(0.0, 30.0), (35.0, 30.0)]
    assert len(long) == 2 * 2 * 2  # clips x flips x repeats
    assert all(it.caption == "ohwx, rock, guitar" and it.lyrics == "" for it in long)  # clips leave lyrics out
    assert {(it.seconds, it.caption) for it in items if it.path.endswith("short.flac")} == {(12.0, "ohwx, jazz")}
    assert not any(it.path.endswith("tiny.mp3") for it in items) and any("too short" in m for m in logs)
    assert all(it.is_audio and not it.is_video for it in items)
    whole = ds.build_audio_items(sets, 0, flip=False)
    assert {(os.path.basename(it.path), it.seconds, it.lyrics) for it in whole} == \
        {("long.wav", 65.0, "[Verse]\nla la"), ("short.flac", 12.0, "")}
    # a clip's cache key is its own; batches keep one length
    assert len({it.key() for it in long}) == 4
    batches = ds.make_batches(items, 4, random.Random(0))
    assert all(len({items[i].seconds for i in b}) == 1 for b in batches)
    assert bucket_summary(items)[0] == "12 training items from 2 songs, 8 of 30 s, 4 of 12 s"
    # a song trains once per epoch (per flip and repeat), as one of its clips picked at random
    rng = random.Random(0)
    epochs = [[items[i] for i in ds.epoch_items(items, rng)] for _ in range(20)]
    assert all(len(e) == 8 and sum(it.path.endswith("long.wav") for it in e) == 4 for e in epochs)
    assert {it.start for e in epochs for it in e if it.path.endswith("long.wav")} == {0.0, 35.0}
    assert logs[-1].endswith("2 songs, 2 repeats -> 12 items, 8 per epoch (a clip of each song)")
    assert not any(it.group for it in whole)  # whole songs are one item each


def test_reading_a_clip(tmp_path):
    _tone(tmp_path / "a.wav", 6, rate=44100, channels=1)
    wav = ds.read_audio(str(tmp_path / "a.wav"), 2.0, M.SAMPLE_RATE, 2, start=3.0)
    assert wav.shape == (2, 2 * M.SAMPLE_RATE) and torch.equal(wav[0], wav[1])  # mono becomes both channels
    assert ds.probe_audio(str(tmp_path / "a.wav")) == {"duration": pytest.approx(6, abs=0.05), "sample_rate": 44100,
                                                       "channels": 1}


# --- settings, sample cards and conditions --------------------------------------------------------------
def test_music_job_settings():
    s = check_settings({**YuE2Driver.info.training_defaults, "resolution": 1024, "sample_prompts": [
        {"prompt": "ohwx, rock", "lyrics": " [Verse]\nhi ", "seconds": 20, "width": 999},
        {"prompt": "ohwx, jazz"}]}, 16, audio=True)
    assert "resolution" not in s and (s["clip_seconds"], s["sample_seconds"]) == (30, 30)
    assert s["sample_prompts"] == [{"prompt": "ohwx, rock", "seconds": 20, "lyrics": "[Verse]\nhi"}, {"prompt": "ohwx, jazz"}]
    for bad in ({"clip_seconds": 3}, {"clip_seconds": 400}, {"sample_seconds": 0}, {"bleed_enabled": True, "bleed_prompts": ["x"]},
                {"sample_prompts": [{"prompt": "x", "seconds": 1000}]}):
        with pytest.raises(HTTPException):
            check_settings({**YuE2Driver.info.training_defaults, **bad}, 16, audio=True)
    assert check_settings({**YuE2Driver.info.training_defaults, "clip_seconds": 0}, 16, audio=True)["clip_seconds"] == 0


def test_sample_specs_and_conditions():
    t = Trainer.__new__(Trainer)
    t.driver = YuE2Driver({}, torch.device("cpu"))
    t.slider = None
    t.s = {"sample_seconds": 40, "sample_seed": 5, "sample_prompts": [
        {"prompt": "ohwx, rock", "lyrics": "[Verse]\nhi", "seconds": 20}, {"prompt": "ohwx, jazz", "lora_scale": 1}]}
    a, b = t.sample_specs()
    assert (a["seconds"], a["lyrics"], a["lora_scale"], a["seed"]) == (20, "[Verse]\nhi", LORA_STRENGTH, 5)
    assert (b["seconds"], b["lyrics"], b["lora_scale"], b["seed"]) == (40, "", 1, 6)
    assert id_of(a) != id_of(dict(a, lyrics="other")) != id_of(dict(a, seconds=21))
    prompt, images = t.driver.sample_condition(a)
    assert images is None and parse_condition(prompt) == {"style": "ohwx, rock", "lyrics": "[Verse]\nhi", "seconds": 20,
                                                          "sample": True}
    assert t.driver.encode_condition(prompt, None)["sample"]  # sample cards aren't encoded until they're sampled
    assert parse_condition("") == {"style": "", "lyrics": ""} and parse_condition("{oops") == {"style": "{oops", "lyrics": ""}
    item = ds.Item("x.wav", "ohwx, rock", (0, 0), lyrics="la", seconds=30.0)
    assert t.driver.item_conditions(item, {}) == {"text": (song_condition("ohwx, rock", "la"), None)}
    assert YuE2Driver.info.to_json()["lora_strength"] == LORA_STRENGTH
    assert YuE2Driver.info.to_json()["streams_text_encoder"] is False  # the composer isn't streamed in low VRAM mode
    assert YuE2Driver({"sample_cot": "full"}, torch.device("cpu")).cfg_scale(None) == 1.0
    assert YuE2Driver({}, torch.device("cpu")).cfg_scale(None) == 1.01  # planning off: the release's guidance


def test_another_checkpoint_loads_its_own_composer_and_vae(tmp_path, monkeypatch):
    """The Sandbox can point the driver at another checkpoint (``Session.use_model``): the tokenizer, composer and
    VAE come from it then, not from the one loaded before; from the same one they're kept."""
    from safetensors.torch import save_file

    from dainamo.engine.families.yue2 import driver as yd

    files = []
    for name in ("a", "b"):
        path = str(tmp_path / f"{name}.safetensors")
        save_file({"text_encoders.yue2_tokenizer_json": torch.tensor(list(name.encode()), dtype=torch.uint8)}, path)
        files.append(path)
    loads = []

    def load_state_dict(path, strip_prefix="", **kw):
        loads.append((os.path.basename(path), strip_prefix))
        return SimpleNamespace(tensors={})

    class Part(torch.nn.Module):  # the composer's and the VAE's stand-in
        def load_state_dict(self, sd, assign=False):
            pass

    monkeypatch.setattr(yd.comfy_io, "load_state_dict", load_state_dict)
    monkeypatch.setattr(yd, "materialize", lambda *a, **kw: None)
    monkeypatch.setattr(yd, "assert_materialized", lambda m: None)
    for k, v in (("TextTokenizer", lambda blob: blob), ("Composer", Part), ("VAE", Part), ("fold_weight_norm", dict)):
        monkeypatch.setattr(M, k, v)
    d = YuE2Driver({"diffusion_model": files[0]}, torch.device("cpu"), log=lambda *_: None)
    d.load_text_encoder()
    d.load_vae()
    composer, vae = d.text_encoder, d.vae
    d.load_text_encoder()
    d.load_vae()
    assert d.text_encoder is composer and d.vae is vae and d._tokenizer() == "a"
    assert loads == [("a.safetensors", "text_encoders."), ("a.safetensors", "vae.")]
    d.profile["diffusion_model"] = files[1]  # as use_model does
    d.load_text_encoder()
    d.load_vae()
    assert d.text_encoder is not composer and d.vae is not vae and d._tokenizer() == "b"
    assert loads[2:] == [("b.safetensors", "text_encoders."), ("b.safetensors", "vae.")]


# --- the acoustic model -----------------------------------------------------------------------------------
def _nar():
    torch.manual_seed(0)
    model = M.NAR(layers=1).to(DEV, torch.bfloat16)
    model.latent_pos_embed.pe.normal_(0, 0.02)  # filled from the checkpoint in real use
    return model.eval()


def test_padded_conditions_match_unpadded():
    nar = _nar()
    torch.manual_seed(1)
    short = torch.randn(1, 1, 5, M.KV_HEADS, M.HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    long = torch.randn(1, 1, 9, M.KV_HEADS, M.HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    x, t = torch.randn(2, 6, M.LATENT_DIM, device=DEV), torch.tensor([0.3, 0.7], device=DEV)
    with torch.no_grad():
        alone = nar(x[:1], t[:1], short, short, torch.tensor([5], device=DEV))
        pad = torch.nn.functional.pad(short, (0, 0, 0, 0, 0, 4))
        both = nar(x, t, torch.cat([pad, long], 1), torch.cat([pad, long], 1), torch.tensor([5, 9], device=DEV))
        other = nar(x[1:], t[1:], long, long, torch.tensor([9], device=DEV))
    assert both.shape == (2, 6, M.LATENT_DIM)
    assert torch.allclose(both[:1], alone, atol=2e-2) and torch.allclose(both[1:], other, atol=2e-2)


def test_training_loss_trains_the_lora():
    nar, d = _nar(), YuE2Driver({}, DEV)
    net = LoRANetwork(nar, d.lora_targets, 4, 4)
    net.to(DEV, torch.float32)
    assert len(net.modules) == 4  # qkv, o, gate_up and down of the one layer
    kv = torch.randn(1, 7, M.KV_HEADS, M.HEAD_DIM, dtype=torch.bfloat16)
    conds = [{"text": {"k": kv, "v": kv}}, {"text": {"k": kv[:, :4], "v": kv[:, :4]}}]
    payloads = [{"latent": torch.randn(10, M.LATENT_DIM)} for _ in range(2)]
    nar.train()
    loss = d.training_loss(nar, payloads, conds, {"timestep_sampling": "uniform"})
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    sd = net.state_dict(torch.float32)
    assert "diffusion_model.model.layers.0.self_attn.qkv_proj.lora_A.weight" in sd  # ComfyUI's native YuE2 keys
    assert "diffusion_model.model.layers.0.mlp.gate_up_proj.lora_B.weight" in sd


def test_sampler_and_time():
    noise = torch.randn(4, 3)
    target = torch.randn(4, 3)
    # a constant velocity noise - x0 is integrated exactly from t = 1 to 0
    out = M.solve(lambda x, t: noise - target, noise, steps=5)
    assert torch.allclose(out, target, atol=1e-5)
    calls = []
    M.solve(lambda x, t: torch.zeros_like(x), noise, steps=3, callback=lambda s, n: calls.append((s, n)))
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert M.model_time(torch.tensor([0.5, 1e-12, 1.0])).tolist() == pytest.approx([0.5, 0.0, 1.0], abs=1e-6)


def test_weight_norm_folding():
    conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 4, 5))
    g = conv.parametrizations.weight.original0.detach()
    v = conv.parametrizations.weight.original1.detach()
    folded = M.fold_weight_norm({"c.weight_g": g, "c.weight_v": v, "c.bias": conv.bias.detach()})
    assert set(folded) == {"c.weight", "c.bias"} and torch.allclose(folded["c.weight"], conv.weight.detach(), atol=1e-6)


# --- output and the health check ---------------------------------------------------------------------------
def test_saved_song(tmp_path):
    t = np.linspace(0, 2, 2 * M.SAMPLE_RATE, dtype=np.float32)
    wav = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]) * 0.3
    saved = save_media({"audio": wav, "sample_rate": M.SAMPLE_RATE, "score": "X:1\nK:C\nCDEF|"}, str(tmp_path), "s")
    assert saved == {"file": "s.mp3", "poster": "s_poster.png", "kind": "audio", "score": "s.abc"}
    assert ds.probe_audio(str(tmp_path / "s.mp3"))["channels"] == 2
    assert (tmp_path / "s.abc").read_text() == "X:1\nK:C\nCDEF|"
    from PIL import Image
    assert Image.open(tmp_path / "s_poster.png").size == (1024, 256)
    assert "score" not in save_media({"audio": wav, "sample_rate": M.SAMPLE_RATE}, str(tmp_path), "n")


def test_health_check_on_songs(tmp_path):
    _tone(tmp_path / "ok.wav", 40)
    _tone(tmp_path / "mono.wav", 20, channels=1)
    _tone(tmp_path / "phone.wav", 20, rate=16000)
    _tone(tmp_path / "blip.wav", 2)
    (tmp_path / "ok.txt").write_text("rock\n[Lyrics]\nla la")
    for f in ("mono", "phone", "blip"):
        (tmp_path / f"{f}.txt").write_text("rock")
    from PIL import Image
    Image.new("RGB", (64, 64)).save(tmp_path / "cover.png")
    r = dataset_check.check(str(tmp_path), "ohwx", clip_seconds=30)
    ids = {i["id"]: i for i in r["issues"]}
    assert r["audio"] == 4 and r["images"] == 1
    assert {"song_too_short", "low_sample_rate", "mono", "song_short_clip", "lyrics_unused", "visual_skipped"} <= set(ids)
    assert [f["name"] for f in ids["song_too_short"]["files"]] == ["blip.wav"]
    assert "small" not in ids and "duplicates" not in ids
    assert "lyrics_unused" not in {i["id"] for i in dataset_check.check(str(tmp_path), clip_seconds=0)["issues"]}
    # an image model skips the audio files, and says so
    assert "audio_skipped" in {i["id"] for i in dataset_check.check(str(tmp_path), resolution=[512])["issues"]}


# --- with the real checkpoint ----------------------------------------------------------------------------------
@needs_ckpt
def test_prompt_format_and_projection():
    d = YuE2Driver({"diffusion_model": CKPT}, torch.device("cpu"))
    tok = d._tokenizer()
    assert tok.decode(tok.encode("Pop, upbeat\n[Verse]")) == "Pop, upbeat\n[Verse]"
    off = M.prefix_ids(tok, "off", "pop", "la")
    assert off[0] == M.EOD and off[-3:] == [M.ABC_START, M.ABC_END, M.MUSIC_START]
    assert tok.decode(off[1:-3]) == f"{M.INSTRUCTIONS['off']}\n[Tags]\npop\n[Lyrics]\nla\n"
    assert d.training_prefix("pop", "la") == off + [M.MUSIC_END]
    full = M.prefix_ids(tok, "full", "pop", "la", [100, 200])
    assert full[-5:] == [M.ABC_START, 100, 200, M.ABC_END, M.MUSIC_START]
    assert M.negative_ids(tok, "off")[-1] == M.MUSIC_START
    assert YuE2Driver.validate_profile({"diffusion_model": CKPT})[0]["level"] == "ok"
    proj = memory.project(YuE2Driver, {"diffusion_model": CKPT}, {"clip_seconds": 30, "sample_seconds": 30, "rank": 32})
    phases = {p["id"]: dict(p["parts"]) for p in proj["phases"]}
    assert proj["blocks"] == 28 and set(phases) == {"training", "sampling", "decoding", "latents", "captions"}
    assert 3.5 < phases["sampling"]["Composer"] < 4.5 and 0.3 < phases["latents"]["VAE"] < 0.7
    whole = memory.project(YuE2Driver, {"diffusion_model": CKPT}, {"clip_seconds": 0, "rank": 32})
    assert dict(whole["phases"][0]["parts"])["Activations"] > 5 * phases["training"]["Activations"]


@needs_ckpt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the VAE check runs on the GPU")
def test_vae_keeps_a_chords_envelope(tmp_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=6", "-f", "lavfi", "-i", "sine=f=330:d=6",
                    "-filter_complex", "amix=inputs=2,volume=0.8,pan=stereo|c0=c0|c1=c0", "-ar", "48000",
                    str(tmp_path / "chord.wav")], check=True)
    d = YuE2Driver({"diffusion_model": CKPT}, DEV)
    d.load_vae()
    x = ds.read_audio(str(tmp_path / "chord.wav"), 6, M.SAMPLE_RATE, 2)
    with torch.no_grad():
        z = d.vae.encode(x)
        y = d.vae.decode(z)
    d.unload_vae()
    assert z.shape == (6 * M.FRAME_RATE, M.LATENT_DIM)

    def env(a):
        m = a.mean(0)[: a.shape[1] // 480 * 480].reshape(-1, 480)
        return 20 * np.log10(np.sqrt((m ** 2).mean(1)) + 1e-9)

    a, b = env(x.numpy()[:, : y.shape[1]]), env(y.numpy())
    assert np.abs(a[10:-10] - b[10:-10]).max() < 1.5  # the loudness, 10 ms at a time, within 1.5 dB

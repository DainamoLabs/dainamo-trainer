"""LTX 2.5: the driver's glue around diffusers' LTX-2 classes, on tiny random models (no downloads).

The key mappings and the mel filterbank are also checked against the real ComfyUI files when they're in
``DAINAMO_TEST_MODELS``'s sibling folders (only their headers and two small tensors are read).

    .venv/bin/python -m pytest tests/test_ltx2.py -v
"""

import json
import os
import random
import struct

import pytest
import torch

from dainamo.engine.families.ltx2 import model as M
from dainamo.engine.families.ltx2.driver import LTX25Driver

# the folder holding diffusion_models/, text_encoders/ and vae/ (DAINAMO_TEST_MODELS is its diffusion_models/)
MODELS = os.path.dirname(os.environ.get("DAINAMO_TEST_MODELS",
                                        os.path.expanduser("~/AI_APPS/COMFY_CACHE/models/diffusion_models")))
DIT = os.path.join(MODELS, "diffusion_models", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
TE = os.path.join(MODELS, "text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors")
VAE = os.path.join(MODELS, "vae", "ltx-2.5-video-vae-bf16.safetensors")
AUDIO = os.path.join(MODELS, "vae", "ltx-2.5-audio-vae-bf16.safetensors")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_frames_schedules_and_audio_length():
    assert [M.snap_frames(f) for f in (1, 8, 9, 48, 49, 97, 100)] == [1, 1, 9, 41, 49, 97, 97]
    assert M.latent_frames(49) == 7 and M.latent_frames(1) == 1
    assert M.audio_latent_frames(49, 24) == 51  # 25 audio latents a second, as the pipeline rounds them
    s = M.shifted_sigmas(30, 4096)
    assert len(s) == 31 and s[0] == pytest.approx(1.0) and s[-2] == pytest.approx(M.SHIFT_TERMINAL) and s[-1] == 0
    assert all(a > b for a, b in zip(s, s[1:]))
    d = LTX25Driver({"variant": "distilled"}, torch.device("cpu"))
    assert d._schedule("default", 8, 4096, None) == M.DISTILLED_SIGMAS + [0.0]
    assert (d.sample_steps(), d.cfg_scale(None)) == (8, 1.0)
    dev_model = LTX25Driver({"variant": "dev"}, torch.device("cpu"))
    assert (dev_model.sample_steps(), dev_model.cfg_scale(None), dev_model.cfg_scale("2.5")) == (30, 4.0, 2.5)
    assert dev_model._schedule("default", 8, 4096, None) == M.shifted_sigmas(8, 4096)
    # 768x512, 49 frames: 24x16 latents over 7 frames, and 51 audio latents
    assert LTX25Driver.sequence_tokens(768, 512, 49) == 24 * 16 * 7 + 51


def test_neighborhood_attention_matches_diffusers_mask():
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import _neighborhood_block_mask
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    for (t, h, w), kernel, tile in [((5, 9, 20), (3, 7, 7), (2, 3, 8)), ((6, 10, 11), (11, 11, 11), (1, 16, 32)),
                                    ((1, 6, 7), (11, 11, 11), (1, 4, 4))]:
        q, k, v = (torch.randn(1, t, h, w, 2, 16, device=DEV) for _ in range(3))
        flat = lambda x: x.reshape(1, -1, 2, 16).transpose(1, 2)  # noqa: E731
        ref = flex_attention(flat(q), flat(k), flat(v), block_mask=_neighborhood_block_mask(t, h, w, kernel, DEV),
                             scale=1.0).transpose(1, 2).reshape(q.shape)
        assert torch.allclose(M.neighborhood_attention(q, k, v, kernel, tile), ref, atol=1e-5)


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


@pytest.mark.skipif(not all(os.path.exists(p) for p in (DIT, TE, VAE, AUDIO)), reason="needs the LTX 2.5 files")
def test_key_mappings_cover_the_real_files():
    """Every tensor of the ComfyUI files lands on exactly the parameters of diffusers' modules."""
    from dainamo.engine.comfy_io import SIDECAR_SUFFIXES

    dit = {k.removeprefix("model.diffusion_model."): v["shape"] for k, v in _header(DIT).items()
           if not k.endswith(SIDECAR_SUFFIXES)}
    model = M.build_transformer(dit)
    assert {M.transformer_key(k) for k in dit if not k.startswith(M.CONNECTOR_PARTS)} == set(model.state_dict())
    te = _header(TE)
    got = {M.connector_key(k) for k in list(dit) + list(te)
           if k.startswith(M.CONNECTOR_PARTS) or k.startswith("text_embedding_projection.")}
    assert got == set(M.build_connectors(3840, 48).state_dict())
    vae = _header(VAE)
    assert {M.encoder_key(k[8:]) for k in vae if k.startswith("encoder.")} == set(M.build_encoder().state_dict())
    fake = {k[8:]: torch.empty(v["shape"], device="meta") for k, v in vae.items() if k.startswith("decoder.")}
    fake.pop("type_emb", None)
    assert {"decoder." + k for k in M.decoder_state(fake)} | {"latents_mean", "latents_std"} == \
        set(M.build_diffusion_decoder().state_dict())
    audio = _header(AUDIO)
    assert {M.audio_vae_key(k[10:]) for k in audio if k.startswith("audio_vae.")} == \
        set(M.build_audio_vae().state_dict())
    assert {M.vocoder_key(k[8:]) for k in audio if k.startswith("vocoder.")} == set(M.build_vocoder().state_dict())


@pytest.mark.skipif(not os.path.exists(AUDIO), reason="needs the LTX 2.5 audio VAE file")
def test_mel_filters_reproduce_the_vocoders_own():
    """The mel filterbank (and so the audio VAE's input) uses the conventions of LTX's stored one."""
    from safetensors import safe_open

    with safe_open(AUDIO, "pt") as f:
        stored = f.get_tensor("vocoder.mel_stft.mel_basis").float()  # n_fft 512 at 16 kHz
        window = f.get_tensor("vocoder.mel_stft.stft_fn.forward_basis")[0, 0].float()  # the k = 0 row: the window
    assert torch.equal(M.mel_filters(16000, 512, 64, 0.0, 8000.0).to(torch.bfloat16).float(), stored)
    assert torch.allclose(torch.hann_window(512, periodic=True), window, atol=4e-3)


def test_log_mel_frames():
    wav = torch.randn(1, 2, 16000)  # a second at 16 kHz: 100 frames, frame i ends at sample 160 (i + 1)
    mel = M.log_mel(wav)
    assert mel.shape == (1, 2, 100, 64)
    late = wav.clone()
    late[..., 8000:] = 0  # changing the second half leaves the first half's frames alone (causal)
    assert torch.equal(M.log_mel(late)[:, :, :50], mel[:, :, :50])


def tiny_driver():
    """A driver with a 2-block random LTX-2.5 DiT (real channel counts at the edges, tiny inside)."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    torch.manual_seed(0)
    cfg = M.transformer_config({"transformer_blocks.1.x": None, "prompt_adaln_single.linear.weight": None})
    cfg.update(num_attention_heads=2, attention_head_dim=16, cross_attention_dim=32, audio_num_attention_heads=2,
               audio_attention_head_dim=8, audio_cross_attention_dim=16)
    model = LTX2VideoTransformer3DModel(**cfg).to(DEV)
    d = LTX25Driver({"variant": "distilled"}, DEV)
    return d, model


def cond(seed=0, tokens=M.TEXT_TOKENS):
    g = torch.Generator().manual_seed(seed)
    return {"video": torch.randn(tokens, 32, generator=g), "audio": torch.randn(tokens, 16, generator=g)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (bf16 autocast)")
def test_training_loss_branches():
    d, model = tiny_driver()
    t, h, w = 3, 2, 3  # 17 frames at 96x64
    video = {"video": torch.randn(128, t, h, w), "frames": 17}
    with_audio = dict(video, audio=torch.randn(M.audio_latent_frames(17, 24), 128))
    image = {"video": torch.randn(128, 1, h, w), "frames": 1}
    settings = {"i2v_probability": 0.0, "train_audio": True}
    for payload in (video, with_audio, image):
        loss = d.training_loss(model, [payload, payload], [{"text": cond(0)}, {"text": cond(1)}], settings,
                               rng=random.Random(0))
        assert torch.isfinite(loss)
        loss.backward()

    # validation: the same noise and sigma give the same loss
    def val():
        return d.training_loss(model, [with_audio], [{"text": cond()}], settings, sigma=0.4,
                               generator=torch.Generator().manual_seed(3), deterministic=True).item()
    assert val() == val()

    # image-to-video: the first latent frame goes in clean, at timestep 0, and isn't in the loss
    seen = {}
    real = d._predict

    def spy(model_, video_, audio_, conds, video_t, *rest):
        seen.update(video=video_.clone(), video_t=video_t.clone())
        return real(model_, video_, audio_, conds, video_t, *rest)
    d._predict = spy
    d.training_loss(model, [video], [{"text": cond()}], {"i2v_probability": 1.0}, rng=random.Random(0))
    first = video["video"][:, :1].flatten(1).T.to(DEV)
    assert torch.equal(seen["video"][0, : h * w], first)
    assert (seen["video_t"][0, : h * w] == 0).all() and (seen["video_t"][0, h * w:] > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (bf16 autocast)")
def test_sampling_keeps_the_first_frame_and_returns_audio():
    d, model = tiny_driver()
    model.eval()
    first = torch.randn(1, 128, 1, 2, 3, device=DEV)
    d._first_frame_latent = lambda path, w, h: first
    d.video_vae = object()  # loaded, as far as sample() checks
    out = d.sample(model, cond(0), cond(1), 96, 64, 4, 3.0, seed=5, frames=17, images=["first.png"])
    assert out["video"].shape == (1, 128, 3, 2, 3) and out["audio"].shape == (M.audio_latent_frames(17, 24), 128)
    assert torch.equal(out["video"][:, :, :1], first.cpu())
    again = d.sample(model, cond(0), cond(1), 96, 64, 4, 3.0, seed=5, frames=17, images=["first.png"])
    assert torch.equal(out["video"], again["video"])
    # each sampler and schedule runs (distilled: its 8-step schedule by default)
    for sampler, scheduler in (("euler", "default"), ("heun", "simple"), ("dpmpp_2m", "beta")):
        res = d.sample(model, cond(0), None, 64, 64, 8, 1.0, seed=1, frames=9, sampler=sampler, scheduler=scheduler)
        assert torch.isfinite(res["video"]).all() and torch.isfinite(res["audio"]).all()

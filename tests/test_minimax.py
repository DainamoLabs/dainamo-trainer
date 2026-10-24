"""MiniMax H3: driver plumbing around the packed sequence, checked with a recording stand-in model.

These run on the CPU in seconds; the real-weights checks (text encoder, VAEs and NVFP4 against
ai-toolkit) need the 40 GB of model files and are run by hand.

    .venv/bin/python -m pytest tests/test_minimax.py -v
"""

import random

import torch
from PIL import Image

from dainamo.engine.families.minimax_h3 import packing as P
from dainamo.engine.families.minimax_h3.driver import MinimaxH3Driver

CPU = torch.device("cpu")


class Recorder(torch.nn.Module):
    """Stands in for the DiT: records its inputs and predicts zeros (so the loss is the target's power)."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kw):
        self.calls.append({k: v.detach().clone() for k, v in kw.items()})
        return torch.zeros_like(kw["hidden_states"]), torch.zeros_like(kw["audio_hidden_states"])


def text_cond(n=7, vision=0):
    tags = torch.cat([torch.zeros(vision, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    return {"embeds": torch.randn(n + vision, 5120).bfloat16(), "tags": tags}


def video_payload(frames=22, h=4, w=6, audio=True, first=False, refs=0):
    t = P.video_latent_num_frames(frames)
    p = {"video": torch.randn(24, t, h, w), "frames": frames}
    if audio:
        p["audio"] = torch.randn(P.audio_latent_num_frames(frames) * P.AUDIO_CHANNELS, 32)
    if first:
        p["first"] = torch.randn(24, 1, h, w)
    if refs:
        p["refs"] = [torch.randn(24, 1, h, w) for _ in range(refs)]
    return p


class AlwaysLow:
    def random(self):
        return 0.0


def test_frame_counts_follow_17n_plus_5():
    assert [P.align_num_frames_down(n) for n in (5, 21, 22, 38, 39, 40, 121)] == [5, 5, 22, 22, 39, 39, 107]
    assert [P.video_latent_num_frames(n) for n in (5, 22, 39)] == [2, 7, 12]
    assert P.audio_latent_num_frames(24) == 40  # 40 Hz audio latents


def test_audio_sigma_tracks_video_schedule():
    s = torch.linspace(0, 1, 11)
    a = P.remap_sigma(s, P.VIDEO_SIGMA_SHIFT, P.AUDIO_SIGMA_SHIFT)
    assert a[0] == 0 and torch.isclose(a[-1], torch.tensor(1.0))
    assert torch.all(a[1:-1] < s[1:-1])  # shift 3 sits below shift 12
    assert torch.all(a.diff() > 0)
    sched = P.build_sigma_schedule(20)
    assert len(sched) == 21 and sched[0] == 1 and sched[-1] == 0


def test_t2va_step_packs_text_audio_video_with_per_row_times():
    d = MinimaxH3Driver({"variant": "fl2va"}, CPU)
    m = Recorder()
    p = video_payload()
    loss = d.training_loss(m, [p], [{"text": text_cond()}], {"train_audio": True}, rng=random.Random(0), sigma=0.5)
    call = m.calls[0]
    t, h, w = p["video"].shape[1:]
    video_rows = t * (h // 2) * (w // 2)
    assert call["hidden_states"].shape[1] == video_rows  # no condition rows without a keyframe
    assert call["audio_hidden_states"].shape[1] == p["audio"].shape[0]
    rt = call["row_timesteps"][0]
    assert torch.allclose(rt[call["video_indices"]], torch.tensor(0.5))
    sa = float(P.remap_sigma(torch.tensor(0.5), P.VIDEO_SIGMA_SHIFT, P.AUDIO_SIGMA_SHIFT))
    assert torch.allclose(rt[call["audio_indices"]], torch.tensor(1 - sa))
    # zero prediction: loss = E[(x0 - noise)^2] for video + audio, about 2 + 2
    assert 3.0 < loss.item() < 5.0


def test_silent_clip_and_images_have_no_audio_loss():
    d = MinimaxH3Driver({"variant": "fl2va"}, CPU)
    torch.manual_seed(0)
    silent = d.training_loss(Recorder(), [video_payload(audio=False)], [{"text": text_cond()}], {}, sigma=0.5)
    assert 1.5 < silent.item() < 2.5  # video term only
    image = {"video": torch.randn(24, 1, 4, 6), "frames": 1}
    m = Recorder()
    d.training_loss(m, [image], [{"text": text_cond()}], {}, sigma=0.5)
    assert m.calls[0]["hidden_states"].shape[1] == 6


def test_fl2va_keyframe_rows_lead_and_stay_clean():
    d = MinimaxH3Driver({"variant": "fl2va"}, CPU)
    m = Recorder()
    p = video_payload(first=True)
    conds = {"text": text_cond(5), "i2v": text_cond(5, vision=9)}
    d.training_loss(m, [p], [conds], {"i2v_probability": 1.0}, rng=AlwaysLow(), sigma=0.3)
    call = m.calls[0]
    per_frame = (4 // 2) * (6 // 2)
    t = p["video"].shape[1]
    assert call["hidden_states"].shape[1] == per_frame * (1 + t)  # keyframe rows, then the target
    assert call["encoder_hidden_states"].shape[1] == 14  # the i2v presentation (vision block + caption)
    rt = call["row_timesteps"][0][call["video_indices"]]
    assert torch.allclose(rt[:per_frame], torch.tensor(P.KEYFRAME_NOISE_AUG_T))
    assert torch.allclose(rt[per_frame:], torch.tensor(0.7))
    # deterministic (validation) steps never take the keyframe branch
    m2 = Recorder()
    d.training_loss(m2, [p], [conds], {"i2v_probability": 1.0}, rng=AlwaysLow(), sigma=0.3, deterministic=True)
    assert m2.calls[0]["hidden_states"].shape[1] == per_frame * t


def test_ref2va_uses_reference_rows_when_present():
    d = MinimaxH3Driver({"variant": "ref2va"}, CPU)
    m = Recorder()
    p = video_payload(refs=2)
    conds = {"text": text_cond(), "ref": text_cond(7, vision=12)}
    d.training_loss(m, [p], [conds], {}, sigma=0.5)
    per_frame = (4 // 2) * (6 // 2)
    t = p["video"].shape[1]
    assert m.calls[0]["hidden_states"].shape[1] == per_frame * (2 + t)
    assert m.calls[0]["encoder_hidden_states"].shape[1] == 19
    # an item without references trains text-only
    m2 = Recorder()
    d.training_loss(m2, [video_payload()], [{"text": text_cond()}], {}, sigma=0.5)
    assert m2.calls[0]["hidden_states"].shape[1] == per_frame * t


class Echo(Recorder):
    """Records its inputs and predicts them back, so each item's loss depends on its own rows."""

    def forward(self, **kw):
        super().forward(**kw)
        return kw["hidden_states"].float(), kw["audio_hidden_states"].float()


def test_ref2va_batch_runs_a_forward_per_reference_layout():
    """Items with different references in one batch (one, two, none): a forward each, the loss the mean over all."""
    d = MinimaxH3Driver({"variant": "ref2va"}, CPU)
    d.dtype = torch.float32
    one, two, none = video_payload(refs=1), video_payload(refs=2), video_payload()
    two["refs"][1] = torch.randn(24, 1, 6, 4)  # a portrait reference
    conds = [{"text": text_cond(), "ref": text_cond(7, vision=12)}, {"text": text_cond(), "ref": text_cond(7, vision=20)},
             {"text": text_cond()}]
    m = Echo()
    loss = d.training_loss(m, [one, two, none], conds, {}, sigma=0.5, generator=torch.Generator().manual_seed(0))
    per_frame, t = 2 * 3, P.video_latent_num_frames(22)
    assert [tuple(c["hidden_states"].shape[:2]) for c in m.calls] == [
        (1, per_frame * (1 + t)), (1, per_frame * (2 + t)), (1, per_frame * t)]
    assert [c["encoder_hidden_states"].shape[1] for c in m.calls] == [19, 27, 7]
    # the same draws by hand: the video and audio noise come first, for the whole batch
    g = torch.Generator().manual_seed(0)
    x0, xa = torch.stack([p["video"] for p in (one, two, none)]), torch.stack([p["audio"] for p in (one, two, none)])
    noise, na = torch.randn(x0.shape, generator=g), torch.randn(xa.shape, generator=g)
    sa = float(P.remap_sigma(torch.tensor(0.5), P.VIDEO_SIGMA_SHIFT, P.AUDIO_SIGMA_SHIFT))
    want = (torch.mean((0.5 * x0 + 0.5 * noise - (x0 - noise)) ** 2)
            + torch.mean(((1 - sa) * xa + sa * na - (xa - na)) ** 2))
    assert torch.allclose(loss, want)


class FakeAudioVAE:
    """Encodes like the H3 audio VAE in shape: (B, 1, samples) -> (B, 32, ceil(samples / 800)), latent i of input b
    holding 1000 * b + i, so the rows say where they came from."""

    def encode(self, wav):
        t = -(-wav.shape[-1] // 800)
        return (1000.0 * torch.arange(wav.shape[0])[:, None, None] + torch.arange(t)).expand(-1, 32, -1).clone()


def _encoding_driver(variant, monkeypatch):
    """A driver whose VAEs are stand-ins: a clip's frames aren't read and its video encodes to zeros."""
    from dainamo.engine import dataset as ds

    d = MinimaxH3Driver({"variant": variant}, CPU)
    d.video_vae = type("VAE", (), {"encode": staticmethod(
        lambda px, sample=True, **kw: torch.zeros(1, 24, P.video_latent_num_frames(px.shape[2]) if px.shape[2] > 1
                                                  else 1, px.shape[3] // 16, px.shape[4] // 16))})()
    d.audio_vae = FakeAudioVAE()
    monkeypatch.setattr(ds, "read_video_frames", lambda path, size, frames, fps, flip=False, *a, **kw:
                        torch.zeros(1, 3, frames, size[1], size[0]))
    return d


def test_a_clips_audio_encodes_to_the_models_latent_count(tmp_path, monkeypatch):
    """At 56 frames the VAE gives 94 latents a channel (74 667 samples, padded to 800s) to the model's 93: each
    channel is cut before packing, so channel 1 starts at row a_lat."""
    import subprocess

    from dainamo.engine import dataset as ds

    clip = tmp_path / "clip.wav"  # the soundtrack is all encode_item reads of the clip (its frames are stubbed)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=3", "-ac", "2",
                    str(clip)], check=True)
    d = _encoding_driver("fl2va", monkeypatch)
    item = ds.Item(path=str(clip), caption="", size=(96, 64), frames=56, has_audio=True)
    a_lat = P.audio_latent_num_frames(56)
    assert a_lat == 93
    rows = d.encode_item(item, {"train_audio": True})["audio"]
    assert rows.shape == (2 * a_lat, 32)
    assert torch.equal(rows[:, 0], torch.cat([torch.arange(a_lat), 1000 + torch.arange(a_lat)]).float())
    # a payload cached before the cut (94 latents a channel) trains on the same rows
    d.dtype = torch.float32
    stale = {"video": torch.randn(24, P.video_latent_num_frames(56), 4, 6), "frames": 56,
             "audio": P.pack_audio_latents(d.audio_vae.encode(torch.zeros(2, 1, 74667))[None])[0]}
    assert stale["audio"].shape[0] == 2 * 94
    m = Recorder()
    d.training_loss(m, [stale], [{"text": text_cond()}], {}, sigma=0.0)
    assert torch.equal(m.calls[0]["audio_hidden_states"][0], rows)


def test_ref2va_mirrors_references_with_a_flipped_item(tmp_path, monkeypatch):
    from dainamo.engine import dataset as ds
    from dainamo.engine.trainer import cond_key

    img, ref = tmp_path / "a.png", tmp_path / "a_ref.png"
    Image.new("RGB", (96, 64)).save(img)
    face = Image.new("RGB", (96, 64))
    face.paste((255, 0, 0), (0, 0, 30, 64))  # red on the left only
    face.save(ref)
    d = _encoding_driver("ref2va", monkeypatch)
    seen = []
    d._encode_keyframe = lambda px: seen.append(px) or torch.zeros(1, 24, 1, 4, 6)
    items = [ds.Item(path=str(img), caption="a", size=(96, 64), flip=f, refs=[str(ref)]) for f in (False, True)]
    plain, flipped = (d.item_conditions(it, {})["ref"][1][0] for it in items)
    assert flipped.tobytes() == plain.transpose(Image.Transpose.FLIP_LEFT_RIGHT).tobytes()
    assert cond_key("a", [plain]) != cond_key("a", [flipped]) and items[0].key() != items[1].key()
    for it in items:
        d.encode_item(it, {})
    assert torch.equal(seen[1], seen[0].flip(-1))


def test_conditioning_pixels_reach_the_vision_tower_in_fp32():
    """The text model runs in bf16 but the vision tower in fp32: the pixels aren't rounded to bf16 on the way."""
    from dainamo.engine.families.minimax_h3.text_encoder import encode_minimax_h3_prompt

    tok = type("Tok", (), {"__call__": lambda self, text, add_special_tokens=False: {"input_ids": [5] * len(text)},
                           "convert_tokens_to_ids": lambda self, t: 1, "pad_token_id": 0})()
    pixels = torch.randn(16, 8, dtype=torch.float32)
    processor = type("Proc", (), {
        "image_processor": type("IP", (), {"merge_size": 2, "__call__": lambda self, images, return_tensors: {
            "pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 4, 4]])}})(),
        "create_mm_token_type_ids": staticmethod(lambda ids: [[0] * len(ids[0])])})()
    seen = {}

    def model(input_ids, pixel_values=None, **kw):
        seen["pixels"] = pixel_values
        return type("Out", (), {"hidden_states": [torch.zeros(1, input_ids.shape[1], 8)] * 51})()

    cfg = type("Cfg", (), {"text_config": type("TC", (), {"num_hidden_layers": 50})()})()
    te = type("TE", (), {"config": cfg, "device": CPU, "dtype": torch.bfloat16, "model": staticmethod(model)})()
    encode_minimax_h3_prompt(te, tok, processor, "a cat", keyframes=[Image.new("RGB", (64, 64))])
    assert seen["pixels"].dtype == torch.float32 and torch.equal(seen["pixels"], pixels)


def test_sampler_steps_and_shapes():
    d = MinimaxH3Driver({"variant": "fl2va"}, CPU)
    m = Recorder()
    out = d.sample(m, text_cond(), None, 96, 64, 4, 0, 7, frames=24)
    assert out["frames"] == 22  # snapped down to 17n+5
    assert out["video"].shape == (1, 24, 7, 4, 6)
    assert out["audio"].shape[1] == P.audio_latent_num_frames(22) * P.AUDIO_CHANNELS
    assert len(m.calls) == 4
    still = d.sample(Recorder(), text_cond(), None, 96, 64, 2, 0, 7, frames=1)
    assert still["frames"] == 1 and still["video"].shape[2] == 1


def test_fl2va_takes_references_with_reference_to_video(tmp_path):
    """An fl2va render's image is its first frame, unless Reference-to-Video makes its images references."""
    d = MinimaxH3Driver({"variant": "fl2va"}, CPU)
    assert "r2v" in [w.id for w in d.info.workflows if not w.variants or "fl2va" in w.variants]
    d.video_vae = object()  # the encode is stubbed
    d._encode_keyframe = lambda px: torch.randn(1, 24, 1, 4, 6)
    paths = []
    for i in range(2):
        Image.new("RGB", (96, 64)).save(tmp_path / f"{i}.png")
        paths.append(str(tmp_path / f"{i}.png"))
    spec = {"prompt": "a cat", "images": paths, "width": 96, "height": 64, "frames": 22}
    assert len(d.sample_condition(spec)[1]) == 1
    assert len(d.sample_condition({**spec, "references": True})[1]) == 2
    per_frame, t = 2 * 3, P.video_latent_num_frames(22)
    first, refs = Recorder(), Recorder()
    d.sample(first, text_cond(), None, 96, 64, 1, 0, 7, frames=22, images=paths)
    d.sample(refs, text_cond(), None, 96, 64, 1, 0, 7, frames=22, images=paths, references=True)
    assert first.calls[0]["hidden_states"].shape[1] == per_frame * (1 + t)  # one keyframe
    assert refs.calls[0]["hidden_states"].shape[1] == per_frame * (2 + t)  # two reference blocks


class Oracle(torch.nn.Module):
    """Predicts the exact velocity toward fixed clean rows, reading sigma from the row times (t = 1 - sigma)."""

    def __init__(self, x0_video, x0_audio):
        super().__init__()
        self.x0_video, self.x0_audio = x0_video, x0_audio

    def forward(self, hidden_states, audio_hidden_states, row_timesteps, video_indices, audio_indices, **kw):
        sv = 1 - row_timesteps[0, video_indices[-1]]
        sa = 1 - row_timesteps[0, audio_indices[0]]
        v = (self.x0_video - hidden_states.float()) / sv
        a = (self.x0_audio - audio_hidden_states.float()) / sa
        return v, a


def test_sampler_lands_on_the_clean_latent_given_the_true_velocity():
    d = MinimaxH3Driver({"variant": "fl2va"}, CPU)
    d.dtype = torch.float32
    a_rows = P.audio_latent_num_frames(22) * P.AUDIO_CHANNELS
    x0v, x0a = torch.randn(1, 7 * 2 * 3, 96), torch.randn(1, a_rows, 32)
    out = d.sample(Oracle(x0v, x0a), text_cond(), None, 96, 64, 6, 0, 3, frames=22)
    assert torch.allclose(P.patchify_video_latents(out["video"]), x0v, atol=1e-4)
    assert torch.allclose(out["audio"], x0a, atol=1e-4)


def test_loss_weighting_follows_each_items_own_noise_level():
    """Per-row timesteps, but every row in the loss has its item's noise level: each item's video loss is weighted
    by its sigma and its audio by the audio's, also through ref2va's reordered groups."""
    d = MinimaxH3Driver({"variant": "ref2va"}, CPU)
    d.dtype = torch.float32
    items = [video_payload(refs=1), video_payload(), video_payload(refs=1)]  # groups: items 0 and 2, then 1
    conds = [{"text": text_cond(), "ref": text_cond(7, vision=12)}, {"text": text_cond()},
             {"text": text_cond(), "ref": text_cond(7, vision=12)}]
    sv = torch.tensor([0.2, 0.5, 0.8])
    d.sample_timesteps = lambda b, hw, settings: sv.clone()

    def loss(settings, **kw):
        return d.training_loss(Echo(), items, conds, settings, generator=torch.Generator().manual_seed(0), **kw)

    # the same draws by hand (Echo predicts its noisy input back)
    g = torch.Generator().manual_seed(0)
    x0, xa = torch.stack([p["video"] for p in items]), torch.stack([p["audio"] for p in items])
    noise, na = torch.randn(x0.shape, generator=g), torch.randn(xa.shape, generator=g)
    sa = P.remap_sigma(sv, P.VIDEO_SIGMA_SHIFT, P.AUDIO_SIGMA_SHIFT)
    s5, s3 = sv.view(-1, 1, 1, 1, 1), sa.view(-1, 1, 1)
    err_v = (((1 - s5) * x0 + s5 * noise - (x0 - noise)) ** 2).flatten(1).mean(1)
    err_a = (((1 - s3) * xa + s3 * na - (xa - na)) ** 2).flatten(1).mean(1)
    assert torch.allclose(loss({}), err_v.mean() + err_a.mean())
    s = {"loss_weighting": "cosmap"}
    w_v, w_a = d.loss_weights(sv, s), d.loss_weights(sa, s)
    assert torch.allclose(loss(s), (err_v * w_v).mean() + (err_a * w_a).mean())
    # a pinned sigma (validation) isn't weighted
    assert torch.equal(loss(s, sigma=0.5), loss({}, sigma=0.5))


def test_the_noise_range_reaches_the_rows():
    d = MinimaxH3Driver({"variant": "fl2va"}, CPU)
    m = Recorder()
    d.training_loss(m, [video_payload() for _ in range(32)], [{"text": text_cond()}] * 32,
                    {"timestep_min": 0.4, "timestep_max": 0.6})
    call = m.calls[0]
    video_t = call["row_timesteps"][:, call["video_indices"]]  # the model's t is 1 - sigma
    assert float(video_t.min()) >= 0.4 - 1e-6 and float(video_t.max()) <= 0.6 + 1e-6

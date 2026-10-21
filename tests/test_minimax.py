"""MiniMax H3: driver plumbing around the packed sequence, checked with a recording stand-in model.

These run on the CPU in seconds; the real-weights checks (text encoder, VAEs and NVFP4 against
ai-toolkit) need the 40 GB of model files and are run by hand.

    .venv/bin/python -m pytest tests/test_minimax.py -v
"""

import random

import torch

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

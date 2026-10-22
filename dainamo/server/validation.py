"""Checks for a job's training settings; each raises a 400 with a readable message."""

from __future__ import annotations

import os

from fastapi import HTTPException

MAX_FRAMES = 241  # 10 s at 24 fps
MAX_RESOLUTIONS = 6
MAX_SLIDER_IMAGES = 200
MAX_SLIDER_PRESERVE = 12
LOKR_FACTOR = 16  # a LoKr's default split
MIN_SONG_SECONDS, MAX_SONG_SECONDS = 5, 360  # music clips and samples (YuE2 writes up to 6 minutes)
MAX_LYRICS = 6000


def check_settings(s: dict, div: int, audio: bool = False) -> dict:
    """Coerce and sanity-check the numeric training settings; raises 400 with a readable message.
    ``audio``: a music family's job, which trains on clips of songs rather than at resolutions."""
    def num(key, kind, lo=None, hi=None):
        try:
            v = kind(s[key])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, f"Setting {key!r} must be a number")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise HTTPException(400, f"Setting {key!r}={v} is out of range")
        s[key] = v

    num("steps", int, 1)
    num("rank", int, 1, 1024)
    num("alpha", float, 1e-6)
    num("learning_rate", float, 1e-9, 10)
    num("batch_size", int, 1, 64)
    if audio:
        s.pop("resolution", None)
        check_song_seconds(s)
    else:
        s["resolution"] = check_resolution(s.get("resolution"))
    for key, kind, lo in (("grad_accum", int, 1), ("save_every", int, 0), ("sample_every", int, 0),
                          ("keep_last", int, 0), ("warmup_steps", int, 0), ("caption_dropout", float, 0),
                          ("dropout", float, 0), ("max_grad_norm", float, 0), ("weight_decay", float, 0),
                          ("sample_width", int, div), ("sample_height", int, div),
                          ("num_frames", int, 1), ("sample_frames", int, 1), ("i2v_probability", float, 0),
                          ("audio_loss_weight", float, 0), ("offload", float, 0)):
        if key in s:
            num(key, kind, lo)
    for key, hi in (("num_frames", MAX_FRAMES), ("sample_frames", MAX_FRAMES), ("i2v_probability", 1),
                    ("offload", 100)):
        if key in s and s[key] > hi:
            raise HTTPException(400, f"Setting {key!r}={s[key]} is out of range (max {hi})")
    for key in ("train_audio", "keep_alpha", "low_vram", "auto_frames"):
        if key in s:
            s[key] = bool(s[key])
    if s.get("auto_frames"):  # each clip at its own length: a clip trains at most the clip length, so the longest
        s["num_frames"] = MAX_FRAMES
    if s.get("optimizer") not in (None, "adamw8bit", "adamw", "adafactor", "prodigy"):
        raise HTTPException(400, f"Unknown optimizer {s.get('optimizer')!r}")
    check_network(s)
    s["sample_prompts"] = check_sample_prompts(s.get("sample_prompts", []), div, audio)
    check_evaluation(s, div, audio)
    check_slider(s)
    return s


def check_network(s: dict) -> None:
    """The adapter trained: a LoRA (``rank``, ``alpha``) or a LoKr split by ``lokr_factor`` (-1: evenly)."""
    kind = s.get("network_type") or "lora"
    if kind not in ("lora", "lokr"):
        raise HTTPException(400, f"Unknown network type {kind!r}")
    if "network_type" in s:
        s["network_type"] = kind
    if kind == "lokr":
        try:
            factor = int(s.get("lokr_factor", LOKR_FACTOR))
        except (TypeError, ValueError):
            raise HTTPException(400, "Setting 'lokr_factor' must be a whole number")
        if factor != -1 and not 2 <= factor <= 1024:
            raise HTTPException(400, f"Setting 'lokr_factor'={factor} must be -1 (an even split) or 2 to 1024")
        s["lokr_factor"] = factor


def strength_list(raw, what: str, most: int) -> list[float]:
    """LoRA strengths as typed ("0.6, 0.8") or a list: 1 to ``most`` of them, each -10 to 10."""
    parts = raw.replace(",", " ").split() if isinstance(raw, str) else list(raw)
    try:
        values = [float(x) for x in parts]
    except (TypeError, ValueError):
        raise HTTPException(400, f"{what} must be numbers, e.g. 0.6, 0.8, 1.0")
    if not 1 <= len(values) <= most or any(not -10 <= x <= 10 for x in values):
        raise HTTPException(400, f"{what} need 1–{most} strengths between -10 and 10")
    return values


def check_slider(s: dict) -> None:
    """A concept slider (``engine.slider``): its three prompts, strength, sample strengths, how many images
    to render when the job has no datasets, preserved attributes and whether it keeps the layout. The
    checkpoint picker doesn't apply: a slider isn't trained to reproduce its images, so their validation
    loss says nothing about it."""
    s["slider_enabled"] = bool(s.get("slider_enabled", False))
    if not s["slider_enabled"]:
        return
    if s.get("network_type") == "lokr":
        raise HTTPException(400, "A slider trains as a LoRA, not a LoKr")
    for key, label in (("slider_target", "target"), ("slider_positive", "positive"), ("slider_negative", "negative")):
        s[key] = str(s.get(key) or "").strip()
        if not s[key]:
            raise HTTPException(400, f"A slider needs a {label} prompt")
    if s["slider_positive"] == s["slider_negative"]:
        raise HTTPException(400, "A slider's positive and negative prompts must differ")
    for key, kind, default, lo, hi in (("slider_strength", float, 3.0, 0.1, 20.0),
                                       ("slider_images", int, 16, 1, MAX_SLIDER_IMAGES),
                                       ("slider_layout_sigma", float, 0.9, 0.05, 0.99)):
        try:
            v = kind(s.get(key) if s.get(key) not in (None, "") else default)
        except (TypeError, ValueError):
            raise HTTPException(400, f"Setting {key!r} must be a number")
        if not lo <= v <= hi:
            raise HTTPException(400, f"Setting {key!r}={v} is out of range ({lo:g}–{hi:g})")
        s[key] = v
    s["slider_sample_strengths"] = strength_list(s.get("slider_sample_strengths") or [-1.0, 1.0],
                                                 "Slider sample strengths", 6)
    raw = s.get("slider_preserve") or []
    raw = raw.splitlines() if isinstance(raw, str) else raw
    if not isinstance(raw, list):
        raise HTTPException(400, "slider_preserve must be a list of attributes")
    preserve = list(dict.fromkeys(str(p).strip().strip(",").strip() for p in raw))
    preserve = [p for p in preserve if p]
    if len(preserve) > MAX_SLIDER_PRESERVE or any(len(p) > 200 for p in preserve):
        raise HTTPException(400, f"A slider preserves up to {MAX_SLIDER_PRESERVE} attributes, each a short phrase")
    s["slider_preserve"] = preserve
    s["slider_keep_layout"] = bool(s.get("slider_keep_layout", False))
    if s.get("picker_enabled"):
        raise HTTPException(400, "The checkpoint picker doesn't apply to sliders; concept bleed does")


def check_resolution(raw) -> int | list[int]:
    """One training resolution (an int) or several ("512, 1024" or a list): sorted, without repeats.
    A single value stays an int, as in jobs from before multi-resolution training."""
    parts = raw.replace(",", " ").split() if isinstance(raw, str) else raw if isinstance(raw, list) else [raw]
    try:
        values = sorted({int(v) for v in parts})
    except (TypeError, ValueError):
        raise HTTPException(400, "Setting 'resolution' must be a number or a list of numbers")
    if not values:
        raise HTTPException(400, "Choose at least one training resolution")
    if len(values) > MAX_RESOLUTIONS:
        raise HTTPException(400, f"Up to {MAX_RESOLUTIONS} training resolutions")
    if any(not 128 <= v <= 4096 for v in values):
        raise HTTPException(400, "Training resolutions must be 128–4096")
    return values[0] if len(values) == 1 else values


def check_song_seconds(s: dict) -> None:
    """Music: ``clip_seconds`` (0 trains whole songs, up to the longest YuE2 writes) and ``sample_seconds``."""
    for key, default, zero in (("clip_seconds", 30, True), ("sample_seconds", 30, False)):
        try:
            v = int(s.get(key) if s.get(key) not in (None, "") else default)
        except (TypeError, ValueError):
            raise HTTPException(400, f"Setting {key!r} must be a whole number of seconds")
        if not (MIN_SONG_SECONDS <= v <= MAX_SONG_SECONDS or (zero and v == 0)):
            raise HTTPException(400, f"Setting {key!r}={v} is out of range ({'0 or ' if zero else ''}"
                                     f"{MIN_SONG_SECONDS}–{MAX_SONG_SECONDS} s)")
        s[key] = v


def check_evaluation(s: dict, div: int, audio: bool = False) -> None:
    """Best-checkpoint picker and concept-bleed settings (both optional, off by default)."""
    def flag(key, default=False):
        s[key] = bool(s.get(key, default))
        return s[key]

    def num(key, kind, default, lo, hi):
        try:
            v = kind(s.get(key, default) if s.get(key) not in (None, "") else default)
        except (TypeError, ValueError):
            raise HTTPException(400, f"Setting {key!r} must be a number")
        if not lo <= v <= hi:
            raise HTTPException(400, f"Setting {key!r}={v} is out of range ({lo}–{hi})")
        s[key] = v

    if flag("picker_enabled"):
        num("val_holdout", int, 2, 1, 100)
        num("val_timesteps", int, 8, 1, 64)
        s["val_dataset"] = str(s.get("val_dataset") or "")
        if flag("sweep_enabled", True):
            s["sweep_strengths"] = strength_list(s.get("sweep_strengths", [0.6, 0.8, 1.0, 1.2]),
                                                 "Strength sweep values", 8)
    if flag("bleed_enabled"):
        if audio:
            raise HTTPException(400, "Concept bleed compares images, so it doesn't apply to music")
        prompts = [str(p).strip() for p in s.get("bleed_prompts") or [] if str(p).strip()]
        if not 1 <= len(prompts) <= 8:
            raise HTTPException(400, "Concept bleed needs 1–8 probe prompts")
        s["bleed_prompts"] = prompts
        num("bleed_width", int, 512, div, 2048)
        num("bleed_height", int, 512, div, 2048)
        num("bleed_seed", int, 1234, 0, 2**63 - 1)
        num("bleed_max", float, 25.0, 0.0, 1000.0)


def check_sample_prompts(cards, div: int, audio: bool = False) -> list[dict]:
    """Sample prompt cards: {prompt, width?, height?, seed?, lora_scale?, frames?, images?}, and for music
    {prompt (the style), lyrics?, seconds?, seed?, lora_scale?}; blank fields use the Sampling defaults.
    ``images`` are conditioning image paths (video families: a first frame or references). Plain strings are
    accepted as prompt-only cards. Empty prompts are dropped."""
    if not isinstance(cards, list):
        raise HTTPException(400, "sample_prompts must be a list")
    out = []
    for n, card in enumerate(cards, 1):
        card = {"prompt": card} if isinstance(card, str) else card
        if not isinstance(card, dict):
            raise HTTPException(400, f"Sample prompt {n} is malformed")
        prompt = str(card.get("prompt") or "").strip()
        if not prompt:
            continue
        clean = {"prompt": prompt}
        fields = ((("seconds", int, MIN_SONG_SECONDS, MAX_SONG_SECONDS),) if audio else
                  (("width", int, div, 4096), ("height", int, div, 4096), ("frames", int, 1, MAX_FRAMES)))
        if audio:
            lyrics = str(card.get("lyrics") or "").strip()
            if len(lyrics) > MAX_LYRICS:
                raise HTTPException(400, f"Sample prompt {n}: lyrics are limited to {MAX_LYRICS} characters")
            if lyrics:
                clean["lyrics"] = lyrics
        for key, kind, lo, hi in fields + (("seed", int, 0, 2**63 - 1), ("lora_scale", float, -10.0, 10.0)):
            v = card.get(key)
            if v in (None, ""):
                continue
            try:
                v = kind(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"Sample prompt {n}: {key} must be a number")
            if not lo <= v <= hi:
                raise HTTPException(400, f"Sample prompt {n}: {key}={v} is out of range")
            clean[key] = v
        images = card.get("images") or []
        images = [images] if isinstance(images, str) else images
        if not isinstance(images, list) or len(images) > 8:
            raise HTTPException(400, f"Sample prompt {n}: images must be a list of up to 8 paths")
        paths = []
        for im in images:
            path = os.path.expanduser(str(im).strip())
            if not path:
                continue
            if not os.path.isfile(path):
                raise HTTPException(400, f"Sample prompt {n}: image not found: {path}")
            paths.append(path)
        if paths:
            clean["images"] = paths
        out.append(clean)
    return out

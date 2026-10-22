"""RefMods: MiniMax H3 references saved as files, in the format of ComfyUI-MiniMaxH3Mod (MIT,
https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod), so a RefMod made here loads there and one made there
loads here.

A RefMod holds the video VAE's latents of a set of references (images and clips of one subject or style), stacked
along time: (1, 24, T, h, w) in the VAE's normalized latent space. Generating with the ref2va model, they go in
through the same reference tokens as reference images do, without being encoded again. Making one needs only the
VAE:
  * full: each reference's encode as it is, every one cover-cropped to the first one's size. The most detail,
    and the most tokens: (h / 2) * (w / 2) per latent frame
  * compressed: each encode averaged down to a small grid (its long edge ``refmod_grid``, the short one from the
    first reference's shape), then refined so the grid scaled back up matches the full encode (a few hundred
    Adam steps on its own ~2K numbers, no diffusion model). Far fewer tokens, less detail. ``refmod_merge``
    refines one grid against every reference instead of stacking one per reference
A cap on the tokens (``refmod_max_tokens``) drops near-duplicate frames, then spreads the rest evenly.

An audio RefMod (a voice, a singer, a style of music, a sound) holds the audio VAE's latents of a sound instead:
(1, 32, 2, T), 32 latent channels by 2 stereo channels by 40 latents a second, normalized, the posterior mean (the
pack's layout of what the released pipeline encodes an audio reference to). ``make_audio`` takes the datasets'
sound files and clip soundtracks in order, up to ``refmod_seconds``; it puts 2 tokens per latent into the model.

Files: a standalone RefMod is a .safetensors file with one ``latent`` tensor (fp16) and its description as JSON
under the header's ``refmod_meta`` key (format 4; formats 2 and 3 read the same way; the older H3AudioMod's voices
under ``audio_refmod_meta``). A bundle (format 5) holds several, ``ref_0``, ``ref_1``, …, described by its
``members``.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from ..fsutil import replace
from . import dataset as ds

META_KEY = "refmod_meta"
AUDIO_META_KEY = "audio_refmod_meta"  # voices saved by the older H3AudioMod
FORMAT = 4
BUNDLE_FORMAT = 5
CHANNELS = 24  # the H3 video VAE's latent channels
AUDIO_LATENT_CHANNELS = 32  # the H3 audio VAE's
AUDIO_RATE = 32000  # the audio VAE's sample rate
AUDIO_HOP = 800  # samples per audio latent
AUDIO_PER_SECOND = AUDIO_RATE // AUDIO_HOP  # 40
# seconds encoded at a time (~0.8 GB a 10 s): the pack encodes 10 s at a time, which leaves a seam in the latents
# at 10 s that a generation's longest reference (its own length, up to 10 s) can reach
AUDIO_CHUNK = 30
SCALE = 16  # pixels per latent cell
# what a RefMod represents: kept in the file (the pack's loaders show it), it doesn't change how it's made
CONCEPTS = ("generic", "identity", "pose_motion", "clothing", "background", "style")
AUDIO_CONCEPTS = ("generic", "voice", "singing", "music_style", "sound_fx", "ambience")
MODES = {"full": "encode", "compressed": "training"}  # Dainamo's names -> the file's
DEDUP = 0.02  # a frame this close to the last one kept (mean difference / magnitude) counts as a duplicate


# --- files ------------------------------------------------------------------
def read_meta(path: str) -> dict | None:
    """A file's RefMod description (its header only), or None if it isn't a RefMod."""
    from safetensors import safe_open

    try:
        with safe_open(path, "pt") as f:
            header = f.metadata() or {}
        raw = header.get(META_KEY) or header.get(AUDIO_META_KEY)
        meta = json.loads(raw) if raw else None
    except Exception:  # unreadable, not a safetensors file (SafetensorError), or a header that isn't JSON
        return None
    return meta if isinstance(meta, dict) and meta.get("kind") in ("image", "video", "audio", "bundle") else None


def load(path: str) -> list[dict]:
    """A RefMod's references: [{"name", "kind" (image, video or audio), "latent" (float32)}], one for a standalone
    file, each member of a bundle in order."""
    from safetensors.torch import load_file

    meta = read_meta(path)
    if meta is None:
        raise ValueError(f"{os.path.basename(path)} isn't a RefMod (it has no {META_KEY} description)")
    tensors = load_file(path)
    if meta["kind"] != "bundle":
        members = [(meta, tensors.get("latent"))]
    else:
        members = [(m, tensors.get(f"ref_{i}")) for i, m in enumerate(meta.get("members") or [])]
    out = []
    for m, t in members:
        if t is None:
            raise ValueError(f"{os.path.basename(path)} is missing the latent of {m.get('name') or 'a reference'}")
        visual = m.get("kind") != "audio"
        if ((visual and (t.dim() != 5 or t.shape[1] != CHANNELS))
                or (not visual and (t.dim() != 4 or tuple(t.shape[:3]) != (1, AUDIO_LATENT_CHANNELS, 2)))):
            raise ValueError(f"{os.path.basename(path)}: unexpected latent shape {tuple(t.shape)}")
        out.append({"name": m.get("name") or os.path.splitext(os.path.basename(path))[0],
                    "kind": m.get("kind", "image"), "latent": t.float()})
    return out


def visual(path: str) -> list[dict]:
    """The image and video references of a RefMod."""
    return [r for r in load(path) if r["kind"] != "audio"]


def member_kinds(meta: dict) -> list[str]:
    """What a RefMod's description says it holds, in order: "image", "video" or "audio" per reference."""
    members = (meta.get("members") or []) if meta.get("kind") == "bundle" else [meta]
    return [m.get("kind", "image") for m in members if isinstance(m, dict)]


def save(path: str, latent: torch.Tensor, meta: dict) -> str:
    """Write a standalone RefMod: ``latent`` ((1, 24, T, h, w), or (1, 32, 2, T) audio) as fp16, ``meta``
    (``describe`` or ``describe_audio``) in the header."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    save_file({"latent": latent.to(torch.float16).contiguous()}, tmp, metadata={META_KEY: json.dumps(meta)})
    replace(tmp, path)
    return path


def describe(name: str, latent: torch.Tensor, mode: str, source: str, source_shape: str, pool: str,
             refine_steps: int, tags: list[str], description: str = "", concept: str = "generic") -> dict:
    """A standalone RefMod's header description (format 4)."""
    t, h, w = (int(x) for x in latent.shape[2:])
    return {"name": name, "kind": "video" if t > 1 else "image", "latent_h": h, "latent_w": w, "latent_t": t,
            "mode": MODES[mode], "source": source, "source_shape": source_shape, "pool": pool,
            "optimize_steps": int(refine_steps), "tags": tags, "description": description, "concept_type": concept,
            "_format_version": FORMAT, "sample_rate": AUDIO_RATE}


def describe_audio(name: str, latent: torch.Tensor, tags: list[str], description: str = "",
                   concept: str = "voice") -> dict:
    """An audio RefMod's header description (format 4, as the pack writes one)."""
    t = int(latent.shape[-1])
    return {"name": name, "kind": "audio", "latent_h": 0, "latent_w": 0, "latent_t": t, "mode": MODES["full"],
            "source": "audio", "source_shape": "x".join(str(x) for x in latent.shape[1:]), "pool": "none",
            "optimize_steps": 0, "tags": tags, "description": description, "concept_type": concept,
            "_format_version": FORMAT, "sample_rate": AUDIO_RATE}


# --- latents ----------------------------------------------------------------
def tokens(latent: torch.Tensor) -> int:
    """The reference tokens a (1, 24, T, h, w) latent puts into the model: one per 2x2 cell of each frame; an audio
    one (1, 32, 2, T), one per latent of each stereo channel."""
    if latent.dim() == 4:
        return 2 * int(latent.shape[-1])
    t, h, w = latent.shape[2:]
    return int(t) * (int(h) // 2) * (int(w) // 2)


def ref_size(width: int, height: int, short_edge: int) -> tuple[int, int]:
    """A reference's encode size: scaled down (never up) to ``short_edge``, in multiples of 32 (the VAE's 16 times
    the model's 2x2 patch)."""
    scale = min(1.0, short_edge / min(width, height))
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def aspect_grid(long_edge: int, width: int, height: int) -> tuple[int, int]:
    """(h, w) of a compressed grid: ``long_edge`` cells along the longer side, the other side keeping the shape, both
    even (the model's 2x2 patch), so a portrait subject isn't squeezed into a square."""
    if height >= width:
        h, w = long_edge, long_edge * width / height
    else:
        h, w = long_edge * height / width, long_edge
    return max(2, round(h / 2) * 2), max(2, round(w / 2) * 2)


def pool(z: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    """Average ``z`` (1, 24, T, H, W) down to (1, 24, t, h, w)."""
    if tuple(z.shape[2:]) == (t, h, w):
        return z
    return F.adaptive_avg_pool3d(z.float(), (t, h, w))


def refine(small: torch.Tensor, targets: list[torch.Tensor], steps: int, lr: float = 0.02) -> torch.Tensor:
    """``small`` adjusted so that, scaled up (trilinear) to each target's size, it matches the targets as closely as
    it can (mean squared error, averaged over them): nothing but the small latent is learned."""
    if steps <= 0:
        return small
    targets = [t.float() for t in targets]
    param = torch.nn.Parameter(small.detach().clone().float())
    opt = torch.optim.Adam([param], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            loss = sum(F.mse_loss(F.interpolate(param, size=tuple(t.shape[2:]), mode="trilinear",
                                                align_corners=False), t) for t in targets) / len(targets)
            loss.backward()
            opt.step()
    return param.detach()


def distinct_frames(z: torch.Tensor, threshold: float = DEDUP) -> list[int]:
    """The frames of ``z`` that differ from the last one kept by at least ``threshold`` (mean absolute difference
    over the two frames' mean magnitude): a still shot's frames, which differ by little more than noise, keep one."""
    frames = z[0].float().transpose(0, 1)  # (T, 24, h, w)
    kept = [0]
    for i in range(1, frames.shape[0]):
        a, b = frames[kept[-1]], frames[i]
        if (b - a).abs().mean() / ((a.abs().mean() + b.abs().mean()) / 2 + 1e-6) >= threshold:
            kept.append(i)
    return kept


def fit_tokens(z: torch.Tensor, budget: int, log=print) -> torch.Tensor:
    """``z`` cut to at most ``budget`` tokens (0: no cap), by time only (resizing a frame would move its reference
    grid): near-duplicate frames go first, then the rest are spread evenly."""
    per_frame = tokens(z[:, :, :1])
    if budget <= 0 or tokens(z) <= budget:
        return z
    if per_frame > budget:
        raise ValueError(f"One frame takes {per_frame} tokens, more than the cap of {budget}: lower the size "
                         "(or the grid), or raise the cap")
    kept = distinct_frames(z)
    if len(kept) < z.shape[2]:
        log(f"  over the {budget}-token cap: dropped {z.shape[2] - len(kept)} near-duplicate frame(s)")
        z = z[:, :, kept]
    fit = budget // per_frame
    if z.shape[2] > fit:
        idx = torch.linspace(0, z.shape[2] - 1, fit).round().long()
        log(f"  still over the cap: kept {fit} of {z.shape[2]} frames, spread evenly")
        z = z[:, :, idx]
    return z


def weaken(z: torch.Tensor, strength: float) -> torch.Tensor:
    """A reference at ``strength`` (0–1): blended toward a blurred copy of itself (an eighth of its size, scaled
    back up), which keeps it a plausible latent while taking away the detail that makes it a strong reference
    (blending toward zero or noise would read as grey, or as some other picture). An audio one (1, 32, 2, T) is
    blurred along time, as the pack does."""
    strength = min(1.0, max(0.0, float(strength)))
    if strength >= 1.0:
        return z
    if z.dim() == 4:
        t = z.shape[-1]
        if t <= 1:
            return z
        flat = z.reshape(-1, 1, t).float()
        blur = F.interpolate(F.adaptive_avg_pool1d(flat, max(1, t // 8)), size=t, mode="linear",
                             align_corners=False).reshape(z.shape)
        return strength * z.float() + (1.0 - strength) * blur
    t, h, w = z.shape[2:]
    blur = F.interpolate(F.adaptive_avg_pool3d(z.float(), (t, max(1, h // 8), max(1, w // 8))), size=(t, h, w),
                         mode="trilinear", align_corners=False)
    return strength * z.float() + (1.0 - strength) * blur


# --- making one -------------------------------------------------------------
def sources(datasets: list[dict]) -> list[str]:
    """The images and clips of ``datasets``, in order (each dataset's ``_refs`` folder isn't one of them)."""
    out = []
    for d in datasets:
        out += [p for p in ds.list_media(d["path"]) if ds.media_kind(p) in ("image", "video")]
    return out


def audio_sources(datasets: list[dict]) -> list[str]:
    """The sound files of ``datasets`` and their clips with a soundtrack, in order."""
    out = []
    for d in datasets:
        for p in ds.list_media(d["path"]):
            kind = ds.media_kind(p)
            if kind == "audio" or (kind == "video" and ds.probe_video(p).get("has_audio")):
                out.append(p)
    return out


def audio_rows(z: torch.Tensor) -> torch.Tensor:
    """An audio RefMod's (1, 32, 2, T) -> the model's (2T, 32) channel-major rows (channel 0's T, then channel 1's)."""
    return z[0].permute(1, 2, 0).reshape(-1, z.shape[1])


def read_sound(path: str, seconds: float, info: dict | None = None) -> torch.Tensor:
    """A sound's first ``seconds`` as (2, samples) at 32 kHz; a mono one in both channels at its own level, as the
    released pipeline brings one up to stereo (ffmpeg's own upmix is 3 dB quieter). ``info``: its ``probe_audio``."""
    if (info or ds.probe_audio(path))["channels"] == 1:
        return ds.read_audio(path, seconds, AUDIO_RATE, 1).repeat(2, 1)
    return ds.read_audio(path, seconds, AUDIO_RATE, 2)


def audio_latents(seconds: float) -> int:
    """The audio latents of ``seconds`` of sound (the VAE pads the last one out)."""
    return -(-int(seconds * AUDIO_RATE) // AUDIO_HOP)


@torch.no_grad()
def encode_audio(audio_vae, wav: torch.Tensor, device) -> torch.Tensor:
    """A (2, samples) waveform at 32 kHz -> (1, 32, 2, T): the audio VAE's encode of each stereo channel,
    ``AUDIO_CHUNK`` seconds at a time."""
    chunk = AUDIO_CHUNK * AUDIO_RATE
    parts = []
    for start in range(0, wav.shape[-1], chunk):
        z = audio_vae.encode(wav[:, None, start:start + chunk].to(device))  # (2, 32, t)
        parts.append(z.float().cpu())
    return torch.cat(parts, dim=-1).permute(1, 0, 2)[None]


def _media_size(path: str) -> tuple[int, int]:
    if ds.media_kind(path) == "video":
        info = ds.probe_video(path)
        return info["width"], info["height"]
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).size


def _pixels(path: str, size: tuple[int, int], frames: int, fps: int, snap) -> torch.Tensor:
    """A reference as (1, 3, T, H, W) in [-1, 1], cover-cropped to ``size``: an image, or a clip's first ``frames``
    frames at ``fps`` (fewer if it's shorter, snapped down to a length the VAE takes; a clip too short for one is
    its first frame)."""
    if ds.media_kind(path) != "video":
        return ds.load_and_crop(path, size).unsqueeze(2)
    have = int(ds.probe_video(path)["duration"] * fps)
    n = min(frames, have)
    return ds.read_video_frames(path, size, snap(n) if n >= 5 else 1, fps)  # under 5: the VAE's shortest clip


def make_audio(driver, paths: list[str], s: dict, log=print, check=lambda: None) -> tuple[torch.Tensor, dict]:
    """An audio RefMod of ``paths`` (sound files and clips' soundtracks), each in turn from its start until there
    are ``refmod_seconds`` in all, encoding through ``driver.audio_vae``: -> (latent (1, 32, 2, T) float32, the
    description for ``save``, less its name). A token cap trims the end."""
    total = float(s["refmod_seconds"])
    parts, used = [], []
    left = total
    for path in paths:
        if left <= 0:
            break
        check()
        try:
            info = ds.probe_audio(path)
        except ValueError:
            log(f"  {os.path.basename(path)}: no sound, skipped")
            continue
        have = info["duration"]
        take = min(left, have)
        if take * AUDIO_RATE < AUDIO_HOP:
            continue
        parts.append(encode_audio(driver.audio_vae, read_sound(path, take, info), driver.device))
        used.append(path)
        left -= take
        log(f"  {len(used)}. {os.path.basename(path)}: {take:.1f} s" + (f" of {have:.1f}" if take < have else ""))
    if not parts:
        raise RuntimeError("None of the selected datasets' files has any sound")
    latent = torch.cat(parts, dim=-1)
    tags = [f"{len(used)} sound(s), {latent.shape[-1] / AUDIO_PER_SECOND:.1f} s"]
    cap = int(s.get("refmod_max_tokens") or 0)
    if cap and tokens(latent) > cap:
        if cap < 2:
            raise ValueError("The token cap is below one audio latent (2 tokens)")
        latent = latent[..., : cap // 2]
        log(f"  over the {cap}-token cap: kept the first {latent.shape[-1] / AUDIO_PER_SECOND:.1f} s")
        tags.append(f"capped at {cap} tokens")
    meta = describe_audio("", latent, tags, str(s.get("refmod_description") or "").strip(),
                          str(s.get("refmod_concept") or "voice"))
    return latent, meta


def make(driver, paths: list[str], s: dict, log=print, check=lambda: None) -> tuple[torch.Tensor, dict]:
    """A RefMod of ``paths`` (images and clips) with the job's settings ``s``, encoding through ``driver`` (its video
    VAE loaded): -> (latent (1, 24, T, h, w) float32, the description for ``save``, less its name)."""
    mode = s["refmod_mode"]
    short = int(s["refmod_size"])
    frames = int(s["refmod_frames"])
    sizes = [_media_size(p) for p in paths]
    canvas = ref_size(*sizes[0], short) if mode == "full" else None
    grid = aspect_grid(int(s["refmod_grid"]), *sizes[0]) if mode == "compressed" else None
    steps = int(s["refmod_refine"]) if mode == "compressed" else 0
    merge = mode == "compressed" and bool(s.get("refmod_merge")) and len(paths) > 1
    parts, fulls, shapes = [], [], []
    n_img = n_vid = 0
    for i, (path, (w, h)) in enumerate(zip(paths, sizes)):
        check()
        size = canvas or ref_size(w, h, short)
        px = _pixels(path, size, frames, driver.fps, driver.snap_frames)
        n_vid += px.shape[2] > 1
        n_img += px.shape[2] == 1
        try:
            z = driver.encode_reference(px).float().cpu()  # (1, 24, t, H/16, W/16)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(f"Out of VRAM encoding {os.path.basename(path)} at {size[0]}×{size[1]}"
                               + (f" ({px.shape[2]} frames)" if px.shape[2] > 1 else "")
                               + ": lower the size" + (" or the frames per clip" if px.shape[2] > 1 else ""))
        del px
        shapes.append("x".join(str(x) for x in z.shape[2:]))
        if mode == "full":
            parts.append(z)
        else:
            small = pool(z, z.shape[2], *grid)
            if merge:
                parts.append(small)
                fulls.append(z)
            else:
                parts.append(refine(small, [z], steps))
        log(f"  {i + 1}/{len(paths)} {os.path.basename(path)}: {size[0]}×{size[1]}"
            + (f", {z.shape[2]} latent frames" if z.shape[2] > 1 else ""))
    if merge:
        # one grid for all of them: the references' grids averaged (at the longest one's frames), then refined
        # against every full encode at once, so it settles on what they have in common
        t = max(p.shape[2] for p in parts)
        start = torch.stack([pool(p, t, *grid) for p in parts]).mean(0)
        log(f"  merging {len(parts)} references into one {t}×{grid[0]}×{grid[1]} grid"
            + (f", refined over {steps} steps" if steps else ""))
        latent = refine(start, fulls, steps)
    else:
        if steps:
            log(f"  each grid refined over {steps} steps")
        latent = torch.cat(parts, dim=2)
    before = tokens(latent)
    latent = fit_tokens(latent, int(s.get("refmod_max_tokens") or 0), log)
    t, h, w = latent.shape[2:]
    pool_desc = (f"full-res {w * SCALE}x{h * SCALE}px (short-edge cap {short}px)" if mode == "full"
                 else f"{t}x{h}x{w}")
    source = f"merge {len(paths)} refs" if merge else "stack" if len(paths) > 1 else ("video" if n_vid else "image")
    tags = [f"{n_img} img, {n_vid} vid"] + ([f"merged {len(paths)} refs"] if merge else [])
    if tokens(latent) < before:
        tags.append(f"capped at {int(s['refmod_max_tokens'])} tokens")
    meta = describe("", latent, mode, source, " +".join(shapes), pool_desc, steps, tags,
                    str(s.get("refmod_description") or "").strip(), str(s.get("refmod_concept") or "generic"))
    return latent, meta

"""Measure a family's activation memory for the VRAM projection (``MemoryProfile`` in its driver).

Builds the diffusion model with two and then four of its real blocks (read from the real file, so the
quantized layer types match), attaches a rank-32 LoRA, and records the peak memory of a training step
(with and without gradient checkpointing) and of an inference forward at a few sizes. The VAE's
encode and decode are measured at the same sizes. Prints the fitted per-token / per-pixel constants.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/measure_memory.py krea2 \\
        --dit .../krea2_turbo_fp8_scaled.safetensors --vae .../qwen_image_vae.safetensors

Music families measure clips of ``--sizes`` seconds instead (their VAE is in the checkpoint):

    PYTHONPATH=. .venv/bin/python scripts/measure_memory.py yue2 --dit .../yue2_3b_int8_convrot.safetensors \\
        --sizes 10,30,60
"""

from __future__ import annotations

import argparse
import gc
import random

import torch

from dainamo.engine import comfy_io
from dainamo.engine.families import get_family
from dainamo.engine.families.base import block_index, count_blocks, transformer_shapes
from dainamo.engine.fp8_linear import assert_materialized, materialize
from dainamo.engine.lora import LoRANetwork

TEXT = 256  # caption tokens in the dummy conditions (MemoryProfile.text_tokens)
SONG_TEXT = 64  # music: prompt tokens in the dummy conditions


def dummy(family: str, w: int, h: int, frames: int):
    """(payload, cond, settings) shaped like the family's cached training inputs (music: ``w`` seconds)."""
    bf = torch.bfloat16
    if family == "yue2":
        from dainamo.engine.families.yue2 import model as M

        kv = torch.randn(M.LAYERS, SONG_TEXT, M.KV_HEADS, M.HEAD_DIM, dtype=bf)
        return {"latent": torch.randn(w * M.FRAME_RATE, M.LATENT_DIM)}, {"text": {"k": kv, "v": kv}}, {}
    if family == "krea2":
        return {"latent": torch.randn(16, h // 8, w // 8)}, {"text": torch.randn(TEXT, 12, 2560, dtype=bf)}, {}
    if family == "moss_image":
        return {"latent": torch.randn(128, h // 16, w // 16)}, {"text": torch.randn(TEXT, 4096, dtype=bf)}, {}
    if family == "qwen_image_2":
        cond = {"embeds": torch.randn(TEXT, 4096, dtype=bf), "slots": torch.zeros(TEXT, dtype=torch.bool)}
        return {"latent": torch.randn(64, h // 16, w // 16), "refs": []}, {"text": cond}, {}
    if family == "minimax_h3":
        from dainamo.engine.families.minimax_h3 import packing as P

        t_lat = P.video_latent_num_frames(frames) if frames > 1 else 1
        cond = {"embeds": torch.randn(TEXT, 5120, dtype=bf), "tags": torch.full((TEXT,), P.TEXT_TAG)}
        payload = {"video": torch.randn(24, t_lat, h // 16, w // 16), "frames": frames,
                   "first": torch.randn(24, 1, h // 16, w // 16),
                   "audio": torch.randn(P.audio_latent_num_frames(frames) * 2, 32)}
        # worst case: every step trains with the first frame as a keyframe, and with audio
        return payload, {"text": cond, "i2v": cond}, {"i2v_probability": 1.0, "train_audio": True}
    if family == "ltx2_5":
        from dainamo.engine.families.ltx2 import model as M

        t_lat = M.latent_frames(frames) if frames > 1 else 1
        cond = {"video": torch.randn(M.TEXT_TOKENS, 4096, dtype=bf),
                "audio": torch.randn(M.TEXT_TOKENS, 2048, dtype=bf)}
        payload = {"video": torch.randn(128, t_lat, h // 32, w // 32), "frames": frames,
                   "audio": torch.randn(M.audio_latent_frames(frames, 24), 128)}
        # worst case: every step trains with the first frame (per-token timesteps), and with audio
        return payload, {"text": cond}, {"i2v_probability": 1.0, "train_audio": True}
    raise SystemExit(f"no dummy inputs for {family}")


def build(driver, path: str, depth: int):
    prefix, shapes = transformer_shapes(path)
    n = count_blocks(shapes, driver.block_list)
    keep = {k: v for k, v in shapes.items()
            if not k.startswith(driver.block_list + ".") or block_index(k, driver.block_list) < depth}
    model = driver.build_transformer(keep)
    skip = tuple(f"{driver.block_list}.{i}." for i in range(depth, n)) + driver.text_encoder_parts
    loaded = comfy_io.load_state_dict(path, strip_prefix=prefix or None, only_prefixed=bool(prefix),
                                      skip_prefixes=skip, device=driver.device)
    driver.check_transformer_state(path, loaded)
    materialize(model, loaded, driver.dtype, keep_fp8=True, device=None, dtype_for=driver.transformer_dtype_for)
    assert_materialized(model)
    return model


def peak(fn) -> float:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def measure_depth(driver, args, depth: int, sizes: list[int], rng) -> dict:
    """Peak memory of training steps (with and without checkpointing) and of inference passes."""
    dev = driver.device
    runs = {}
    model = build(driver, args.dit, depth)
    network = LoRANetwork(model, driver.lora_targets, 32, 32.0)
    network.to(dev, torch.float32)
    model.train()
    for ckpt in (True, False):
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing(ckpt)
        else:
            model.gradient_checkpointing = ckpt
        for s in sizes:
            payload, cond, settings = dummy(args.family, s, s, args.frames)

            def step():
                loss = driver.training_loss(model, [payload], [cond], settings, rng=rng)
                loss.backward()
            runs[(depth, ckpt, s)] = peak(step)
            for p in network.parameters():
                p.grad = None
    model.eval()
    for s in sizes:
        payload, cond, settings = dummy(args.family, s, s, args.frames)

        @torch.no_grad()
        def forward():
            driver.training_loss(model, [payload], [cond], settings, rng=rng)
        runs[(depth, "infer", s)] = peak(forward)
    return runs


def fit(xs, ys):
    """Least-squares slope and intercept."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", default="")
    ap.add_argument("--audio-vae", default="")
    ap.add_argument("--sizes", default="512,768,1024")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--depths", default="2,4", help="two block counts; their difference isolates the per-block cost")
    args = ap.parse_args()

    dev = torch.device("cuda")
    profile = {"diffusion_model": args.dit, "vae": args.vae, "audio_vae": args.audio_vae}
    driver = get_family(args.family)(profile, dev, log=lambda *_: None)
    sizes = [int(s) for s in args.sizes.split(",")]
    text = driver.memory.text_tokens if driver.memory else 256
    # the dummy captions' share of the sequence (LTX's text is cross-attention context, not in its sequence)
    dummy_text = 0 if args.family == "ltx2_5" else TEXT
    if driver.is_audio:
        tokens = {s: driver.sequence_tokens(0, 0, s * driver.fps) - text + SONG_TEXT for s in sizes}
    else:
        tokens = {s: driver.sequence_tokens(s, s, args.frames) - text + dummy_text for s in sizes}
    rng = random.Random(0)

    lo, hi = (int(x) for x in args.depths.split(","))
    runs = {}
    for depth in (lo, hi):
        runs.update(measure_depth(driver, args, depth, sizes, rng))

    gb = 1e9
    for key, v in sorted(runs.items(), key=str):
        print(f"  depth {key[0]} {str(key[1]):5} {key[2]:5}{'s' if driver.is_audio else 'px'}  "
              f"T={tokens[key[2]]:6}  {v / gb:6.2f} GB")
    ts = [tokens[s] for s in sizes]
    saved = [(runs[(hi, True, s)] - runs[(lo, True, s)]) / (hi - lo) / tokens[s] for s in sizes]
    full = [(runs[(hi, False, s)] - runs[(lo, False, s)]) / (hi - lo) / tokens[s] for s in sizes]
    work, c_work = fit(ts, [runs[(lo, True, s)] - lo * saved[i] * tokens[s] for i, s in enumerate(sizes)])
    infer, c_infer = fit(ts, [runs[(lo, "infer", s)] for s in sizes])
    print(f"saved per token per block: {[round(x) for x in saved]}")
    print(f"full per token per block:  {[round(x) for x in full]}")
    print(f"work per token: {work:.0f} (+{c_work / gb:.2f} GB)   infer per token: {infer:.0f} (+{c_infer / gb:.2f} GB)")

    # VAE
    before = torch.cuda.memory_allocated()
    driver.load_vae()
    print(f"VAE weights on the GPU: {(torch.cuda.memory_allocated() - before) / gb:.2f} GB")
    enc, dec = [], []
    for s in sizes:
        with torch.no_grad():
            if driver.is_audio:
                wav = torch.randn(2, s * driver.fps * 1920) * 0.1  # 1920 samples a latent frame
                lat = driver.vae.encode(wav)
                enc.append(peak(lambda: driver.vae.encode(wav)))
                dec.append(peak(lambda: driver.vae.decode(lat)))
            elif args.family in ("minimax_h3", "ltx2_5"):
                px = torch.randn(1, 3, args.frames, s, s, device=dev)
                lat = driver.video_vae.encode(px)
                enc.append(peak(lambda: driver.video_vae.encode(px)))
                dec.append(peak(lambda: driver.video_vae.decode(lat)))
            else:
                px = torch.rand(1, 3, s, s) * 2 - 1
                lat = driver.encode_image(px)
                enc.append(peak(lambda: driver.encode_image(px)))
                dec.append(peak(lambda: driver.decode_latents(lat[None])))
        print(f"  {s}{'s' if driver.is_audio else 'px'}: encode {enc[-1] / gb:.2f} GB, decode {dec[-1] / gb:.2f} GB")
    pixels = [s * driver.fps if driver.is_audio else s * s * args.frames for s in sizes]
    unit = "latent frame" if driver.is_audio else "pixel"
    print(f"vae_encode per {unit}: {max(e / p for e, p in zip(enc, pixels)):.0f}, "
          f"vae_decode per {unit}: {max(d / p for d, p in zip(dec, pixels)):.0f}")


if __name__ == "__main__":
    main()

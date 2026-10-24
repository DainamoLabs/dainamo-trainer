"""VRAM projection for a training job or a Sandbox generation, from the model files' headers (no
weights are read) and the family's measured activation memory (``MemoryProfile``).

A run goes through phases that never overlap on the GPU: caption encoding (text encoder), image
encoding (VAE), then training steps and sampling (diffusion model, plus the VAE to decode). Only the
last two hold the diffusion model, so only they depend on how many of its blocks are offloaded. The
projection lists each block's size so a client can total any offload setting instantly:

    weights(k) = resident_gb + sum(block_gb[: blocks - k]) + (inflight_gb if k else 0)
    phase(k)   = (weights(k) if phase["model"] else 0) + sum(phase["parts"]) + overhead_gb

``phase_totals`` does exactly this, and the web UI mirrors it. In low VRAM mode the text encoder's layers
and large parts stream from system RAM too (``streamed_text_encoder``), so caption encoding holds only the
rest of it, and samples are made in two steps: denoising without the VAE (and with the optimizer state in
RAM), then decoding with the diffusion model and LoRA in RAM (if the GPU hasn't room for them beside the VAE;
the projection assumes it hasn't). Training steps keep the blocks' saved activations in system RAM
(``offload.ActivationOffload``), with only a few on the GPU at a time.

Each phase also lists the system RAM it holds (``ram_totals``): its ``ram`` parts, plus the diffusion
model's streamed blocks (``model_ram`` "streamed", the default for a phase with the model) or all of it
("all": parked while samples decode), plus the process itself.

A full finetune (``engine.finetune``) holds the whole model in bf16, quantized layers expanded; its optimizer
state is the trained part beside it. Its offloaded blocks train as they stream, and a stepped block's copy stays
on the GPU until it's copied back to RAM: one more block in flight.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import struct
from functools import lru_cache

from .dataset import resolutions
from .families.base import block_index
from .lora import factorize
from .offload import BIG_MODULE

GB = 1024**3  # GiB, as nvidia-smi reports memory
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
QUANT_DTYPES = ("I8", "U8", "F8_E4M3", "F8_E5M2")
# ComfyUI per-layer JSON blobs and markers: read at load, never kept
DROPPED_SUFFIXES = (".comfy_quant", ".comfy_attention.config", "scaled_fp8")
# quantization scales: quantized layers keep them in fp32
SCALE_SUFFIXES = (".weight_scale", ".input_scale", ".scale_weight", ".scale_input", ".weight_scale_2",
                  ".pre_quant_scale")
# optimizer state per trainable parameter, on top of the fp32 weight and gradient
OPTIMIZER_BYTES = {"adamw8bit": 2, "adamw": 8, "adafactor": 4, "prodigy": 16}
CONTEXT_BYTES = 0.5 * GB  # CUDA context and cuBLAS workspaces, outside PyTorch's allocator
FRAGMENTATION = 0.03  # allocator slack, as a share of the peak
# streamed blocks' worth of VRAM while offloading: the running and prefetched copies, plus the one
# autograd still holds during backward (on Krea 2 at 50%, within 0.7 GB of nvidia-smi's peak)
IN_FLIGHT_BLOCKS = 3
# system RAM a training or Sandbox process holds besides its models: Python, PyTorch, CUDA and the libraries
# (1.7-2.5 GB measured in H3 and LTX 2.5 jobs), and the room left in the pinned slabs (``offload.PinnedArena``)
PROCESS_RAM = 4 * GB
# a text encoder's repeated layers (a language model's ``layers``, a vision tower's ``blocks``), which low VRAM
# mode streams (``offload.stream_layers``) with its large parts: the running layer and the next one, or one
# large part, are on the GPU at a time
TEXT_LAYER = re.compile(r"(?:^|\.)(?:layers|blocks)\.\d+\.")
TEXT_IN_FLIGHT = 2
# blocks' saved activations on the GPU at a time in low VRAM mode, where the rest wait in system RAM: one
# being copied out while the next is made, in backward the one in use and the one fetched ahead, and the
# first block's input (which needs no grad, so it stays)
ACTIVATIONS_IN_FLIGHT = 3


@lru_cache(maxsize=32)
def _header(path: str, mtime: float) -> dict[str, tuple[str, tuple[int, ...]]]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 << 20:  # not a .safetensors file (a .gguf's first bytes would read it all)
            raise ValueError("not a .safetensors file")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in header.items()}


def tensors(path: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """{key: (safetensors dtype, shape)} from a .safetensors header, or the headers of every shard of a sharded
    checkpoint (a folder or an index file, ``comfy_io.shard_files``)."""
    from .comfy_io import shard_files

    out: dict[str, tuple[str, tuple[int, ...]]] = {}
    for p in shard_files(os.path.expanduser(path)):
        out.update(_header(p, os.path.getmtime(p)))
    return out


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def gpu_bytes(key: str, dtype: str, shape, float_bytes: int = 2) -> int:
    """Bytes a tensor takes once loaded: quantized codes stay as they are, other floats become the
    compute dtype (``float_bytes``), quantization metadata blobs are dropped."""
    if key.endswith(DROPPED_SUFFIXES):
        return 0
    n = _numel(shape)
    if dtype in QUANT_DTYPES:
        return n
    if dtype in ("F32", "F16", "BF16", "F64"):
        return n * (4 if key.endswith(SCALE_SUFFIXES) else float_bytes)
    return n * DTYPE_BYTES.get(dtype, 4)


def file_gpu_bytes(path: str, skip: tuple[str, ...] = (), fp32: tuple[str, ...] = ()) -> int:
    """Loaded size of a whole file (text encoder, VAE), skipping keys that contain any of ``skip``; float
    tensors whose keys contain any of ``fp32`` load in fp32."""
    if not path or not os.path.exists(os.path.expanduser(path)):
        return 0
    return sum(_loaded_bytes(k, dt, shape, fp32) for k, (dt, shape) in tensors(path).items()
               if not any(s in k for s in skip))


def _loaded_bytes(key: str, dtype: str, shape, fp32: tuple[str, ...] = ()) -> int:
    b = gpu_bytes(key, dtype, shape, 4 if any(s in key for s in fp32) else 2)
    if dtype in QUANT_DTYPES and key.endswith("embed_tokens.weight"):
        b *= 2  # quantized embeddings are expanded to bf16
    return b


def streamed_text_encoder(driver_cls, path: str) -> tuple[int, int, int]:
    """(bytes left on the GPU, bytes streamed from RAM, bytes in flight) of a text encoder file in low VRAM mode:
    its layers, and each module outside them of ``offload.BIG_MODULE`` or more, stream."""
    if not path or not os.path.exists(os.path.expanduser(path)):
        return 0, 0, 0
    skip, fp32 = driver_cls.text_encoder_skip, driver_cls.text_encoder_fp32
    layers, parts = {}, {}
    for k, (dt, shape) in tensors(path).items():
        if any(s in k for s in skip):
            continue
        b, m = _loaded_bytes(k, dt, shape, fp32), TEXT_LAYER.search(k)
        group, key = (layers, k[:m.end()]) if m else (parts, k.rsplit(".", 1)[0])
        group[key] = group.get(key, 0) + b
    big = [b for b in parts.values() if b >= BIG_MODULE]
    streamed = sum(layers.values()) + sum(big)
    inflight = max(TEXT_IN_FLIGHT * max(layers.values(), default=0), max(big, default=0))
    return file_gpu_bytes(path, skip, fp32) - streamed, streamed, inflight


def bundled_bytes(path: str, prefix: str, float_bytes: int) -> int:
    """Loaded size of the part of a single-file checkpoint under ``prefix`` (YuE2's composer or VAE), with
    every weight at ``float_bytes`` (quantized ones are expanded)."""
    return sum(_numel(shape) * float_bytes for k, (dt, shape) in tensors(path).items()
               if k.startswith(prefix) and not k.endswith(DROPPED_SUFFIXES + SCALE_SUFFIXES))


def lora_params(shapes: dict, targets: list[str], rank: int, kind: str = "lora", factor: int = -1) -> int:
    """Trainable parameters of a LoRA (or a LoKr split by ``factor``) on the Linear weights matching
    ``targets``."""
    total = 0
    for k, (dt, shape) in shapes.items():
        if not k.endswith(".weight") or len(shape) != 2:
            continue
        name = k[: -len(".weight")]
        if any(fnmatch.fnmatchcase(name, p) for p in targets):
            out_f, in_f = shape
            if dt == "U8":
                in_f *= 2  # nvfp4: two codes per byte
            if kind == "lokr":
                (out_m, out_n), (in_m, in_n) = factorize(out_f, factor), factorize(in_f, factor)
                total += out_m * in_m + out_n * in_n
            else:
                total += rank * (in_f + out_f)
    return total


def adapter_bytes(path: str) -> int | None:
    """GPU bytes of a training adapter (bf16), or None when it isn't on disk yet."""
    path = (path or "").strip()
    if not path:
        return 0
    local = os.path.expanduser(path)
    if not os.path.exists(local):
        parts = path.split("/")
        if len(parts) < 3 or path.startswith("/"):
            return None
        try:
            from huggingface_hub import try_to_load_from_cache
            local = try_to_load_from_cache("/".join(parts[:2]), "/".join(parts[2:]))
        except Exception:
            local = None
        if not isinstance(local, str):
            return None
    # every tensor ``lora.adapter_layers`` loads (a LoRA's lora_A/lora_B, a LoKr's lokr_w1/lokr_w2 or their low-rank
    # halves, whole changes' diff/diff_b), but not the alphas, which it reads as numbers
    return sum(_numel(shape) * 2 for k, (_, shape) in tensors(local).items()
               if not k.endswith(".alpha") and _numel(shape) > 1)


def model_weights(driver_cls, path: str, expand: bool = False):
    """(resident bytes outside the blocks, [bytes per block], {key: (dtype, shape)} without prefix). ``expand``: as a
    finetune loads it, every weight in bf16 (or its wider dtype) and no quantization scales."""
    raw = tensors(path)
    full = any(k.startswith("model.diffusion_model.") for k in raw)
    shapes = {k.removeprefix("model.diffusion_model."): v for k, v in raw.items()
              if not full or k.startswith("model.diffusion_model.")}
    head = driver_cls.block_list + "."
    blocks: dict[int, int] = {}
    resident = 0
    probe = driver_cls.__new__(driver_cls)  # transformer_dtype_for needs no state
    for k, (dt, shape) in shapes.items():
        if k.startswith(driver_cls.text_encoder_parts):  # loaded with the text encoder
            continue
        wide = probe.transformer_dtype_for(k)
        width = 4 if wide is not None and "32" in str(wide) else 2
        if expand:
            b = 0 if k.endswith(DROPPED_SUFFIXES + SCALE_SUFFIXES) else _numel(shape) * (2 if dt == "U8" else 1) * width
        else:
            b = gpu_bytes(k, dt, shape, float_bytes=width)
        if k.startswith(head):
            i = block_index(k, driver_cls.block_list)
            blocks[i] = blocks.get(i, 0) + b
        else:
            resident += b
    return resident, [blocks[i] for i in sorted(blocks)], shapes


def finetune_parts(shapes: dict, optimizer: str) -> tuple[int, int, int]:
    """(trainable weights, optimizer-state bytes, bytes of one weight's update) of a finetune of the model with these
    {key: (dtype, shape)}. Adafactor keeps a row and a column per matrix (all of a vector), AdamW 8-bit 2 bytes a
    weight; a weight's update holds its fp32 copy and gradient, its bf16 gradient and the rounding's temporaries."""
    params, state, largest = 0, 0, 0
    for k, (dt, shape) in shapes.items():
        if k.endswith(DROPPED_SUFFIXES + SCALE_SUFFIXES):
            continue
        shape = list(shape)
        if dt == "U8" and len(shape) == 2:
            shape[-1] *= 2  # nvfp4: two codes per byte
        n = _numel(shape)
        params += n
        largest = max(largest, n)
        if optimizer == "adafactor":
            state += 4 * (_numel(shape[:-1]) + _numel(shape[:-2] + shape[-1:]) if len(shape) >= 2 else n)
        else:
            state += 2 * n
    return params, state, largest * 18


def _vae_work(spec: tuple[float, float], pixels: int, cap: int | None = None) -> float:
    base, per_pixel = spec
    return base + per_pixel * (min(pixels, cap) if cap else pixels)


def vae_bytes(driver_cls, profile: dict) -> int:
    """The VAE(s) on the GPU, including one bundled in the diffusion-model file; in fp32 for a family that runs it
    so (``vae_fp32``)."""
    fp32 = ("",) if getattr(driver_cls, "vae_fp32", False) else ()  # "" is in every key
    total = sum(file_gpu_bytes(profile.get(f, ""), fp32=fp32) for f in ("vae", "audio_vae"))
    if "vae" in driver_cls.bundled:
        total += bundled_bytes(profile["diffusion_model"], *driver_cls.bundled["vae"])
    return total


def decode_bytes(driver, width: int, height: int, frames: int = 1, loaded: bool = False) -> int:
    """VRAM decoding one sample takes: its working memory, and the VAE unless it's ``loaded``."""
    mem = driver.memory
    work = _vae_work(mem.vae_decode, width * height * frames, mem.decode_tile_pixels) if mem else 2 * GB
    return int(work) + (0 if loaded else vae_bytes(driver, driver.profile))


def encode_bytes(driver_cls, profile: dict, pixels: int, loaded: bool = False) -> int:
    """VRAM VAE-encoding ``pixels`` (width × height × frames) takes: its working memory, and the VAE unless it's
    ``loaded``."""
    mem = driver_cls.memory
    work = _vae_work(mem.vae_encode, pixels, mem.encode_tile_pixels) if mem else 2 * GB
    return int(work) + (0 if loaded else vae_bytes(driver_cls, profile))


def project(driver_cls, profile: dict, settings: dict, kind: str = "train") -> dict:
    """VRAM projection for a training job (``kind="train"``, ``settings`` as the job's) or a Sandbox
    generation (``kind="sample"``: width, height, frames, loras). Sizes are in GB."""
    mem = driver_cls.memory
    finetune = kind == "train" and bool(settings.get("finetune_enabled"))
    resident, blocks, shapes = model_weights(driver_cls, profile["diffusion_model"], expand=finetune)
    vae = vae_bytes(driver_cls, profile)
    te = file_gpu_bytes(profile.get("text_encoder", ""), skip=driver_cls.text_encoder_skip,
                        fp32=driver_cls.text_encoder_fp32)
    if "text_encoder" in driver_cls.bundled:  # part of the diffusion-model file
        te += bundled_bytes(profile["diffusion_model"], *driver_cls.bundled["text_encoder"])
    if te and driver_cls.text_encoder_parts:  # diffusion-model parts that run with the text encoder
        raw = tensors(profile["diffusion_model"])
        te += sum(gpu_bytes(k, dt, shape) for k, (dt, shape) in raw.items()
                  if k.removeprefix("model.diffusion_model.").startswith(driver_cls.text_encoder_parts))
    notes = []
    adapter = adapter_bytes(profile.get("training_adapter", ""))
    if adapter is None:
        notes.append("The training adapter isn't downloaded yet, so it isn't counted (usually under 1 GB).")
        adapter = 0
    video = "video" in driver_cls.info.media
    audio = "audio" in driver_cls.info.media
    low_vram = bool(settings.get("low_vram")) and driver_cls.info.streams_text_encoder
    checkpointing = settings.get("gradient_checkpointing", True)
    offload_saved = low_vram and checkpointing  # the blocks' saved activations wait in system RAM

    def tokens(w, h, f):
        return driver_cls.sequence_tokens(int(w), int(h), int(f or 1))

    def activations(n_tokens, train):
        if mem is None:
            return 0.0
        if not train:
            return mem.infer * n_tokens
        per_block = mem.saved if checkpointing else mem.full
        held = min(len(blocks), ACTIVATIONS_IN_FLIGHT) if offload_saved else len(blocks)
        return n_tokens * (held * per_block + mem.work)

    phases = []
    params = 0  # the trained adapter's (training only)
    if audio:
        return _project_audio(driver_cls, settings, kind, phases, notes, resident, blocks, shapes, vae, te, adapter,
                              tokens, activations)
    if kind == "train":
        frames = int(settings.get("num_frames", 1) or 1) if video else 1
        # each dataset at its largest size, clips at the clip length, with the most reference images a file of it
        # trains with (``datasets``: [{resolution, video, refs}]; without it, everything at the job's largest size);
        # the biggest of them sets the peak
        default = settings.get("resolution") or 1024
        sizes = [(max(resolutions(d.get("resolution") or default)), frames if d.get("video") else 1,
                  max(0, int(d.get("refs") or 0) - driver_cls.references_counted))
                 for d in settings.get("datasets") or []] or [(max(resolutions(default)), frames, 0)]
        res = max(r for r, _, _ in sizes)
        batch = int(settings.get("batch_size", 1) or 1)
        rank = int(settings.get("rank", 32) or 32)
        net = settings.get("network_type") or "lora"
        if finetune:  # the trained weights are the model's, counted with it, beside the training adapter (if any)
            model = {k: v for k, v in shapes.items() if not k.startswith(driver_cls.text_encoder_parts or ("\0",))}
            params, opt, update = finetune_parts(model, settings.get("optimizer", "adafactor"))
            name, lora, weights = "Weight updates", update, 0
            notes.append(f"A finetune trains every weight in bf16 and saves the whole model each time: about "
                         f"{params * 2 / 1e9:.1f} GB a file.")
            if any(dt in QUANT_DTYPES for k, (dt, _) in model.items() if not k.endswith(DROPPED_SUFFIXES)):
                notes.append("The model file is quantized: its layers are expanded to bf16 to train, so its rounding "
                             "is trained in. The bf16 release trains better.")
        else:
            params = lora_params(shapes, driver_cls.lora_targets, rank, net, int(settings.get("lokr_factor", -1) or -1))
            name = "LoKr" if net == "lokr" else "LoRA"
            # fp32 weights and gradients, plus the adapter; EMA keeps an fp32 average beside them (``engine.ema``)
            ema = params * 4 if settings.get("ema_enabled") else 0
            lora = params * 8 + adapter + ema
            weights = params * 4 + adapter + ema  # while sampling: no gradients
            opt = params * OPTIMIZER_BYTES.get(settings.get("optimizer", "adamw8bit"), 8)
        n_tok = max(tokens(r, r, f) + refs * driver_cls.reference_tokens(r, r) for r, f, refs in sizes) * batch
        saved = n_tok * len(blocks) * mem.saved if offload_saved and mem is not None else 0
        # a finetune's training adapter, apart from its weights (a LoRA's is counted with it)
        beside = [["Training adapter", adapter / GB]] if finetune and adapter else []
        phases.append({"id": "training", "label": "Training step", "model": True, "parts": [
            [name, lora / GB], ["Optimizer", opt / GB], *beside, ["Activations", activations(n_tok, True) / GB]],
            "ram": [["Saved activations", saved / GB]] if saved else []})
        if saved:
            notes.append(f"Low VRAM: training steps keep the blocks' saved activations ({saved / GB:.1f} GB at "
                         "the largest size) in system RAM and copy each back as the backward pass needs it.")
        sw, sh = int(settings.get("sample_width", res)), int(settings.get("sample_height", res))
        sf = int(settings.get("sample_frames", 1) or 1) if video else 1
        if mem is not None:
            decode = _vae_work(mem.vae_decode, sw * sh * sf, mem.decode_tile_pixels)
            infer = activations(tokens(sw, sh, sf), False)
            held = beside if finetune else [[name, weights / GB]]  # a finetune's are the model's
            if low_vram:  # denoise with the optimizer state in RAM, then decode with the model and LoRA there too
                # sample cards with images (a first frame, references) have the VAE encode them while sampling
                cards = [c for c in settings.get("sample_prompts") or [] if isinstance(c, dict)]
                with_vae = [["VAE", vae / GB]] if any(c.get("images") for c in cards) else []
                phases.append({"id": "sampling", "label": "Sampling", "model": True, "parts": [
                    *held, *with_vae, ["Working memory", infer / GB]], "ram": [["Optimizer", opt / GB]]})
                # a training adapter (if any) stays on the GPU while the LoRA and its optimizer state wait in RAM
                phases.append({"id": "decoding", "label": "Decoding samples", "model": False, "model_ram": "all",
                               "parts": [["Training adapter", adapter / GB], ["VAE", vae / GB],
                                         ["Working memory", decode / GB]],
                               "ram": [*([] if finetune else [[name, params * 4 / GB]]), ["Optimizer", opt / GB]]})
            else:
                phases.append({"id": "sampling", "label": "Sampling", "model": True, "parts": [
                    *([["Optimizer", opt / GB], *beside] if finetune else [[f"{name} + optimizer", (lora + opt) / GB]]),
                    ["VAE", vae / GB], ["Working memory", max(infer, decode) / GB]]})
            phases.append({"id": "latents", "label": "Encoding images", "model": False, "parts": [
                ["VAE", vae / GB], ["Working memory", _vae_work(mem.vae_encode, max(r * r * f for r, f, _ in sizes)) / GB]]})
        if not checkpointing:
            notes.append("Gradient checkpointing is off: every block's activations stay in VRAM, and "
                         "offloading saves little.")
    else:
        w, h = int(settings.get("width", 1024)), int(settings.get("height", 1024))
        f = int(settings.get("frames", 1) or 1) if video else 1
        loras = sum(os.path.getsize(p) for p in (settings.get("loras") or []) if os.path.exists(p))
        infer = activations(tokens(w, h, f), False)
        decode = _vae_work(mem.vae_decode, w * h * f, mem.decode_tile_pixels) if mem is not None else 0
        if low_vram:  # the VAE waits in RAM unless input images need it, and decodes with the model in RAM
            with_vae = bool(settings.get("images"))
            phases.append({"id": "generation", "label": "Generation", "model": True, "parts": [
                ["LoRAs", loras / GB], *([["VAE", vae / GB]] if with_vae else []), ["Working memory", infer / GB]],
                "ram": [] if with_vae else [["VAE", vae / GB]]})
            phases.append({"id": "decoding", "label": "Decoding", "model": False, "model_ram": "all",
                           "parts": [["LoRAs", loras / GB], ["VAE", vae / GB], ["Working memory", decode / GB]]})
        else:
            phases.append({"id": "generation", "label": "Generation", "model": True, "parts": [
                ["LoRAs", loras / GB], ["VAE", vae / GB], ["Working memory", max(infer, decode) / GB]]})
    if te:
        parts, ram = [["Text encoder", te / GB], ["Working memory", mem.text_work if mem else 0.8]], []
        if low_vram:
            _, streamed, inflight = streamed_text_encoder(driver_cls, profile.get("text_encoder", ""))
            if streamed:
                parts = [["Text encoder", (te - streamed) / GB], ["Streamed parts", inflight / GB], parts[1]]
                ram = [["Text encoder", te / GB]]  # all of it while it loads, then the streamed parts
                notes.append(f"Low VRAM: the text encoder's layers and large parts ({streamed / GB:.1f} GB) stream "
                             "from system RAM, which slows only " + (
                                 "caption encoding (once per caption; then they're cached)." if kind == "train"
                                 else "encoding a new prompt.")
                             + " Samples decode with the diffusion model in system RAM if the GPU hasn't room "
                             "for both.")
        # a Sandbox encodes new prompts with the diffusion model loaded; a job encodes captions before loading it
        phases.append({"id": "captions", "label": "Encoding captions" if kind == "train" else "Encoding the prompt",
                       "model": False, "model_ram": "streamed" if kind != "train" else None, "parts": parts,
                       "ram": ram})
    return _projection(mem, phases, notes, resident, blocks, params, IN_FLIGHT_BLOCKS + (1 if finetune else 0))


def _projection(mem, phases, notes, resident, blocks, params, in_flight: int = IN_FLIGHT_BLOCKS) -> dict:
    if mem is None:
        notes.append("No activation measurements for this model yet, so the projection counts weights only.")
    biggest = max(blocks) if blocks else 0
    return {
        "blocks": len(blocks),
        "block_gb": [b / GB for b in blocks],
        "resident_gb": resident / GB,
        "inflight_gb": in_flight * biggest / GB,
        "context_gb": CONTEXT_BYTES / GB,
        "fragmentation": FRAGMENTATION,
        "model_gb": (resident + sum(blocks)) / GB,
        "process_ram_gb": PROCESS_RAM / GB,
        "phases": phases,
        "notes": notes,
        "adapter_params": params,
    }


# the composer's key/value cache per token (YuE2: 28 layers x keys and values x 8 heads x 128, bf16),
# doubled for the CFG branch
COMPOSER_KV_BYTES = 28 * 2 * 8 * 128 * 2 * 2
MAX_SONG_SECONDS = 360


def _project_audio(driver_cls, settings, kind, phases, notes, resident, blocks, shapes, vae, te, adapter, tokens,
                   activations) -> dict:
    """Music (YuE2): clips of ``clip_seconds`` (0: whole songs, up to 6 minutes) train; songs sample with the
    composer (the text encoder) beside the diffusion model, and decode with it still loaded."""
    mem = driver_cls.memory
    rate = driver_cls.fps  # latent frames a second

    def frames(seconds):
        return int(float(seconds or 0) * rate) or MAX_SONG_SECONDS * rate

    params = 0
    if kind == "train":
        batch = int(settings.get("batch_size", 1) or 1)
        rank = int(settings.get("rank", 32) or 32)
        net = settings.get("network_type") or "lora"
        params = lora_params(shapes, driver_cls.lora_targets, rank, net, int(settings.get("lokr_factor", -1) or -1))
        name = "LoKr" if net == "lokr" else "LoRA"
        lora = params * 8 + adapter
        opt = params * OPTIMIZER_BYTES.get(settings.get("optimizer", "adamw8bit"), 8)
        clip = frames(settings.get("clip_seconds", 30))
        phases.append({"id": "training", "label": "Training step", "model": True, "parts": [
            [name, lora / GB], ["Optimizer", opt / GB], ["Activations", activations(tokens(0, 0, clip) * batch, True) / GB]]})
        held = [[f"{name} + optimizer", (lora + opt) / GB]]
        sample = frames(settings.get("sample_seconds", 30))
        encode = clip
    else:
        loras = sum(os.path.getsize(p) for p in (settings.get("loras") or []) if os.path.exists(p))
        held = [["LoRAs", loras / GB]]
        sample = frames(settings.get("seconds", 30))
        encode = 0
    if mem is not None:
        work = activations(tokens(0, 0, sample), False) + COMPOSER_KV_BYTES * sample
        phases.append({"id": "sampling" if kind == "train" else "generation", "label": "Sampling" if kind == "train"
                       else "Generation", "model": True, "parts": held + [["Composer", te / GB],
                                                                           ["Working memory", work / GB]]})
        phases.append({"id": "decoding", "label": "Decoding audio", "model": True, "parts": held + [
            ["Composer", te / GB], ["VAE", vae / GB],
            ["Working memory", _vae_work(mem.vae_decode, sample, mem.decode_tile_pixels) / GB]]})
        if encode:
            phases.append({"id": "latents", "label": "Encoding audio", "model": False, "parts": [
                ["VAE", vae / GB], ["Working memory", _vae_work(mem.vae_encode, encode, mem.encode_tile_pixels) / GB]]})
    if kind == "train":
        phases.append({"id": "captions", "label": "Encoding captions", "model": False,
                       "parts": [["Composer", te / GB], ["Working memory", 0.3]]})
        if not settings.get("gradient_checkpointing", True):
            notes.append("Gradient checkpointing is off: every block's activations stay in VRAM, and "
                         "offloading saves little.")
    return _projection(mem, phases, notes, resident, blocks, params)


def streamed_for(proj: dict, offload_pct: float) -> int:
    """Blocks streamed at an offload percentage (as ``FamilyDriver.streamed_blocks``)."""
    n = proj["blocks"]
    return max(0, min(n, math.floor(float(offload_pct) / 100 * n + 0.5)))


def weights_gb(proj: dict, streamed: int) -> float:
    n = proj["blocks"]
    return proj["resident_gb"] + sum(proj["block_gb"][: n - streamed]) + (proj["inflight_gb"] if streamed else 0.0)


def ram_totals(proj: dict, streamed: int) -> dict[str, float]:
    """GB of system RAM per phase with ``streamed`` blocks offloaded: the process, the phase's ``ram`` parts and
    the diffusion model's share (see the module docstring)."""
    n, out = proj["blocks"], {}
    for ph in proj["phases"]:
        held = ph.get("model_ram", "streamed" if ph["model"] else None)
        model = proj["model_gb"] if held == "all" else sum(proj["block_gb"][n - streamed:]) if held == "streamed" else 0
        out[ph["id"]] = proj["process_ram_gb"] + model + sum(v for _, v in ph.get("ram", []))
    return out


def phase_totals(proj: dict, streamed: int) -> dict[str, float]:
    """GB per phase with ``streamed`` blocks offloaded, including context and allocator slack."""
    out = {}
    for ph in proj["phases"]:
        allocated = (weights_gb(proj, streamed) if ph["model"] else 0.0) + sum(v for _, v in ph["parts"])
        out[ph["id"]] = allocated * (1 + proj["fragmentation"]) + proj["context_gb"]
    return out


def peak_gb(proj: dict, streamed: int) -> float:
    return max(phase_totals(proj, streamed).values(), default=0.0)

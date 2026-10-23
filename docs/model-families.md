# Model families

| Family | Files a profile needs | Notes |
| --- | --- | --- |
| Krea 2 | Krea 2 Raw or Turbo, Qwen3-VL-4B, Qwen-Image VAE | Train Turbo with Ostris's turbo training adapter; it's on only during training. Can be [finetuned](finetuning.md). |
| MOSS-Image | MOSS-Image Base or SFT, Qwen3.5-9B, FLUX.2 VAE | The ComfyUI single file. Samples like the reference pipeline: shift 5, Euler, CFG 3. Can be [finetuned](finetuning.md). |
| Qwen-Image 2.1 | the Qwen-Image 2.1 DiT (base or Turbo), Qwen3-VL-8B, Qwen-Image 2.1 VAE (RGBA) | Images with references in `_refs` train as edits (the caption is the instruction); images without train text-to-image. Transparent PNGs train as RGBA. Sample cards can add reference images to render an edit. Turbo needs its training adapter. |
| LTX 2.5 (video + audio) | the 22B DiT (dev or distilled), Gemma 4 12B with LTX's projection, video VAE (diffusion decoder or conv), audio VAE | Trains clips and images; clips with sound train audio too. Sample cards can set a frame count and a first frame. Samples are MP4 with 48 kHz audio. |
| MiniMax H3 (video + audio) | the fl2va or ref2va DiT, Qwen3-VL-32B (NVFP4), video VAE, audio VAE | Trains clips and images; clips with sound train audio too. ref2va trains with references from `_refs`. Sample cards can set a frame count and a first frame or references. Samples without CFG (it's guidance-distilled), as MP4 with AAC audio. Makes [RefMods](refmods.md). |
| YuE2 (music) | the YuE2-3B single file (int8_convrot) | Trains the acoustic model on songs; LoRAs default to strength 2. See [Music](music.md). |

To find and download these files, see
[Profiles and model files](profiles-and-model-files.md).

**LTX 2.5 dev or distilled:** a LoRA can be trained on either file. The
profile's variant only sets how samples are made: distilled uses its own 8-step
schedule without CFG, dev about 30 steps at CFG 4. Dev is the usual base for LTX
LoRAs.

## Speed and memory on an RTX 3090

Training at LoRA rank 32 with gradient checkpointing:

- **Krea 2**: about 8–9 s/step at 1024 px, peaking at about 18 GB.
- **MOSS-Image**: about 4 s/step at 1024 px, peaking at about 10 GB. A CFG
  sample takes about 110 s at the default 50 steps.
- **Qwen-Image 2.1** (int8_convrot): about 2 s/step on 512 px edit pairs,
  peaking at about 12 GB. A 1024 px sample takes about 55 s.
- **MiniMax H3**: about 5.4 s/step on 39-frame clips at 384 px with 40%
  offloaded, peaking at about 20 GB. Offloaded blocks take about 0.4 GB of
  system RAM each (about 8 GB at the default 40%), so two MiniMax jobs in
  parallel need twice that. The model files total about 40 GB.
- **LTX 2.5** (int8_convrot, 60% offloaded): about 4.5 s/step on 25-frame clips
  at 384 px (rank 16), with about 11 GB of blocks in system RAM. Caption
  encoding needs about 19 GB of VRAM unless Low VRAM is on.
- **YuE2**: about 0.21 s/step on 20 s clips, peaking at about 2.3 GB; a
  1500-step job took under 8 minutes.

See [Memory](memory.md) for offloading and Low VRAM.

## Weight formats

Each model file can use any of these, and a profile's files can mix them.

| Format | Stored on the GPU as |
| --- | --- |
| bf16 / fp16 / fp32 | bf16 |
| ComfyUI `fp8_scaled` (per-tensor or per-row scale; current `weight_scale` and older `scale_weight` names) | fp8 |
| ComfyUI `int8_tensorwise` (per-row scale) | int8 |
| ComfyUI `int8_tensorwise` with convrot (block regular-Hadamard rotation) | int8, input rotated at runtime |
| ComfyUI `nvfp4` (E2M1 codes, swizzled E4M3 block scales, optional AWQ `pre_quant_scale`) | nvfp4, expanded per matmul |

MXFP4, GGUF and block-wise scaled fp8 aren't supported yet. The profile form
flags them, and loading them fails with a clear error rather than producing
wrong weights.

## Licenses

No ComfyUI code is included; see `THIRD_PARTY_NOTICES.md`. Dainamo ships no
weights. Whether you may run a model is between you and its license.

- **LTX 2.5**: the LTX-2.x Community License. Entities with $10M or more in
  annual revenue need a paid license from Lightricks for anything beyond
  non-commercial testing and research, and a shared LoRA must carry the
  agreement and its use restrictions.
- **Qwen-Image 2.1**: the Qwen Research License. Non-commercial use only, and a
  distributed model trained with it must say "Built with Qwen".
- **YuE2**: CC BY-NC 4.0, plus a permission (2026-09-16) letting individuals,
  such as creators and musicians, use the weights free of charge and sell the
  songs they make. Companies need a commercial license from M-A-P. A shared LoRA
  likely counts as adapted weights, which CC BY-NC's attribution and
  non-commercial terms cover.
- **MiniMax H3**: the MiniMax H3 Community License (see `NOTICE`). Its grant
  excludes the EU, the UK, South Korea and the US, and it bans using MiniMax H3
  or its outputs to improve other models.

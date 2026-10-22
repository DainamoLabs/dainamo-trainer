# Third-Party Notices

Dainamo Trainer contains code adapted from the projects below. Each component
remains under its upstream license.

## ai-toolkit (Ostris, LLC) — MIT License

Upstream: https://github.com/ostris/ai-toolkit

- `dainamo/engine/families/krea2/model.py` — the Krea 2 single-stream MMDiT,
  ported from `extensions_built_in/diffusion_models/krea2/src/mmdit.py`
  (edit/reference-image and KV-cache paths removed).
- The Krea 2 prompt template, selected text-encoder layers and flow-matching
  schedule constants in `dainamo/engine/families/krea2/driver.py` follow
  ai-toolkit's `krea2` implementation.

```
MIT License

Copyright (c) 2024 Ostris, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MOSS-Image (OpenMOSS) — Apache License 2.0

Upstream: https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT (`pipeline_moss_image.py`),
https://github.com/OpenMOSS/MOSS-Image. Copyright 2026 OpenMOSS.

- `dainamo/engine/families/moss_image/model.py`: the MOSS-Image DiT, ported from
  `MOSSImageTransformer`. Changes: q/k/v fused into one `proj.qkv` Linear
  (ComfyUI's checkpoint layout), the compute dtype is a model attribute,
  per-block gradient checkpointing, and RoPE bands derived as a non-persistent
  fp32 buffer.
- `dainamo/engine/families/moss_image/driver.py`: the sampling schedule follows
  `MOSSImageScheduler`, and the latent packing and batch-norm normalisation follow
  `MOSSImagePipeline`. The training sigma distribution implements eq. (4) of the
  MOSS-Image Technical Report.

Licensed under the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0

## Qwen-Image 2.1 (Qwen-Image Team, Hugging Face) — Apache License 2.0; weights under the Qwen Research License

Upstream: https://huggingface.co/Qwen/Qwen-Image-2.1 and diffusers' `QwenImage21Transformer2DModel` /
`AutoencoderKLQwenImage21` (Copyright 2026 The Qwen Team / Qwen-Image Team and The HuggingFace Team,
Apache License 2.0), via ai-toolkit (Ostris, LLC — MIT), `extensions_built_in/diffusion_models/qwen_image_2/`.

- `dainamo/engine/families/qwen_image_2/model.py`: the DiT. Changes: a plain `nn.Module`; the fused
  ComfyUI `img_mlp.gate_up` (as in ai-toolkit); attention is the exact multi-pass SDPA form of the
  block-causal mask (flex-attention and KV-cache inference paths not ported); lazy RoPE tables.
- `dainamo/engine/families/qwen_image_2/vae.py`: the VAE, with ai-toolkit's loader mixin removed and
  its ComfyUI key conversion kept.
- `dainamo/engine/families/qwen_image_2/driver.py`: Dainamo's own. The prompt template, reference
  sizing and sampling schedule follow the release pipeline as vendored by ai-toolkit.

Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

The Qwen-Image 2.1 **model weights** are under the Qwen Research License Agreement (copy in
`dainamo/engine/families/qwen_image_2/QWEN_RESEARCH_LICENSE.txt`): **non-commercial use only**
(research or evaluation) without a separate commercial license from Alibaba's Tongyi Lab. A model
trained or fine-tuned with Qwen that you distribute must say "Built with Qwen" or "Improved using
Qwen" in its documentation. Dainamo ships no weights.

## MiniMax H3 (MiniMax) — MiniMax H3 Community License Agreement

Upstream: https://huggingface.co/MiniMaxAI/MiniMax-H3. Architecture code ported via ai-toolkit
(Ostris, LLC — MIT License, above), `extensions_built_in/diffusion_models/minimax_h3/src/`.

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax.
All Rights Reserved. The full agreement is in
`dainamo/engine/families/minimax_h3/MINIMAX_H3_LICENSE.txt`. **Its grant covers only the
"Applicable Territory": worldwide excluding the European Union, the United Kingdom, the Republic of
Korea and the United States of America.** It also carries an Acceptable Use Policy (Exhibit A), a
ban on using MiniMax H3 or its outputs to improve other AI models, and extra terms for commercial
services. Model weights are not included with Dainamo; you download them yourself and are
responsible for complying with the agreement.

- `packing.py`, `text_encoder.py`: verbatim from ai-toolkit.
- `transformer.py`: the FastH3 sparse-attention (VSA) path removed; block count read from the
  checkpoint; no loader mixin.
- `vae.py`, `audio_vae.py`: no ai-toolkit loader mixin.
- `driver.py`: Dainamo's own; the training recipe (packed sequences, sigma schedules, keyframe and
  reference conditioning, the released Euler sampler) follows ai-toolkit's MiniMax H3 trainer.

## LTX 2.5 (Lightricks) — LTX-2.x Community License Agreement

Upstream: https://huggingface.co/Lightricks/LTX-2.5, https://github.com/Lightricks/LTX-2.

The model code is diffusers' (Apache License 2.0, Copyright 2025-2026 The Lightricks team and The
HuggingFace Team), used as a library: `LTX2VideoTransformer3DModel`, `LTX2TextConnectors`,
`LTX2VideoEncoder3d`, `LTX2VideoDiffusionDecoderModel`, `AutoencoderKLLTX2Video`, `AutoencoderKLLTX2Audio`
and `LTX2VocoderWithBWE`, with Gemma 4 from Hugging Face `transformers` (Apache-2.0).

- `dainamo/engine/families/ltx2/model.py`: the original-to-diffusers key mapping and the model configs follow
  diffusers' `scripts/convert_ltx2_to_diffusers.py` (`--version 2.5`, Apache-2.0); the prompt encoding
  follows diffusers' `LTX2Pipeline`. Dainamo's own: the halo neighborhood attention that replaces NATTEN
  in the diffusion decoder, and the causal log-mel preprocessing (its filterbank reproduces the one stored
  in the vocoder).
- `dainamo/engine/families/ltx2/driver.py`: Dainamo's own. The training recipe (velocity target, first-frame
  conditioning with the conditioned tokens left out of the loss, joint audio) follows ai-toolkit's
  `extensions_built_in/diffusion_models/ltx2/ltx2.py` (Ostris, LLC — MIT); the distilled sigmas and the
  image-conditioning CRF are diffusers' `DISTILLED_SIGMA_VALUES` and `LTX2_5_IMAGE_CRF`.

LTX-2.x is licensed under the LTX-2.x Community License Agreement (copy in
`dainamo/engine/families/ltx2/LTX_2_COMMUNITY_LICENSE.txt`); the Licensor is Lightricks Ltd. **Entities with
annual revenues of at least $10,000,000 need a paid commercial license** for any use other than
non-commercial testing and research, including training a model for commercial use. A LoRA trained on LTX-2.x
is a Derivative under the agreement: distributing it requires passing on the agreement and its use
restrictions (Attachment A). Model weights are not included with Dainamo.

## YuE2 (M-A-P) — Apache License 2.0; weights under CC BY-NC 4.0

Upstream: https://github.com/multimodal-art-projection/YuE, https://huggingface.co/m-a-p/YuE2-3B.

- `dainamo/engine/families/yue2/model.py` is ported from M-A-P's `yue2` package (`yue2-infer`, Apache
  License 2.0): `modeling_yue2.py` (the AR and NAR experts and their embeddings), `nar.py` (the prefill and
  the midpoint ODE), `protocol.py` (the prompt format and token ids), `sampling.py` (the token sampler) and
  `modeling_vae.py` (the Oobleck VAE). It is modified to read ComfyUI's single-file layout, with fused
  `qkv_proj` and `gate_up_proj`, and to cache the composer's keys and values as conditioning. The VAE's
  Oobleck blocks and SnakeBeta activation come, through `modeling_vae.py`, from stable-audio-tools and
  BigVGAN (MIT; notices below).
- `dainamo/engine/families/yue2/driver.py`: Dainamo's own. Training a LoRA on the NAR alone, with the
  release's flow matching and text-only (planning off) conditioning, follows Starnodes'
  ComfyUI-YuE2-Trainer (https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer, MIT; notice below), as do
  the rank 32 / alpha 32 / 1e-4 defaults and the recommended LoRA strength of 2, which Dainamo's own test
  confirmed (README, Music). No code is taken from it.

The YuE2-3B and YuE2-Vae weights are licensed under Creative Commons Attribution-NonCommercial 4.0
International (https://creativecommons.org/licenses/by-nc/4.0/), with an additional permission (2026-09-16)
for individuals acting in a personal capacity to use the weights free of charge and monetize their outputs.
That permission doesn't cover commercial use of the weights by companies, or redistributing or selling the
weights. Model weights are not included with Dainamo.

stable-audio-tools (Oobleck):

```
MIT License

Copyright (c) 2023 Stability AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

BigVGAN (SnakeBeta):

```
MIT License

Copyright (c) 2022 NVIDIA CORPORATION.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

ComfyUI-YuE2-Trainer:

```
MIT License

Copyright (c) 2026 Starnodes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Sampling algorithms

`dainamo/engine/sampling.py` implements Heun's method and DPM-Solver++(2M) (Lu et al., "DPM-Solver++:
Fast Solver for Guided Sampling of Diffusion Probabilistic Models", 2022) from their published
formulas, in flow-matching form. `tests/test_sampling.py` compares the latter with a transcription of
k-diffusion's `sample_dpmpp_2m` (Katherine Crowson, MIT License, https://github.com/crowsonkb/k-diffusion).
The beta schedule follows "Beta Sampling is All You Need" (Lee et al., 2024), with a = b = 0.6.

## LoKr

`LoKrLinear` and `kron_delta` in `dainamo/engine/lora.py` implement LoKr, the Kronecker-product adapter of
LyCORIS (Yeh et al., "Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model
Evaluation", ICLR 2024; https://github.com/KohakuBlueleaf/LyCORIS, Apache License 2.0). How a layer's
sizes are split by the factor, the initialization (w1 random, w2 zero), the alpha rule for low-rank
factors and the `lokr_w1` / `lokr_w2` key layout follow LyCORIS. Training both factors as full matrices
follows ai-toolkit's default for LoKr (MIT License, above). The code is written from these descriptions;
none of it is copied.

## ComfyUI-MiniMaxH3Mod (Luisa) — MIT License

Upstream: https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod

`dainamo/engine/refmod.py` reads and writes RefMods in this pack's file format (the `refmod_meta`
header, format 4 standalone files and format 5 bundles, audio ones included), and makes them by its
methods: references encoded to a short edge in multiples of 32, compressed grids pooled to the first
reference's aspect and refined with Adam against the full encode, merging, the token cap's
near-duplicate pass and even spread, audio as the normalized (1, 32, 2, T) posterior mean, and
strength as a blend toward a blurred copy (an eighth of the size, or of the length for audio). The code
is Dainamo's own, written to match them.

```
MIT License

Copyright (c) 2026 Luisa (luisacaotica)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Libraries used (not vendored)

- Hugging Face `diffusers` (Apache-2.0) — `AutoencoderKLQwenImage` and
  `AutoencoderKLFlux2`, plus its Wan and LDM VAE key converters.
- Hugging Face `transformers` (Apache-2.0) — Qwen3-VL and Qwen3.5 text models, DINOv2.
- DINOv2 weights (`facebook/dinov2-small`, Apache-2.0), downloaded at first use by the concept bleed detector.
- `bitsandbytes` (MIT), `prodigyopt` (MIT), FastAPI (MIT), React (MIT), Vite (MIT).

No ComfyUI code is included. ComfyUI-format model files are read using the
plain safetensors format and its documented fp8 scaling convention.

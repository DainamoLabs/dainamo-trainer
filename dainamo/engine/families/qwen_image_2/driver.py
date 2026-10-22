"""Qwen-Image 2.1 driver: text-to-image and image-editing LoRAs from one checkpoint.

Loads ComfyUI-format files directly:
  * diffusion model: the 32-block single-stream DiT (bf16 or ComfyUI int8_convrot)
  * text encoder: Qwen3-VL-8B (``model.layers.*`` + ``model.visual.*``), read at the last decoder layer
    before its final RMSNorm. The vision tower stays: edit prompts carry their reference images.
  * VAE: RGBA, 16x spatial, 64 latent channels, in ComfyUI's Wan-native layout

Editing is not a separate mode. An item with reference images (``<dataset>/_refs/<stem>.png``,
``<stem>_1.png``, ...) trains as an edit: the caption is the instruction, the references enter the
Qwen3-VL prompt as ``<imageN>`` vision blocks, and their latents are dropped into the slots those
blocks reserve. Items without references train text-to-image. Images with an alpha channel train as
RGBA unless ``keep_alpha`` is off.

The prompt template, reference sizing, timestep convention (t = sigma, 1 = noise, target noise - clean)
and the sampler (dynamic exponential shift, terminal sigma 0.02) follow the release's diffusers
pipeline as vendored by ai-toolkit.
"""

from __future__ import annotations

import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from ... import comfy_io
from ... import dataset as ds
from ...fp8_linear import assert_materialized, materialize
from ...hf_assets import qwen_image_2_assets
from ..base import FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, Workflow
from .model import QwenImage21Config, QwenImage21DiT
from .vae import AutoencoderKLQwenImage21

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
SYSTEM_TURN = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
PROMPT_TEMPLATE = SYSTEM_TURN + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

VAE_FACTOR = 16
VISION_PX = 32  # one Qwen3-VL vision token (patch 16, merge 2) = 2x2 latent tokens
MIN_REF_PIXELS = 65536  # the processor upsamples anything smaller, which would break the slot count
TILE_DECODE_ABOVE = 1024 * 1024

# FlowMatchEulerDiscreteScheduler config shipped with the release
BASE_SEQ, MAX_SEQ, BASE_SHIFT, MAX_SHIFT = 256, 8192, 0.5, 0.9
SHIFT_TERMINAL = 0.02
# Qwen-Image-2.1-Turbo's own 8-step schedule (model_index.json of Qwen/Qwen-Image-2.1-Turbo)
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def mu_for(seq_len: int) -> float:
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ)
    return seq_len * slope + BASE_SHIFT - slope * BASE_SEQ


def time_shift(mu: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


def sampling_sigmas(steps: int, seq_len: int, turbo: bool = False) -> list[float]:
    """The release's schedule: linspace(1, 1/n, n) through the exponential shift, stretched so the last
    sigma lands on 0.02, then 0. Turbo uses its fixed schedule."""
    if turbo:
        return TURBO_SIGMAS + [0.0]
    s = time_shift(mu_for(seq_len), torch.from_numpy(np.linspace(1.0, 1.0 / steps, steps)).float())
    one_minus = 1.0 - s
    s = 1.0 - one_minus / (one_minus[-1] / (1.0 - SHIFT_TERMINAL))
    return s.tolist() + [0.0]


def _snap(size: float) -> int:
    return max(VISION_PX, round(size / VISION_PX) * VISION_PX)


def reference_size(width: int, height: int, target_pixels: int) -> tuple[int, int]:
    """A reference scaled to the target's area, aspect kept, on the 32 px grid."""
    budget = max(target_pixels, MIN_REF_PIXELS)
    w = math.sqrt(budget * width / height)
    return _snap(w), _snap(w * height / width)


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)


def open_image(path: str, keep_alpha: bool) -> Image.Image:
    """RGBA when the file has transparency and alpha is kept, else RGB (transparency over white)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGBA") if keep_alpha and has_alpha(im) else ds.to_rgb(im)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> (1, C, H, W) in [0, 1], C = 3 or 4."""
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]


def prepare_reference(img: Image.Image, target_pixels: int) -> torch.Tensor:
    """(1, C, H, W) in [0, 1] at ``reference_size`` (bicubic + antialias, as the release). The same
    tensor feeds the vision tower and the VAE, so their slot and latent counts agree."""
    x = to_tensor(img)
    w, h = reference_size(img.width, img.height, target_pixels)
    if (h, w) == tuple(x.shape[2:]):
        return x
    return F.interpolate(x, size=(h, w), mode="bicubic", antialias=True).clamp(0, 1)


def vision_image(x: torch.Tensor) -> Image.Image:
    """A prepared reference as the vision tower sees it: RGB, any alpha composited over white."""
    x = x[0].float().clamp(0, 1)
    if x.shape[0] == 4:
        x = x[:3] * x[3:4] + (1.0 - x[3:4])
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), mode="RGB")


def pack(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B, h*w, C); 2.1 consumes latents unpatched."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(1, 2)


class QwenImage2Driver(FamilyDriver):
    info = FamilyInfo(
        id="qwen_image_2",
        label="Qwen-Image 2.1 (text-to-image + edit)",
        media=["image"],
        profile_fields=[
            ProfileField("diffusion_model", "Diffusion model", hint="Qwen-Image 2.1 (base or Turbo)",
                         help="bf16 or ComfyUI int8_convrot. One checkpoint trains both text-to-image and edit LoRAs."),
            ProfileField("text_encoder", "Text encoder (CLIP)", hint="Qwen3-VL-8B",
                         help="Qwen3-VL-8B as one .safetensors file, any precision (e.g. ComfyUI's qwen3vl_8b "
                              "files). Not Krea 2's 4B."),
            ProfileField("vae", "VAE", hint="Qwen-Image 2.1 VAE",
                         help="The 2.1 VAE (RGBA, 64 latent channels). The original Qwen-Image VAE won't work."),
            ProfileField("training_adapter", "Training adapter", required=False, default="",
                         hint="Qwen-Image 2.1 Turbo training adapter",
                         help="For Turbo: a de-distill LoRA applied only while training, e.g. "
                              "ostris/qwen_image_2_turbo_training_adapter/qwen_image_2_turbo_training_adapter_v1.safetensors. "
                              "Leave empty for the base model."),
            ProfileField("sample_schedule", "Sampling schedule", kind="select", options=["standard", "turbo"],
                         default="standard", help="turbo: the Turbo checkpoint's fixed 8-step schedule"),
            ProfileField("sample_steps", "Sample steps", kind="number", default=40),
            ProfileField("sample_cfg", "Sampling CFG (1 = off)", kind="number", default=1,
                         help="2.1 is meant to be sampled without guidance. Edit samples never use CFG."),
            ProfileField("sample_negative", "Sampling negative prompt", kind="text", required=False, default=""),
        ],
        training_defaults={
            "resolution": 1024, "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 2000,
            "batch_size": 1, "optimizer": "adamw8bit", "timestep_sampling": "shift", "keep_alpha": True,
            "sample_width": 1024, "sample_height": 1024,
        },
        workflows=[
            Workflow("t2i"),
            Workflow("edit", images={
                "label": "Reference images (edit)", "min": 1, "max": 6,
                "help": "For edits, put each image's source image(s) in a _refs folder inside the dataset, named "
                        "after it: _refs/photo.png, _refs/photo_1.png, … The caption is the edit instruction. "
                        "Images without references train text-to-image."}),
        ],
        sliders=True,
    )
    bucket_divisibility = VISION_PX
    block_list = "transformer_blocks"
    lora_targets = ["transformer_blocks.*.attn.to_q", "transformer_blocks.*.attn.to_k",
                    "transformer_blocks.*.attn.to_v", "transformer_blocks.*.attn.to_out.0",
                    "transformer_blocks.*.img_mlp.gate_up", "transformer_blocks.*.img_mlp.out"]
    memory = MemoryProfile(saved=30_400, full=1_066_000, work=1_749_000, infer=1_084_000,
                           vae_encode=(0.1e9, 2_330), vae_decode=(0.2e9, 6_730), decode_tile_pixels=TILE_DECODE_ABOVE)
    text_encoder_fp32 = ("visual.",)  # the vision tower runs in fp32
    lora_metadata = {"modelspec.architecture": "qwen-image-2.1/lora", "dainamo.family": "qwen_image_2"}
    signature_keys = {
        "diffusion_model": ["transformer_blocks.0.img_mlp.gate_up.weight", "txt_in.in_layer.weight"],
        "text_encoder": ["layers.35.mlp.down_proj.weight", "visual.deepstack_merger_list.0.linear_fc1.weight"],
        "vae": ["decoder.head.2.weight", "encoder.conv1.weight"],
    }
    signature_shapes = {
        "text_encoder": {"layers.35.mlp.down_proj.weight": (4096, 12288)},  # 8B, not Krea's 4B
        "vae": {"encoder.conv1.weight": (96, 4)},  # RGBA input, not the Qwen-Image 1 VAE
    }

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.processor = self.text_encoder = self.vae = None
        self.keep_alpha = True

    # --- text -------------------------------------------------------------
    def load_text_encoder(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel, Qwen3VLTextRotaryEmbedding, Qwen3VLVisionRotaryEmbedding,
        )

        processor, config, _ = qwen_image_2_assets()
        config.tie_word_embeddings = False
        for c in (config, config.text_config, config.vision_config):
            c._attn_implementation = "sdpa"
        path = self.profile["text_encoder"]
        self.log(f"Loading text encoder {path}")
        # lm_head is unused, and the final norm is replaced by Identity below
        loaded = comfy_io.load_state_dict(path, skip_prefixes=("lm_head.", "model.norm.", "model.language_model.norm."),
                                          device=self.text_encoder_device)

        def rename(k: str) -> str:  # ComfyUI: model.layers.* / model.visual.*; transformers: language_model.* / visual.*
            if k.startswith("model.visual."):
                return k[len("model."):]
            if k.startswith("model.language_model."):
                return k[len("model."):]
            return "language_model." + k[len("model."):] if k.startswith("model.") else k

        loaded.tensors = {rename(k): v for k, v in loaded.tensors.items()}
        loaded.quant = {rename(k): v for k, v in loaded.quant.items()}
        with torch.device("meta"):
            model = Qwen3VLModel(config)
        model.language_model.norm = torch.nn.Identity()  # the DiT reads the last layer before the final norm
        # the vision tower runs in fp32 (bf16 drifts by several percent; it's small and only sees references)
        materialize(model, loaded, self.dtype, keep_fp8=True, device=None,
                    dtype_for=lambda k: torch.float32 if k.startswith("visual.") else None)
        del loaded  # else its tensors stay in RAM beside the streamed copies
        model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config.text_config, device=self.device)
        model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.vision_config).to(self.device)
        assert_materialized(model)
        self.processor, self.text_encoder = processor, self.stream_text_encoder(model).eval()
        tok = processor.tokenizer
        self.drop_tokens = len(tok(SYSTEM_TURN, add_special_tokens=False)["input_ids"])
        self.image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

    @torch.no_grad()
    def encode_condition(self, prompt: str, images: list | None):
        """-> {"embeds": (L, 4096), "slots": (L,) bool marking image slots}. ``images`` are prepared
        references (PIL, already on the 32 px grid)."""
        images = list(images or [])
        user = " ".join(f"<image{i + 1}>{VISION_BLOCK}" for i in range(len(images))) + (prompt.strip() or " ")
        inputs = self.processor(text=[PROMPT_TEMPLATE.format(user)], images=images or None, return_tensors="pt")
        if images:
            grid = inputs["image_grid_thw"]  # (n, 3) in 16 px patches
            got = [(int(g[1]) * 16, int(g[2]) * 16) for g in grid]
            want = [(im.height, im.width) for im in images]
            if got != want:
                raise RuntimeError(f"Qwen3-VL resized reference images {want} to {got}; their slots would not "
                                   "line up with the VAE latents")
        kw = {k: inputs[k].to(self.device) for k in ("input_ids", "attention_mask", "image_grid_thw",
                                                     "mm_token_type_ids") if k in inputs}
        if images:
            kw["pixel_values"] = inputs["pixel_values"].to(self.device)
        hidden = self.text_encoder(**kw, use_cache=False).last_hidden_state[0]
        ids = inputs["input_ids"][0]
        return {"embeds": hidden[self.drop_tokens:].to("cpu", torch.bfloat16),
                "slots": (ids == self.image_token_id)[self.drop_tokens:].cpu()}

    def encode_prompt(self, prompt: str):
        return self.encode_condition(prompt, None)

    # --- references ----------------------------------------------------------
    def _references(self, paths: list[str], target_pixels: int, flip: bool = False) -> list[torch.Tensor]:
        out = []
        for p in paths:
            img = open_image(p, self.keep_alpha)
            if flip:
                img = ImageOps.mirror(img)
            out.append(prepare_reference(img, target_pixels))
        return out

    def item_conditions(self, item, settings: dict):
        self.keep_alpha = bool(settings.get("keep_alpha", True))
        refs = ds.reference_images(item.path)
        if not refs:
            return {"text": (item.caption, None)}
        w, h = item.size
        return {"ref": (item.caption, [vision_image(r) for r in self._references(refs, w * h, item.flip)])}

    def _sample_refs(self, spec: dict) -> list[torch.Tensor]:
        return self._references(spec.get("images") or [], spec["width"] * spec["height"])

    def sample_condition(self, spec: dict):
        refs = self._sample_refs(spec)
        return spec["prompt"], [vision_image(r) for r in refs] or None

    # --- vae --------------------------------------------------------------
    def cache_inputs(self, kind: str, settings: dict) -> dict:
        inputs = super().cache_inputs(kind, settings)
        return inputs if kind == "text" else {**inputs, "keep_alpha": bool(settings.get("keep_alpha", True))}

    def load_vae(self):
        _, _, vae_config = qwen_image_2_assets()
        self.log(f"Loading VAE {self.profile['vae']}")
        sd = comfy_io.dequantize_all(comfy_io.load_state_dict(self.profile["vae"]), self.dtype)
        sd = AutoencoderKLQwenImage21.convert_state_dict_on_load(sd)
        vae = AutoencoderKLQwenImage21(**vae_config)
        vae.load_state_dict(sd, strict=True)
        self.vae = vae.to(self.device, self.dtype).eval().requires_grad_(False)
        z = vae_config["z_dim"]
        self._lat_mean = torch.tensor(vae_config["latents_mean"]).view(1, z, 1, 1, 1)
        self._lat_std = torch.tensor(vae_config["latents_std"]).view(1, z, 1, 1, 1)

    @torch.no_grad()
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """(1, 3 or 4, H, W) in [-1, 1] -> (64, h, w). RGB gets an opaque alpha channel."""
        x = pixels.to(self.device, self.dtype)
        if x.shape[1] == 3:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        lat = self.vae.encode(x.unsqueeze(2)).latent_dist.mode()
        lat = (lat - self._lat_mean.to(lat)) / self._lat_std.to(lat)
        return lat[0, :, 0].to("cpu", torch.bfloat16)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """-> (B, 4, H, W) RGBA in [-1, 1]."""
        lat = latents.to(self.device, self.dtype).unsqueeze(2)
        lat = lat * self._lat_std.to(lat) + self._lat_mean.to(lat)
        tiled = lat.shape[-2] * lat.shape[-1] * VAE_FACTOR ** 2 > TILE_DECODE_ABOVE
        if tiled:  # a one-shot decode of this VAE's 16x stack is heavy above ~1 MP
            self.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                                   tile_sample_stride_height=768, tile_sample_stride_width=768)
        try:
            img = self.vae.decode(lat).sample
        finally:
            if tiled:
                self.vae.disable_tiling()
        return img[:, :, 0].float().clamp(-1, 1)

    def to_media(self, latents) -> list:
        """PNG-ready images: RGBA when the result is visibly transparent, else RGB."""
        out = []
        for img in self.decode_latents(latents):
            arr = ((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
            transparent = (arr[..., 3] < 128).mean() > 0.01
            out.append(Image.fromarray(arr if transparent else arr[..., :3]))
        return out

    def unload_vae(self):
        self.vae = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_item(self, item, settings: dict) -> dict:
        self.keep_alpha = bool(settings.get("keep_alpha", True))
        img = ImageOps.fit(open_image(item.path, self.keep_alpha), item.size, Image.Resampling.LANCZOS)
        if item.flip:
            img = ImageOps.mirror(img)
        payload = {"latent": self.encode_image(to_tensor(img) * 2 - 1)}
        refs = ds.reference_images(item.path)
        if refs:
            w, h = item.size
            payload["refs"] = [self.encode_image(r * 2 - 1) for r in self._references(refs, w * h, item.flip)]
        return payload

    # --- transformer --------------------------------------------------------
    def build_transformer(self, shapes):
        with torch.device("meta"):
            return QwenImage21DiT(QwenImage21Config.from_shapes(shapes))

    def sample_timesteps(self, batch: int, latent_hw, settings: dict) -> torch.Tensor:
        mode = settings.get("timestep_sampling", "shift")
        if mode == "uniform":
            return torch.rand(batch).clamp(1e-3, 1.0)
        u = torch.sigmoid(torch.randn(batch)) if mode == "logit_normal" else torch.rand(batch)
        return time_shift(mu_for(latent_hw[0] * latent_hw[1]), u.clamp(1e-4, 1 - 1e-4))

    def _forward(self, model, noisy: torch.Tensor, t: torch.Tensor, conds: list[dict], refs: list[list]):
        """Velocity for (B, 64, h, w) noisy target latents. All samples must share the reference layout."""
        b, _, h, w = noisy.shape
        d = self.device
        shapes = [tuple(r.shape[-2:]) for r in refs[0]]
        if any([tuple(r.shape[-2:]) for r in rs] != shapes for rs in refs):
            raise ValueError("samples in one forward must share their reference image sizes")
        want = sum(rh * rw for rh, rw in shapes) // 4
        for c in conds:
            if int(c["slots"].sum()) != want:
                raise ValueError(f"prompt holds {int(c['slots'].sum())} image slots but the references need {want}")
        length = max(c["embeds"].shape[0] for c in conds)
        embeds = torch.zeros(b, length, conds[0]["embeds"].shape[-1], device=d, dtype=self.dtype)
        valid = torch.zeros(b, length, dtype=torch.bool, device=d)
        slots = torch.zeros(b, length, dtype=torch.bool, device=d)
        for i, c in enumerate(conds):
            n = c["embeds"].shape[0]
            embeds[i, :n] = c["embeds"].to(d, self.dtype)
            valid[i, :n] = True
            slots[i, :n] = c["slots"].to(d)
        hidden = pack(noisy.to(self.dtype))
        if shapes:
            ref_rows = torch.stack([torch.cat([pack(r[None].to(d, self.dtype))[0] for r in rs]) for rs in refs])
            hidden = torch.cat([ref_rows, hidden], dim=1)
        img_mask = torch.cat([slots, torch.ones(b, h * w // 4, dtype=torch.bool, device=d)], dim=1)
        img_shapes = [*[(1, rh, rw) for rh, rw in shapes], (1, h, w)]
        out = model(hidden, embeds, t.to(d), img_shapes, img_mask, None if bool(valid.all()) else valid)
        return out[:, -h * w:].transpose(1, 2).reshape(b, -1, h, w)

    def predict(self, model, noisy, t, text):
        """Text-to-image velocity (no references), as the shared slider step uses it."""
        return self._forward(model, noisy, t, text, [[] for _ in text])

    def training_loss(self, model, payloads, conds, settings, rng=None, sigma=None, generator=None,
                      deterministic=False):
        d = self.device
        lat = torch.stack([p["latent"] for p in payloads]).to(d, torch.float32)
        noise = (torch.randn(lat.shape, generator=generator).to(d) if generator is not None
                 else torch.randn_like(lat))
        if sigma is not None:
            t = torch.full((lat.shape[0],), float(sigma), device=d)
        else:
            t = self.sample_timesteps(lat.shape[0], lat.shape[-2:], settings).to(d)
        tb = t.view(-1, 1, 1, 1)
        noisy = (1 - tb) * lat + tb * noise
        # an edit item trains with its references; a dropped caption trains plain text-to-image
        items = []
        for p, c in zip(payloads, conds):
            if p.get("refs") and "ref" in c:
                items.append((c["ref"], p["refs"]))
            else:
                items.append((c["text"] if "text" in c else c["ref"], []))
        # one forward per reference layout (samples in a forward must share it)
        groups: dict[tuple, list[int]] = {}
        for i, (_, refs) in enumerate(items):
            groups.setdefault(tuple(tuple(r.shape) for r in refs), []).append(i)
        pred = torch.empty_like(lat)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for idx in groups.values():
                pred[idx] = self._forward(model, noisy[idx], t[idx], [items[i][0] for i in idx],
                                          [items[i][1] for i in idx]).float()
        return F.mse_loss(pred, noise - lat)

    # --- sampling -----------------------------------------------------------
    def default_shift(self, width, height):
        return math.exp(mu_for((height // VAE_FACTOR) * (width // VAE_FACTOR)))

    def default_sigmas(self, steps, width, height):
        turbo = str(self.profile.get("sample_schedule") or "standard") == "turbo"
        return sampling_sigmas(int(steps), (height // VAE_FACTOR) * (width // VAE_FACTOR), turbo)

    def prepare_sampling(self, model, cond, uncond, width, height, cfg, seed, images=None):
        h, w = height // VAE_FACTOR, width // VAE_FACTOR
        refs = []
        if images:
            if self.vae is None:
                raise RuntimeError("the VAE must be loaded to encode reference images")
            refs = [self.encode_image(r * 2 - 1) for r in
                    self._sample_refs({"images": images, "width": width, "height": height})]
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn(1, 64, h, w, generator=g).to(self.device, torch.float32)
        use_cfg = cfg and float(cfg) > 1 and uncond is not None and not refs  # edits sample without CFG

        def velocity(x, s):
            t = torch.full((1,), s, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self._forward(model, x, t, [cond], [refs]).float()
                if use_cfg:
                    vu = self._forward(model, x, t, [uncond], [[]]).float()
                    v = vu + float(cfg) * (v - vu)
            return v
        return x, velocity

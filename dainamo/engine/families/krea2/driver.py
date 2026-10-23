"""Krea 2 driver.

Loads ComfyUI-format files directly:
  * diffusion model: native Krea 2 key layout, bf16 / fp8_scaled / int8 (incl. convrot)
  * text encoder: Qwen3-VL-4B (``model.layers.*``), bf16 / fp8_scaled / int8
  * VAE: Qwen-Image / Wan 2.1 layout, converted to diffusers' AutoencoderKLQwenImage

The prompt template, selected layers and schedule constants follow the Krea 2
reference as implemented in ai-toolkit (MIT).
"""

from __future__ import annotations

import gc
import math

import torch
from einops import rearrange, repeat

from ... import comfy_io
from ...fp8_linear import assert_materialized, materialize
from ...hf_assets import qwen3vl_tokenizer_and_config
from ..base import Download, FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, count_blocks
from .model import Krea2Config, Krea2DiT

# where the model manager downloads the files from, at the commit they were checked against
HF_REPO, HF_REV = "Comfy-Org/Krea-2", "1171f4dc7e325bf3734586942e737872506c73e1"
ADAPTER_REPO, ADAPTER_REV = "ostris/krea2_turbo_training_adapter", "64ba06d2354120be472d26380247c971b2fb6af3"

SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34
MAX_TEXT_TOKENS = 512

VAE_FACTOR = 8
PATCH = 2
# resolution-dependent shift: mu interpolated in image-token count
SCHED_X1 = (256 // (VAE_FACTOR * PATCH)) ** 2
SCHED_X2 = (1280 // (VAE_FACTOR * PATCH)) ** 2
SCHED_Y1, SCHED_Y2 = 0.5, 1.15


def mu_for(seq_len: int) -> float:
    slope = (SCHED_Y2 - SCHED_Y1) / (SCHED_X2 - SCHED_X1)
    return slope * seq_len + (SCHED_Y1 - slope * SCHED_X1)


def time_shift(mu: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


class Krea2Driver(FamilyDriver):
    info = FamilyInfo(
        id="krea2",
        label="Krea 2",
        media=["image"],
        profile_fields=[
            ProfileField("diffusion_model", "Diffusion model", hint="Krea 2 (Raw or Turbo)",
                         help="bf16, or ComfyUI fp8_scaled / int8_convrot. Turbo is distilled: train it with the "
                              "training adapter below."),
            ProfileField("text_encoder", "Text encoder (CLIP)", hint="Qwen3-VL-4B",
                         help="Qwen3-VL-4B-Instruct as one .safetensors file, any precision (e.g. ComfyUI's "
                              "qwen3vl_4b files). Not the 8B."),
            ProfileField("vae", "VAE", hint="Qwen-Image VAE",
                         help="The Qwen-Image VAE. The Wan 2.1 VAE has the same layout and passes the check, but "
                              "its decoder differs, so samples would come out wrong."),
            ProfileField("training_adapter", "Training adapter", required=False, hint="Krea 2 Turbo training adapter",
                         help="For Turbo models: a de-distill LoRA applied only while training (Download puts it in "
                              "your loras folder). Leave empty for Raw.",
                         default=""),
            ProfileField("sample_steps", "Sample steps", kind="number", default=8),
            ProfileField("sample_cfg", "Sampling CFG (0 = off)", kind="number", default=0),
            ProfileField("sample_mu", "Sampling fixed shift (mu, blank = auto)", kind="text", required=False, default="1.15",
                         help="Turbo was distilled at mu=1.15. Leave blank for Raw."),
        ],
        training_defaults={
            "resolution": 1024, "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 2000,
            "batch_size": 1, "optimizer": "adamw8bit", "timestep_sampling": "shift",
        },
        sliders=True,
        # 26 GB in bf16, so a finetune streams some blocks from system RAM on a 24 GB card, and each save is as big
        finetune={"learning_rate": 1e-5, "optimizer": "adafactor", "keep_last": 1},
    )
    bucket_divisibility = VAE_FACTOR * PATCH
    caption_tokens = MAX_TEXT_TOKENS
    lora_targets = ["blocks.*.attn.w?", "blocks.*.attn.gate", "blocks.*.mlp.*"]
    memory = MemoryProfile(saved=12_800, full=351_500, work=464_000, infer=198_000,
                           vae_encode=(0.75e9, 30), vae_decode=(0.3e9, 10))
    text_encoder_skip = ("lm_head.", "visual.")
    lora_metadata = {"modelspec.architecture": "krea2/lora", "dainamo.family": "krea2"}
    signature_keys = {
        "diffusion_model": ["blocks.27.attn.wq.weight", "txtfusion.projector.weight"],
        "text_encoder": ["layers.35.mlp.down_proj.weight"],
        "vae": ["decoder.head.2.weight"],
    }
    signature_shapes = {
        "text_encoder": {"layers.35.mlp.down_proj.weight": (2560, 9728)},  # 4B, not Qwen-Image 2.1's 8B
        "vae": {"decoder.head.2.weight": (3, 96)},  # RGB, not the Qwen-Image 2.1 RGBA VAE
    }
    downloads = [
        Download("diffusion_model", HF_REPO, "diffusion_models/krea2_turbo_int8_convrot.safetensors", 13_492_686_496,
                 "Turbo · int8 convrot", HF_REV, recommended=True,
                 sha256="8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1"),
        Download("diffusion_model", HF_REPO, "diffusion_models/krea2_turbo_fp8_scaled.safetensors", 13_141_730_784,
                 "Turbo · fp8 scaled", HF_REV,
                 sha256="eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1"),
        Download("diffusion_model", HF_REPO, "diffusion_models/krea2_turbo_nvfp4.safetensors", 7_673_668_448,
                 "Turbo · nvfp4", HF_REV, sha256="61527003b2d537055494d01bc8efe51d6e86e64192ba23e3721a5647231fe394"),
        Download("diffusion_model", HF_REPO, "diffusion_models/krea2_turbo_bf16.safetensors", 26_283_332_608,
                 "Turbo · bf16", HF_REV, sha256="78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3"),
        Download("diffusion_model", HF_REPO, "diffusion_models/krea2_raw_int8_convrot.safetensors", 13_492_686_496,
                 "Raw · int8 convrot", HF_REV,
                 sha256="5585a4a38c4bcfb6fde2d480a4aa6edf7f665721ebde56d30662c35a45f5fa5c"),
        Download("diffusion_model", HF_REPO, "diffusion_models/krea2_raw_fp8_scaled.safetensors", 13_141_730_784,
                 "Raw · fp8 scaled", HF_REV,
                 sha256="48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61"),
        Download("diffusion_model", HF_REPO, "diffusion_models/krea2_raw_bf16.safetensors", 26_283_332_608,
                 "Raw · bf16", HF_REV, sha256="f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7"),
        Download("text_encoder", HF_REPO, "text_encoders/qwen3vl_4b_int8_convrot.safetensors", 4_858_559_240,
                 "int8 convrot", HF_REV, recommended=True,
                 sha256="32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c"),
        Download("text_encoder", HF_REPO, "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5_242_467_968,
                 "fp8 scaled", HF_REV, sha256="54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094"),
        Download("text_encoder", HF_REPO, "text_encoders/qwen3vl_4b_bf16.safetensors", 8_875_719_384, "bf16", HF_REV,
                 sha256="36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34"),
        Download("vae", HF_REPO, "vae/qwen_image_vae.safetensors", 253_806_246, "bf16", HF_REV, recommended=True,
                 sha256="a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
        Download("training_adapter", ADAPTER_REPO, "krea2_turbo_training_adapter_v1.safetensors", 228_587_504,
                 "v1, for Turbo", ADAPTER_REV, recommended=True, folder="loras",
                 sha256="d6898b9a4c741c0cc513a7ec89f03e7cc81f10ad0376bda8ef417819723af6e9"),
    ]

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None

    # --- text -------------------------------------------------------------
    def load_text_encoder(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

        tokenizer, config = qwen3vl_tokenizer_and_config()
        text_cfg = config.text_config
        text_cfg._attn_implementation = "sdpa"
        self.log(f"Loading text encoder {self.profile['text_encoder']}")
        loaded = comfy_io.load_state_dict(self.profile["text_encoder"], strip_prefix="model.",
                                          skip_prefixes=("visual.", "lm_head.", "language_model.visual."),
                                          device=self.text_encoder_device)
        if "language_model.embed_tokens.weight" in loaded.tensors:  # HF-layout export
            loaded.tensors = {k.removeprefix("language_model."): v for k, v in loaded.tensors.items()}
            loaded.quant = {k.removeprefix("language_model."): v for k, v in loaded.quant.items()}
        with torch.device("meta"):
            model = Qwen3VLTextModel(text_cfg)
        materialize(model, loaded, self.dtype, keep_fp8=True, device=None)
        del loaded  # else its tensors stay in RAM beside the streamed copies
        model.rotary_emb = Qwen3VLTextRotaryEmbedding(text_cfg, device=self.device)
        assert_materialized(model)
        self.stream_text_encoder(model).eval()
        self.tokenizer, self.text_encoder = tokenizer, model

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        # leading whitespace would merge with the prefix's last token and shift PREFIX_TOKENS
        prompt = prompt.strip()
        tok = self.tokenizer
        ids = tok(PROMPT_PREFIX + prompt, truncation=True, max_length=MAX_TEXT_TOKENS + PREFIX_TOKENS,
                  return_tensors="pt")["input_ids"]
        suffix = tok(PROMPT_SUFFIX, return_tensors="pt")["input_ids"]
        ids = torch.cat([ids, suffix], dim=1).to(self.device)
        out = self.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        hidden = torch.stack([out.hidden_states[i] for i in SELECT_LAYERS], dim=2)  # (1, L, n, d)
        return hidden[0, PREFIX_TOKENS:].to("cpu", torch.bfloat16)  # (L, n, d)

    # --- vae --------------------------------------------------------------
    def load_vae(self):
        from diffusers import AutoencoderKLQwenImage
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

        self.log(f"Loading VAE {self.profile['vae']}")
        sd = comfy_io.dequantize_all(comfy_io.load_state_dict(self.profile["vae"]), self.dtype)
        if not any(k.startswith("encoder.conv_in") for k in sd):
            sd = convert_wan_vae_to_diffusers(sd)
        vae = AutoencoderKLQwenImage()
        vae.load_state_dict(sd, strict=True)
        self.vae = vae.to(self.device, self.dtype).eval().requires_grad_(False)
        cfg = self.vae.config
        self._lat_mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1)
        self._lat_std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1)

    @torch.no_grad()
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        x = pixels.to(self.device, self.dtype).unsqueeze(2)
        lat = self.vae.encode(x).latent_dist.mode()
        lat = (lat - self._lat_mean.to(lat)) / self._lat_std.to(lat)
        return lat[0, :, 0].to("cpu", torch.bfloat16)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        lat = latents.to(self.device, self.dtype).unsqueeze(2)
        lat = lat * self._lat_std.to(lat) + self._lat_mean.to(lat)
        self.vae.enable_tiling()
        img = self.vae.decode(lat).sample
        return img[:, :, 0].float().clamp(-1, 1)

    def unload_vae(self):
        self.vae = None
        gc.collect()
        torch.cuda.empty_cache()

    # --- transformer --------------------------------------------------------
    def build_transformer(self, shapes):
        with torch.device("meta"):
            return Krea2DiT(Krea2Config(layers=count_blocks(shapes, "blocks")))

    def sample_timesteps(self, batch: int, latent_hw: tuple[int, int], settings: dict) -> torch.Tensor:
        mode = settings.get("timestep_sampling", "shift")
        if mode == "uniform":
            return torch.rand(batch)
        u = torch.sigmoid(torch.randn(batch)) if mode == "logit_normal" else torch.rand(batch)
        if mode in ("shift", "logit_normal"):
            seq = (latent_hw[0] // PATCH) * (latent_hw[1] // PATCH)
            u = time_shift(mu_for(seq), u.clamp(1e-4, 1 - 1e-4))
        return u

    def _prepare(self, latents: torch.Tensor, text: list[torch.Tensor]):
        b, _, h, w = latents.shape
        dev = latents.device
        lens = [t.shape[0] for t in text]
        n, d = text[0].shape[1], text[0].shape[2]
        ctx = torch.zeros(b, max(lens), n, d, device=dev, dtype=self.dtype)
        txtmask = torch.zeros(b, max(lens), dtype=torch.bool, device=dev)
        for i, t in enumerate(text):
            ctx[i, : t.shape[0]] = t.to(dev, self.dtype)
            txtmask[i, : t.shape[0]] = True
        h_, w_ = h // PATCH, w // PATCH
        ids = torch.zeros(h_, w_, 3, device=dev)
        ids[..., 1] = torch.arange(h_, device=dev)[:, None]
        ids[..., 2] = torch.arange(w_, device=dev)[None, :]
        imgpos = repeat(ids, "h w c -> b (h w) c", b=b)
        tokens = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=PATCH, pw=PATCH)
        pos = torch.cat([torch.zeros(b, ctx.shape[1], 3, device=dev), imgpos], dim=1)
        mask = torch.cat([txtmask, torch.ones(b, h_ * w_, dtype=torch.bool, device=dev)], dim=1)
        return tokens, ctx, pos, mask

    def predict(self, model, noisy, t, text):
        _, _, h, w = noisy.shape
        tokens, ctx, pos, mask = self._prepare(noisy.to(self.dtype), text)
        out = model(img=tokens, context=ctx, t=t.to(self.device, torch.float32), pos=pos, mask=mask)
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=PATCH, pw=PATCH, h=h // PATCH, w=w // PATCH)

    # --- sampling -----------------------------------------------------------
    def cfg_scale(self, profile_cfg) -> float:
        cfg = float(profile_cfg or 0)  # Krea profiles store the extra guidance (0 = off)
        return 1.0 + cfg if cfg > 0 else 1.0

    def _mu(self, width: int, height: int) -> float:
        mu = self.profile.get("sample_mu")
        return float(mu) if mu not in (None, "") else mu_for((height // VAE_FACTOR // PATCH) * (width // VAE_FACTOR // PATCH))

    def default_shift(self, width, height):
        return math.exp(self._mu(width, height))

    def default_sigmas(self, steps, width, height):
        mu = self._mu(width, height)
        ts = torch.linspace(1, 0, steps + 1)
        return [float(time_shift(mu, t)) if 0 < t < 1 else float(t) for t in ts]

    def prepare_sampling(self, model, cond, uncond, width, height, cfg, seed, images=None):
        h, w = height // VAE_FACTOR, width // VAE_FACTOR
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn(1, 16, h, w, generator=g).to(self.device, torch.float32)

        def velocity(x, s):
            t = torch.full((1,), s, device=self.device)
            v = self.predict(model, x, t, [cond])
            if cfg is not None and cfg > 0 and cfg != 1 and uncond is not None:
                v = v + (cfg - 1) * (v - self.predict(model, x, t, [uncond]))
            return v
        return x, velocity

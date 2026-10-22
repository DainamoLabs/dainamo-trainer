"""MOSS-Image driver.

Loads ComfyUI-format files directly:
  * diffusion model: MOSS-Image DiT (fused ``proj.qkv``), bf16 / fp8_scaled / int8 (incl. convrot)
  * text encoder: Qwen3.5-9B (``model.language_model.*``), bf16 / fp8_scaled / int8; vision tower,
    LM head and MTP head are never read
  * VAE: FLUX.2 VAE, BFL layout (``encoder.down.N...`` + ``bn.*``) or diffusers layout

Conditioning, latent packing, the sampler and the training noise distribution follow OpenMOSS's
reference pipeline (``pipeline_moss_image.py``, Apache-2.0) and the MOSS-Image technical report:
  * prompt -> Qwen chat template (one user turn, generation prompt, thinking on) -> Qwen3.5's final
    hidden state, every token, at its true length (no padding)
  * FLUX.2 latents packed 2x2 to 128 channels and normalised with the VAE's batch-norm statistics
  * the DiT is told 1000 * sigma; target noise - x0; unweighted MSE (report eq. 3)
  * training sigma: the last pretraining interval's distribution (report eq. 4, Table 6)
  * sampling: shift-5 grid snapped to the 1000 training sigmas, Euler, CFG with block 22 skipped
    in the negative branch
"""

from __future__ import annotations

import gc

import torch

from ... import comfy_io
from ...fp8_linear import assert_materialized, materialize
from ...hf_assets import moss_tokenizer_and_config
from ..base import FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, count_blocks
from .model import MossConfig, MossImageDiT

T_TRAIN = 1000
MAX_TEXT_TOKENS = 4096
# report eq. (4): Z ~ N(mu, s^2), S = 1 - sigmoid(Z), truncated at the cutoff, 1% uniform
PRETRAIN_MU, PRETRAIN_S, PRETRAIN_CUTOFF, PRETRAIN_UNIFORM = -1.09, 0.95, 0.95, 0.01


def _sigma_cdf(x: torch.Tensor) -> torch.Tensor:
    """F(x) = P(S <= x) = 1 - Phi((log((1 - x) / x) - mu) / s)."""
    return 1 - torch.special.ndtr((torch.log((1 - x) / x) - PRETRAIN_MU) / PRETRAIN_S)


def pretrain_sigmas(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw n sigmas from report eq. (4): sigma_j = j / T with
    p_j = (1 - rho) * (F(a j / T) - F(a (j - 1) / T)) / F(a) + rho / T."""
    u = torch.rand(n, generator=generator, dtype=torch.float64)
    pick_uniform = torch.rand(n, generator=generator, dtype=torch.float64) < PRETRAIN_UNIFORM
    j_uniform = torch.randint(1, T_TRAIN + 1, (n,), generator=generator)
    # inverse CDF of the truncated distribution: p uniform on (0, F(a)]
    p = u.clamp_min(1e-12) * _sigma_cdf(torch.tensor(PRETRAIN_CUTOFF, dtype=torch.float64))
    z = PRETRAIN_MU + PRETRAIN_S * torch.special.ndtri(1 - p)
    s = torch.sigmoid(-z)
    j = torch.ceil(T_TRAIN * s / PRETRAIN_CUTOFF).clamp(1, T_TRAIN).long()
    return torch.where(pick_uniform, j_uniform, j).double() / T_TRAIN


def sampling_sigmas(steps: int, shift: float) -> torch.Tensor:
    """MOSSImageScheduler: a shifted grid snapped to the nearest of the 1000 training sigmas,
    then the terminal 0 (as diffusers' FlowMatchEulerDiscreteScheduler appends)."""
    def grid(count, s):
        lower = s / (T_TRAIN + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(T_TRAIN, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=T_TRAIN - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.cat([torch.unique_consecutive(training[nearest]), torch.zeros(1)])


def pack_latents(z: torch.Tensor) -> torch.Tensor:
    """(B, 32, 2h, 2w) -> (B, 128, h, w), channel = c * 4 + 2 * dy + dx."""
    b, c, h, w = z.shape
    return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpack_latents(z: torch.Tensor) -> torch.Tensor:
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def _decoder_channels(sd: dict[str, torch.Tensor]) -> list[int] | None:
    """Decoder widths in encoder order, read from the weights (FLUX.2 VAEs ship full or slim decoders)."""
    bfl = [sd.get(f"decoder.up.{i}.block.0.conv2.weight") for i in range(4)]
    if all(t is not None for t in bfl):
        return [int(t.shape[0]) for t in bfl]
    dif = [sd.get(f"decoder.up_blocks.{i}.resnets.0.conv2.weight") for i in range(4)]
    if all(t is not None for t in dif):
        return [int(t.shape[0]) for t in reversed(dif)]
    return None


class MossImageDriver(FamilyDriver):
    info = FamilyInfo(
        id="moss_image",
        label="MOSS-Image",
        media=["image"],
        profile_fields=[
            ProfileField("diffusion_model", "Diffusion model", hint="MOSS-Image (Base or SFT)",
                         help="A ComfyUI single file (fused q/k/v): bf16, fp8_scaled or int8_convrot."),
            ProfileField("text_encoder", "Text encoder (CLIP)", hint="Qwen3.5-9B",
                         help="Qwen3.5-9B as one .safetensors file, any precision (e.g. ComfyUI's qwen3.5_9b files)."),
            ProfileField("vae", "VAE", hint="FLUX.2 VAE",
                         help="The FLUX.2 VAE. The FLUX.1 VAE (ae.safetensors) won't work."),
            ProfileField("sample_steps", "Sample steps", kind="number", default=50,
                         help="MOSS-Image's user guide recommends 50 or more; fewer makes samples quicker but rougher."),
            ProfileField("sample_cfg", "Sampling CFG (1 = off)", kind="number", default=3.0),
            ProfileField("sample_shift", "Sampling shift", kind="number", default=5.0),
            ProfileField("sample_uncond_skip", "Skip blocks in the negative branch", kind="text", required=False,
                         default="22", help="Comma-separated block indices (0-based). Blank disables it."),
            ProfileField("sample_negative", "Sampling negative prompt", kind="text", required=False, default="",
                         help='The user guide suggests "nsfw, worst quality, bad anatomy, watermark, jpeg artifact".'),
        ],
        training_defaults={
            "resolution": 1024, "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 2000,
            "batch_size": 1, "optimizer": "adamw8bit", "timestep_sampling": "pretrain",
        },
        timestep_modes=[
            ("pretrain", "MOSS pretraining schedule (report eq. 4)"),
            ("logit_normal", "Logit-normal (0, 1)"),
            ("uniform", "Uniform"),
        ],
        sliders=True,
        # 5.1B weights: 10.2 GB in bf16, with Adafactor's factored state beside them
        finetune={"learning_rate": 1e-5, "optimizer": "adafactor", "keep_last": 2},
    )
    bucket_divisibility = 16  # VAE /8, then 2x2 packing
    caption_tokens = MAX_TEXT_TOKENS
    block_list = "layers"
    lora_targets = ["layers.*.proj.*"]
    memory = MemoryProfile(saved=6_300, full=273_000, work=377_000, infer=195_000,
                           vae_encode=(0.1e9, 2_580), vae_decode=(0.1e9, 3_860))
    lora_metadata = {"modelspec.architecture": "moss-image/lora", "dainamo.family": "moss_image"}
    signature_keys = {
        "diffusion_model": ["layers.41.proj.ffn_down.weight", "text_embedder.proj.weight"],
        "text_encoder": ["layers.31.mlp.down_proj.weight", "layers.0.linear_attn.in_proj_qkv.weight"],
        "vae": ["bn.running_mean", "encoder.conv_in.weight"],
    }

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None

    # --- text -------------------------------------------------------------
    def load_text_encoder(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding

        tokenizer, config = moss_tokenizer_and_config()
        text_cfg = config.text_config
        text_cfg._attn_implementation = "sdpa"
        path = self.profile["text_encoder"]
        self.log(f"Loading text encoder {path}")
        keys, _ = comfy_io.read_header(path)
        prefix = next((p for p in ("model.language_model.", "language_model.", "model.")
                       if any(k.startswith(p + "layers.") for k in keys)), None)
        if prefix is None:
            raise ValueError(f"{path} doesn't look like a Qwen3.5 text encoder")
        loaded = comfy_io.load_state_dict(path, strip_prefix=prefix, only_prefixed=True,
                                          device=self.text_encoder_device)
        with torch.device("meta"):
            model = Qwen3_5TextModel(text_cfg)
        materialize(model, loaded, self.dtype, keep_fp8=True, device=None)
        del loaded  # else its tensors stay in RAM beside the streamed copies
        model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=text_cfg, device=self.device)
        assert_materialized(model)
        self.stream_text_encoder(model).eval()
        self.tokenizer, self.text_encoder = tokenizer, model

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_TOKENS)
        out = self.text_encoder(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device), use_cache=False)
        return out.last_hidden_state[0].to("cpu", torch.bfloat16)  # (L, 4096)

    # --- vae --------------------------------------------------------------
    def load_vae(self):
        from diffusers import AutoencoderKLFlux2
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

        path = self.profile["vae"]
        self.log(f"Loading VAE {path}")
        sd = comfy_io.dequantize_all(comfy_io.load_state_dict(path), torch.float32)
        vae = AutoencoderKLFlux2(decoder_block_out_channels=_decoder_channels(sd))
        if not any(k.startswith("encoder.down_blocks.") for k in sd):  # BFL layout -> diffusers names
            conv = convert_ldm_vae_checkpoint(sd, vae.config)
            for new, old in (("quant_conv", "encoder.quant_conv"), ("post_quant_conv", "decoder.post_quant_conv")):
                for p in ("weight", "bias"):
                    if f"{old}.{p}" in sd:
                        conv[f"{new}.{p}"] = sd[f"{old}.{p}"]
            conv.update({k: v for k, v in sd.items() if k.startswith("bn.")})
            sd = conv
        result = vae.load_state_dict(sd, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
        if missing or result.unexpected_keys:
            raise ValueError(f"{path} isn't a FLUX.2 VAE: missing={missing[:6]} unexpected={result.unexpected_keys[:6]}")
        # the reference runs the VAE in fp32; it's small
        self.vae = vae.to(self.device, torch.float32).eval().requires_grad_(False)
        eps = self.vae.config.batch_norm_eps
        self._bn_mean = self.vae.bn.running_mean.float().view(1, -1, 1, 1)
        self._bn_std = (self.vae.bn.running_var.float() + eps).sqrt().view(1, -1, 1, 1)

    @torch.no_grad()
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(pixels.to(self.device, torch.float32)).latent_dist.mode()
        z = (pack_latents(z) - self._bn_mean) / self._bn_std
        return z[0].to("cpu", torch.float32)  # (128, H/16, W/16)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(self.device, torch.float32) * self._bn_std + self._bn_mean
        return self.vae.decode(unpack_latents(z)).sample.float().clamp(-1, 1)

    def unload_vae(self):
        self.vae = None
        gc.collect()
        torch.cuda.empty_cache()

    # --- transformer --------------------------------------------------------
    def build_transformer(self, shapes):
        if "layers.0.proj.q.weight" in shapes:
            raise ValueError(f"{self.profile['diffusion_model']} stores q/k/v separately; use a ComfyUI "
                             "MOSS-Image file (fused proj.qkv)")
        with torch.device("meta"):
            model = MossImageDiT(MossConfig(num_layers=count_blocks(shapes, "layers")))
        model.compute_dtype = self.dtype
        return model

    def check_transformer_state(self, path, loaded):
        loaded.tensors.pop("rope.bands", None)  # derived constant, recomputed in fp32

    def sample_timesteps(self, batch: int, latent_hw: tuple[int, int], settings: dict) -> torch.Tensor:
        mode = settings.get("timestep_sampling", "pretrain")
        if mode == "uniform":
            return torch.rand(batch).clamp(1e-3, 1.0)
        if mode == "logit_normal":
            return torch.sigmoid(torch.randn(batch))
        return pretrain_sigmas(batch).float()

    def predict(self, model, noisy, t, text):
        return model(noisy, t.to(self.device, torch.float32) * T_TRAIN, text)

    @torch.no_grad()
    def default_shift(self, width, height):
        return float(self.profile.get("sample_shift") or 5.0)

    def default_sigmas(self, steps, width, height):
        return sampling_sigmas(int(steps), self.default_shift(width, height)).tolist()

    def prepare_sampling(self, model, cond, uncond, width, height, cfg, seed, images=None):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn(1, model.config.latent_channels, height // 16, width // 16, generator=g,
                        dtype=torch.float32).to(self.device)
        skip = frozenset(int(s) for s in str(self.profile.get("sample_uncond_skip") or "").replace(" ", "").split(",")
                         if s.strip().isdigit() and int(s) < model.config.num_layers)
        use_cfg = cfg is not None and cfg != 1 and cfg > 0 and uncond is not None

        def velocity(x, s):
            t = torch.full((1,), s * T_TRAIN, device=self.device)
            v = model(x, t, [cond]).float()
            if use_cfg:
                v_uncond = model(x, t, [uncond], skip_layers=skip).float()
                v = v_uncond + cfg * (v - v_uncond)
            return v
        return x, velocity

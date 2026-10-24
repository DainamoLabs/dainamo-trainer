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

import torch

from ... import comfy_io
from ...fp8_linear import assert_materialized, materialize
from ...hf_assets import moss_tokenizer_and_config
from .. import flux2_vae
from ..base import Download, FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, count_blocks
from ..flux2_vae import Flux2VAE, pack_latents, unpack_latents  # noqa: F401  (the packing is MOSS's too)
from .model import MossConfig, MossImageDiT

# where the model manager downloads the files from, at the commit they were checked against
HF_REPO, HF_REV = "Comfy-Org/MOSS-Image", "24147db865fcdc207270f844777587223fed8d6e"

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
    vae_fp32 = True  # the FLUX.2 VAE runs in fp32 (``flux2_vae``)
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
    downloads = [
        Download("diffusion_model", HF_REPO, "diffusion_models/moss_image_sft_int8_convrot.safetensors", 5_413_698_958,
                 "SFT · int8 convrot", HF_REV, recommended=True,
                 sha256="b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f"),
        Download("diffusion_model", HF_REPO, "diffusion_models/moss_image_sft_bf16.safetensors", 10_164_655_808,
                 "SFT · bf16", HF_REV, sha256="f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f"),
        Download("diffusion_model", HF_REPO, "diffusion_models/moss_image_base_int8_convrot.safetensors",
                 5_413_698_958, "Base · int8 convrot", HF_REV,
                 sha256="f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f"),
        Download("diffusion_model", HF_REPO, "diffusion_models/moss_image_base_bf16.safetensors", 10_164_655_808,
                 "Base · bf16", HF_REV, sha256="cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f"),
        Download("text_encoder", HF_REPO, "text_encoders/qwen3.5_9b_int8_convrot.safetensors", 9_748_121_532,
                 "int8 convrot", HF_REV, recommended=True,
                 sha256="32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a"),
        Download("text_encoder", HF_REPO, "text_encoders/qwen3.5_9b_bf16.safetensors", 19_306_312_328, "bf16", HF_REV,
                 sha256="7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53"),
        Download("vae", HF_REPO, "vae/flux2-vae.safetensors", 336_211_292, "bf16", HF_REV, recommended=True,
                 sha256="868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"),
    ]

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
        self.vae = Flux2VAE(self.profile["vae"], self.device, self.log)

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels)  # (128, H/16, W/16): packed and normalised

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)

    def unload_vae(self):
        self.vae = None
        flux2_vae.unload(None)

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

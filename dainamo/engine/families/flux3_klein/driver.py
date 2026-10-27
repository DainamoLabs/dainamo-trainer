"""FLUX 3 Image Klein driver: text-to-image and edit LoRAs on the Base model.

Loads ComfyUI-format files directly:
  * diffusion model: FLUX 3 Image Klein Base, ComfyUI's single file (q/k/v fused into ``qkv``), bf16 / int8
    convrot / nvfp4. Its per-block ``comfy_attention.config`` hints are skipped. The distilled FLUX 3 Image Klein
    (a flow map, with ``ref_time_in``) is refused: it samples without CFG at a time it jumps to, which a velocity
    LoRA doesn't train.
  * text encoder: Qwen3-VL-4B-Instruct's text model (``model.layers.*``), any precision; the vision tower and LM
    head are never read. The same file as Krea 2's.
  * VAE: the FLUX.2 VAE (``flux2_vae``): 128-channel latents at 1/16 of the image, normalised by its batch norm.

Conditioning, token layout and sampling follow Black Forest Labs' reference code (``flux3``, Apache 2.0:
``text_encoder.py``, ``pipeline.py``, ``sampling.py``, ``autoencoder.py``):
  * prompt -> Qwen chat template (one user turn, generation prompt) -> right-padded to a multiple of 80 tokens
    (at most 4096), positions the mask's running count -> hidden states of layers 4, 8, ..., 32 side by side,
    (L, 20480). The DiT reads every token, padding included, at ids (0, 0, 0, position).
  * image tokens: the packed latents, row by row, at ids (0, row / 2, col / 2, 0) (``IMAGE_ROPE_SCALE``)
  * edits: each reference image VAE-encoded the same way into the ``image_cond`` mode, at ids whose last axis is
    -(1 + its index), clean (timestep 0); the loss is on the target's tokens only
  * the velocity noise - clean, Euler, sigmas t' = 16 t / (1 + 15 t) for t from 1 to 0 (``ALPHA``), CFG 4 against
    the empty prompt; training noise levels drawn uniformly through the same shift by default

The model was trained on captions its prompt upsampler writes (long, detailed descriptions), so captions and
prompts should be written that way. The upsampler isn't part of training or sampling; Model files downloads it for
the caption provider, and the "FLUX 3 (prompt upsampler style)" caption style asks any captioner for its captions.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from PIL import Image, ImageOps

from ... import comfy_io
from ... import dataset as ds
from ...fp8_linear import assert_materialized, materialize
from ...hf_assets import qwen3vl_tokenizer_and_config
from .. import flux2_vae
from ..base import Download, FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, Workflow, count_blocks
from ..flux2_vae import TOKEN_CHANNELS, TOKEN_PIXELS, Flux2VAE
from .model import FLOW_MAP_KEY, Flux3Config, Flux3KleinDiT, Pack

# where the model manager downloads the files from, at the commit they were checked against (auto-gated: sign in
# and accept the FLUX Kommunity License on Hugging Face first)
HF_REPO, HF_REV = "black-forest-labs/flux-3-image-klein-comfy", "feef60b0aaf63a4773ace96636ebbcf43b881312"

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)  # Qwen3-VL-4B hidden states the DiT reads
# tokens a caption may use: the health check's limit, and where encoding cuts a caption off (before the chat
# template goes round it, which stays whole). The reference reads up to 4096, template included; 512 keeps caches
# and VRAM bounded
CAPTION_TOKENS = 512
PAD_MULTIPLE = 80
IMAGE_ROPE_SCALE = 0.5  # image rows and columns step RoPE by half a position
ALPHA = 16.0  # the schedule's shift, at every size
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024  # each reference of a several-reference edit
SINGLE_REFERENCE_MAX_PIXELS = 4 * 1024 * 1024
DISTILLED = ("This is the distilled FLUX 3 Image Klein (a flow map, sampled in 16 steps without CFG), which Dainamo "
             "can't train or sample yet. Use FLUX 3 Image Klein Base.")


def shift(u: torch.Tensor, alpha: float = ALPHA) -> torch.Tensor:
    return alpha * u / (1.0 + (alpha - 1.0) * u)


def sampling_sigmas(steps: int, alpha: float = ALPHA) -> torch.Tensor:
    """The reference's schedule (``sampling.timesteps``): steps + 1 sigmas from 1 to 0."""
    return shift(torch.linspace(1.0, 0.0, steps + 1), alpha)


def image_ids(h: int, w: int, index: int | None = None, device=None) -> torch.Tensor:
    """(h * w, 4) RoPE ids of a latent grid, row by row: (0, row / 2, col / 2, 0), or for reference ``index`` the
    last axis -(1 + index)."""
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    ids = torch.zeros(h, w, 4, device=device)
    ids[..., 1] = rows * IMAGE_ROPE_SCALE
    ids[..., 2] = cols * IMAGE_ROPE_SCALE
    if index is not None:
        ids[..., 3] = -(1 + index)
    return ids.reshape(h * w, 4)


def text_ids(length: int, device=None) -> torch.Tensor:
    ids = torch.zeros(length, 4, device=device)
    ids[:, 3] = torch.arange(length, device=device)
    return ids


def to_tokens(latents: torch.Tensor) -> torch.Tensor:
    """(B, 128, h, w) -> (B * h * w, 128), each sample's tokens row by row."""
    b, c, h, w = latents.shape
    return latents.permute(0, 2, 3, 1).reshape(b * h * w, c)


def from_tokens(tokens: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    return tokens.reshape(b, h, w, -1).permute(0, 3, 1, 2)


def reference_size(width: int, height: int, target_pixels: int, count: int = 1) -> tuple[int, int]:
    """A reference scaled to the target's area (at most 4 MP alone, 1 MP each of several, as the reference pipeline
    caps them), aspect kept, on the 16 px grid."""
    budget = min(target_pixels, SINGLE_REFERENCE_MAX_PIXELS if count == 1 else MULTI_REFERENCE_MAX_PIXELS)
    w = math.sqrt(budget * width / height)
    snap = lambda v: max(TOKEN_PIXELS, round(v / TOKEN_PIXELS) * TOKEN_PIXELS)  # noqa: E731
    return snap(w), snap(w * height / width)


def prepare_reference(img: Image.Image, target_pixels: int, count: int = 1, flip: bool = False) -> torch.Tensor:
    """A reference as the VAE takes it: RGB, cropped to ``reference_size``'s aspect and resized (Lanczos), -> (1, 3,
    H, W) in [-1, 1]."""
    img = ds.to_rgb(ImageOps.exif_transpose(img))
    img = ImageOps.fit(img, reference_size(img.width, img.height, target_pixels, count), Image.Resampling.LANCZOS)
    if flip:
        img = ImageOps.mirror(img)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]


def templated(tok, prompt: str) -> tuple[str, int]:
    """The prompt in the reference's chat template (one user turn, the generation prompt), its first
    ``CAPTION_TOKENS`` tokens only, and the tokens it's padded to: a multiple of ``PAD_MULTIPLE``."""
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > CAPTION_TOKENS:
        prompt = tok.decode(ids[:CAPTION_TOKENS])
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    real = len(tok(text)["input_ids"])
    return text, math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE


class Flux3KleinDriver(FamilyDriver):
    info = FamilyInfo(
        id="flux3_klein",
        label="FLUX 3 Image Klein",
        media=["image"],
        profile_fields=[
            ProfileField("diffusion_model", "Diffusion model",
                         hint="FLUX 3 Image Klein Base (bf16 / int8 convrot / nvfp4)",
                         help="FLUX Kommunity License. A ComfyUI single file (fused q/k/v) of the Base model; the "
                              "distilled one can't be trained. It was trained on long, detailed captions: caption "
                              "your images as full descriptions."),
            ProfileField("text_encoder", "Text encoder (CLIP)", hint="Qwen3-VL-4B",
                         help="Qwen3-VL-4B-Instruct as one .safetensors file, any precision (e.g. ComfyUI's qwen3vl_4b "
                              "files, the same as Krea 2's). Not the 8B."),
            ProfileField("vae", "VAE", hint="FLUX.2 VAE",
                         help="The FLUX.2 VAE (flux2-vae). The FLUX.1 VAE (ae.safetensors) and the Qwen-Image VAE "
                              "won't work."),
            ProfileField("sample_steps", "Sample steps", kind="number", default=28,
                         help="The reference samples Base in 50 steps; fewer is quicker and a little rougher."),
            ProfileField("sample_cfg", "Sampling CFG (1 = off)", kind="number", default=4.0),
            ProfileField("sample_negative", "Sampling negative prompt", kind="text", required=False, default="",
                         help="Blank: the empty prompt, as the reference's unconditional branch."),
        ],
        training_defaults={
            "resolution": 1024, "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 2000,
            "batch_size": 1, "optimizer": "adamw8bit", "timestep_sampling": "shift",
        },
        timestep_modes=[("shift", "Shift 16 (the sampling schedule's)"), ("logit_normal", "Logit-normal + shift 16"),
                        ("uniform", "Uniform")],
        workflows=[
            Workflow("t2i"),
            Workflow("edit", images={"label": "Reference images", "min": 1, "max": MAX_REFERENCES,
                                     "help": "Scaled to the output's size (1 MP at most each, with several)."}),
        ],
        sliders=True,
        sources={"label": "Source dataset", "noun": "source image", "max": 3, "variants": None,
                 "without": "train text-to-image",
                 "help": "For an edit LoRA: a dataset of the images before the edit, paired with this dataset's "
                         "images (the results) by file name: photo.png pairs with photo.jpg. Each result's caption "
                         "is then the edit instruction. A second or third source dataset adds more reference images "
                         "to each edit. With none, or for an image without a source, it trains text-to-image."},
    )
    vae_fp32 = True  # the FLUX.2 VAE runs in fp32 (``flux2_vae``)
    bucket_divisibility = TOKEN_PIXELS
    caption_tokens = CAPTION_TOKENS
    # the 28 shared blocks are what offloading streams; the 15 mode blocks (5 per mode) stay on the GPU
    block_list = "single_blocks"
    lora_targets = [f"{blocks}.*.{layer}" for blocks in ("single_blocks", "content_mode_blocks", "txt_mode_blocks")
                    for layer in ("qkv", "attn_out", "mlp_in", "mlp_out")]
    # measured with scripts/measure_memory.py on the int8 convrot Base file (the mode blocks' share is in ``work``);
    # captions as the model wants them (long descriptions) come to about 320 tokens with the template and padding
    memory = MemoryProfile(saved=6_200, full=182_000, work=355_000, infer=267_000,
                           vae_encode=(0.1e9, 2_580), vae_decode=(0.1e9, 3_860), text_tokens=320)
    text_encoder_skip = ("lm_head.", "visual.")
    lora_metadata = {"modelspec.architecture": "flux-3-image-klein/lora", "dainamo.family": "flux3_klein"}
    signature_keys = {
        "diffusion_model": ["single_blocks.27.qkv.weight", "content_mode_blocks.image_cond.4.qkv.weight",
                            "early_stream_modulations.txt.lin.weight"],
        "text_encoder": ["layers.35.mlp.down_proj.weight", "visual.blocks.0.attn.qkv.weight"],  # Qwen3-VL, not Qwen3
        "vae": ["bn.running_mean", "encoder.conv_in.weight"],
    }
    signature_shapes = {
        "text_encoder": {"layers.35.mlp.down_proj.weight": (2560, 9728)},  # 4B, not Qwen-Image 2.1's 8B
        "vae": {"bn.running_mean": (128,)},  # the FLUX.2 VAE's 32 x 2 x 2 statistics
    }
    downloads = [
        Download("diffusion_model", HF_REPO, "diffusion_models/flux3-image-klein-base-int8-convrot.safetensors",
                 5_807_427_339, "Base · int8 convrot", HF_REV, gated=True, recommended=True,
                 sha256="8949ecfc1bf3871b08be02cf43ec7276a4914e1a86eade6b5eb8f4204426bc92"),
        Download("diffusion_model", HF_REPO, "diffusion_models/flux3-image-klein-base-bf16.safetensors",
                 11_076_957_072, "Base · bf16", HF_REV, gated=True,
                 sha256="5c921759dc03663dbb67df11d8e3d317d2d5b893b31b969c95dc77ebf861b19c"),
        Download("diffusion_model", HF_REPO, "diffusion_models/flux3-image-klein-base-nvfp4.safetensors",
                 5_257_238_192, "Base · nvfp4", HF_REV, gated=True,
                 sha256="f99714a57ada7b7f39ebb7d916a1b842324c8ec483943816f5b3ea5c062b08a9"),
        Download("text_encoder", HF_REPO, "text_encoders/qwen3vl_4b_int8_convrot.safetensors", 4_858_559_240,
                 "int8 convrot", HF_REV, gated=True, recommended=True,
                 sha256="32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c"),
        Download("text_encoder", HF_REPO, "text_encoders/qwen3vl_4b_bf16.safetensors", 8_875_719_408, "bf16", HF_REV,
                 gated=True, sha256="f0b24fc5c8d7c0ea6ca909da14c20af5cac53f22506be942c6c956f838b9630f"),
        Download("vae", HF_REPO, "vae/flux2-vae.safetensors", 336_211_292, "bf16", HF_REV, gated=True,
                 recommended=True, sha256="868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"),
        # the prompt upsampler (a Qwen3.5-9B, Apache 2.0), which wrote the captions the model was trained on: no
        # profile uses it, but the caption provider can run it as a text encoder (docs/captions.md). Its bf16 file is
        # byte for byte black-forest-labs/flux-3-image-klein-pup's model.safetensors
        Download("prompt_upsampler", HF_REPO,
                 "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors", 9_748_113_996,
                 "int8 convrot", HF_REV, gated=True, recommended=True,
                 sha256="cb8cb9b9dcd2569509de755dca54d96fd93a16a7aabfd2f4051f0a144230620f"),
        Download("prompt_upsampler", HF_REPO, "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors",
                 18_819_722_392, "bf16", HF_REV, gated=True,
                 sha256="f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4"),
    ]
    extra_fields = {"prompt_upsampler": ("Prompt upsampler", "for captions")}

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None

    @classmethod
    def validate_profile(cls, profile: dict) -> list[dict]:
        out = super().validate_profile(profile)
        path = os.path.expanduser((profile.get("diffusion_model") or "").strip())
        entry = next((o for o in out if o["field"] == "diffusion_model" and o["level"] == "ok"), None)
        if entry is not None:
            try:
                distilled = FLOW_MAP_KEY in comfy_io.read_header(path)[0]
            except Exception:
                distilled = False
            if distilled:
                entry.update(level="error", message=DISTILLED)
        return out

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

    def cache_inputs(self, kind: str, settings: dict) -> dict:
        inputs = super().cache_inputs(kind, settings)
        # captions encoded before a long one was cut ahead of the template (``templated``) are encoded again
        return {**inputs, "caption_cut": "before template"} if kind == "text" else inputs

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """-> (L, 20480) bf16, L the templated prompt's tokens rounded up to a multiple of 80 (padding included)."""
        tok = self.tokenizer
        text, length = templated(tok, prompt)
        enc = tok(text, return_tensors="pt", padding="max_length", max_length=length)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        out = self.text_encoder(input_ids=ids, attention_mask=mask, position_ids=positions,
                                output_hidden_states=True, use_cache=False)
        stacked = torch.stack([out.hidden_states[k][0] for k in OUTPUT_LAYERS], dim=1)  # (L, 8, 2560)
        return stacked.reshape(length, -1).to("cpu", torch.bfloat16)

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

    # --- references ----------------------------------------------------------
    def _references(self, paths: list[str], target_pixels: int, flip: bool = False) -> list[torch.Tensor]:
        if len(paths) > MAX_REFERENCES:
            raise ValueError(f"FLUX 3 Image Klein takes at most {MAX_REFERENCES} reference images")
        out = []
        for p in paths:
            with Image.open(p) as im:
                out.append(prepare_reference(im, target_pixels, len(paths), flip))
        return out

    @torch.no_grad()
    def encode_item(self, item, settings: dict) -> dict:
        payload = super().encode_item(item, settings)
        if item.refs:
            w, h = item.size
            payload["refs"] = [self.encode_image(r) for r in self._references(item.refs, w * h, item.flip)]
        return payload

    # --- transformer --------------------------------------------------------
    def build_transformer(self, shapes):
        if FLOW_MAP_KEY in shapes:
            raise ValueError(f"{self.profile.get('diffusion_model')}: {DISTILLED}")
        if "single_blocks.0.q_proj.weight" in shapes:
            raise ValueError(f"{self.profile.get('diffusion_model')} stores q/k/v separately (Black Forest Labs' own "
                             "layout); use the ComfyUI file (black-forest-labs/flux-3-image-klein-comfy)")
        with torch.device("meta"):
            model = Flux3KleinDiT(Flux3Config(depth=count_blocks(shapes, "txt_mode_blocks"),
                                              depth_single=count_blocks(shapes, "single_blocks")))
        model.compute_dtype = self.dtype
        return model

    def check_transformer_state(self, path, loaded):
        for k in [k for k in loaded.tensors if k.endswith(".comfy_attention.config")]:  # ComfyUI runtime hints
            del loaded.tensors[k]

    def sample_timesteps(self, batch: int, latent_hw, settings: dict) -> torch.Tensor:
        mode = settings.get("timestep_sampling", "shift")
        if mode == "uniform":
            return torch.rand(batch).clamp(1e-3, 1.0)
        u = torch.sigmoid(torch.randn(batch)) if mode == "logit_normal" else torch.rand(batch)
        return shift(u.clamp(1e-4, 1 - 1e-4))

    def _forward(self, model, noisy: torch.Tensor, t: torch.Tensor, texts: list[torch.Tensor],
                 refs: list[list[torch.Tensor]]) -> torch.Tensor:
        """Velocity (B, 128, h, w) for noisy target latents, sample i with text ``texts[i]`` (L_i, 20480) and
        reference latents ``refs[i]`` (each (128, h', w'), any number and size)."""
        b, _, h, w = noisy.shape
        d = self.device
        n = h * w
        data, ids, spans = {}, {}, {}
        data["image"] = to_tokens(noisy)
        ids["image"] = image_ids(h, w, device=d).repeat(b, 1)
        spans["image"] = [(i * n, (i + 1) * n) for i in range(b)]
        cond, cond_ids, cond_spans, start = [], [], [], 0
        for rs in refs:
            for j, r in enumerate(rs):
                cond.append(to_tokens(r[None].to(d)))
                cond_ids.append(image_ids(r.shape[-2], r.shape[-1], index=j, device=d))
            end = start + sum(r.shape[-2] * r.shape[-1] for r in rs)
            cond_spans.append((start, end))
            start = end
        data["image_cond"] = torch.cat(cond) if cond else noisy.new_zeros(0, TOKEN_CHANNELS)
        ids["image_cond"] = torch.cat(cond_ids) if cond_ids else torch.zeros(0, 4, device=d)
        spans["image_cond"] = cond_spans
        lengths = [x.shape[0] for x in texts]
        data["text"] = torch.cat([x.to(d) for x in texts])
        ids["text"] = torch.cat([text_ids(n_, device=d) for n_ in lengths])
        bounds = np.cumsum([0, *lengths]).tolist()
        spans["text"] = list(zip(bounds[:-1], bounds[1:]))
        zero = torch.zeros(b, device=d)
        pack = Pack(data, ids, spans, {"image": t.to(d, torch.float32), "image_cond": zero, "text": zero})
        return from_tokens(model(pack), b, h, w)

    def predict(self, model, noisy, t, text):
        """Text-to-image velocity (no references), as the shared slider step uses it."""
        return self._forward(model, noisy, t, text, [[] for _ in text])

    def training_loss(self, model, payloads, conds, settings, rng=None, sigma=None, generator=None,
                      deterministic=False):
        d = self.device
        lat = torch.stack([p["latent"] if isinstance(p, dict) else p for p in payloads]).to(d, torch.float32)
        noise = (torch.randn(lat.shape, generator=generator).to(d) if generator is not None
                 else torch.randn_like(lat))
        if sigma is not None:
            t = torch.full((lat.shape[0],), float(sigma), device=d)
        else:
            t = self.timesteps(lat.shape[0], lat.shape[-2:], settings).to(d)
        tb = t.view(-1, 1, 1, 1)
        noisy = (1 - tb) * lat + tb * noise
        # an edit's references stay with a dropped caption: that's the reference's unconditional edit branch
        refs = [list(p.get("refs") or []) if isinstance(p, dict) else [] for p in payloads]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self._forward(model, noisy, t, [c["text"] for c in conds], refs)
        return self.weighted_loss(pred, self.loss_target(lat, noise), t, settings, payloads, sigma is not None)

    # --- sampling -----------------------------------------------------------
    def default_shift(self, width, height):
        return ALPHA

    def default_sigmas(self, steps, width, height):
        return sampling_sigmas(int(steps)).tolist()

    def prepare_sampling(self, model, cond, uncond, width, height, cfg, seed, images=None):
        refs = []
        if images:
            if self.vae is None:
                raise RuntimeError("the VAE must be loaded to encode reference images")
            refs = [self.encode_image(r) for r in self._references(list(images), width * height)]
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn(1, TOKEN_CHANNELS, height // TOKEN_PIXELS, width // TOKEN_PIXELS, generator=g).to(self.device)
        use_cfg = cfg is not None and float(cfg) != 1 and uncond is not None

        def velocity(x, s):
            t = torch.full((1,), float(s), device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self._forward(model, x, t, [cond], [refs]).float()
                if use_cfg:  # the references condition both branches, as in the reference
                    vu = self._forward(model, x, t, [uncond], [refs]).float()
                    v = vu + float(cfg) * (v - vu)
            return v
        return x, velocity

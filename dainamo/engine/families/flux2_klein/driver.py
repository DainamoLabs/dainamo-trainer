"""FLUX.2 Klein driver: the 4B and 9B, Base or distilled, text-to-image and edit LoRAs from one checkpoint.

Loads:
  * diffusion model: Black Forest Labs' single file in its own key layout (``double_blocks`` / ``single_blocks``),
    which ComfyUI's split files share: bf16, or the fp8 releases (ComfyUI ``_quantization_metadata``, kept fp8).
    4B or 9B is read from the tensor shapes.
  * text encoder: Qwen3-4B for the 4B, Qwen3-8B for the 9B: a plain causal LM, of which only the first 27 layers
    run (the DiT reads hidden states 9, 18 and 27). One .safetensors file (ComfyUI's ``qwen_3_4b`` ...) or a
    Hugging Face folder of shards (``comfy_io.shard_files``); ``model.``-prefixed or bare keys. The tokenizer and
    config are bundled (``hf_assets.qwen3_tokenizer_and_config``).
  * VAE: the FLUX.2 VAE (``flux2_vae``): latents packed 2x2 to 128 channels at 1/16 of the image, normalised.

Conditioning, latents and sampling follow Black Forest Labs' reference (``src/flux2/text_encoder.py``,
``sampling.py``; Apache 2.0) and diffusers' ``Flux2KleinPipeline``:
  * prompt -> Qwen3 chat template (one user turn, generation prompt, thinking off), padded to 512 tokens; hidden
    states 9, 18 and 27 side by side, (512, 3 x hidden). The DiT reads all 512 tokens, pads included, at RoPE ids
    (0, 0, 0, l); image tokens at (0, h, w, 0).
  * the DiT is told the noise level (1 = noise) and predicts noise - x0
  * sampling: ``get_schedule`` (the empirical mu for the size and step count, then the exponential time shift);
    Base with CFG against an empty prompt, distilled in 4 steps without it
  * edits: each reference image is VAE-encoded (here at the target's area, on the 16 px grid) and its tokens, at ids
    (10 * (i + 1), h, w, 0), follow the image tokens with the same noise level and full attention, as the reference's
    ``denoise`` runs the Klein checkpoints; their prediction is dropped. Items without a reference train
    text-to-image.
Training noise levels: a uniform (or logit-normal) draw through the time shift whose mu grows linearly with the image
tokens, the release scheduler's base/max shift (0.5 at 256 tokens, 1.15 at 4096).
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
from ...hf_assets import qwen3_tokenizer_and_config
from .. import flux2_vae
from ..base import Download, FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, Workflow
from ..flux2_vae import TOKEN_CHANNELS, TOKEN_PIXELS, Flux2VAE
from .model import Flux2KleinConfig, Flux2KleinDiT

MAX_TEXT_TOKENS = 512
OUTPUT_LAYERS = (9, 18, 27)  # Qwen3 hidden states the DiT reads (0 = the embeddings)
TEMPLATE_TOKENS = 12  # the chat template around an empty prompt
QWEN3_WIDTHS = {2560: "4B", 4096: "8B"}  # hidden size -> Qwen3 model
DIT_SIZES = {7680: "4B", 12288: "9B"}  # txt_in width (3 x the text encoder's hidden size) -> Klein model
TEXT_FOR_DIT = {"4B": "4B", "9B": "8B"}

# FlowMatchEulerDiscreteScheduler config of the release (base/max shift over image tokens)
BASE_SEQ, MAX_SEQ, BASE_SHIFT, MAX_SHIFT = 256, 4096, 0.5, 1.15
BASE_STEPS, BASE_CFG, DISTILLED_STEPS = 28, 4.0, 4

REF_TIME_SCALE = 10  # reference i sits at RoPE time 10 * (i + 1)
MAX_REFS = 4

# where the model manager downloads the files from, at the commits they were checked against
COMFY_4B, COMFY_4B_REV = "Comfy-Org/flux2-klein-4B", "533c4cbc3d3b89af1743d9466c434e89810391c0"
COMFY_9B, COMFY_9B_REV = "Comfy-Org/flux2-klein-9B", "3f62d9d8ae1fec33c6e91453d5c712855b096b55"
NON_COMMERCIAL = "FLUX Non-Commercial License: accept it on the model's Hugging Face page first"


def mu_for(seq_len: int) -> float:
    """The release scheduler's dynamic shift: mu linear in image tokens through (256, 0.5) and (4096, 1.15)."""
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ)
    return seq_len * slope + BASE_SHIFT - slope * BASE_SEQ


def time_shift(mu: float, t: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """The reference's ``generalized_time_snr_shift`` (t = 1 stays 1, t = 0 goes to 0)."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """The reference's ``compute_empirical_mu``: the sampling shift for a size and step count."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    return float(a * num_steps + m_200 - 200.0 * a)


def sampling_sigmas(steps: int, image_seq_len: int) -> list[float]:
    """The reference's ``get_schedule``: linspace(1, 0, steps + 1) through the time shift at the empirical mu."""
    ts = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return time_shift(empirical_mu(image_seq_len, steps), ts).float().tolist()


def image_ids(h: int, w: int, t: int = 0, device=None) -> torch.Tensor:
    """(h*w, 4) RoPE ids (t, row, column, 0) of an image's tokens, row-major like ``prc_img``."""
    ids = torch.zeros(h, w, 4, device=device)
    ids[..., 0] = t
    ids[..., 1] = torch.arange(h, device=device)[:, None]
    ids[..., 2] = torch.arange(w, device=device)[None, :]
    return ids.reshape(h * w, 4)


def text_ids(length: int, device=None) -> torch.Tensor:
    """(L, 4) RoPE ids (0, 0, 0, l) of the text tokens, like ``prc_txt``."""
    ids = torch.zeros(length, 4, device=device)
    ids[:, 3] = torch.arange(length, device=device)
    return ids


def tokens(latents: torch.Tensor) -> torch.Tensor:
    """(B, 128, h, w) -> (B, h*w, 128)."""
    return latents.flatten(2).transpose(1, 2)


def reference_size(width: int, height: int, target_pixels: int) -> tuple[int, int]:
    """A reference image's size: the target's area with its own aspect, on the 16 px grid."""
    w = math.sqrt(target_pixels * width / height)
    snap = lambda v: max(TOKEN_PIXELS, round(v / TOKEN_PIXELS) * TOKEN_PIXELS)  # noqa: E731
    return snap(w), snap(w * height / width)


def prepare_reference(path: str, target_pixels: int, flip: bool = False) -> torch.Tensor:
    """A reference image as the VAE takes it: (1, 3, H, W) in [-1, 1] at ``reference_size``."""
    with Image.open(path) as im:
        img = ds.to_rgb(ImageOps.exif_transpose(im))
    if flip:
        img = ImageOps.mirror(img)
    size = reference_size(img.width, img.height, target_pixels)
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]


def header(path: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """{key: (dtype, shape)} of a file, or of every shard of a sharded checkpoint."""
    from ...memory import tensors
    out = {}
    for p in comfy_io.shard_files(path):
        out.update(tensors(p))
    return out


def text_encoder_size(keys: dict[str, tuple[str, tuple[int, ...]]]) -> str | None:
    """"4B" or "8B" for a Qwen3 text encoder's tensors (None: neither)."""
    emb = next((v for k, v in keys.items() if k.endswith("embed_tokens.weight")), None)
    return QWEN3_WIDTHS.get(int(emb[1][1])) if emb and len(emb[1]) == 2 else None


def dit_size(keys: dict[str, tuple[str, tuple[int, ...]]]) -> str | None:
    """"4B" or "9B" for a Klein diffusion model's tensors (None: neither), from txt_in's width."""
    txt = next((v for k, v in keys.items() if k.endswith("txt_in.weight")), None)
    return DIT_SIZES.get(int(txt[1][1])) if txt and len(txt[1]) == 2 else None


class Flux2KleinDriver(FamilyDriver):
    info = FamilyInfo(
        id="flux2_klein",
        label="FLUX.2 Klein",
        media=["image"],
        profile_fields=[
            ProfileField("diffusion_model", "Diffusion model", hint="FLUX.2 Klein 4B or 9B, Base or distilled",
                         help="Black Forest Labs' or ComfyUI's single file, bf16 or fp8. Train on Base: the distilled "
                              "models are tuned to sample in 4 steps, which a LoRA trained on them wears away."),
            ProfileField("text_encoder", "Text encoder (CLIP)", hint="Qwen3-4B for the 4B, Qwen3-8B for the 9B",
                         help="Qwen3-4B for the 4B model, Qwen3-8B for the 9B, any precision: one .safetensors file "
                              "(e.g. ComfyUI's qwen_3_4b) or a Hugging Face folder of shards (e.g. Qwen/Qwen3-4B's, or "
                              "its model.safetensors.index.json). Not Qwen3-VL."),
            ProfileField("vae", "VAE", hint="FLUX.2 VAE",
                         help="The FLUX.2 VAE (flux2-vae.safetensors). The FLUX.1 VAE (ae.safetensors) won't work."),
            ProfileField("variant", "Variant", kind="select", options=["base", "distilled"], default="base",
                         help="base: sampled with CFG in many steps. distilled: the 4-step release, sampled in 4 "
                              "steps without CFG. Sets the sampling defaults below."),
            ProfileField("sample_steps", "Sample steps (blank = the variant's)", kind="number", required=False,
                         help=f"Blank: {BASE_STEPS} for Base (the reference uses 50), {DISTILLED_STEPS} for distilled."),
            ProfileField("sample_cfg", "Sampling CFG (1 = off, blank = the variant's)", kind="number", required=False,
                         help=f"Blank: {BASE_CFG:g} for Base, 1 (off) for distilled, which samples without it."),
            ProfileField("sample_negative", "Sampling negative prompt", kind="text", required=False, default="",
                         help="Base's CFG branch. The reference uses an empty prompt."),
        ],
        training_defaults={
            "resolution": 1024, "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 2000,
            "batch_size": 1, "optimizer": "adamw8bit", "timestep_sampling": "shift",
        },
        workflows=[
            Workflow("t2i"),
            Workflow("edit", images={"label": "Reference images (edit)", "min": 1, "max": MAX_REFS,
                                     "help": "Scaled to the output's area."}),
        ],
        sliders=True,
        # 4B: 7.8 GB in bf16; 9B: 18 GB, so a 9B finetune streams some blocks from system RAM on a 24 GB card
        finetune={"learning_rate": 1e-5, "optimizer": "adafactor", "keep_last": 2},
        sources={"label": "Source dataset", "noun": "source image", "max": MAX_REFS, "variants": None,
                 "without": "train text-to-image",
                 "help": "For an edit LoRA: a dataset of the images before the edit, paired with this dataset's "
                         "images (the results) by file name: photo.png pairs with photo.jpg. Each result's caption "
                         "is then the edit instruction. More source datasets add more reference images to each "
                         "edit. With none, or for an image without a source, it trains text-to-image, its caption a "
                         "prompt."},
    )
    vae_fp32 = True  # the FLUX.2 VAE runs in fp32 (``flux2_vae``)
    bucket_divisibility = TOKEN_PIXELS
    caption_tokens = MAX_TEXT_TOKENS - TEMPLATE_TOKENS
    # the long list of the two (double_blocks are 5 or 8 blocks): offloading streams these
    block_list = "single_blocks"
    lora_targets = ["double_blocks.*.img_attn.qkv", "double_blocks.*.img_attn.proj",
                    "double_blocks.*.txt_attn.qkv", "double_blocks.*.txt_attn.proj",
                    "double_blocks.*.img_mlp.[02]", "double_blocks.*.txt_mlp.[02]",
                    "single_blocks.*.linear1", "single_blocks.*.linear2"]
    # measured on Klein 9B (fp8), the larger: 4B's per-token costs are about 3/4 of these. The double blocks aren't
    # in block_list, so without checkpointing their activations (192 KB a token each on 9B) are spread over the
    # single blocks in ``full`` (215 KB measured + 8/24 of 192 KB); with it they're part of ``work``.
    memory = MemoryProfile(saved=8_600, full=280_000, work=436_000, infer=326_000,
                           vae_encode=(0.1e9, 2_580), vae_decode=(0.1e9, 3_860), text_tokens=MAX_TEXT_TOKENS)
    # the text encoder's layers past the last one read, its final norm and LM head are never loaded
    text_encoder_skip = ("lm_head.", "model.norm.", *(f"layers.{i}." for i in range(max(OUTPUT_LAYERS), 64)))
    lora_metadata = {"modelspec.architecture": "flux-2-klein/lora", "dainamo.family": "flux2_klein"}
    signature_keys = {
        "diffusion_model": ["double_stream_modulation_img.lin.weight", "single_stream_modulation.lin.weight",
                            "txt_in.weight"],
        "text_encoder": ["layers.35.mlp.down_proj.weight", "layers.0.self_attn.q_norm.weight", "embed_tokens.weight"],
        "vae": ["bn.running_mean", "encoder.conv_in.weight"],
    }
    signature_shapes = {
        "vae": {"bn.running_mean": (TOKEN_CHANNELS,)},  # FLUX.2's 32 channels packed 2x2
    }
    downloads = [
        Download("diffusion_model", COMFY_4B, "split_files/diffusion_models/flux-2-klein-base-4b.safetensors",
                 7_751_105_712, "Base 4B · bf16", COMFY_4B_REV, folder="diffusion_models", variant="base",
                 recommended=True, sha256="9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92"),
        Download("diffusion_model", "black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors",
                 4_089_498_488, "Base 4B · fp8", "103db268c10d4d3921101b46057671f9ac460da6",
                 folder="diffusion_models", variant="base",
                 sha256="44bab3a86fe98b85d21dd2a4729ebdc3ae51fb8a39f76e457e18c724219e6840"),
        Download("diffusion_model", "black-forest-labs/FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors",
                 18_157_185_168, "Base 9B · bf16 · non-commercial", "32773329fbe7e81a90ef971740e8ba4b0364ecf3",
                 folder="diffusion_models", variant="base", gated=True, note=NON_COMMERCIAL,
                 sha256="4a54fad7f5f741b99eee217198daac20b8d8e515e2a1f5b064fd51cf074f95bd"),
        Download("diffusion_model", "black-forest-labs/FLUX.2-klein-base-9b-fp8", "flux-2-klein-base-9b-fp8.safetensors",
                 9_567_278_472, "Base 9B · fp8 · non-commercial", "9ecf2143d71542449960c5584340269c6d401449",
                 folder="diffusion_models", variant="base", gated=True, note=NON_COMMERCIAL,
                 sha256="a9f5028c24a7a96f4f45beb883aad287d9bccc246227a6803edc898ddda42cf4"),
        Download("diffusion_model", COMFY_4B, "split_files/diffusion_models/flux-2-klein-4b.safetensors",
                 7_751_105_712, "Distilled 4B · bf16", COMFY_4B_REV, folder="diffusion_models", variant="distilled",
                 recommended=True, sha256="ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343"),
        Download("diffusion_model", "black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors",
                 4_070_624_520, "Distilled 4B · fp8", "5b4408e59397a4a37ccb46afe426d8ed86379441",
                 folder="diffusion_models", variant="distilled",
                 sha256="97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6"),
        Download("diffusion_model", "black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors",
                 18_157_185_168, "Distilled 9B · bf16 · non-commercial", "92196c8e11f7b6cf2b7493e037d8c5345c559216",
                 folder="diffusion_models", variant="distilled", gated=True, note=NON_COMMERCIAL,
                 sha256="0975d6b77b5f510b99547d6724a208e36527df654e8f6134f59ece3f9f30da58"),
        Download("diffusion_model", "black-forest-labs/FLUX.2-klein-9b-fp8", "flux-2-klein-9b-fp8.safetensors",
                 9_433_061_528, "Distilled 9B · fp8 · non-commercial", "902d9d510b51533e07729f19211414a3648b77d2",
                 folder="diffusion_models", variant="distilled", gated=True, note=NON_COMMERCIAL,
                 sha256="865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee"),
        Download("text_encoder", COMFY_4B, "split_files/text_encoders/qwen_3_4b.safetensors", 8_044_982_048,
                 "Qwen3-4B, for the 4B · bf16", COMFY_4B_REV, folder="text_encoders", recommended=True,
                 sha256="6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a"),
        Download("text_encoder", COMFY_9B, "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", 8_664_848_742,
                 "Qwen3-8B, for the 9B · fp8 mixed", COMFY_9B_REV, folder="text_encoders",
                 sha256="abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6"),
        Download("text_encoder", COMFY_9B, "split_files/text_encoders/qwen_3_8b.safetensors", 16_381_517_176,
                 "Qwen3-8B, for the 9B · bf16", COMFY_9B_REV, folder="text_encoders",
                 sha256="f0ff9239d56269ca1d05e5f86da6a79fac111af464955681f11c7ab0ec5ef6c1"),
        Download("vae", COMFY_4B, "split_files/vae/flux2-vae.safetensors", 336_211_292, "bf16", COMFY_4B_REV,
                 folder="vae", recommended=True,
                 sha256="868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"),
    ]

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None

    # --- profile ------------------------------------------------------------
    @property
    def distilled(self) -> bool:
        return str(self.profile_value(self.profile, "variant", "base")) == "distilled"

    def sample_steps(self) -> int:
        v = self.profile.get("sample_steps")
        return int(v) if v not in (None, "") else (DISTILLED_STEPS if self.distilled else BASE_STEPS)

    def cfg_scale(self, profile_cfg) -> float:
        if profile_cfg in (None, ""):
            return 1.0 if self.distilled else BASE_CFG
        return float(profile_cfg)

    @classmethod
    def matches(cls, field: str, tensors) -> bool:
        if not super().matches(field, tensors):
            return False
        if field == "text_encoder":  # Qwen3-4B or -8B, not Qwen3-VL (whose language model has the same shapes)
            return text_encoder_size(tensors) is not None and not any("visual." in k for k in tensors)
        if field == "diffusion_model":  # Klein's widths, not FLUX.2 [dev]'s
            return dit_size(tensors) is not None
        return True

    @classmethod
    def validate_profile(cls, profile: dict) -> list[dict]:
        te = os.path.expanduser((profile.get("text_encoder") or "").strip())
        sharded = bool(te) and comfy_io.is_sharded(te)
        out = super().validate_profile({**profile, "text_encoder": ""} if sharded else profile)
        if sharded:  # a Hugging Face folder: the base check reads single files only
            out = [r for r in out if r["field"] != "text_encoder"]
            out.append(cls._check_folder(te))
        # the 4B takes Qwen3-4B and the 9B Qwen3-8B: their widths must agree
        bad = {r["field"] for r in out if r["level"] == "error"}
        dm = os.path.expanduser((profile.get("diffusion_model") or "").strip())
        if te and dm and not bad & {"text_encoder", "diffusion_model"} and os.path.exists(dm) and os.path.exists(te):
            try:
                dit, enc = dit_size(header(dm)), text_encoder_size(header(te))
            except Exception:
                dit = enc = None
            if dit and enc and TEXT_FOR_DIT[dit] != enc:
                out = [r for r in out if r["field"] != "text_encoder"]
                out.append({"field": "text_encoder", "level": "error",
                            "message": f"This is Qwen3-{enc}; the {dit} diffusion model takes Qwen3-{TEXT_FOR_DIT[dit]}"})
        name = os.path.basename(dm).lower()
        variant = str(profile.get("variant") or "base")
        if "klein" in name and ("base" in name) != (variant == "base"):
            looks = "Base" if "base" in name else "distilled"
            out.append({"field": "variant", "level": "warning",
                        "message": f"The diffusion model's name says {looks}, but the variant is {variant}: samples "
                                   "would use the other's steps and CFG"})
        return out

    @classmethod
    def _check_folder(cls, path: str) -> dict:
        try:
            keys = header(path)
        except Exception as e:
            return {"field": "text_encoder", "level": "error", "message": f"Not a readable Hugging Face folder: {e}"}
        if not cls.matches("text_encoder", keys):
            return {"field": "text_encoder", "level": "error",
                    "message": "Wrong files? Expected Qwen3-4B for the 4B or Qwen3-8B for the 9B"}
        labels, unsupported = comfy_io.describe_quant(path)
        if unsupported:
            return {"field": "text_encoder", "level": "error",
                    "message": f"Uses quantization Dainamo can't read yet: {', '.join(sorted(unsupported))}"}
        shards = len(comfy_io.shard_files(path))
        return {"field": "text_encoder", "level": "ok",
                "message": f"{len(keys)} tensors in {shards} file{'s' if shards > 1 else ''}"
                           + (f" · {', '.join(sorted(labels))}" if labels else "")}

    # --- text -------------------------------------------------------------
    def cache_inputs(self, kind: str, settings: dict) -> dict:
        inputs = super().cache_inputs(kind, settings)
        te = inputs["files"][0] if kind == "text" else None
        if te and str(te).endswith(".index.json"):  # an index's shards beside it are what encode the text
            inputs["files"] = [os.path.dirname(os.path.expanduser(te))]
        return inputs

    def load_text_encoder(self):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

        path = self.profile["text_encoder"]
        self.log(f"Loading text encoder {path}")
        keys = header(path)
        size = text_encoder_size(keys)
        if size is None or any("visual." in k for k in keys):
            raise ValueError(f"{path} isn't Qwen3-4B or Qwen3-8B")
        try:
            dit = dit_size(header(self.profile["diffusion_model"]))
        except Exception:  # (the diffusion model's own load says what's wrong with it)
            dit = None
        if dit and TEXT_FOR_DIT[dit] != size:
            raise ValueError(f"The text encoder is Qwen3-{size}, but the {dit} diffusion model takes "
                             f"Qwen3-{TEXT_FOR_DIT[dit]}")
        tokenizer, config = qwen3_tokenizer_and_config(size)
        cfg = type(config).from_dict(config.to_dict())  # (the cached config stays whole)
        cfg.num_hidden_layers = max(OUTPUT_LAYERS)
        cfg.layer_types = list(config.layer_types[: cfg.num_hidden_layers])
        cfg._attn_implementation = "sdpa"
        prefix = "model." if any(k.startswith("model.layers.") for k in keys) else ""
        skip = ("lm_head.", "norm.", *(f"layers.{i}." for i in range(cfg.num_hidden_layers, config.num_hidden_layers)))
        loaded = comfy_io.load_state_dict(path, strip_prefix=prefix or None, only_prefixed=bool(prefix),
                                          skip_prefixes=skip, device=self.text_encoder_device)
        with torch.device("meta"):
            model = Qwen3Model(cfg)
        model.norm = torch.nn.Identity()  # the last layer read is mid-network: no final norm
        materialize(model, loaded, self.dtype, keep_fp8=True, device=None)
        del loaded  # else its tensors stay in RAM beside the streamed copies
        model.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device=self.device)
        assert_materialized(model)
        self.stream_text_encoder(model).eval()
        self.tokenizer, self.text_encoder = tokenizer, model

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """-> (512, 3 x hidden) bf16 on the CPU: every token of the padded prompt."""
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        enc = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                             max_length=MAX_TEXT_TOKENS)
        out = self.text_encoder(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device),
                                output_hidden_states=True, use_cache=False)
        hidden = torch.cat([out.hidden_states[i] for i in OUTPUT_LAYERS], dim=-1)
        return hidden[0].to("cpu", torch.bfloat16)

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

    def _references(self, paths: list[str], target_pixels: int, flip: bool = False) -> list[torch.Tensor]:
        """Reference images' latents (128, h, w), each at the target's area."""
        return [self.encode_image(prepare_reference(p, target_pixels, flip)) for p in paths[:MAX_REFS]]

    @torch.no_grad()
    def encode_item(self, item, settings: dict) -> dict:
        payload = super().encode_item(item, settings)
        if item.refs:
            w, h = item.size
            payload["refs"] = self._references(item.refs, w * h, item.flip)
        return payload

    # --- transformer --------------------------------------------------------
    def build_transformer(self, shapes):
        config = Flux2KleinConfig.from_shapes(shapes)
        if config.context_in_dim not in DIT_SIZES:
            raise ValueError(f"Not a FLUX.2 Klein 4B or 9B diffusion model (text width {config.context_in_dim})")
        with torch.device("meta"):
            return Flux2KleinDiT(config)

    @classmethod
    def sequence_tokens(cls, width: int, height: int, frames: int = 1) -> int:
        return (width // TOKEN_PIXELS) * (height // TOKEN_PIXELS) + MAX_TEXT_TOKENS

    def sample_timesteps(self, batch: int, latent_hw, settings: dict) -> torch.Tensor:
        mode = settings.get("timestep_sampling", "shift")
        if mode == "uniform":
            return torch.rand(batch).clamp(1e-3, 1.0)
        u = torch.sigmoid(torch.randn(batch)) if mode == "logit_normal" else torch.rand(batch)
        return time_shift(mu_for(latent_hw[0] * latent_hw[1]), u.clamp(1e-4, 1 - 1e-4))

    def _forward(self, model, noisy: torch.Tensor, t: torch.Tensor, text: list[torch.Tensor],
                 refs: list[list[torch.Tensor]]) -> torch.Tensor:
        """Velocity (B, 128, h, w) of noisy latents (B, 128, h, w) under per-sample text conditions and reference
        latents (samples in one forward share the references' sizes)."""
        b, c, h, w = noisy.shape
        d = self.device
        shapes = [tuple(r.shape[-2:]) for r in refs[0]]
        if any([tuple(r.shape[-2:]) for r in rs] != shapes for rs in refs):
            raise ValueError("samples in one forward must share their reference image sizes")
        txt = torch.stack([x.to(d, self.dtype) for x in text])
        ids = image_ids(h, w, device=d).expand(b, -1, -1)
        ref = ref_ids = None
        if shapes:
            ref = torch.stack([torch.cat([tokens(r[None].to(d, self.dtype))[0] for r in rs]) for rs in refs])
            ref_ids = torch.cat([image_ids(rh, rw, REF_TIME_SCALE * (i + 1), d) for i, (rh, rw) in enumerate(shapes)])
            ref_ids = ref_ids.expand(b, -1, -1)
        out = model(tokens(noisy.to(self.dtype)), ids, txt, text_ids(txt.shape[1], d).expand(b, -1, -1),
                    t.to(d, torch.float32), ref, ref_ids)
        return out.transpose(1, 2).reshape(b, c, h, w)

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
        refs = [list(p.get("refs") or []) if isinstance(p, dict) else [] for p in payloads]
        # one forward per reference layout (samples in a forward must share it)
        groups: dict[tuple, list[int]] = {}
        for i, rs in enumerate(refs):
            groups.setdefault(tuple(tuple(r.shape) for r in rs), []).append(i)
        pred = torch.empty_like(lat)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for idx in groups.values():
                pred[idx] = self._forward(model, noisy[idx], t[idx], [conds[i]["text"] for i in idx],
                                          [refs[i] for i in idx]).float()
        return self.weighted_loss(pred, self.loss_target(lat, noise), t, settings, payloads, sigma is not None)

    # --- sampling -----------------------------------------------------------
    def default_shift(self, width, height):
        return math.exp(empirical_mu((height // TOKEN_PIXELS) * (width // TOKEN_PIXELS), self.sample_steps()))

    def default_sigmas(self, steps, width, height):
        return sampling_sigmas(int(steps), (height // TOKEN_PIXELS) * (width // TOKEN_PIXELS))

    def prepare_sampling(self, model, cond, uncond, width, height, cfg, seed, images=None):
        h, w = height // TOKEN_PIXELS, width // TOKEN_PIXELS
        refs = []
        if images:
            if self.vae is None:
                raise RuntimeError("the VAE must be loaded to encode reference images")
            refs = self._references(list(images), width * height)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn(1, TOKEN_CHANNELS, h, w, generator=g).to(self.device, torch.float32)
        use_cfg = cfg is not None and float(cfg) != 1 and float(cfg) > 0 and uncond is not None

        def velocity(x, s):
            t = torch.full((1,), float(s), device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self._forward(model, x, t, [cond], [refs]).float()
                if use_cfg:  # the reference's CFG keeps the references in both branches
                    vu = self._forward(model, x, t, [uncond], [refs]).float()
                    v = vu + float(cfg) * (v - vu)
            return v
        return x, velocity

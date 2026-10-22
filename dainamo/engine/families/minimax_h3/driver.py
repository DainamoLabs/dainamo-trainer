"""MiniMax H3 driver: joint video + audio LoRA training for the fl2va and ref2va models.

Loads ComfyUI-format files directly:
  * diffusion model: the pruned int8-ConvRot DiT (fl2va or ref2va), ~19B parameters. By default 40%
    of its blocks stay in system RAM and are streamed in per forward (the job's offload setting;
    see ``engine.offload``)
  * text encoder: Qwen3-VL-32B, 50 decoder layers in NVFP4 (AWQ) + int8 embeddings + bf16 vision
    tower; conditioning is the unnormalised output of layer 49
  * video VAE (int8-ConvRot decoder) and audio VAE (fp32)

Training follows the released recipe as implemented in ai-toolkit (MIT): packed
[text | conditions | audio | video] sequences, video sigma on the shift-12 schedule with the
audio sigma remapped to shift 3, the model consumes t = 1 - sigma and predicts clean - noise.
Clips with a soundtrack train audio jointly; silent clips and images carry noised silence with
no audio loss. fl2va trains with the clip's own first frame as a keyframe on a share of steps
(and plain text otherwise); ref2va trains with reference images from ``<dataset>/_refs``.
Conditioning images also enter the Qwen3-VL presentation (``<Picture i>:`` + vision block).

It also makes RefMods (``engine.refmod``), references encoded once and saved; ref2va sampling takes them as
references beside or instead of images (the text encoder doesn't see visual ones, as with the ComfyUI pack's Apply).
ref2va sampling also takes audio references (sound files, and audio RefMods) as the released pipeline does: clean
audio rows ahead of the generated audio, each cut to the generated length and labelled ``<Audio j>:`` in the
prompt, beside at least one image or visual RefMod.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from ... import comfy_io
from ... import dataset as ds
from ... import refmod
from ...fp8_linear import assert_materialized, materialize
from ...hf_assets import minimax_assets
from ..base import FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, Workflow
from . import packing as P
from .audio_vae import MiniMaxH3AudioVAE
from .text_encoder import TEXT_ENCODER_LAYER, AudioRef, encode_minimax_h3_prompt, trim_caption_tokens
from .transformer import MiniMaxH3Transformer
from .vae import MiniMaxH3VideoVAE, RotaryEmbedding3d, IMAGENET_MEAN, IMAGENET_STD

MAX_TEXT_TOKENS = 512
# the checkpoint's float32 islands (and the fp16-stored AdaLN projections, computed in fp32)
FP32_PREFIXES = MiniMaxH3Transformer.FP32_KEY_PREFIXES + ("adaln_t_table",)


def dit_dtype_for(key: str):
    if key.startswith(FP32_PREFIXES) or ".adaln_proj." in key:
        return torch.float32
    return None


class _TextEncoderView:
    """What ``encode_minimax_h3_prompt`` expects of a transformers Qwen3-VL wrapper."""

    def __init__(self, model, config, device, dtype):
        self.model, self.config, self.device, self.dtype = model, config, device, dtype


def _to_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ds.to_rgb(ImageOps.exif_transpose(im))


class MinimaxH3Driver(FamilyDriver):
    info = FamilyInfo(
        id="minimax_h3",
        label="MiniMax H3 (video + audio)",
        media=["image", "video"],
        profile_fields=[
            ProfileField("variant", "Variant", kind="select", options=["fl2va", "ref2va"], default="fl2va",
                         help="fl2va: text or first frame to video. ref2va: reference images to video. "
                              "Must match the diffusion model file."),
            ProfileField("diffusion_model", "Diffusion model", hint="MiniMax H3 (fl2va or ref2va)",
                         help="The fl2va or ref2va DiT, matching the variant above (ComfyUI int8_convrot)."),
            ProfileField("text_encoder", "Text encoder (CLIP)", hint="MiniMax H3's Qwen3-VL-32B",
                         help="The Qwen3-VL-32B text encoder released for MiniMax H3 (cut to 50 layers), any "
                              "precision, e.g. nvfp4. A full 64-layer Qwen3-VL-32B won't load."),
            ProfileField("vae", "Video VAE", hint="MiniMax H3 video VAE",
                         help="The video VAE from the MiniMax H3 release."),
            ProfileField("audio_vae", "Audio VAE", hint="MiniMax H3 audio VAE",
                         help="The audio VAE from the MiniMax H3 release."),
            ProfileField("sample_steps", "Sample steps", kind="number", default=20,
                         help="The release uses 28. No CFG: the model is guidance-distilled."),
        ],
        training_defaults={
            "resolution": 384, "num_frames": 39, "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 1000,
            "batch_size": 1, "offload": 40, "optimizer": "adamw8bit", "timestep_sampling": "shift", "i2v_probability": 0.5,
            "train_audio": True, "audio_loss_weight": 1.0, "sample_frames": 39, "sample_width": 640,
            "sample_height": 384, "bleed_width": 384, "bleed_height": 384,
        },
        workflows=[
            Workflow("t2v"),
            Workflow("i2v", images={"label": "First frame", "min": 1, "max": 1, "help": ""}, variants=["fl2va"]),
            Workflow("r2v", images={"label": "Reference images", "min": 1, "max": 4,
                                    "help": "Reference images per clip go in a _refs folder inside the dataset, "
                                            "named after the clip: _refs/clip.png, _refs/clip_1.png, …"},
                     variants=["ref2va"], refmods=True,
                     audio={"label": "Reference audio", "max": 3,
                            "help": "A voice, music or a sound for the video to take after, beside at least one "
                                    "reference image or visual RefMod. Each is cut to the video's length."}),
        ],
        timestep_modes=[
            ("shift", "Shift 12 (released schedule)"),
            ("logit_normal", "Logit-normal + shift 12"),
            ("uniform", "Uniform"),
        ],
        frame_rule="17n + 5 (22, 39, 56, …)",
        refmods=True,
    )
    bucket_divisibility = 32  # 16x VAE, 2x2 patch
    caption_tokens = MAX_TEXT_TOKENS
    snap_frames = staticmethod(P.align_num_frames_down)
    fps = P.FPS
    lora_targets = ["blocks.*.attn.qkv_proj", "blocks.*.attn.out_proj", "blocks.*.mlp.fc1", "blocks.*.mlp.fc2"]
    memory = MemoryProfile(saved=10_800, full=447_000, work=585_000, infer=339_000,
                           vae_encode=(3.6e9, 33), vae_decode=(0.33e9, 31),
                           text_work=2.0)  # the NVFP4 text encoder's weights are expanded per matmul
    text_encoder_fp32 = ("visual.",)
    lora_metadata = {"modelspec.architecture": "minimax-h3/lora", "dainamo.family": "minimax_h3"}
    signature_keys = {
        "diffusion_model": ["blocks.0.attn.qkv_proj.weight", "audio_patch_proj.weight"],
        "text_encoder": ["layers.49.mlp.down_proj.weight", "visual.merger.linear_fc1.weight"],
        "vae": ["decoder.x_embedder.weight", "encoder.conv_in.weight"],
        "audio_vae": ["dec_in_proj.weight", "pre_block.attn.qkv.weight"],
    }

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.variant = str(profile.get("variant") or "fl2va")
        self.tokenizer = self.processor = self.text_encoder = None
        self.video_vae = self.audio_vae = None

    # --- text -------------------------------------------------------------
    def load_text_encoder(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel, Qwen3VLTextRotaryEmbedding, Qwen3VLVisionRotaryEmbedding,
        )

        tokenizer, processor, config = minimax_assets()
        config.text_config.num_hidden_layers = TEXT_ENCODER_LAYER  # only hidden_states[50] is used
        config.tie_word_embeddings = False
        for c in (config, config.text_config, config.vision_config):
            c._attn_implementation = "sdpa"
        path = self.profile["text_encoder"]
        self.log(f"Loading text encoder {path}")
        loaded = comfy_io.load_state_dict(path, skip_prefixes=("lm_head.",), device=self.text_encoder_device)

        def rename(k):
            return "language_model." + k[len("model."):] if k.startswith("model.") else k

        loaded.tensors = {rename(k): v for k, v in loaded.tensors.items()}
        loaded.quant = {rename(k): v for k, v in loaded.quant.items()}
        with torch.device("meta"):
            model = Qwen3VLModel(config)
        model.language_model.norm = nn.Identity()  # hidden_states[-1] stays the unnormalised layer-49 output
        # the vision tower runs in fp32: in bf16 its output drifts ~5% from the fp32 result (and differs
        # between attention kernels by about as much); it's small and only sees conditioning images
        materialize(model, loaded, self.dtype, keep_fp8=True, device=None,
                    dtype_for=lambda k: torch.float32 if k.startswith("visual.") else None)
        del loaded  # else its tensors stay in RAM beside the streamed copies
        model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config.text_config, device=self.device)
        model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.vision_config).to(self.device)
        assert_materialized(model)
        self.stream_text_encoder(model).eval()
        self.tokenizer, self.processor = tokenizer, processor
        self.text_encoder = _TextEncoderView(model, config, self.device, self.dtype)

    @torch.no_grad()
    def encode_condition(self, prompt: str, images: list | None):
        embeds, tags = encode_minimax_h3_prompt(self.text_encoder, self.tokenizer, self.processor, prompt.strip(),
                                                keyframes=images or None, device=self.device, dtype=self.dtype,
                                                max_length=MAX_TEXT_TOKENS)
        return {"embeds": embeds.to("cpu", torch.bfloat16), "tags": tags.cpu()}

    def encode_prompt(self, prompt: str):
        return self.encode_condition(prompt, None)

    def _prepared_refs(self, path: str, size: tuple[int, int]) -> list[Image.Image]:
        w, h = size
        return [P.prepare_reference_image(_to_pil(f), h, w) for f in ds.reference_images(path)]

    def item_conditions(self, item, settings: dict):
        conds = {"text": (item.caption, None)}
        if self.variant == "fl2va" and item.is_video and float(settings.get("i2v_probability", 0.5)) > 0:
            conds["i2v"] = (item.caption, [ds.first_frame(item.path, item.size, self.fps, item.flip)])
        if self.variant == "ref2va":
            refs = self._prepared_refs(item.path, item.size)
            if refs:
                conds["ref"] = (item.caption, refs)
        return conds

    def _sample_images(self, spec: dict) -> list[Image.Image]:
        paths = spec.get("images") or []
        if not paths:
            return []
        if self.variant == "fl2va":  # one first-frame keyframe, stretched onto the canvas
            return [P.prepare_keyframe_image(_to_pil(paths[0]), spec["height"], spec["width"], stretch=True)]
        return [P.prepare_reference_image(_to_pil(p), spec["height"], spec["width"]) for p in paths]

    def _audio_refs(self, spec: dict) -> int:
        """How many audio references a ref2va render takes (the audio RefMods' members and the sound files)."""
        if self.variant != "ref2va" or int(spec.get("frames") or 1) <= 1:
            return 0
        n = len(spec.get("audio") or [])
        for r in spec.get("refmods") or []:
            n += refmod.member_kinds(refmod.read_meta(r["path"]) or {}).count("audio")
        return n

    def sample_condition(self, spec: dict):
        # each audio reference's "<Audio j>: " label, after the pictures' (the order they pack in)
        return spec["prompt"], (self._sample_images(spec) + [AudioRef()] * self._audio_refs(spec)) or None

    # --- vae --------------------------------------------------------------
    def cache_inputs(self, kind: str, settings: dict) -> dict:
        if kind == "text":
            return super().cache_inputs(kind, settings)
        # the variant decides the keyframes (fl2va) or reference images (ref2va) encoded with each item
        return {"files": [self.profile["vae"], self.profile.get("audio_vae")], "variant": self.variant,
                "train_audio": bool(settings.get("train_audio", True))}

    def load_vae(self, audio: bool = True, video: bool = True):
        """The video VAE unless ``video`` is off, and the audio VAE unless ``audio`` is (a RefMod needs one)."""
        if video:
            self._load_video_vae()
        if audio:
            apath = self.profile["audio_vae"]
            self.log(f"Loading audio VAE {apath}")
            sd = comfy_io.dequantize_all(comfy_io.load_state_dict(apath), torch.float32)
            self.audio_vae = MiniMaxH3AudioVAE.load_from_state_dict(sd).to(self.device)

    def _load_video_vae(self):
        path = self.profile["vae"]
        self.log(f"Loading video VAE {path}")
        loaded = comfy_io.load_state_dict(path, device=self.device)
        stats = {k: loaded.tensors.pop(k).float() for k in ("latents_mean", "latents_std") if k in loaded.tensors}
        with torch.device("meta"):
            vae = MiniMaxH3VideoVAE()
        materialize(vae, loaded, torch.float32, keep_fp8=True, device=self.device)
        # non-persistent buffers built on meta: re-derive them
        vae.latents_mean = stats.get("latents_mean", torch.zeros(24)).to(self.device)
        vae.latents_std = stats.get("latents_std", torch.ones(24)).to(self.device)
        vae.pixel_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1, 1)
        vae.pixel_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1, 1)
        dec = vae.decoder
        head_dim = dec.transformer_blocks[0].attn.head_dim
        dec.rope = RotaryEmbedding3d(int(head_dim * 0.75), theta=100.0).to(self.device)
        assert_materialized(vae)
        self.video_vae = vae.eval().requires_grad_(False)

    def unload_vae(self):
        self.video_vae = self.audio_vae = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_keyframe(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, 3, 1, H, W) in [-1, 1] -> (B, 24, 1, h, w): the released conditioning recipe (posterior
        sample with seed 42, rounded through fp16 before normalisation)."""
        g = torch.Generator(device="cpu").manual_seed(P.KEYFRAME_ENCODE_SEED)
        return self.video_vae.encode(pixels.to(self.device), sample=True, generator=g, fp16_round=True).float()

    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """(1, 3, T, H, W) in [-1, 1] (an image, or a 17n+5-frame clip) -> (1, 24, t, h, w): a RefMod's reference,
        encoded as conditioning images are."""
        return self._encode_keyframe(pixels)

    @torch.no_grad()
    def encode_item(self, item, settings: dict) -> dict:
        if item.is_video:
            px = ds.read_video_frames(item.path, item.size, item.frames, self.fps, item.flip).to(self.device)
            payload = {"video": self.video_vae.encode(px, sample=True)[0].float().cpu(), "frames": item.frames}
            if self.variant == "fl2va":
                payload["first"] = self._encode_keyframe(px[:, :, :1])[0].cpu()
            if settings.get("train_audio", True) and item.has_audio:
                wav = ds.read_audio(item.path, item.frames / self.fps, P.AUDIO_SAMPLE_RATE, P.AUDIO_CHANNELS)
                z = self.audio_vae.encode(wav.unsqueeze(1).to(self.device))  # (2, 32, T): each channel separately
                payload["audio"] = P.pack_audio_latents(z.unsqueeze(0))[0].float().cpu()  # (2T, 32)
        else:
            px = ds.load_and_crop(item.path, item.size, item.flip).to(self.device).unsqueeze(2)
            payload = {"video": self.video_vae.encode(px, sample=True)[0].float().cpu(), "frames": 1}
        if self.variant == "ref2va":
            refs = self._prepared_refs(item.path, item.size)
            payload["refs"] = [self._encode_keyframe(self._pil_to_px(r))[0].cpu() for r in refs]
        return payload

    @staticmethod
    def _pil_to_px(img: Image.Image) -> torch.Tensor:
        arr = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0)
        return arr.permute(2, 0, 1)[None, :, None]  # (1, 3, 1, H, W)

    # --- transformer --------------------------------------------------------
    def build_transformer(self, shapes):
        with torch.device("meta"):
            return MiniMaxH3Transformer(MiniMaxH3Transformer.config_from_shapes(shapes))

    def transformer_dtype_for(self, key):
        return dit_dtype_for(key)

    @classmethod
    def sequence_tokens(cls, width, height, frames=1):
        """Video rows, audio rows (two channels) and text, plus one latent frame of keyframe or
        reference rows (the larger fl2va / ref2va case)."""
        per_frame = (width // 32) * (height // 32)
        if frames > 1:
            frames = P.align_num_frames_down(frames)
            rows = P.video_latent_num_frames(frames) * per_frame + P.audio_latent_num_frames(frames) * P.AUDIO_CHANNELS
        else:
            rows = per_frame
        return rows + per_frame + (cls.memory.text_tokens if cls.memory else 256)

    def sample_timesteps(self, batch: int, latent_hw, settings: dict) -> torch.Tensor:
        mode = settings.get("timestep_sampling", "shift")
        if mode == "uniform":
            return torch.rand(batch).clamp(1e-3, 1.0)
        u = torch.sigmoid(torch.randn(batch)) if mode == "logit_normal" else torch.rand(batch)
        return P.shift_sigma(u, P.VIDEO_SIGMA_SHIFT).clamp(1e-3, 1.0)

    def _text_batch(self, conds: list[dict]):
        embeds, tags = [], []
        for c in conds:
            e, t = trim_caption_tokens(c["embeds"], c["tags"], MAX_TEXT_TOKENS)
            embeds.append(e)
            tags.append(t)
        return embeds, tags

    def _forward(self, model, video_rows, audio_rows, embeds, tags, latent_shape, a_lat, row_t_fn,
                 anchors=(), ref_blocks=()):
        """Pack, run the DiT, and return the velocity rows of the target video and audio. ``audio_rows`` starts
        with any reference audio rows (``(0, 0, 0, a)`` and soundtrack blocks), which ride clean (t = 1)."""
        t_lat, h_lat, w_lat = latent_shape
        layouts = [P.build_packed_sequence(text_token_tags=t.to(torch.long), num_latent_frames=t_lat,
                                           latent_height=h_lat, latent_width=w_lat, num_audio_latents=a_lat,
                                           keyframe_anchors=anchors, ref_blocks=ref_blocks) for t in tags]
        position_ids, token_tags, video_idx, audio_idx, text_idx, _ = P.pad_layouts_to_batch(layouts)
        num_cond = layouts[0].num_condition_video_rows
        num_cond_audio = layouts[0].num_condition_audio_rows
        b = len(embeds)
        text_batch = torch.zeros(b, int(text_idx.shape[0]), embeds[0].shape[-1], device=self.device, dtype=self.dtype)
        for i, e in enumerate(embeds):
            text_batch[i, : e.shape[0]] = e.to(self.device, self.dtype)
        row_t = row_t_fn(token_tags.shape[1], video_idx, audio_idx, num_cond)
        if num_cond_audio:
            row_t[:, audio_idx[:num_cond_audio].to(row_t.device)] = 1.0
        row_t = row_t.to(self.device)
        d = self.device
        video_pred, audio_pred = model(
            hidden_states=video_rows.to(self.dtype), audio_hidden_states=audio_rows.to(self.dtype),
            encoder_hidden_states=text_batch, row_timesteps=row_t, token_tags=token_tags.to(d),
            position_ids=position_ids.to(d), video_indices=video_idx.to(d), audio_indices=audio_idx.to(d),
            text_indices=text_idx.to(d))
        return video_pred[:, num_cond:], audio_pred[:, num_cond_audio:]

    def training_loss(self, model, payloads, conds, settings, rng=None, sigma=None, generator=None,
                      deterministic=False):
        d = self.device

        def randn(shape):
            return torch.randn(shape, generator=generator).to(d) if generator is not None else torch.randn(shape, device=d)

        b = len(payloads)
        x0 = torch.stack([p["video"] for p in payloads]).to(d, torch.float32)  # (B, 24, t, h, w)
        t_lat, h_lat, w_lat = x0.shape[2:]
        frames = int(payloads[0].get("frames", 1))
        a_lat = P.audio_latent_num_frames(frames)
        if sigma is not None:
            sv = torch.full((b,), float(sigma), device=d)
        else:
            sv = self.sample_timesteps(b, None, settings).to(d)
        sa = P.remap_sigma(sv, P.VIDEO_SIGMA_SHIFT, P.AUDIO_SIGMA_SHIFT)
        tv, ta = 1.0 - sv, 1.0 - sa

        noise = randn(x0.shape)
        sv5 = sv.view(-1, 1, 1, 1, 1)
        noisy = (1 - sv5) * x0 + sv5 * noise
        target_v = x0 - noise  # the model predicts clean - noise

        audio_target = None
        sa3 = sa.view(-1, 1, 1)
        if settings.get("train_audio", True) and frames > 1 and all("audio" in p for p in payloads):
            rows = a_lat * P.AUDIO_CHANNELS
            xa = torch.stack([F.pad(p["audio"][:rows], (0, 0, 0, max(0, rows - p["audio"].shape[0])))
                              for p in payloads]).to(d, torch.float32)
            na = randn(xa.shape)
            audio_rows = (1 - sa3) * xa + sa3 * na
            audio_target = xa - na
        else:
            audio_rows = sa3 * randn((b, a_lat * P.AUDIO_CHANNELS, 32))

        # conditioning: fl2va keyframe on a share of steps, ref2va references when the item has them
        anchors, ref_blocks, cond_rows, role = (), (), None, "text"
        if (self.variant == "fl2va" and frames > 1 and not deterministic and rng is not None
                and all("first" in p for p in payloads) and all("i2v" in c for c in conds)
                and rng.random() < float(settings.get("i2v_probability", 0.5))):
            first = torch.stack([p["first"] for p in payloads]).to(d, torch.float32)
            first = P.KEYFRAME_NOISE_AUG_T * first + (1 - P.KEYFRAME_NOISE_AUG_T) * randn(first.shape)
            cond_rows, anchors, role = P.patchify_video_latents(first), ("first",), "i2v"
        elif (self.variant == "ref2va" and all(p.get("refs") for p in payloads)
              and all("ref" in c for c in conds)):
            parts, blocks = [], []
            for j in range(len(payloads[0]["refs"])):
                ref = torch.stack([p["refs"][j] for p in payloads]).to(d, torch.float32)
                ref = P.KEYFRAME_NOISE_AUG_T * ref + (1 - P.KEYFRAME_NOISE_AUG_T) * randn(ref.shape)
                parts.append(P.patchify_video_latents(ref))
                blocks.append((ref.shape[2], ref.shape[3], ref.shape[4], 0))
            cond_rows, ref_blocks, role = torch.cat(parts, dim=1), tuple(blocks), "ref"
        embeds, tags = self._text_batch([c.get(role, c["text"]) for c in conds])

        video_rows = P.patchify_video_latents(noisy)
        if cond_rows is not None:
            video_rows = torch.cat([cond_rows, video_rows], dim=1)

        def row_t_fn(seq_len, video_idx, audio_idx, num_cond):
            row_t = tv.view(-1, 1).expand(-1, seq_len).clone()
            row_t[:, audio_idx] = ta.view(-1, 1)
            if num_cond:
                row_t[:, video_idx[:num_cond]] = torch.clamp(tv, min=P.KEYFRAME_NOISE_AUG_T).view(-1, 1)
            return row_t

        video_pred, audio_pred = self._forward(model, video_rows, audio_rows, embeds, tags, (t_lat, h_lat, w_lat),
                                               a_lat, row_t_fn, anchors, ref_blocks)
        pred_v = P.unpatchify_video_tokens(video_pred.float(), t_lat, h_lat, w_lat)
        loss = F.mse_loss(pred_v, target_v)
        if audio_target is not None:
            loss = loss + float(settings.get("audio_loss_weight", 1.0)) * F.mse_loss(audio_pred.float(), audio_target)
        return loss

    # --- sampling -----------------------------------------------------------
    supports_samplers = False  # the released joint video + audio Euler loop

    @torch.no_grad()
    def sample(self, model, cond, uncond, width, height, steps, cfg, seed, frames=1, images=None, sampler="euler",
               scheduler="default", shift=None, callback=None, refmods=None, audio=None):
        """``refmods`` (ref2va): [{"path", "strength"}], each RefMod's references put in after the images, at
        their strength (``refmod.weaken``). ``audio`` (ref2va, a video): sound files, audio references after
        those. Audio references (these and audio RefMods) are cut to the generated length, and need ``cond``
        encoded with their labels (``sample_condition``)."""
        d = self.device
        g = torch.Generator(device="cpu").manual_seed(int(seed))

        def randn(shape):
            return torch.randn(shape, generator=g, dtype=torch.float32).to(d)

        if frames > 1:
            frames = P.align_num_frames_down(frames)
            t_lat = P.video_latent_num_frames(frames)
        else:
            frames, t_lat = 1, 1
        h_lat, w_lat = height // 16, width // 16
        a_lat = P.audio_latent_num_frames(frames)
        embeds, tags = self._text_batch([cond])
        pil = self._sample_images({"images": images, "width": width, "height": height}) if images else []

        # conditioning rows (draw order as released: condition noise, video, audio)
        anchors, ref_blocks, cond_rows = (), (), None
        parts, blocks, audio_parts = [], [], []
        max_audio = refmod.audio_latents(frames / self.fps)

        def add(lat):  # a reference's latent (1, 24, t, h, w), noise-augmented as released
            cond_noise = randn(lat.shape)
            parts.append(P.patchify_video_latents(P.KEYFRAME_NOISE_AUG_T * lat.to(d, torch.float32)
                                                  + (1 - P.KEYFRAME_NOISE_AUG_T) * cond_noise))
            blocks.append(tuple(int(x) for x in lat.shape[2:]))

        def add_audio(lat):  # an audio reference's (1, 32, 2, T), cut to the generated length: clean rows
            lat = lat[..., :max_audio]
            audio_parts.append(refmod.audio_rows(lat).to(d, torch.float32))
            blocks.append((0, 0, 0, int(lat.shape[-1])))

        if pil:
            if self.video_vae is None:
                raise RuntimeError("video VAE must be loaded to encode conditioning images")
            for img in pil:
                add(self._encode_keyframe(self._pil_to_px(img)))
        if refmods:
            if self.variant != "ref2va":
                raise ValueError("RefMods need the ref2va model")
            for r in refmods:
                strength = float(r.get("strength", 1.0))
                for ref in refmod.load(r["path"]):
                    if ref["kind"] != "audio":
                        add(refmod.weaken(ref["latent"], strength))
                    elif frames > 1:  # an image takes no sound
                        add_audio(refmod.weaken(ref["latent"][..., :max_audio], strength))
        if audio and frames > 1:
            if self.variant != "ref2va":
                raise ValueError("Audio references need the ref2va model")
            if self.audio_vae is None:
                raise RuntimeError("audio VAE must be loaded to encode reference audio")
            for path in audio:
                info = ds.probe_audio(path)
                wav = refmod.read_sound(path, min(frames / self.fps, info["duration"]), info)
                add_audio(refmod.encode_audio(self.audio_vae, wav, d))
        if audio_parts and not parts:
            raise ValueError("An audio reference needs an image or a visual RefMod beside it")
        if parts:
            cond_rows = torch.cat(parts, dim=1)
            if self.variant == "fl2va":
                anchors, cond_rows = ("first",), parts[0]
            else:
                ref_blocks = tuple(blocks)

        video_rows = P.patchify_video_latents(randn((1, 24, t_lat, h_lat, w_lat)))
        audio_rows = P.pack_audio_latents(randn((1, P.AUDIO_CHANNELS, 32, a_lat)))
        cond_audio = torch.cat(audio_parts, dim=0)[None] if audio_parts else None
        sigmas_v = P.build_sigma_schedule(int(steps), P.VIDEO_SIGMA_SHIFT).to(d)
        sigmas_a = P.remap_sigma(sigmas_v, P.VIDEO_SIGMA_SHIFT, P.AUDIO_SIGMA_SHIFT)
        for i in range(sigmas_v.shape[0] - 1):
            sv, sv_next, sa, sa_next = sigmas_v[i], sigmas_v[i + 1], sigmas_a[i], sigmas_a[i + 1]
            tv, ta = 1.0 - float(sv), 1.0 - float(sa)

            def row_t_fn(seq_len, video_idx, audio_idx, num_cond):
                row_t = torch.full((1, seq_len), tv, dtype=torch.float32)
                row_t[:, audio_idx.cpu()] = ta
                if num_cond:
                    row_t[:, video_idx[:num_cond].cpu()] = max(tv, P.KEYFRAME_NOISE_AUG_T)
                return row_t

            video_in = video_rows if cond_rows is None else torch.cat([cond_rows, video_rows], dim=1)
            audio_in = audio_rows if cond_audio is None else torch.cat([cond_audio, audio_rows], dim=1)
            v_video, v_audio = self._forward(model, video_in, audio_in, embeds, tags, (t_lat, h_lat, w_lat),
                                             a_lat, row_t_fn, anchors, ref_blocks)
            denoised_v = video_rows + sv * v_video.float()
            ratio_v = sv_next / sv
            video_rows = ratio_v * video_rows + (1.0 - ratio_v) * denoised_v
            denoised_a = audio_rows + sa * v_audio.float()
            ratio_a = sa_next / sa if float(sa) != 0.0 else 0.0
            audio_rows = ratio_a * audio_rows + (1.0 - ratio_a) * denoised_a
            if callback is not None:
                callback(i + 1, sigmas_v.shape[0] - 1)
        return {"video": P.unpatchify_video_tokens(video_rows, t_lat, h_lat, w_lat), "audio": audio_rows,
                "a_lat": a_lat, "frames": frames}

    @torch.no_grad()
    def to_media(self, latents) -> list:
        video = self.video_vae.decode(latents["video"].to(self.device))  # (1, 3, T, H, W) in [-1, 1]
        frames = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 3, 0).cpu().numpy()
        if latents["frames"] == 1:
            return [Image.fromarray(frames[0])]
        audio = P.unpack_audio_tokens(latents["audio"], latents["a_lat"])[0]  # (2, 32, A)
        wav = self.audio_vae.decode(audio.float().to(self.device))[:, 0].cpu().numpy()  # (2, samples)
        return [{"frames": frames, "audio": wav, "fps": self.fps, "sample_rate": P.AUDIO_SAMPLE_RATE}]

    def decode_latents(self, latents):
        raise NotImplementedError("MiniMax H3 decodes through to_media")

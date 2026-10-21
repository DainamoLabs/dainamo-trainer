"""LTX 2.5 driver: joint video + audio LoRA training, text-to-video and first-frame image-to-video.

Loads ComfyUI-format files directly:
  * diffusion model: the 22B audio-video DiT, dev or distilled, int8 convrot or bf16 (``model``: diffusers'
    LTX2VideoTransformer3DModel with the original key layout mapped onto it). By default 60% of its blocks
    stay in system RAM and are streamed in (the job's offload setting; see ``engine.offload``)
  * text encoder: Gemma 4 12B with LTX's per-modality text projections; its tokenizer is embedded in the
    file. The text connectors (16 small transformer blocks that ship in the diffusion-model file) run with
    the text encoder, so each caption is cached as the DiT's final video and audio text conditions
  * video VAE: the causal encoder with the diffusion decoder (or the classic conv VAE); audio VAE + vocoder

Training follows ai-toolkit's LTX-2 trainer (MIT): flow matching with the velocity target noise - x0 on the
video and audio latents at one shared sigma, shifted by the video token count as LTX's scheduler shifts
(mu 0.95 at 1024 tokens to 2.05 at 4096). On a share of clip steps the first latent frame is the clip's
own first frame, clean at timestep 0 and left out of the loss (image-to-video). Clips with a soundtrack
train audio jointly; silent clips and images get pure-noise audio at sigma 1 and no audio loss.
"""

from __future__ import annotations

import gc
import json
import math
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from ... import comfy_io
from ... import dataset as ds
from ...fp8_linear import assert_materialized, materialize
from ..base import FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, Workflow
from . import model as M

IMAGE_CRF = 18  # first-frame images are re-compressed like this, as LTX 2.5 was trained (diffusers' LTX2_5_IMAGE_CRF)
DECODE_SEED = 0  # the diffusion decoder samples noise; a fixed seed keeps a latent's decode the same


def _pack(video: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, T*H*W, C) tokens in frame, row, column order."""
    return video.flatten(2).transpose(1, 2)


def _unpack(tokens: torch.Tensor, shape) -> torch.Tensor:
    b, c, t, h, w = shape
    return tokens.transpose(1, 2).reshape(b, c, t, h, w)


def _crf(img: Image.Image, crf: int) -> Image.Image:
    """``img`` through H.264 at ``crf`` and back (even width and height), as LTX's conditioning images are."""
    w, h = img.width // 2 * 2, img.height // 2 * 2
    img = img.convert("RGB").crop((0, 0, w, h))
    enc = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                          "-f", "h264", "-"], input=img.tobytes(), capture_output=True, check=True, timeout=60)
    dec = subprocess.run(["ffmpeg", "-v", "error", "-f", "h264", "-i", "-", "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], input=enc.stdout, capture_output=True, check=True, timeout=60)
    return Image.frombytes("RGB", (w, h), dec.stdout[: w * h * 3])


class LTX25Driver(FamilyDriver):
    info = FamilyInfo(
        id="ltx2_5",
        label="LTX 2.5 (video + audio)",
        media=["image", "video"],
        profile_fields=[
            ProfileField("variant", "Variant", kind="select", options=["distilled", "dev"], default="distilled",
                         help="Must match the diffusion model file. Distilled samples in 8 steps without CFG. "
                              "Dev takes about 30 steps with CFG, and is the better base to train a LoRA on."),
            ProfileField("diffusion_model", "Diffusion model", hint="LTX 2.5 22B (dev or distilled)",
                         help="The transformer file, e.g. ltx-2.5-22b-dev-transformer-comfy-int8-convrot "
                              "(ComfyUI int8_convrot or bf16). It also holds the text connectors."),
            ProfileField("text_encoder", "Text encoder (CLIP)", hint="Gemma 4 12B with LTX 2.5's projection",
                         help="gemma4-12b-with-proj-ltx-2.5: Gemma 4 with LTX's text projections and the "
                              "tokenizer in one file."),
            ProfileField("vae", "Video VAE", hint="LTX 2.5 video VAE",
                         help="ltx-2.5-video-vae (the diffusion decoder) or ltx-2.5-video-vae-conv."),
            ProfileField("audio_vae", "Audio VAE", hint="LTX 2.5 audio VAE",
                         help="ltx-2.5-audio-vae, which also holds the vocoder."),
            ProfileField("sample_steps", "Sample steps", kind="number", required=False,
                         help="Blank: 8 for distilled (its own schedule), 30 for dev."),
            ProfileField("sample_cfg", "Sampling CFG", kind="number", required=False,
                         help="Blank: 1 (off) for distilled, 4 for dev."),
            ProfileField("sample_negative", "Sampling negative prompt", kind="text", required=False, default="",
                         help="Used with CFG above 1 (dev)."),
        ],
        training_defaults={
            "resolution": 512, "num_frames": 49, "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 2000,
            "batch_size": 1, "offload": 60, "optimizer": "adamw8bit", "timestep_sampling": "logit_normal",
            "i2v_probability": 0.5, "train_audio": True, "audio_loss_weight": 1.0, "sample_frames": 49,
            "sample_width": 768, "sample_height": 512, "bleed_width": 512, "bleed_height": 512,
        },
        workflows=[Workflow("t2v"), Workflow("i2v", images={"label": "First frame", "min": 1, "max": 1, "help": ""})],
        timestep_modes=[
            ("logit_normal", "Logit-normal + resolution shift"),
            ("shift", "Uniform + resolution shift"),
            ("uniform", "Uniform"),
        ],
        frame_rule="8n + 1 (9, 17, 25, … 49, 97)",
    )
    bucket_divisibility = M.VAE_SPATIAL
    caption_tokens = M.TEXT_TOKENS
    snap_frames = staticmethod(M.snap_frames)
    fps = 24
    block_list = "transformer_blocks"
    text_encoder_parts = M.CONNECTOR_PARTS
    # every attention (video, audio, text cross-attention, audio<->video) and both feed-forwards; not the
    # per-head gates
    lora_targets = ["transformer_blocks.*attn*.to_[qkv]", "transformer_blocks.*attn*.to_out.0",
                    "transformer_blocks.*ff.net.0.proj", "transformer_blocks.*ff.net.2"]
    # measured on 25-frame clips at 384-768 px (scripts/measure_memory.py), worst case: first-frame steps
    # with audio; per video/audio token (sequence_tokens has no text). Decoding is tiled at 512 px x 32 frames
    memory = MemoryProfile(saved=14_000, full=600_000, work=700_000, infer=330_000, vae_encode=(0.1e9, 122),
                           vae_decode=(0.2e9, 450), decode_tile_pixels=512 * 512 * 32, text_tokens=0,
                           # every layer's hidden states, padded to 1024 tokens, through the connectors
                           text_work=2.8)
    text_encoder_skip = ("vision_model.", "audio_projector.", "multi_modal_projector.", "tokenizer_json",
                         "hf_asset__")
    lora_metadata = {"modelspec.architecture": "ltx2.5/lora", "dainamo.family": "ltx2_5"}
    signature_keys = {
        "diffusion_model": ["transformer_blocks.0.audio_to_video_attn.to_gate_logits.weight",
                            "video_embeddings_connector.learnable_registers"],
        "text_encoder": ["text_embedding_projection.video_aggregate_embed.weight", "layers.0.layer_scalar"],
        "vae": ["encoder.down_blocks.0.res_blocks.0.conv1.conv.weight", "per_channel_statistics.mean-of-means"],
        "audio_vae": ["audio_vae.encoder.conv_in.conv.weight", "vocoder.bwe_generator.conv_pre.weight"],
    }
    signature_shapes = {
        # LTX's per-modality projection of all 49 Gemma hidden states (layer_scalar above is what tells Gemma 4
        # from LTX 2.3's Gemma 3, whose projection has this shape too)
        "text_encoder": {"text_embedding_projection.video_aggregate_embed.weight": (4096, 188160)},
    }

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.distilled = str(profile.get("variant") or "distilled") == "distilled"
        self.text_encoder = None  # model.TextStack
        self.video_vae = None  # model.VideoVAE
        self.audio_vae = None  # model.AudioCodec

    # --- defaults that depend on the variant ----------------------------------------------------------
    def sample_steps(self) -> int:
        v = self.profile_value(self.profile, "sample_steps")
        return int(v) if v not in (None, "") else (8 if self.distilled else 30)

    def cfg_scale(self, profile_cfg) -> float:
        if profile_cfg in (None, ""):
            return 1.0 if self.distilled else 4.0
        return float(profile_cfg)

    # --- text -----------------------------------------------------------------------------------------
    def load_text_encoder(self):
        from safetensors import safe_open
        from tokenizers import Tokenizer
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedTextModel, Gemma4UnifiedTextRotaryEmbedding,
        )

        path = self.profile["text_encoder"]
        self.log(f"Loading text encoder {path}")
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
            tokenizer = Tokenizer.from_str(f.get_tensor("tokenizer_json").numpy().tobytes().decode("utf-8"))
        text_cfg = json.loads(meta["gemma_config"])["text_config"]
        config = Gemma4UnifiedTextConfig(**{k: v for k, v in text_cfg.items() if k != "dtype"})
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            gemma = Gemma4UnifiedTextModel(config)
        weights = comfy_io.load_state_dict(path, strip_prefix="model.", only_prefixed=True,
                                           device=self.text_encoder_device)
        materialize(gemma, weights, self.dtype, keep_fp8=True, device=None)
        del weights
        # buffers computed at construction, not stored in the file
        gemma.rotary_emb = Gemma4UnifiedTextRotaryEmbedding(config).to(self.device)
        gemma.embed_tokens.embed_scale = torch.tensor(config.hidden_size**0.5, device=self.device)
        assert_materialized(gemma)

        # the text projections (text-encoder file) and connectors (diffusion-model file) as one module
        connectors = M.build_connectors(config.hidden_size, config.num_hidden_layers)
        proj = comfy_io.load_state_dict(path, include=("text_embedding_projection.",), device=self.text_encoder_device)
        dit = self.profile["diffusion_model"]
        prefix = "model.diffusion_model." if any(k.startswith("model.diffusion_model.")
                                                 for k in comfy_io.read_header(dit)[0]) else None
        conn = comfy_io.load_state_dict(dit, strip_prefix=prefix, only_prefixed=bool(prefix),
                                        include=M.CONNECTOR_PARTS, device=self.text_encoder_device)
        both = comfy_io.LoadedWeights({M.connector_key(k): v for k, v in {**proj.tensors, **conn.tensors}.items()},
                                      {M.connector_key(k): q for k, q in conn.quant.items()})
        materialize(connectors, both, self.dtype, keep_fp8=True, device=None)
        del proj, conn, both  # else their tensors stay in RAM beside the streamed copies
        assert_materialized(connectors)
        # low VRAM: Gemma's layers and the large text projections stream (the connectors' blocks are small)
        self.text_encoder = self.stream_text_encoder(
            M.TextStack(gemma.eval(), connectors.eval(), tokenizer, config.bos_token_id))

    def encode_condition(self, prompt: str, images: list | None):
        return self.text_encoder.encode(prompt, self.device)

    def encode_prompt(self, prompt: str):
        return self.encode_condition(prompt, None)

    # --- vae ------------------------------------------------------------------------------------------
    def load_vae(self):
        path = self.profile["vae"]
        self.log(f"Loading video VAE {path}")
        t = comfy_io.load_state_dict(path, device=self.device).tensors
        mean, std = t["per_channel_statistics.mean-of-means"], t["per_channel_statistics.std-of-means"]
        if "decoder.det_stages.0.0.attn.qkv.weight" in t:  # the diffusion decoder (LTX 2.5's default VAE)
            encoder = M.build_encoder()
            materialize(encoder, comfy_io.LoadedWeights(
                {M.encoder_key(k[len("encoder."):]): v for k, v in t.items() if k.startswith("encoder.")}),
                self.dtype, device=self.device)
            raw = {k[len("decoder."):]: v for k, v in t.items() if k.startswith("decoder.")}
            # not a decoder input (decoding with it added to the latents reconstructs worse), and diffusers'
            # converter doesn't map it either
            raw.pop("type_emb", None)
            sd = {"decoder." + k: v for k, v in M.decoder_state(raw).items()}
            sd.update(latents_mean=mean, latents_std=std)
            decoder = M.build_diffusion_decoder()
            materialize(decoder, comfy_io.LoadedWeights(sd), self.dtype, device=self.device)
            M.use_halo_attention(decoder)
            # decode the last stages in overlapping tiles of 32 frames x 512 px, so a long clip fits next to the DiT
            decoder.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512, tile_sample_min_num_frames=32,
                                  tile_sample_stride_height=448, tile_sample_stride_width=448,
                                  tile_sample_stride_num_frames=24)
            vae = M.VideoVAE(encoder, decoder, diffusion=True)
        else:  # the classic conv VAE
            conv = M.build_conv_vae()
            sd = {("encoder." + M.encoder_key(k[len("encoder."):])) if k.startswith("encoder.") else
                  ("decoder." + M.conv_decoder_key(k[len("decoder."):])): v
                  for k, v in t.items() if k.startswith(("encoder.", "decoder."))}
            sd.update(latents_mean=mean, latents_std=std)
            materialize(conv, comfy_io.LoadedWeights(sd), self.dtype, device=self.device)
            conv.enable_tiling()
            vae = M.VideoVAE(conv.encoder, conv, diffusion=False)
        vae.latents_mean.copy_(mean.float())
        vae.latents_std.copy_(std.float())
        self.video_vae = vae.to(self.device).eval().requires_grad_(False)
        assert_materialized(self.video_vae)
        del t

        apath = self.profile["audio_vae"]
        self.log(f"Loading audio VAE {apath}")
        t = comfy_io.load_state_dict(apath, device=self.device).tensors
        audio = M.build_audio_vae()
        materialize(audio, comfy_io.LoadedWeights(
            {M.audio_vae_key(k[len("audio_vae."):]): v for k, v in t.items() if k.startswith("audio_vae.")}),
            torch.float32, device=self.device)
        vocoder = M.build_vocoder()
        materialize(vocoder, comfy_io.LoadedWeights(
            {M.vocoder_key(k[len("vocoder."):]): v for k, v in t.items() if k.startswith("vocoder.")}),
            torch.float32, device=self.device)
        self.audio_vae = M.AudioCodec(audio, vocoder).to(self.device).eval().requires_grad_(False)
        assert_materialized(self.audio_vae)

    def unload_vae(self):
        self.video_vae = self.audio_vae = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_item(self, item, settings: dict) -> dict:
        if item.is_video:
            px = ds.read_video_frames(item.path, item.size, item.frames, self.fps, item.flip)
            payload = {"video": self.video_vae.encode(px)[0].cpu(), "frames": item.frames}
            if settings.get("train_audio", True) and item.has_audio:
                wav = ds.read_audio(item.path, item.frames / self.fps, M.AUDIO_SAMPLE_RATE, 2)
                if item.flip:  # a mirrored picture hears the left and right channels swapped
                    wav = wav.flip(0)
                payload["audio"] = self.audio_vae.encode(wav, M.audio_latent_frames(item.frames, self.fps)).cpu()
        else:
            px = ds.load_and_crop(item.path, item.size, item.flip).unsqueeze(2)
            payload = {"video": self.video_vae.encode(px)[0].cpu(), "frames": 1}
        return payload

    def _first_frame_latent(self, path: str, width: int, height: int) -> torch.Tensor:
        """A first-frame image, cover-cropped to the size and re-compressed, encoded: (1, 128, 1, h, w)."""
        with Image.open(path) as im:
            img = ImageOps.fit(ds.to_rgb(ImageOps.exif_transpose(im)), (width, height), Image.Resampling.LANCZOS)
        img = _crf(img, IMAGE_CRF).resize((width, height), Image.Resampling.LANCZOS)
        px = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None]
        return self.video_vae.encode(px)

    # --- transformer ----------------------------------------------------------------------------------
    def build_transformer(self, shapes):
        return M.build_transformer(shapes)

    def check_transformer_state(self, path, loaded):
        loaded.tensors = {M.transformer_key(k): v for k, v in loaded.tensors.items()}
        loaded.quant = {M.transformer_key(k): q for k, q in loaded.quant.items()}

    @classmethod
    def sequence_tokens(cls, width, height, frames=1):
        """Video and audio tokens. The text (always 1024 tokens) is cross-attention context, not part of the
        sequence; its fixed cost is in the measured constants."""
        frames = M.snap_frames(frames) if frames > 1 else 1
        video = (width // M.VAE_SPATIAL) * (height // M.VAE_SPATIAL) * M.latent_frames(frames)
        return video + M.audio_latent_frames(frames, cls.fps)

    def _sigmas(self, batch: int, video_tokens: int, settings: dict) -> torch.Tensor:
        mode = settings.get("timestep_sampling", "logit_normal")
        if mode == "uniform":
            return torch.rand(batch).clamp(1e-3, 1.0)
        u = torch.sigmoid(torch.randn(batch)) if mode == "logit_normal" else torch.rand(batch)
        return M.time_shift(M.mu_for(video_tokens), u.clamp(1e-4, 1 - 1e-4)).clamp(1e-3, 1.0)

    def sample_timesteps(self, batch: int, latent_hw, settings: dict) -> torch.Tensor:
        """The training sigma distribution at this latent size (clips at the job's clip length)."""
        frames = int(settings.get("num_frames", 1) or 1)
        t = M.latent_frames(M.snap_frames(frames)) if frames > 1 else 1
        return self._sigmas(batch, int(latent_hw[0]) * int(latent_hw[1]) * t, settings)

    def _predict(self, model, video, audio, conds: list[dict], video_t, sigma, audio_sigma, shape, audio_frames):
        """Run the DiT: video tokens (B, N, 128), audio tokens (B, A, 128); ``video_t`` (B,) or per token
        (B, N), ``sigma`` / ``audio_sigma`` (B,), all in [0, 1]. Returns the velocity (video, audio)."""
        d = self.device
        _, _, t, h, w = shape
        text_v = torch.stack([c["video"] for c in conds]).to(d, self.dtype)
        text_a = torch.stack([c["audio"] for c in conds]).to(d, self.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pv, pa = model(
                hidden_states=video.to(d, self.dtype), audio_hidden_states=audio.to(d, self.dtype),
                encoder_hidden_states=text_v, audio_encoder_hidden_states=text_a,
                timestep=video_t.to(d, torch.float32) * 1000, audio_timestep=audio_sigma.to(d, torch.float32) * 1000,
                sigma=sigma.to(d, torch.float32) * 1000, audio_sigma=audio_sigma.to(d, torch.float32) * 1000,
                num_frames=t, height=h, width=w, fps=float(self.fps), audio_num_frames=audio_frames,
                use_cross_timestep=True, return_dict=False)
        return pv.float(), pa.float()

    def training_loss(self, model, payloads, conds, settings, rng=None, sigma=None, generator=None,
                      deterministic=False):
        d = self.device

        def randn(shape):
            if generator is not None:
                return torch.randn(shape, generator=generator).to(d)
            return torch.randn(shape, device=d)

        x0 = torch.stack([p["video"] for p in payloads]).to(d, torch.float32)  # (B, 128, t, h, w)
        b, _, t, h, w = x0.shape
        frames = int(payloads[0].get("frames", 1))
        a_frames = M.audio_latent_frames(frames, self.fps)
        s = torch.full((b,), float(sigma), device=d) if sigma is not None else \
            self._sigmas(b, t * h * w, settings).to(d)
        noise = randn(x0.shape)
        s5 = s.view(-1, 1, 1, 1, 1)
        noisy = (1 - s5) * x0 + s5 * noise
        target = noise - x0  # the velocity

        # image-to-video on a share of clip steps: the first latent frame clean, at timestep 0, not in the loss
        first = (frames > 1 and t > 1 and not deterministic and rng is not None
                 and rng.random() < float(settings.get("i2v_probability", 0.5)))
        video_t = s
        if first:
            noisy[:, :, :1] = x0[:, :, :1]
            video_t = s.view(-1, 1).repeat(1, t * h * w)
            video_t[:, : h * w] = 0

        audio_target = None
        if settings.get("train_audio", True) and frames > 1 and all("audio" in p for p in payloads):
            xa = torch.stack([p["audio"] for p in payloads]).to(d, torch.float32)  # (B, A, 128)
            na = randn(xa.shape)
            audio = (1 - s.view(-1, 1, 1)) * xa + s.view(-1, 1, 1) * na
            audio_sigma, audio_target = s, na - xa
        else:  # no soundtrack: pure noise, and the model is told so
            audio, audio_sigma = randn((b, a_frames, M.LATENT_CHANNELS)), torch.ones(b, device=d)

        pv, pa = self._predict(model, _pack(noisy), audio, [c["text"] for c in conds], video_t, s, audio_sigma,
                               x0.shape, audio.shape[1])
        err = (_unpack(pv, x0.shape) - target) ** 2
        loss = err[:, :, 1:].mean() if first else err.mean()
        if audio_target is not None:
            loss = loss + float(settings.get("audio_loss_weight", 1.0)) * F.mse_loss(pa, audio_target)
        return loss

    # --- sampling -------------------------------------------------------------------------------------
    def _schedule(self, scheduler: str, steps: int, video_tokens: int, shift: float | None) -> list[float]:
        from ...sampling import shift_sigma, spacing

        if scheduler == "default" and shift is None:
            if self.distilled and steps == len(M.DISTILLED_SIGMAS):
                return M.DISTILLED_SIGMAS + [0.0]
            return M.shifted_sigmas(steps, video_tokens)
        s = float(shift) if shift is not None else math.exp(M.mu_for(video_tokens))
        return [shift_sigma(u, s) for u in spacing("simple" if scheduler == "default" else scheduler, steps)]

    @torch.no_grad()
    def sample(self, model, cond, uncond, width, height, steps, cfg, seed, frames=1, images=None, sampler="euler",
               scheduler="default", shift=None, callback=None):
        """Video and audio latents, denoised together as one state for the generic samplers; with a
        first-frame image its latent frame stays clean (timestep 0, zero velocity)."""
        from ... import sampling

        d = self.device
        frames = M.snap_frames(frames) if frames > 1 else 1
        shape = (1, M.LATENT_CHANNELS, M.latent_frames(frames), height // M.VAE_SPATIAL, width // M.VAE_SPATIAL)
        _, _, t, h, w = shape
        a_frames = M.audio_latent_frames(frames, self.fps)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        video = torch.randn(shape, generator=g)
        audio = torch.randn((1, a_frames, M.LATENT_CHANNELS), generator=g)
        first = None
        if images and frames > 1:
            if self.video_vae is None:
                raise RuntimeError("The video VAE must be loaded to encode the first frame")
            first = self._first_frame_latent(images[0], width, height).to(d)
            video[:, :, :1] = first.cpu()
        nv = video.numel()
        use_cfg = cfg is not None and cfg != 1 and uncond is not None
        conds = [cond, uncond] if use_cfg else [cond]
        b = len(conds)

        def velocity(x, s):
            v_lat = x[:nv].view(shape).clone()
            if first is not None:
                v_lat[:, :, :1] = first
            a_lat = x[nv:].view(1, a_frames, -1).to(d)
            sig = torch.full((b,), float(s), device=d)
            video_t = sig
            if first is not None:
                video_t = sig.view(-1, 1).repeat(1, t * h * w)
                video_t[:, : h * w] = 0
            pv, pa = self._predict(model, _pack(v_lat).expand(b, -1, -1), a_lat.expand(b, -1, -1), conds, video_t,
                                   sig, sig, shape, a_frames)
            if use_cfg:
                pv, pa = pv[1:] + cfg * (pv[:1] - pv[1:]), pa[1:] + cfg * (pa[:1] - pa[1:])
            pv = _unpack(pv, shape)
            if first is not None:
                pv[:, :, :1] = 0
            return torch.cat([pv.flatten(), pa.flatten()])

        sigmas = self._schedule(scheduler, int(steps), t * h * w, shift)
        x = sampling.run(sampler, torch.cat([video.flatten(), audio.flatten()]).to(d), sigmas, velocity, callback)
        return {"video": x[:nv].view(shape).cpu(), "audio": x[nv:].view(a_frames, -1).cpu(), "frames": frames}

    @torch.no_grad()
    def to_media(self, latents) -> list:
        g = torch.Generator(device=self.device).manual_seed(DECODE_SEED)
        video = self.video_vae.decode(latents["video"], generator=g)  # (1, 3, F, H, W) in [-1, 1]
        frames = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 3, 0).cpu().numpy()
        if latents["frames"] == 1:
            return [Image.fromarray(frames[0])]
        wav = self.audio_vae.decode(latents["audio"]).cpu().numpy()  # (2, samples) at 48 kHz
        wav = wav[:, : round(latents["frames"] / self.fps * 48000)]
        return [{"frames": frames, "audio": wav, "fps": self.fps, "sample_rate": 48000}]

    def decode_latents(self, latents):
        raise NotImplementedError("LTX 2.5 decodes through to_media")

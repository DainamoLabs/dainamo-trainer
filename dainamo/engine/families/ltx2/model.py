"""LTX-2.5 components, built from diffusers' LTX-2 classes and loaded from ComfyUI-format files.

The diffusion model, text connectors, video encoder, diffusion video decoder, audio VAE and vocoder
are diffusers' own classes (Apache-2.0, used as a library, not vendored). What's here maps the
original (ComfyUI) key layout onto them, following diffusers' ``scripts/convert_ltx2_to_diffusers.py``
for ``--version 2.5``, so the int8 weights load straight into them without a converted copy on disk.

The DiT's Linear layers keep their original names under this mapping (``transformer_blocks.N.attn1.to_q``,
``ff.net.0.proj``, ...), so a LoRA trained on them saves with the keys ComfyUI's LTX loader uses.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VAE_SPATIAL = 32  # video VAE: 32x32 pixels per latent, 8 frames per latent frame (the first frame alone)
VAE_TEMPORAL = 8
LATENT_CHANNELS = 128
AUDIO_SAMPLE_RATE = 16000  # the audio VAE's input; the vocoder's bandwidth extension outputs 48 kHz
AUDIO_HOP = 160
AUDIO_N_FFT = 1024
AUDIO_MEL_BINS = 64
AUDIO_DOWNSAMPLE = 4  # mel frames per audio latent frame (and mel bins per latent bin)
AUDIO_LATENTS_PER_SECOND = AUDIO_SAMPLE_RATE / AUDIO_HOP / AUDIO_DOWNSAMPLE  # 25
TEXT_TOKENS = 1024  # the prompt is padded to this, and the connectors fill the padding with registers
# the distilled model's schedule (diffusers' DISTILLED_SIGMA_VALUES, from Lightricks' ltx-pipelines)
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
# resolution-dependent shift of the training and sampling sigmas: mu linear in video tokens
BASE_SEQ, MAX_SEQ, BASE_SHIFT, MAX_SHIFT = 1024, 4096, 0.95, 2.05
SHIFT_TERMINAL = 0.1  # the non-distilled schedule is stretched so its last step lands here


# --- schedule --------------------------------------------------------------------------------------
def snap_frames(frames: int) -> int:
    """Clip lengths the VAE takes: 8k + 1 (rounded down)."""
    return max(1, (int(frames) - 1) // VAE_TEMPORAL * VAE_TEMPORAL + 1)


def latent_frames(frames: int) -> int:
    return (int(frames) - 1) // VAE_TEMPORAL + 1


def audio_latent_frames(frames: int, fps: float) -> int:
    return max(1, round(int(frames) / fps * AUDIO_LATENTS_PER_SECOND))


def mu_for(video_tokens: int) -> float:
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ)
    return BASE_SHIFT + slope * (video_tokens - BASE_SEQ)


def time_shift(mu: float, sigma):
    """FlowMatchEulerDiscreteScheduler's exponential shift: e^mu / (e^mu + (1 / sigma - 1))."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0))


def shifted_sigmas(steps: int, video_tokens: int, shift: float | None = None) -> list[float]:
    """The non-distilled schedule: linear from 1 to 1/steps, shifted (by the token count's mu, or e^mu =
    ``shift``), stretched so the last step is SHIFT_TERMINAL, then 0."""
    if steps <= 1:  # nothing to stretch: one step from noise straight to the image
        return [1.0, 0.0]
    mu = math.log(shift) if shift is not None else mu_for(video_tokens)
    base = [1.0 - i * (1.0 - 1.0 / steps) / max(1, steps - 1) for i in range(steps)]
    s = [time_shift(mu, x) for x in base]
    scale = (1.0 - s[-1]) / (1.0 - SHIFT_TERMINAL)
    return [1.0 - (1.0 - x) / scale for x in s] + [0.0]


# --- the diffusion model -----------------------------------------------------------------------------
TRANSFORMER_RENAMES = [  # original -> diffusers, applied in order (as the converter's 2.3/2.5 dict)
    ("patchify_proj", "proj_in"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
]
CONNECTOR_PARTS = ("video_embeddings_connector.", "audio_embeddings_connector.")


def transformer_key(key: str) -> str:
    """A diffusion-model key (prefix stripped) in diffusers' naming."""
    for old, new in TRANSFORMER_RENAMES:
        key = key.replace(old, new)
    if key.startswith("adaln_single."):
        key = "time_embed." + key[len("adaln_single."):]
    elif key.startswith("audio_adaln_single."):
        key = "audio_time_embed." + key[len("audio_adaln_single."):]
    return key


def transformer_config(shapes: dict) -> dict:
    """LTX2VideoTransformer3DModel config for LTX-2.5 (diffusers' converter, ``--version 2.5``); the
    block count, the video FFN bias and the keyframe embedding are read from the checkpoint."""
    layers = 1 + max(int(k.split(".")[1]) for k in shapes if k.startswith("transformer_blocks."))
    return dict(
        in_channels=128, out_channels=128, patch_size=1, patch_size_t=1, num_attention_heads=32,
        attention_head_dim=128, cross_attention_dim=4096, vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20,
        base_height=2048, base_width=2048, gated_attn=True, cross_attn_mod=True, audio_in_channels=128,
        audio_out_channels=128, audio_patch_size=1, audio_patch_size_t=1, audio_num_attention_heads=32,
        audio_attention_head_dim=64, audio_cross_attention_dim=2048, audio_scale_factor=4,
        audio_pos_embed_max_pos=20, audio_sampling_rate=16000, audio_hop_length=160, audio_gated_attn=True,
        audio_cross_attn_mod=True, num_layers=layers, activation_fn="gelu-approximate",
        qk_norm="rms_norm_across_heads", norm_elementwise_affine=False, norm_eps=1e-6, caption_channels=3840,
        attention_bias=True, attention_out_bias=True, rope_theta=10000.0, rope_double_precision=True,
        causal_offset=1, timestep_scale_multiplier=1000, cross_attn_timestep_scale_multiplier=1000,
        rope_type="split", use_prompt_embeddings=False, perturbed_attn=False,
        ff_bias="transformer_blocks.0.ff.net.0.proj.bias" in shapes,
        use_prompt_adaln_single="prompt_adaln_single.linear.weight" in shapes,
        use_keyframes_abs_pos_embedding="keyframes_abs_pos_embedding" in shapes,
    )


def build_transformer(shapes: dict) -> nn.Module:
    """The diffusion model on the meta device. ``enable_gradient_checkpointing(bool)`` as the trainer calls it."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    class LTX2Transformer(LTX2VideoTransformer3DModel):
        def enable_gradient_checkpointing(self, enabled=True):
            if enabled:
                super().enable_gradient_checkpointing()
            else:
                self.disable_gradient_checkpointing()

    with torch.device("meta"):
        return LTX2Transformer(**transformer_config(shapes))


# --- text: Gemma 4 hidden states -> per-modality projections -> connectors ---------------------------
def connector_key(key: str) -> str:
    """A connector or text-projection key in diffusers' ``LTX2TextConnectors`` naming."""
    for old, new in (("video_embeddings_connector", "video_connector"),
                     ("audio_embeddings_connector", "audio_connector"), ("transformer_1d_blocks", "transformer_blocks"),
                     ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
                     ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
                     ("q_norm", "norm_q"), ("k_norm", "norm_k")):
        key = key.replace(old, new)
    return key


def build_connectors(caption_channels: int, num_layers: int) -> nn.Module:
    """LTX2TextConnectors for LTX-2.5 on the meta device (diffusers' converter, ``--version 2.5``)."""
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    with torch.device("meta"):
        return LTX2TextConnectors(
            caption_channels=caption_channels, text_proj_in_factor=num_layers + 1,
            video_connector_num_attention_heads=32, video_connector_attention_head_dim=128,
            video_connector_num_layers=8, video_connector_num_learnable_registers=128, video_gated_attn=True,
            audio_connector_num_attention_heads=32, audio_connector_attention_head_dim=64,
            audio_connector_num_layers=8, audio_connector_num_learnable_registers=128, audio_gated_attn=True,
            connector_rope_base_seq_len=4096, rope_theta=10000.0, rope_double_precision=True,
            causal_temporal_positioning=False, rope_type="split", per_modality_projections=True,
            video_hidden_dim=4096, audio_hidden_dim=2048, proj_bias=True)


class TextStack(nn.Module):
    """Gemma 4's text decoder and the LTX connectors: one module, so the Sandbox parks them together."""

    def __init__(self, gemma: nn.Module, connectors: nn.Module, tokenizer, bos_id: int):
        super().__init__()
        self.gemma, self.connectors = gemma, connectors
        self.tokenizer, self.bos_id = tokenizer, bos_id

    @torch.no_grad()
    def encode(self, prompt: str, device) -> dict[str, torch.Tensor]:
        """{"video": (1024, 4096), "audio": (1024, 2048)}: the DiT's text conditions for ``prompt``, as the
        pipeline makes them (BOS + prompt, left-padded to 1024; the connectors replace padding with registers)."""
        ids = [self.bos_id] + self.tokenizer.encode(prompt.strip(), add_special_tokens=False).ids[: TEXT_TOKENS - 1]
        out = self.gemma(input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=-1).flatten(2, 3)  # (1, n, hidden * (layers + 1))
        n = hidden.shape[1]
        padded = hidden.new_zeros(1, TEXT_TOKENS, hidden.shape[2])
        padded[:, TEXT_TOKENS - n:] = hidden
        mask = torch.zeros(1, TEXT_TOKENS, dtype=torch.long, device=device)
        mask[:, TEXT_TOKENS - n:] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
            video, audio, _ = self.connectors(padded, mask)
        return {"video": video[0].to("cpu", torch.bfloat16), "audio": audio[0].to("cpu", torch.bfloat16)}


# --- video VAE ----------------------------------------------------------------------------------------
ENCODER_RENAMES = [  # the converter's LTX-2.3/2.5 video VAE dict, encoder half
    ("down_blocks.1", "down_blocks.0.downsamplers.0"), ("down_blocks.2", "down_blocks.1"),
    ("down_blocks.3", "down_blocks.1.downsamplers.0"), ("down_blocks.4", "down_blocks.2"),
    ("down_blocks.5", "down_blocks.2.downsamplers.0"), ("down_blocks.6", "down_blocks.3"),
    ("down_blocks.7", "down_blocks.3.downsamplers.0"), ("down_blocks.8", "mid_block"), ("res_blocks", "resnets"),
]
DECODER_RENAMES = [  # the converter's LTX-2.5 diffusion-decoder dict
    ("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1."),
    ("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2."),
    (".attn.proj.", ".attn.to_out.0."), (".attn.q_norm.", ".attn.norm_q."), (".attn.k_norm.", ".attn.norm_k."),
]
# a static AdaLN gate some checkpoints carry is folded into the Linear it gates (the decoder's residuals are ungated)
GATE_FOLDS = {".attn.to_out.0.weight": ".gate_msa", ".attn.to_out.0.bias": ".gate_msa",
              ".mlp.w_down.weight": ".gate_mlp", ".context_proj.weight": ".gate_ctx", ".context_proj.bias": ".gate_ctx"}


def encoder_key(key: str) -> str:
    """A ``encoder.*`` VAE key (prefix stripped) in diffusers' ``LTX2VideoEncoder3d`` naming."""
    for old, new in ENCODER_RENAMES:
        key = key.replace(old, new)
    return key


def build_encoder() -> nn.Module:
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoEncoder3d

    with torch.device("meta"):
        return LTX2VideoEncoder3d(
            in_channels=3, out_channels=LATENT_CHANNELS, block_out_channels=(256, 512, 1024, 1024),
            spatio_temporal_scaling=(True, True, True, True), layers_per_block=(4, 6, 4, 2, 2),
            downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"), patch_size=4,
            patch_size_t=1, resnet_norm_eps=1e-6, is_causal=True, spatial_padding_mode="zeros")


def decoder_state(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The ``decoder.*`` tensors of an LTX-2.5 diffusion-decoder VAE in ``LTX2VideoDiffusionDecoderModel``
    naming (without the ``decoder.`` prefix): renamed, static gates folded in, fused qkv split."""
    gates = {k: v for k, v in tensors.items() if k.endswith((".gate_msa", ".gate_mlp", ".gate_ctx"))}
    out = {}
    for key, value in tensors.items():
        if key in gates or key.startswith("coarse_") or ".coarse_" in key:
            continue
        new = key
        for old, rep in DECODER_RENAMES:
            new = new.replace(old, rep)
        leaf = next((s for s in GATE_FOLDS if new.endswith(s)), None)
        gate = gates.get(new[: -len(leaf)] + GATE_FOLDS[leaf]) if leaf else None  # a sibling: same block prefix
        if gate is not None:
            value = ((gate.float().unsqueeze(1) if value.ndim == 2 else gate.float()) * value.float()).to(value.dtype)
        if new.endswith((".qkv.weight", ".qkv.bias")):
            leaf = new.rsplit(".", 1)[1]
            prefix = new[: -len(f"qkv.{leaf}")]
            q, k, v = value.chunk(3, dim=0)
            out[f"{prefix}to_q.{leaf}"], out[f"{prefix}to_k.{leaf}"], out[f"{prefix}to_v.{leaf}"] = q, k, v
            continue
        out[new] = value
    return out


def _window_starts(n: int, size: int, lo: int, hi: int, device) -> torch.Tensor:
    """First key index of each query's window along one axis, for queries lo..hi-1: centred, shifted
    inward at the borders so every window holds exactly ``size`` positions (NATTEN's rule)."""
    return (torch.arange(lo, hi, device=device) - size // 2).clamp(0, n - size)


def neighborhood_attention(q, k, v, kernel: tuple[int, int, int], tile: tuple[int, int, int] = (1, 16, 32)):
    """3D neighborhood attention on (B, T, H, W, heads, dim) tensors (q already scaled): each query attends
    to the ``kernel`` window around it, shifted inward at the grid borders. Computed tile by tile: a tile's
    queries attend to the union of their windows (the tile plus a halo) through SDPA with an exact mask,
    so memory stays small and only the halo is computed beyond what NATTEN would."""
    b, t, h, w, heads, dim = q.shape
    kt, kh, kw = min(kernel[0], t), min(kernel[1], h), min(kernel[2], w)
    out = torch.empty_like(q)
    dev = q.device
    for t0 in range(0, t, tile[0]):
        t1 = min(t, t0 + tile[0])
        st = _window_starts(t, kt, t0, t1, dev)
        rt0, rt1 = int(st[0]), int(st[-1]) + kt
        ar = torch.arange(rt0, rt1, device=dev)[None]
        mt = (ar >= st[:, None]) & (ar < st[:, None] + kt)
        for h0 in range(0, h, tile[1]):
            h1 = min(h, h0 + tile[1])
            sh = _window_starts(h, kh, h0, h1, dev)
            rh0, rh1 = int(sh[0]), int(sh[-1]) + kh
            ar = torch.arange(rh0, rh1, device=dev)[None]
            mh = (ar >= sh[:, None]) & (ar < sh[:, None] + kh)
            for w0 in range(0, w, tile[2]):
                w1 = min(w, w0 + tile[2])
                sw = _window_starts(w, kw, w0, w1, dev)
                rw0, rw1 = int(sw[0]), int(sw[-1]) + kw
                ar = torch.arange(rw0, rw1, device=dev)[None]
                mw = (ar >= sw[:, None]) & (ar < sw[:, None] + kw)
                mask = (mt[:, None, None, :, None, None] & mh[None, :, None, None, :, None]
                        & mw[None, None, :, None, None, :]).reshape((t1 - t0) * (h1 - h0) * (w1 - w0), -1)
                qq = q[:, t0:t1, h0:h1, w0:w1].reshape(b, -1, heads, dim).transpose(1, 2)
                kk = k[:, rt0:rt1, rh0:rh1, rw0:rw1].reshape(b, -1, heads, dim).transpose(1, 2)
                vv = v[:, rt0:rt1, rh0:rh1, rw0:rw1].reshape(b, -1, heads, dim).transpose(1, 2)
                o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=1.0)
                out[:, t0:t1, h0:h1, w0:w1] = o.transpose(1, 2).reshape(b, t1 - t0, h1 - h0, w1 - w0, heads, dim)
    return out


def use_halo_attention(decoder: nn.Module) -> None:
    """Run the diffusion decoder's neighborhood attention with ``neighborhood_attention`` (plain SDPA)
    instead of NATTEN, or an uncompiled FlexAttention mask that materialises every score. A grid smaller
    than the kernel (a still image, a clip under 17 frames) narrows the window to the grid, as diffusers'
    mask builder does, rather than being refused."""
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodAttention

    class HaloNeighborhoodAttention(LTX2VideoVaeNeighborhoodAttention):
        def build_block_mask(self, hidden_states):
            return None

        def forward(self, hidden_states, block_mask=None):
            b, t, h, w, c = hidden_states.shape
            q, k, v = self.project_qkv(hidden_states)
            return self.to_out[0](neighborhood_attention(q, k, v, self.kernel_size).reshape(b, t, h, w, c))

    for m in decoder.modules():
        if isinstance(m, LTX2VideoVaeNeighborhoodAttention):
            m.__class__ = HaloNeighborhoodAttention


def build_diffusion_decoder() -> nn.Module:
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoDiffusionDecoderModel

    with torch.device("meta"):
        return LTX2VideoDiffusionDecoderModel(
            out_channels=3, latent_channels=LATENT_CHANNELS, patch_size=4, decoder_head_dim=64,
            decoder_stage_channels=(2048, 1024, 512, 512, 256), decoder_stage_depths=(4, 6, 4, 2, 8),
            decoder_stage_kernels=((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
            decoder_upsample_strides=((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
            decoder_upsample_channel_reductions=(2, 2, 1, 2), decoder_stage5_kernel=(11, 11, 11),
            decoder_t_emb_dim=384, decoder_timestep_scale_multiplier=1000.0, decoder_model_output_type="x0",
            decoder_num_inference_steps=1, spatial_compression_ratio=32, temporal_compression_ratio=8)


def build_conv_vae() -> nn.Module:
    """The classic convolutional VAE (``ltx-2.5-video-vae-conv``): diffusers' converter config for 2.5."""
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video

    with torch.device("meta"):
        return AutoencoderKLLTX2Video(
            in_channels=3, out_channels=3, latent_channels=LATENT_CHANNELS, block_out_channels=(256, 512, 1024, 1024),
            decoder_block_out_channels=(256, 512, 512, 1024), layers_per_block=(4, 6, 4, 2, 2),
            decoder_layers_per_block=(4, 6, 4, 2, 2), spatio_temporal_scaling=(True, True, True, True),
            decoder_spatio_temporal_scaling=(True, True, True, True),
            decoder_inject_noise=(False, False, False, False, False),
            downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
            upsample_residual=(False, False, False, False), upsample_factor=(2, 2, 1, 2), timestep_conditioning=False,
            patch_size=4, patch_size_t=1, resnet_norm_eps=1e-6, encoder_causal=True, decoder_causal=False,
            encoder_spatial_padding_mode="zeros", decoder_spatial_padding_mode="zeros",
            spatial_compression_ratio=32, temporal_compression_ratio=8)


CONV_DECODER_RENAMES = [  # the converter's LTX-2.3/2.5 dict, decoder half
    ("up_blocks.0", "mid_block"), ("up_blocks.1", "up_blocks.0.upsamplers.0"), ("up_blocks.2", "up_blocks.0"),
    ("up_blocks.3", "up_blocks.1.upsamplers.0"), ("up_blocks.4", "up_blocks.1"),
    ("up_blocks.5", "up_blocks.2.upsamplers.0"), ("up_blocks.6", "up_blocks.2"),
    ("last_time_embedder", "time_embedder"), ("last_scale_shift_table", "scale_shift_table"),
    ("up_blocks.7", "up_blocks.3.upsamplers.0"), ("up_blocks.8", "up_blocks.3"), ("res_blocks", "resnets"),
]


def conv_decoder_key(key: str) -> str:
    for old, new in CONV_DECODER_RENAMES:
        key = key.replace(old, new)
    return key


class VideoVAE(nn.Module):
    """The causal video encoder and either decoder (the diffusion decoder, or the whole conv VAE), with the
    latent statistics. ``encode`` gives the DiT's normalised latents; ``decode`` takes them back to pixels
    in [-1, 1]."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, diffusion: bool):
        super().__init__()
        self.encoder, self.decoder, self.diffusion = encoder, decoder, diffusion
        self.register_buffer("latents_mean", torch.zeros(LATENT_CHANNELS))
        self.register_buffer("latents_std", torch.ones(LATENT_CHANNELS))

    def _stats(self, like: torch.Tensor):
        return (self.latents_mean.view(1, -1, 1, 1, 1).to(like), self.latents_std.view(1, -1, 1, 1, 1).to(like))

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, 3, F, H, W) in [-1, 1] -> (B, 128, latent frames, H/32, W/32), normalised (the posterior mean)."""
        p = next(self.encoder.parameters())
        moments = self.encoder(pixels.to(p.device, p.dtype))
        mean, std = self._stats(moments)
        return ((moments[:, :LATENT_CHANNELS] - mean) / std).float()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        p = next(self.decoder.parameters())
        z = latents.to(p.device, p.dtype)
        mean, std = self._stats(z)
        z = z * std + mean
        if self.diffusion:
            return self.decoder.decode(z, generator=generator, return_dict=False)[0].float()
        return self.decoder.decode(z, return_dict=False)[0].float()


# --- audio ----------------------------------------------------------------------------------------------
def build_audio_vae() -> nn.Module:
    from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio

    with torch.device("meta"):
        return AutoencoderKLLTX2Audio(
            base_channels=128, output_channels=2, ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=None,
            in_channels=2, resolution=256, latent_channels=8, norm_type="pixel", causality_axis="height",
            dropout=0.0, mid_block_add_attention=False, sample_rate=AUDIO_SAMPLE_RATE, mel_hop_length=AUDIO_HOP,
            is_causal=True, mel_bins=AUDIO_MEL_BINS, double_z=True)


def audio_vae_key(key: str) -> str:
    return key.replace("per_channel_statistics.mean-of-means", "latents_mean") \
        .replace("per_channel_statistics.std-of-means", "latents_std")


def build_vocoder() -> nn.Module:
    """On the CPU, not the meta device: its resampling filter is a buffer computed at construction, not
    stored in the file (it's small; move it to the GPU after loading)."""
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    return LTX2VocoderWithBWE(
        in_channels=128, hidden_channels=1536, out_channels=2, upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        upsample_factors=[5, 2, 2, 2, 2, 2], resnet_kernel_sizes=[3, 7, 11],
        resnet_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], act_fn="snakebeta", leaky_relu_negative_slope=0.1,
        antialias=True, antialias_ratio=2, antialias_kernel_size=12, final_act_fn=None, final_bias=False,
        bwe_in_channels=128, bwe_hidden_channels=512, bwe_out_channels=2,
        bwe_upsample_kernel_sizes=[12, 11, 4, 4, 4], bwe_upsample_factors=[6, 5, 2, 2, 2],
        bwe_resnet_kernel_sizes=[3, 7, 11], bwe_resnet_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        bwe_act_fn="snakebeta", bwe_leaky_relu_negative_slope=0.1, bwe_antialias=True, bwe_antialias_ratio=2,
        bwe_antialias_kernel_size=12, bwe_final_act_fn=None, bwe_final_bias=False, filter_length=512,
        hop_length=80, window_length=512, num_mel_channels=64, input_sampling_rate=16000,
        output_sampling_rate=48000)


def vocoder_key(key: str) -> str:
    """A ``vocoder.*`` key (prefix stripped) in diffusers' ``LTX2VocoderWithBWE`` naming."""
    for old, new in (("resblocks", "resnets"), ("conv_pre", "conv_in"), ("conv_post", "conv_out"),
                     ("act_post", "act_out"), ("downsample.lowpass", "downsample"), (".ups.", ".upsamplers.")):
        key = key.replace(old, new)
    return key


def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    f_sp, min_hz = 200.0 / 3, 1000.0
    return torch.where(f >= min_hz, min_hz / f_sp + torch.log(f / min_hz) / (math.log(6.4) / 27.0), f / f_sp)


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    f_sp, min_hz = 200.0 / 3, 1000.0
    return torch.where(m >= min_hz / f_sp, min_hz * torch.exp(math.log(6.4) / 27.0 * (m - min_hz / f_sp)), f_sp * m)


def mel_filters(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0,
                fmax: float | None = None) -> torch.Tensor:
    """(n_mels, n_fft // 2 + 1) triangular filters on Slaney's mel scale with Slaney (area) normalisation,
    as the LTX audio models use (it reproduces the vocoder's stored filterbank exactly)."""
    fmax = sample_rate / 2 if fmax is None else fmax
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    lo, hi = _hz_to_mel(torch.tensor(fmin, dtype=torch.float64)), _hz_to_mel(torch.tensor(fmax, dtype=torch.float64))
    pts = _mel_to_hz(torch.linspace(float(lo), float(hi), n_mels + 2, dtype=torch.float64))
    ramps = pts[:, None] - freqs[None, :]
    widths = pts[1:] - pts[:-1]
    w = torch.clamp(torch.minimum(-ramps[:-2] / widths[:-1, None], ramps[2:] / widths[1:, None]), min=0)
    return (w * (2.0 / (pts[2:] - pts[:-2]))[:, None]).float()


def log_mel(wav: torch.Tensor) -> torch.Tensor:
    """(B, channels, samples) at 16 kHz -> (B, channels, frames, 64) log-mel spectrogram, the audio VAE's
    input: a causal STFT (1024-sample periodic Hann window, hop 160, left-padded so frame i ends at sample
    160 * (i + 1)), magnitudes, Slaney mel filters 0-8 kHz, natural log clamped at 1e-5."""
    b, c, n = wav.shape
    x = F.pad(wav.reshape(b * c, n).float(), (AUDIO_N_FFT - AUDIO_HOP, 0))
    spec = torch.stft(x, AUDIO_N_FFT, AUDIO_HOP, AUDIO_N_FFT, torch.hann_window(AUDIO_N_FFT, device=x.device),
                      center=False, return_complex=True).abs()  # (B*C, freqs, frames)
    mel = mel_filters(AUDIO_SAMPLE_RATE, AUDIO_N_FFT, AUDIO_MEL_BINS, 0.0, AUDIO_SAMPLE_RATE / 2).to(x.device) @ spec
    return torch.log(mel.clamp(min=1e-5)).reshape(b, c, AUDIO_MEL_BINS, -1).transpose(2, 3)


class AudioCodec(nn.Module):
    """Audio VAE and vocoder. ``encode``: waveform -> packed, normalised latents (frames, 128) as the DiT
    takes them; ``decode``: those latents -> 48 kHz stereo waveform."""

    def __init__(self, vae: nn.Module, vocoder: nn.Module):
        super().__init__()
        self.vae, self.vocoder = vae, vocoder

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, latent_frames_: int) -> torch.Tensor:
        """(channels, samples) at 16 kHz -> (latent_frames_, 128); stereo, cropped or padded to the length."""
        p = next(self.vae.parameters())
        wav = wav.to(p.device, torch.float32)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        z = self.vae.encode(log_mel(wav[None]).to(p.dtype)).latent_dist.mode()  # (1, 8, T, 16)
        packed = z.transpose(1, 2).flatten(2, 3)[0].float()  # (T, 128): channels then mel bins
        packed = (packed - self.vae.latents_mean.float()) / self.vae.latents_std.float()
        if packed.shape[0] < latent_frames_:
            packed = F.pad(packed, (0, 0, 0, latent_frames_ - packed.shape[0]))
        return packed[:latent_frames_]

    @torch.no_grad()
    def decode(self, packed: torch.Tensor) -> torch.Tensor:
        """(frames, 128) -> (2, samples) at 48 kHz in [-1, 1]."""
        p = next(self.vae.parameters())
        x = packed.to(p.device, torch.float32) * self.vae.latents_std.float() + self.vae.latents_mean.float()
        z = x.unflatten(1, (8, -1)).transpose(0, 1)[None].to(p.dtype)  # (1, 8, T, 16)
        mel = self.vae.decode(z, return_dict=False)[0]
        return self.vocoder(mel.to(next(self.vocoder.parameters()).dtype))[0].float()

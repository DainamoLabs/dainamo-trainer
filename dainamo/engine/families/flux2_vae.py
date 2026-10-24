"""The FLUX.2 image autoencoder, shared by the families built on it (MOSS-Image, FLUX.2 Klein, FLUX 3 Image Klein).

An 8x VAE with 32 latent channels, whose latents are packed 2x2 into 128-channel tokens at 1/16 of the image and
normalised by the batch-norm statistics the file carries (``bn.running_mean`` / ``bn.running_var``). Loads
ComfyUI's single file (BFL layout) or a diffusers export into diffusers' ``AutoencoderKLFlux2``; the reference
pipelines run it in fp32, and so does this.

Families keep latents in the packed, normalised form: ``encode`` gives (128, H/16, W/16), the layout the
diffusion models take their image tokens in (``pack_latents``: channel = c * 4 + 2 * dy + dx).
"""

from __future__ import annotations

import gc

import torch

from .. import comfy_io

VAE_FACTOR = 8  # pixels per latent
PATCH = 2  # latents per token side
TOKEN_PIXELS = VAE_FACTOR * PATCH  # 16 pixels per token side
LATENT_CHANNELS = 32
TOKEN_CHANNELS = LATENT_CHANNELS * PATCH * PATCH  # 128


def pack_latents(z: torch.Tensor) -> torch.Tensor:
    """(B, 32, 2h, 2w) -> (B, 128, h, w), channel = c * 4 + 2 * dy + dx."""
    b, c, h, w = z.shape
    return z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpack_latents(z: torch.Tensor) -> torch.Tensor:
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def decoder_channels(sd: dict[str, torch.Tensor]) -> list[int] | None:
    """Decoder widths in encoder order, read from the weights (FLUX.2 VAEs ship full or slim decoders)."""
    bfl = [sd.get(f"decoder.up.{i}.block.0.conv2.weight") for i in range(4)]
    if all(t is not None for t in bfl):
        return [int(t.shape[0]) for t in bfl]
    dif = [sd.get(f"decoder.up_blocks.{i}.resnets.0.conv2.weight") for i in range(4)]
    if all(t is not None for t in dif):
        return [int(t.shape[0]) for t in reversed(dif)]
    return None


class Flux2VAE:
    """The loaded autoencoder on ``device``, in fp32, with its batch-norm statistics ready to apply."""

    def __init__(self, path: str, device: torch.device, log=print):
        from diffusers import AutoencoderKLFlux2
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

        log(f"Loading VAE {path}")
        sd = comfy_io.dequantize_all(comfy_io.load_state_dict(path), torch.float32)
        vae = AutoencoderKLFlux2(decoder_block_out_channels=decoder_channels(sd))
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
        self.device = device
        # the reference runs the VAE in fp32; it's small
        self.vae = vae.to(device, torch.float32).eval().requires_grad_(False)
        eps = self.vae.config.batch_norm_eps
        self.bn_mean = self.vae.bn.running_mean.float().view(1, -1, 1, 1)
        self.bn_std = (self.vae.bn.running_var.float() + eps).sqrt().view(1, -1, 1, 1)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels (1, 3, H, W) in [-1, 1] -> packed, normalised latents (128, H/16, W/16), fp32 on the CPU."""
        z = self.vae.encode(pixels.to(self.device, torch.float32)).latent_dist.mode()
        z = (pack_latents(z) - self.bn_mean) / self.bn_std
        return z[0].to("cpu", torch.float32)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """packed latents (B, 128, h, w) -> pixels (B, 3, 16h, 16w) in [-1, 1], fp32."""
        z = latents.to(self.device, torch.float32) * self.bn_std + self.bn_mean
        return self.vae.decode(unpack_latents(z)).sample.float().clamp(-1, 1)


def unload(vae: Flux2VAE | None) -> None:
    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

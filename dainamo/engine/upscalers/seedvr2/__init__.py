"""SeedVR2 (ByteDance's one-step diffusion video restorer, the 3B release) as an upscaler: ``model`` (the DiT),
``vae`` (its causal video VAE), ``pipeline`` (``SeedVR2Upscaler``, the ``engine.upscale.Upscaler``)."""

from .pipeline import SeedVR2Upscaler

__all__ = ["SeedVR2Upscaler"]

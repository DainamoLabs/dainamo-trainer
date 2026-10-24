"""YuE2 driver: music LoRAs on the acoustic (NAR) expert; songs from style tags and lyrics.

Loads ComfyUI's single-file checkpoint (bf16 or int8 convrot), which holds everything: the composer (the AR
expert, which writes a score and semantic tokens), the acoustic expert (the NAR, flow matching on VAE
latents), the 48 kHz stereo VAE and the tokenizer (see ``model``).

Only the NAR trains. The composer can't learn from recordings, since the tokenizer that turns audio into its
semantic tokens isn't released, and so it stays frozen and acts as the text encoder: each condition is
cached as the composer's keys and values over YuE2's prompt with planning off and no semantic tokens
(style tags, and lyrics when whole songs are trained). That is the paper's "direct audio generation" task,
one of the four the model was trained on, so the LoRA learns the sound of the songs: their style,
instruments and timbre. The melody and the lyrics' timing come from the frozen composer when it samples.

Training is the release's flow matching: z_t = (1 - t) z_0 + t noise, the velocity target noise - z_0, and
t logit-normal (a standard-normal logit) unless set to uniform. Samples run the full release pipeline: an
optional score, the semantic tokens, then 32 midpoint steps of the NAR (with the LoRA) and the VAE.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ... import comfy_io
from ... import dataset as ds
from ...fp8_linear import assert_materialized, materialize
from ..base import Download, FamilyDriver, FamilyInfo, MemoryProfile, ProfileField, Workflow, count_blocks
from . import model as M

# where the model manager downloads the files from, at the commit they were checked against
HF_REPO, HF_REV = "Comfy-Org/YuE2", "2f76ca75e6ee094169de899cc7fc99d6887e2196"

MAX_SECONDS = 360  # longest song or clip (the release's 9000 semantic tokens)
MIN_SECONDS = 5
# A LoRA learns in the regime it trains in (no semantic tokens) and acts at about half that strength once the
# composer's tokens condition the song, as they always do in generation: a telephone-effect LoRA took the
# low end (under 250 Hz) of text-only songs from 24-56% of their energy to under 0.3%, but of normal songs only
# from 66-74% to 38-68% at strength 1, and to 2-7% at 2. So its samples, and the Sandbox, start at 2.
LORA_STRENGTH = 2.0


def song_condition(style: str, lyrics: str = "", **extra) -> str:
    """The cache key ("prompt") of a YuE2 condition: style tags and lyrics, plus a sample's length."""
    return json.dumps({"style": style, "lyrics": lyrics, **extra}, sort_keys=True)


def parse_condition(prompt: str) -> dict:
    """``song_condition``'s fields; a plain prompt (caption dropout's "", a negative prompt) is style only."""
    if prompt.startswith("{"):
        try:
            return json.loads(prompt)
        except json.JSONDecodeError:
            pass
    return {"style": prompt, "lyrics": ""}


class YuE2Driver(FamilyDriver):
    info = FamilyInfo(
        id="yue2",
        label="YuE2 (music)",
        media=["audio"],
        profile_fields=[
            ProfileField("diffusion_model", "Checkpoint", hint="YuE2 3B (ComfyUI's single-file checkpoint)",
                         help="yue2_3b_bf16 or yue2_3b_int8_convrot (Comfy-Org/YuE2): the composer, the acoustic "
                              "model, the VAE and the tokenizer in one file."),
            ProfileField("sample_cot", "Score planning", kind="select", options=["off", "melody", "full"],
                         default="off", required=False,
                         help="Off: semantic tokens straight from the prompt, the closest match to how the LoRA "
                              "trains. Full writes a melody-and-chords score first (the release's default), "
                              "melody a melody-only one; both take longer."),
            ProfileField("sample_steps", "Sample steps", kind="number", required=False, default=M.ODE_STEPS,
                         help="Flow-matching (midpoint) steps of the acoustic model; the release uses 32."),
            ProfileField("sample_cfg", "Sampling CFG", kind="number", required=False,
                         help="Guidance on the semantic tokens. Blank: the release's 1.01 with planning off, else 1 "
                              "(off). Anything but 1 runs the composer twice per token."),
        ],
        training_defaults={
            "rank": 32, "alpha": 32, "learning_rate": 1e-4, "steps": 2000, "batch_size": 1, "offload": 0,
            "optimizer": "adamw8bit", "timestep_sampling": "logit_normal", "clip_seconds": 30, "sample_seconds": 30,
            "caption_dropout": 0.05,
        },
        timestep_modes=[("logit_normal", "Logit-normal"), ("uniform", "Uniform")],
        workflows=[Workflow("t2a")],
        lora_strength=LORA_STRENGTH,
        streams_text_encoder=False,
    )
    block_list = "model.layers"
    lora_targets = ["model.layers.*.self_attn.qkv_proj", "model.layers.*.self_attn.o_proj",
                    "model.layers.*.mlp.gate_up_proj", "model.layers.*.mlp.down_proj"]
    lora_metadata = {"modelspec.architecture": "yue2/lora", "dainamo.family": "yue2",
                     "dainamo.recommended_strength": str(LORA_STRENGTH)}
    signature_keys = {"diffusion_model": ["model.diffusion_model.llm2vae.weight", "text_encoders.yue2_tokenizer_json",
                                          "vae.decoder.layers.0.weight_v"]}
    downloads = [
        Download("diffusion_model", HF_REPO, "checkpoints/yue2_3b_int8_convrot.safetensors", 3_960_938_800,
                 "int8 convrot", HF_REV, recommended=True,
                 sha256="96fe199377309001ed8cd26a944baeee8cc31a20ba7c36d1d3c0a7e1f4149db6"),
        Download("diffusion_model", HF_REPO, "checkpoints/yue2_3b_bf16.safetensors", 7_799_983_228, "bf16", HF_REV,
                 sha256="33765adbf9813c9a50318218760b2fd819a319862460a04884607581961c6fee"),
    ]
    supports_samplers = False  # the release's midpoint ODE
    fps = M.FRAME_RATE  # latent frames a second
    # measured with scripts/measure_memory.py on the int8 convrot checkpoint (10-60 s clips); VAE costs are per
    # latent frame, its encode in 750-frame chunks and decode in 1024-frame tiles, each with a halo
    memory = MemoryProfile(saved=4_600, full=133_000, work=150_600, infer=100_800,
                           vae_encode=(0, 2_490_000), vae_decode=(0, 2_230_000), decode_tile_pixels=1056,
                           encode_tile_pixels=878, text_tokens=64)
    # the composer samples in bf16 (quantized layers expanded); the VAE runs in fp32
    bundled = {"text_encoder": ("text_encoders.model.", 2), "vae": ("vae.", 4)}

    def __init__(self, profile: dict, device: torch.device, log=print):
        super().__init__(profile, device, log)
        self.tokenizer: M.TextTokenizer | None = None
        self.text_encoder: M.Composer | None = None  # the composer (the Sandbox parks it in RAM between runs)
        self.vae: M.VAE | None = None
        # the checkpoint each part ("tokenizer", "composer", "vae") was loaded from: the Sandbox may point the profile
        # at another one (``Session.use_model``), whose parts are loaded in their place
        self.loaded_from: dict[str, str] = {}

    # --- settings -------------------------------------------------------------------------------------
    def cot(self) -> str:
        v = str(self.profile_value(self.profile, "sample_cot", "off") or "off")
        return v if v in M.INSTRUCTIONS else "off"

    def cfg_scale(self, profile_cfg) -> float:
        if profile_cfg in (None, ""):
            return 1.01 if self.cot() == "off" else 1.0
        return float(profile_cfg)

    @classmethod
    def sequence_tokens(cls, width, height, frames=1):
        """The clip's latent frames (``frames``, 25 a second) with its start and end, plus a typical prompt."""
        return frames + 2 + cls.memory.text_tokens

    # --- the composer (text encoder) ------------------------------------------------------------------
    def _tokenizer(self) -> M.TextTokenizer:
        path = self.profile["diffusion_model"]
        if self.tokenizer is None or self.loaded_from.get("tokenizer") != path:
            with safe_open(path, "pt") as f:
                blob = f.get_tensor("text_encoders.yue2_tokenizer_json").numpy().tobytes().decode("utf-8")
            self.tokenizer = M.TextTokenizer(blob)
            self.loaded_from["tokenizer"] = path
        return self.tokenizer

    def load_text_encoder(self):
        """The composer on the GPU, expanded to bf16 (it samples token by token, where expanding quantized
        weights per token would dominate)."""
        path = self.profile["diffusion_model"]
        if self.text_encoder is not None:
            if self.loaded_from.get("composer") == path:
                self.text_encoder.to(self.device)
                return
            self.unload_text_encoder()  # another checkpoint's: freed before this one's loads
        self.log(f"Loading the composer from {path}")
        self._tokenizer()
        with torch.device("meta"):
            comp = M.Composer()
        weights = comfy_io.load_state_dict(path, strip_prefix="text_encoders.", only_prefixed=True,
                                           include=("model.",), device=self.device)
        materialize(comp, weights, self.dtype, keep_fp8=False, device=self.device)
        del weights
        assert_materialized(comp)
        self.text_encoder = comp.eval()
        self.loaded_from["composer"] = path

    def training_prefix(self, style: str, lyrics: str) -> list[int]:
        """The conditioning a training item is shown: planning off, no semantic tokens."""
        return M.prefix_ids(self._tokenizer(), "off", style, lyrics) + [M.MUSIC_END]

    def encode_condition(self, prompt: str, images):
        c = parse_condition(prompt)
        if c.get("sample"):  # a sample card: the composer runs when it's sampled
            return c
        k, v = self.text_encoder.keys_values(self.training_prefix(c.get("style", ""), c.get("lyrics", "")), self.device)
        return {"k": k.cpu(), "v": v.cpu()}

    def encode_prompt(self, prompt: str):
        return self.encode_condition(prompt, None)

    def item_conditions(self, item, settings):
        return {"text": (song_condition(item.caption, item.lyrics), None)}

    def sample_condition(self, spec):
        return song_condition(spec["prompt"], spec.get("lyrics", ""), seconds=spec.get("seconds", 30),
                              sample=True), None

    # --- vae ------------------------------------------------------------------------------------------
    def cache_inputs(self, kind: str, settings: dict) -> dict:
        return {"files": [self.profile["diffusion_model"]]}  # the composer and the VAE are in the checkpoint

    def load_vae(self):
        path = self.profile["diffusion_model"]
        if self.vae is not None and self.loaded_from.get("vae") == path:
            return
        self.vae = None  # another checkpoint's
        weights = comfy_io.load_state_dict(path, strip_prefix="vae.", only_prefixed=True, device=self.device).tensors
        with torch.device("meta"):
            vae = M.VAE()
        vae.load_state_dict(M.fold_weight_norm(weights), assign=True)
        del weights
        self.vae = vae.to(self.device).eval().requires_grad_(False)
        assert_materialized(self.vae)
        self.loaded_from["vae"] = path

    def unload_vae(self):
        """With the VAE goes the composer, which sampling loaded: training continues without either."""
        self.vae = None
        self.unload_text_encoder()

    @torch.no_grad()
    def encode_item(self, item, settings):
        wav = ds.read_audio(item.path, item.seconds, M.SAMPLE_RATE, 2, start=item.start)
        if item.flip:  # swapped stereo channels
            wav = wav.flip(0)
        return {"latent": self.vae.encode(wav).cpu()}

    # --- the acoustic model -------------------------------------------------------------------------------
    def build_transformer(self, shapes):
        return M.NAR(count_blocks(shapes, self.block_list))

    def _conds(self, conds: list[dict]):
        """Per-item conditioning keys and values -> padded (L, B, n, 8, 128) on the GPU, and the lengths."""
        lengths = [c["k"].shape[1] for c in conds]
        n = max(lengths)

        def pad(t):
            return F.pad(t, (0, 0, 0, 0, 0, n - t.shape[1]))

        k = torch.stack([pad(c["k"]) for c in conds], dim=1).to(self.device, self.dtype)
        v = torch.stack([pad(c["v"]) for c in conds], dim=1).to(self.device, self.dtype)
        return k, v, torch.tensor(lengths, device=self.device)

    def sample_timesteps(self, batch, latent_hw, settings):
        if settings.get("timestep_sampling", "logit_normal") == "uniform":
            return torch.rand(batch).clamp(1e-3, 1 - 1e-3)
        return torch.sigmoid(torch.randn(batch))

    def training_loss(self, model, payloads, conds, settings, rng=None, sigma=None, generator=None,
                      deterministic=False):
        d = self.device
        x0 = torch.stack([p["latent"] for p in payloads]).to(d, torch.float32)  # (B, T, 64)
        b = x0.shape[0]
        noise = torch.randn(x0.shape, generator=generator).to(d) if generator is not None else torch.randn_like(x0)
        t = torch.full((b,), float(sigma)) if sigma is not None else self.timesteps(b, None, settings)
        t = t.to(d)
        noisy = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * noise
        k, v, n = self._conds([c["text"] for c in conds])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(noisy, t, k, v, n)
        w = self.step_weights(t, settings, sigma is not None)  # one noise level per clip; songs have no masks
        if w is None:
            return F.mse_loss(pred.float(), noise - x0)
        return (self.per_item_mean((pred.float() - (noise - x0)) ** 2) * w).mean()

    # --- sampling ---------------------------------------------------------------------------------------
    def sample_steps(self) -> int:
        return int(self.profile_value(self.profile, "sample_steps", M.ODE_STEPS))

    @torch.no_grad()
    def sample(self, model, cond, uncond, width, height, steps, cfg, seed, frames=1, images=None, sampler="euler",
               scheduler="default", shift=None, callback=None):
        """A song from a sample condition ({style, lyrics, seconds}): the composer writes the score (if planning
        is on) and the semantic tokens, then the NAR (``model``, with any LoRA) renders them. -> latents for
        ``to_media``. ``callback(step, total)`` follows the acoustic model's steps."""
        d = self.device
        self.load_text_encoder()
        tok, comp, cot = self._tokenizer(), self.text_encoder, self.cot()
        style, lyrics = cond.get("style", ""), cond.get("lyrics", "")
        seconds = min(MAX_SECONDS, max(1.0, float(cond.get("seconds") or 30)))
        prefix = M.prefix_ids(tok, cot, style, lyrics)
        abc = None
        if cot != "off":
            abc, _ = comp.generate(prefix, M.ABC_SAMPLING, seed, "abc", d, callback=self._noting(
                lambda n, _: f"Writing the score: {n} tokens"))
            prefix = M.prefix_ids(tok, cot, style, lyrics, abc)
        frames = min(round(seconds * M.FRAME_RATE), M.max_frames(len(prefix) + 1))
        negative = M.negative_ids(tok, cot, abc) if cfg != 1 else None
        codes, _ = comp.generate(prefix, M.SEMANTIC_SAMPLING, seed, "semantic", d, negative=negative, cfg=cfg,
                                 legacy_off=cot == "off", max_tokens=frames, callback=self._noting(
                                     lambda n, total: f"Writing the song: {n // M.FRAME_RATE} of "
                                                      f"{total // M.FRAME_RATE} s"))
        if not codes:
            raise RuntimeError("The composer ended the song before writing any of it")
        ids = prefix + codes + [M.MUSIC_END]
        k, v = comp.keys_values(ids, d)
        k, v, n = k.unsqueeze(1), v.unsqueeze(1), torch.tensor([len(ids)], device=d)
        noise = torch.randn(len(codes), M.LATENT_DIM, generator=torch.Generator().manual_seed(int(seed)))

        def velocity(x, t):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(x.unsqueeze(0), torch.full((1,), t, device=d), k, v, n)[0]

        lat = M.solve(velocity, noise.to(d), int(steps) or M.ODE_STEPS, callback)
        return {"latent": lat.cpu(), "score": tok.decode(abc) if abc else None}

    def _noting(self, text):
        """A composer callback passing ``text(done, total)`` to ``progress_note`` every second's worth of tokens."""
        if self.progress_note is None:
            return None
        return lambda done, total: self.progress_note(text(done, total)) if done % M.FRAME_RATE == 0 else None

    @torch.no_grad()
    def to_media(self, latents) -> list:
        wav = self.vae.decode(latents["latent"]).clamp(-1, 1).numpy()
        return [{"audio": wav, "sample_rate": M.SAMPLE_RATE, "score": latents.get("score")}]

    def decode_latents(self, latents):
        raise NotImplementedError("YuE2 decodes through to_media")

"""YuE2 (M-A-P): the AR-NAR Mixture-of-Transformers and its Oobleck VAE, laid out as ComfyUI's single-file
checkpoint stores them:

  text_encoders.model.*              the AR expert, the "composer": token embedding, 28 layers, norm, LM head
  model.diffusion_model.*            the NAR expert: 28 layers, norm, the latent projections in and out, and the
                                     timestep and frame-position embeddings
  vae.*                              the 48 kHz stereo Oobleck VAE: 25 latent frames a second, 64 channels
  text_encoders.yue2_tokenizer_json  the text tokenizer

Each layer of the released model has an AR and a NAR expert around one shared attention. AR positions never
attend to NAR ones, so the AR experts' keys and values over the conditioning (text, then the score and
semantic tokens when there are any) don't depend on the audio, and a NAR layer attends to them plus its own
positions. Those keys and values are all the conditioning the NAR (the part a LoRA trains) gets, so they
are computed once per prompt, like a text encoder's output. ComfyUI's file fuses q/k/v into ``qkv_proj``
and gate/up into ``gate_up_proj``.

Ported from M-A-P's yue2 package (https://github.com/multimodal-art-projection/YuE, Apache-2.0):
``modeling_yue2.py`` (the experts and embeddings), ``nar.py`` (the prefill and the midpoint ODE),
``protocol.py`` (the prompt format and token ids) and ``sampling.py`` (the token sampler). The VAE follows
``modeling_vae.py``, whose Oobleck and SnakeBeta come from stable-audio-tools (Stability AI, MIT) and
BigVGAN (NVIDIA, MIT). See THIRD_PARTY_NOTICES.md.

Changes for Dainamo: plain ``nn.Module``s (no transformers classes) laid out as ComfyUI's file is, with its
fused ``qkv_proj`` and ``gate_up_proj``, so the checkpoint loads as-is and LoRA keys match its weight names;
the composer's keys and values are a separate step (``Composer.keys_values``) so they can be cached as
conditioning, and the NAR takes them padded, with their lengths, for batches of several prompts; the VAE
folds its weight norm at load and encodes long songs in overlapping chunks (decoding is tiled as in the release).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# token ids (protocol.py)
EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END = 184621, 184622
CONTEXT = 24576  # positions: conditioning tokens + latent frames + 2
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec "
              "tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given "
            "conditions.",
}

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, MLP_DIM = 2048, 28, 16, 8, 128, 6144
EPS, ROPE_THETA = 1e-6, 1e6
LATENT_DIM = 64
SAMPLE_RATE = 48000
HOP = 1920  # audio samples per latent frame
FRAME_RATE = SAMPLE_RATE // HOP  # 25 latent frames (and semantic tokens) a second
ODE_STEPS = 32


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000


ABC_SAMPLING = Sampling(0.7, 0.9, 30, 1.005, 100, 32, 4096)
SEMANTIC_SAMPLING = Sampling()


# --- prompts -----------------------------------------------------------------------------------------
class TextTokenizer:
    """The text and ABC tokenizer (Qwen's BPE), from the JSON embedded in ComfyUI's file."""

    def __init__(self, tokenizer_json: str):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_str(tokenizer_json)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids if 0 <= int(i) < EOD])


def song_text(cot: str, style: str, lyrics: str) -> str:
    return f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"


def prefix_ids(tokenizer: TextTokenizer, cot: str, style: str, lyrics: str, abc_ids=None) -> list[int]:
    """The AR prefix up to where the composer writes: semantic tokens with ``cot`` "off" (or a given
    score), else the score."""
    base = [EOD] + tokenizer.encode(song_text(cot, style, lyrics))
    if cot == "off":
        return base + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        return base + [ABC_START]
    return base + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def negative_ids(tokenizer: TextTokenizer, cot: str, abc_ids=None) -> list[int]:
    """The CFG branch: the instruction alone (and the same score)."""
    base = [EOD] + tokenizer.encode(INSTRUCTIONS[cot])
    if cot == "off":
        return base + [MUSIC_START]
    return base + [ABC_START] + list(abc_ids or []) + [ABC_END, MUSIC_START]


def max_frames(prefix_len: int) -> int:
    """Latent frames one NAR pass takes after a conditioning of ``prefix_len`` tokens (nar.chunk_ranges)."""
    return (CONTEXT - prefix_len - 3) // 2


# --- shared pieces -------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS).to(x.dtype) * self.weight


def rope(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """positions (..., T) -> cos, sin (..., T, HEAD_DIM / 2) in fp32."""
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=positions.device) / HEAD_DIM)
    angles = positions.float().unsqueeze(-1) * inv
    return angles.cos(), angles.sin()


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def qkv(self, x, cos, sin):
        """x (B, T, HIDDEN), cos/sin (B, T, HEAD_DIM / 2) -> q (B, T, 16, 128), k and v (B, T, 8, 128)."""
        b, t, _ = x.shape
        q, k, v = self.qkv_proj(x).split([HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
        q = self.q_norm(q.reshape(b, t, HEADS, HEAD_DIM))
        k = self.k_norm(k.reshape(b, t, KV_HEADS, HEAD_DIM))
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        return _rotate(q, cos, sin), _rotate(k, cos, sin), v.reshape(b, t, KV_HEADS, HEAD_DIM)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(HIDDEN, 2 * MLP_DIM, bias=False)
        self.down_proj = nn.Linear(MLP_DIM, HIDDEN, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def _attend(q, k, v, mask=None, causal=False):
    """(B, T, H, D) queries over (B, S, KV, D) keys and values -> (B, T, H * D)."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype),
                                         attn_mask=mask, is_causal=causal, enable_gqa=True)
    return out.transpose(1, 2).flatten(2)


class Layer(nn.Module):
    """One expert's half of a layer (the AR's or the NAR's): its norms, projections and MLP."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN)
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.mlp = MLP()

    def forward(self, x, ar_k, ar_v, cos, sin, mask=None):
        """NAR: the latent positions attend to the conditioning's AR keys and values plus each other."""
        q, k, v = self.self_attn.qkv(self.input_layernorm(x), cos, sin)
        k, v = torch.cat([ar_k.to(k.dtype), k], dim=1), torch.cat([ar_v.to(v.dtype), v], dim=1)
        x = x + self.self_attn.o_proj(_attend(q, k, v, mask))
        return x + self.mlp(self.post_attention_layernorm(x))


class Stack(nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(layers))
        self.norm = RMSNorm(HIDDEN)


# --- the NAR expert (what a LoRA trains) ----------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))

    def forward(self, t):
        """t (B,) flow time in [0, 1] -> (B, HIDDEN)."""
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1).to(torch.bfloat16))


class PositionTable(nn.Module):
    """The fixed sinusoidal frame-position embedding, stored in the file."""

    def __init__(self):
        super().__init__()
        self.register_buffer("pe", torch.empty(CONTEXT, HIDDEN))


def model_time(t: torch.Tensor) -> torch.Tensor:
    """The flow time the NAR is shown: the release passes it as a logit (clamped to +-20) through a bf16
    sigmoid (``_shift_t_value``, shift 1), so it arrives rounded to bf16."""
    raw = torch.logit(t.double()).clamp(-20, 20)
    return torch.sigmoid(raw.to(torch.bfloat16)).float()


class NAR(nn.Module):
    """The NAR expert (``model.diffusion_model.*``): predicts the flow velocity of the latent frames."""

    def __init__(self, layers: int = LAYERS):
        super().__init__()
        self.model = Stack(layers)
        self.llm2vae = nn.Linear(HIDDEN, LATENT_DIM)
        self.vae2llm = nn.Linear(LATENT_DIM, HIDDEN)
        self.time_embedder = TimestepEmbedder()
        self.latent_pos_embed = PositionTable()
        self.gradient_checkpointing = False

    def forward(self, x, t, ar_k, ar_v, ar_len):
        """x (B, T, 64) noisy latents; t (B,) flow time; ar_k / ar_v (L, B, n, 8, 128): each item's conditioning
        keys and values, padded to the longest (n); ar_len (B,) their lengths. -> velocity (B, T, 64)."""
        b, frames, _ = x.shape
        n_nar = frames + 2  # LATENT_START, the frames, LATENT_END: the start and end get a zero latent
        dtype = torch.bfloat16
        h = self.vae2llm(F.pad(x, (0, 0, 1, 1)).to(dtype))
        h = h + self.time_embedder(model_time(t)).unsqueeze(1).to(h.dtype)
        h = h + self.latent_pos_embed.pe[:n_nar].to(h.dtype)
        ar_len = torch.as_tensor(ar_len, device=x.device)
        positions = ar_len.view(-1, 1) + torch.arange(n_nar, device=x.device)
        cos, sin = rope(positions)
        n = ar_k.shape[2]
        mask = None
        if bool((ar_len < n).any()):  # padded conditioning: hide the padding
            keys = torch.arange(n + n_nar, device=x.device)
            mask = ((keys < ar_len.view(-1, 1)) | (keys >= n)).view(b, 1, 1, -1)
        ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for i, layer in enumerate(self.model.layers):
            if ckpt:
                h = checkpoint(layer, h, ar_k[i], ar_v[i], cos, sin, mask, use_reentrant=False)
            else:
                h = layer(h, ar_k[i], ar_v[i], cos, sin, mask)
        return self.llm2vae(self.model.norm(h))[:, 1:-1].float()


def solve(velocity, noise: torch.Tensor, steps: int = ODE_STEPS, callback=None) -> torch.Tensor:
    """The release's sampler: flow time 1 -> 0 in ``steps`` equal midpoint steps. ``velocity(x, t)`` with t a
    float; ``callback(step, total)`` after each step."""
    x = noise.float()
    dt = 1.0 / steps
    for step in range(steps):
        t = 1.0 - step * dt
        mid = x - velocity(x, t) * (dt / 2)
        x = x - velocity(mid, t - dt / 2) * dt
        if callback is not None:
            callback(step + 1, steps)
    return x


# --- the AR expert (the composer) -------------------------------------------------------------------------
class ComposerStack(Stack):
    def __init__(self, layers: int):
        super().__init__(layers)
        self.embed_tokens = nn.Embedding(184704, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, 184704, bias=False)


class KVCache:
    def __init__(self, layers: int, capacity: int, device, dtype=torch.bfloat16):
        self.k = [torch.empty(1, capacity, KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for _ in range(layers)]
        self.v = [torch.empty_like(t) for t in self.k]
        self.n = 0


class Composer(nn.Module):
    """The AR expert (``text_encoders.*``): writes the score and the semantic tokens, and gives the NAR its
    conditioning keys and values."""

    def __init__(self, layers: int = LAYERS):
        super().__init__()
        self.model = ComposerStack(layers)

    def _run(self, ids: torch.Tensor, cache: KVCache | None, collect: bool = False):
        """ids (1, T) after ``cache.n`` cached tokens (or from 0 without a cache) -> the last hidden state, and
        each layer's keys and values when ``collect``."""
        m = self.model
        start = cache.n if cache is not None else 0
        t = ids.shape[1]
        if t > 1 and start:
            raise ValueError("a prefill must start the cache")
        x = m.embed_tokens(ids)
        cos, sin = rope(torch.arange(start, start + t, device=ids.device).unsqueeze(0))
        kvs = []
        for i, layer in enumerate(m.layers):
            q, k, v = layer.self_attn.qkv(layer.input_layernorm(x), cos, sin)
            if cache is not None:
                cache.k[i][:, start:start + t] = k
                cache.v[i][:, start:start + t] = v
                k, v = cache.k[i][:, :start + t], cache.v[i][:, :start + t]
            if collect:
                kvs.append((k[0], v[0]))
            x = x + layer.self_attn.o_proj(_attend(q, k, v, causal=t > 1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if cache is not None:
            cache.n = start + t
        return x[:, -1:], kvs

    @torch.no_grad()
    def keys_values(self, ids: list[int], device) -> tuple[torch.Tensor, torch.Tensor]:
        """The conditioning ``ids`` -> keys and values (L, n, 8, 128) for the NAR (``nar.CachedNAR._prefill``)."""
        _, kvs = self._run(torch.tensor([ids], device=device), None, collect=True)
        return torch.stack([k for k, _ in kvs]), torch.stack([v for _, v in kvs])

    def logits(self, hidden):
        return self.model.lm_head(self.model.norm(hidden))[:, -1]

    @torch.no_grad()
    def generate(self, prefix: list[int], sampling: Sampling, seed: int, phase: str, device, negative=None,
                 cfg: float = 1.0, legacy_off: bool = False, max_tokens: int | None = None, callback=None):
        """Sample the score (``phase`` "abc") or the semantic tokens ("semantic") after ``prefix``, as the
        release does (``sampling.generate_tokens``). -> (tokens without the end token, truncated)."""
        limit = min(sampling.max_tokens, max_tokens or sampling.max_tokens)
        if len(prefix) + limit > CONTEXT:
            raise ValueError("The prompt and the requested length don't fit in YuE2's 24576-token context")
        g = torch.Generator(device=device).manual_seed(int(seed))
        n_layers = len(self.model.layers)

        def prefill(ids):
            cache = KVCache(n_layers, len(ids) + limit, device)
            return self.logits(self._run(torch.tensor([ids], device=device), cache)[0]), cache

        cond, cache = prefill(prefix)
        uncond, neg_cache = prefill(negative) if cfg != 1.0 else (None, None)
        end = ABC_END if phase == "abc" else MUSIC_END
        out, done = [], False
        for step in range(limit):
            logits = cond if uncond is None else uncond + cfg * (cond - uncond)
            scores = distribution(logits, sampling, out, step, phase, legacy_off)
            if sampling.temperature == 0:
                token = int(scores.argmax(-1))
            else:
                token = int(torch.multinomial(scores.softmax(-1), 1, generator=g))
            if token == end:
                done = True
                break
            out.append(token)
            if callback is not None:
                callback(len(out), limit)
            if step + 1 < limit:
                nxt = torch.tensor([[token]], device=device)
                cond = self.logits(self._run(nxt, cache)[0])
                if neg_cache is not None:
                    uncond = self.logits(self._run(nxt, neg_cache)[0])
        return out, not done


def window_penalty(logits, recent, penalty):
    if penalty == 1.0 or not recent:
        return logits
    ids = torch.as_tensor(recent, dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, ids, torch.ones_like(ids, dtype=logits.dtype))
    alpha = penalty ** freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def distribution(logits, sampling: Sampling, history, step, phase, legacy_off=False):
    """Next-token scores: only the phase's own vocabulary and end token, no end before ``min_tokens``, a
    repetition penalty over the last ``penalty_window`` tokens, then temperature, top-k and top-p (with
    ``legacy_off`` the release's bf16 arithmetic for ``cot`` off)."""
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-sampling.penalty_window:], sampling.repetition_penalty)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probs = values.softmax(-1)
        removed = probs.cumsum(-1) - probs > sampling.top_p
        removed[..., :3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


# --- the Oobleck VAE ------------------------------------------------------------------------------------------
class Snake(nn.Module):
    """SnakeBeta with log-scale alpha and beta: x + sin^2(x e^alpha) / e^beta."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp().view(1, -1, 1)
        beta = self.beta.exp().view(1, -1, 1)
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


def _conv(cin, cout, k, stride=1, dilation=1, padding=0, bias=True):
    return nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation, padding=padding, bias=bias)


class Residual(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(Snake(ch), _conv(ch, ch, 7, dilation=dilation, padding=dilation * 3),
                                    Snake(ch), _conv(ch, ch, 1))

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.layers = nn.Sequential(Residual(cin, 1), Residual(cin, 3), Residual(cin, 9), Snake(cin),
                                    _conv(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2)))

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.layers = nn.Sequential(Snake(cin), nn.ConvTranspose1d(cin, cout, 2 * stride, stride=stride,
                                                                   padding=math.ceil(stride / 2)),
                                    Residual(cout, 1), Residual(cout, 3), Residual(cout, 9))

    def forward(self, x):
        return self.layers(x)


CHANNELS, MULTS, STRIDES = 64, (1, 1, 2, 4, 8, 16, 32), (2, 2, 4, 4, 5, 6)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [CHANNELS * m for m in MULTS]
        layers = [_conv(2, ch[0], 7, padding=3)]
        layers += [EncoderBlock(ch[i], ch[i + 1], s) for i, s in enumerate(STRIDES)]
        layers += [Snake(ch[-1]), _conv(ch[-1], 2 * LATENT_DIM, 3, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [CHANNELS * m for m in MULTS]
        layers = [_conv(LATENT_DIM, ch[-1], 7, padding=3)]
        layers += [DecoderBlock(ch[i + 1], ch[i], STRIDES[i]) for i in reversed(range(len(STRIDES)))]
        layers += [Snake(ch[0]), _conv(ch[0], 2, 7, padding=3, bias=False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def fold_weight_norm(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``weight_g`` / ``weight_v`` pairs (torch's weight_norm over dim 0) -> plain ``weight``s, in fp32."""
    out = {}
    for k, t in sd.items():
        if k.endswith(".weight_v"):
            base = k[: -len("_v")]
            g, v = sd[base + "_g"].float(), t.float()
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[base] = g * v / norm
        elif not k.endswith(".weight_g"):
            out[k] = t.float()
    return out


class VAE(nn.Module):
    """The 48 kHz stereo Oobleck VAE (fp32, as the release requires)."""

    halo = 16  # latent frames of context each side of a decoded tile (the release's decode_halo_frames)

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, chunk_frames: int = 750, halo_frames: int = 64) -> torch.Tensor:
        """audio (2, samples) at 48 kHz -> posterior mean (frames, 64). Long audio is encoded in overlapping
        chunks, each cropped to its core."""
        dev = next(self.parameters()).device
        x = audio.to(dev, torch.float32).unsqueeze(0)
        frames = x.shape[-1] // HOP
        x = x[..., : frames * HOP]
        if frames <= chunk_frames + 2 * halo_frames:
            return self.encoder(x)[0, :LATENT_DIM].T.contiguous()
        out = []
        for start in range(0, frames, chunk_frames):
            end = min(frames, start + chunk_frames)
            left, right = max(0, start - halo_frames), min(frames, end + halo_frames)
            z = self.encoder(x[..., left * HOP: right * HOP])[0, :LATENT_DIM]
            out.append(z[:, start - left: start - left + end - start])
        return torch.cat(out, dim=1).T.contiguous()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, core_frames: int = 1024) -> torch.Tensor:
        """latents (frames, 64) -> audio (2, samples) on the CPU, decoded in tiles of ``core_frames`` with
        ``halo`` frames of context each side, cropped to their cores (``decode_tiled``)."""
        dev = next(self.parameters()).device
        z = latents.to(dev, torch.float32).T.unsqueeze(0)
        frames = z.shape[-1]
        tiles = []
        for start in range(0, frames, core_frames):
            end = min(frames, start + core_frames)
            left, right = max(0, start - self.halo), min(frames, end + self.halo)
            audio = self.decoder(z[..., left:right])[0]
            crop = audio[:, (start - left) * HOP:]
            tiles.append(crop[:, : (end - start) * HOP] if end < frames else crop)
        return torch.cat(tiles, dim=1).cpu()

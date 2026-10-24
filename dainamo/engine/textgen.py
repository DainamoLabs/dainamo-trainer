"""Captions and prompts from a model's text encoder (Settings → Caption provider → Text encoder).

The text encoders of Qwen-Image 2.1 (Qwen3-VL-8B), Krea 2 (Qwen3-VL-4B) and MOSS Image (Qwen3.5-9B) are whole
chat models with their vision part: a diffusion model reads one of their layers for its conditioning, but the
file also has the output layer (or shares it with the input embeddings), so the encoder a profile already has can
caption a dataset or rewrite a prompt with nothing else to download. Two can't: MiniMax H3's is cut short, and
LTX 2.5's is Lightricks' custom Gemma 4, changed for LTX so that it no longer predicts text (no layer of it does).
A Gemma 4 file that still does (Gemma4UnifiedForConditionalGeneration) loads too. Every encoder is checked once
it's loaded: one that can't continue a plain English sentence is refused rather than left to write gibberish.
FLUX 3 Image Klein's prompt upsampler is a Qwen3.5-9B too (the same layout, tokenizer and chat template as MOSS
Image's): it isn't a text encoder, but it wrote the captions FLUX 3 was trained on, and loads the same way. Its
header doesn't say which it is, so it's told by its file name.

    python -m dainamo.engine.textgen --model FILE --port 8090

serves it with the requests llama-server answers (/health: 503 while the model loads; /v1/models;
/v1/chat/completions, with images as data: URLs), so Dainamo runs it like llama-server
(dainamo/server/llama_server.py) and the captioner talks to it the same way. It generates with transformers'
``generate`` on the engine's loaders, on the one GPU CUDA_VISIBLE_DEVICES leaves it. int8/fp8 layers are expanded
to bf16 while the GPU has room: a quantized layer is expanded for every token otherwise, about 3x slower.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SUPPORTED = ("It writes text with the text encoders of Qwen-Image 2.1 (Qwen3-VL-8B), Krea 2 (Qwen3-VL-4B) and "
             "MOSS Image (Qwen3.5-9B).")
# (vocabulary, hidden size, layers) of the Qwen encoders: (kind, name, whose, output layer shared with the embeddings)
QWEN = {
    (151936, 4096, 36): ("qwen3vl", "Qwen3-VL-8B", "Qwen-Image 2.1", False),
    (151936, 2560, 36): ("qwen3vl", "Qwen3-VL-4B", "Krea 2", True),
    (248320, 4096, 32): ("qwen3_5", "Qwen3.5-9B", "MOSS Image", False),
}
GEMMA_SIZES = {3840: "12B"}  # hidden size → name
# a Qwen3.5-9B named like FLUX 3 Image Klein's prompt upsampler (qwen3.5_9b_flux3_prompt_upsampler_*.safetensors,
# or "pup" as a word): BFL's files carry no metadata that tells it from MOSS Image's encoder
UPSAMPLER_NAME = re.compile(r"upsampler|(?<![a-z0-9])pup(?![a-z0-9])", re.I)
# ComfyUI's names → Gemma4UnifiedForConditionalGeneration's (the first matching prefix)
GEMMA_NAMES = (("model.", "model.language_model."), ("vision_model.", "model.embed_vision."),
               ("multi_modal_projector.", "model.embed_vision.multimodal_embedder."),
               ("audio_projector.", "model.embed_audio."))
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<turn|>", "<end_of_turn>", "<eos>")
RESERVE = 4 * 2**30  # GPU memory kept free for the images, the KV cache and the activations
# the load check: a language model continues this at about 2-4 nats a token; one that can't write text, 10 or more
CHECK_TEXT = ("The quick brown fox jumps over the lazy dog. Paris is the capital of France, and Berlin is the "
              "capital of Germany. Water boils at one hundred degrees Celsius.")
MAX_CHECK_LOSS = 6.0
MAX_BODY = 64 * 2**20


class EncoderError(ValueError):
    """The file can't write text; the message says why, for the user."""


class RequestError(ValueError):
    """A request the server can't answer (HTTP 400)."""


@dataclass(frozen=True)
class Encoder:
    kind: str  # "qwen3vl", "qwen3_5" or "gemma4"
    name: str  # e.g. "Qwen3-VL-8B"
    whose: str  # the model it's the text encoder of
    vision: bool  # the file has the vision part, so it can see images
    tied: bool  # no lm_head in the file: the output layer is the input embeddings
    tuned_from: str = ""  # a model that isn't a text encoder (FLUX 3's prompt upsampler): the model it was tuned from

    @property
    def label(self) -> str:
        """What it is, e.g. "Qwen3-VL-4B (Krea 2's text encoder)", "FLUX 3 prompt upsampler (Qwen3.5-9B)"."""
        return f"{self.name} ({self.tuned_from})" if self.tuned_from else f"{self.name} ({self.whose}'s text encoder)"

    def to_json(self) -> dict:
        return {"kind": self.kind, "name": self.name, "whose": self.whose, "vision": self.vision,
                **({"tuned_from": self.tuned_from} if self.tuned_from else {})}


def detect(path: str) -> Encoder:
    """Which text encoder ``path`` is, from its header only. Raises EncoderError when it can't write text."""
    from safetensors import safe_open

    path = os.path.expanduser(path)
    base = os.path.basename(path)
    if not os.path.isfile(path):
        raise EncoderError(f"The text encoder wasn't found: {path}")
    if not path.lower().endswith(".safetensors"):
        raise EncoderError("Choose the text encoder's .safetensors file")
    try:
        with safe_open(path, "pt") as f:
            keys = set(f.keys())
            meta = f.metadata() or {}
            shape = {k: tuple(f.get_slice(k).get_shape()) for k in keys if k.endswith(
                ("layers.0.mlp.down_proj.weight", "embed_tokens.weight"))}
    except Exception as e:
        raise EncoderError(f"{base} can't be read as a .safetensors file: {e}")

    if "gemma_config" in meta:
        cfg = json.loads(meta["gemma_config"])
        text = cfg.get("text_config", cfg)
        if "ltx" in str(cfg.get("gemma_version", "")).lower():
            raise EncoderError(f"{base} is LTX 2.5's text encoder, Lightricks' custom Gemma 4: it was changed for LTX "
                               f"and no longer predicts text, so it can't write captions. {SUPPORTED}")
        last = text.get("num_hidden_layers", 0) - 1
        if (cfg.get("model_type") != "gemma4_unified" or "model.norm.weight" not in keys
                or f"model.layers.{last}.mlp.down_proj.weight" not in keys):
            raise EncoderError(f"{base} isn't a whole Gemma 4 text encoder, so it can't write text. {SUPPORTED}")
        size = GEMMA_SIZES.get(text.get("hidden_size"), "")
        whose = "LTX" if any(k.startswith("text_embedding_projection.") for k in keys) else "Gemma 4"
        return Encoder("gemma4", f"Gemma 4 {size}".strip(), whose, "vision_model.patch_dense.weight" in keys, True)

    prefix = next((p for p in ("model.language_model.", "language_model.", "model.")
                   if f"{p}layers.0.mlp.down_proj.weight" in keys), None)
    if prefix is None:
        if any(".encoder.layers." in k or k.startswith("text_model.") for k in keys):
            raise EncoderError(f"{base} is a CLIP text encoder: it only turns text into numbers and can't write "
                               f"text. {SUPPORTED}")
        if any(k.startswith(("encoder.block.", "shared.")) for k in keys):
            raise EncoderError(f"{base} is a T5 encoder: it only turns text into numbers and can't write text. "
                               f"{SUPPORTED}")
        raise EncoderError(f"{base} isn't a text encoder Dainamo can write text with. {SUPPORTED}")
    layers = 1 + max(int(k[len(prefix) + 7:].split(".", 1)[0]) for k in keys if k.startswith(prefix + "layers."))
    hidden, _ = shape[f"{prefix}layers.0.mlp.down_proj.weight"]
    vocab = shape.get(f"{prefix}embed_tokens.weight", (0,))[0]
    norm, head = f"{prefix}norm.weight" in keys, "lm_head.weight" in keys
    if hidden == 5120 and not norm:
        raise EncoderError(f"{base} is MiniMax H3's text encoder: it stops after {layers} of Qwen3-VL-32B's 64 layers "
                           f"and has no output layer, so it can't write text. {SUPPORTED}")
    known = QWEN.get((vocab, hidden, layers))
    if known is None:
        raise EncoderError(f"{base} is a {layers}-layer Qwen model Dainamo doesn't know. {SUPPORTED}")
    kind, name, whose, tied = known
    if not norm:
        raise EncoderError(f"{base} has no final norm: this {name} was cut short, so it can't write text")
    if not tied and not head:
        raise EncoderError(f"{base} has no output layer (lm_head), so it can't write text. ComfyUI's {name} "
                           "files keep it.")
    vision = any(k.startswith(("model.visual.", "visual.")) for k in keys)
    hinted = any("upsampler" in str(v).lower() for v in meta.values())  # a header that says so (BFL's don't)
    if kind == "qwen3_5" and (UPSAMPLER_NAME.search(base) or hinted):
        return Encoder(kind, "FLUX 3 prompt upsampler", "FLUX 3 Image Klein", vision, tied, tuned_from=name)
    return Encoder(kind, name, whose, vision, tied)


# --- loading ---------------------------------------------------------------------------------------
def _qwen_name(k: str) -> str:
    """ComfyUI's model.layers.* / model.visual.* (or a transformers export's names) → the
    ForConditionalGeneration's model.language_model.* / model.visual.*."""
    if k.startswith(("model.language_model.", "model.visual.", "lm_head")):
        return k
    if k.startswith(("visual.", "language_model.")):
        return "model." + k
    return "model.language_model." + k[len("model."):] if k.startswith("model.") else k


def _gemma_name(k: str) -> str:
    return next((new + k[len(old):] for old, new in GEMMA_NAMES if k.startswith(old)), k)


def _sdpa(config):
    for c in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if c is not None:
            c._attn_implementation = "sdpa"


class Engine:
    """The encoder as a chat model on the GPU."""

    def __init__(self, enc: Encoder, path: str, log=print):
        self.enc, self.path, self.log = enc, os.path.expanduser(path), log
        self.model = self.processor = self.tokenizer = None
        self.eos: list[int] = []
        self.suppress: list[int] = []
        self.lock = threading.Lock()  # one request at a time: they share the GPU

    def _weights(self, rename, skip: tuple[str, ...]):
        """The file's weights on the GPU, renamed; without an lm_head, the output layer is the input embeddings."""
        from . import comfy_io

        loaded = comfy_io.load_state_dict(self.path, skip_prefixes=skip, device="cuda:0")
        loaded.tensors = {rename(k): v for k, v in loaded.tensors.items()}
        loaded.quant = {rename(k): q for k, q in loaded.quant.items()}
        embed = "model.language_model.embed_tokens"
        if "lm_head.weight" not in loaded.tensors:
            loaded.tensors["lm_head.weight"] = loaded.tensors[f"{embed}.weight"]
            if embed in loaded.quant:
                loaded.quant["lm_head"] = loaded.quant[embed]
        return loaded

    def _materialize(self, model, loaded):
        import torch

        from .fp8_linear import materialize

        if not self.enc.vision:  # a text-only file: no vision weights to load (images are refused)
            model.model.visual = torch.nn.Identity()
        materialize(model, loaded, torch.bfloat16, keep_fp8=True, device=None)

    def _qwen(self):
        import torch

        from . import hf_assets

        processor, config, _ = hf_assets.qwen_image_2_assets()
        if self.enc.kind == "qwen3_5":
            from transformers import Qwen3VLProcessor
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForConditionalGeneration as Model, Qwen3_5TextRotaryEmbedding as TextRope,
                Qwen3_5VisionRotaryEmbedding as VisionRope,
            )
            # MOSS's Qwen3.5 tokenizer and chat template, with Qwen3-VL's image processor (the same patches)
            tokenizer, config = hf_assets.moss_tokenizer_and_config()
            processor = Qwen3VLProcessor(image_processor=processor.image_processor, tokenizer=tokenizer,
                                         video_processor=processor.video_processor,
                                         chat_template=tokenizer.chat_template)
            skip = ("mtp.",)  # the multi-token-prediction head: transformers doesn't use it
        else:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration as Model, Qwen3VLTextRotaryEmbedding as TextRope,
                Qwen3VLVisionRotaryEmbedding as VisionRope,
            )
            if self.enc.name == "Qwen3-VL-4B":
                config = hf_assets.qwen3vl_tokenizer_and_config()[1]
            skip = ()
        _sdpa(config)
        for c in (config, config.text_config):
            c.tie_word_embeddings = self.enc.tied
        loaded = self._weights(_qwen_name, skip)
        with torch.device("meta"):
            model = Model(config)
        self._materialize(model, loaded)
        del loaded
        device = torch.device("cuda", 0)
        model.model.language_model.rotary_emb = TextRope(config.text_config, device=device)
        if self.enc.vision:
            model.model.visual.rotary_pos_emb = VisionRope(config.vision_config).to(device)
        return model, processor

    def _gemma(self):
        import torch
        from safetensors import safe_open
        from transformers import AutoProcessor
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedTextRotaryEmbedding,
        )

        # the tokenizer, chat template and processor settings travel in the file
        with safe_open(self.path, "pt") as f, tempfile.TemporaryDirectory() as tmp:
            meta = f.metadata()
            for key in f.keys():
                if key.startswith("hf_asset__") or key == "tokenizer_json":
                    name = "tokenizer.json" if key == "tokenizer_json" else key[len("hf_asset__"):]
                    with open(os.path.join(tmp, name), "wb") as out:
                        out.write(f.get_tensor(key).numpy().tobytes())
            processor = AutoProcessor.from_pretrained(tmp)
            if os.path.exists(os.path.join(tmp, "generation_config.json")):
                with open(os.path.join(tmp, "generation_config.json")) as g:
                    gen = json.load(g)
                self.eos = [i for i in gen.get("eos_token_id") or [] if isinstance(i, int)]
                self.suppress = list(gen.get("suppress_tokens") or [])
        config = Gemma4UnifiedConfig(**{k: v for k, v in json.loads(meta["gemma_config"]).items() if k != "dtype"})
        _sdpa(config)
        # LTX's text projections are for the video model, not for writing text
        loaded = self._weights(_gemma_name, ("text_embedding_projection.", "hf_asset__", "tokenizer_json"))
        with torch.device("meta"):
            model = Gemma4UnifiedForConditionalGeneration(config)
        if not self.enc.vision:
            model.model.embed_vision = torch.nn.Identity()
        from .fp8_linear import materialize
        materialize(model, loaded, torch.bfloat16, keep_fp8=True, device=None)
        del loaded
        device = torch.device("cuda", 0)
        lm = model.model.language_model
        lm.rotary_emb = Gemma4UnifiedTextRotaryEmbedding(config.text_config).to(device)
        lm.embed_tokens.embed_scale = torch.tensor(config.text_config.hidden_size**0.5, device=device)
        return model, processor

    def load(self):
        import torch

        from .fp8_linear import assert_materialized

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU is visible: the text encoder runs on a GPU")
        start = time.time()
        self.log(f"Loading {self.enc.label} from {self.path}")
        self.model, self.processor = self._gemma() if self.enc.kind == "gemma4" else self._qwen()
        self.model.eval()
        assert_materialized(self.model)
        self.tokenizer = self.processor.tokenizer
        vocab = self.tokenizer.get_vocab()
        ends = {vocab[t] for t in END_TOKENS if t in vocab} | {self.tokenizer.eos_token_id}
        self.eos = sorted({*self.eos, *ends} - {None})
        loss = self.check()
        if loss > MAX_CHECK_LOSS:
            raise RuntimeError(f"{self.enc.name} doesn't predict text: it scores {loss:.1f} on a plain English sentence, "
                               "where a language model scores about 3, so it can't write captions. Its model may have "
                               "retrained it as an encoder.")
        expanded, kept = expand_quantized(self.model, RESERVE)
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 2**30
        layers = f"; {expanded} quantized layers expanded to bf16" + (f", {kept} kept quantized" if kept else "") \
            if expanded or kept else ""
        self.log(f"Loaded in {time.time() - start:.0f} s: {used:.1f} GB on the GPU{layers}; it continues a test "
                 f"sentence at {loss:.1f} nats a token")

    def check(self) -> float:
        """How well it continues CHECK_TEXT: the mean next-token loss, in nats."""
        import torch
        import torch.nn.functional as F

        ids = self.tokenizer(CHECK_TEXT, return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        return F.cross_entropy(logits, ids[0, 1:]).item()

    def chat(self, body: dict) -> dict:
        """An OpenAI chat completion for ``body``."""
        import torch
        from transformers import GenerationConfig

        messages, images = to_chat(body.get("messages"))
        if images and not self.enc.vision:
            raise RequestError("This text encoder file has no vision part, so it can't see images")
        try:
            max_new = max(1, min(32768, int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)))
            temperature = float(body.get("temperature", 0.7))
            gen = {"max_new_tokens": max_new, "do_sample": temperature > 0, "eos_token_id": self.eos,
                   "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                   else self.eos[0]}
            if temperature > 0:
                gen.update(temperature=temperature, top_p=float(body.get("top_p", 0.95)),
                           top_k=int(body.get("top_k", 64)), min_p=float(body.get("min_p", 0.05)))
            penalty = body.get("repetition_penalty", body.get("repeat_penalty"))
            if penalty:
                gen["repetition_penalty"] = float(penalty)
            seed = body.get("seed")
            seed = None if seed is None else int(seed)
        except (TypeError, ValueError) as e:
            raise RequestError(f"Bad sampling setting: {e}")
        if self.suppress:
            gen["suppress_tokens"] = self.suppress
        template = {"enable_thinking": False, **(body.get("chat_template_kwargs") or {})}
        # only what the template uses (Qwen3-VL's has no thinking): transformers takes the rest for the processor
        template = {k: v for k, v in template.items() if k in str(self.processor.chat_template)}
        text =self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template)
        with self.lock:
            start = time.time()
            inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to("cuda:0")
            prompt = int(inputs["input_ids"].shape[1])
            if seed is not None:
                torch.manual_seed(seed)
            with torch.inference_mode():
                out = self.model.generate(**inputs, generation_config=GenerationConfig(**gen))
            new = out[0, prompt:].tolist()
            took = time.time() - start
        stopped = bool(new) and new[-1] in self.eos
        reply = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        self.log(f"Request: {len(images)} image(s), {prompt} prompt tokens → {len(new)} tokens in {took:.1f} s "
                 f"({len(new) / max(took, 1e-6):.1f}/s)")
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                "model": self.enc.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop" if stopped or len(new) < max_new else "length"}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": len(new), "total_tokens": prompt + len(new)}}


def expand_quantized(model, reserve: int) -> tuple[int, int]:
    """Turn the language model's int8/fp8 layers into bf16 ones while the GPU keeps ``reserve`` bytes free: a
    quantized layer is expanded for every matmul, so for one token at a time it runs about 3x slower. The output
    layer goes first (the biggest, and run for every token); the vision part only runs once per request and stays
    quantized. Returns (layers expanded, language-model layers left quantized)."""
    import torch
    import torch.nn as nn

    from .comfy_io import QuantLayer, dequantize
    from .fp8_linear import QuantLinear, _get_parent

    # names only: holding the modules would keep each quantized weight after it's replaced
    todo = [n for n, m in model.named_modules() if isinstance(m, QuantLinear)
            and not n.startswith(("model.visual.", "model.embed_vision.", "model.embed_audio."))]
    todo.sort(key=lambda n: n != "lm_head")
    expanded = 0
    for name in todo:
        m = model.get_submodule(name)
        free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if free - m.weight.numel() * 2 < reserve:  # the bf16 copy is made before the quantized one is freed
            break
        weight = dequantize(m.weight, QuantLayer("int8_tensorwise", m.scale, rot_size=m.rot_size), torch.bfloat16)
        lin = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None, device="meta")
        lin.weight = nn.Parameter(weight, requires_grad=False)
        if m.bias is not None:
            lin.bias = nn.Parameter(m.bias.to(torch.bfloat16), requires_grad=False)
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, lin)
        del m, weight
        expanded += 1
    return expanded, len(todo) - expanded


def _image(url) -> "Image.Image":
    from PIL import Image

    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
        raise RequestError("Send images as data: URLs (base64)")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=False)))
        img.load()
    except (binascii.Error, OSError, ValueError) as e:
        raise RequestError(f"An image couldn't be read: {e}")
    return img.convert("RGB")


def to_chat(messages) -> tuple[list[dict], list]:
    """OpenAI chat messages → (chat-template messages with {"type": "image"} placeholders, the images in order)."""
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    out, images = [], []
    for m in messages:
        role, content = (m.get("role"), m.get("content")) if isinstance(m, dict) else (None, None)
        if role not in ("system", "user", "assistant"):
            raise RequestError(f"Unknown message role {role!r}")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for c in content:
                kind = c.get("type") if isinstance(c, dict) else None
                if kind == "text":
                    parts.append({"type": "text", "text": str(c.get("text") or "")})
                elif kind == "image_url":
                    url = c.get("image_url")
                    images.append(_image(url.get("url") if isinstance(url, dict) else url))
                    parts.append({"type": "image"})
                else:
                    raise RequestError(f"Unsupported message part {kind!r}: send text and image_url parts")
        else:
            raise RequestError("A message's content must be text or a list of parts")
        out.append({"role": role, "content": parts})
    return out, images


# --- the server ------------------------------------------------------------------------------------
class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine: Engine):
        super().__init__(address, Handler)
        self.engine = engine
        self.ready = False


class Handler(BaseHTTPRequestHandler):
    server: Server

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, message: str, kind: str = "invalid_request_error"):
        self._send(code, {"error": {"code": code, "message": message, "type": kind}})

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            if self.server.ready:
                self._send(200, {"status": "ok"})
            else:
                self._error(503, "Loading model", "unavailable_error")
        elif path in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [{"id": self.server.engine.enc.name, "object": "model",
                                                         "owned_by": "dainamo"}]})
        else:
            self._error(404, f"No {path}")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            return self._error(404, f"No {path}")
        if not self.server.ready:
            return self._error(503, "Loading model", "unavailable_error")
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_BODY:
                return self._error(413, "The request is too large")
            body = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(body, dict):
                raise RequestError("The request must be a JSON object")
            if body.get("stream"):
                raise RequestError("Streaming isn't supported: send stream=false")
            self._send(200, self.server.engine.chat(body))
        except (RequestError, json.JSONDecodeError) as e:
            self._error(400, str(e))
        except Exception as e:
            import torch

            if isinstance(e, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
                message = f"Out of GPU memory: {str(e).splitlines()[0]}"
            else:
                traceback.print_exc()
                message = f"{type(e).__name__}: {e}"
            print(f"Request failed: {message}", flush=True)
            self._error(500, message, "server_error")

    def log_message(self, *args):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the text encoder's .safetensors file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args(argv)
    try:
        enc = detect(args.model)
    except EncoderError as e:
        print(f"error: {e}", flush=True)
        sys.exit(1)
    engine = Engine(enc, args.model, log=lambda s: print(s, flush=True))
    try:
        server = Server((args.host, args.port), engine)  # listening first: /health says 503 while it loads
    except OSError as e:
        print(f"error: couldn't bind to {args.host}:{args.port}: {e}", flush=True)
        sys.exit(1)

    def load():
        try:
            engine.load()
        except Exception as e:
            traceback.print_exc()
            print(f"error: {e}", flush=True)
            os._exit(1)
        server.ready = True
        print(f"Ready: http://{args.host}:{args.port}/v1", flush=True)

    threading.Thread(target=load, daemon=True).start()
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

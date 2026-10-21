"""Every test run gets a throwaway data folder, never the real data/. This runs before any test module
imports dainamo.server (which reads DAINAMO_DATA once, at import)."""

import os
import tempfile

os.environ["DAINAMO_DATA"] = tempfile.mkdtemp(prefix="dainamo-test-")
os.environ["DAINAMO_ALLOWED_HOSTS"] = "127.0.0.1,localhost,testserver"

import sys  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def fake_llama(tmp_path_factory):
    """A stand-in llama-server (tests/fake_llama_server.py, run by a shell script) with a model and a vision
    projector for it: {binary, model, mmproj}."""
    d = tmp_path_factory.mktemp("llama")
    binary = d / "llama-server"
    stub = os.path.join(os.path.dirname(__file__), "fake_llama_server.py")
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub}" "$@"\n')
    binary.chmod(0o755)
    for name in ("Qwen2.5-VL-7B-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-7B-f16.gguf"):
        (d / name).write_bytes(b"GGUF")
    return {"binary": str(binary), "model": str(d / "Qwen2.5-VL-7B-Q4_K_M.gguf"),
            "mmproj": str(d / "mmproj-Qwen2.5-VL-7B-f16.gguf")}


def header_only_safetensors(path, tensors: dict[str, tuple[str, tuple[int, ...]]], metadata: dict | None = None):
    """A .safetensors file with ``tensors`` ({name: (dtype, shape)}) whose data is a hole: sparse, so a 9 GB
    encoder's header costs nothing on disk. Enough for code that only reads headers."""
    import json
    import struct

    sizes = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "F8_E4M3": 1}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = sizes[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        f.truncate(8 + len(blob) + offset)
    return str(path)


def _qwen_encoder(vocab, hidden, inter, layers, prefix="model.", head=False, norm=True, vision=True, extra=()):
    t = {f"{prefix}embed_tokens.weight": ("BF16", (vocab, hidden)),
         f"{prefix}layers.0.mlp.down_proj.weight": ("BF16", (hidden, inter))}
    t.update({f"{prefix}layers.{i}.mlp.down_proj.weight": ("BF16", (1,)) for i in range(1, layers)})
    if norm:
        t[f"{prefix}norm.weight"] = ("BF16", (hidden,))
    if head:
        t["lm_head.weight"] = ("I8", (vocab, hidden))
    if vision:
        t["model.visual.patch_embed.proj.weight"] = ("BF16", (1,))
    t.update({k: ("BF16", (1,)) for k in extra})
    return t


def _gemma_encoder(version):
    import json

    cfg = {"model_type": "gemma4_unified", "architectures": ["Gemma4UnifiedForConditionalGeneration"],
           "text_config": {"num_hidden_layers": 2, "hidden_size": 3840}}
    if version:
        cfg["gemma_version"] = version
    return ({"model.layers.0.mlp.down_proj.weight": ("BF16", (1,)), "model.layers.1.mlp.down_proj.weight": ("BF16", (1,)),
             "model.norm.weight": ("BF16", (1,)), "vision_model.patch_dense.weight": ("BF16", (1,))},
            {"gemma_config": json.dumps(cfg)})


# name → (tensors, metadata): the headers of the text encoders Dainamo's families use
FAKE_ENCODERS = {
    "krea": (_qwen_encoder(151936, 2560, 9728, 36), None),  # Qwen3-VL-4B: the output layer is the embeddings
    "qwen_image": (_qwen_encoder(151936, 4096, 12288, 36, head=True), None),
    "qwen_image_no_head": (_qwen_encoder(151936, 4096, 12288, 36), None),
    "moss": (_qwen_encoder(248320, 4096, 12288, 32, prefix="model.language_model.", head=True,
                           extra=("mtp.fc.weight",)), None),
    "minimax": (_qwen_encoder(151936, 5120, 25600, 50, norm=False), None),
    "gemma": _gemma_encoder(""),
    "ltx": _gemma_encoder("gemma4-12b-ltx-v1"),
    "clip": ({"text_model.encoder.layers.0.mlp.fc1.weight": ("BF16", (1,))}, None),
    "t5": ({"encoder.block.0.layer.0.SelfAttention.q.weight": ("BF16", (1,)), "shared.weight": ("BF16", (1,))}, None),
}


@pytest.fixture
def fake_encoder(tmp_path_factory):
    """``fake_encoder(name)``: the path of a header-only copy of a text encoder in FAKE_ENCODERS."""
    d = tmp_path_factory.mktemp("encoders")

    def make(name: str) -> str:
        tensors, metadata = FAKE_ENCODERS[name]
        return header_only_safetensors(d / f"{name}.safetensors", tensors, metadata)
    return make

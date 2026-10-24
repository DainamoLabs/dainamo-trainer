"""Captions from a model's text encoder (dainamo/engine/textgen.py): which encoder files can write text (from
their headers, with header-only stand-ins), OpenAI messages turned into chat-template ones, the server's requests,
quantized layers expanded to bf16, and a tiny random Qwen3-VL in ComfyUI's layout loaded, checked and asked.

    .venv/bin/python -m pytest tests/test_textgen.py -v
"""

import base64
import io
import threading

import httpx
import pytest
import torch
import torch.nn as nn
from PIL import Image

from dainamo.engine import textgen
from dainamo.engine.textgen import Encoder, EncoderError, RequestError, detect, expand_quantized, to_chat

KREA = Encoder("qwen3vl", "Qwen3-VL-4B", "Krea 2", True, True)


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_knows_the_text_encoders_that_can_write(fake_encoder):
    assert detect(fake_encoder("krea")) == KREA  # no lm_head: it shares the input embeddings
    assert detect(fake_encoder("qwen_image")) == Encoder("qwen3vl", "Qwen3-VL-8B", "Qwen-Image 2.1", True, False)
    assert detect(fake_encoder("moss")) == Encoder("qwen3_5", "Qwen3.5-9B", "MOSS Image", True, False)
    assert detect(fake_encoder("gemma")) == Encoder("gemma4", "Gemma 4 12B", "Gemma 4", True, True)


def test_names_flux3s_prompt_upsampler_from_its_file(tmp_path):
    """A Qwen3.5-9B like MOSS Image's encoder, told apart by its name (BFL's header says nothing), or a header that
    says so."""
    from conftest import FAKE_ENCODERS, header_only_safetensors

    moss, _ = FAKE_ENCODERS["moss"]
    pup = Encoder("qwen3_5", "FLUX 3 prompt upsampler", "FLUX 3 Image Klein", True, False, tuned_from="Qwen3.5-9B")
    for name, meta, want in (("qwen3.5_9b_flux3_prompt_upsampler_int8_convrot", None, pup),
                             ("flux3-pup-bf16", None, pup), ("mine", {"modelspec.title": "FLUX 3 prompt upsampler"}, pup),
                             ("pupil", None, Encoder("qwen3_5", "Qwen3.5-9B", "MOSS Image", True, False))):
        assert detect(header_only_safetensors(tmp_path / f"{name}.safetensors", moss, meta)) == want, name
    assert pup.label == "FLUX 3 prompt upsampler (Qwen3.5-9B)" and pup.to_json()["tuned_from"] == "Qwen3.5-9B"
    assert KREA.label == "Qwen3-VL-4B (Krea 2's text encoder)" and "tuned_from" not in KREA.to_json()
    krea, _ = FAKE_ENCODERS["krea"]  # only a Qwen3.5-9B can be the upsampler
    assert detect(header_only_safetensors(tmp_path / "krea_upsampler.safetensors", krea)) == KREA


@pytest.mark.parametrize("name, said", [
    ("minimax", "MiniMax H3's text encoder: it stops after 50 of Qwen3-VL-32B's 64 layers"),
    ("ltx", "LTX 2.5's text encoder, Lightricks' custom Gemma 4"),
    ("qwen_image_no_head", "has no output layer (lm_head), so it can't write text. ComfyUI's Qwen3-VL-8B files keep it"),
    ("clip", "is a CLIP text encoder"),
    ("t5", "is a T5 encoder"),
])
def test_says_why_a_text_encoder_cant_write(fake_encoder, name, said):
    with pytest.raises(EncoderError) as e:
        detect(fake_encoder(name))
    assert said in str(e.value)
    if name != "qwen_image_no_head":  # the right model in the wrong file: no list of the others
        assert "Qwen-Image 2.1 (Qwen3-VL-8B)" in str(e.value)  # which ones can


def test_says_why_a_file_isnt_one(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    (tmp_path / "broken.safetensors").write_bytes(b"not a safetensors file")
    for path, said in ((tmp_path / "nope.safetensors", "wasn't found"), (tmp_path / "model.gguf", ".safetensors file"),
                       (tmp_path / "broken.safetensors", "can't be read")):
        with pytest.raises(EncoderError, match=said):
            detect(str(path))


def test_openai_messages_become_chat_template_ones():
    url = data_url(Image.new("RGBA", (8, 6), "red"))
    messages, images = to_chat([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": url}}]},
    ])
    assert messages == [{"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                        {"role": "user", "content": [{"type": "text", "text": "Describe"}, {"type": "image"}]}]
    assert [(im.size, im.mode) for im in images] == [((8, 6), "RGB")]
    image = lambda u: [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}}]}]  # noqa: E731
    for bad, said in ((image("http://example.com/a.png"), "data: URLs"), (image("data:image/png;base64,AAAA"), "couldn't be read"),
                      ([{"role": "tool", "content": "x"}], "role"), ([], "non-empty"),
                      ([{"role": "user", "content": [{"type": "input_audio"}]}], "Unsupported message part")):
        with pytest.raises(RequestError, match=said):
            to_chat(bad)


class StubEngine:
    enc = KREA

    def chat(self, body):
        if body["messages"] == "bad":
            raise RequestError("messages must be a non-empty list")
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "A red square."},
                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}


def test_the_server_answers_like_llama_server():
    from dainamo.server import captioner

    srv = textgen.Server(("127.0.0.1", 0), StubEngine())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    ask = {"messages": [{"role": "user", "content": "hi"}]}
    try:
        with httpx.Client() as c:
            assert c.get(url + "/health").status_code == 503  # the model is loading
            assert c.post(url + "/v1/chat/completions", json=ask).status_code == 503
            srv.ready = True
            assert c.get(url + "/health").json() == {"status": "ok"}
            assert c.get(url + "/v1/models").json()["data"][0]["id"] == "Qwen3-VL-4B"
            assert c.post(url + "/v1/chat/completions", json=ask).json()["choices"][0]["message"]["content"] == "A red square."
            r = c.post(url + "/v1/chat/completions", json={"messages": "bad"})
            assert r.status_code == 400 and r.json()["error"]["message"] == "messages must be a non-empty list"
            assert "Streaming" in c.post(url + "/v1/chat/completions", json={**ask, "stream": True}).json()["error"]["message"]
            assert c.post(url + "/v1/chat/completions", content=b"{not json").status_code == 400
            assert c.post(url + "/v1/completions", json=ask).status_code == 404
            # what Auto-Caption and Enhance prompt send
            assert captioner.chat({"endpoint": url + "/v1"}, c, ask["messages"], 100, 0.2) == "A red square."
    finally:
        srv.shutdown()
        srv.server_close()


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_quantized_layers_become_bf16_while_the_gpu_has_room():
    from dainamo.engine.fp8_linear import QuantLinear

    torch.manual_seed(0)

    def quant(out_f, in_f, rot=1):
        w = torch.randint(-127, 128, (out_f, in_f), dtype=torch.int8, device="cuda")
        return QuantLinear(in_f, out_f, w, torch.rand(out_f, device="cuda") / 100, torch.randn(out_f, device="cuda"),
                           torch.bfloat16, rot_size=rot)

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([quant(64, 256, rot=256), quant(32, 64)])  # one in the convrot basis
    model.model.visual = nn.Module()
    model.model.visual.proj = quant(16, 64)  # runs once a request: stays quantized
    model.lm_head = quant(100, 64)
    xs = {"model.layers.0": torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)}
    xs.update({k: torch.randn(3, 64, device="cuda", dtype=torch.bfloat16) for k in ("model.layers.1", "lm_head")})
    before = {k: model.get_submodule(k)(x).float() for k, x in xs.items()}
    assert expand_quantized(model, reserve=1 << 60) == (0, 3)  # no room
    assert expand_quantized(model, reserve=0) == (3, 0)
    assert isinstance(model.lm_head, nn.Linear) and model.lm_head.weight.dtype == torch.bfloat16
    assert isinstance(model.model.visual.proj, QuantLinear)
    for k, x in xs.items():
        after = model.get_submodule(k)(x).float()
        assert torch.allclose(after, before[k], rtol=0.02, atol=0.02 * before[k].abs().max().item()), k


def tiny_qwen3vl_config():
    from transformers import Qwen3VLConfig

    from dainamo.engine import hf_assets

    try:
        _, real, _ = hf_assets.qwen_image_2_assets()
    except Exception as e:  # not downloaded, and no network
        pytest.skip(f"Qwen-Image 2.1's processor isn't available: {e}")
    text, vision = real.text_config.to_dict(), real.vision_config.to_dict()
    text.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                head_dim=32, rope_parameters={**text["rope_parameters"], "mrope_section": [6, 5, 5]})
    vision.update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  deepstack_visual_indexes=[0, 1])
    return Qwen3VLConfig(text_config=text, vision_config=vision, tie_word_embeddings=True,
                         **{k: getattr(real, k) for k in ("image_token_id", "video_token_id", "vision_start_token_id",
                                                           "vision_end_token_id")})


@cuda
def test_a_tiny_qwen3_vl_in_comfyui_layout_is_loaded_checked_and_asked(tmp_path, monkeypatch):
    """Random weights, so it can't write text: the load check refuses it; with the check off it answers, with an
    image, through transformers' generate."""
    from safetensors.torch import save_file
    from transformers import Qwen3VLForConditionalGeneration

    from dainamo.engine import hf_assets

    config = tiny_qwen3vl_config()
    torch.manual_seed(0)
    state = Qwen3VLForConditionalGeneration(config).state_dict()
    # ComfyUI's names: model.layers.* / model.embed_tokens / model.norm / model.visual.*, and no lm_head (tied)
    comfy = {("model." + k[len("model.language_model."):] if k.startswith("model.language_model.") else k):
             v.to(torch.bfloat16).contiguous().clone() for k, v in state.items() if k != "lm_head.weight"}
    path = str(tmp_path / "tiny_qwen3vl.safetensors")
    save_file(comfy, path)
    monkeypatch.setattr(hf_assets, "qwen3vl_tokenizer_and_config", lambda: (None, tiny_qwen3vl_config()))

    with pytest.raises(RuntimeError, match="doesn't predict text"):
        textgen.Engine(KREA, path, log=lambda s: None).load()

    monkeypatch.setattr(textgen, "MAX_CHECK_LOSS", 1e9)
    logs = []
    engine = textgen.Engine(KREA, path, log=logs.append)
    engine.load()
    lm = engine.model.model.language_model
    assert engine.model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr()  # tied, as in the file
    assert "Loaded in" in logs[-1] and "nats a token" in logs[-1]
    ask = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this."},
                                                      {"type": "image_url", "image_url": {"url": data_url(Image.new("RGB", (64, 64), "red"))}}]}],
           "max_tokens": 5, "temperature": 0}
    out = engine.chat(ask)
    usage = out["usage"]
    assert usage["completion_tokens"] <= 5 and usage["prompt_tokens"] > 4  # 64x64 px is 4 image tokens, plus the text
    assert out["choices"][0]["finish_reason"] in ("stop", "length") and out["model"] == "Qwen3-VL-4B"
    sampled = {**ask, "temperature": 0.8, "seed": 7}
    assert engine.chat(sampled)["choices"][0]["message"] == engine.chat(sampled)["choices"][0]["message"]  # seeded
    assert any(line.startswith("Request: 1 image(s)") for line in logs)

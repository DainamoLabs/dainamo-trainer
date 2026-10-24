"""FLUX.2 Klein: the DiT against diffusers' independent port (text-to-image, references appended, the KV
checkpoint's causal references), sizes from checkpoint shapes, the text encoder's truncated read, the sampling
and training schedules against the release's scheduler, the driver's edit plumbing, and the profile checks.

The tests with real files skip themselves when the files aren't there.

    .venv/bin/python -m pytest tests/test_flux2_klein.py -v
"""

import json
import os
import re

import numpy as np
import pytest
import torch
from conftest import header_only_safetensors
from safetensors.torch import save_file

from dainamo.engine import comfy_io
from dainamo.engine.families.flux2_klein import driver as K
from dainamo.engine.families.flux2_klein.driver import Flux2KleinDriver
from dainamo.engine.families.flux2_klein.model import KLEIN_4B, KLEIN_9B, Flux2KleinConfig, Flux2KleinDiT
from dainamo.engine.lora import LoRANetwork

CPU = torch.device("cpu")
MODELS = os.environ.get("DAINAMO_TEST_MODELS", os.path.expanduser("~/AI_APPS/COMFY_CACHE/models/diffusion_models"))
VAES = os.path.join(os.path.dirname(MODELS), "vae")
HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/AI_MODELS/hub"))
REAL_DITS = [os.path.join(MODELS, n) for n in ("flux-2-klein-base-4b-fp8.safetensors", "flux-2-klein-9b-fp8.safetensors")]
REAL_TEXT = [os.path.join(HUB, f"models--Qwen--Qwen3-{s}") for s in ("4B", "8B")]
SMALL = dict(context_in_dim=48, hidden_size=64, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=(8, 8, 8, 8))


def small_model(seed: int = 0, **kw) -> Flux2KleinDiT:
    m = Flux2KleinDiT(Flux2KleinConfig(**{**SMALL, **kw})).eval()
    g = torch.Generator().manual_seed(seed)
    m.load_state_dict({k: torch.randn(v.shape, generator=g) * (0.3 if k.endswith("scale") else 0.08) + (
        1.0 if k.endswith("scale") else 0.0) for k, v in sorted(m.state_dict().items())})
    return m


def diffusers_twin(ours: Flux2KleinDiT):
    """diffusers' Flux2Transformer2DModel with the same weights, converted by diffusers' own key mapping."""
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

    c = ours.config
    twin = Flux2Transformer2DModel(
        patch_size=1, in_channels=c.in_channels, num_layers=c.depth, num_single_layers=c.depth_single_blocks,
        attention_head_dim=c.hidden_size // c.num_heads, num_attention_heads=c.num_heads,
        joint_attention_dim=c.context_in_dim, timestep_guidance_channels=256, mlp_ratio=c.mlp_ratio,
        axes_dims_rope=tuple(c.axes_dim), rope_theta=c.theta, guidance_embeds=False).eval()
    sd = convert_flux2_transformer_checkpoint_to_diffusers({k: v.clone() for k, v in ours.state_dict().items()})
    twin.load_state_dict(sd, strict=True)
    return twin


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def _inputs(g, b=2, h=3, w=4, n_txt=7):
    img = torch.randn(b, h * w, 128, generator=g)
    txt = torch.randn(b, n_txt, SMALL["context_in_dim"], generator=g)
    ids = K.image_ids(h, w)[None].expand(b, -1, -1)
    tids = K.text_ids(n_txt)[None].expand(b, -1, -1)
    return img, ids, txt, tids


def test_dit_matches_diffusers():
    """Text-to-image with two noise levels, two references appended after the image (the Klein checkpoints'
    edits), and the KV checkpoint's layout (references first, at timestep 0, attending only to themselves)."""
    torch.manual_seed(0)
    ours = small_model()
    twin = diffusers_twin(ours)
    g = torch.Generator().manual_seed(1)
    img, ids, txt, tids = _inputs(g)
    t = torch.tensor([0.83, 0.21])
    ref = torch.randn(2, 4 + 6, 128, generator=g)
    ref_ids = torch.cat([K.image_ids(2, 2, 10), K.image_ids(2, 3, 20)])[None].expand(2, -1, -1)
    with torch.no_grad():
        want = twin(img, txt, t, ids, tids, return_dict=False)[0]
        assert _rel(ours(img, ids, txt, tids, t), want) < 1e-5
        # edits: the references follow the image tokens; only the image's prediction is kept
        want = twin(torch.cat([img, ref], 1), txt, t, torch.cat([ids, ref_ids], 1), tids, return_dict=False)[0]
        got = ours(img, ids, txt, tids, t, ref, ref_ids)
        assert got.shape == img.shape and _rel(got, want[:, : img.shape[1]]) < 1e-5
        assert _rel(got, ours(img, ids, txt, tids, t)) > 1e-3  # and the references change it
        # the KV checkpoint: references first, modulated at t = 0, attending only to each other (diffusers' KV
        # pipeline swaps in its KV attention processors for this)
        from diffusers.models.transformers.transformer_flux2 import (Flux2KVAttnProcessor,
                                                                     Flux2KVParallelSelfAttnProcessor)

        for block in twin.transformer_blocks:
            block.attn.set_processor(Flux2KVAttnProcessor())
        for block in twin.single_transformer_blocks:
            block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())
        want = twin(torch.cat([ref, img], 1), txt, t, torch.cat([ref_ids, ids], 1), tids, return_dict=False,
                    kv_cache_mode="extract", num_ref_tokens=ref.shape[1])[0]
        got = ours(img, ids, txt, tids, t, ref, ref_ids, causal_refs=True)
        assert _rel(got, want) < 1e-5
        # negative control: the comparison is sensitive to the order of the fused q/k/v
        w = ours.single_blocks[0].linear1.weight
        d = SMALL["hidden_size"]
        w.data = torch.cat([w[d: 2 * d], w[:d], w[2 * d:]])
        assert _rel(ours(img, ids, txt, tids, t), twin(img, txt, t, ids, tids, return_dict=False)[0]) > 1e-3


def test_config_from_checkpoint_shapes():
    for cfg in (KLEIN_4B, KLEIN_9B, Flux2KleinConfig(**SMALL)):
        with torch.device("meta"):
            m = Flux2KleinDiT(cfg)
        shapes = {k: list(v.shape) for k, v in m.state_dict().items()}
        assert Flux2KleinConfig.from_shapes(shapes) == cfg
        # measurement builds a few single blocks: the double blocks are all still read
        few = {k: v for k, v in shapes.items() if not k.startswith("single_blocks.") or k.startswith("single_blocks.0.")}
        assert Flux2KleinConfig.from_shapes(few).depth_single_blocks == 1
        if cfg is KLEIN_4B:
            assert sum(v.numel() for v in m.state_dict().values()) == 3_875_544_576
    with pytest.raises(ValueError, match="dev"):
        Flux2KleinConfig.from_shapes({**shapes, "guidance_in.in_layer.weight": [64, 256]})


@pytest.mark.parametrize("path", REAL_DITS)
def test_real_diffusion_models_build_and_carry_the_signature(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not here")
    from dainamo.engine.families.base import transformer_shapes
    from dainamo.engine.memory import tensors

    _, shapes = transformer_shapes(path)
    d = Flux2KleinDriver({"diffusion_model": path}, CPU)
    model = d.build_transformer(shapes)
    want = {k: list(v.shape) for k, v in model.state_dict().items()}
    have = {k: v for k, v in shapes.items() if not k.endswith(comfy_io.SIDECAR_SUFFIXES)}
    assert want == have  # every key the checkpoint's, no mapping
    assert model.config == (KLEIN_4B if "4b" in path else KLEIN_9B)
    t = tensors(path)
    assert Flux2KleinDriver.matches("diffusion_model", t)
    assert not Flux2KleinDriver.matches("text_encoder", t) and not Flux2KleinDriver.matches("vae", t)


def test_real_text_encoders_and_vae_carry_the_signature():
    from dainamo.engine.memory import tensors

    seen = 0
    for path, size in zip(REAL_TEXT, ("4B", "8B")):
        if not os.path.isdir(path):
            continue
        seen += 1
        keys = K.header(path)  # a Hugging Face cache folder, read as its main snapshot
        assert Flux2KleinDriver.matches("text_encoder", keys) and K.text_encoder_size(keys) == size
        assert not Flux2KleinDriver.matches("diffusion_model", keys)
    vl = os.path.join(os.path.dirname(MODELS), "clip", "qwen3vl_4b_fp8_scaled.safetensors")
    if os.path.exists(vl):  # Krea 2's Qwen3-VL-4B has Qwen3-4B's shapes, but a vision tower
        assert not Flux2KleinDriver.matches("text_encoder", tensors(vl))
    vae = os.path.join(VAES, "flux2-vae.safetensors")
    if os.path.exists(vae):
        seen += 1
        assert Flux2KleinDriver.matches("vae", tensors(vae))
    for other in ("qwen_image_vae.safetensors", "ae.safetensors"):
        if os.path.exists(os.path.join(VAES, other)):
            assert not Flux2KleinDriver.matches("vae", tensors(os.path.join(VAES, other)))
    if not seen:
        pytest.skip("no real files here")


def test_sampling_schedule_matches_the_release_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

    sched = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096, max_shift=1.15, num_train_timesteps=1000,
        shift=3.0, time_shift_type="exponential", use_dynamic_shifting=True)
    for steps, seq in ((28, 32 * 32), (50, 64 * 64), (4, 64 * 64), (28, 96 * 64), (8, 16 * 16)):
        mu = compute_empirical_mu(seq, steps)
        assert K.empirical_mu(seq, steps) == pytest.approx(mu, abs=1e-12)
        sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
        ours = K.sampling_sigmas(steps, seq)
        assert len(ours) == steps + 1 and ours[0] == 1.0 and ours[-1] == 0.0
        assert np.allclose(ours, sched.sigmas.numpy(), atol=1e-6)
    d = Flux2KleinDriver({}, CPU)
    assert d.default_sigmas(28, 512, 512) == K.sampling_sigmas(28, 1024)
    assert d.default_shift(512, 512) == pytest.approx(np.exp(compute_empirical_mu(1024, 28)))


def test_training_noise_levels_follow_the_scheduler_shift():
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    for seq in (256, 1024, 4096, 6144):
        assert K.mu_for(seq) == pytest.approx(calculate_shift(seq, 256, 4096, 0.5, 1.15))
    d = Flux2KleinDriver({}, CPU)
    torch.manual_seed(0)
    t = d.sample_timesteps(20000, (64, 64), {"timestep_sampling": "shift"})
    assert 0 < float(t.min()) and float(t.max()) < 1
    assert float(t.median()) == pytest.approx(np.exp(1.15) / (np.exp(1.15) + 1), abs=0.01)  # the shifted 0.5
    small = d.sample_timesteps(20000, (16, 16), {"timestep_sampling": "shift"})
    assert float(small.median()) < float(t.median())  # smaller images, less shift


def test_sequence_and_caption_tokens():
    from dainamo.engine.hf_assets import qwen3_tokenizer_and_config

    assert Flux2KleinDriver.sequence_tokens(1024, 768) == 64 * 48 + 512
    assert Flux2KleinDriver.reference_tokens(1024, 768) == 64 * 48
    tok, _ = qwen3_tokenizer_and_config("4B")
    template = tok.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    assert Flux2KleinDriver.caption_tokens == 512 - len(tok(template)["input_ids"])


def test_downloads_are_pinned_with_checksums():
    seen = set()
    for dl in Flux2KleinDriver.downloads:
        assert re.fullmatch(r"[0-9a-f]{40}", dl.revision) and re.fullmatch(r"[0-9a-f]{64}", dl.sha256), dl.id
        assert dl.size > 0 and dl.kind_folder in ("diffusion_models", "text_encoders", "vae")
        assert dl.variant in ("", "base", "distilled")
        seen.add((dl.field, dl.variant))
        assert ("non-commercial" in dl.label) == ("9B" in dl.label and dl.field == "diffusion_model"), dl.label
    for key in seen:
        assert sum(d.recommended for d in Flux2KleinDriver.downloads if (d.field, d.variant) == key) == 1, key


# --- the text encoder -------------------------------------------------------------
TINY_QWEN = dict(vocab_size=151936, hidden_size=64, intermediate_size=96, num_hidden_layers=36, num_attention_heads=2,
                 num_key_value_heads=1, head_dim=32, max_position_embeddings=1024, tie_word_embeddings=True,
                 rope_theta=1e6, rms_norm_eps=1e-6)


@pytest.mark.parametrize("layout", ["folder", "index", "file"])
def test_text_encoder_reads_layers_9_18_27_of_the_padded_prompt(tmp_path, monkeypatch, layout):
    """A tiny Qwen3 saved as Hugging Face shards (or one ComfyUI-style file): the driver loads only the first 27
    layers, without the final norm, and returns hidden states 9, 18 and 27 of the 512-token padded prompt, side
    by side, as the reference's Qwen3Embedder does with the whole model."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from dainamo.engine.hf_assets import qwen3_tokenizer_and_config

    tok, _ = qwen3_tokenizer_and_config("4B")
    cfg = Qwen3Config(**TINY_QWEN)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    full = Qwen3ForCausalLM(cfg).eval()
    sd = {k: v.contiguous() for k, v in full.state_dict().items() if k != "lm_head.weight"}
    if layout == "file":
        path = str(tmp_path / "qwen_3_tiny.safetensors")
        save_file(sd, path)
    else:
        keys = sorted(sd)
        parts = {"model-00001-of-00002.safetensors": keys[: len(keys) // 2],
                 "model-00002-of-00002.safetensors": keys[len(keys) // 2:]}
        for name, ks in parts.items():
            save_file({k: sd[k] for k in ks}, str(tmp_path / name))
        (tmp_path / comfy_io.INDEX_NAME).write_text(json.dumps(
            {"weight_map": {k: n for n, ks in parts.items() for k in ks}}))
        path = str(tmp_path) if layout == "folder" else str(tmp_path / comfy_io.INDEX_NAME)
    monkeypatch.setitem(K.QWEN3_WIDTHS, 64, "4B")
    monkeypatch.setattr(K, "qwen3_tokenizer_and_config", lambda size: (tok, cfg))
    d = Flux2KleinDriver({"text_encoder": path}, CPU)
    d.dtype = torch.float32
    d.load_text_encoder()
    assert len(d.text_encoder.layers) == 27
    prompt = "a red apple on a wooden table"
    got = d.encode_prompt(prompt)

    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        out = full(**enc, output_hidden_states=True, use_cache=False)
    want = torch.stack([out.hidden_states[k] for k in (9, 18, 27)], dim=1)  # (b, 3, L, d) -> (b, L, 3 d)
    want = want.permute(0, 2, 1, 3).reshape(1, 512, -1)[0].bfloat16()
    assert got.shape == (512, 3 * 64) and got.dtype == torch.bfloat16
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=1e-2)
    assert int(enc["attention_mask"].sum()) < 40  # mostly padding, which the DiT still reads


def test_profile_checks_folders_sizes_and_variant(tmp_path):
    def qwen(folder, hidden, inter, vision=False):
        folder.mkdir()
        t = {"model.embed_tokens.weight": ("BF16", (151936, hidden)),
             "model.layers.0.self_attn.q_norm.weight": ("BF16", (128,)),
             "model.layers.35.mlp.down_proj.weight": ("BF16", (hidden, inter))}
        if vision:
            t["model.visual.patch_embed.proj.weight"] = ("BF16", (1,))
        header_only_safetensors(folder / "model-00001-of-00001.safetensors", t)
        (folder / comfy_io.INDEX_NAME).write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors"
                                                                             for k in t}}))
        return str(folder)

    def dit(name, hidden, ctx):
        return header_only_safetensors(tmp_path / name, {
            "double_stream_modulation_img.lin.weight": ("BF16", (6 * hidden, hidden)),
            "single_stream_modulation.lin.weight": ("BF16", (3 * hidden, hidden)),
            "txt_in.weight": ("BF16", (hidden, ctx))})

    vae = header_only_safetensors(tmp_path / "flux2-vae.safetensors", {"bn.running_mean": ("F32", (128,)),
                                                                       "encoder.conv_in.weight": ("F32", (128, 3, 3, 3))})
    q4, q8 = qwen(tmp_path / "q4", 2560, 9728), qwen(tmp_path / "q8", 4096, 12288)
    d4, d9 = dit("flux-2-klein-base-4b.safetensors", 3072, 7680), dit("flux-2-klein-9b.safetensors", 4096, 12288)

    def check(**profile):
        return {r["field"]: r for r in Flux2KleinDriver.validate_profile({"vae": vae, **profile})}

    r = check(diffusion_model=d4, text_encoder=q4)
    assert r["text_encoder"]["level"] == "ok" and "1 file" in r["text_encoder"]["message"]
    assert all(x["level"] == "ok" for x in r.values())
    assert check(diffusion_model=d9, text_encoder=os.path.join(q8, comfy_io.INDEX_NAME),
                 variant="distilled")["text_encoder"]["level"] == "ok"
    r = check(diffusion_model=d9, text_encoder=q4, variant="distilled")
    assert r["text_encoder"]["level"] == "error" and "takes Qwen3-8B" in r["text_encoder"]["message"]
    assert check(diffusion_model=d4, text_encoder=qwen(tmp_path / "vl", 2560, 9728, vision=True))[
        "text_encoder"]["level"] == "error"  # Qwen3-VL
    assert check(diffusion_model=dit("flux2-dev.safetensors", 6144, 15360), text_encoder=q4)[
        "diffusion_model"]["level"] == "error"
    assert check(diffusion_model=d9, text_encoder=q8)["variant"]["level"] == "warning"  # "base", by its name distilled
    assert "variant" not in check(diffusion_model=d4, text_encoder=q4, variant="base")


def test_variant_sets_the_sampling_defaults():
    base, distilled = Flux2KleinDriver({}, CPU), Flux2KleinDriver({"variant": "distilled"}, CPU)
    assert (base.sample_steps(), base.cfg_scale(None)) == (28, 4.0)
    assert (distilled.sample_steps(), distilled.cfg_scale("")) == (4, 1.0)
    chosen = Flux2KleinDriver({"variant": "distilled", "sample_steps": 8, "sample_cfg": 2.5}, CPU)
    assert (chosen.sample_steps(), chosen.cfg_scale(2.5)) == (8, 2.5)


# --- the driver on a tiny model ----------------------------------------------------
def tiny_driver(**profile):
    d = Flux2KleinDriver(profile, CPU)
    d.dtype = torch.float32
    return d


def test_training_reaches_every_lora_target_and_sliders_work():
    torch.manual_seed(0)
    model = small_model()
    net = LoRANetwork(model, Flux2KleinDriver.lora_targets, rank=4, alpha=4)
    per_double = 8  # qkv, proj and the two MLP linears of each stream
    assert len(net.modules) == per_double * SMALL["depth"] + 2 * SMALL["depth_single_blocks"]
    for m in net.modules.values():
        torch.nn.init.normal_(m.lora_B.weight, std=0.01)
    model.gradient_checkpointing = True
    model.train()
    d = tiny_driver()
    cond = torch.randn(9, SMALL["context_in_dim"]).bfloat16()
    payloads = [{"latent": torch.randn(128, 3, 4)}, {"latent": torch.randn(128, 3, 4),
                                                     "refs": [torch.randn(128, 2, 2)]}]
    loss = d.training_loss(model, payloads, [{"text": cond}, {"text": cond}], {})
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    assert "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight" in net.state_dict()
    assert "diffusion_model.single_blocks.1.linear2.lora_B.weight" in net.state_dict()
    batch = d.slider_batch(payloads, {})
    v = d.slider_velocity(model, batch, cond)
    assert v.shape == (2, 128, 3, 4) and v.dtype == torch.float32


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, img, img_ids, txt, txt_ids, t, ref=None, ref_ids=None):
        self.calls.append(dict(img=img.shape, txt=txt.shape, t=t.clone(), img_ids=img_ids, txt_ids=txt_ids,
                               ref=None if ref is None else ref.shape, ref_ids=ref_ids))
        return torch.zeros_like(img)


def test_edit_items_carry_their_references_and_others_train_text_to_image():
    d = tiny_driver()
    m = Recorder()
    lat = torch.randn(128, 3, 4)
    edit = {"latent": lat, "refs": [torch.randn(128, 3, 4), torch.randn(128, 2, 2)]}
    cond = {"text": torch.randn(512, 48).bfloat16()}
    d.training_loss(m, [edit, {"latent": lat}, edit, {"latent": lat}], [cond] * 4, {}, sigma=0.5)
    assert len(m.calls) == 2  # one forward per reference layout
    e, t = m.calls
    assert e["img"] == (2, 12, 128) and e["ref"] == (2, 12 + 4, 128) and t["ref"] is None and t["img"][0] == 2
    assert e["txt"] == (2, 512, 48)
    # ids: the image at (0, h, w, 0), the references at t = 10 and 20, the text at (0, 0, 0, l)
    assert torch.equal(e["img_ids"][0], K.image_ids(3, 4))
    assert e["img_ids"][0, 5].tolist() == [0, 1, 1, 0]
    assert torch.equal(e["ref_ids"][0], torch.cat([K.image_ids(3, 4, 10), K.image_ids(2, 2, 20)]))
    assert e["txt_ids"][0, 7].tolist() == [0, 0, 0, 7] and e["txt_ids"].shape == (2, 512, 4)
    assert torch.equal(e["t"], torch.full((2,), 0.5))


def test_references_are_scaled_to_the_target_area(tmp_path):
    from PIL import Image

    for (w, h), target in (((1920, 1080), 1024 * 1024), ((300, 500), 512 * 512), ((64, 48), 256 * 256)):
        rw, rh = K.reference_size(w, h, target)
        assert rw % 16 == 0 and rh % 16 == 0
        assert abs(rw / rh - w / h) / (w / h) < 0.05 and abs(rw * rh - target) / target < 0.06
    p = tmp_path / "ref.png"
    Image.new("RGBA", (300, 200), (255, 0, 0, 0)).save(p)  # fully transparent: over white
    x = K.prepare_reference(str(p), 256 * 256)
    assert x.shape == (1, 3, 208, 320) and float(x.min()) == pytest.approx(1.0)


class Oracle(torch.nn.Module):
    """Predicts the exact velocity (noise - clean) toward fixed clean tokens, reading sigma = t."""

    def __init__(self, x0):
        super().__init__()
        self.x0 = x0

    def forward(self, img, img_ids, txt, txt_ids, t, ref=None, ref_ids=None):
        return (img.float() - self.x0) / t.float().view(-1, 1, 1)


def test_sampler_lands_on_the_clean_latent_given_the_true_velocity():
    d = tiny_driver(sample_steps=12)
    x0 = torch.randn(1, 4 * 6, 128)
    cond = torch.randn(512, 48)
    out = d.sample(Oracle(x0), cond, cond, 96, 64, 12, 4.0, 3)
    assert out.shape == (1, 128, 4, 6)
    assert torch.allclose(K.tokens(out), x0, atol=1e-4)


def test_cfg_keeps_references_in_both_branches(tmp_path):
    from PIL import Image

    d = tiny_driver()
    p = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (10, 200, 30)).save(p)

    class VAE:
        def encode(self, pixels):
            return torch.zeros(128, pixels.shape[-2] // 16, pixels.shape[-1] // 16)

    d.vae = VAE()
    m = Recorder()
    x, velocity = d.prepare_sampling(m, torch.randn(512, 48), torch.randn(512, 48), 64, 32, 4.0, 0, [str(p)])
    velocity(x, 0.7)
    assert [c["ref"] for c in m.calls] == [(1, 9, 128), (1, 9, 128)]  # 64 x 32 = 2048 px as a square: 48 x 48, 3 x 3 tokens
    m.calls.clear()
    x, velocity = d.prepare_sampling(m, torch.randn(512, 48), torch.randn(512, 48), 64, 32, 1.0, 0, None)
    velocity(x, 0.7)
    assert len(m.calls) == 1 and m.calls[0]["ref"] is None  # CFG 1: one branch

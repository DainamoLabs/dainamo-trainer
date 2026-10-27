"""Layer offloading through the shared loader, and the VRAM projection read from file headers.

    .venv/bin/python -m pytest tests/test_memory.py -v
"""

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

from dainamo.engine import memory
from dainamo.engine.families.base import FamilyDriver, FamilyInfo, MemoryProfile, count_blocks
from dainamo.engine.lora import LoRANetwork

D, N = 64, 6


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.Linear(d, d, bias=False)  # int8 in the file
        self.mlp = nn.Linear(d, d)  # bf16

    def forward(self, x):
        return x + self.mlp(torch.relu(self.attn(self.norm(x))))


class TinyDiT(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.proj_in = nn.Linear(d, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n))
        self.gradient_checkpointing = False

    def forward(self, x):
        x = self.proj_in(x)
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.gradient_checkpointing and torch.is_grad_enabled() else b(x)
        return x


class TinyDriver(FamilyDriver):
    info = FamilyInfo(id="tiny", label="Tiny", media=["image"], profile_fields=[])
    lora_targets = ["blocks.*.attn", "blocks.*.mlp"]
    memory = MemoryProfile(saved=1000, full=10_000, work=5000, infer=2000, vae_encode=(0.0, 10.0),
                           vae_decode=(0.0, 20.0))

    def build_transformer(self, shapes):
        with torch.device("meta"):
            return TinyDiT(shapes["proj_in.weight"][0], count_blocks(shapes, "blocks"))


def write_model(path, full_checkpoint=False, dtype=torch.bfloat16):
    torch.manual_seed(0)
    ref = TinyDiT(D, N)
    sd = {}
    for k, v in ref.state_dict().items():
        if k.endswith("attn.weight"):
            scale = v.abs().amax(dim=1) / 127
            sd[k] = torch.round(v / scale[:, None]).clamp(-127, 127).to(torch.int8)
            sd[k + "_scale"] = scale.float()
            sd[k.removesuffix(".weight") + ".comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
        else:
            sd[k] = v.to(dtype)
    if full_checkpoint:  # a ComfyUI checkpoint bundling a text encoder
        sd = {"model.diffusion_model." + k: v for k, v in sd.items()}
        sd["text_encoders.te.weight"] = torch.zeros(128, 128, dtype=torch.bfloat16)
    save_file(sd, str(path))
    return str(path)


def test_block_rounding_matches_between_loader_and_projection():
    d = TinyDriver({}, torch.device("cpu"))
    for n in range(1, 61):
        for pct in range(101):
            d.offload = pct / 100
            assert d.streamed_blocks(n) == memory.streamed_for({"blocks": n}, pct), (n, pct)
    d.offload = 0.25
    assert d.streamed_blocks(2) == 1  # halves round up, as in the web UI


@pytest.mark.parametrize("full,dtype", [(False, torch.bfloat16), (True, torch.bfloat16), (False, torch.float32)])
def test_header_sizes_match_the_loaded_model(tmp_path, full, dtype):
    """Also for an fp32 file, which loads as bf16 (cast as it's read)."""
    path = write_model(tmp_path / "m.safetensors", full, dtype)
    resident, blocks, shapes = memory.model_weights(TinyDriver, path)
    assert len(blocks) == N and "text_encoders.te.weight" not in shapes
    d = TinyDriver({"diffusion_model": path}, torch.device("cpu"), log=lambda *_: None)
    model = d.load_transformer()

    def size(m):
        return sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
    assert resident == size(model) - sum(size(b) for b in model.blocks)
    assert blocks == [size(b) for b in model.blocks]
    assert memory.lora_params(shapes, TinyDriver.lora_targets, 4) == N * 2 * 4 * (D + D)


def test_projection_adds_up(tmp_path):
    path = write_model(tmp_path / "m.safetensors")
    settings = {"resolution": 1024, "batch_size": 2, "rank": 4, "optimizer": "adamw", "sample_width": 512,
                "sample_height": 512}
    p = memory.project(TinyDriver, {"diffusion_model": path}, settings)
    assert [ph["id"] for ph in p["phases"]] == ["training", "sampling", "latents"]  # no text encoder file
    train = dict(p["phases"][0]["parts"])
    tokens = 2 * ((1024 // 16) ** 2 + 256)
    assert train["Activations"] == pytest.approx(tokens * (N * 1000 + 5000) / memory.GB)
    params = N * 2 * 4 * (D + D)
    assert train["LoRA"] == pytest.approx(params * 8 / memory.GB) and train["Optimizer"] == pytest.approx(params * 8 / memory.GB)
    assert p["adapter_params"] == params
    # a LoKr counts its two matrices per layer (D splits as 2 x D/2 at factor 2)
    lokr = memory.project(TinyDriver, {"diffusion_model": path}, {**settings, "network_type": "lokr", "lokr_factor": 2})
    assert lokr["adapter_params"] == N * 2 * (2 * 2 + (D // 2) ** 2)
    assert dict(lokr["phases"][0]["parts"])["LoKr"] == pytest.approx(lokr["adapter_params"] * 8 / memory.GB)
    with_te = memory.project(TinyDriver, {"diffusion_model": path, "text_encoder": path}, {**settings, "network_type": "lokr"})
    assert with_te["phases"][-1]["label"] == "Encoding captions"  # still a training job's projection
    full = memory.weights_gb(p, 0)
    assert full == pytest.approx(p["model_gb"])
    assert memory.weights_gb(p, N) == pytest.approx(p["resident_gb"] + p["inflight_gb"])
    peaks = [memory.peak_gb(p, k) for k in range(1, N + 1)]
    assert all(a >= b for a, b in zip(peaks, peaks[1:]))
    no_ckpt = memory.project(TinyDriver, {"diffusion_model": path}, {**settings, "gradient_checkpointing": False})
    assert dict(no_ckpt["phases"][0]["parts"])["Activations"] == pytest.approx(tokens * (N * 10_000 + 5000) / memory.GB)
    assert any("checkpointing is off" in n for n in no_ckpt["notes"])
    multi = memory.project(TinyDriver, {"diffusion_model": path}, {**settings, "resolution": [512, 1024]})
    assert multi["phases"] == p["phases"]  # several resolutions peak at the largest


def test_a_training_adapter_counts_every_tensor_it_loads(tmp_path):
    """A LoKr training adapter (or whole changes) counts too, not only a LoRA's .weight keys; alphas don't."""
    k = "diffusion_model.blocks.0.attn"
    lokr, lora, whole = tmp_path / "lokr.safetensors", tmp_path / "lora.safetensors", tmp_path / "whole.safetensors"
    save_file({f"{k}.lokr_w1": torch.zeros(2, 2), f"{k}.lokr_w2": torch.zeros(32, 32, dtype=torch.bfloat16),
               f"{k}.alpha": torch.tensor(1.0)}, str(lokr))
    save_file({f"{k}.lora_A.weight": torch.zeros(4, D), f"{k}.lora_B.weight": torch.zeros(D, 4),
               f"{k}.alpha": torch.tensor(4.0)}, str(lora))
    save_file({f"{k}.diff": torch.zeros(D, D), f"{k}.diff_b": torch.zeros(D)}, str(whole))
    assert memory.adapter_bytes(str(lokr)) == (2 * 2 + 32 * 32) * 2  # in bf16
    assert memory.adapter_bytes(str(lora)) == 2 * 4 * D * 2
    assert memory.adapter_bytes(str(whole)) == (D * D + D) * 2
    assert memory.adapter_bytes("") == 0 and memory.adapter_bytes(str(tmp_path / "gone.safetensors")) is None
    p = memory.project(TinyDriver, {"diffusion_model": write_model(tmp_path / "m.safetensors"),
                                    "training_adapter": str(lokr)}, {"resolution": 512, "rank": 4})
    params = N * 2 * 4 * (D + D)
    assert dict(p["phases"][0]["parts"])["LoRA"] == pytest.approx((params * 8 + (2 * 2 + 32 * 32) * 2) / memory.GB)


class TinyVideoDriver(TinyDriver):
    info = FamilyInfo(id="tinyvideo", label="Tiny video", media=["image", "video"], profile_fields=[])

    @classmethod
    def sequence_tokens(cls, width, height, frames=1):
        return (width // 16) * (height // 16) * frames + 256


def test_projection_takes_each_datasets_size_and_kind(tmp_path):
    path = write_model(tmp_path / "m.safetensors")
    base = {"batch_size": 1, "rank": 4, "num_frames": 9, "sample_width": 64, "sample_height": 64, "sample_frames": 1}

    def activations(settings):
        p = memory.project(TinyVideoDriver, {"diffusion_model": path}, {**base, **settings})
        return dict(p["phases"][0]["parts"])["Activations"] * memory.GB / (N * 1000 + 5000)

    # stills at 1024 and clips at 512: the clips (32 x 32 x 9 tokens) are the bigger step
    mixed = {"resolution": [512, 1024], "datasets": [{"resolution": [768, 1024], "video": False},
                                                     {"resolution": 512, "video": True}]}
    assert activations(mixed) == pytest.approx(32 * 32 * 9 + 256)
    assert activations({**mixed, "datasets": mixed["datasets"][:1]}) == pytest.approx(64 * 64 + 256)
    # without per-dataset sizes (older callers), everything counts as a clip at the largest size
    assert activations({"resolution": [512, 1024]}) == pytest.approx(64 * 64 * 9 + 256)


def test_projection_samples_the_largest_prompt_card(tmp_path):
    path = write_model(tmp_path / "m.safetensors")
    base = {"batch_size": 1, "rank": 4, "num_frames": 9, "sample_width": 64, "sample_height": 64, "sample_frames": 1}

    def working(cards):
        p = memory.project(TinyVideoDriver, {"diffusion_model": path}, {**base, "sample_prompts": cards})
        return dict(next(ph for ph in p["phases"] if ph["id"] == "sampling")["parts"])["Working memory"]

    still = working([{"prompt": "a", "frames": ""}])
    assert still == working([])  # blank fields take the defaults
    assert working([{"prompt": "a", "frames": "9"}]) > working([{"prompt": "a", "frames": "5"}]) > still
    assert working([{"prompt": "a", "width": 128, "height": "128"}]) > still
    assert working([{"prompt": "", "frames": 9}]) == still  # an empty card isn't rendered
    assert working([{"prompt": "a", "frames": "9e"}]) == still  # a half-typed field is skipped, not an error
    assert working([{"prompt": "a", "frames": "1e999"}]) == still  # so is one too big to be a number
    # sized as training renders it: rounded down to the model's multiple (16)
    assert working([{"prompt": "a", "width": 79, "height": 64}]) == still


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_offloaded_load_streams_the_last_blocks_and_matches(tmp_path):
    path = write_model(tmp_path / "m.safetensors")
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    x = torch.randn(3, D, device=dev, dtype=torch.bfloat16)
    results = []
    for offload in (0.0, 0.5):
        d = TinyDriver({"diffusion_model": path}, dev, log=lambda *_: None)
        d.offload = offload
        model = d.load_transformer()
        model.gradient_checkpointing = True
        if offload:
            assert d.swapper.swapped == [3, 4, 5]
            assert all(not t.is_cuda and t.is_pinned() for b in model.blocks[3:] for t in b.state_dict().values())
            assert all(t.is_cuda for b in model.blocks[:3] for t in b.state_dict().values())
        else:
            assert d.swapper is None
        torch.manual_seed(1)
        net = LoRANetwork(model, TinyDriver.lora_targets, 4, 4.0)
        for m in net.modules.values():
            nn.init.normal_(m.lora_B.weight, std=0.1)  # nonzero, so the LoRA path matters
        net.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out.float().pow(2).mean().backward()
        results.append((out.detach(), [p.grad.clone() for p in net.parameters()]))
    (a, ga), (b, gb) = results
    assert torch.equal(a, b)
    assert all(torch.allclose(g1, g2) for g1, g2 in zip(ga, gb))


class TinyText(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.embed_tokens = nn.Embedding(256, d)
        self.layers = nn.ModuleList(Block(d) for _ in range(n))
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for b in self.layers:
            x = b(x)
        return self.norm(x)


def write_text_encoder(path):
    """A text encoder file as ComfyUI saves one ("model." keys), with int8 attention like ``write_model``."""
    torch.manual_seed(2)
    sd = {}
    for k, v in TinyText(D, 4).state_dict().items():
        if k.endswith("attn.weight"):
            scale = v.abs().amax(dim=1) / 127
            sd["model." + k] = torch.round(v / scale[:, None]).clamp(-127, 127).to(torch.int8)
            sd["model." + k + "_scale"] = scale.float()
            sd["model." + k.removesuffix(".weight") + ".comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
        else:
            sd["model." + k] = v.to(torch.bfloat16)
    save_file(sd, str(path))
    return str(path)


class TinyTextDriver(TinyDriver):
    """Loads its text encoder the way the families do: to ``text_encoder_device``, then ``stream_text_encoder``."""

    def load_text_encoder(self):
        from dainamo.engine import comfy_io
        from dainamo.engine.fp8_linear import materialize

        loaded = comfy_io.load_state_dict(self.profile["text_encoder"], strip_prefix="model.", only_prefixed=True,
                                          device=self.text_encoder_device)
        with torch.device("meta"):
            model = TinyText(D, 4)
        materialize(model, loaded, self.dtype, keep_fp8=True, device=None)
        self.text_encoder = self.stream_text_encoder(model).eval()

    @torch.no_grad()
    def encode_prompt(self, prompt):
        ids = torch.tensor([ord(c) for c in prompt], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.text_encoder(ids).float().cpu()


class TinyMusicDriver(TinyTextDriver):
    info = FamilyInfo(id="tinymusic", label="Tiny music", media=["image"], profile_fields=[],
                      streams_text_encoder=False)


def test_low_vram_projects_only_the_text_encoders_unstreamed_part(tmp_path, monkeypatch):
    path, te = write_model(tmp_path / "m.safetensors"), write_text_encoder(tmp_path / "te.safetensors")
    profile, settings = {"diffusion_model": path, "text_encoder": te}, {"resolution": 512}

    def captions(driver, **extra):
        p = memory.project(driver, profile, {**settings, **extra})
        return dict(next(ph for ph in p["phases"] if ph["id"] == "captions")["parts"]), p["notes"]

    whole, notes = captions(TinyTextDriver)
    assert whole["Text encoder"] == pytest.approx(memory.file_gpu_bytes(te) / memory.GB) and not notes
    layer = sum(memory.gpu_bytes(k, dt, shape) for k, (dt, shape) in memory.tensors(te).items()
                if k.startswith("model.layers.0."))
    assert layer > 0
    streamed, notes = captions(TinyTextDriver, low_vram=True)
    assert streamed["Text encoder"] == pytest.approx(whole["Text encoder"] - 4 * layer / memory.GB)
    assert streamed["Streamed parts"] == pytest.approx(2 * layer / memory.GB)  # the running one and the next
    assert streamed["Working memory"] == whole["Working memory"]
    assert any(n.startswith("Low VRAM") for n in notes)
    assert captions(TinyMusicDriver, low_vram=True) == (whole, [])  # a family that doesn't stream it
    # a large part outside the layers (the embedding table, at this size) streams too, on its own
    embed = 256 * D * 2
    monkeypatch.setattr(memory, "BIG_MODULE", embed)
    big, _ = captions(TinyTextDriver, low_vram=True)
    assert big["Text encoder"] == pytest.approx(streamed["Text encoder"] - embed / memory.GB)
    assert embed > 2 * layer and big["Streamed parts"] == pytest.approx(embed / memory.GB)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_low_vram_text_encoder_encodes_the_same_with_less_on_the_gpu(tmp_path, monkeypatch):
    from dainamo.engine import offload

    te = write_text_encoder(tmp_path / "te.safetensors")
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    out, on_gpu = {}, {}
    # whole, the layers streamed, and the embedding table too (a large part, at this size)
    for mode, swappers in (("whole", 0), ("layers", 1), ("big", 2)):
        if mode == "big":
            monkeypatch.setattr(offload, "BIG_MODULE", 256 * D * 2)
        d = TinyTextDriver({"text_encoder": te}, dev, log=lambda *_: None)
        d.low_vram = mode != "whole"
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(dev)
        d.load_text_encoder()
        on_gpu[mode] = torch.cuda.memory_allocated(dev) - before
        assert len(d.text_swappers) == swappers
        out[mode] = [d.encode_prompt(p) for p in ("a fox in the snow", "x")]
        d.unload_text_encoder()
        assert d.text_encoder is None and d.text_swappers == []
    assert all(torch.equal(a, b) for a, b in zip(out["whole"], out["layers"]))
    assert all(torch.equal(a, b) for a, b in zip(out["whole"], out["big"]))
    assert on_gpu["big"] < on_gpu["layers"] < on_gpu["whole"]
    # a family that doesn't stream its text encoder ignores the setting
    d = TinyMusicDriver({"text_encoder": te}, dev, log=lambda *_: None)
    d.low_vram = True
    d.load_text_encoder()
    assert d.text_swappers == [] and d.text_encoder_device == dev
    d.unload_text_encoder()


def test_low_vram_samples_in_two_steps(tmp_path):
    path = write_model(tmp_path / "m.safetensors")
    vae = tmp_path / "vae.safetensors"
    save_file({"decoder.weight": torch.zeros(512, 512, dtype=torch.bfloat16)}, str(vae))
    profile = {"diffusion_model": path, "vae": str(vae)}
    settings = {"resolution": 512, "sample_width": 512, "sample_height": 512}
    vae_gb = 512 * 512 * 2 / memory.GB

    def phases(**extra):
        p = memory.project(TinyDriver, profile, {**settings, **extra})
        return p, {ph["id"]: ph for ph in p["phases"]}

    _, off = phases()
    assert "decoding" not in off and dict(off["sampling"]["parts"])["VAE"] == pytest.approx(vae_gb)
    on_proj, on = phases(low_vram=True)
    # denoising without the VAE and the optimizer state, then decoding without the diffusion model and LoRA
    assert "VAE" not in dict(on["sampling"]["parts"]) and "Optimizer" in dict(on["sampling"]["ram"])
    assert on["decoding"]["model"] is False and dict(on["decoding"]["parts"])["VAE"] == pytest.approx(vae_gb)
    assert memory.phase_totals(on_proj, N)["sampling"] < memory.phase_totals(phases()[0], N)["sampling"]
    # a sample card with a first frame or reference images needs the VAE while sampling
    _, cond = phases(low_vram=True, sample_prompts=[{"prompt": "a fox", "images": ["fox.png"]}])
    assert dict(cond["sampling"]["parts"])["VAE"] == pytest.approx(vae_gb)
    # the Sandbox: the VAE waits in system RAM unless input images need it
    gen = {"width": 512, "height": 512}
    sample = {ph["id"]: ph for ph in memory.project(TinyDriver, profile, {**gen, "low_vram": True}, "sample")["phases"]}
    assert "VAE" not in dict(sample["generation"]["parts"]) and dict(sample["generation"]["ram"])["VAE"] > 0
    assert sample["decoding"]["model"] is False
    sample = {ph["id"]: ph for ph in memory.project(TinyDriver, profile, {**gen, "low_vram": True, "images": ["a.png"]},
                                                    "sample")["phases"]}
    assert "VAE" in dict(sample["generation"]["parts"])


def test_low_vram_keeps_saved_activations_in_system_ram(tmp_path):
    path = write_model(tmp_path / "m.safetensors")

    def training(**settings):
        p = memory.project(TinyDriver, {"diffusion_model": path}, {"resolution": 512, **settings})
        return next(ph for ph in p["phases"] if ph["id"] == "training")

    tokens, saved = TinyDriver.sequence_tokens(512, 512, 1), TinyDriver.memory.saved
    off, on = training(), training(low_vram=True)
    # only a few blocks' saved activations stay on the GPU; all of them are in RAM
    activations = [dict(ph["parts"])["Activations"] for ph in (off, on)]
    assert activations[0] - activations[1] == pytest.approx(tokens * (N - memory.ACTIVATIONS_IN_FLIGHT) * saved / memory.GB)
    assert dict(on["ram"]) == {"Saved activations": pytest.approx(tokens * N * saved / memory.GB)}
    assert not off.get("ram")
    # without gradient checkpointing there's nothing to offload
    assert not training(low_vram=True, gradient_checkpointing=False).get("ram")


def test_ram_per_phase_counts_streamed_blocks_and_parked_parts(tmp_path):
    path = write_model(tmp_path / "m.safetensors")
    p = memory.project(TinyDriver, {"diffusion_model": path}, {"resolution": 512, "low_vram": True})
    base, blocks = p["process_ram_gb"], p["block_gb"]
    phases = {ph["id"]: ph for ph in p["phases"]}
    base_step = base + sum(v for _, v in phases["training"]["ram"])  # the saved activations
    assert memory.ram_totals(p, 0)["training"] == pytest.approx(base_step)
    assert memory.ram_totals(p, 2)["training"] == pytest.approx(base_step + sum(blocks[-2:]))
    # while samples decode, all of the diffusion model, the LoRA and the optimizer state are in RAM
    held = sum(v for _, v in phases["decoding"]["ram"])
    assert held > 0
    assert memory.ram_totals(p, 0)["decoding"] == pytest.approx(base + p["model_gb"] + held)

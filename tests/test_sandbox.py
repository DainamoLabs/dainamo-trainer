"""Sandbox: request validation per family/workflow, the GPU rules shared with the training queue, and
the inference LoRA stack.

    .venv/bin/python -m pytest tests/test_sandbox.py -v
"""

import io
import os
import sys
import tempfile

if "dainamo.server.db" not in sys.modules:
    os.environ["DAINAMO_DATA"] = tempfile.mkdtemp(prefix="dainamo-test-")
    os.environ["DAINAMO_ALLOWED_HOSTS"] = "127.0.0.1,localhost,testserver"

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from dainamo.engine.lora import AdapterStack  # noqa: E402
from dainamo.server import db  # noqa: E402
from dainamo.server import queue as qmod  # noqa: E402
from dainamo.server import sandbox as sbx  # noqa: E402
from dainamo.server.app import app  # noqa: E402
from dainamo.server.routes.sandbox import SandboxIn, check_sandbox  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        c.put("/api/settings", json={"queue": {"gpus": [99]}})  # no GPU to reserve or occupy
        yield c


@pytest.fixture
def profiles(client):
    make = lambda name, fam, data: client.post("/api/profiles", json={"name": name, "family": fam, "data": data}).json()["id"]  # noqa: E731
    return {"qwen": make("q", "qwen_image_2", {}), "krea": make("k", "krea2", {}),
            "fl2va": make("m1", "minimax_h3", {"variant": "fl2va"}), "ref2va": make("m2", "minimax_h3", {"variant": "ref2va"})}


def check(**kw):
    return check_sandbox(SandboxIn(**{"prompt": "a cat", **kw}))


def test_workflows_follow_the_model(profiles, tmp_path, monkeypatch):
    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    for pid, wf in ((profiles["krea"], "t2v"), (profiles["krea"], "edit"), (profiles["qwen"], "i2v"),
                    (profiles["fl2va"], "t2i"), (profiles["ref2va"], "i2v")):
        with pytest.raises(HTTPException, match="isn't available"):
            check(profile_id=pid, workflow=wf, images=[str(img)])
    with pytest.raises(HTTPException, match="needs 1"):
        check(profile_id=profiles["qwen"], workflow="edit")
    with pytest.raises(HTTPException, match="takes no images"):
        check(profile_id=profiles["qwen"], workflow="t2i", images=[str(img)])
    with pytest.raises(HTTPException, match="needs 1 first frame"):
        check(profile_id=profiles["fl2va"], workflow="i2v", images=[str(img), str(img)])
    _, p, gpu = check(profile_id=profiles["qwen"], workflow="edit", images=[str(img)], width=1000, height=700)
    assert (p["width"], p["height"], p["frames"], gpu) == (992, 672, 1, 1)  # /32, the last GPU by default
    _, p, _ = check(profile_id=profiles["fl2va"], workflow="i2v", images=[str(img)], frames=39, sampler="heun",
                    scheduler="beta", cfg=4)
    assert (p["frames"], p["sampler"], p["scheduler"]) == (39, "euler", "default")  # MiniMax: released sampler
    # fl2va wasn't trained on references, but takes them
    assert check(profile_id=profiles["fl2va"], workflow="r2v", images=[str(img)] * 2, frames=39)[1]["workflow"] == "r2v"
    _, p, _ = check(profile_id=profiles["krea"], workflow="t2i", seed=-1, frames=40)
    assert p["seed"] >= 0 and p["frames"] == 1 and p["low_vram"] is False  # off unless asked for
    assert check(profile_id=profiles["fl2va"], workflow="t2v", low_vram=True)[1]["low_vram"] is True
    for bad in ({"steps": 0}, {"cfg": 99}, {"sampler": "ddim"}, {"scheduler": "karras"}, {"shift": 0},
                {"count": 9}, {"width": 64}, {"prompt": "  "}, {"loras": [{"path": "/nope.safetensors"}]},
                {"loras": [{"path": str(img)}]}):
        with pytest.raises(HTTPException):
            check(**{"profile_id": profiles["krea"], "workflow": "t2i", **bad})


class FakeProc:
    stdin = io.StringIO()

    def poll(self):
        return None


def _queue_job(gpu="auto"):
    jid = db.new_id()
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES "
               "(?,?,?,?,?,?,?,?)", (jid, "j", "queued", "krea2", gpu, db.now(), db.now(), "{}"))
    return jid


@pytest.fixture
def clean_jobs():
    yield
    db.execute("UPDATE jobs SET status='stopped' WHERE status IN ('queued','running')")


def test_two_gpus_the_queue_skips_the_reserved_gpu(client, monkeypatch, clean_jobs):
    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    monkeypatch.setattr(sbx.sandbox, "proc", FakeProc())
    monkeypatch.setattr(sbx.sandbox, "gpu", 0)
    spawned = []
    monkeypatch.setattr(qmod.scheduler, "_spawn", lambda job, gpu: spawned.append(gpu))
    monkeypatch.setattr(db, "get_settings", lambda: {"queue": {"mode": "sequential", "gpus": [0, 1], "max_per_gpu": 1}})
    _queue_job()
    pinned = _queue_job(gpu="0")
    qmod.scheduler._launch()
    assert spawned == [1]  # sequential would use GPU 0; the Sandbox holds it, so GPU 1
    db.execute("UPDATE jobs SET status='stopped' WHERE id<>?", (pinned,))
    spawned.clear()
    qmod.scheduler._launch()
    assert spawned == []  # a job pinned to the reserved GPU waits for the release


def test_one_gpu_training_comes_first(client, monkeypatch, clean_jobs):
    monkeypatch.setattr(sbx, "gpu_count", lambda: 1)
    monkeypatch.setattr(sbx.sandbox, "proc", FakeProc())
    monkeypatch.setattr(sbx.sandbox, "gpu", 0)
    yielded, spawned = [], []
    monkeypatch.setattr(sbx.sandbox, "yield_gpu", lambda: yielded.append(True))
    monkeypatch.setattr(qmod.scheduler, "_spawn", lambda job, gpu: spawned.append(gpu))
    monkeypatch.setattr(db, "get_settings", lambda: {"queue": {"mode": "sequential", "gpus": [0], "max_per_gpu": 1}})
    assert sbx.sandbox.check_gpu(0) is None
    _queue_job()
    assert "busy with training" in sbx.sandbox.check_gpu(0)  # no new generations while a job waits
    qmod.scheduler._launch()
    assert yielded and not spawned  # the Sandbox unloads first; the job starts on a later tick


def test_reserve_refuses_a_training_gpu(client, monkeypatch, clean_jobs):
    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    jid = _queue_job()
    db.execute("UPDATE jobs SET status='running', assigned_gpu=1 WHERE id=?", (jid,))
    assert "GPU 1 is training" in sbx.sandbox.check_gpu(1)
    assert sbx.sandbox.check_gpu(0) is None


def test_status_endpoint(client):
    st = client.get("/api/sandbox").json()
    assert {"mode", "session", "samplers", "schedulers", "runs"} <= set(st)
    assert st["samplers"]["dpmpp_2m"] == "DPM++ 2M"


# ---- LoRA stack -----------------------------------------------------------
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)) for _ in range(2)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def lora_sd(prefix="diffusion_model.", rank=2, alpha=1.0, seed=0, kohya=False):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name in ("blocks.0.0", "blocks.1.1"):
        a, b = torch.randn(rank, 8, generator=g), torch.randn(8, rank, generator=g)
        if kohya:
            k = "lora_unet_" + name.replace(".", "_")
            sd |= {f"{k}.lora_down.weight": a, f"{k}.lora_up.weight": b, f"{k}.alpha": torch.tensor(alpha)}
        else:
            sd |= {f"{prefix}{name}.lora_A.weight": a, f"{prefix}{name}.lora_B.weight": b,
                   f"{prefix}{name}.alpha": torch.tensor(alpha)}
    return sd


def reference(model, x, loras):
    """Base model with each LoRA's delta W += strength * alpha/rank * B @ A folded in."""
    import copy

    m = copy.deepcopy(model)
    with torch.no_grad():
        for sd, strength in loras:
            for name in ("blocks.0.0", "blocks.1.1"):
                a, b = sd[f"diffusion_model.{name}.lora_A.weight"], sd[f"diffusion_model.{name}.lora_B.weight"]
                scale = float(sd[f"diffusion_model.{name}.alpha"]) / a.shape[0]
                m.get_submodule(name).weight += strength * scale * (b @ a)
    return m(x)


def test_adapter_stack_strengths_stacking_and_base_toggle():
    torch.manual_seed(0)
    model = Tiny().double()
    x = torch.randn(3, 8, dtype=torch.float64)
    base = model(x)
    stack = AdapterStack(model, torch.device("cpu"), torch.float64)
    one, two = lora_sd(seed=1), lora_sd(seed=2, alpha=4.0)
    report = stack.set([("one", one, 0.7), ("two", two, -0.4)])
    assert [r["matched"] for r in report] == [2, 2]
    assert torch.allclose(model(x), reference(_clone(model), x, [(one, 0.7), (two, -0.4)]))
    stack.set_enabled(False)
    assert torch.allclose(model(x), base)
    stack.set([("one", one, 1.0)])
    assert torch.allclose(model(x), reference(_clone(model), x, [(one, 1.0)]))
    stack.set([])
    assert torch.allclose(model(x), base)


def _clone(model):
    """A plain Tiny with the wrapped model's base weights."""
    t = Tiny().double()
    sd = {k.replace(".base.", "."): v for k, v in model.state_dict().items()}
    t.load_state_dict(sd)
    return t


def test_adapter_stack_reads_kohya_and_rejects_wrong_files():
    model = Tiny().double()
    x = torch.randn(3, 8, dtype=torch.float64)
    stack = AdapterStack(model, torch.device("cpu"), torch.float64)
    stack.set([("peft", lora_sd(seed=3), 1.0)])
    peft = model(x)
    stack.set([("kohya", lora_sd(seed=3, kohya=True), 1.0)])
    assert torch.allclose(model(x), peft)
    with pytest.raises(ValueError, match="no LoRA weights"):
        stack.set([("model", {"blocks.0.0.weight": torch.zeros(8, 8)}, 1.0)])
    with pytest.raises(ValueError, match="matches no layers"):
        stack.set([("other", lora_sd(prefix="diffusion_model.other_model."), 1.0)])


def test_a_failed_lora_set_leaves_nothing_applied(tmp_path, monkeypatch):
    """``AdapterStack.set`` clears what's applied before it can fail, so after a failure the LoRAs applied before
    are applied again when asked for, not taken as still there (which would render the base model as theirs)."""
    from dainamo.engine import sandbox

    monkeypatch.setattr(sandbox, "emit", lambda *a, **kw: None)
    session = sandbox.Session()
    calls = []

    class Adapters:
        def set(self, loras):
            calls.append([label for label, _, _ in loras])
            if "b.safetensors" in calls[-1]:
                raise ValueError("LoRA b.safetensors matches no layers of this model (wrong model family?)")
            return [{"label": label, "matched": 2, "total": 2} for label in calls[-1]]

    session.adapters = Adapters()
    monkeypatch.setattr(session, "_lora_file", lambda path: {})
    a, b = [], []
    for name, loras in (("a", a), ("b", b)):
        (tmp_path / f"{name}.safetensors").write_bytes(b"")
        loras.append({"path": str(tmp_path / f"{name}.safetensors"), "strength": 1.0})
    report = session.apply_loras(a)
    assert report == [{"label": "a.safetensors", "matched": 2, "total": 2}]
    assert session.apply_loras(a) == report and len(calls) == 1  # applied already
    with pytest.raises(ValueError, match="matches no layers"):
        session.apply_loras(b)
    assert session.lora_key is None and session.lora_report == []
    assert session.apply_loras(a) == report
    assert calls == [["a.safetensors"], ["b.safetensors"], ["a.safetensors"]]


def test_prompt_encodings_keep_the_most_recently_used(monkeypatch):
    from dainamo.engine import sandbox

    monkeypatch.setattr(sandbox, "emit", lambda *a, **kw: None)
    session = sandbox.Session()
    encoded = []

    class Driver:
        def encode_condition(self, prompt, images):
            encoded.append(prompt)
            return torch.zeros(4)

    session.driver, session.te_where = Driver(), "gpu"
    monkeypatch.setattr(session, "_park_text_encoder", lambda: None)
    for i in range(sandbox.TEXTS_KEPT):
        session.encode({f"p{i}": (f"p{i}", None)})
    session.encode({"p0": ("p0", None), "": ("", None)})  # p0 again (and a negative): now the most recent
    assert encoded[-1] == "" and encoded.count("p0") == 1
    assert len(session.texts) == sandbox.TEXTS_KEPT and "p1" not in session.texts  # the least recently used went
    assert list(session.texts)[-2:] == ["p0", ""]
    session.encode({"p1": ("p1", None)})
    assert encoded[-1] == "p1" and "p0" in session.texts and "p2" not in session.texts


def test_another_model_file_reloads_what_came_from_the_last_one(monkeypatch):
    """A request's ``model`` in place of the profile's: the diffusion model reloads, and so do a text encoder and VAE
    bundled in its file (YuE2's), with what that text encoder encoded; a family with its own files keeps them."""
    from dainamo.engine import sandbox

    monkeypatch.setattr(sandbox, "emit", lambda *a, **kw: None)
    session = sandbox.Session()
    unloaded = []

    class Driver:
        swapper = None
        bundled = {"text_encoder": ("text_encoders.model.", 2), "vae": ("vae.", 4)}

        def __init__(self):
            self.profile = {"diffusion_model": "a.safetensors"}

        def unload_text_encoder(self):
            unloaded.append(self.profile["diffusion_model"])

    session.driver, session.model = Driver(), nn.Linear(2, 2)
    session.te_where, session.vae_loaded, session.texts["x"] = "cpu", True, torch.zeros(1)
    session.use_model("a.safetensors")  # the same file: nothing changes
    assert session.model is not None and session.vae_loaded and session.texts
    session.use_model("b.safetensors")
    assert session.model is None and session.driver.profile["diffusion_model"] == "b.safetensors"
    assert unloaded == ["b.safetensors"] and session.te_where is None and not session.texts
    assert not session.vae_loaded and session.vae_parked is None
    Driver.bundled = {}
    session.te_where, session.vae_loaded, session.texts["x"] = "cpu", True, torch.zeros(1)
    session.use_model("c.safetensors")
    assert session.te_where == "cpu" and session.vae_loaded and session.texts and unloaded == ["b.safetensors"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_park_and_unpark_round_trip_with_block_swapping():
    from dainamo.engine.offload import BlockSwapper, park, unpark

    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    model = Tiny()
    model.blocks.extend([nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)) for _ in range(2)])
    model.requires_grad_(False)
    swapper = BlockSwapper(model.blocks, 2, dev, log=lambda *_: None)
    x = torch.randn(3, 8, device=dev)
    before = model(x)
    resident = sum(p.numel() * 4 for p in model.parameters() if p.is_cuda)
    allocated = torch.cuda.memory_allocated(dev)
    moved = park([model], swapper)
    assert moved and all(not p.is_cuda for p in model.parameters())
    assert allocated - torch.cuda.memory_allocated(dev) >= resident  # the GPU copies are freed
    unpark(moved)
    assert model.blocks[0][0].weight.is_cuda and not model.blocks[3][0].weight.is_cuda  # streamed blocks stay in RAM
    assert torch.equal(model(x), before)
    swapper.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_low_vram_keeps_the_vae_in_ram_until_a_render_decodes(monkeypatch):
    from dainamo.engine import offload, sandbox

    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    session = sandbox.Session()
    session.model = nn.Linear(8, 8).to(dev)  # the diffusion model
    seen = []

    class Driver:
        swapper, memory, bundled, profile = None, None, {}, {}
        vae = None

        def load_vae(self):
            self.vae = nn.Linear(8, 8).to(dev)

        def to_media(self, lat):
            seen.append((self.vae.weight.is_cuda, session.model.weight.is_cuda))
            return [self.vae(lat)]

    session.driver = d = Driver()
    session.ensure_models(vae=False)
    assert d.vae is None  # not loaded until a render decodes
    lat = torch.randn(1, 8, device=dev)
    media = session.decode(lat, True, 8, 8)
    assert seen[-1] == (True, True)  # room for both: the diffusion model stays
    assert torch.allclose(media, nn.functional.linear(lat, d.vae.weight.to(dev), d.vae.bias.to(dev)))
    assert not d.vae.weight.is_cuda and session.vae_parked  # back in system RAM
    monkeypatch.setattr(offload, "ROOM_MARGIN", 2**50)  # no room for both
    session.decode(lat, True, 8, 8)
    assert seen[-1] == (True, False) and session.model.weight.is_cuda and not d.vae.weight.is_cuda
    session.decode(lat, True, 8, 8, keep_vae=True)  # input images: the next render samples with the VAE
    assert seen[-1] == (True, False) and session.model.weight.is_cuda and d.vae.weight.is_cuda
    assert session.vae_parked is None
    session.ensure_models(vae=True)  # without low VRAM (or with input images) it's on the GPU throughout
    assert d.vae.weight.is_cuda and session.vae_parked is None
    session.decode(lat, False, 8, 8)
    assert seen[-1] == (True, True) and d.vae.weight.is_cuda
    session.ensure_models(vae=False)
    assert not d.vae.weight.is_cuda and session.vae_parked


def test_an_enhanced_run_keeps_what_was_typed(profiles, monkeypatch):
    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    _, p, _ = check(profile_id=profiles["krea"], workflow="t2i", prompt="a long enhanced prompt", source_prompt="a cat")
    assert p["prompt"] == "a long enhanced prompt" and p["source_prompt"] == "a cat"
    _, p, _ = check(profile_id=profiles["krea"], workflow="t2i", source_prompt="a cat")
    assert "source_prompt" not in p  # not enhanced: the prompt is what was typed


def test_progress_carries_the_step_counter_and_brings_times_up_to_now(monkeypatch):
    from dainamo.server import sandbox as sbx

    m = sbx.SandboxManager()
    clock = [1000.0]
    monkeypatch.setattr(sbx.time, "time", lambda: clock[0])
    m._on_event({"event": "progress", "id": "r", "done": 25, "total": 40, "message": "Sampling · seed 43",
                 "image": 2, "images": 2, "step": 5, "steps": 20, "sec_per_step": 2.0, "eta": 31.0, "started": 960.0})
    clock[0] = 1004.0  # polled 4 s after the event
    p = m.snapshot()["progress"]
    assert p == {"done": 25, "total": 40, "image": 2, "images": 2, "step": 5, "steps": 20, "sec_per_step": 2.0,
                 "elapsed": 44.0, "eta": 27.0}
    assert m.message == "Sampling · seed 43"
    clock[0] = 1100.0  # running late: the time left doesn't go below zero
    assert m.snapshot()["progress"]["eta"] == 0.0
    m._on_event({"event": "progress", "id": "r", "done": 0, "total": 40, "eta": None, "started": 1100.0,
                 "step": 0, "steps": 20})
    assert m.snapshot()["progress"]["eta"] is None  # nothing timed yet


class Worker:
    """A stand-in worker process: what it's sent, and whether it has exited (waiting for it lets it)."""

    def __init__(self):
        self.stdin, self.code = io.StringIO(), None

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.code = 0
        return 0

    def kill(self):
        self.code = -9


def test_what_is_sent_while_the_sandbox_unloads_runs_in_a_new_worker(monkeypatch):
    import json

    m = sbx.SandboxManager()
    started = []

    def start(gpu):
        m.proc, m.gpu, m.state = Worker(), gpu, "starting"
        started.append(gpu)

    def unloading():
        old = Worker()
        old.stdin.close()  # as _stop leaves it
        m.proc, m.gpu, m.state, m.pending = old, 0, "stopping", []
        return old

    monkeypatch.setattr(m, "_start", start)
    old = unloading()
    m.submit({"id": "r1", "params": {}}, 0, {"id": "p", "family": "krea2", "data": {}})
    assert old.code == 0 and started == [0]  # it waited for the old one to exit, then started one on the same GPU
    assert m.pending == ["r1"] and json.loads(m.proc.stdin.getvalue())["id"] == "r1"
    old = unloading()
    assert m.submit_refmod({"id": "m1", "name": "x"}, 0)  # a RefMod too
    assert old.code == 0 and started == [0, 0] and json.loads(m.proc.stdin.getvalue())["cmd"] == "refmod"
    assert not m.submit_refmod({"id": "m2", "name": "y"}, 0) and m.pending == ["m1"]  # one at a time
    m.state = "ready"  # one that's running takes the next as it is
    m.submit({"id": "r2", "params": {}}, 0, {"id": "p", "family": "krea2", "data": {}})
    assert started == [0, 0] and m.pending == ["m1", "r2"]


def test_checkpoint_grids_are_checked_and_labelled(profiles, monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    jid = db.new_id()
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
               (jid, "fox", "completed", "krea2", "99", 0.0, db.now(), "{}"))
    out = os.path.join(qmod.job_dir(jid), "output")
    os.makedirs(out)
    files = ["fox_000250.safetensors", "fox_000500.safetensors", "fox.safetensors"]
    for f in files:
        open(os.path.join(out, f), "wb").close()
    with open(os.path.join(qmod.job_dir(jid), "status.json"), "w") as fh:
        json.dump({"checkpoints": [{"step": 250, "file": files[0], "final": False},
                                   {"step": 500, "file": files[1], "final": False},
                                   {"step": 750, "file": files[2], "final": True}]}, fh)
    paths = [os.path.join(out, f) for f in files]
    other = tmp_path / "style.safetensors"
    other.write_bytes(b"")

    def grid(**g):
        return check(profile_id=profiles["krea"], workflow="t2i", loras=[{"path": str(other), "strength": 0.5}],
                     compare_base=True, grid={"job_id": jid, "checkpoints": paths, "strengths": "0.8, 1", **g})

    _, p, _ = grid(base=True)
    assert p["grid"] == {"job_id": jid, "job_name": "fox", "base": True, "strengths": [0.8, 1.0],
                         "checkpoints": [{"path": paths[0], "label": "step 250", "step": 250},
                                         {"path": paths[1], "label": "step 500", "step": 500},
                                         {"path": paths[2], "label": "final", "step": 750}]}
    assert not p["compare_base"] and p["loras"] == [{"path": str(other), "strength": 0.5}]  # the grid has its own base
    for bad, words in (({"checkpoints": []}, "1–12"), ({"checkpoints": [paths[0], "/nope.safetensors"]}, "not found"),
                       ({"strengths": "a, b"}, "numbers"), ({"strengths": "1 2 3 4 5 6 7"}, "1–6"),
                       ({"checkpoints": paths * 4, "strengths": "0.5 0.6 0.7 0.8 0.9 1.0"}, "keep a grid")):
        with pytest.raises(HTTPException) as e:
            grid(**bad)
        assert words in e.value.detail
    _, p, _ = check(profile_id=profiles["krea"], workflow="t2i", grid=None)
    assert p["grid"] is None
    db.execute("DELETE FROM jobs WHERE id=?", (jid,))


def _refmod_file(path, kind="image"):
    import json

    from safetensors.torch import save_file

    if kind == "audio":
        save_file({"latent": torch.zeros(1, 32, 2, 50)}, str(path),  # as the older H3AudioMod saved voices
                  metadata={"audio_refmod_meta": json.dumps({"name": "voice", "kind": "audio"})})
    else:
        save_file({"latent": torch.zeros(1, 24, 1, 4, 4)}, str(path),
                  metadata={"refmod_meta": json.dumps({"name": "fox", "kind": "image", "latent_t": 1})})
    return str(path)


def test_reference_to_video_takes_refmods(profiles, monkeypatch, tmp_path):
    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    fox = _refmod_file(tmp_path / "fox.safetensors")
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    r2v = {"profile_id": profiles["ref2va"], "workflow": "r2v"}
    # a RefMod stands in for the reference images, or goes with them
    _, p, _ = check(**r2v, refmods=[{"path": fox, "strength": 0.6}])
    assert p["refmods"] == [{"path": fox, "strength": 0.6}] and p["images"] == []
    _, p, _ = check(**r2v, images=[str(img)], refmods=[{"path": fox}, {"path": "  "}])
    assert p["refmods"] == [{"path": fox, "strength": 1.0}]
    # at strength 0 it's left out, so the images are needed again
    with pytest.raises(HTTPException, match="needs 1–4"):
        check(**r2v, refmods=[{"path": fox, "strength": 0}])
    lora = tmp_path / "lora.safetensors"
    from safetensors.torch import save_file
    save_file({"w": torch.zeros(1)}, str(lora))
    for bad, words in (([{"path": str(lora)}], "isn't a RefMod"), ([{"path": "/nope.safetensors"}], "not found"),
                       ([{"path": fox, "strength": 1.5}], "between 0 and 1"), ([{"path": fox}] * 5, "Up to 4")):
        with pytest.raises(HTTPException, match=words):
            check(**r2v, refmods=bad)
    assert check(profile_id=profiles["ref2va"], workflow="t2v")[1]["refmods"] == []
    with pytest.raises(HTTPException, match="takes no RefMods"):
        check(profile_id=profiles["ref2va"], workflow="t2v", refmods=[{"path": fox}])


def test_reference_to_video_takes_reference_audio(profiles, monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    fox = _refmod_file(tmp_path / "fox.safetensors")
    voice = _refmod_file(tmp_path / "voice.safetensors", "audio")
    hum = tmp_path / "hum.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.5", str(hum)], check=True)
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    r2v = {"profile_id": profiles["ref2va"], "workflow": "r2v", "frames": 39}
    # sound files, and audio RefMods, beside a picture reference (an image or a visual RefMod)
    _, p, _ = check(**r2v, refmods=[{"path": fox}, {"path": voice}], audio=[str(hum), " "])
    assert p["audio"] == [str(hum)] and len(p["refmods"]) == 2 and p["images"] == []
    assert check(**r2v, images=[str(img)], audio=[str(hum)])[1]["audio"] == [str(hum)]
    assert check(**r2v, images=[str(img)])[1]["audio"] == []
    for kw, words in (({"refmods": [{"path": voice}], "audio": [str(hum)]}, "needs 1–4"),
                      ({"images": [str(img)], "audio": [str(hum)], "frames": 1}, "need a video"),
                      ({"images": [str(img)], "refmods": [{"path": voice}], "frames": 1}, "need a video"),
                      ({"images": [str(img)], "refmods": [{"path": voice}], "audio": [str(hum)] * 3}, "Up to 3"),
                      ({"images": [str(img)], "audio": ["/nope.wav"]}, "Audio not found"),
                      ({"images": [str(img)], "audio": [str(img)]}, "has no sound")):
        with pytest.raises(HTTPException, match=words):
            check(**{**r2v, **kw})
    with pytest.raises(HTTPException, match="takes no reference audio"):
        check(profile_id=profiles["ref2va"], workflow="t2v", frames=39, audio=[str(hum)])


def test_refmod_jobs_are_checked_listed_and_made_again(client, profiles, tmp_path, clean_jobs):
    import json

    from dainamo.server.routes import jobs as jobs_routes

    did = client.post("/api/datasets", json={"name": "refmod refs", "path": str(tmp_path)}).json()["id"]
    body = {"name": "fox ref", "profile_id": profiles["ref2va"], "datasets": [{"id": did}], "gpu": "auto",
            "settings": {"refmod_enabled": True, "slider_enabled": True, "picker_enabled": True}}
    assert "doesn't make RefMods" in client.post("/api/jobs", json={**body, "profile_id": profiles["krea"]}).json()["detail"]
    for bad in ({"refmod_grid": 15}, {"refmod_mode": "pooled"}, {"refmod_size": 100}, {"refmod_concept": "voice"},
                {"refmod_copy_to": "relative/folder"}, {"refmod_copy_to": "/nope/not/here"}, {"refmod_kind": "sound"},
                {"refmod_kind": "audio", "refmod_concept": "identity"}, {"refmod_seconds": 0}, {"refmod_seconds": 61},
                {"refmod_seconds": "long"}):
        r = client.post("/api/jobs", json={**body, "settings": {**body["settings"], **bad}})
        assert r.status_code == 400, bad
    job = client.post("/api/jobs", json=body).json()
    s = job["config"]["settings"]
    assert s["refmod_mode"] == "compressed" and s["refmod_grid"] == 16 and s["refmod_max_tokens"] == 5120
    assert s["refmod_kind"] == "visual" and s["refmod_seconds"] == 10
    assert not (s["slider_enabled"] or s["picker_enabled"] or s["bleed_enabled"])  # nothing is trained
    jid = job["id"]
    assert next(j for j in client.get("/api/jobs").json() if j["id"] == jid)["refmod"] is True

    # once made: its file is a RefMod for the Sandbox, not a LoRA, and it has nothing to continue or evaluate
    d = qmod.job_dir(jid)
    os.makedirs(os.path.join(d, "output"), exist_ok=True)
    out = _refmod_file(os.path.join(d, "output", "fox_ref.safetensors"))
    with open(os.path.join(d, "status.json"), "w") as fh:
        json.dump({"phase": "completed", "checkpoints": [{"step": 0, "file": "fox_ref.safetensors", "final": True}],
                   "refmod": {"file": "fox_ref.safetensors", "kind": "image", "mode": "compressed", "references": 1,
                              "frames": 1, "tokens": 4, "preview": [{"file": "refmod_00.png", "frame": 0}]}}, fh)
    db.execute("UPDATE jobs SET status='completed' WHERE id=?", (jid,))
    listed = client.get("/api/sandbox/refmods", params={"family": "minimax_h3"}).json()
    assert {"job_id": jid, "path": out, "tokens": 4, "mode": "compressed"}.items() <= listed[0].items()
    assert all(o["job_id"] != jid for o in client.get("/api/sandbox/loras", params={"family": "minimax_h3"}).json())
    assert next(j for j in client.get("/api/jobs").json() if j["id"] == jid)["last_sample"]["file"] == "refmod_00.png"
    assert client.post(f"/api/jobs/{jid}/action", json={"action": "extend", "steps": 10}).status_code == 400
    assert "no checkpoints" in client.post(f"/api/jobs/{jid}/evaluate", json={"settings": {}}).json()["detail"]

    # an edit without changes keeps it; one with changes makes it again
    edit = {**body, "settings": job["config"]["settings"]}
    assert client.put(f"/api/jobs/{jid}", json=edit).json()["status"] == "completed"
    r = client.put(f"/api/jobs/{jid}", json={**edit, "settings": {**edit["settings"], "refmod_grid": 8}}).json()
    assert r["status"] == "queued" and not os.path.exists(out) and r["config"]["settings"]["refmod_grid"] == 8
    assert jobs_routes._restart_reasons(r["config"], {**r["config"], "settings": {**r["config"]["settings"],
                                                                                   "refmod_enabled": False}})
    db.execute("DELETE FROM jobs WHERE id=?", (jid,))

    # a sound RefMod: listed with its length, its spectrogram the thumbnail
    job = client.post("/api/jobs", json={**body, "settings": {**body["settings"], "refmod_kind": "audio",
                                                              "refmod_concept": "voice", "refmod_seconds": 12.5}}).json()
    assert job["config"]["settings"]["refmod_seconds"] == 12.5
    jid = job["id"]
    d = qmod.job_dir(jid)
    os.makedirs(os.path.join(d, "output"), exist_ok=True)
    out = _refmod_file(os.path.join(d, "output", "her_voice.safetensors"), "audio")
    with open(os.path.join(d, "status.json"), "w") as fh:
        json.dump({"phase": "completed", "refmod": {"file": "her_voice.safetensors", "kind": "audio", "mode": "full",
                                                    "references": 2, "frames": 0, "seconds": 1.25, "tokens": 100,
                                                    "preview": [], "audio": {"file": "refmod_audio.mp3",
                                                                             "poster": "refmod_audio_poster.png"}}}, fh)
    listed = next(o for o in client.get("/api/sandbox/refmods", params={"family": "minimax_h3"}).json() if o["job_id"] == jid)
    assert listed["kind"] == "audio" and listed["seconds"] == 1.25 and listed["path"] == out
    assert next(j for j in client.get("/api/jobs").json() if j["id"] == jid)["last_sample"]["file"] == "refmod_audio_poster.png"
    db.execute("DELETE FROM jobs WHERE id=?", (jid,))


def test_the_sandbox_makes_refmods(client, profiles, monkeypatch, tmp_path):
    """Making a RefMod in the Sandbox: checked and estimated (a warning when a clip may not fit beside the model),
    a sheet previewed, queued on the Sandbox's GPU, tracked apart from the runs, and listed with the jobs' ones."""
    import json
    import shutil
    import subprocess

    from PIL import Image

    from dainamo.server import hardware
    from dainamo.server.routes import refmods as rm

    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    for k, v in (("proc", FakeProc()), ("gpu", 1), ("refmod", None), ("pending", []), ("current", None),
                 ("state", "ready"), ("message", ""), ("progress", None), ("last_used", 0.0)):
        monkeypatch.setattr(sbx.sandbox, k, v)
    monkeypatch.setattr(hardware, "gpus", lambda: [{"index": 1, "name": "RTX", "mem_used": 22000, "mem_total": 24576,
                                                    "util": 0, "temp": 40}])
    front, side, clip = tmp_path / "front.png", tmp_path / "side.png", tmp_path / "walk.mp4"
    Image.new("RGB", (600, 800), (200, 0, 0)).save(front)
    Image.new("RGB", (600, 800), (0, 200, 0)).save(side)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x384:rate=24:duration=2",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    body = {"profile_id": profiles["ref2va"], "paths": [str(front), str(side)],
            "settings": {"refmod_mode": "full", "refmod_size": 512, "refmod_frames": 39}}
    r = client.post("/api/sandbox/refmod/check", json=body).json()
    assert r["refs"][0] == {"kind": "image", "width": 512, "height": 672, "frames": 1}
    assert (r["tokens"], r["per_frame"], r["name"], r["sheet"], r["error"]) == (672, 336, "front_refmod", None, None)
    assert r["vram"]["gpu"] == 1 and r["vram"]["loaded"] and r["warning"] is None  # images: no warning
    r = client.post("/api/sandbox/refmod/check", json={**body, "paths": [str(front), str(clip)]}).json()
    assert r["refs"][1] == {"kind": "video", "width": 512, "height": 672, "frames": 39}
    assert "may run out of memory" in r["warning"] and "release the Sandbox" in r["warning"]  # 2.5 GB free
    r = client.post("/api/sandbox/refmod/check", json={**body, "sheet": True}).json()
    assert r["sheet"] == {"columns": 2, "rows": 1} and len(r["refs"]) == 1 and r["name"] == "front_sheet"
    capped = {**body, "settings": {**body["settings"], "refmod_max_tokens": 100}}
    assert "more than the cap" in client.post("/api/sandbox/refmod/check", json=capped).json()["error"]
    for bad in ({"sheet": True, "paths": [str(front), str(clip)]}, {"paths": [str(tmp_path / "gone.png")]},
                {"paths": []}, {"profile_id": profiles["krea"]}, {"settings": {"refmod_grid": 15}}):
        assert client.post("/api/sandbox/refmod/check", json={**body, **bad}).status_code == 400, bad
    r = client.get("/api/sandbox/refmod/sheet", params={"path": [str(front), str(side)], "size": 512})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    sent = []
    monkeypatch.setattr(sbx.sandbox, "_enqueue", lambda gpu, msg: sent.append((msg, gpu)))
    made = client.post("/api/sandbox/refmod", json={**body, "name": "Jake: front/side.safetensors", "gpu": 0}).json()
    [(make, gpu)] = sent
    assert made["name"] == make["name"] == "Jake_front_side" and gpu == 1  # the Sandbox's GPU, where its VAE is
    assert make["cmd"] == "refmod" and make["out_dir"] == rm.made_dir(made["id"])
    assert make["settings"]["refmod_mode"] == "full"
    assert make["profile"]["family"] == "minimax_h3" and make["paths"] == [str(front), str(side)]
    assert sbx.sandbox.refmod["state"] == "queued"
    assert client.post("/api/sandbox/refmod", json=body).status_code == 409 and len(sent) == 1  # one at a time

    # the worker's events update it, not a run, and the Sandbox is ready again once it's done
    monkeypatch.setattr(sbx.sandbox, "refmod", {"id": "m1", "name": "x", "state": "queued", "message": "", "made": None})
    monkeypatch.setattr(sbx.sandbox, "pending", ["m1"])
    sbx.sandbox._on_event({"event": "started", "id": "m1"})
    sbx.sandbox._on_event({"event": "progress", "id": "m1", "message": "1/2 front.png: 512×672"})
    assert sbx.sandbox.snapshot()["refmod"]["state"] == "running" and sbx.sandbox.message == "1/2 front.png: 512×672"
    sbx.sandbox._on_event({"event": "done", "id": "m1", "made": {"name": "x"}})
    assert sbx.sandbox.refmod["state"] == "done" and sbx.sandbox.refmod["made"] == {"name": "x"}
    assert sbx.sandbox.current is None and sbx.sandbox.state == "ready"
    assert db.query_one("SELECT id FROM sandbox_runs WHERE id='m1'") is None

    # listed for its family, newest first, with the jobs' RefMods
    folder = rm.made_dir("abc123")
    os.makedirs(folder)
    try:
        _refmod_file(os.path.join(folder, "jake_sheet.safetensors"))
        Image.new("RGB", (8, 8)).save(os.path.join(folder, "refmod_00.png"))
        with open(os.path.join(folder, "made.json"), "w") as f:
            json.dump({"name": "jake_sheet", "file": "jake_sheet.safetensors", "family": "minimax_h3", "kind": "image",
                       "mode": "full", "tokens": 672, "references": 2, "frames": 1, "sheet": True,
                       "created": db.now()}, f)
        [first, *_] = client.get("/api/sandbox/refmods", params={"family": "minimax_h3"}).json()
        assert {"job_id": None, "job_name": "jake_sheet", "sheet": True, "tokens": 672,
                "path": os.path.join(folder, "jake_sheet.safetensors")}.items() <= first.items()
        assert client.get("/api/sandbox/refmods", params={"family": "krea2"}).json() == []
        assert client.get("/api/sandbox/refmod/file/abc123/refmod_00.png").status_code == 200
        assert client.get("/api/sandbox/refmod/file/abc123/made.json").status_code == 404
    finally:
        shutil.rmtree(folder)

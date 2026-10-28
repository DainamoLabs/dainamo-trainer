"""Sandbox: request validation per family/workflow, the GPU rules shared with the training queue, and
the inference LoRA stack.

    .venv/bin/python -m pytest tests/test_sandbox.py -v
"""

import io
import os
import sys
import tempfile
import time

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
    def __init__(self):
        self.stdin = io.StringIO()  # its own: a stopped one's is closed

    def poll(self):
        return None

    def wait(self, timeout=None):  # the manager's tick may stop an idle fake worker
        return 0


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
    # its layers' names, another size's widths (a FLUX.2 Klein 4B LoRA on the 9B): refused, not a matmul error later
    wide = {k: torch.zeros(2, 12) if k.endswith("lora_A.weight") else v for k, v in lora_sd().items()}
    with pytest.raises(ValueError, match=r"another size of this model: its blocks.0.0 is 12 -> 8, this model's 8 -> 8"):
        stack.set([("wide", wide, 1.0)])
    assert torch.allclose(model(x), _clone(model)(x))  # nothing left applied


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


def test_a_failed_encode_still_parks_the_text_encoder(monkeypatch):
    """An encode that raises (out of memory, a bad reference image) moves the text encoder off the GPU as one that
    worked does: left marked "gpu", later requests skipped parking and ran it beside the diffusion model."""
    from dainamo.engine import sandbox

    monkeypatch.setattr(sandbox, "emit", lambda *a, **kw: None)
    session = sandbox.Session()

    class Driver:
        def encode_condition(self, prompt, images):
            raise RuntimeError("CUDA out of memory")

    session.driver, session.te_where = Driver(), "gpu"
    parked = []

    def park():
        parked.append(True)
        session.te_where = "cpu"

    monkeypatch.setattr(session, "_park_text_encoder", park)
    with pytest.raises(RuntimeError, match="out of memory"):
        session.encode({"p": ("p", None)})
    assert parked == [True] and session.te_where == "cpu" and "p" not in session.texts


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


def test_generate_right_after_release_runs_in_a_new_worker(monkeypatch):
    """Release cancels what's queued and closes the worker's stdin; the "cancelled" events it answers with don't
    make it "ready" again (Generate then wrote to the closed stdin, and the run failed when it exited)."""
    import json

    m = sbx.SandboxManager()
    started = []

    def start(gpu):
        m.proc, m.gpu, m.state = Worker(), gpu, "starting"
        started.append(gpu)

    monkeypatch.setattr(m, "_start", start)
    monkeypatch.setattr(db, "execute", lambda *a, **kw: None)  # (the runs' rows)
    old = Worker()
    m.proc, m.gpu, m.state, m.pending = old, 0, "ready", ["r0"]
    m.release()
    assert old.stdin.closed and m.state == "stopping"
    m._on_event({"event": "cancelled", "id": "r0"})
    assert m.state == "stopping" and not m.pending
    m.submit({"id": "r1", "params": {}}, 0, {"id": "p", "family": "krea2", "data": {}})
    assert old.code == 0 and started == [0]  # waited for the released one to exit, then started a new one
    assert json.loads(m.proc.stdin.getvalue())["id"] == "r1"


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
                 ("state", "ready"), ("message", ""), ("progress", None), ("last_used", time.time())):  # (not idle)
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


# --- upscaling in the worker ---------------------------------------------------------------------------------------
class FakeUpscaler:
    """Stands in for an ``engine.upscale.Upscaler``: a nearest-neighbour resize to the size asked, noting what it's
    given, and telling ``progress`` of each frame as the pixel upscaler does."""
    unloaded: list = []

    def __init__(self, path, kind="pixel", scale=2, video=False):
        from dainamo.engine.upscale import Info

        self.info, self.scale, self.calls = Info(path, kind, os.path.basename(path), scale=scale, video=video), scale, []
        self.after = None  # called after each upscale (to cancel from a test)

    def upscale(self, frames, size, progress=None, check=None, **kw):
        self.calls.append((tuple(frames.shape), tuple(size), kw))
        assert frames.dtype == torch.float32 and frames.shape[1] == 3 and 0 <= frames.min() and frames.max() <= 1
        for i in range(frames.shape[0]):
            if check:
                check()
            if progress and frames.shape[0] > 1:
                progress(f"Frame {i + 1} of {frames.shape[0]}")
        out = nn.functional.interpolate(frames, size=(size[1], size[0]), mode="nearest")
        if self.after:
            self.after()
        return out

    def unload(self):
        FakeUpscaler.unloaded.append(self.info.path)


@pytest.fixture
def upscale_worker(monkeypatch):
    """A worker Session whose upscalers are FakeUpscalers (loads counted), and the events it sends."""
    from dainamo.engine import sandbox as worker
    from dainamo.engine import upscale

    events = io.StringIO()
    monkeypatch.setattr(worker, "EVENTS", events)
    loads = []

    def load(info, device, log=print):
        loads.append(info.path)
        return FakeUpscaler(info.path, video=info.kind == "seedvr2")

    monkeypatch.setattr(upscale, "identify", lambda path, files=None: upscale.Info(
        path, "seedvr2" if "seedvr2" in path else "pixel", os.path.basename(path), scale=2))
    monkeypatch.setattr(upscale, "load", load)
    session = worker.Session()

    def sent():
        import json
        return [json.loads(line) for line in events.getvalue().splitlines()]
    return session, loads, sent


def _clip(path, size="32x24", rate=12, frames=2, audio=True):
    import subprocess

    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-shortest"]
    subprocess.run(cmd + ["-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _probe(path):
    import json
    import subprocess

    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"]


def test_the_worker_upscales_an_image_and_a_clip(upscale_worker, tmp_path):
    """The "upscale" command: each input upscaled to the factor asked (a clip to even sides, with its frame rate
    and sound), one output each with its source and size, progress counted in frames, and the upscaler kept loaded
    for the next run; another model replaces it."""
    from PIL import Image

    session, loads, sent = upscale_worker
    img = Image.new("RGBA", (24, 15), (200, 30, 30, 255))
    img.putpixel((0, 0), (0, 0, 0, 0))  # transparency, which carries through
    img.save(tmp_path / "pic.png")
    clip = _clip(tmp_path / "walk.mp4")
    out_dir = tmp_path / "run"
    msg = {"id": "u1", "out_dir": str(out_dir), "model": "/m/4x.pth", "factor": 2,
           "inputs": [{"path": str(tmp_path / "pic.png"), "kind": "image"}, {"path": clip, "kind": "video"}]}
    session.upscale(msg, lambda: False)
    events = sent()
    assert events[0] == {"event": "started", "id": "u1"}
    assert events[-1]["event"] == "done" and events[-1]["elapsed"] >= 0
    outputs = [e["item"] for e in events if e["event"] == "output"]
    assert outputs[0] == {"file": "0_pic_x2.png", "kind": "image", "source": str(tmp_path / "pic.png"), "width": 48,
                          "height": 30}
    assert outputs[1] == {"file": "1_walk_x2.mp4", "poster": "1_walk_x2_poster.png", "kind": "video", "source": clip,
                          "width": 64, "height": 48}
    out = Image.open(out_dir / "0_pic_x2.png")
    assert out.mode == "RGBA" and out.size == (48, 30)
    assert out.getpixel((0, 0))[3] < 128 and out.getpixel((40, 20))[3] == 255
    video, audio = _probe(out_dir / "1_walk_x2.mp4")
    assert (video["width"], video["height"], video["nb_read_frames"], video["avg_frame_rate"]) == (64, 48, "2", "12/1")
    assert audio["codec_type"] == "audio" and (out_dir / "1_walk_x2_poster.png").exists()
    progress = [e for e in events if e["event"] == "progress"]
    assert all(e["total"] == 3 and e["inputs"] == 2 and e["started"] for e in progress)  # an image and two frames
    assert [e["done"] for e in progress if e["message"].startswith("walk.mp4: frame")] == [2, 3]
    assert progress[0]["message"] == "pic.png: 24×15 to 48×30" and progress[-1]["eta"] is not None
    up = session.upscaler
    assert [c[:2] for c in up.calls] == [((1, 3, 15, 24), (48, 30))] + [((1, 3, 24, 32), (64, 48))] * 2
    assert loads == ["/m/4x.pth"] and session.upscaler_path == "/m/4x.pth"

    # the same model again: still loaded; options reach it
    session.upscale({**msg, "id": "u2", "factor": 1.5, "inputs": msg["inputs"][:1], "options": {"tiles": 2}},
                    lambda: False)
    assert loads == ["/m/4x.pth"] and up.calls[-1] == ((1, 3, 15, 24), (36, 22), {"options": {"tiles": 2}})
    # another one replaces it
    FakeUpscaler.unloaded.clear()
    session.upscale({**msg, "id": "u3", "model": "/m/seedvr2.safetensors", "inputs": msg["inputs"][1:]}, lambda: False)
    assert loads == ["/m/4x.pth", "/m/seedvr2.safetensors"] and FakeUpscaler.unloaded == ["/m/4x.pth"]
    assert session.upscaler.calls[0][0] == (2, 3, 24, 32)  # one that takes clips as clips gets the frames together
    session.unload_all()
    assert session.upscaler is None and FakeUpscaler.unloaded[-1] == "/m/seedvr2.safetensors"


def test_upscaling_cancels_between_inputs(upscale_worker, tmp_path):
    from PIL import Image

    from dainamo.engine.sampling import Cancelled

    session, _, sent = upscale_worker
    for name in ("a", "b"):
        Image.new("RGB", (8, 8), (10, 200, 10)).save(tmp_path / f"{name}.png")
    flag = []
    session.upscaler = FakeUpscaler("/m/4x.pth")
    session.upscaler_path = "/m/4x.pth"
    session.upscaler.after = lambda: flag.append(1)
    with pytest.raises(Cancelled):
        session.upscale({"id": "u", "out_dir": str(tmp_path / "run"), "model": "/m/4x.pth", "factor": 2,
                         "inputs": [{"path": str(tmp_path / f"{n}.png"), "kind": "image"} for n in "ab"]},
                        lambda: bool(flag))
    assert len(session.upscaler.calls) == 1 and [e["event"] for e in sent()].count("output") == 1
    assert session.upscaler is not None  # still loaded for the next run


class FakeClipUpscaler(FakeUpscaler):
    """One that takes clips as clips, in passes of 5 frames sharing 2 (as SeedVR2 says through ``frames_per_pass``
    and ``overlap_frames``); each call adds 0.2 times the calls before it, so where the worker's pieces meet
    shows."""

    def __init__(self, path):
        super().__init__(path, kind="seedvr2", scale=0, video=True)
        self.hints = []

    def frames_per_pass(self, size, options=None):
        self.hints.append(options)
        return 5

    def overlap_frames(self, size, options=None):
        return 2

    @torch.inference_mode()  # as SeedVR2's: the frames it gives can't be changed in place outside it
    def upscale(self, frames, size, progress=None, check=None, **kw):
        k = len(self.calls)
        return super().upscale(frames, size, progress, check, **kw) + 0.2 * k


def _grey_clip(tmp_path, frames=24):
    """A clip of flat greys, a different one each frame, at 30000/1001 with sound -> (path, its frames' greys as
    the worker reads them)."""
    import numpy as np

    from dainamo.engine import sandbox as worker
    from dainamo.engine.media import VideoWriter

    greys = (40 + 3 * np.arange(frames)).astype(np.uint8)
    with VideoWriter(str(tmp_path), "long", "30000/1001", audio=_clip(tmp_path / "sound.mp4", frames=30)) as video:
        video.write(np.ascontiguousarray(np.broadcast_to(greys[:, None, None, None], (frames, 24, 32, 3))))
    path = str(tmp_path / "long.mp4")
    read = next(worker._clip_frames(path, 32, 24, 1000))
    return path, read.reshape(frames, -1).mean(1)


def _upscale_clip_msg(tmp_path, path, model="/m/seedvr2.safetensors", options=None):
    return {"id": "c", "out_dir": str(tmp_path / "run"), "model": model, "factor": 2,
            "inputs": [{"path": path, "kind": "video"}], **({"options": options} if options else {})}


def test_a_long_clip_is_upscaled_in_pieces_that_crossfade(upscale_worker, tmp_path, monkeypatch):
    """A clip whose frames don't fit in CLIP_RAM goes in pieces of whole passes (m * (5 - 2) + 2 frames), each
    starting with the last 2 frames of the piece before; those are crossfaded, as passes are, and written once.
    The video streams out as it's made: every frame, the frame rate as it was, the sound. Progress counts each
    frame once. A clip that fits goes whole, the upscaler making its own passes."""
    import numpy as np

    from dainamo.engine import sandbox as worker

    session, _, sent = upscale_worker
    path, greys = _grey_clip(tmp_path)
    session.upscaler, session.upscaler_path = FakeClipUpscaler("/m/seedvr2.safetensors"), "/m/seedvr2.safetensors"
    per = 4 * 3 * (32 * 24 + 64 * 48)  # a frame in and out, as floats
    monkeypatch.setattr(worker, "CLIP_RAM", 12 * per)  # 12 frames: pieces of 11, three passes each
    session.upscale(_upscale_clip_msg(tmp_path, path, options={"max_frames": 5}), lambda: False)
    up = session.upscaler
    assert [c[0][0] for c in up.calls] == [11, 11, 6]  # frames 0-10, 9-19, 18-23
    assert up.hints == [{"max_frames": 5}]  # asked with the options, once
    events = sent()
    out = [e["item"] for e in events if e["event"] == "output"]
    assert out == [{"file": "0_long_x2.mp4", "poster": "0_long_x2_poster.png", "kind": "video", "source": path,
                    "width": 64, "height": 48}]
    video, audio = _probe(tmp_path / "run" / "0_long_x2.mp4")
    assert (video["width"], video["height"], video["nb_read_frames"], video["avg_frame_rate"]) == \
        (64, 48, "24", "30000/1001")
    assert audio["codec_type"] == "audio"
    made = next(worker._clip_frames(str(tmp_path / "run" / "0_long_x2.mp4"), 64, 48, 1000))
    shift = made.reshape(24, -1).mean(1) - greys  # what each piece added, crossfaded where they meet
    ramp = [0] * 9 + [0.05, 0.15] + [0.2] * 7 + [0.25, 0.35] + [0.4] * 4
    assert np.abs(shift - 255 * np.array(ramp)).max() < 2.5, shift.round(1)
    progress = [e for e in events if e["event"] == "progress" and ": frame " in e["message"]]
    assert [(e["done"], e["message"]) for e in progress] == [(11, "long.mp4: frame 11 of 24"),
                                                              (20, "long.mp4: frame 20 of 24"),
                                                              (24, "long.mp4: frame 24 of 24")]
    assert any(e["message"].startswith("long.mp4: part 2 of 3: Frame") for e in events if e["event"] == "progress")

    monkeypatch.setattr(worker, "CLIP_RAM", 8e9)  # it fits: one call, the whole clip
    up.calls.clear()
    session.upscale({**_upscale_clip_msg(tmp_path, path), "id": "c2"}, lambda: False)
    assert [c[0][0] for c in up.calls] == [24]
    monkeypatch.setattr(worker, "CLIP_RAM", 2 * per)  # not even the 3 frames a piece needs
    with pytest.raises(ValueError, match="system RAM"):
        session.upscale({**_upscale_clip_msg(tmp_path, path), "id": "c3"}, lambda: False)


def test_a_clip_that_fails_part_way_leaves_no_video(upscale_worker, tmp_path, monkeypatch):
    """An error after some pieces are written, or cancelling, removes the partial video and its poster."""
    from dainamo.engine import sandbox as worker
    from dainamo.engine.sampling import Cancelled

    session, _, _ = upscale_worker
    path, _ = _grey_clip(tmp_path)
    session.upscaler, session.upscaler_path = FakeClipUpscaler("/m/seedvr2.safetensors"), "/m/seedvr2.safetensors"
    monkeypatch.setattr(worker, "CLIP_RAM", 12 * 4 * 3 * (32 * 24 + 64 * 48))

    def fail():
        if len(session.upscaler.calls) == 2:
            raise RuntimeError("CUDA out of memory")

    session.upscaler.after = fail
    with pytest.raises(RuntimeError):
        session.upscale(_upscale_clip_msg(tmp_path, path), lambda: False)
    assert os.listdir(tmp_path / "run") == []
    session.upscaler.calls.clear()
    session.upscaler.after = None
    with pytest.raises(Cancelled):
        session.upscale(_upscale_clip_msg(tmp_path, path), lambda: len(session.upscaler.calls) >= 2)
    assert os.listdir(tmp_path / "run") == []
    pixel = FakeUpscaler("/m/4x.pth")  # a pixel upscaler, a frame at a time
    session.upscaler, session.upscaler_path = pixel, "/m/4x.pth"
    with pytest.raises(Cancelled):
        session.upscale(_upscale_clip_msg(tmp_path, path, model="/m/4x.pth"), lambda: len(pixel.calls) >= 5)
    assert os.listdir(tmp_path / "run") == [] and len(pixel.calls) == 5


def test_the_worker_upscales_a_datasets_small_images_in_place(upscale_worker, tmp_path):
    """"upscale_dataset": small images replaced by <stem>.png at the size a job trains at (by the max factor at
    most), the originals in _originals beside them, captions untouched; large images, clips and clashing or broken
    files skipped or reported, with counts. Cancelling stops between images."""
    from PIL import Image

    from dainamo.engine.dataset import ORIGINALS_DIR, list_media
    from dainamo.engine.sampling import Cancelled

    session, loads, sent = upscale_worker
    ds = tmp_path / "ds"
    (ds / "sub").mkdir(parents=True)
    Image.new("RGB", (64, 48), (90, 20, 20)).save(ds / "small.jpg")
    (ds / "small.txt").write_text("a red square")
    Image.new("L", (32, 32), 128).save(ds / "sub" / "grey.png")  # already a PNG; in a subfolder
    Image.new("RGB", (512, 512)).save(ds / "big.png")
    Image.new("RGB", (40, 40)).save(ds / "clash.jpg")
    Image.new("RGB", (600, 600)).save(ds / "clash.png")  # another image where clash.jpg's result would go
    (ds / "broken.jpg").write_bytes(b"not an image")
    clip = _clip(ds / "walk.mp4", audio=False)
    paths = [str(ds / n) for n in ("small.jpg", "sub/grey.png", "big.png", "clash.jpg", "broken.jpg", "walk.mp4")]
    session.upscale_dataset({"id": "d1", "model": "/m/4x.pth", "paths": paths, "resolution": 128, "max_factor": 4},
                            lambda: False)
    events = sent()
    made = events[-1]["made"]
    assert events[-1]["event"] == "done" and (made["upscaled"], made["skipped"]) == (2, 2)  # big.png and the clip
    assert len(made["errors"]) == 2 and made["errors"][0].startswith("clash.jpg: clash.png is there already")
    assert made["errors"][1].startswith("broken.jpg: ")
    # 64x48 to an area of 128²: 2.31x
    assert Image.open(ds / "small.png").size == (148, 111) and not (ds / "small.jpg").exists()
    assert Image.open(ds / ORIGINALS_DIR / "small.jpg").size == (64, 48)
    assert (ds / "small.txt").read_text() == "a red square"
    grey = Image.open(ds / "sub" / "grey.png")
    assert grey.size == (128, 128) and grey.mode == "L"  # 4x, the most it may
    assert Image.open(ds / "sub" / ORIGINALS_DIR / "grey.png").size == (32, 32)
    assert (ds / "clash.jpg").exists() and Image.open(ds / "clash.png").size == (600, 600)  # left as they were
    assert made["files"] == [
        {"path": str(ds / "small.png"), "original": str(ds / ORIGINALS_DIR / "small.jpg"), "width": 148, "height": 111},
        {"path": str(ds / "sub" / "grey.png"), "original": str(ds / "sub" / ORIGINALS_DIR / "grey.png"), "width": 128,
         "height": 128}]
    assert sorted(os.path.relpath(p, ds) for p in list_media(str(ds))) == [
        "big.png", "broken.jpg", "clash.jpg", "clash.png", "small.png", "sub/grey.png", "walk.mp4"]  # no originals
    assert not [f for f in os.listdir(ds) if f.startswith(".")]  # no temporary files left
    progress = [e for e in events if e["event"] == "progress"]
    assert progress[0] == {"event": "progress", "id": "d1", "done": 0, "total": 6,
                           "message": "small.jpg: 64×48 to 148×111", "made": {"upscaled": 0, "skipped": 0, "errors": []}}
    assert progress[-1]["done"] == 6 and loads == ["/m/4x.pth"] and clip in paths

    # again, the upscaled PNG now small for a bigger resolution: the first original is kept, the PNG kept beside it,
    # and a third time, beside that under another name
    Image.new("RGB", (16, 16)).save(ds / "tiny.png")
    flag = []
    session.upscaler.after = lambda: flag.append(1)
    with pytest.raises(Cancelled):  # cancelled after the first image
        session.upscale_dataset({"id": "d2", "model": "/m/4x.pth", "resolution": 256, "max_factor": 4,
                                 "paths": [str(ds / "small.png"), str(ds / "tiny.png")]}, lambda: bool(flag))
    assert Image.open(ds / "small.png").size == (296, 222)
    assert Image.open(ds / ORIGINALS_DIR / "small.jpg").size == (64, 48)
    assert Image.open(ds / ORIGINALS_DIR / "small.png").size == (148, 111)
    assert Image.open(ds / "tiny.png").size == (16, 16)  # not reached
    flag.append(0)
    session.upscaler.after = None
    session.upscale_dataset({"id": "d3", "model": "/m/4x.pth", "paths": [str(ds / "small.png")], "resolution": 512,
                             "max_factor": 4}, lambda: False)
    assert Image.open(ds / ORIGINALS_DIR / "small_2.png").size == (296, 222)
    assert Image.open(ds / ORIGINALS_DIR / "small.png").size == (148, 111)


def test_upscalers_make_room_and_say_what_to_try_out_of_memory(upscale_worker):
    from dainamo.engine import sandbox as worker

    session, _, _ = upscale_worker
    session.upscaler, session.upscaler_path = FakeUpscaler("/m/4x.pth"), "/m/4x.pth"
    session.upscaler_bytes = 40e6
    FakeUpscaler.unloaded.clear()
    session.drop_upscaler(big_only=True)  # a pixel upscaler stays beside a generation's models
    assert session.upscaler is not None and not FakeUpscaler.unloaded
    session.upscaler_bytes = 8e9  # SeedVR2's DiT doesn't
    session.drop_upscaler(big_only=True)
    assert session.upscaler is None and FakeUpscaler.unloaded == ["/m/4x.pth"]
    up = {"model": "/m/4x.pth"}
    assert worker._out_of_memory("upscale", up, session).startswith("Out of GPU memory: try a smaller factor. ")
    assert "max factor" in worker._out_of_memory("upscale_dataset", up, session)
    assert "SeedVR2 tile" in worker._out_of_memory("upscale", {"model": "/m/seedvr2_3b.safetensors"}, session)
    assert "fewer LoRAs" in worker._out_of_memory("generate", {}, session)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_an_upscaler_without_room_beside_the_models_has_them_parked(upscale_worker, tmp_path, monkeypatch):
    from PIL import Image

    from dainamo.engine import sandbox as worker

    session, _, _ = upscale_worker
    session.model = nn.Linear(8, 8).to(session.device)  # the profile's diffusion model

    class Driver:
        swapper, vae = None, None

    session.driver = Driver()
    session.upscaler, session.upscaler_path = FakeUpscaler("/m/4x.pth"), "/m/4x.pth"
    seen = []
    session.upscaler.after = lambda: seen.append(session.model.weight.is_cuda)
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    msg = {"id": "u", "out_dir": str(tmp_path / "run"), "model": "/m/4x.pth", "factor": 2,
           "inputs": [{"path": str(tmp_path / "a.png"), "kind": "image"}]}
    monkeypatch.setitem(worker.UPSCALE_WORK, "pixel", 2**50)  # no room beside it
    session.upscale(msg, lambda: False)
    assert seen == [False] and session.model.weight.is_cuda  # in system RAM while it ran, back after
    monkeypatch.setitem(worker.UPSCALE_WORK, "pixel", 0)
    session.upscale(msg, lambda: False)
    assert seen == [False, True]


# --- image-to-image and the hi-res fix in the worker -------------------------------------------------------------
def _gen_driver():
    from PIL import Image

    from dainamo.engine.families.base import FamilyDriver, FamilyInfo

    class GenDriver(FamilyDriver):
        """Stands in for an image family in the worker's ``generate``: an image's latents are its 8 px averages, and
        sampling gives back the input image's latents (or grey from noise), noting what it was asked."""
        info = FamilyInfo(id="tiny", label="Tiny", media=["image"], profile_fields=[])

        def __init__(self):
            super().__init__({"diffusion_model": "dm.safetensors"}, torch.device("cpu"), log=lambda m: None)
            self.calls, self.encoded, self.spec = [], [], None

        def sample_condition(self, spec):
            self.spec = spec
            return spec["prompt"], None

        def load_vae(self):
            pass

        def encode_image(self, pixels):
            self.encoded.append(tuple(pixels.shape))
            return nn.functional.avg_pool2d(pixels, 8)[0]

        def sample(self, model, cond, uncond, width, height, steps, cfg, seed, frames=1, images=None,
                   sampler="euler", scheduler="default", shift=None, callback=None, init=None, strength=None):
            self.calls.append({"size": (width, height), "steps": steps, "strength": strength, "images": images,
                               "init": None if init is None else tuple(init.shape)})
            for k in range(steps):
                callback(k + 1, steps)
            return init[None] if init is not None else torch.zeros(1, 3, height // 8, width // 8)

        def to_media(self, latents):
            arr = ((latents[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
            img = Image.fromarray(arr)
            return [img.resize((img.width * 8, img.height * 8), Image.Resampling.NEAREST)]

    return GenDriver()


@pytest.fixture
def gen_worker(upscale_worker, monkeypatch):
    """The upscale worker's Session with a stand-in image family loaded (prompts encode to nothing, LoRAs apply to
    nothing), the profile it's for, and the upscalers it made room for."""
    import json

    session, loads, sent = upscale_worker
    d = _gen_driver()
    profile = {"id": "p", "family": "tiny", "data": dict(d.profile)}
    session.driver, session.profile_key = d, ("p", json.dumps(profile["data"], sort_keys=True))
    session.model, session.vae_loaded = nn.Linear(2, 2), True
    enabled = []

    class Adapters:
        def set(self, files):
            return []

        def set_enabled(self, on):
            enabled.append(on)

    session.adapters = Adapters()
    monkeypatch.setattr(session, "encode", lambda conds: session.texts.update({k: torch.zeros(1) for k in conds}))
    monkeypatch.setattr(session, "apply_loras", lambda loras: [])
    rooms = []
    monkeypatch.setattr(session, "_upscaler_room", lambda path: rooms.append(path) or [])
    return session, d, profile, loads, sent, rooms, enabled


def test_the_hires_fix_renders_again_bigger_from_the_upscaled_first_pass(gen_worker, tmp_path):
    """Hi-res fix: the render, decoded, upscaled to the size asked with the Session's cached upscaler (the profile's
    models given room), encoded, and sampled again from there at the second pass's strength and steps; the first
    pass is an output (variant "pass1") before the final, each with its size."""
    from PIL import Image

    session, d, profile, loads, sent, rooms, _ = gen_worker
    out = tmp_path / "g"
    hires = {"factor": 2, "model": "/m/4x.pth", "model_name": "4x", "strength": 0.35, "steps": 3, "width": 128,
             "height": 96}
    msg = {"id": "g", "out_dir": str(out), "profile": profile,
           "params": {"workflow": "t2i", "prompt": "a cat", "width": 64, "height": 48, "steps": 4, "seed": 5,
                      "count": 1, "cfg": 1.0, "hires": hires}}
    session.generate(msg, lambda: False)
    events = sent()
    items = [e["item"] for e in events if e["event"] == "output"]
    assert items == [
        {"file": "0_lora_pass1.png", "kind": "image", "seed": 5, "variant": "pass1", "of": "lora", "width": 64,
         "height": 48},
        {"file": "0_lora.png", "kind": "image", "seed": 5, "variant": "lora", "width": 128, "height": 96}]
    assert Image.open(out / "0_lora_pass1.png").size == (64, 48) and Image.open(out / "0_lora.png").size == (128, 96)
    assert [(c["size"], c["steps"], c["init"], c["strength"]) for c in d.calls] == [
        ((64, 48), 4, None, None), ((128, 96), 3, (3, 12, 16), 0.35)]
    assert d.encoded == [(1, 3, 96, 128)]  # the upscaled first pass
    assert loads == ["/m/4x.pth"] and rooms == ["/m/4x.pth"] and session.upscaler.calls[0][:2] == ((1, 3, 48, 64),
                                                                                                    (128, 96))
    assert session.upscaler_path == "/m/4x.pth"  # a pixel upscaler stays loaded for the next run
    progress = [e for e in events if e["event"] == "progress"]
    assert progress[-1]["done"] == progress[-1]["total"] == 4 + 3 and progress[-1]["images"] == 2
    assert {(e["image"], e["steps"]) for e in progress} == {(1, 4), (2, 3)}
    assert any("upscaling to 128×96 with 4x" in (e.get("message") or "") for e in progress)
    assert events[-1]["event"] == "done"


def test_the_hires_fix_samples_its_second_pass_with_a_condition_for_its_size(gen_worker, tmp_path, monkeypatch):
    """An edit model's references are sized for the render (Qwen-Image Edit): the hi-res fix's second pass is
    sampled with a condition made for its own size, not the first pass's (that failed on pass 2)."""
    session, d, profile, *_ = gen_worker
    d.sample_condition = lambda spec: (spec["prompt"], [f"refs for {spec['width']}x{spec['height']}"])
    monkeypatch.setattr(session, "encode",
                        lambda conds: session.texts.update({k: images[0] for k, (_, images) in conds.items()}))
    used = []
    sample = d.sample
    d.sample = lambda model, cond, *a, **kw: used.append(cond) or sample(model, cond, *a, **kw)
    hires = {"factor": 2, "model": "resize", "model_name": "Resize", "strength": 0.35, "steps": 3, "width": 128,
             "height": 96}
    msg = {"id": "g", "out_dir": str(tmp_path / "g"), "profile": profile,
           "params": {"workflow": "t2i", "prompt": "a cat", "width": 64, "height": 48, "steps": 4, "seed": 5,
                      "count": 1, "cfg": 1.0, "hires": hires}}
    session.generate(msg, lambda: False)
    assert used == ["refs for 64x48", "refs for 128x96"]


def test_image_to_image_starts_from_the_input_and_compares_with_the_hires_fix_on_both(gen_worker, tmp_path):
    """Image-to-Image: the input image, cover-cropped to the size and encoded once, is where every render starts (at
    the strength), not a condition; with Compare without LoRAs and a hi-res fix that only resizes, both renders get
    a second pass, and no upscaler is loaded."""
    from PIL import Image

    session, d, profile, loads, sent, rooms, enabled = gen_worker
    src = tmp_path / "in.png"
    Image.new("RGB", (100, 50), (200, 40, 40)).save(src)
    hires = {"factor": 1.5, "model": "resize", "model_name": "Resize", "strength": 0.4, "steps": None, "width": 96,
             "height": 72}
    msg = {"id": "g", "out_dir": str(tmp_path / "g"), "profile": profile,
           "params": {"workflow": "i2i", "prompt": "a cat", "width": 64, "height": 48, "steps": 4, "seed": 9,
                      "count": 1, "cfg": 1.0, "images": [str(src)], "strength": 0.6, "hires": hires,
                      "compare_base": True, "loras": [{"path": "a.safetensors", "strength": 1.0}]}}
    session.generate(msg, lambda: False)
    assert d.spec["images"] == [] and d.encoded[0] == (1, 3, 48, 64)
    assert [(c["size"], c["steps"], c["init"], c["strength"], c["images"]) for c in d.calls] == [
        ((64, 48), 4, (3, 6, 8), 0.6, None), ((64, 48), 4, (3, 6, 8), 0.6, None),
        ((96, 72), 4, (3, 9, 12), 0.4, None), ((96, 72), 4, (3, 9, 12), 0.4, None)]
    items = [e["item"] for e in sent() if e["event"] == "output"]
    assert [(o["variant"], o.get("of"), o["width"]) for o in items] == [
        ("pass1", "lora", 64), ("pass1", "base", 64), ("lora", None, 96), ("base", None, 96)]
    first = Image.open(tmp_path / "g" / "0_lora_pass1.png").convert("RGB")
    assert first.size == (64, 48) and first.getpixel((32, 24))[0] > 150  # the input, as it came back
    assert enabled == [True, False, True, False, True] and not loads and not rooms


def test_image_to_image_and_the_hires_fix_are_checked(client, profiles, monkeypatch, tmp_path):
    """Image-to-Image is offered for every image family with the generic samplers (not video or music ones), takes
    one input image and a strength; the hi-res fix is for images, at 1.5× or 2× with a usable upscaler or a resize,
    and its final size within the caps. Both are kept in the run's params."""
    from PIL import Image

    from dainamo.engine import upscale
    from dainamo.engine.families import FAMILIES
    from dainamo.engine.families.base import sandbox_workflows

    monkeypatch.setattr(sbx, "gpu_count", lambda: 2)
    for fid, F in FAMILIES.items():
        ids = [w.id for w in sandbox_workflows(F, {"variant": "fl2va"})]
        assert ("i2i" in ids) == (F.info.media == ["image"] and F.supports_samplers), fid
        if "i2i" in ids:
            assert ids.index("i2i") == ids.index("t2i") + 1
    d = client.get(f"/api/sandbox/defaults?profile_id={profiles['krea']}").json()
    i2i = next(w for w in d["workflows"] if w["id"] == "i2i")
    assert i2i["label"] == "Image-to-Image" and (i2i["images"]["min"], i2i["images"]["max"]) == (1, 1)
    assert "i2i" not in [w["id"] for w in client.get(f"/api/sandbox/defaults?profile_id={profiles['ref2va']}").json()["workflows"]]
    assert any(w["id"] == "i2i" for w in client.get("/api/system").json()["workflows"])

    img = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(img)
    with pytest.raises(HTTPException, match="needs 1 input image"):
        check(profile_id=profiles["krea"], workflow="i2i")
    with pytest.raises(HTTPException, match="needs 1 input image"):
        check(profile_id=profiles["krea"], workflow="i2i", images=[str(img)] * 2)
    for bad in (0.01, 1.2):
        with pytest.raises(HTTPException, match="Strength must be 0.05–1"):
            check(profile_id=profiles["krea"], workflow="i2i", images=[str(img)], strength=bad)
    _, p, _ = check(profile_id=profiles["krea"], workflow="i2i", images=[str(img)], strength=0.6)
    assert p["workflow"] == "i2i" and p["strength"] == 0.6 and p["images"] == [str(img)] and p["hires"] is None
    assert "strength" not in check(profile_id=profiles["krea"], workflow="t2i", strength=0.3)[1]

    # the hi-res fix
    _, p, _ = check(profile_id=profiles["krea"], workflow="t2i", width=1000, height=760,
                    hires={"factor": 2, "model": "resize", "strength": 0.35, "steps": None})
    assert p["hires"] == {"factor": 2.0, "model": "resize", "model_name": "Resize", "strength": 0.35, "steps": None,
                          "width": 1984, "height": 1504}  # (992 × 752) × 2, on the 16 px grid
    _, p, _ = check(profile_id=profiles["qwen"], workflow="edit", images=[str(img)], width=1024, height=768,
                    hires={"factor": 1.5, "model": "resize", "strength": 0.5, "steps": 12})
    assert (p["hires"]["width"], p["hires"]["height"], p["hires"]["steps"]) == (1536, 1152, 12)
    up = tmp_path / "4x.pth"
    up.write_bytes(b"x")
    monkeypatch.setattr(upscale, "identify", lambda path, files=None: upscale.Info(
        path, "unknown" if "bad" in path else "pixel", "4x", scale=4, error="unreadable" if "bad" in path else None))
    _, p, _ = check(profile_id=profiles["krea"], workflow="i2i", images=[str(img)],
                    hires={"factor": 2, "model": str(up), "strength": 0.3})
    assert p["hires"]["model"] == str(up) and p["hires"]["model_name"] == "4x" and p["strength"] == 0.6
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"x")
    for hires, why in (({"factor": 3, "model": "resize"}, "1.5× or 2×"),
                       ({"factor": 2, "model": "resize", "strength": 0.95}, "0.1–0.9"),
                       ({"factor": 2, "model": "resize", "steps": 0}, "1–200"),
                       ({"factor": 2, "model": ""}, "Choose an upscaler"),
                       ({"factor": 2, "model": str(tmp_path / "nope.pth")}, "not found"),
                       ({"factor": 2, "model": str(bad)}, "unreadable")):
        with pytest.raises(HTTPException, match=why):
            check(profile_id=profiles["krea"], workflow="t2i", hires=hires)
    with pytest.raises(HTTPException, match="at most 4096"):
        check(profile_id=profiles["krea"], workflow="t2i", width=2304, height=1024, hires={"factor": 2, "model": "resize"})
    with pytest.raises(HTTPException, match="for images"):
        check(profile_id=profiles["fl2va"], workflow="t2v", hires={"factor": 2, "model": "resize"})

"""LoRAs merged into one (dainamo.engine.merge): exact stacking, cutting to a rank, LoKrs and whole changes, kohya
files, the Sandbox applying a merge as it applies its inputs, the job that makes one, and the Merge & extract
routes (a LoRA extracted from any file, too)."""

import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from dainamo.engine import merge
from dainamo.engine.lora import AdapterStack, adapter_layers
from tests.test_memory import D, N, TinyDriver, write_model

LINEARS = [f"blocks.{i}.{m}" for i in range(N) for m in ("attn", "mlp")]


def lora(path, rank, alpha, seed, layers=LINEARS, kohya=False, extra=None):
    """A LoRA of ``rank`` on ``layers``, alpha ``alpha``, its up matrices not zero (as after training)."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name in layers:
        a, b = torch.randn(rank, D, generator=g) * 0.1, torch.randn(D, rank, generator=g) * 0.1
        if kohya:
            stem = "lora_unet_" + name.replace(".", "_")
            sd |= {f"{stem}.lora_down.weight": a, f"{stem}.lora_up.weight": b, f"{stem}.alpha": torch.tensor(alpha)}
        else:
            stem = "diffusion_model." + name
            sd |= {f"{stem}.lora_A.weight": a, f"{stem}.lora_B.weight": b, f"{stem}.alpha": torch.tensor(alpha)}
    sd |= extra or {}
    save_file({k: v.bfloat16() if v.dim() else v for k, v in sd.items()}, str(path), {"dainamo.trigger": f"t{seed}"})
    return str(path)


def change(path) -> dict[str, torch.Tensor]:
    """{module: its whole change} of a LoRA file, as the Sandbox applies it."""
    out = {}
    for name, a in adapter_layers(load_file(path)):
        key = name + (".diff_b" if a.bias is not None else "")
        out[key] = out.get(key, 0) + a.dense()
    return out


def summed(*inputs) -> dict[str, torch.Tensor]:
    out = {}
    for path, s in inputs:
        for k, v in change(path).items():
            out[k] = out.get(k, 0) + s * v
    return out


@pytest.fixture
def layers(tmp_path):
    return merge.model_layers(TinyDriver, {"diffusion_model": write_model(tmp_path / "base.safetensors")})


def test_the_model_is_built_from_its_header(layers):
    assert layers.linears["blocks.0.attn"] == (D, D) and "proj_in" in layers.linears
    assert layers.tensors["blocks.0.norm.weight"] == (D,)
    assert layers.kohya["blocks_0_attn"] == "blocks.0.attn"


def test_an_exact_merge_stacks_the_loras(tmp_path, layers):
    a, b = lora(tmp_path / "a.safetensors", 4, 8.0, 1), lora(tmp_path / "b.safetensors", 2, 2.0, 2, LINEARS[:4])
    out = str(tmp_path / "m.safetensors")
    made = merge.merge([(a, 1.0), (b, -0.5)], out, 0, layers, "cpu", {"dainamo.trigger": "x"})
    assert made["kept"] == 1.0 and made["cut"] == 0 and made["worst"] is None
    assert made["layers"] == len(LINEARS) and made["whole"] == 0
    assert [i["matched"] for i in made["inputs"]] == [len(LINEARS), 4]
    sd = load_file(out)
    assert sd["diffusion_model.blocks.0.attn.lora_A.weight"].shape == (6, D)  # the ranks added up
    assert sd["diffusion_model.blocks.5.mlp.lora_A.weight"].shape == (4, D)
    want, got = summed((a, 1.0), (b, -0.5)), change(out)
    assert set(got) == set(want)
    for k in want:
        assert torch.allclose(got[k], want[k], atol=2e-3 * want[k].abs().max())
    with safe_open(out, "pt") as f:
        assert f.metadata()["dainamo.trigger"] == "x" and f.metadata()["dainamo.kept"] == "1.0"
    # about as big as the plan said (it leaves out each layer's alpha, a scalar)
    plan = merge.plan([a, b], layers)
    assert plan["sizes"]["0"] == sum(v.numel() * v.element_size() for k, v in sd.items() if not k.endswith(".alpha"))
    assert [(i["matched"], i["ranks"], i["trigger"]) for i in plan["inputs"]] == [(12, [4, 4], "t1"), (4, [2, 2], "t2")]


def test_a_merge_cut_to_a_rank_is_the_best_approximation_of_the_sum(tmp_path, layers):
    a, b = lora(tmp_path / "a.safetensors", 4, 4.0, 1), lora(tmp_path / "b.safetensors", 4, 4.0, 2)
    out = str(tmp_path / "m.safetensors")
    made = merge.merge([(a, 1.0), (b, 0.7)], out, 3, layers, "cpu")
    assert made["cut"] == len(LINEARS) and 0.3 < made["kept"] < 1.0 and made["worst"]["kept"] <= made["kept"]
    want, got = summed((a, 1.0), (b, 0.7)), change(out)
    total = kept = 0.0
    for k, d in want.items():
        U, S, Vh = torch.linalg.svd(d)
        best = U[:, :3] * S[:3] @ Vh[:3]
        assert torch.allclose(got[k], best, atol=1e-2 * best.abs().max())
        total, kept = total + float((S ** 2).sum()), kept + float((S[:3] ** 2).sum())
    assert made["kept"] == pytest.approx(kept / total, abs=1e-3)
    assert load_file(out)["diffusion_model.blocks.0.attn.lora_A.weight"].shape == (3, D)
    # a rank above the stacked one keeps it as it is; one where a LoRA is no smaller saves the change whole
    made = merge.merge([(a, 1.0)], out, 16, layers, "cpu")
    assert made["cut"] == 0 and made["kept"] == 1.0 and made["layers"] == len(LINEARS)
    made = merge.merge([(a, 1.0), (b, 1.0)], out, 0, layers, "cpu")
    assert made["layers"] == len(LINEARS)  # rank 8: 8 * (64 + 64) < 64 * 64
    big = lora(tmp_path / "big.safetensors", 28, 28.0, 3)
    made = merge.merge([(a, 1.0), (big, 1.0)], out, 0, layers, "cpu")  # rank 32 stacked: 32 * (64 + 64) is no smaller
    assert made["whole"] == len(LINEARS) and made["layers"] == 0 and made["kept"] == 1.0
    assert "diffusion_model.blocks.0.attn.diff" in load_file(out)


def test_a_kohya_lora_merges_as_the_same_lora_in_peft_keys(tmp_path, layers):
    peft, kohya = lora(tmp_path / "p.safetensors", 4, 2.0, 1), lora(tmp_path / "k.safetensors", 4, 2.0, 1, kohya=True)
    one, two = str(tmp_path / "1.safetensors"), str(tmp_path / "2.safetensors")
    merge.merge([(peft, 0.8)], one, 0, layers, "cpu")
    made = merge.merge([(kohya, 0.8)], two, 0, layers, "cpu")
    assert made["inputs"][0]["matched"] == len(LINEARS)
    assert load_file(one).keys() == load_file(two).keys()
    for k, v in load_file(one).items():
        assert torch.equal(v, load_file(two)[k])
    assert merge.describe(kohya, layers)["matched"] == len(LINEARS)


def test_loras_that_spell_a_layer_differently_are_merged_together(tmp_path, layers):
    """A kohya (``lora_unet_``) and a ``diffusion_model.`` LoRA name the same layers: each layer's changes stack,
    rather than one input's overwriting the other's."""
    a = lora(tmp_path / "a.safetensors", 4, 4.0, 1)
    b = lora(tmp_path / "b.safetensors", 2, 2.0, 2, kohya=True)
    out = str(tmp_path / "m.safetensors")
    made = merge.merge([(a, 1.0), (b, 0.5)], out, 0, layers, "cpu")
    assert made["layers"] == len(LINEARS) and made["kept"] == 1.0
    assert load_file(out)["diffusion_model.blocks.0.attn.lora_A.weight"].shape == (6, D)
    same_b = lora(tmp_path / "b_peft.safetensors", 2, 2.0, 2)  # b's tensors under PEFT keys, as change() reads them
    want, got = summed((a, 1.0), (same_b, 0.5)), change(out)
    assert set(got) == set(want)
    for k in want:
        assert torch.allclose(got[k], want[k], atol=2e-3 * want[k].abs().max())


def test_a_lokr_merges_only_at_a_rank(tmp_path, layers):
    g = torch.Generator().manual_seed(4)
    sd = {}
    for name in LINEARS[:2]:  # an 8 x 8 kron 8 x 8 of rank 2: rank 16 in all
        stem = "diffusion_model." + name
        sd |= {f"{stem}.lokr_w1": torch.randn(8, 8, generator=g), f"{stem}.lokr_w2_a": torch.randn(8, 2, generator=g),
               f"{stem}.lokr_w2_b": torch.randn(2, 8, generator=g) * 0.1, f"{stem}.alpha": torch.tensor(2.0)}
    lokr = str(tmp_path / "lokr.safetensors")
    save_file(sd, lokr)
    a = lora(tmp_path / "a.safetensors", 4, 4.0, 1)
    out = str(tmp_path / "m.safetensors")
    with pytest.raises(ValueError, match="choose a rank"):
        merge.merge([(a, 1.0), (lokr, 1.0)], out, 0, layers, "cpu")
    plan = merge.plan([a, lokr], layers)
    assert plan["lokr"] and plan["sizes"]["0"] is None and plan["sizes"]["16"]
    assert plan["inputs"][1]["kinds"] == ["lokr"] and plan["inputs"][1]["matched"] == 2
    made = merge.merge([(a, 1.0), (lokr, 0.5)], out, 20, layers, "cpu")  # rank 4 + 16 holds all of it
    assert made["kept"] == pytest.approx(1.0, abs=1e-4)
    want, got = summed((a, 1.0), (lokr, 0.5)), change(out)
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-2 * want[k].abs().max())


def test_whole_changes_are_added_up_and_a_weights_cut_to_the_rank(tmp_path, layers):
    """An extracted LoRA's: a matrix kept whole (``.diff``), a norm's scale and a bias's change."""
    g = torch.Generator().manual_seed(5)
    whole = {"diffusion_model.blocks.0.mlp.diff": torch.randn(D, D, generator=g) * 0.01,
             "diffusion_model.blocks.0.norm.diff": torch.randn(D, generator=g) * 0.01,
             "diffusion_model.proj_in.diff_b": torch.randn(D, generator=g) * 0.01}
    ex = lora(tmp_path / "ex.safetensors", 2, 2.0, 1, LINEARS[2:4], extra=whole)
    a = lora(tmp_path / "a.safetensors", 4, 4.0, 2)
    out = str(tmp_path / "m.safetensors")
    made = merge.merge([(ex, 2.0), (a, 1.0)], out, 0, layers, "cpu")
    sd, src = load_file(out), load_file(ex)  # (saved in bf16)
    for k in ("diffusion_model.blocks.0.norm.diff", "diffusion_model.proj_in.diff_b"):
        assert torch.equal(sd[k], 2 * src[k].float())
    assert sd["diffusion_model.proj_in.diff_b"].dtype == torch.float32  # small: kept in fp32
    assert "diffusion_model.blocks.0.mlp.diff" in sd and made["whole"] == 3  # exact: the matrix stays whole
    assert made["kept"] == 1.0
    want, got = summed((ex, 2.0), (a, 1.0)), change(out)
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-2 * want[k].abs().max()), k
    made = merge.merge([(ex, 2.0), (a, 1.0)], out, 4, layers, "cpu")  # cut: the matrix too
    sd = load_file(out)
    assert "diffusion_model.blocks.0.mlp.diff" not in sd and sd["diffusion_model.blocks.0.mlp.lora_A.weight"].shape == (4, D)
    assert "diffusion_model.blocks.0.norm.diff" in sd and made["kept"] < 1.0


def test_layers_the_model_lacks_are_left_out_and_a_lora_for_another_model_refused(tmp_path, layers):
    extra = {"diffusion_model.blocks.9.attn.lora_A.weight": torch.randn(4, D),
             "diffusion_model.blocks.9.attn.lora_B.weight": torch.randn(D, 4),
             "diffusion_model.blocks.0.attn.lora_A.weight": torch.randn(4, 2 * D)}  # not this model's size
    a = lora(tmp_path / "a.safetensors", 4, 4.0, 1, LINEARS[1:], extra=extra | {
        "diffusion_model.blocks.0.attn.lora_B.weight": torch.randn(D, 4)})
    d = merge.describe(a, layers)
    assert (d["layers"], d["matched"]) == (len(LINEARS) + 1, len(LINEARS) - 1)
    out = str(tmp_path / "m.safetensors")
    made = merge.merge([(a, 1.0)], out, 0, layers, "cpu")
    assert made["inputs"][0]["matched"] == len(LINEARS) - 1 and made["layers"] == len(LINEARS) - 1
    assert not any(k.startswith(("diffusion_model.blocks.9.", "diffusion_model.blocks.0.attn.")) for k in load_file(out))
    other = lora(tmp_path / "other.safetensors", 4, 4.0, 1, ["double_blocks.0.img_attn.qkv"])
    with pytest.raises(ValueError, match="matches no layers"):
        merge.merge([(a, 1.0), (other, 1.0)], out, 0, layers, "cpu")
    dora = lora(tmp_path / "dora.safetensors", 4, 4.0, 1, extra={"diffusion_model.blocks.0.attn.dora_scale": torch.ones(D)})
    with pytest.raises(ValueError, match="DoRA"):
        merge.describe(dora, layers)


def test_the_sandbox_applies_a_merge_as_it_applies_its_loras(tmp_path, layers):
    from tests.test_memory import TinyDiT

    a, b = lora(tmp_path / "a.safetensors", 4, 4.0, 1), lora(tmp_path / "b.safetensors", 2, 1.0, 2)
    out = str(tmp_path / "m.safetensors")
    merge.merge([(a, 0.6), (b, -1.2)], out, 0, layers, "cpu")
    torch.manual_seed(0)
    model = TinyDiT(D, N)
    x = torch.randn(5, D)
    stack = AdapterStack(model, "cpu", torch.float32)
    stack.set([("a", load_file(a), 0.6), ("b", load_file(b), -1.2)])
    with torch.no_grad():
        want = model(x)
    stack.set([("m", load_file(out), 1.0)])
    with torch.no_grad():
        got = model(x)
    stack.clear()
    with torch.no_grad():
        before = model(x)
    assert float((got - want).norm() / (want - before).norm()) < 0.01


# --- the job ------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer needs a GPU")
def test_a_merge_job_saves_the_merged_lora(tmp_path, monkeypatch):
    from dainamo.engine import trainer as trainer_mod

    base = write_model(tmp_path / "base.safetensors")
    a, b = lora(tmp_path / "a.safetensors", 4, 4.0, 1), lora(tmp_path / "b.safetensors", 4, 4.0, 2, LINEARS[:3])
    monkeypatch.setattr(trainer_mod, "get_family", lambda fid: TinyDriver)
    job = tmp_path / "job"
    job.mkdir()
    (job / "job.json").write_text(json.dumps({
        "name": "a+b_r6", "family": "tiny", "profile": {"diffusion_model": base}, "datasets": [],
        "settings": {"steps": 0, "merge_enabled": True, "merge_rank": 6,
                     "merge_loras": [{"path": a, "strength": 1.0, "trigger": "sks"},
                                     {"path": b, "strength": 0.5, "trigger": "ohwx"}]}}))
    t = trainer_mod.Trainer(str(job))
    assert t.run_merge() == "completed"
    assert t.status["checkpoints"] == [{"step": 0, "file": "a_b_r6.safetensors", "final": True}]
    made = t.status["merge"]
    assert made["rank"] == 6 and made["cut"] == 3 and made["layers"] == len(LINEARS) and made["kept"] < 1.0
    assert t.status["phase"] == "completed" and t.status["step"] == 0
    with safe_open(str(job / "output" / "a_b_r6.safetensors"), "pt") as f:
        meta = f.metadata()
    assert (meta["dainamo.network"], meta["dainamo.trigger"], meta["ss_network_dim"]) == ("lora", "sks, ohwx", "6")
    assert json.loads(meta["dainamo.merged_from"]) == [{"file": "a.safetensors", "strength": 1.0},
                                                       {"file": "b.safetensors", "strength": 0.5}]


# --- the server ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from dainamo.server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def held(monkeypatch):
    """The jobs a test queues aren't started, and are stopped after it, so later tests' queues start empty."""
    from dainamo.server import db
    from dainamo.server.queue import scheduler

    monkeypatch.setattr(scheduler, "_launch", lambda: None)
    yield
    db.execute("UPDATE jobs SET status='stopped' WHERE status='queued'")


def test_loras_are_checked_and_queued_to_merge(client, held, tmp_path, monkeypatch):
    from dainamo.server import db
    from dainamo.server.queue import job_dir
    from dainamo.server.routes import tools

    monkeypatch.setattr(tools, "get_family", lambda fid: TinyDriver)
    base = write_model(tmp_path / "base.safetensors")
    pid = client.post("/api/profiles", json={"name": "merge moss", "family": "moss_image",
                                             "data": {"diffusion_model": base}}).json()["id"]
    a, b = lora(tmp_path / "a.safetensors", 4, 4.0, 1), lora(tmp_path / "b.safetensors", 4, 4.0, 2, LINEARS[:3])
    check = client.post("/api/tools/merge/check", json={"profile_id": pid, "loras": [a, b]}).json()
    assert [(i["file"], i["matched"], i["trigger"]) for i in check["inputs"]] == [("a.safetensors", 12, "t1"),
                                                                                   ("b.safetensors", 3, "t2")]
    # rank 8 stacked: 16 keeps all of it
    assert check["sizes"]["16"] == check["sizes"]["0"] > check["sizes"]["4"] and not check["lokr"]
    assert "twice" in client.post("/api/tools/merge/check", json={"profile_id": pid, "loras": [a, a]}).json()["detail"]
    assert "not found" in client.post("/api/tools/merge/check", json={
        "profile_id": pid, "loras": [str(tmp_path / "gone.safetensors")]}).json()["detail"]

    body = {"profile_id": pid, "loras": [{"path": a, "strength": 1.0}, {"path": b, "strength": 0.5}], "rank": 0}
    assert "not 0" in client.post("/api/tools/merge", json={**body, "loras": [{"path": a, "strength": 0}]}).json()["detail"]
    assert "rank" in client.post("/api/tools/merge", json={**body, "rank": 5000}).json()["detail"]
    other = lora(tmp_path / "other.safetensors", 4, 4.0, 1, ["double_blocks.0.img_attn.qkv"])
    assert "matches no layers" in client.post("/api/tools/merge", json={
        **body, "loras": [{"path": other}]}).json()["detail"]
    r = client.post("/api/tools/merge", json=body)
    assert r.status_code == 200
    m = r.json()
    s = m["config"]["settings"]
    assert m["status"] == "queued" and m["name"] == "a+b" and m["family"] == "moss_image"
    assert s["merge_enabled"] and s["merge_rank"] == 0 and m["config"]["profile"]["diffusion_model"] == base
    assert s["merge_loras"] == [{"path": a, "strength": 1.0, "trigger": "t1"}, {"path": b, "strength": 0.5, "trigger": "t2"}]
    assert next(j for j in client.get("/api/jobs").json() if j["id"] == m["id"])["merge"]
    one = client.post("/api/tools/merge", json={**body, "loras": [{"path": a}], "rank": 2, "name": " mine "}).json()
    assert one["name"] == "mine" and one["config"]["settings"]["merge_rank"] == 2
    assert client.post("/api/tools/merge", json={**body, "loras": [{"path": a}], "rank": 2}).json()["name"] == "a_r2"
    # it's merged again rather than edited, continued or scored
    assert "merge again" in client.put(f"/api/jobs/{m['id']}", json={
        "name": "x", "profile_id": pid, "datasets": [], "settings": {}}).json()["detail"]
    db.execute("UPDATE jobs SET status='completed' WHERE id=?", (m["id"],))
    assert "trains nothing" in client.post(f"/api/jobs/{m['id']}/action", json={"action": "extend", "steps": 9}).json()["detail"]
    assert "merged LoRA has one file" in client.post(f"/api/jobs/{m['id']}/evaluate", json={"settings": {}}).json()["detail"]
    # once made, its LoRA is in the Sandbox with the family's others
    out = os.path.join(job_dir(m["id"]), "output")
    os.makedirs(out)
    open(os.path.join(out, "a_b.safetensors"), "wb").close()
    with open(os.path.join(job_dir(m["id"]), "status.json"), "w") as f:
        json.dump({"phase": "completed", "checkpoints": [{"step": 0, "file": "a_b.safetensors", "final": True}]}, f)
    listed = client.get("/api/sandbox/loras", params={"family": "moss_image"}).json()
    assert [x["file"] for x in listed if x["job_id"] == m["id"]] == ["a_b.safetensors"]


def test_a_lora_is_queued_to_extract_from_any_finetuned_file(client, held, tmp_path):
    from tests.test_extract import finetuned

    base, tuned, _, _ = finetuned(tmp_path)
    pid = client.post("/api/profiles", json={"name": "extract moss", "family": "moss_image",
                                             "data": {"diffusion_model": base}}).json()["id"]
    sizes = client.get("/api/tools/extract/sizes", params={"profile_id": pid, "path": tuned}).json()
    assert sizes["base"] == base and sizes["base_found"] and int(sizes["sizes"]["16"]) < int(sizes["sizes"]["64"])
    gone = client.get("/api/tools/extract/sizes", params={"profile_id": pid, "path": tuned,
                                                          "base": str(tmp_path / "gone.safetensors")}).json()
    assert not gone["base_found"] and "not found" in gone["base_error"]
    body = {"profile_id": pid, "tuned": tuned, "rank": 16, "trigger": " sks "}
    assert "same file" in client.post("/api/tools/extract", json={**body, "tuned": base}).json()["detail"]
    assert "rank" in client.post("/api/tools/extract", json={**body, "rank": 0}).json()["detail"]
    r = client.post("/api/tools/extract", json=body)
    assert r.status_code == 200
    ex = r.json()
    s = ex["config"]["settings"]
    assert ex["name"] == "tuned_r16" and ex["config"]["datasets"] == []
    assert s["extract_from"] == {"file": "tuned.safetensors", "step": 0, "trigger": "sks"}
    assert (s["extract_source"], s["extract_base"]) == (tuned, base)
    assert next(j for j in client.get("/api/jobs").json() if j["id"] == ex["id"])["extract"]

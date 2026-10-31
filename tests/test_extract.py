"""LoRAs extracted from a finetune (dainamo.engine.extract): the change as low-rank factors and whole tensors,
applied by the Sandbox's adapter stack, and the job that makes one."""

import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from dainamo.engine import extract
from dainamo.engine.finetune import Layout, save_model
from dainamo.engine.lora import AdapterStack
from tests.test_memory import D, N, TinyDriver, write_model


def test_the_factors_are_the_best_approximation_of_their_rank():
    torch.manual_seed(0)
    for shape in ((40, 24), (24, 40)):
        d = torch.randn(*shape)
        up, down, kept = extract.low_rank(d, 6)
        assert up.shape == (shape[0], 6) and down.shape == (6, shape[1])
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        best = U[:, :6] * S[:6] @ Vh[:6]
        assert torch.allclose(up @ down, best, atol=1e-4)
        assert kept == pytest.approx(float((S[:6] ** 2).sum() / (S ** 2).sum()), rel=1e-4)
        assert float(up.norm()) == pytest.approx(float(down.norm()), rel=1e-3)  # the singular values split between them
        # a change of lower rank than asked for is held exactly, with no factors for what isn't there
        low = torch.randn(shape[0], 3) @ torch.randn(3, shape[1])
        up, down, kept = extract.low_rank(low, 8)
        assert torch.allclose(up @ down, low, atol=1e-4) and kept == pytest.approx(1.0)
        assert up.isfinite().all() and down.isfinite().all()


def finetuned(tmp_path):
    """A tiny model's file (its attn layers int8) and a "finetune" of it: each block's attn changed by a rank-2
    matrix (which bf16's rounding adds a little noise to), its mlp by a full-rank one, its norm's scale and the
    input's bias nudged, and the last block left alone. -> (base, tuned, base model, tuned model)."""
    base = write_model(tmp_path / "base.safetensors")
    d = TinyDriver({"diffusion_model": base}, torch.device("cpu"), log=lambda *_: None)
    model = d.load_transformer(trainable=True)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(1)
    with torch.no_grad():
        for b in model.blocks[:-1]:
            b.attn.weight += (torch.randn(D, 2) @ torch.randn(2, D) * 5e-3).bfloat16()
            b.mlp.weight += (torch.randn(D, D) * 3e-3).bfloat16()
            b.norm.weight += (torch.randn(D) * 1e-2).bfloat16()
        model.proj_in.bias += (torch.randn(D) * 1e-2).bfloat16()
    tuned = str(tmp_path / "tuned.safetensors")
    save_model(model, tuned, Layout(base, model), {"dainamo.base": "base.safetensors"})
    base_model = TinyDriver({"diffusion_model": base}, torch.device("cpu")).load_transformer(trainable=True)
    for k, v in base_model.state_dict().items():
        assert torch.equal(v, original[k])
    return base, tuned, base_model, model


def test_a_finetune_is_extracted_as_loras_and_whole_changes(tmp_path):
    base, tuned, base_model, model = finetuned(tmp_path)
    out = str(tmp_path / "lora.safetensors")
    made = extract.extract(base, tuned, out, 4, "blocks", "cpu", {"dainamo.family": "tiny"}, log=lambda *_: None)
    sd = load_file(out)
    p = "diffusion_model."
    for i in range(N - 1):
        assert sd[f"{p}blocks.{i}.attn.lora_A.weight"].shape == (4, D)
        assert sd[f"{p}blocks.{i}.attn.lora_B.weight"].shape == (D, 4) and float(sd[f"{p}blocks.{i}.attn.alpha"]) == 4
        # the int8 original is compared as training expanded it, so the change is what was added, not rounding
        change = model.blocks[i].attn.weight.float() - base_model.blocks[i].attn.weight.float()
        lora = sd[f"{p}blocks.{i}.attn.lora_B.weight"].float() @ sd[f"{p}blocks.{i}.attn.lora_A.weight"].float()
        U, S, Vh = torch.linalg.svd(change)
        assert (lora - U[:, :4] * S[:4] @ Vh[:4]).norm() / change.norm() < 0.01  # its best rank-4 approximation
        assert (lora - change).norm() / change.norm() < 0.05  # which is most of a rank-2 change
        # a norm's scale is kept whole, in fp32, exactly
        norm = sd[f"{p}blocks.{i}.norm.diff"]
        assert norm.dtype == torch.float32
        assert torch.equal(norm, model.blocks[i].norm.weight.float() - base_model.blocks[i].norm.weight.float())
    assert torch.equal(sd[f"{p}proj_in.diff_b"], model.proj_in.bias.float() - base_model.proj_in.bias.float())
    assert not any(k.startswith(f"{p}blocks.{N - 1}.") for k in sd)  # the block that didn't change
    assert made["layers"] == 2 * (N - 1) and made["whole"] == (N - 1) + 1
    assert made["unchanged"] == len(base_model.state_dict()) - made["layers"] - made["whole"]
    assert 0.3 < made["kept"] < 0.95  # the mlp changes are full-rank: rank 4 keeps part of them
    assert made["worst"]["name"].endswith(".mlp") and made["worst"]["kept"] < made["kept"]
    assert 0 < made["change"] < extract.WRONG_BASE_CHANGE and made["bytes"] == os.path.getsize(out)
    with safe_open(out, "pt") as f:
        meta = f.metadata()
    assert meta["dainamo.family"] == "tiny" and float(meta["dainamo.kept"]) == made["kept"]
    # at a rank where a LoRA would be no smaller, every matrix is kept whole: all of the change
    full = extract.extract(base, tuned, str(tmp_path / "full.safetensors"), 64, "blocks", "cpu", log=lambda *_: None)
    assert full["layers"] == 0 and full["kept"] == 1.0
    assert extract.sizes(tuned, (4, 64)) == {
        4: (2 * N + 1) * 2 * 4 * (2 * D) + 4 * (N * 3 * D + D),  # every matrix a LoRA, every vector whole in fp32
        64: (2 * N + 1) * 2 * D * D + 4 * (N * 3 * D + D)}


def test_extracting_needs_a_change_from_the_same_model(tmp_path):
    base, tuned, _, _ = finetuned(tmp_path)
    with pytest.raises(ValueError, match="no change"):
        extract.extract(base, base, str(tmp_path / "x.safetensors"), 4, "blocks", "cpu")
    from safetensors.torch import save_file

    other = str(tmp_path / "other.safetensors")
    sd = load_file(tuned)
    sd["proj_in.weight"] = torch.zeros(D, D // 2, dtype=torch.bfloat16)
    save_file(sd, other)
    with pytest.raises(ValueError, match="aren't the same model"):
        extract.extract(base, other, str(tmp_path / "x.safetensors"), 4, "blocks", "cpu")
    with pytest.raises(ValueError, match="rank"):
        extract.extract(base, tuned, str(tmp_path / "x.safetensors"), 0, "blocks", "cpu")


def test_the_sandbox_applies_an_extracted_lora_as_the_finetune(tmp_path):
    base, tuned, base_model, model = finetuned(tmp_path)
    exact = str(tmp_path / "exact.safetensors")
    extract.extract(base, tuned, exact, 64, "blocks", "cpu", log=lambda *_: None)  # all of the change, whole
    low = str(tmp_path / "low.safetensors")
    extract.extract(base, tuned, low, 4, "blocks", "cpu", log=lambda *_: None)
    torch.manual_seed(2)
    x = torch.randn(5, D)
    model, base_model = model.float(), base_model.float()
    norms = {k: v.clone() for k, v in base_model.state_dict().items() if "norm" in k}
    with torch.no_grad():
        want, before = model(x), base_model(x)
    stack = AdapterStack(base_model, "cpu", torch.float32)

    def err():
        with torch.no_grad():
            return float((base_model(x) - want).norm() / (before - want).norm())

    report = stack.set([("exact", load_file(exact), 1.0)])
    assert report[0]["matched"] == report[0]["total"]  # norms' scales and biases too
    assert err() < 0.01
    stack.set_enabled(False)  # the base model, for "compare with base"
    with torch.no_grad():
        assert torch.equal(base_model(x), before)
    stack.set_enabled(True)
    assert err() < 0.01
    stack.set([("low", load_file(low), 1.0)])
    assert 0.01 < err() < 0.8  # rank 4 holds some of the change
    stack.set([("exact", load_file(exact), 0.5)])
    assert 0.3 < err() < 0.7  # half the change
    stack.clear()
    with torch.no_grad():
        assert torch.equal(base_model(x), before)
    for k, v in base_model.state_dict().items():
        if k in norms:
            assert torch.equal(v, norms[k])


# --- the job ------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer needs a GPU")
def test_an_extract_job_takes_the_checkpoint_at_its_step(tmp_path, monkeypatch):
    from dainamo.engine import trainer as trainer_mod

    base, tuned, _, _ = finetuned(tmp_path)
    monkeypatch.setattr(trainer_mod, "get_family", lambda fid: TinyDriver)
    jobs = tmp_path / "jobs"
    src = jobs / "ft_src"  # its folder, named after it
    (src / "output").mkdir(parents=True)
    # the finetune trained on after this was chosen, so its final model was renamed
    os.replace(tuned, src / "output" / "ft_000100.safetensors")
    (src / "status.json").write_text(json.dumps({"checkpoints": [{"step": 100, "file": "ft_000100.safetensors"}]}))
    job = jobs / "ex"
    job.mkdir()
    (job / "job.json").write_text(json.dumps({
        "name": "ft_r4", "family": "tiny", "profile": {"diffusion_model": base}, "datasets": [],
        "settings": {"steps": 0, "extract_enabled": True, "extract_rank": 4,
                     "extract_source": str(src / "output" / "ft.safetensors"), "extract_base": base,
                     "extract_from": {"job_id": "src", "job_name": "ft", "file": "ft.safetensors", "step": 100,
                                      "trigger": "sks", "folder": "ft_src"}}}))
    t = trainer_mod.Trainer(str(job))
    assert t.run_extract() == "completed"
    assert t.status["checkpoints"] == [{"step": 100, "file": "ft_r4.safetensors", "final": True}]
    made = t.status["extract"]
    assert made["source"] == "ft_000100.safetensors" and made["base"] == "base.safetensors" and made["rank"] == 4
    assert t.status["phase"] == "completed" and t.status["step"] == 0
    with safe_open(str(job / "output" / "ft_r4.safetensors"), "pt") as f:
        meta = f.metadata()
    assert (meta["dainamo.network"], meta["dainamo.step"], meta["dainamo.trigger"]) == ("lora", "100", "sks")
    assert meta["ss_network_dim"] == "4"


# --- the server ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from dainamo.server.app import app

    with TestClient(app) as c:
        yield c


def test_a_finetune_checkpoint_is_queued_to_extract_a_lora(client, tmp_path, monkeypatch):
    from dainamo.server import db
    from dainamo.server.queue import job_dir, scheduler

    monkeypatch.setattr(scheduler, "_launch", lambda: None)
    base, tuned, _, _ = finetuned(tmp_path)
    pid = client.post("/api/profiles", json={"name": "ex moss", "family": "moss_image", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "ex", "path": str(tmp_path), "trigger": "sks"}).json()["id"]

    def job(**settings):
        jid = client.post("/api/jobs", json={"name": "ft", "profile_id": pid, "datasets": [{"id": did}],
                                             "settings": {"optimizer": "adafactor", **settings}}).json()["id"]
        cfg = client.get(f"/api/jobs/{jid}/config").json()
        cfg["profile"]["diffusion_model"] = base
        d = job_dir(jid)
        os.makedirs(os.path.join(d, "output"))
        os.link(tuned, os.path.join(d, "output", "ft_000200.safetensors"))
        with open(os.path.join(d, "status.json"), "w") as f:
            json.dump({"phase": "completed", "checkpoints": [{"step": 200, "file": "ft_000200.safetensors"}]}, f)
        db.execute("UPDATE jobs SET status='completed', config=? WHERE id=?", (json.dumps(cfg), jid))
        return jid

    ft, lora = job(finetune_enabled=True), job()
    sizes = client.get(f"/api/jobs/{ft}/extract", params={"file": "ft_000200.safetensors"}).json()
    assert sizes["base"] == base and sizes["base_found"]
    assert int(sizes["sizes"]["16"]) < int(sizes["sizes"]["64"])
    body = {"file": "ft_000200.safetensors", "rank": 16}
    assert "Only a finetune" in client.post(f"/api/jobs/{lora}/extract", json=body).json()["detail"]
    assert client.post(f"/api/jobs/{ft}/extract", json={**body, "file": "gone.safetensors"}).status_code == 404
    assert "rank" in client.post(f"/api/jobs/{ft}/extract", json={**body, "rank": 0}).json()["detail"]
    r = client.post(f"/api/jobs/{ft}/extract", json=body)
    assert r.status_code == 200
    ex = r.json()
    s = ex["config"]["settings"]
    assert ex["status"] == "queued" and ex["name"] == "ft_000200_r16" and ex["config"]["datasets"] == []
    assert s["extract_from"] == {"job_id": ft, "job_name": "ft", "file": "ft_000200.safetensors", "step": 200,
                                 "trigger": "sks", "folder": os.path.basename(job_dir(ft))}
    assert s["extract_base"] == base and s["extract_source"].endswith("ft_000200.safetensors")
    assert next(j for j in client.get("/api/jobs").json() if j["id"] == ex["id"])["extract"]
    # it's made again rather than edited, continued or scored
    assert "extract it again" in client.put(f"/api/jobs/{ex['id']}", json={
        "name": "x", "profile_id": pid, "datasets": [{"id": did}], "settings": {}}).json()["detail"]
    db.execute("UPDATE jobs SET status='completed' WHERE id=?", (ex["id"],))
    assert "trains nothing" in client.post(f"/api/jobs/{ex['id']}/action", json={"action": "extend", "steps": 9}).json()["detail"]
    assert "no datasets" in client.post(f"/api/jobs/{ex['id']}/evaluate", json={"settings": {}}).json()["detail"]
    # once made, its LoRA is in the Sandbox with the family's others; the finetune's models aren't
    out = os.path.join(job_dir(ex["id"]), "output")
    os.makedirs(out)
    open(os.path.join(out, "ft_000200_r16.safetensors"), "wb").close()
    with open(os.path.join(job_dir(ex["id"]), "status.json"), "w") as f:
        json.dump({"phase": "completed", "checkpoints": [{"step": 200, "file": "ft_000200_r16.safetensors",
                                                          "final": True}]}, f)
    listed = client.get("/api/sandbox/loras", params={"family": "moss_image"}).json()
    assert [x["file"] for x in listed if x["job_id"] in (ex["id"], ft)] == ["ft_000200_r16.safetensors"]
    # the model it was finetuned from has to be there: the LoRA is the change from it
    cfg = client.get(f"/api/jobs/{ft}/config").json()
    cfg["profile"]["diffusion_model"] = str(tmp_path / "moved.safetensors")
    db.execute("UPDATE jobs SET config=? WHERE id=?", (json.dumps(cfg), ft))
    assert "isn't there any more" in client.post(f"/api/jobs/{ft}/extract", json=body).json()["detail"]

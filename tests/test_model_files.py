"""The model manager: the families' download catalogs, finding model files, the models folder setting, and
downloads (from a local server standing in for Hugging Face)."""

import hashlib
import http.server
import os
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from conftest import header_only_safetensors


@pytest.fixture(scope="module")
def client():
    from dainamo.server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def hf_cache(monkeypatch, tmp_path_factory):
    """An empty Hugging Face cache instead of this machine's, for tests to fill (``hf_repo``)."""
    from dainamo.server import model_files

    cache = tmp_path_factory.mktemp("hub")
    monkeypatch.setattr(model_files, "hf_cache_folder", lambda: str(cache))
    return cache


def hf_repo(cache, repo: str, files: dict, indexed=(), sha="5" * 40):
    """A ``models--owner--repo`` folder as huggingface_hub leaves it: ``files`` ({path in the repo: tensors}) as
    blobs, linked from the snapshot ``refs/main`` names; each folder in ``indexed`` ("" for the top) gets a
    model.safetensors.index.json of its .safetensors files' tensors. -> the repo folder."""
    import json

    root = cache / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / sha
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(sha)
    (root / "blobs").mkdir()
    for i, (rel, tensors) in enumerate(files.items()):
        link = snap / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if tensors is None:  # a small file that isn't a model
            (root / "blobs" / f"b{i}").write_text("{}")
        else:
            header_only_safetensors(root / "blobs" / f"b{i}", tensors)
        link.symlink_to(os.path.relpath(root / "blobs" / f"b{i}", link.parent))
    for folder in indexed:
        weight_map = {k: os.path.basename(rel) for rel, t in files.items()
                      if t and os.path.dirname(rel) == folder and rel.endswith(".safetensors") for k in t}
        (snap / folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return root


def qwen3_4b(prefix="model."):
    """Qwen3-4B's tensors that tell it apart (FLUX.2 Klein 4B's text encoder), in two shards."""
    return ({f"{prefix}embed_tokens.weight": ("BF16", (151936, 2560)),
             f"{prefix}layers.0.self_attn.q_norm.weight": ("BF16", (128,))},
            {f"{prefix}layers.35.mlp.down_proj.weight": ("BF16", (2560, 9728))})


FLUX2_VAE = {"bn.running_mean": ("F32", (128,)), "encoder.conv_in.weight": ("F32", (128, 3, 3, 3))}


def _fields(ov: dict, family: str) -> dict:
    return {f["key"]: f for f in next(x for x in ov["families"] if x["id"] == family)["fields"]}


def test_every_catalog_file_fits_its_family():
    from dainamo.engine.families import FAMILIES
    from dainamo.server.model_files import KIND_FOLDERS

    ids = set()
    for F in FAMILIES.values():
        paths = {f.key for f in F.info.profile_fields if f.kind == "path"} | set(F.extra_fields)
        assert not set(F.extra_fields) & {f.key for f in F.info.profile_fields}, F.info.id
        variants = ["", *(next((f.options for f in F.info.profile_fields if f.key == "variant"), None) or [])]
        for d in F.downloads:
            assert d.field in paths, (F.info.id, d.field)
            assert d.kind_folder in KIND_FOLDERS and d.name.endswith(".safetensors") and d.size > 0
            assert re.fullmatch(r"[0-9a-f]{40}", d.revision) and re.fullmatch(r"([0-9a-f]{64})?", d.sha256)
            assert d.sha256 or d.gated, f"{d.id}: only gated repos hide their checksums"
            assert d.variant in variants, (d.id, d.variant)
            assert d.id not in ids
            ids.add(d.id)
            assert not re.search(r"w4a8|w6a8|gptq|mxfp8", d.name), f"{d.id}: a format Dainamo can't read"
        for field in {d.field for d in F.downloads}:
            for v in {d.variant for d in F.downloads if d.field == field}:
                picks = [d for d in F.downloads if d.field == field and d.variant == v and d.recommended]
                assert len(picks) == 1, (F.info.id, field, v)
        required = {f.key for f in F.info.profile_fields if f.kind == "path" and f.required}
        assert required <= {d.field for d in F.downloads}, f"{F.info.id}: nothing to download for a required file"


def test_models_folder_setting(client, tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "vae").mkdir(parents=True)
    r = client.put("/api/settings", json={"models": {"folder": str(comfy)}})
    assert r.status_code == 200 and r.json()["models"]["folder"] == str(comfy / "models")  # ComfyUI's own folder
    r = client.put("/api/settings", json={"models": {"folder": str(tmp_path / "missing")}})
    assert r.status_code == 400
    assert client.put("/api/settings", json={"models": {"folder": ""}}).json()["models"]["folder"] == ""


def test_files_are_found_by_what_is_in_them(client, tmp_path):
    models = tmp_path / "models"
    (models / "unet").mkdir(parents=True)
    (models / "clip").mkdir()
    (models / "loras").mkdir()
    header_only_safetensors(models / "unet" / "my krea.safetensors",
                            {"blocks.27.attn.wq.weight": ("BF16", (8, 8)), "txtfusion.projector.weight": ("BF16", (8, 8))})
    header_only_safetensors(models / "clip" / "te.safetensors",  # Krea's Qwen3-VL-4B: its size, and a vision tower
                            {"model.layers.35.mlp.down_proj.weight": ("I8", (2560, 9728)),
                             "model.visual.blocks.0.attn.qkv.weight": ("BF16", (8, 8))})
    header_only_safetensors(models / "loras" / "style.safetensors", {"lora_A.weight": ("BF16", (4, 8))})
    (models / "loras" / "notes.txt").write_text("not a model")
    client.put("/api/settings", json={"models": {"folder": str(models)}})
    try:
        ov = client.get("/api/models").json()
        assert ov["folder"] == str(models) and ov["folders"]["loras"] == str(models / "loras")
        krea = {f["key"]: f for f in next(x for x in ov["families"] if x["id"] == "krea2")["fields"]}
        assert [x["where"] for x in krea["diffusion_model"]["found"]] == [os.path.join("unet", "my krea.safetensors")]
        assert [x["name"] for x in krea["text_encoder"]["found"]] == ["te.safetensors"]
        assert krea["training_adapter"]["found"] == []  # a LoRA isn't taken for an adapter by its looks
        assert all(o["have"] is None for o in krea["diffusion_model"]["downloads"])  # not the catalog's file
        qwen = {f["key"]: f for f in next(x for x in ov["families"] if x["id"] == "qwen_image_2")["fields"]}
        assert qwen["text_encoder"]["found"] == []  # the 4B isn't Qwen-Image 2.1's 8B
        suggest = client.get("/api/models/suggest").json()["folders"]
        assert isinstance(suggest, list)
    finally:
        client.put("/api/settings", json={"models": {"folder": ""}})


def test_the_flux3_prompt_upsampler_is_an_optional_row_for_captions(client, tmp_path, monkeypatch):
    """No profile takes it: a row of its own in Model files, never a MOSS Image text encoder though it has the
    tensors of one, and a file the caption provider's text encoder can use."""
    import dataclasses

    from conftest import FAKE_ENCODERS
    from dainamo.engine.families import FAMILIES

    models = tmp_path / "models"
    (models / "text_encoders").mkdir(parents=True)
    moss = {**FAKE_ENCODERS["moss"][0], "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": ("BF16", (1,))}
    name = "qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors"
    pup = header_only_safetensors(models / "text_encoders" / name, moss)
    own = header_only_safetensors(models / "text_encoders" / "qwen3.5_9b_int8_convrot.safetensors", moss)
    F = FAMILIES["flux3_klein"]  # the catalog's file is this one: the same name and size
    monkeypatch.setattr(F, "downloads", [dataclasses.replace(d, size=os.path.getsize(pup)) if d.name == name else d
                                         for d in F.downloads])
    client.put("/api/settings", json={"models": {"folder": str(models)}})
    try:
        ov = client.get("/api/models").json()
        row = _fields(ov, "flux3_klein")["prompt_upsampler"]
        assert (row["label"], row["hint"], row["required"], row["profile"]) == ("Prompt upsampler", "for captions",
                                                                               False, False)
        assert [x["path"] for x in row["found"]] == [pup] and len(row["downloads"]) == 2
        assert all(o["folder"] == "text_encoders" for o in row["downloads"])
        assert "profile" not in _fields(ov, "flux3_klein")["text_encoder"]
        assert [x["path"] for x in _fields(ov, "moss_image")["text_encoder"]["found"]] == [own]
        listed = {e["path"]: e for e in client.get("/api/llm/encoders").json()}
        assert listed[pup]["name"] == "FLUX 3 prompt upsampler" and listed[pup]["profiles"] == []
        assert own not in listed  # no profile uses it, and it isn't an upsampler
    finally:
        client.put("/api/settings", json={"models": {"folder": ""}})


def test_the_hugging_face_cache_is_searched_too(client, hf_cache):
    """A sharded repo is one model, by its repo folder; a diffusers pipeline's text_encoder/ (shards) and vae/ (one
    file) are models of their own; the pipeline's diffusers-layout transformer matches no diffusion model."""
    shard1, shard2 = qwen3_4b()
    qwen = hf_repo(hf_cache, "Qwen/Qwen3-4B", {"model-00001-of-00002.safetensors": shard1,
                                                "model-00002-of-00002.safetensors": shard2,
                                                "config.json": None}, indexed=[""])
    te1, te2 = qwen3_4b()
    pipe = hf_repo(hf_cache, "acme/klein-pipeline", {
        "model_index.json": None,
        "text_encoder/model-00001-of-00002.safetensors": te1, "text_encoder/model-00002-of-00002.safetensors": te2,
        "vae/diffusion_pytorch_model.safetensors": FLUX2_VAE,
        "transformer/diffusion_pytorch_model.safetensors": {"x_embedder.weight": ("BF16", (3072, 128)),
                                                            "context_embedder.weight": ("BF16", (3072, 7680))},
    }, indexed=["text_encoder"])
    snap = pipe / "snapshots" / ("5" * 40)
    (hf_cache / "models--half--done" / "refs").mkdir(parents=True)  # refs/main, but nothing downloaded
    (hf_cache / "models--half--done" / "refs" / "main").write_text("6" * 40)

    ov = client.get("/api/models").json()
    assert ov["hf_cache"] == str(hf_cache) and ov["hf_cache_ok"]
    klein = _fields(ov, "flux2_klein")
    te = {x["name"]: x for x in klein["text_encoder"]["found"]}
    assert set(te) == {"Qwen/Qwen3-4B", "acme/klein-pipeline/text_encoder"}
    assert te["Qwen/Qwen3-4B"]["path"] == str(qwen)  # read at refs/main (comfy_io.shard_files)
    assert te["Qwen/Qwen3-4B"]["size"] == sum(os.path.getsize(qwen / "snapshots" / ("5" * 40) / n)
                                              for n in ("model-00001-of-00002.safetensors",
                                                        "model-00002-of-00002.safetensors"))
    assert te["acme/klein-pipeline/text_encoder"]["path"] == str(snap / "text_encoder")
    assert all(x["source"] == "hf_cache" for x in te.values())
    assert {x["repo"] for x in te.values()} == {"Qwen/Qwen3-4B", "acme/klein-pipeline"}
    [vae] = klein["vae"]["found"]
    assert (vae["name"], vae["path"], vae["repo"]) == ("acme/klein-pipeline/vae/diffusion_pytorch_model.safetensors",
                                                       str(snap / "vae" / "diffusion_pytorch_model.safetensors"),
                                                       "acme/klein-pipeline")
    assert klein["diffusion_model"]["found"] == []  # diffusers' key names aren't Klein's
    assert _fields(ov, "krea2")["text_encoder"]["found"] == []  # Qwen3, not Qwen3-VL

    # what a profile editor's pick puts in the field checks out
    v = client.post("/api/profiles/validate", json={"name": "k", "family": "flux2_klein",
                                                    "data": {"text_encoder": str(qwen), "vae": vae["path"]}}).json()
    assert {c["field"]: c["level"] for c in v if c["field"] in ("text_encoder", "vae")} == {"text_encoder": "ok",
                                                                                           "vae": "ok"}


def test_the_models_folder_copy_comes_first(client, hf_cache, tmp_path):
    """Both places have a Qwen3-4B: the overview says where each is from, and lists the ComfyUI one first (what
    Create profile picks). A file linked into the models folder from the cache is listed once."""
    shard1, shard2 = qwen3_4b()
    hf_repo(hf_cache, "Qwen/Qwen3-4B", {"model-00001-of-00002.safetensors": shard1,
                                        "model-00002-of-00002.safetensors": shard2}, indexed=[""])
    vae_repo = hf_repo(hf_cache, "acme/flux2-vae", {"ae.safetensors": FLUX2_VAE})
    models = tmp_path / "models"
    (models / "text_encoders").mkdir(parents=True)
    (models / "vae").mkdir()
    header_only_safetensors(models / "text_encoders" / "zz_qwen_3_4b.safetensors", {**shard1, **shard2})
    (models / "vae" / "flux2-vae.safetensors").symlink_to(vae_repo / "snapshots" / ("5" * 40) / "ae.safetensors")
    client.put("/api/settings", json={"models": {"folder": str(models)}})
    try:
        ov = client.get("/api/models").json()
        klein = _fields(ov, "flux2_klein")
        assert [(x["name"], x["source"]) for x in klein["text_encoder"]["found"]] == [
            ("zz_qwen_3_4b.safetensors", "models"), ("Qwen/Qwen3-4B", "hf_cache")]
        assert klein["text_encoder"]["found"][0]["repo"] == ""
        assert [(x["path"], x["source"]) for x in klein["vae"]["found"]] == [
            (str(models / "vae" / "flux2-vae.safetensors"), "models")]
    finally:
        client.put("/api/settings", json={"models": {"folder": ""}})


def test_a_cache_without_models_adds_nothing(client, hf_cache):
    """Whisper, a repo whose shards aren't all downloaded, one with refs/main but no snapshot: nothing a family
    takes. The overview is the same as with no cache at all."""
    from dainamo.server import model_files

    def found(ov):
        return {(fam["id"], f["key"]): f["found"] for fam in ov["families"] for f in fam["fields"]}

    before = found(client.get("/api/models").json())
    hf_repo(hf_cache, "openai/whisper-tiny", {"model.safetensors": {"model.encoder.conv1.weight": ("F32", (384, 80, 3))}})
    shard1, shard2 = qwen3_4b()
    partly = hf_repo(hf_cache, "Qwen/Qwen3-4B", {"model-00001-of-00002.safetensors": shard1,
                                                 "model-00002-of-00002.safetensors": shard2}, indexed=[""])
    (partly / "snapshots" / ("5" * 40) / "model-00002-of-00002.safetensors").unlink()  # still downloading
    (hf_cache / "models--gone--away" / "refs").mkdir(parents=True)
    (hf_cache / "models--gone--away" / "refs" / "main").write_text("7" * 40)
    (hf_cache / "datasets--some--data").mkdir()
    ov = client.get("/api/models").json()
    assert found(ov) == before
    assert not any(x["source"] == "hf_cache" for fs in found(ov).values() for x in fs)
    assert [f["name"] for f in model_files.scan() if f["source"] == "hf_cache"] == [
        "openai/whisper-tiny/model.safetensors"]  # looked at, and taken for nothing


def test_a_cache_of_many_repos_lists_quickly_once_scanned(client, hf_cache, monkeypatch):
    """Headers are read once; after that a scan reads only folders, until a download changes one."""
    from dainamo.server import model_files

    for i in range(20):
        a, b = qwen3_4b()
        hf_repo(hf_cache, f"owner/repo-{i}", {"model-00001-of-00002.safetensors": a,
                                              "model-00002-of-00002.safetensors": b,
                                              "vae/ae.safetensors": FLUX2_VAE}, indexed=[""])
    first = model_files.scan()
    assert len([f for f in first if f["source"] == "hf_cache"]) == 40
    read = []
    real = model_files._identify
    monkeypatch.setattr(model_files, "_identify", lambda path: read.append(path) or real(path))
    t = time.monotonic()
    again = model_files.scan()
    assert time.monotonic() - t < 0.5 and not read
    assert [f["path"] for f in again] == [f["path"] for f in first]
    # a new file in one: that folder changed, so it's looked at again
    folder = hf_cache / "models--owner--repo-3" / "snapshots" / ("5" * 40) / "vae"
    header_only_safetensors(hf_cache / "models--owner--repo-3" / "blobs" / "new", FLUX2_VAE)
    (folder / "ae2.safetensors").symlink_to(os.path.relpath(hf_cache / "models--owner--repo-3" / "blobs" / "new", folder))
    st = os.stat(folder)
    os.utime(folder, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # a second later, as a real download would be
    names = [f["name"] for f in model_files.scan()]
    assert "owner/repo-3/vae/ae2.safetensors" in names and read == [str(folder / "ae2.safetensors")]


def test_a_turbo_model_without_its_adapter_is_flagged(client):
    def adapter_check(data):
        v = client.post("/api/profiles/validate", json={"name": "x", "family": "krea2", "data": data}).json()
        return next((c for c in v if c["field"] == "training_adapter"), None)

    turbo = "/nowhere/krea2_turbo_int8_convrot.safetensors"
    assert "training adapter" in adapter_check({"diffusion_model": turbo})["message"]
    assert adapter_check({"diffusion_model": "/nowhere/krea2_raw_int8_convrot.safetensors"}) is None
    from dainamo.engine.families import FAMILIES

    # blank by default (not a Hugging Face path, which would download into its hidden cache)
    assert next(f for f in FAMILIES["krea2"].info.profile_fields if f.key == "training_adapter").default == ""


# ---------------------------------------------------------------------------------------------- downloads ---
class _Hub(http.server.BaseHTTPRequestHandler):
    """Serves ``files`` ({url path: bytes}) with Range requests, like Hugging Face's resolve URLs; "/gated/" paths
    answer 401. ``slow`` paths dribble out, so a test can pause them."""

    files: dict[str, bytes] = {}
    slow: set[str] = set()

    def log_message(self, *a):
        pass

    def do_GET(self):
        if "/gated/" in self.path:
            self.send_response(401)
            self.end_headers()
            return
        data = self.files.get(self.path)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        m = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range", ""))
        if m and int(m.group(1)) >= len(data):  # as the CDN answers a range from the end of the file
            self.send_response(416)
            self.end_headers()
            return
        if m:
            start = int(m.group(1))
            self.send_response(206)
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            for i in range(0, len(body), 4096):
                self.wfile.write(body[i:i + 4096])
                if self.path in self.slow:
                    time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A stand-in Hugging Face with one model file, listed in Krea 2's catalog: (Download, its bytes)."""
    import torch
    from safetensors.torch import save_file

    from dainamo.engine.families import FAMILIES
    from dainamo.engine.families.base import Download
    from dainamo.server import model_files

    src = tmp_path / "src.safetensors"
    save_file({"w": torch.arange(60000, dtype=torch.float32)}, str(src))
    data = src.read_bytes()
    d = Download("training_adapter", "test/repo", "adapter.safetensors", len(data), "test", "0" * 40,
                 sha256=hashlib.sha256(data).hexdigest(), folder="loras")
    gated = Download("training_adapter", "test/gated", "x.safetensors", 10, "gated", "0" * 40, gated=True,
                     folder="loras")
    monkeypatch.setattr(FAMILIES["krea2"], "downloads", [d, gated])
    _Hub.files = {f"/test/repo/resolve/{'0' * 40}/adapter.safetensors": data}
    _Hub.slow = set(_Hub.files)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(model_files, "HF_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(model_files, "hf_token", lambda: None)
    monkeypatch.setattr(model_files, "CHUNK", 4096)  # progress as it goes, for a file this small
    yield d, data
    server.shutdown()


def _wait(client, tid, until=lambda t: t["state"] not in ("queued", "checking", "downloading", "verifying")):
    for _ in range(600):
        t = next(t for t in client.get("/api/models/downloads").json() if t["id"] == tid)
        if until(t):
            return t
        time.sleep(0.02)
    raise AssertionError(f"download stuck: {t}")


def test_download_pauses_resumes_and_checks_the_file(client, hub, tmp_path):
    d, data = hub
    folder = tmp_path / "models" / "loras"
    t = client.post("/api/models/downloads", json={"id": d.id, "folder": str(folder)}).json()
    assert t["dest"] == str(folder / d.name)
    t = _wait(client, t["id"], lambda t: t["done"] > 20000)
    v = client.post("/api/profiles/validate", json={"name": "x", "family": "krea2",
                                                    "data": {"training_adapter": t["dest"]}}).json()
    check = next(c for c in v if c["field"] == "training_adapter")
    assert check["level"] == "warning" and check["message"].startswith(("Downloading", "Waiting"))
    client.post(f"/api/models/downloads/{t['id']}/cancel")
    t = _wait(client, t["id"])
    assert t["state"] == "cancelled" and 0 < os.path.getsize(t["dest"] + ".part") < len(data)
    assert not os.path.exists(t["dest"])  # a part never looks finished
    _Hub.slow = set()
    client.post(f"/api/models/downloads/{t['id']}/resume")
    t = _wait(client, t["id"])
    assert t["state"] == "done", t["error"]
    assert (folder / d.name).read_bytes() == data and not os.path.exists(t["dest"] + ".part")
    r = client.post("/api/models/downloads", json={"id": d.id, "folder": str(folder)})
    assert r.status_code == 400 and "already there" in r.json()["detail"]
    assert client.delete(f"/api/models/downloads/{t['id']}").status_code == 200
    assert (folder / d.name).exists()  # clearing a finished download keeps its file


def test_a_download_whose_part_is_complete_is_checked_not_fetched(client, hub, tmp_path):
    """Stopped while it was being checked, a part is the whole file: it's verified and kept (a request for the
    range from its end was refused every time, so Resume never got further)."""
    d, data = hub
    _Hub.slow = set()
    folder = tmp_path / "models" / "loras"
    folder.mkdir(parents=True)
    (folder / (d.name + ".part")).write_bytes(data)
    t = client.post("/api/models/downloads", json={"id": d.id, "folder": str(folder)}).json()
    t = _wait(client, t["id"])
    assert t["state"] == "done", t["error"]
    assert (folder / d.name).read_bytes() == data
    client.delete(f"/api/models/downloads/{t['id']}")


def test_a_bad_download_is_thrown_away(client, hub, tmp_path, monkeypatch):
    from dainamo.engine.families import FAMILIES

    d, data = hub
    _Hub.slow = set()
    wrong = type(d)(**{**d.__dict__, "sha256": "f" * 64})
    monkeypatch.setattr(FAMILIES["krea2"], "downloads", [wrong])
    t = client.post("/api/models/downloads", json={"id": d.id, "folder": str(tmp_path)}).json()
    t = _wait(client, t["id"])
    assert t["state"] == "failed" and "checksum" in t["error"]
    assert not os.path.exists(t["dest"]) and not os.path.exists(t["dest"] + ".part")
    client.delete(f"/api/models/downloads/{t['id']}")


def test_gated_and_impossible_downloads_say_why(client, hub, tmp_path, monkeypatch):
    from dainamo.server import model_files

    t = client.post("/api/models/downloads", json={"id": "test/gated/x.safetensors", "folder": str(tmp_path)}).json()
    t = _wait(client, t["id"])
    assert t["state"] == "failed" and "hf auth login" in t["error"] and "huggingface.co/test/gated" in t["error"]
    client.delete(f"/api/models/downloads/{t['id']}")
    d, _ = hub
    assert client.post("/api/models/downloads", json={"id": d.id, "folder": ""}).status_code == 400
    assert client.post("/api/models/downloads", json={"id": "no/such/file", "folder": str(tmp_path)}).status_code == 400
    monkeypatch.setattr(model_files, "SPARE_BYTES", 10**18)
    r = client.post("/api/models/downloads", json={"id": d.id, "folder": str(tmp_path)})
    assert r.status_code == 400 and "Not enough space" in r.json()["detail"]


# ---------------------------------------------------------------------------------------------- upscalers ---
def test_the_upscaler_downloads_go_to_upscale_models():
    from dainamo.engine import upscale
    from dainamo.server.model_files import UPSCALER_DOWNLOADS, find_download

    assert {d.name for d in UPSCALER_DOWNLOADS} == {"RealESRGAN_x4plus.safetensors", "seedvr2_3b_fp16.safetensors",
                                                    "seedvr2_ema_vae_fp16.safetensors"}  # as ComfyUI names them
    for d in UPSCALER_DOWNLOADS:
        assert d.kind_folder == upscale.FOLDER and d.name.endswith(".safetensors") and d.size > 0
        assert re.fullmatch(r"[0-9a-f]{40}", d.revision) and re.fullmatch(r"[0-9a-f]{64}", d.sha256)
        assert find_download(d.id) == ("upscalers", "Upscalers", d)
    assert sum(d.recommended for d in UPSCALER_DOWNLOADS) == 1


def test_the_upscalers_card(client, tmp_path, monkeypatch):
    """Model files' Upscalers card: the files in upscale_models (the catalog's known by name and size) and what each
    is, and the downloads, into that folder."""
    from dainamo.engine import upscale
    from dainamo.server.model_files import UPSCALER_DOWNLOADS

    models = tmp_path / "models"
    folder = models / "upscale_models"
    folder.mkdir(parents=True)
    esrgan = next(d for d in UPSCALER_DOWNLOADS if d.name.startswith("RealESRGAN"))
    with open(folder / esrgan.name, "wb") as f:  # the catalog's file: its name and size (sparse)
        f.truncate(esrgan.size)
    header_only_safetensors(folder / "4x-UltraSharp.safetensors", {"w": ("BF16", (4,))})
    (folder / "put_esrgan_and_other_upscale_models_here").write_text("")
    infos = [upscale.Info(str(folder / "4x-UltraSharp.safetensors"), "pixel", "4x-UltraSharp", scale=4),
             upscale.Info(str(folder / esrgan.name), "pixel", "RealESRGAN_x4plus", scale=4)]
    monkeypatch.setattr(upscale, "list_models", lambda base: infos)
    client.put("/api/settings", json={"models": {"folder": str(models)}})
    try:
        ov = client.get("/api/models").json()
        assert ov["folders"]["upscale_models"] == str(folder)
        up = ov["upscalers"]
        assert (up["id"], up["label"], up["folder"], up["folder_ok"]) == ("upscalers", "Upscalers", str(folder), True)
        assert [(m["name"], m["kind"], m["scale"]) for m in up["models"]] == [("4x-UltraSharp", "pixel", 4),
                                                                               ("RealESRGAN_x4plus", "pixel", 4)]
        [field] = up["fields"]
        have = {o["name"]: o["have"] for o in field["downloads"]}
        assert have == {esrgan.name: str(folder / esrgan.name), "seedvr2_3b_fp16.safetensors": None,
                        "seedvr2_ema_vae_fp16.safetensors": None}
        assert all(o["folder"] == "upscale_models" for o in field["downloads"])
        found = {f["name"]: f for f in field["found"]}
        assert set(found) == {esrgan.name, "4x-UltraSharp.safetensors"}  # model files only
        assert found[esrgan.name]["download"] == esrgan.id and found[esrgan.name]["where"] == \
            os.path.join("upscale_models", esrgan.name)
    finally:
        client.put("/api/settings", json={"models": {"folder": ""}})
    assert client.get("/api/models").json()["upscalers"]["models"] == []


def test_an_upscaler_downloads_like_a_model_file(client, hub, tmp_path, monkeypatch):
    from dainamo.engine.families.base import Download
    from dainamo.server import model_files

    _, data = hub
    d = Download("upscaler", "test/upscalers", "up.safetensors", len(data), "test", "0" * 40,
                 sha256=hashlib.sha256(data).hexdigest(), folder="upscale_models")
    monkeypatch.setattr(model_files, "UPSCALER_DOWNLOADS", [d])
    _Hub.files[f"/test/upscalers/resolve/{'0' * 40}/up.safetensors"] = data
    _Hub.slow = set()
    folder = tmp_path / "models" / "upscale_models"
    t = client.post("/api/models/downloads", json={"id": d.id, "folder": str(folder)}).json()
    assert (t["family"], t["family_label"], t["dest"]) == ("upscalers", "Upscalers", str(folder / "up.safetensors"))
    t = _wait(client, t["id"])
    assert t["state"] == "done", t["error"]
    assert (folder / "up.safetensors").read_bytes() == data
    client.delete(f"/api/models/downloads/{t['id']}")

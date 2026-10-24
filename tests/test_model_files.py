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


def test_every_catalog_file_fits_its_family():
    from dainamo.engine.families import FAMILIES
    from dainamo.server.model_files import KIND_FOLDERS

    ids = set()
    for F in FAMILIES.values():
        paths = {f.key for f in F.info.profile_fields if f.kind == "path"}
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

"""API checks that don't need a GPU: settings merging, job validation, path sandboxing, host check."""

import os
import sys
import tempfile
from types import SimpleNamespace

os.environ["DAINAMO_DATA"] = tempfile.mkdtemp(prefix="dainamo-test-")
os.environ["DAINAMO_ALLOWED_HOSTS"] = "127.0.0.1,localhost,testserver"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from dainamo.server.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_settings_partial_update_keeps_other_fields(client):
    client.put("/api/settings", json={"captioner": {"model": "my-vl"}})
    r = client.put("/api/settings", json={"captioner": {"endpoint": "http://x:1/v1"}}).json()
    assert r["captioner"]["model"] == "my-vl" and r["captioner"]["endpoint"] == "http://x:1/v1"
    assert r["captioner"]["max_tokens"] == 600  # default still present


def test_queue_settings_validated(client):
    assert client.put("/api/settings", json={"queue": {"mode": "bogus"}}).status_code == 400
    r = client.put("/api/settings", json={"queue": {"mode": "parallel_multi", "gpus": [1, 0, 1], "max_per_gpu": 0}}).json()
    assert r["queue"] == {"mode": "parallel_multi", "gpus": [0, 1], "max_per_gpu": 1}
    client.put("/api/settings", json={"queue": {"mode": "sequential", "gpus": [0]}})


def test_job_settings_validated(client, tmp_path):
    pid = client.post("/api/profiles", json={"name": "p", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "d", "path": str(tmp_path)}).json()["id"]
    base = {"name": "j", "profile_id": pid, "datasets": [{"id": did}], "gpu": "auto"}
    for bad in ({"rank": 0}, {"steps": -5}, {"learning_rate": "abc"}, {"optimizer": "sgd"}, {"offload": 101},
                {"offload": -1}):
        r = client.post("/api/jobs", json={**base, "settings": bad})
        assert r.status_code == 400, bad
    assert client.post("/api/jobs", json={**base, "gpu": "1; rm -rf", "settings": {}}).status_code == 400
    # don't let the scheduler actually launch the valid one
    client.put("/api/settings", json={"queue": {"gpus": [99]}})


def test_estimate_reports_unreadable_models(client):
    pid = client.post("/api/profiles", json={"name": "e", "family": "krea2",
                                             "data": {"diffusion_model": "/nope.safetensors"}}).json()["id"]
    r = client.post("/api/estimate", json={"profile_id": pid, "settings": {}}).json()
    assert "wasn't found" in r["error"]


def test_media_outside_datasets_forbidden(client):
    assert client.get("/api/media", params={"path": "/etc/passwd"}).status_code == 403
    assert client.get("/api/thumb", params={"path": "/etc/hostname"}).status_code == 403


def test_foreign_host_rejected(client):
    assert client.get("/api/settings", headers={"host": "evil.example.com"}).status_code == 400
    assert client.get("/api/settings", headers={"host": "127.0.0.1:7865"}).status_code == 200


def test_sample_prompt_cards_validated(client, tmp_path):
    from dainamo.server.validation import check_sample_prompts
    from fastapi import HTTPException

    cards = check_sample_prompts(
        ["legacy string prompt", {"prompt": "  "}, {"prompt": "card", "width": "768", "height": 512, "seed": "",
                                                   "lora_scale": "0.5"}], 16)
    assert cards == [{"prompt": "legacy string prompt"},
                     {"prompt": "card", "width": 768, "height": 512, "lora_scale": 0.5}]
    for bad in ({"prompt": "x", "width": "wide"}, {"prompt": "x", "lora_scale": 99}, {"prompt": "x", "height": 8}, 5):
        with pytest.raises(HTTPException):
            check_sample_prompts([bad], 16)


def test_trainer_fills_card_defaults():
    from dainamo.engine.trainer import Trainer

    t = Trainer.__new__(Trainer)  # no GPU or job dir needed for this
    t.driver = type("D", (), {"bucket_divisibility": 16, "supports_video": False, "is_audio": False, "snap_frames": None,
                              "info": SimpleNamespace(lora_strength=1.0)})()
    t.s = {"sample_width": 1024, "sample_height": 768, "sample_seed": 40,
           "sample_prompts": ["first", {"prompt": "second", "width": 500, "lora_scale": 0.0},
                              {"prompt": "third", "seed": 7}]}
    assert t.sample_specs() == [
        {"prompt": "first", "width": 1024, "height": 768, "seed": 40, "lora_scale": 1.0, "frames": 1, "images": []},
        {"prompt": "second", "width": 496, "height": 768, "seed": 41, "lora_scale": 0.0, "frames": 1,
         "images": []},  # rounded to /16
        {"prompt": "third", "width": 1024, "height": 768, "seed": 7, "lora_scale": 1.0, "frames": 1, "images": []},
    ]


def test_video_card_fields_validated(tmp_path):
    from dainamo.server.validation import check_sample_prompts, check_settings
    from fastapi import HTTPException

    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    cards = check_sample_prompts([{"prompt": "v", "frames": "39", "images": [str(img), " "]},
                                   {"prompt": "w", "images": str(img)}], 32)
    assert cards == [{"prompt": "v", "frames": 39, "images": [str(img)]}, {"prompt": "w", "images": [str(img)]}]
    for bad in ({"prompt": "x", "frames": 0}, {"prompt": "x", "frames": 999},
                {"prompt": "x", "images": [str(tmp_path / "missing.png")]}, {"prompt": "x", "images": {"a": 1}}):
        with pytest.raises(HTTPException):
            check_sample_prompts([bad], 32)
    base = {"steps": 10, "rank": 8, "alpha": 8, "learning_rate": 1e-4, "batch_size": 1, "resolution": 384}
    s = check_settings({**base, "num_frames": "39", "i2v_probability": "0.25", "train_audio": 0}, 32)
    assert (s["num_frames"], s["i2v_probability"], s["train_audio"]) == (39, 0.25, False)
    for bad in ({"num_frames": 0}, {"num_frames": 500}, {"i2v_probability": 1.5}, {"audio_loss_weight": -1}):
        with pytest.raises(HTTPException):
            check_settings({**base, **bad}, 32)


def test_video_specs_snap_frames_and_split_images():
    from dainamo.engine.families.minimax_h3 import packing as P
    from dainamo.engine.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.driver = type("D", (), {"bucket_divisibility": 32, "supports_video": True, "is_audio": False,
                              "info": SimpleNamespace(lora_strength=1.0),
                              "snap_frames": staticmethod(P.align_num_frames_down)})()
    t.s = {"sample_width": 640, "sample_height": 384, "sample_frames": 40,
           "sample_prompts": [{"prompt": "a"}, {"prompt": "b", "frames": 1, "images": "/x.png, /y.png"}]}
    specs = t.sample_specs()
    assert [s["frames"] for s in specs] == [39, 1]
    assert specs[1]["images"] == ["/x.png", "/y.png"]


def _header_only(path, tensors: dict) -> str:
    """A .safetensors file with these {key: (dtype, shape)} and a sparse (unwritten) data section."""
    import json
    import struct

    header, off = {}, 0
    size = {"BF16": 2, "I8": 1, "U8": 1, "F32": 4}
    for k, (dt, shape) in tensors.items():
        n = size[dt]
        for s in shape:
            n *= s
        header[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        f.truncate(8 + len(blob) + off)
    return str(path)


def test_an_unknown_model_family_is_refused(client, tmp_path):
    said = "Unknown model family 'ltx2'"
    for r in (client.post("/api/profiles", json={"name": "bad", "family": "ltx2", "data": {}}),
              client.post("/api/profiles/validate", json={"name": "bad", "family": "ltx2", "data": {}})):
        assert r.status_code == 400 and r.json()["detail"] == said
    assert not any(p["name"] == "bad" for p in client.get("/api/profiles").json())
    pid = client.post("/api/profiles", json={"name": "fam", "family": "ltx2_5", "data": {}}).json()["id"]
    try:
        r = client.put(f"/api/profiles/{pid}", json={"name": "fam", "family": "ltx2", "data": {}})
        assert r.status_code == 400 and r.json()["detail"] == said
        assert next(p for p in client.get("/api/profiles").json() if p["id"] == pid)["family"] == "ltx2_5"  # unchanged
    finally:
        client.delete(f"/api/profiles/{pid}")
    did = client.post("/api/datasets", json={"name": "fam", "path": str(tmp_path)}).json()["id"]
    r = client.get(f"/api/datasets/{did}/check", params={"family": "ltx2"})
    assert r.status_code == 400 and r.json()["detail"] == said


def test_profile_check_tells_text_encoders_and_vaes_apart(tmp_path):
    from dainamo.engine.families import get_family

    krea, qwen = get_family("krea2"), get_family("qwen_image_2")
    key = "model.layers.35.mlp.down_proj.weight"
    vis = {"model.visual.blocks.0.attn.qkv.weight": ("BF16", (8, 8))}  # Qwen3-VL's vision tower (plain Qwen3 has none)
    te4 = _header_only(tmp_path / "te4.safetensors", {key: ("I8", (2560, 9728)), **vis})
    te4_nvfp4 = _header_only(tmp_path / "te4fp4.safetensors", {key: ("U8", (2560, 4864)), **vis})  # two codes per byte
    te8 = _header_only(tmp_path / "te8.safetensors", {key: ("I8", (4096, 12288))})
    te4_plain = _header_only(tmp_path / "te4plain.safetensors", {key: ("I8", (2560, 9728))})  # Qwen3-4B: the same size
    rgb = _header_only(tmp_path / "rgb.safetensors", {"decoder.head.2.weight": ("BF16", (3, 96, 3, 3, 3))})

    def check(F, field, path):
        return next(r for r in F.validate_profile({field: path}) if r["field"] == field)
    assert check(krea, "text_encoder", te4)["level"] == "ok"
    assert check(krea, "text_encoder", te4_nvfp4)["level"] == "ok"
    bad = check(krea, "text_encoder", te8)
    assert bad["level"] == "error" and "Qwen3-VL-4B" in bad["message"]
    assert check(krea, "text_encoder", te4_plain)["level"] == "error"  # FLUX.2 Klein's Qwen3-4B isn't Qwen3-VL-4B
    assert check(get_family("flux3_klein"), "text_encoder", te4_plain)["level"] == "error"
    assert check(get_family("flux3_klein"), "text_encoder", te4)["level"] == "ok"
    assert check(qwen, "text_encoder", te8)["level"] == "error"  # no vision tower in this file
    assert check(krea, "vae", rgb)["level"] == "ok"
    rgba = _header_only(tmp_path / "rgba.safetensors", {"decoder.head.2.weight": ("BF16", (4, 144, 1, 3, 3))})
    assert "Qwen-Image VAE" in check(krea, "vae", rgba)["message"]
    from dainamo.engine.families import FAMILIES
    assert all(f.hint for F in FAMILIES.values() for f in F.info.profile_fields if f.kind == "path")


def test_a_hugging_face_folder_passes_the_profile_check(tmp_path):
    """Model files offers a Hugging Face cache folder of shards (here Qwen3-VL-4B-Instruct, in transformers' key
    layout) for Krea 2's text encoder: the profile check reads every shard, of the cache folder at refs/main too."""
    import json

    from dainamo.engine.families import get_family

    repo = tmp_path / "models--Qwen--Qwen3-VL-4B-Instruct"
    snap = repo / "snapshots" / ("1" * 40)
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("1" * 40)
    shards = {"model-00001-of-00002.safetensors": {"model.language_model.layers.35.mlp.down_proj.weight":
                                                   ("BF16", (2560, 9728))},
              "model-00002-of-00002.safetensors": {"model.visual.blocks.0.attn.qkv.weight": ("BF16", (8, 8))}}
    for name, tensors in shards.items():
        _header_only(snap / name, tensors)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        k: name for name, tensors in shards.items() for k in tensors}}))
    for path in (repo, snap, snap / "model.safetensors.index.json"):
        r = next(r for r in get_family("krea2").validate_profile({"text_encoder": str(path)})
                 if r["field"] == "text_encoder")
        assert r["level"] == "ok" and r["message"].startswith("2 tensors"), (path, r)


def test_log_chunks_resume_where_they_left_off(client, tmp_path):
    import os

    from dainamo.server import db
    from dainamo.server.routes.jobs import LOG_CHUNK, decode_complete
    from dainamo.server.queue import job_dir

    assert decode_complete("ab█".encode()) == ("ab█", 5)
    assert decode_complete("ab█".encode()[:-1]) == ("ab", 2)  # the cut-off character waits
    jid = db.new_id()
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
               (jid, "log job", "completed", "krea2", "auto", db.now(), db.now(), "{}"))
    os.makedirs(job_dir(jid), exist_ok=True)
    path = os.path.join(job_dir(jid), "log.txt")
    with open(path, "wb") as f:
        f.write(b"x" * (LOG_CHUNK + 100) + b"\nfirst line\n\rstep 1/2 \xe2\x96")  # ends mid-character
    c = client.get(f"/api/jobs/{jid}/log/chunk?offset=-1").json()
    assert c["reset"] and c["start"] > 0 and c["text"] == "first line\n\rstep 1/2 "  # starts at a line
    with open(path, "ab") as f:
        f.write(b"\x88\rstep 2/2 \xe2\x96\x88\nDone\n")
    d = client.get(f"/api/jobs/{jid}/log/chunk?offset={c['offset']}&file={c['file']}").json()
    assert not d["reset"] and d["text"] == "█\rstep 2/2 █\nDone\n" and not d["more"]
    stale = client.get(f"/api/jobs/{jid}/log/chunk?offset={d['offset']}&file=other").json()
    assert stale["reset"]  # a replaced log starts over
    r = client.get(f"/api/jobs/{jid}/log.txt")
    assert r.status_code == 200 and r.content.endswith(b"Done\n") and "log_job.log.txt" in r.headers["content-disposition"]


def test_resolution_takes_one_or_several(client, tmp_path):
    from dainamo.server.validation import check_resolution
    from fastapi import HTTPException

    assert check_resolution(1024) == 1024 and check_resolution([1024]) == 1024
    assert check_resolution("1024, 512 512") == [512, 1024] and check_resolution([768, 512]) == [512, 768]
    for bad in ([], "abc", [64], [512, 640, 768, 896, 1024, 1280, 1536], None):
        with pytest.raises(HTTPException):
            check_resolution(bad)
    client.put("/api/settings", json={"queue": {"gpus": [99]}})
    pid = client.post("/api/profiles", json={"name": "mr", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "mr", "path": str(tmp_path)}).json()["id"]
    r = client.post("/api/jobs", json={"name": "mr", "profile_id": pid, "datasets": [{"id": did}],
                                       "settings": {"resolution": [1024, 512]}})
    assert r.status_code == 200 and r.json()["config"]["settings"]["resolution"] == [512, 1024]
    client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})


def test_job_folders_are_named_after_the_job(client, tmp_path):
    from dainamo.server import db
    from dainamo.server.queue import JOBS_DIR, job_dir

    client.put("/api/settings", json={"queue": {"gpus": [99]}})
    pid = client.post("/api/profiles", json={"name": "jf", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "jf", "path": str(tmp_path)}).json()["id"]
    jobs = [client.post("/api/jobs", json={"name": name, "profile_id": pid, "datasets": [{"id": did}],
                                           "settings": {}}).json() for name in ("My LoRA: v1/2", "My LoRA: v1/2", "日本")]
    assert [os.path.basename(job_dir(j["id"])) for j in jobs] == \
        [f"My_LoRA_v1_2_{jobs[0]['id']}", f"My_LoRA_v1_2_{jobs[1]['id']}", jobs[2]["id"]]  # same name, own folder
    assert client.get(f"/api/jobs/{jobs[0]['id']}").json()["dir"] == job_dir(jobs[0]["id"])
    # renaming a job leaves its files where they are
    edit = {"name": "renamed", "profile_id": pid, "datasets": [{"id": did}], "settings": {}}
    assert client.put(f"/api/jobs/{jobs[0]['id']}", json=edit).status_code == 200
    assert os.path.basename(job_dir(jobs[0]["id"])).startswith("My_LoRA_v1_2_")
    # a job from before folders were named keeps its id as its folder
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
               ("oldjob123456", "old", "completed", "krea2", "auto", 0, 0, "{}"))
    assert job_dir("oldjob123456") == os.path.join(JOBS_DIR, "oldjob123456")
    for j in jobs:
        client.post(f"/api/jobs/{j['id']}/action", json={"action": "stop"})


def test_an_auto_clip_length_trains_each_clip_whole(client, tmp_path):
    client.put("/api/settings", json={"queue": {"gpus": [99]}})
    pid = client.post("/api/profiles", json={"name": "ac", "family": "minimax_h3", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "ac", "path": str(tmp_path)}).json()["id"]
    r = client.post("/api/jobs", json={"name": "ac", "profile_id": pid, "datasets": [{"id": did}],
                                       "settings": {"auto_frames": True, "num_frames": 39}})
    s = r.json()["config"]["settings"]
    # the trainer then takes each clip at its own length, up to the longest a clip trains at
    assert r.status_code == 200 and (s["auto_frames"], s["num_frames"]) == (True, 241)
    client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})
    check = client.get(f"/api/datasets/{did}/check?family=minimax_h3&auto=true").json()
    assert check["longest_clip"] == 0  # no clips


def test_each_dataset_has_its_own_resolutions(client, tmp_path):
    from PIL import Image

    (tmp_path / "stills").mkdir()
    (tmp_path / "clips").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "stills" / "a.png")
    (tmp_path / "clips" / "b.mp4").write_bytes(b"")
    client.put("/api/settings", json={"queue": {"gpus": [99]}})
    pid = client.post("/api/profiles", json={"name": "pd", "family": "krea2", "data": {}}).json()["id"]
    stills = client.post("/api/datasets", json={"name": "stills", "path": str(tmp_path / "stills")}).json()["id"]
    clips = client.post("/api/datasets", json={"name": "clips", "path": str(tmp_path / "clips")}).json()["id"]
    listed = {d["name"]: d for d in client.get("/api/datasets").json()}
    assert (listed["stills"]["images"], listed["stills"]["videos"]) == (1, 0)
    assert (listed["clips"]["images"], listed["clips"]["videos"]) == (0, 1)

    r = client.post("/api/jobs", json={"name": "pd", "profile_id": pid, "settings": {"resolution": 768, "flip": True,
                                                                                      "caption_dropout": 0.1},
                                       "datasets": [{"id": stills, "repeats": 2, "resolution": [1024, 512], "flip": False,
                                                     "caption_dropout": 0.3},
                                                    {"id": clips, "resolution": "384"}, {"id": stills}]})
    assert r.status_code == 200
    cfg = r.json()["config"]
    # what a dataset leaves out is the job's
    assert [(d["name"], d["repeats"], d["resolution"], d["flip"], d["caption_dropout"]) for d in cfg["datasets"]] == [
        ("stills", 2, [512, 1024], False, 0.3), ("clips", 1, 384, True, 0.1), ("stills", 1, 768, True, 0.1)]
    assert cfg["settings"]["resolution"] == [384, 512, 768, 1024]  # the job's: all of them together
    client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})
    bad = client.post("/api/jobs", json={"name": "pd", "profile_id": pid, "settings": {},
                                         "datasets": [{"id": clips, "resolution": [64]}]})
    assert bad.status_code == 400 and bad.json()["detail"].startswith("clips: ")
    bad = client.post("/api/jobs", json={"name": "pd", "profile_id": pid, "settings": {},
                                         "datasets": [{"id": clips, "caption_dropout": 1.5}]})
    assert bad.status_code == 400 and "dropout" in bad.json()["detail"]


def test_slider_jobs_need_their_prompts_and_no_dataset(client):
    client.put("/api/settings", json={"queue": {"gpus": [99]}})
    pid = client.post("/api/profiles", json={"name": "sl", "family": "krea2", "data": {}}).json()["id"]
    prompts = {"slider_enabled": True, "slider_target": " a photo of a person ", "slider_positive": "a person, smiling",
               "slider_negative": "a person, frowning"}
    settings = {**prompts, "resolution": 768, "slider_sample_strengths": "-1.5, 0 1.5"}
    r = client.post("/api/jobs", json={"name": "sl", "profile_id": pid, "datasets": [], "settings": settings})
    assert r.status_code == 200
    s = r.json()["config"]["settings"]
    assert (s["slider_target"], s["slider_strength"], s["slider_images"], s["slider_sample_strengths"], s["resolution"]) == \
        ("a photo of a person", 3.0, 16, [-1.5, 0.0, 1.5], 768)
    assert next(j for j in client.get("/api/jobs").json() if j["id"] == r.json()["id"])["slider"] is True
    assert (s["slider_preserve"], s["slider_keep_layout"], s["slider_layout_sigma"]) == ([], False, 0.9)
    client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})
    r = client.post("/api/jobs", json={"name": "sl", "profile_id": pid, "datasets": [], "settings": {
        **prompts, "slider_preserve": [" in a park,", "", "indoors", "in a park"], "slider_keep_layout": 1,
        "slider_layout_sigma": "0.8"}})
    s = r.json()["config"]["settings"]
    assert (s["slider_preserve"], s["slider_keep_layout"], s["slider_layout_sigma"]) == (["in a park", "indoors"], True, 0.8)
    client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})

    def bad(settings, profile=pid):
        r = client.post("/api/jobs", json={"name": "sl", "profile_id": profile, "datasets": [], "settings": settings})
        assert r.status_code == 400, settings
        return r.json()["detail"]

    assert "dataset" in bad({})  # a LoRA still needs one
    assert "negative prompt" in bad({**prompts, "slider_negative": " "})
    assert "differ" in bad({**prompts, "slider_negative": prompts["slider_positive"]})
    assert "slider_strength" in bad({**prompts, "slider_strength": 0})
    assert "strengths" in bad({**prompts, "slider_sample_strengths": "1 2 3 4 5 6 7"})
    assert "picker" in bad({**prompts, "picker_enabled": True})
    assert "slider_layout_sigma" in bad({**prompts, "slider_layout_sigma": 1.5})
    assert "attributes" in bad({**prompts, "slider_preserve": [f"thing {i}" for i in range(13)]})
    ltx = client.post("/api/profiles", json={"name": "lt", "family": "ltx2_5", "data": {}}).json()["id"]
    assert "can't train sliders" in bad(prompts, ltx)


def test_lokr_settings_are_checked(client, tmp_path):
    from dainamo.server.validation import LOKR_FACTOR

    client.put("/api/settings", json={"queue": {"gpus": [99]}})
    pid = client.post("/api/profiles", json={"name": "lk", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "lk", "path": str(tmp_path)}).json()["id"]

    def post(settings, datasets=({"id": did},)):
        return client.post("/api/jobs", json={"name": "lk", "profile_id": pid, "datasets": list(datasets),
                                              "settings": settings})

    for extra, factor in (({}, LOKR_FACTOR), ({"lokr_factor": "-1"}, -1), ({"lokr_factor": 8}, 8)):
        r = post({"network_type": "lokr", **extra})
        assert r.status_code == 200, r.text
        s = r.json()["config"]["settings"]
        assert (s["network_type"], s["lokr_factor"]) == ("lokr", factor)
        client.post(f"/api/jobs/{r.json()['id']}/action", json={"action": "stop"})
    for settings, why in (({"network_type": "loha"}, "network type"),
                          ({"network_type": "lokr", "lokr_factor": 1}, "lokr_factor"),
                          ({"network_type": "lokr", "lokr_factor": "big"}, "lokr_factor")):
        r = post(settings)
        assert r.status_code == 400 and why in r.json()["detail"], settings
    # a slider is always a LoRA
    slider = {"slider_enabled": True, "slider_target": "a person", "slider_positive": "a person, smiling",
              "slider_negative": "a person, frowning"}
    r = post({**slider, "network_type": "lokr"}, datasets=())
    assert r.status_code == 400 and r.json()["detail"] == "A slider trains as a LoRA, not a LoKr"


def test_sign_in_guards_the_api():
    from dainamo.server import auth

    auth.enable("hunter2")
    try:
        c = TestClient(app)
        assert c.get("/api/settings").status_code == 401
        assert c.get("/api/auth").json() == {"required": True, "signed_in": False}
        assert c.get("/").status_code in (200, 503)  # the UI itself loads (503: not built)
        assert c.post("/api/auth/login", json={"password": "hunter3"}).status_code == 401
        assert c.post("/api/auth/login", json={"password": "hunter2"}).status_code == 200
        assert c.get("/api/settings").status_code == 200 and c.get("/api/auth").json()["signed_in"]
        assert TestClient(app).get("/api/jobs", headers={"Authorization": "Bearer hunter2"}).status_code == 200
        link = TestClient(app)
        r = link.get("/sandbox?key=hunter2&x=1", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/sandbox?x=1"  # the key leaves the address bar
        assert link.get("/api/settings").status_code == 200
        auth.enable("changed")  # a new password signs every browser out
        assert c.get("/api/settings").status_code == 401 and link.get("/api/settings").status_code == 401
        c.post("/api/auth/login", json={"password": "changed"})
        c.post("/api/auth/logout")
        assert c.get("/api/settings").status_code == 401
    finally:
        auth.enable(None)
    assert TestClient(app).get("/api/settings").status_code == 200


@pytest.fixture
def hook():
    """A local webhook receiver: (url, requests received); paths containing "fail" answer 404."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    got = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            got.append({"path": self.path, "body": body, "type": self.headers.get("content-type", "")})
            self.send_response(404 if "fail" in self.path else 200)
            self.end_headers()
            self.wfile.write(b"no such hook" if "fail" in self.path else b"ok")

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", got
    srv.shutdown()


def test_notification_settings_and_test_messages(client, hook):
    import json
    from urllib.parse import parse_qs, urlparse

    url, got = hook
    assert client.put("/api/settings", json={"notifications": {"kind": "pager"}}).status_code == 400
    assert client.put("/api/settings", json={"notifications": {"url": "ftp://x"}}).status_code == 400
    r = client.put("/api/settings", json={"notifications": {"link": "http://box:7865/"}}).json()
    assert r["notifications"]["link"] == "http://box:7865" and r["notifications"]["on_failed"]
    assert client.post("/api/notifications/test", json={"url": f"{url}/topic", "kind": "ntfy"}).json() == {"ok": True}
    q = parse_qs(urlparse(got[-1]["path"]).query)
    assert q["title"] == ["Dainamo Trainer test"] and q["click"] == ["http://box:7865/#/queue"]
    assert got[-1]["body"].decode().startswith("Notifications work")
    client.post("/api/notifications/test", json={"url": f"{url}/discord", "kind": "discord"})
    assert json.loads(got[-1]["body"])["content"].startswith("**Dainamo Trainer test**")
    r = client.post("/api/notifications/test", json={"url": f"{url}/fail", "kind": "json"}).json()
    assert not r["ok"] and "404" in r["error"]
    client.put("/api/settings", json={"notifications": {"link": ""}})


def _running_job(name, status_json):
    import json
    import os

    from dainamo.server import db
    from dainamo.server.queue import job_dir

    jid = db.new_id()
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, started, config) "
               "VALUES (?,?,?,?,?,?,?,?,?)", (jid, name, "running", "krea2", "auto", db.now(), db.now(),
                                              db.now() - 3725, "{}"))
    os.makedirs(job_dir(jid), exist_ok=True)
    with open(os.path.join(job_dir(jid), "status.json"), "w") as f:
        json.dump(status_json, f)
    return jid


def _wait_for(got, n, timeout=5.0):
    import time

    end = time.time() + timeout
    while len(got) < n and time.time() < end:
        time.sleep(0.05)
    return got


def test_ended_jobs_send_notifications(client, hook):
    import json

    from dainamo.server import db
    from dainamo.server.queue import scheduler

    url, got = hook
    client.put("/api/settings", json={"notifications": {"url": f"{url}/hook", "kind": "json", "on_completed": True,
                                                        "on_failed": True, "on_stopped": False}})

    class Exited:
        def __init__(self, code):
            self.code, self.pid = code, 0

        def poll(self):
            return self.code

    oom = _running_job("oom job", {"phase": "failed", "step": 40, "total_steps": 100,
                                   "message": "OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"})
    done = _running_job("done job", {"phase": "completed", "step": 100, "total_steps": 100, "avg_loss": 0.1234,
                                     "checkpoints": [{"step": 100, "file": "done_job.safetensors", "final": True}]})
    quit_ = _running_job("stopped job", {"phase": "stopped", "step": 5})
    with scheduler._lock:
        scheduler.procs.update({oom: Exited(1), done: Exited(0), quit_: Exited(3)})
        scheduler._reap()
    msgs = {m["job_id"]: m for m in (json.loads(g["body"]) for g in _wait_for(got, 2))}
    assert db.query_one("SELECT status FROM jobs WHERE id=?", (oom,))["status"] == "failed"
    assert msgs[oom]["title"] == "oom job failed" and "CUDA out of memory" in msgs[oom]["message"]
    assert msgs[oom]["event"] == "job_failed"
    assert msgs[done]["title"] == "done job finished" and "Step 100/100 · avg loss 0.1234 · took 1h 2m" in msgs[done]["message"]
    assert "done_job.safetensors" in msgs[done]["message"]
    assert quit_ not in msgs  # stopping isn't sent unless asked for

    # a job that ended while the server was down takes the outcome from its status file
    later = _running_job("finished offline", {"phase": "completed", "step": 10, "total_steps": 10})
    lost = _running_job("lost", {"phase": "training", "step": 3})
    scheduler._reattach()
    status = {j: db.query_one("SELECT status FROM jobs WHERE id=?", (j,))["status"] for j in (later, lost)}
    assert status == {later: "completed", lost: "interrupted"}
    titles = {json.loads(g["body"])["title"] for g in _wait_for(got, 4)}
    assert {"finished offline finished", "lost was interrupted"} <= titles
    client.put("/api/settings", json={"notifications": {"url": ""}})


def test_checkpoint_evaluation_queues_and_puts_the_job_back(client, hook, tmp_path):
    import json
    import os
    import time

    from dainamo.server import db
    from dainamo.server.queue import job_dir, scheduler

    url, got = hook
    client.put("/api/settings", json={"notifications": {"url": f"{url}/hook", "kind": "json", "on_completed": True,
                                                        "on_failed": True, "on_stopped": False}})
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    train = client.post("/api/datasets", json={"name": "train", "path": str(tmp_path / "train")}).json()["id"]
    val = client.post("/api/datasets", json={"name": "val", "path": str(tmp_path / "val")}).json()["id"]
    jid, finished = db.new_id(), db.now() - 600
    cfg = {"family": "krea2", "datasets": [{"id": train, "name": "train", "path": str(tmp_path / "train")}],
           "settings": {"picker_enabled": False}}
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, started, finished, exit_code, config) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (jid, "eval job", "completed", "krea2", "auto", 0.0, finished - 60,
                                                  finished - 60, finished, 0, json.dumps(cfg)))
    d = job_dir(jid)
    os.makedirs(os.path.join(d, "output"))
    training = {"phase": "completed", "message": "Done in 1.0 min", "step": 20, "total_steps": 20,
                "checkpoints": [{"step": 20, "file": "e.safetensors", "final": True}]}
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump(training, f)

    def evaluate(**settings):
        return client.post(f"/api/jobs/{jid}/evaluate", json={"settings": settings})

    assert "no saved checkpoints" in evaluate(picker_enabled=True, val_dataset=val).json()["detail"]
    open(os.path.join(d, "output", "e.safetensors"), "wb").close()
    assert "or both" in evaluate(picker_enabled=False, bleed_enabled=False).json()["detail"]
    assert "trained on all of its images" in evaluate(picker_enabled=True).json()["detail"]  # nothing held out
    assert evaluate(bleed_enabled=True, bleed_prompts=[]).status_code == 400

    r = evaluate(picker_enabled=True, val_dataset=val, val_timesteps=4, sweep_strengths="0.5, 1", steps=999)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    ev = r.json()["config"]["evaluate"]
    assert ev["status"] == "completed" and [v["name"] for v in ev["validation_datasets"]] == ["val"]
    assert ev["settings"]["sweep_strengths"] == [0.5, 1.0] and "steps" not in ev["settings"]  # only evaluation settings
    listed = next(j for j in client.get("/api/jobs").json() if j["id"] == jid)
    assert listed["evaluating"]
    assert "finish" in evaluate(picker_enabled=True, val_dataset=val).json()["detail"]  # already queued
    assert "Cancel" in client.post(f"/api/jobs/{jid}/action", json={"action": "restart"}).json()["detail"]

    # cancelled before it started: the job is as it was, and nothing is sent
    client.post(f"/api/jobs/{jid}/action", json={"action": "stop"})
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "completed" and "evaluate" not in j["config"] and j["finished"] == finished

    class Exited:
        def __init__(self, code):
            self.code, self.pid = code, 0

        def poll(self):
            return self.code

    def run_to_end(code, evaluation, phase="evaluating"):
        """Queue it, pretend it ran (writing ``evaluation`` like the trainer would; a trainer that finished
        has put the job's phase back), and reap it."""
        evaluate(picker_enabled=True, val_dataset=val)
        db.execute("UPDATE jobs SET status='running', pid=1 WHERE id=?", (jid,))
        with open(os.path.join(d, "status.json"), "w") as f:
            running = {"phase": phase, "message": "Evaluating checkpoint at step 20"} if phase == "evaluating" else {}
            json.dump({**training, **running, "evaluation": {"started_at": time.time(), **evaluation}}, f)
        with scheduler._lock:
            scheduler.procs[jid] = Exited(code)
            scheduler._reap()
        return client.get(f"/api/jobs/{jid}").json()

    rec = {"step": 20, "file": "e.safetensors", "reason": "Lowest validation loss"}
    with open(os.path.join(d, "status.json")) as f:
        assert "evaluation" not in json.load(f)
    j = run_to_end(0, {"state": "completed", "recommended": rec}, phase="completed")
    assert (j["status"], j["finished"], j["exit_code"]) == ("completed", finished, 0)
    assert "evaluate" not in j["config"]
    msg = json.loads(_wait_for(got, 1)[-1]["body"])
    assert msg["title"] == "eval job: checkpoint evaluation finished" and msg["event"] == "evaluation_completed"
    assert "Recommended: e.safetensors (step 20). Lowest validation loss." in msg["message"]
    assert next(x for x in client.get("/api/jobs").json() if x["id"] == jid)["recommended"] == rec

    # killed (force stop): the status file gets its phase and message back, and the evaluation is marked stopped
    j = run_to_end(-9, {"state": "running", "evals": [{"step": 0, "file": None, "val_loss": 0.2}]})
    st = j["status_detail"]
    assert j["status"] == "completed" and st["phase"] == "completed" and st["message"] == "Done in 1.0 min"
    assert st["evaluation"]["state"] == "stopped" and st["evaluation"]["evals"][0]["val_loss"] == 0.2
    assert len(got) == 1  # stopping isn't sent unless asked for
    client.put("/api/settings", json={"notifications": {"url": ""}})


def test_a_finished_job_can_be_given_more_steps(client):
    import json
    import os

    from dainamo.server import db
    from dainamo.server.queue import job_dir

    jid = db.new_id()
    cfg = {"family": "krea2", "datasets": [], "settings": {"steps": 2000, "lr_scheduler": "constant"}}
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
               (jid, "long job", "completed", "krea2", "99", 0.0, db.now(), json.dumps(cfg)))
    d = job_dir(jid)
    os.makedirs(os.path.join(d, "state"))
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"phase": "completed", "step": 2000, "total_steps": 2000}, f)

    def extend(steps):
        return client.post(f"/api/jobs/{jid}/action", json={"action": "extend", "steps": steps})

    assert not client.get(f"/api/jobs/{jid}").json()["resumable"]
    assert "state wasn't kept" in extend(2500).json()["detail"]
    open(os.path.join(d, "state", "last.pt"), "wb").close()
    assert client.get(f"/api/jobs/{jid}").json()["resumable"]
    assert "more than step 2000" in extend(2000).json()["detail"]
    assert extend(None).status_code == 400
    r = extend(2500)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert r.json()["config"]["settings"]["steps"] == 2500
    listed = next(j for j in client.get("/api/jobs").json() if j["id"] == jid)
    assert listed["progress"]["total_steps"] == 2500
    assert "Wait" in extend(3000).json()["detail"]
    client.post(f"/api/jobs/{jid}/action", json={"action": "stop"})


def test_editing_a_job(client, tmp_path, monkeypatch):
    import json
    import os

    from dainamo.server import db
    from dainamo.server.queue import job_dir, scheduler

    monkeypatch.setattr(scheduler, "_launch", lambda: None)  # its jobs stay queued
    pid = client.post("/api/profiles", json={"name": "ed", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "ed", "path": str(tmp_path)}).json()["id"]
    jid = client.post("/api/jobs", json={"name": "ed", "profile_id": pid, "datasets": [{"id": did}],
                                         "settings": {"steps": 3000}}).json()["id"]

    def edit(settings=None, **job):
        """Save the job as the form does: its settings and datasets as they are, with ``settings`` changed."""
        cfg = client.get(f"/api/jobs/{jid}/config").json()
        datasets = [{k: d[k] for k in ("id", "repeats", "resolution", "flip", "caption_dropout")} for d in cfg["datasets"]]
        return client.put(f"/api/jobs/{jid}", json={"name": cfg["name"], "profile_id": pid, "gpu": "auto",
                                                    "datasets": datasets, "settings": {**cfg["settings"], **(settings or {})},
                                                    **job})

    # before it trains it takes any change, and keeps its place in the queue
    r = edit({"steps": 4000, "rank": 16}, name="ed v1", gpu="1")
    assert r.status_code == 200
    j = r.json()
    assert (j["status"], j["name"], j["gpu"], j["config"]["name"]) == ("queued", "ed v1", "1", "ed v1")
    assert (j["config"]["settings"]["steps"], j["config"]["settings"]["rank"]) == (4000, 16)

    d = job_dir(jid)
    os.makedirs(os.path.join(d, "state"))
    open(os.path.join(d, "state", "last.pt"), "wb").close()
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"phase": "stopped", "step": 1200}, f)
    db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (jid,))

    # once it has trained, its own settings sent back change nothing, and most changes carry on from its last save,
    # each noted in its log: how long it trains, what it saves and samples, how it learns, what it learns from
    assert edit().status_code == 200
    r = edit({"steps": 5000, "sample_every": 100, "offload": 20, "learning_rate": 3.3e-5, "num_frames": 49},
             datasets=[{"id": did, "repeats": 3}])
    s = r.json()["config"]["settings"]
    assert (r.json()["status"], s["steps"], s["sample_every"], s["offload"], s["rank"]) == ("stopped", 5000, 100, 20, 16)
    assert (s["learning_rate"], r.json()["config"]["datasets"][0]["repeats"]) == (3.3e-5, 3)
    assert os.path.exists(os.path.join(d, "state", "last.pt"))
    with open(os.path.join(d, "log.txt"), encoding="utf-8") as f:
        log = f.read()
    assert "carries on from step 1200 with" in log and "learning_rate 0.0001 → 3.3e-05" in log and "datasets" in log
    assert "more than step 1200" in edit({"steps": 1000}).json()["detail"]
    # while the checkpoint picker is on, what it scores can't change mid-run; what it does at the end can
    db.execute("UPDATE jobs SET config=json_set(config, '$.settings.picker_enabled', json('true')) WHERE id=?", (jid,))
    assert edit({"picker_enabled": True, "sweep_enabled": True, "bleed_max": 10}).status_code == 200
    r = edit({"picker_enabled": True, "num_frames": 97})
    assert r.status_code == 409 and "what the checkpoint picker scores" in r.json()["detail"]
    assert edit({"picker_enabled": True}, datasets=[{"id": did, "repeats": 2}]).status_code == 409
    assert edit({"picker_enabled": False, "num_frames": 97}).status_code == 200  # turned off, it scores nothing
    assert edit({"picker_enabled": True}).status_code == 409  # and turned on, checkpoints so far weren't scored
    # another network can't carry on from the saved one: it starts again from step 0, once that's confirmed
    r = edit({"rank": 32, "optimizer": "adamw"})
    assert r.status_code == 409
    assert "network (LoRA or LoKr, rank, alpha) and the optimizer" in r.json()["detail"] and "1200 steps" in r.json()["detail"]
    assert client.get(f"/api/jobs/{jid}/config").json()["settings"]["rank"] == 16
    r = edit({"rank": 32}, restart=True)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert r.json()["config"]["settings"]["rank"] == 32
    assert not os.path.exists(os.path.join(d, "state")) and not os.path.exists(os.path.join(d, "status.json"))

    # a finished job trains on with Continue training, and a running one can't change
    os.makedirs(os.path.join(d, "state"))
    open(os.path.join(d, "state", "last.pt"), "wb").close()
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"phase": "completed", "step": 5000}, f)
    db.execute("UPDATE jobs SET status='completed' WHERE id=?", (jid,))
    assert "Continue training" in edit({"steps": 6000}).json()["detail"]
    assert edit({"sample_every": 50}).json()["config"]["settings"]["sample_every"] == 50
    db.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    assert "Stop the job" in edit({"sample_every": 25}).json()["detail"]
    db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (jid,))


def test_masked_loss_ema_noise_range_and_loss_weighting_are_checked(client, tmp_path, monkeypatch):
    from dainamo.server.queue import scheduler
    from dainamo.server.validation import LOSS_AND_EMA_DEFAULTS

    monkeypatch.setattr(scheduler, "_launch", lambda: None)  # its jobs stay queued
    pid = client.post("/api/profiles", json={"name": "ml", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "ml", "path": str(tmp_path)}).json()["id"]

    def post(settings):
        return client.post("/api/jobs", json={"name": "ml", "profile_id": pid, "datasets": [{"id": did}],
                                              "settings": settings})

    # left out, each is off or the whole range
    s = post({}).json()["config"]["settings"]
    assert {k: s[k] for k in LOSS_AND_EMA_DEFAULTS} == LOSS_AND_EMA_DEFAULTS
    r = post({"masked_loss": True, "mask_min_weight": "0", "ema_enabled": 1, "ema_decay": "0.9999",
              "timestep_min": 0.3, "timestep_max": "1", "loss_weighting": "min_snr", "min_snr_gamma": "2.5"})
    s = r.json()["config"]["settings"]
    assert [s[k] for k in LOSS_AND_EMA_DEFAULTS] == [True, 0.0, True, 0.9999, 0.3, 1.0, "min_snr", 2.5]
    for bad, why in (({"mask_min_weight": 1.5}, "Setting 'mask_min_weight'=1.5 is out of range (0–1)"),
                     ({"ema_decay": 0.5}, "Setting 'ema_decay'=0.5 is out of range (0.9–0.9999)"),
                     ({"ema_decay": 1}, "'ema_decay'=1 is out of range"), ({"ema_decay": "nan"}, "out of range"),
                     ({"timestep_min": -0.1}, "'timestep_min'=-0.1 is out of range (0–1)"),
                     ({"timestep_max": 2}, "'timestep_max'=2 is out of range"),
                     ({"timestep_min": 0.6, "timestep_max": 0.4},
                      "The noise range's minimum (0.6) must be below its maximum (0.4)"),
                     ({"timestep_min": 0.5, "timestep_max": 0.5}, "must be below its maximum"),
                     ({"loss_weighting": "snr"}, "Unknown loss weighting 'snr' (none, min_snr or cosmap)"),
                     ({"min_snr_gamma": 0}, "Setting 'min_snr_gamma'=0 must be above 0"),
                     ({"min_snr_gamma": "a lot"}, "Setting 'min_snr_gamma' must be a number"),
                     ({"mask_min_weight": "some"}, "Setting 'mask_min_weight' must be a number")):
        r = post(bad)
        assert r.status_code == 400 and why in r.json()["detail"], bad
    # a finetune's weights are the whole model, so it keeps no average of them: refused rather than dropped
    finetune = {"finetune_enabled": True, "optimizer": "adafactor"}
    assert post(finetune).status_code == 200
    r = post({**finetune, "ema_enabled": True})
    assert r.status_code == 400 and "Turn EMA off to finetune" in r.json()["detail"]
    for j in client.get("/api/jobs").json():
        if j["name"] == "ml":
            client.delete(f"/api/jobs/{j['id']}")


def test_masks_ema_and_the_noise_range_change_without_starting_again(client, tmp_path, monkeypatch):
    import json

    from dainamo.server import db
    from dainamo.server.queue import job_dir, scheduler

    monkeypatch.setattr(scheduler, "_launch", lambda: None)  # its jobs stay queued
    pid = client.post("/api/profiles", json={"name": "me", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "me", "path": str(tmp_path)}).json()["id"]
    jid = client.post("/api/jobs", json={"name": "me", "profile_id": pid, "datasets": [{"id": did}],
                                         "settings": {"steps": 3000}}).json()["id"]
    d = job_dir(jid)
    os.makedirs(os.path.join(d, "state"))
    open(os.path.join(d, "state", "last.pt"), "wb").close()
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"phase": "stopped", "step": 800}, f)
    # a job from before these settings: it trained without them
    db.execute("UPDATE jobs SET status='stopped', config=json_remove(config, '$.settings.masked_loss', "
               "'$.settings.ema_enabled', '$.settings.timestep_min', '$.settings.loss_weighting') WHERE id=?", (jid,))

    def edit(settings):
        cfg = client.get(f"/api/jobs/{jid}/config").json()
        datasets = [{k: x[k] for k in ("id", "repeats", "resolution", "flip", "caption_dropout")} for x in cfg["datasets"]]
        return client.put(f"/api/jobs/{jid}", json={"name": cfg["name"], "profile_id": pid, "gpu": "auto",
                                                    "datasets": datasets, "settings": {**cfg["settings"], **settings}})

    r = edit({"masked_loss": True, "ema_enabled": True, "ema_decay": 0.995, "timestep_min": 0.2,
              "loss_weighting": "cosmap"})
    assert r.status_code == 200 and r.json()["status"] == "stopped"
    assert os.path.exists(os.path.join(d, "state", "last.pt"))
    with open(os.path.join(d, "log.txt"), encoding="utf-8") as f:
        log = f.read()
    assert "carries on from step 800 with" in log
    for change in ("ema_enabled False → True", "masked_loss False → True", "timestep_min 0.0 → 0.2",
                   "loss_weighting none → cosmap", "ema_decay 0.99 → 0.995"):
        assert change in log, change
    # while the checkpoint picker is on, the noise levels and masks are part of what it scores
    db.execute("UPDATE jobs SET config=json_set(config, '$.settings.picker_enabled', json('true')) WHERE id=?", (jid,))
    for change in ({"timestep_max": 0.8}, {"masked_loss": False}, {"mask_min_weight": 0.5},
                   {"timestep_sampling": "uniform"}):
        r = edit({"picker_enabled": True, **change})
        assert r.status_code == 409 and "what the checkpoint picker scores" in r.json()["detail"], change
    # each checkpoint is scored with the weights it saves, and validation leaves the loss weighting out
    assert edit({"picker_enabled": True, "ema_enabled": False, "loss_weighting": "min_snr"}).status_code == 200
    client.delete(f"/api/jobs/{jid}")


def test_a_job_keeps_its_datasets_once_theyre_removed(client, tmp_path, monkeypatch):
    from dainamo.server.queue import scheduler

    monkeypatch.setattr(scheduler, "_launch", lambda: None)  # its jobs stay queued
    for n in ("after", "before"):
        (tmp_path / n).mkdir()
    pid = client.post("/api/profiles", json={"name": "rm", "family": "qwen_image_2", "data": {}}).json()["id"]
    did, sid = (client.post("/api/datasets", json={"name": n, "path": str(tmp_path / n), "trigger": t}).json()["id"]
                for n, t in (("after", "ohwx"), ("before", "")))
    body = {"name": "rm", "profile_id": pid, "settings": {}, "datasets": [{"id": did, "sources": [sid]}]}
    jid = client.post("/api/jobs", json=body).json()["id"]
    for i in (did, sid):
        client.delete(f"/api/datasets/{i}")
    # renamed and given another learning rate, it keeps them as it had them
    r = client.put(f"/api/jobs/{jid}", json={**body, "name": "rm v2", "settings": {"learning_rate": 5e-5}})
    assert r.status_code == 200, r.text
    d = r.json()["config"]["datasets"][0]
    assert (d["id"], d["name"], d["path"], d["trigger"]) == (did, "after", str(tmp_path / "after"), "ohwx")
    assert d["sources"] == [{"id": sid, "name": "before", "path": str(tmp_path / "before")}]
    assert (r.json()["name"], r.json()["config"]["settings"]["learning_rate"]) == ("rm v2", 5e-5)
    # one it never had is still not found, and so is the removed one for a new job
    assert client.put(f"/api/jobs/{jid}", json={**body, "datasets": [{"id": "nope"}]}).status_code == 404
    assert client.post("/api/jobs", json=body).status_code == 404
    client.post(f"/api/jobs/{jid}/action", json={"action": "stop"})


def test_a_job_stopping_when_the_server_restarted_is_still_forced(client, monkeypatch):
    import time

    from dainamo.server import db, procs
    from dainamo.server.queue import STOP_GRACE, scheduler

    jid = db.new_id()
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, pid, config) VALUES "
               "(?,?,?,?,?,?,?,?,?)", (jid, "slow to stop", "stopping", "krea2", "99", 0.0, db.now(), os.getpid(), "{}"))
    monkeypatch.setattr(procs, "is_trainer", lambda pid, job_id: job_id == jid)  # its trainer outlived the server
    killed = []
    monkeypatch.setattr(procs, "kill_tree", killed.append)
    with scheduler._lock:  # the queue's loop leaves it alone meanwhile
        scheduler._reattach()
        assert time.time() < scheduler.stop_deadline[jid] <= time.time() + STOP_GRACE
        scheduler._reap()
        assert killed == []  # still has time to save
        scheduler.stop_deadline[jid] = time.time() - 1
        scheduler._reap()
        assert killed == [os.getpid()]
        del scheduler.procs[jid], scheduler.stop_deadline[jid]
    db.execute("DELETE FROM jobs WHERE id=?", (jid,))


def test_a_missing_file_or_a_folder_in_a_dataset(client):
    d = client.post("/api/datasets", json={"name": "uploads"}).json()  # uploaded files: deleted from disk
    folder = os.path.join(d["path"], "sub")
    os.makedirs(folder)
    assert client.delete("/api/media", params={"path": folder}).status_code == 400 and os.path.isdir(folder)
    assert client.get("/api/thumb", params={"path": os.path.join(d["path"], "gone.png")}).status_code == 404
    assert client.get("/api/thumb", params={"path": folder}).status_code == 404
    client.delete(f"/api/datasets/{d['id']}", params={"delete_files": True})


def test_clearing_the_shared_cache_spares_what_running_jobs_use(client):
    import json
    import os

    from dainamo.server import db
    from dainamo.server.queue import CACHE_DIR, job_dir

    for rel, size in (("latents/vae1/signature.json", 10), ("latents/vae1/a.pt", 100), ("latents/vae1/b.pt", 200),
                      ("latents/vae1/c.pt.77.tmp", 50), ("text/te1/signature.json", 10), ("text/te1/x.pt", 400)):
        path = os.path.join(CACHE_DIR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)
    jid = db.new_id()
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
               (jid, "busy", "running", "krea2", "99", 0.0, db.now(), json.dumps({"settings": {}})))
    os.makedirs(job_dir(jid))
    with open(os.path.join(job_dir(jid), "cache.json"), "w") as f:
        json.dump({"root": CACHE_DIR, "files": ["latents/vae1/a.pt"]}, f)  # as its trainer lists them

    assert client.get("/api/disk").json()["shared_cache"] == {"total": 770, "in_use": 100}
    assert client.get(f"/api/jobs/{jid}/disk").json()["shared"] == 100  # on its Disk use card
    assert client.post("/api/disk/clean", json={"parts": ["cache"]}).json()["freed"] >= 600
    # what the running job uses stays, and so does a file another job is still writing
    assert sorted(os.listdir(os.path.join(CACHE_DIR, "latents", "vae1"))) == ["a.pt", "c.pt.77.tmp", "signature.json"]
    assert not os.path.exists(os.path.join(CACHE_DIR, "text", "te1"))  # emptied, so gone
    db.execute("UPDATE jobs SET status='completed' WHERE id=?", (jid,))
    client.post("/api/disk/clean", json={"parts": ["cache"]})
    assert sorted(os.listdir(os.path.join(CACHE_DIR, "latents", "vae1"))) == ["c.pt.77.tmp", "signature.json"]


def test_disk_use_and_clearing_what_finished_jobs_dont_need(client):
    import json
    import os

    from dainamo.server import db
    from dainamo.server.queue import job_dir

    def job(name, status):
        jid = db.new_id()
        db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
                   (jid, name, status, "krea2", "99", 0.0, db.now(), json.dumps({"settings": {}})))
        for part, size in (("output", 300), ("samples", 200), ("cache/latents", 5000), ("state", 1000)):
            os.makedirs(os.path.join(job_dir(jid), part))
            with open(os.path.join(job_dir(jid), part, "f.bin"), "wb") as f:
                f.write(b"x" * size)
        with open(os.path.join(job_dir(jid), "log.txt"), "w") as f:
            f.write("x" * 10)
        return jid

    done, stopped, running = job("done", "completed"), job("paused", "stopped"), job("busy", "running")
    assert client.get(f"/api/jobs/{done}/disk").json() == {"output": 300, "samples": 200, "cache": 5000, "state": 1000,
                                                          "other": 10, "total": 6510, "shared": 0}
    assert client.get("/api/disk").json()["jobs"][stopped]["total"] == 6510

    r = client.post("/api/disk/clean", json={"parts": ["state"], "jobs": [done, stopped, running]}).json()
    assert r["freed"] == 1000 and r["jobs"] == 1  # only the completed job's: the stopped one resumes from it
    assert any("resume" in k for k in r["kept"]) and any("running" in k for k in r["kept"])
    assert not client.get(f"/api/jobs/{done}").json()["resumable"]
    r = client.post("/api/disk/clean", json={"parts": ["cache"]}).json()  # every finished job
    disk = client.get("/api/disk").json()["jobs"]
    assert disk[done]["cache"] == 0 and disk[stopped]["cache"] == 0 and disk[running]["cache"] == 5000
    assert disk[stopped]["state"] == 1000 and disk[done]["output"] == 300
    assert client.post("/api/disk/clean", json={"parts": ["output"]}).status_code == 400
    for jid in (done, stopped, running):
        db.execute("DELETE FROM jobs WHERE id=?", (jid,))


def test_bulk_caption_edits_preview_apply_and_undo(client, tmp_path):
    from PIL import Image

    caps = {"a": "ohwx, a man in a red coat", "b": "a man, ohwx2 style", "c": None, "d": "Red car (red) on a road"}
    for name, cap in caps.items():
        Image.new("RGB", (8, 8)).save(tmp_path / f"{name}.png")
        if cap is not None:
            (tmp_path / f"{name}.txt").write_text(cap)
    did = client.post("/api/datasets", json={"name": "caps", "path": str(tmp_path), "trigger": "ohwx"}).json()["id"]

    def bulk(**body):
        return client.post(f"/api/datasets/{did}/captions/bulk", json=body)

    def caption(name):
        p = tmp_path / f"{name}.txt"
        return p.read_text() if p.exists() else None

    r = bulk(op="replace", find="red", replace="blue").json()  # a preview: nothing is written
    assert (r["changed"], r["total"], r["applied"]) == (2, 4, False) and caption("a") == caps["a"]
    assert {x["name"]: x["after"] for x in r["examples"]} == {"a.png": "ohwx, a man in a blue coat",
                                                             "d.png": "blue car (blue) on a road"}
    assert bulk(op="replace", find="red", replace="blue", match_case=True).json()["changed"] == 2
    r = bulk(op="replace", find=r"\((\w+)\)", replace=r"[\1]", regex=True).json()
    assert r["examples"][0]["after"] == "Red car [red] on a road"
    assert bulk(op="replace", find="red", replace=r"\1").json()["examples"][0]["after"] == r"ohwx, a man in a \1 coat"
    assert bulk(op="replace", find="(", regex=True).status_code == 400
    assert bulk(op="replace", find="red", replace=r"\3", regex=True).status_code == 400

    # whole words, tidied; files without a caption only get one when asked
    r = bulk(op="remove", text="OHWX", apply=True).json()
    assert r["changed"] == 1 and caption("a") == "a man in a red coat" and caption("b") == caps["b"]
    assert client.get(f"/api/datasets/{did}/captions/undo").json()["what"] == "Removed “OHWX”"
    r = bulk(op="add", text="ohwx", apply=True).json()
    assert r["changed"] == 3 and caption("a") == "ohwx, a man in a red coat" and caption("c") is None
    assert caption("b") == "ohwx, a man, ohwx2 style" and caption("d") == "ohwx, Red car (red) on a road"
    assert bulk(op="add", text="ohwx").json()["changed"] == 0  # c has no caption to add to
    assert bulk(op="add", text="ohwx", uncaptioned=True).json()["examples"] == [{"name": "c.png", "before": None,
                                                                               "after": "ohwx"}]
    bulk(op="add", text="photo", where="end", uncaptioned=True, apply=True)
    assert caption("c") == "photo" and caption("d") == "ohwx, Red car (red) on a road, photo"

    # undo puts back the last edit only, removing a caption it created
    assert client.post(f"/api/datasets/{did}/captions/undo").json()["restored"] == 4
    assert caption("c") is None and caption("d") == "ohwx, Red car (red) on a road"
    assert caption("a") == "ohwx, a man in a red coat"
    assert not client.get(f"/api/datasets/{did}/captions/undo").json()["available"]
    assert client.post(f"/api/datasets/{did}/captions/undo").status_code == 400
    assert bulk(op="add", text="  ").status_code == 400


def test_a_training_sample_opens_in_the_sandbox(client):
    import json
    import os

    from dainamo.server import db
    from dainamo.server.queue import job_dir

    data = {"diffusion_model": "/models/krea.safetensors", "sample_cfg": 0, "sample_negative": "blurry"}  # steps left out
    pid = client.post("/api/profiles", json={"name": "krea for sandbox", "family": "krea2", "data": data}).json()["id"]
    jid = db.new_id()
    cfg = {"family": "krea2", "profile_name": "krea for sandbox", "profile": data, "datasets": [], "settings": {"offload": 25, "low_vram": True}}
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
               (jid, "fox", "completed", "krea2", "99", 0.0, db.now(), json.dumps(cfg)))
    out = os.path.join(job_dir(jid), "output")
    os.makedirs(out)
    for f in ("fox_000250.safetensors", "fox.safetensors"):
        open(os.path.join(out, f), "wb").close()
    sample = {"prompt": "ohwx fox", "width": 768, "height": 1344, "seed": 43, "lora_scale": 0.8, "frames": 1}
    status = {"samples": [{"step": st, "index": 1, "file": f"s{st}.png", **sample} for st in (0, 250, 300, 500)] +
              [{"step": 100, "index": 0, "prompt": "old", "file": "old.png"}],
              "checkpoints": [{"step": 250, "file": "fox_000250.safetensors", "final": False},
                              {"step": 500, "file": "fox.safetensors", "final": True}]}
    with open(os.path.join(job_dir(jid), "status.json"), "w") as f:
        json.dump(status, f)

    def preset(step, index=1):
        return client.get(f"/api/jobs/{jid}/sandbox?step={step}&index={index}")

    p = preset(250).json()
    assert {k: p[k] for k in ("profile_id", "workflow", "prompt", "negative", "width", "height", "seed", "offload")} == \
        {"profile_id": pid, "workflow": "t2i", "prompt": "ohwx fox", "negative": "blurry", "width": 768, "height": 1344,
         "seed": 43, "offload": 25}
    assert p["low_vram"] is True  # the job's own setting comes along
    assert p["steps"] == 8 and p["cfg"] == 1.0  # the profile form's defaults, as training samples use
    assert p["loras"] == [{"path": os.path.join(out, "fox_000250.safetensors"), "strength": 0.8}] and p["notes"] == []
    p = preset(300).json()
    assert p["loras"][0]["path"].endswith("fox_000250.safetensors") and "nearest" in p["notes"][0]
    assert preset(500).json()["loras"][0]["path"].endswith("fox.safetensors")
    p = preset(0).json()
    assert p["loras"] == [] and "base model" in p["notes"][0]
    assert preset(100, 0).status_code == 400  # from before samples kept their settings
    assert preset(999).status_code == 404
    client.put(f"/api/profiles/{pid}", json={"name": "renamed", "family": "krea2", "data": {**data, "diffusion_model": "/other"}})
    assert "is gone" in preset(250).json()["detail"]
    db.execute("DELETE FROM jobs WHERE id=?", (jid,))


def test_worker_processes_are_found_and_killed_with_their_children():
    import subprocess
    import sys

    import psutil
    from dainamo.server import procs

    code = ("import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
            " time.sleep(60)")
    p = subprocess.Popen([sys.executable, "-c", code, "dainamo.engine.run", "/jobs/abc123"], **procs.detached())
    try:
        child = None
        for _ in range(100):
            kids = psutil.Process(p.pid).children()
            if kids:
                child = kids[0]
                break
            psutil.wait_procs([], timeout=0.05)
        assert child is not None
        assert procs.is_trainer(p.pid, "abc123") and not procs.is_trainer(p.pid, "other")
        assert procs.pid_alive(p.pid)
        procs.kill_tree(p.pid)
        assert p.wait(timeout=10) in procs.KILLED
        child.wait(timeout=10)  # the grandchild went too
        assert not procs.pid_alive(p.pid) and not procs.is_trainer(p.pid, "abc123")
    finally:
        if p.poll() is None:
            p.kill()


def test_a_job_with_the_picker_off_skips_the_strength_sweep():
    from dainamo.engine.trainer import Trainer

    t = Trainer.__new__(Trainer)  # the Create Job form sends the sweep strengths as text
    t.s, t.status = {"picker_enabled": False, "sweep_enabled": True, "sweep_strengths": "0.6, 0.8, 1.0"}, {}
    t.ev = t.status  # a training job's evaluation results live in its status
    assert t.strength_sweep(None, None) is None
    t.s["picker_enabled"], t.status["recommended"] = True, {"step": 1, "file": "x"}
    t.sample_specs = lambda: []  # nothing to render: still no error from the text strengths
    assert t.strength_sweep(None, None) is None


@pytest.fixture
def llm():
    """A stand-in OpenAI-compatible LLM: (base url, requests received, replies to give). Under /textonly it
    refuses requests with images, like a model without vision."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    got, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            got.append({"path": self.path, "body": body})
            content = body["messages"][-1]["content"]
            images = sum(1 for c in content if c.get("type") == "image_url") if isinstance(content, list) else 0
            if ("textonly" in self.path and images) or ("oneimage" in self.path and images > 1):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "image input is not supported"}')
                return
            reply = replies.pop(0) if replies else "A long prompt."  # a dict is a whole response
            out = json.dumps(reply if isinstance(reply, dict) else {"choices": [{"message": {"content": reply}}]})
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", got, replies
    srv.shutdown()


def test_enhance_prompt_asks_the_local_llm(client, llm, tmp_path):
    from PIL import Image

    url, got, replies = llm
    client.put("/api/settings", json={"captioner": {"endpoint": f"{url}/v1", "model": "qwen-vl"}})
    krea = client.post("/api/profiles", json={"name": "ek", "family": "krea2", "data": {}}).json()["id"]
    qwen = client.post("/api/profiles", json={"name": "eq", "family": "qwen_image_2", "data": {}}).json()["id"]
    try:
        replies.append('<think>the user wants…</think> Prompt: "ohwx man in a red coat, studio light"')
        r = client.post("/api/sandbox/enhance", json={"profile_id": krea, "workflow": "t2i", "prompt": " ohwx man ",
                                                      "seed": 7})
        assert r.json() == {"prompt": "ohwx man in a red coat, studio light"}
        body = got[-1]["body"]
        assert got[-1]["path"] == "/v1/chat/completions" and body["seed"] == 7 and body["model"] == "qwen-vl"
        assert "text-to-image" in body["messages"][0]["content"] and body["messages"][1]["content"] == "ohwx man"

        img = tmp_path / "source.png"
        Image.new("RGB", (64, 32)).save(img)
        edit = {"profile_id": qwen, "workflow": "edit", "prompt": "make it night", "images": [str(img)]}
        assert client.post("/api/sandbox/enhance", json=edit).status_code == 200
        body = got[-1]["body"]
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        assert "image-editing" in system and "attached image is the" in system and "seed" not in body
        assert user[0] == {"type": "text", "text": "make it night"} and user[1]["type"] == "image_url"

        # a model without vision gets the text alone
        client.put("/api/settings", json={"captioner": {"endpoint": f"{url}/textonly/v1"}})
        n = len(got)
        assert client.post("/api/sandbox/enhance", json=edit).json() == {"prompt": "A long prompt."}
        assert [isinstance(g["body"]["messages"][1]["content"], list) for g in got[n:]] == [True, False]

        replies.append("<think>still thinking")  # cut off mid-reasoning: no prompt
        r = client.post("/api/sandbox/enhance", json=edit)
        assert r.status_code == 502 and "no text" in r.json()["detail"]
        assert client.post("/api/sandbox/enhance", json={**edit, "prompt": "  "}).status_code == 400
        client.put("/api/settings", json={"captioner": {"endpoint": "http://127.0.0.1:9/v1"}})
        r = client.post("/api/sandbox/enhance", json=edit)
        assert r.status_code == 502 and "Couldn't reach the local LLM" in r.json()["detail"]
    finally:
        client.put("/api/settings", json={"captioner": {"endpoint": "http://127.0.0.1:8080/v1", "model": ""}})


def test_enhance_prompt_for_image_to_image_describes_what_the_image_becomes(client, llm, tmp_path):
    from PIL import Image

    url, got, _ = llm
    client.put("/api/settings", json={"captioner": {"endpoint": f"{url}/v1", "model": "qwen-vl"}})
    krea = client.post("/api/profiles", json={"name": "ei", "family": "krea2", "data": {}}).json()["id"]
    img = tmp_path / "start.png"
    Image.new("RGB", (64, 32)).save(img)
    try:
        r = client.post("/api/sandbox/enhance", json={"profile_id": krea, "workflow": "i2i", "prompt": "as a painting",
                                                      "images": [str(img)]})
        assert r.status_code == 200
        body = got[-1]["body"]
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        assert "text-to-image" in system and system.endswith(
            "The attached image is the starting image; describe what it should become.")
        assert user[0] == {"type": "text", "text": "as a painting"} and user[1]["type"] == "image_url"
    finally:
        client.put("/api/settings", json={"captioner": {"endpoint": "http://127.0.0.1:8080/v1", "model": ""}})


def test_enhancer_settings_and_input_image_size(client, tmp_path):
    from PIL import Image

    assert client.put("/api/settings", json={"enhancer": {"temperature": 5}}).status_code == 400
    assert client.put("/api/settings", json={"enhancer": {"max_tokens": "lots"}}).status_code == 400
    r = client.put("/api/settings", json={"enhancer": {"image": "Be brief.", "edit": "   "}}).json()["enhancer"]
    assert r["image"] == "Be brief." and r["edit"].startswith("You rewrite instructions")  # blank: the default
    client.put("/api/settings", json={"enhancer": {"image": ""}})

    img = tmp_path / "frame.png"
    Image.new("RGB", (640, 360)).save(img)
    assert client.get("/api/sandbox/image-size", params={"path": str(img)}).json() == {"width": 640, "height": 360}
    assert client.get("/api/sandbox/image-size", params={"path": str(tmp_path / "nope.png")}).status_code == 404


def test_caption_styles_are_validated(client):
    good = {"default": "b", "styles": [{"id": "a", "name": "A", "prompt": "x"}, {"id": "b", "name": "B", "prompt": "y"}]}
    assert client.put("/api/settings", json={"caption_styles": good}).json()["caption_styles"] == good
    for bad in ({"styles": []}, {"styles": [{"id": "a", "name": "A", "prompt": " "}]},
                {"styles": [{"id": "a", "name": "A", "prompt": "x"}, {"id": "a", "name": "B", "prompt": "y"}]},
                {"styles": [{"id": "Bad Id!", "name": "A", "prompt": "x"}]}, {"styles": [{"id": "a", "name": "", "prompt": "x"}]}):
        assert client.put("/api/settings", json={"caption_styles": {"default": "a", **bad}}).status_code == 400, bad
    r = client.put("/api/settings", json={"caption_styles": {"default": "gone", "styles": good["styles"]}}).json()
    assert r["caption_styles"]["default"] == "a"  # a default that doesn't exist falls back to the first
    # a system prompt and the style's own sampling are optional, and kept only when set
    own = {"id": "c", "name": "C", "prompt": "z", "system": "Be exact.", "temperature": 0, "max_tokens": 1024}
    blank = {"id": "d", "name": "D", "prompt": "w", "system": " ", "temperature": "", "max_tokens": None}
    r = client.put("/api/settings", json={"caption_styles": {"default": "c", "styles": [own, blank]}}).json()
    assert r["caption_styles"]["styles"] == [own, {"id": "d", "name": "D", "prompt": "w"}]
    for bad in ({"temperature": 3}, {"temperature": "warm"}, {"max_tokens": 8}, {"max_tokens": 100_000}):
        r = client.put("/api/settings", json={"caption_styles": {"default": "c", "styles": [{**own, **bad}]}})
        assert r.status_code == 400 and ("caption provider's" in r.text or "a number" in r.text), bad
    from dainamo.server import db
    client.put("/api/settings", json={"caption_styles": db.DEFAULT_SETTINGS["caption_styles"]})


def test_auto_caption_uses_the_chosen_style_and_remembers_it(client, llm, tmp_path):
    import time

    from PIL import Image

    url, got, replies = llm
    for i in range(2):
        Image.new("RGB", (32, 32)).save(tmp_path / f"img{i}.png")
    client.put("/api/settings", json={"captioner": {"endpoint": f"{url}/v1"}})
    did = client.post("/api/datasets", json={"name": "styled", "path": str(tmp_path), "trigger": "ohwx"}).json()["id"]
    tags = next(st for st in client.get("/api/settings").json()["caption_styles"]["styles"] if st["id"] == "tags")

    def caption(**kw):
        n = len(got)
        r = client.post(f"/api/datasets/{did}/autocaption", json={"overwrite": True, **kw})
        end = time.time() + 10
        while client.get(f"/api/datasets/{did}/autocaption").json()["running"] and time.time() < end:
            time.sleep(0.05)
        return r, [next(c["text"] for c in g["body"]["messages"][-1]["content"] if c["type"] == "text")
                   for g in got[n:]]

    try:
        r, prompts = caption(style="tags")
        assert r.json()["style"] == "tags" and prompts == [tags["prompt"]] * 2
        assert (tmp_path / "img0.txt").read_text() == "ohwx, A long prompt."
        d = next(x for x in client.get("/api/datasets").json() if x["id"] == did)
        assert d["caption_options"] == {"style": "tags", "clip_frames": 0, "speech": ""}
        r, prompts = caption()  # no style given: the one this dataset used last time
        assert r.json()["style"] == "tags" and prompts == [tags["prompt"]] * 2
        assert client.post(f"/api/datasets/{did}/autocaption", json={"style": "gone"}).status_code == 400
        # FLUX 3's style: its system prompt first, the image before its prompt, and its own temperature and length
        from dainamo.server import db
        n = len(got)
        r, prompts = caption(style="flux3")
        assert r.json()["style"] == "flux3" and prompts == [db.FLUX3_CAPTION_STYLE["prompt"]] * 2
        flux3 = db.FLUX3_CAPTION_STYLE
        for g in got[n:]:
            system, user = g["body"]["messages"]
            assert system == {"role": "system", "content": flux3["system"]}
            assert [c["type"] for c in user["content"]] == ["image_url", "text"]
            assert user["content"][1]["text"] == flux3["prompt"]
            assert (g["body"]["temperature"], g["body"]["max_tokens"]) == (0, 1024)
    finally:
        client.put("/api/settings", json={"captioner": {"endpoint": "http://127.0.0.1:8080/v1"}})



def test_auto_caption_never_saves_a_cut_off_or_empty_caption(client, llm, tmp_path):
    from PIL import Image

    url, got, replies = llm
    for name in ("a", "b"):
        Image.new("RGB", (32, 32)).save(tmp_path / f"{name}.png")
    client.put("/api/settings", json={"captioner": {"endpoint": f"{url}/v1"}})
    did = client.post("/api/datasets", json={"name": "cut", "path": str(tmp_path), "trigger": ""}).json()["id"]
    cut = {"choices": [{"message": {"content": "A grey square on a"}, "finish_reason": "length"}],
           "usage": {"prompt_tokens": 300, "completion_tokens": 600}}
    try:
        # a.png: cut off, then whole with more room; b.png: nothing but thinking
        replies.extend([cut, "A grey square.", "<think>a square, grey"])
        _, status = _run_caption(client, did)
        assert (tmp_path / "a.txt").read_text() == "A grey square."
        assert [g["body"]["max_tokens"] for g in got[-3:]] == [600, 2400, 600]
        assert not (tmp_path / "b.txt").exists()
        assert status["errors"] == ["b.png: the LLM returned no text"]
    finally:
        client.put("/api/settings", json={"captioner": {"endpoint": "http://127.0.0.1:8080/v1"}})

def _clip(path, seconds=2, audio=True):
    import subprocess

    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x96:rate=24:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
    return str(path)


def _run_caption(client, did, **kw):
    import time

    r = client.post(f"/api/datasets/{did}/autocaption", json={"overwrite": True, **kw})
    assert r.status_code == 200, r.text
    end = time.time() + 30
    while client.get(f"/api/datasets/{did}/autocaption").json()["running"] and time.time() < end:
        time.sleep(0.05)
    return r.json(), client.get(f"/api/datasets/{did}/autocaption").json()


def test_clips_are_captioned_from_the_trained_span_with_speech(client, llm, tmp_path, monkeypatch):
    import re

    from dainamo.server import captioner

    url, got, replies = llm
    _clip(tmp_path / "talk.mp4")
    _clip(tmp_path / "quiet.mp4", audio=False)
    heard = []

    class FakeWhisper:  # stands in for the subprocess: records what it was asked, answers like it
        def __init__(self, cmd, **kw):
            heard.append(cmd[3:])
            lines = [{"event": "loading"}, {"event": "ready", "device": "cuda"}]
            lines += [{"path": p, "text": "come on, let's go"} for p in cmd[5:]]
            self.stdout = iter(json.dumps(x) + "\n" for x in lines)

        def poll(self):
            return 0

        def wait(self):
            return 0

    import json
    monkeypatch.setattr(captioner, "_start_whisper", FakeWhisper)
    client.put("/api/settings", json={"captioner": {"endpoint": f"{url}/v1", "video_frames": 4}})
    did = client.post("/api/datasets", json={"name": "clips", "path": str(tmp_path)}).json()["id"]
    try:
        n = len(got)
        opts, status = _run_caption(client, did, clip_frames=24, speech="tiny")
        assert opts["speech"] == "tiny" and status["errors"] == [] and status["to_hear"] == 1
        assert heard == [["tiny", "1.0000", str(tmp_path / "talk.mp4")]]  # the clip without sound isn't sent
        asks = {g["body"]["messages"][0]["content"][0]["text"]: g["body"]["messages"][0]["content"] for g in got[n:]}
        talk = next(t for t in asks if "come on" in t)
        quiet = next(t for t in asks if t is not talk)
        assert len(asks[talk]) == 5  # the text and 4 frames
        assert "4 frames from the first 1.0 seconds of a video clip, in order, at 0.1, 0.4, 0.6, 0.9 seconds" in talk
        assert "\"come on, let's go\"" in talk and "speech" not in quiet.split("seconds.")[-1].lower()
        times = [float(t) for t in re.search(r"at ([\d., ]+) seconds", quiet).group(1).split(", ")]
        assert all(0 <= t < 1.0 for t in times)  # only the trained span

        # a model that takes one image: the frames go as one numbered contact sheet
        client.put("/api/settings", json={"captioner": {"endpoint": f"{url}/oneimage/v1"}})
        n = len(got)
        _, status = _run_caption(client, did, speech="")
        assert status["errors"] == [] and status["to_hear"] == 0
        sent = [g["body"]["messages"][0]["content"] for g in got[n:]]
        sheets = [c for c in sent if len(c) == 2 and "grid of 4 frames" in c[0]["text"]]
        assert len(sheets) == 2 and sum(1 for c in sent if len(c) == 5) == 2  # each clip tried frames first
        d = next(x for x in client.get("/api/datasets").json() if x["id"] == did)
        assert d["caption_options"] == {"style": "detailed", "clip_frames": 24, "speech": ""}  # remembered
        assert client.post(f"/api/datasets/{did}/autocaption", json={"speech": "huge"}).status_code == 400
        assert client.post(f"/api/datasets/{did}/autocaption", json={"clip_frames": 999}).status_code == 400
    finally:
        client.put("/api/settings", json={"captioner": {"endpoint": "http://127.0.0.1:8080/v1", "video_frames": 6}})


def test_contact_sheet_and_speech_settings(client):
    from PIL import Image

    from dainamo.engine.transcribe import is_stock_phrase
    from dainamo.server.captioner import contact_sheet

    frames = [Image.new("RGB", (160, 90), (i * 40, 0, 0)) for i in range(6)]
    sheet = contact_sheet(frames, [0.1 * i for i in range(6)])
    assert sheet.size == (2 * 384 + 4, 3 * 216 + 2 * 4)  # 16:9 frames: 2 across, 3 down is nearest square
    assert is_stock_phrase(" Thank you for watching! ") and is_stock_phrase("you") and not is_stock_phrase("you there")
    assert client.put("/api/settings", json={"speech": {"model": "huge"}}).status_code == 400
    assert client.put("/api/settings", json={"speech": {"device": "tpu"}}).status_code == 400
    assert client.put("/api/settings", json={"captioner": {"video_mode": "gif"}}).status_code == 400
    assert client.put("/api/settings", json={"speech": {"model": "base", "device": "cpu"}}).json()["speech"] == \
        {"model": "base", "device": "cpu"}
    client.put("/api/settings", json={"speech": {"model": "large-v3-turbo", "device": "auto"}})
    system = client.get("/api/system").json()
    assert [m["id"] for m in system["whisper_models"]][:2] == ["tiny", "base"]
    assert 0 < system["ram"]["available_gb"] <= system["ram"]["total_gb"]  # for the RAM check beside the VRAM one


LLAMA_OFF = {"binary": "", "model": "", "mmproj": "", "gpu": "auto", "ctx_size": 0, "extra_args": "", "idle_minutes": 10}
ENCODER_OFF = {"path": "", "gpu": "auto", "idle_minutes": 10}
NO_LOCAL_LLM = {"captioner": {"provider": "server"}, "llama": LLAMA_OFF, "text_encoder": ENCODER_OFF}


def test_llama_server_settings_are_validated(client, fake_llama):
    from dainamo.server.llama_server import auto_context
    from dainamo.server import db

    settings = client.get("/api/settings").json()
    assert settings["llama"] == LLAMA_OFF and settings["captioner"]["provider"] == "server"
    for bad, said in (({"model": "/nope/m.gguf"}, "wasn't found"), ({"model": fake_llama["binary"]}, "must be a .gguf file"),
                      ({"gpu": "first"}, "'auto', 'cpu'"), ({"ctx_size": 100}, "2048"),
                      ({"extra_args": "--host 0.0.0.0"}, "can't set --host"),
                      ({"binary": "/nope/llama-server"}, "wasn't found"),
                      ({"idle_minutes": "soon"}, "llama-server's idle minutes must be a number")):
        r = client.put("/api/settings", json={"llama": bad})
        assert r.status_code == 400 and said in r.json()["detail"], bad
    r = client.put("/api/settings", json={"captioner": {"provider": "llama"}})
    assert r.status_code == 400 and r.json()["detail"] == "Choose a model (.gguf) for llama-server, or another caption provider"
    try:
        r = client.put("/api/settings", json={"captioner": {"provider": "llama"},
                                              "llama": {**fake_llama, "gpu": 1, "ctx_size": "8192",
                                                        "extra_args": " -ngl 20 "}}).json()
        llama = r["llama"]
        assert llama["gpu"] == "1" and llama["ctx_size"] == 8192 and llama["extra_args"] == "-ngl 20"
        assert r["captioner"]["provider"] == "llama"
        status = client.get("/api/llm").json()
        assert status["state"] == "off" and status["auto_ctx"] == auto_context(db.get_settings())
        # a provider needs its model: clearing it is refused while it's chosen
        r = client.put("/api/settings", json={"llama": {"model": ""}})
        assert r.status_code == 400 and "Choose a model" in r.json()["detail"]
    finally:
        client.put("/api/settings", json=NO_LOCAL_LLM)


def test_auto_caption_and_enhance_start_the_llama_server_dainamo_runs(client, fake_llama, tmp_path, monkeypatch):
    from PIL import Image

    from dainamo.server.llama_server import llama

    for name in ("a", "b"):
        Image.new("RGB", (32, 32)).save(tmp_path / f"{name}.png")
    did = client.post("/api/datasets", json={"name": "managed", "path": str(tmp_path), "trigger": ""}).json()["id"]
    krea = client.post("/api/profiles", json={"name": "mk", "family": "krea2", "data": {}}).json()["id"]
    client.put("/api/settings", json={"captioner": {"provider": "llama"}, "llama": {**fake_llama, "gpu": "cpu"}})
    try:
        assert client.get("/api/llm").json()["state"] == "off"
        _, status = _run_caption(client, did)
        assert status["errors"] == [] and status["done"] == 2
        assert (tmp_path / "a.txt").read_text().startswith("A caption from Qwen2.5-VL-7B-Q4_K_M.gguf with 1 image(s)")
        s = client.get("/api/llm").json()
        assert s["state"] == "ready" and s["users"] == 0 and s["device"] == "the CPU" and s["provider"] == "llama"
        r = client.post("/api/sandbox/enhance", json={"profile_id": krea, "workflow": "t2i", "prompt": "a cat"})
        assert r.status_code == 200 and r.json()["prompt"].startswith("A caption from")
        assert any("server is listening" in line for line in client.get("/api/llm/log").json()["lines"])
        assert client.post("/api/llm/stop").json()["state"] in ("stopping", "off")
        _wait_for_state(client, "off")

        # a start that fails says why, in Auto-Caption and in the Sandbox
        monkeypatch.setenv("FAKE_LLAMA_FAIL", "ggml_cuda_init: cudaMalloc failed: out of memory")
        _, status = _run_caption(client, did)
        assert len(status["errors"]) == 1 and "stopped while loading the model" in status["errors"][0]
        assert "The model doesn't fit" in status["errors"][0] and status["done"] == 0
        r = client.post("/api/sandbox/enhance", json={"profile_id": krea, "workflow": "t2i", "prompt": "a cat"})
        assert r.status_code == 502 and "out of memory" in r.json()["detail"]
        assert client.get("/api/llm").json()["state"] == "failed"

        monkeypatch.delenv("FAKE_LLAMA_FAIL")
        assert client.post("/api/llm/start").json()["state"] == "starting"  # Settings' Start button
        _wait_for_state(client, "ready")
    finally:
        llama.stop(wait=True)
        client.put("/api/settings", json=NO_LOCAL_LLM)


def _wait_for_state(client, state, timeout=10.0):
    import time

    end = time.time() + timeout
    while client.get("/api/llm").json()["state"] != state and time.time() < end:
        time.sleep(0.05)
    assert client.get("/api/llm").json()["state"] == state


def test_training_takes_the_gpu_from_the_llama_server(client, fake_llama, tmp_path, monkeypatch):
    import time

    from dainamo.server import db
    from dainamo.server.llama_server import llama
    from dainamo.server.queue import scheduler

    spawned = []
    monkeypatch.setattr(scheduler, "_spawn", lambda job, gpu: spawned.append(gpu))
    pid = client.post("/api/profiles", json={"name": "lg", "family": "krea2", "data": {}}).json()["id"]
    did = client.post("/api/datasets", json={"name": "lg", "path": str(tmp_path)}).json()["id"]
    client.put("/api/settings", json={"llama": {**fake_llama, "gpu": "7"}, "queue": {"gpus": [7]}})
    jid = None
    try:
        with llama.serving(db.get_settings()):  # e.g. an Auto-Caption is using it
            assert llama.occupies(7)
            jid = client.post("/api/jobs", json={"name": "lg", "profile_id": pid, "datasets": [{"id": did}],
                                                 "settings": {}}).json()["id"]
            scheduler.tick()
            assert spawned == [] and llama.alive() and llama.yielding == "a training job"
        end = time.time() + 10
        while not spawned and time.time() < end:
            time.sleep(0.05)
        assert spawned[0] == 7 and not llama.alive()
        assert llama.snapshot()["message"] == "Stopped: a training job needed GPU 7"
    finally:
        client.put("/api/settings", json={"queue": {"gpus": [99]}, "llama": LLAMA_OFF})
        if jid:
            client.post(f"/api/jobs/{jid}/action", json={"action": "stop"})
        llama.stop(wait=True)


def test_text_encoder_settings_are_validated(client, fake_encoder):
    krea, ltx = fake_encoder("krea"), fake_encoder("ltx")
    assert client.get("/api/settings").json()["text_encoder"] == ENCODER_OFF
    for bad, said in (({"path": ltx}, "LTX 2.5's text encoder"), ({"path": "/nope/te.safetensors"}, "wasn't found"),
                      ({"gpu": "cpu"}, "The text encoder runs on a GPU: 'auto' or a GPU's number"),
                      ({"gpu": "first"}, "runs on a GPU"), ({"idle_minutes": "soon"}, "must be a number")):
        r = client.put("/api/settings", json={"text_encoder": bad})
        assert r.status_code == 400 and said in r.json()["detail"], bad
    for bad, said in (({"provider": "encoder"}, "Choose a text encoder file, or another caption provider"),
                      ({"provider": "ollama"}, "Unknown caption provider 'ollama'")):
        r = client.put("/api/settings", json={"captioner": bad})
        assert r.status_code == 400 and r.json()["detail"] == said
    # all or none: a bad section saves nothing
    r = client.put("/api/settings", json={"captioner": {"provider": "encoder"}, "text_encoder": {"path": krea},
                                          "llama": {"model": "/nope/m.gguf"}})
    assert r.status_code == 400 and client.get("/api/settings").json()["text_encoder"]["path"] == ""
    try:
        r = client.put("/api/settings", json={"captioner": {"provider": "encoder"},
                                              "text_encoder": {"path": f" {krea} ", "gpu": 1}}).json()
        assert r["captioner"]["provider"] == "encoder" and r["text_encoder"] == {**ENCODER_OFF, "path": krea, "gpu": "1"}
        # the Auto-Caption dialog's check: Dainamo runs it, so it's described, not asked
        assert client.post("/api/captioner/test", json={}).json() == {
            "ok": True, "managed": True, "models": ["Qwen3-VL-4B"],
            "message": "Dainamo runs Qwen3-VL-4B (Krea 2's text encoder): it starts when captioning starts"}
        # with the dialog's own endpoint override, but still this provider
        assert client.post("/api/captioner/test", json={"endpoint": "http://127.0.0.1:1/v1"}).json()["ok"]
    finally:
        client.put("/api/settings", json=NO_LOCAL_LLM)


def test_text_encoders_are_identified_for_the_settings_page(client, fake_encoder, tmp_path):
    krea, ltx = fake_encoder("krea"), fake_encoder("ltx")
    assert client.get("/api/llm/encoder", params={"path": krea}).json() == {
        "path": krea, "ok": True, "kind": "qwen3vl", "name": "Qwen3-VL-4B", "whose": "Krea 2", "vision": True}
    r = client.get("/api/llm/encoder", params={"path": ltx}).json()
    assert not r["ok"] and "LTX 2.5's text encoder" in r["error"]
    made = [client.post("/api/profiles", json={"name": name, "family": "krea2", "data": {"text_encoder": path}}).json()["id"]
            for name, path in (("te-b", krea), ("te-a", krea), ("te-ltx", ltx),
                               ("te-hub", "Comfy-Org/krea2/text_encoders/qwen3vl_4b.safetensors"))]
    try:
        found = {e["path"]: e for e in client.get("/api/llm/encoders").json()}
        assert found[krea]["profiles"] == ["te-a", "te-b"] and found[krea]["name"] == "Qwen3-VL-4B"
        assert found[ltx]["profiles"] == ["te-ltx"] and not found[ltx]["ok"]
        assert not any(p.startswith("Comfy-Org") for p in found)  # not a local file
    finally:
        for pid in made:
            client.delete(f"/api/profiles/{pid}")


def test_auto_caption_runs_the_text_encoder(client, fake_encoder, tmp_path, monkeypatch):
    from PIL import Image

    from dainamo.server import llama_server
    from dainamo.server.llama_server import llama

    monkeypatch.setattr(llama_server, "ENCODER_SERVER",
                        [sys.executable, os.path.join(os.path.dirname(__file__), "fake_llama_server.py")])
    Image.new("RGB", (32, 32)).save(tmp_path / "a.png")
    did = client.post("/api/datasets", json={"name": "te", "path": str(tmp_path), "trigger": ""}).json()["id"]
    r = client.put("/api/settings", json={"captioner": {"provider": "encoder"},
                                          "text_encoder": {"path": fake_encoder("krea"), "gpu": "3"}})
    assert r.status_code == 200, r.text
    try:
        _, status = _run_caption(client, did)
        assert status["errors"] == [] and status["done"] == 1
        assert (tmp_path / "a.txt").read_text().startswith("A caption from krea.safetensors with 1 image(s)")
        s = client.get("/api/llm").json()
        assert s["state"] == "ready" and s["provider"] == "encoder" and s["device"] == "GPU 3"
        assert s["model"] == "Qwen3-VL-4B" and s["what"] == "the text encoder"
        # another provider: the text encoder isn't needed any more
        client.put("/api/settings", json={"captioner": {"provider": "server"}})
        _wait_for_state(client, "off")
        assert client.get("/api/llm").json()["message"] == "Stopped: the caption provider changed"
    finally:
        llama.stop(wait=True)
        client.put("/api/settings", json=NO_LOCAL_LLM)


def test_jobs_carry_a_short_smoothed_loss_curve(client):
    import json
    import math

    from dainamo.server import db
    from dainamo.server.queue import job_dir
    from dainamo.server.routes import jobs as routes

    jid = db.new_id()
    db.execute("INSERT INTO jobs (id, name, status, family, gpu, position, created, config) VALUES (?,?,?,?,?,?,?,?)",
               (jid, "curve", "stopped", "krea2", "auto", db.now(), db.now(), "{}"))
    os.makedirs(job_dir(jid), exist_ok=True)
    path = os.path.join(job_dir(jid), "metrics.jsonl")

    def curve():
        return next(j for j in client.get("/api/jobs").json() if j["id"] == jid)["loss_curve"]

    def expected(losses):  # the Loss tab's smoothing, then 60 values evenly spaced
        ema, sm = None, []
        for _, v in sorted(losses.items()):
            if math.isfinite(v):
                ema = v if ema is None else 0.95 * ema + 0.05 * v
                sm.append(ema)
        n = min(60, len(sm))
        return [round(sm[round((len(sm) - 1) * i / (n - 1))], 5) for i in range(n)]

    def log(f, steps, loss):
        for s in steps:
            f.write(json.dumps({"step": s, "loss": loss(s), "lr": 1e-4}) + "\n")

    assert curve() == []  # nothing trained yet
    losses = {s: 1 / s for s in range(1, 201)}
    with open(path, "w") as f:
        log(f, range(1, 201), lambda s: 1 / s)
        f.write('{"step": 201, "lo')  # a line still being written
    assert curve() == expected(losses) and len(expected(losses)) == 60
    with open(path, "a") as f:
        f.write('ss": NaN, "lr": 0.0001}\n')  # a NaN step doesn't bend the curve
        log(f, range(202, 301), lambda s: 0.5)
        log(f, range(150, 160), lambda s: 0.25)  # a resumed run logs steps again: the last value counts
    losses.update({s: 0.5 for s in range(202, 301)} | {s: 0.25 for s in range(150, 160)})
    assert curve() == expected(losses)
    routes._curves.clear()
    assert curve() == expected(losses)  # read in pieces = read at once

    with open(path, "w") as f:  # started over from step 0, and already longer than before
        log(f, range(1, 400), lambda s: 2.0 + (s % 7) / 10)
    assert curve() == expected({s: 2.0 + (s % 7) / 10 for s in range(1, 400)})
    with open(path, "w") as f:
        log(f, range(1, 6), lambda s: 1.0)
    assert curve() == [1.0] * 5

    client.delete(f"/api/jobs/{jid}")
    client.get("/api/jobs")
    assert jid not in routes._curves

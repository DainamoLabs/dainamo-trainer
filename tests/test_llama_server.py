"""The local LLM server Dainamo runs for the caption provider (dainamo/server/llama_server.py): llama-server, or
a model's text encoder (dainamo/engine/textgen.py), both played by a stand-in (tests/fake_llama_server.py).
Started when something needs it and ready once its model has loaded, a failed start explained, restarted for new
settings only when nothing uses it, and stopped when idle or when training or the Sandbox needs its GPU.

    .venv/bin/python -m pytest tests/test_llama_server.py -v
"""

import json
import os
import sys
import time

import httpx
import psutil
import pytest

from dainamo.server import captioner, db, llama_server
from dainamo.server.llama_server import LlamaError, LlamaServer, auto_context

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stand-in server is started by a shell script")


@pytest.fixture
def settings(fake_llama):
    s = json.loads(json.dumps(db.DEFAULT_SETTINGS))
    s["captioner"]["provider"] = "llama"
    s["llama"].update(gpu="cpu", **fake_llama)
    return s


@pytest.fixture
def encoder(fake_encoder, monkeypatch):
    """Settings for Krea 2's text encoder (a header-only copy) on GPU 1, with the stand-in as its server."""
    from dainamo.server import sandbox

    monkeypatch.setattr(llama_server, "ENCODER_SERVER",
                        [sys.executable, os.path.join(os.path.dirname(__file__), "fake_llama_server.py")])
    monkeypatch.setattr(sandbox, "free_gpu", lambda: 1)
    s = json.loads(json.dumps(db.DEFAULT_SETTINGS))
    s["captioner"]["provider"] = "encoder"
    s["text_encoder"]["path"] = fake_encoder("krea")
    return s


@pytest.fixture
def server():
    srv = LlamaServer()
    yield srv
    srv.stop(wait=True)


def arg(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def wait_until(cond, timeout=10.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.05)
    return cond()


def test_starts_when_needed_and_answers_once_its_model_has_loaded(server, settings, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_LOAD", "0.6")  # /health answers 503 until then
    heard = []
    with server.serving(settings, heard.append) as endpoint:
        assert server.state == "ready" and server.users == 1 and server.load_seconds >= 0.6
        port = server.port
        assert endpoint == f"http://127.0.0.1:{port}/v1" and port >= llama_server.FIRST_PORT  # not 8080
        cmd = server.command
        assert arg(cmd, "--model") == settings["llama"]["model"] and arg(cmd, "--mmproj") == settings["llama"]["mmproj"]
        # room for four images and the prompt, plus a 600-token caption asked for again with 4x the room
        assert arg(cmd, "--ctx-size") == str(auto_context(settings)) == "11264"
        assert arg(cmd, "--parallel") == "1" and arg(cmd, "--n-gpu-layers") == "0"  # the CPU
        assert psutil.Process(server.proc.pid).environ()["CUDA_VISIBLE_DEVICES"] == ""
        assert heard and all(h.startswith("Starting llama-server: loading Qwen2.5-VL-7B-Q4_K_M.gguf on the CPU")
                             for h in heard)
        with httpx.Client() as client:
            text = captioner.chat({"endpoint": endpoint}, client, [{"role": "user", "content": "hi"}], 100, 0.2)
        assert text.startswith("A caption from Qwen2.5-VL-7B-Q4_K_M.gguf")
    assert server.users == 0 and server.alive()
    pid = server.proc.pid
    with server.serving(settings) as again:  # running already: used as it is
        assert again == endpoint and server.proc.pid == pid
    snap = server.snapshot()
    assert snap["state"] == "ready" and snap["device"] == "the CPU" and snap["endpoint"] == endpoint
    assert snap["message"].startswith("Ready on the CPU: loaded in")
    assert any("server is listening" in line for line in server.log)
    with open(llama_server.LOG_PATH) as f:
        assert "--ctx-size 11264" in f.read()
    with open(llama_server.PID_PATH) as f:
        assert f.read() == f"{pid} {port}"
    server.stop(wait=True)
    assert wait_until(lambda: server.snapshot()["state"] == "off" and not os.path.exists(llama_server.PID_PATH))
    assert server.snapshot()["message"] == "Stopped"


def test_auto_takes_a_free_gpu_else_the_cpu(server, settings, monkeypatch):
    from dainamo.server import sandbox

    settings["llama"]["gpu"] = "auto"
    monkeypatch.setattr(sandbox, "free_gpu", lambda: 1)
    with server.serving(settings):
        assert server.gpu == 1 and arg(server.command, "--n-gpu-layers") == "999"
        assert psutil.Process(server.proc.pid).environ()["CUDA_VISIBLE_DEVICES"] == "1"
    server.stop(wait=True)
    monkeypatch.setattr(sandbox, "free_gpu", lambda: None)
    monkeypatch.setattr(sandbox, "gpu_count", lambda: 2)
    server.start(settings)
    assert server.gpu is None and arg(server.command, "--n-gpu-layers") == "0"
    assert server.message.endswith("on the CPU (every GPU is training or running the Sandbox)")
    server.stop(wait=True)
    # a chosen GPU that's training is left to the job
    settings["llama"]["gpu"] = "1"
    monkeypatch.setattr(sandbox, "training_gpus", lambda: {1})
    server.start(settings)
    assert server.gpu is None and server.message.endswith("on the CPU (GPU 1 is training)")
    server.stop(wait=True)
    monkeypatch.setattr(sandbox, "training_gpus", lambda: set())
    # a CPU-only llama.cpp build runs on the CPU, and says so, whatever GPU was asked for
    monkeypatch.setenv("FAKE_LLAMA_NO_GPU", "1")
    settings["llama"]["gpu"] = "0"
    with server.serving(settings):
        assert server.gpu is None and not server.occupies(0) and arg(server.command, "--n-gpu-layers") == "0"
        assert psutil.Process(server.proc.pid).environ()["CUDA_VISIBLE_DEVICES"] == ""
        assert server.snapshot()["message"].startswith("Ready on the CPU (this llama-server has no GPU support: a "
                                                       "CUDA build of llama.cpp runs on the GPU): loaded in")


@pytest.mark.parametrize("line, said", [
    ("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000.00 MiB on device 0: cudaMalloc failed: out of memory",
     "The model doesn't fit"),
    ("error: invalid argument: --flash-attn-bogus", "doesn't know one of the extra arguments"),
    ("llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'qwen9'",
     "too old for the model"),
])
def test_a_failed_start_says_why(server, settings, monkeypatch, line, said):
    monkeypatch.setenv("FAKE_LLAMA_FAIL", line)
    with pytest.raises(LlamaError) as e:
        with server.serving(settings):
            pass
    assert str(e.value).startswith("llama-server stopped while loading the model: ") and line in str(e.value)
    assert said in str(e.value) and server.users == 0
    snap = server.snapshot()
    assert snap["state"] == "failed" and line in snap["error"] and said in snap["hint"]
    monkeypatch.delenv("FAKE_LLAMA_FAIL")
    with server.serving(settings):  # the next request tries again
        assert server.snapshot()["error"] == ""


def test_missing_pieces_are_named(server, settings, monkeypatch):
    def fails(**changes):
        cfg = {**settings, "llama": {**settings["llama"], **changes}}
        with pytest.raises(LlamaError) as e:
            server.start(cfg)
        assert server.snapshot()["state"] == "failed" and not server.alive()
        return str(e.value)

    assert "wasn't found at /nope/llama-server" in fails(binary="/nope/llama-server")
    assert "isn't a program" in fails(binary=settings["llama"]["model"])
    monkeypatch.setattr(llama_server.shutil, "which", lambda name: None)
    assert "wasn't found on the PATH" in fails(binary="")
    assert "Choose a model" in fails(model="")
    assert "The model wasn't found" in fails(model="/nope/model.gguf")
    assert "vision projector wasn't found" in fails(mmproj="/nope/mmproj.gguf")
    assert "can't set --port" in fails(extra_args="-ngl 20 --port 9000")
    assert "Extra arguments" in fails(extra_args='--alias "unclosed')


def test_new_settings_restart_it_only_when_nothing_uses_it(server, settings):
    bigger = {**settings, "llama": {**settings["llama"], "ctx_size": 16384}}
    with server.serving(settings):
        pid = server.proc.pid
        with server.serving(bigger) as endpoint:  # in use: the running one serves
            assert server.proc.pid == pid and arg(server.command, "--ctx-size") == "11264"
            assert endpoint == server.endpoint()
    with server.serving(bigger):
        assert server.proc.pid != pid and arg(server.command, "--ctx-size") == "16384"
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_stops_when_idle_but_not_while_in_use(server, settings, monkeypatch):
    settings["llama"]["idle_minutes"] = 0.01  # 0.6 s
    monkeypatch.setattr(llama_server.db, "get_settings", lambda: settings)
    with server.serving(settings):
        time.sleep(1)
        server.tick()
        assert server.state == "ready"  # a long caption job isn't idle time
    server.tick()
    assert server.state == "ready"  # just used
    time.sleep(0.7)
    server.tick()
    assert server.state == "stopping" and wait_until(lambda: server.snapshot()["state"] == "off")
    assert server.snapshot()["message"] == "Stopped after 0.01 minutes idle"


def test_training_takes_its_gpu_once_captioning_is_done(server, settings):
    settings["llama"]["gpu"] = "0"
    with server.serving(settings):
        assert server.occupies(0) and not server.occupies(1)
        server.yield_gpu()
        assert server.alive() and server.message == "A training job is waiting for GPU 0: stopping once it's done here"
    assert server.state == "stopping" and wait_until(lambda: not server.alive())
    assert server.snapshot()["message"] == "Stopped: a training job needed GPU 0"
    server.start(settings)  # not in use: the Sandbox (or a job) gets the GPU right away
    server.yield_gpu("the Sandbox")
    assert server.state == "stopping" and server.message == "Stopped: the Sandbox needed GPU 0"


def test_one_left_by_a_crashed_server_is_stopped(server, settings):
    server.start(settings)
    server.wait_ready()
    proc, server.proc = server.proc, None  # the server crashed: a new one finds only the pid file
    try:
        LlamaServer().startup()
        assert wait_until(lambda: proc.poll() is not None) and not os.path.exists(llama_server.PID_PATH)
    finally:
        proc.kill()


def test_a_text_encoder_runs_on_a_free_gpu(server, encoder):
    heard = []
    path = encoder["text_encoder"]["path"]
    with server.serving(encoder, heard.append) as endpoint:
        assert server.command[2:] == ["--model", path, "--host", "127.0.0.1", "--port", str(server.port)]
        assert server.gpu == 1 and psutil.Process(server.proc.pid).environ()["CUDA_VISIBLE_DEVICES"] == "1"
        assert heard and all(h.startswith("Starting the text encoder: loading Qwen3-VL-4B on GPU 1") for h in heard)
        with httpx.Client() as client:
            text = captioner.chat({"endpoint": endpoint}, client, [{"role": "user", "content": "hi"}], 100, 0.2)
        assert text == "A caption from krea.safetensors with 0 image(s), ctx none."
    snap = server.snapshot()
    assert snap["provider"] == "encoder" and snap["what"] == "the text encoder" and snap["model"] == "Qwen3-VL-4B"
    assert snap["message"].startswith("Ready on GPU 1: loaded in")
    with open(llama_server.LOG_PATH) as f:
        assert ": the text encoder on GPU 1 =====" in f.read()


def test_a_text_encoder_needs_a_gpu(server, encoder, monkeypatch):
    from dainamo.server import sandbox

    def fails(**changes):
        with pytest.raises(LlamaError) as e:
            server.start({**encoder, "text_encoder": {**encoder["text_encoder"], **changes}})
        assert server.snapshot()["state"] == "failed" and not server.alive()
        return str(e.value)

    assert fails(gpu="cpu") == "The text encoder runs on a GPU: choose one in Settings → Caption provider"
    monkeypatch.setattr(sandbox, "training_gpus", lambda: {1})
    assert "but GPU 1 is training right now: try again later, or use llama.cpp" in fails(gpu="1")
    monkeypatch.setattr(sandbox, "free_gpu", lambda: None)
    monkeypatch.setattr(sandbox, "gpu_count", lambda: 0)
    assert "none was found: use llama.cpp, which can run on the CPU" in fails()
    assert "Choose a text encoder file" in fails(path="")


def test_a_text_encoder_that_cant_write_isnt_started(server, encoder, fake_encoder):
    with pytest.raises(LlamaError, match="LTX 2.5's text encoder"):
        server.start({**encoder, "text_encoder": {**encoder["text_encoder"], "path": fake_encoder("ltx")}})
    assert server.snapshot()["state"] == "failed" and not server.alive()


def test_a_text_encoder_that_doesnt_fit_says_so(server, encoder, monkeypatch):
    # how the text-encoder server ends a failed load: the traceback, then its own error line
    monkeypatch.setenv("FAKE_LLAMA_FAIL", "Traceback (most recent call last):\n  File \"textgen.py\", line 1, in load\n"
                       "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.\n"
                       "error: CUDA out of memory. Tried to allocate 1.16 GiB.")
    with pytest.raises(LlamaError) as e:
        with server.serving(encoder):
            pass
    assert str(e.value) == ("The text encoder stopped while loading the model: error: CUDA out of memory. Tried to "
                            "allocate 1.16 GiB. The text encoder doesn't fit in this GPU's free memory: pick a GPU "
                            "with more free, or a smaller encoder (Krea 2's Qwen3-VL-4B), or use llama.cpp with a "
                            "small GGUF model.")
    assert server.snapshot()["provider"] == "encoder"


def test_switching_provider_restarts_it_when_unused(server, settings, encoder):
    with server.serving(settings):
        pid = server.proc.pid
        with server.serving(encoder) as endpoint:  # in use: llama-server answers until it's done
            assert server.proc.pid == pid and server.provider == "llama" and endpoint == server.endpoint()
    with server.serving(encoder):
        assert server.proc.pid != pid and server.provider == "encoder" and server.model == "Qwen3-VL-4B"


def test_idle_time_comes_from_the_providers_settings(server, encoder, monkeypatch):
    encoder["text_encoder"]["idle_minutes"] = 0.01
    encoder["llama"]["idle_minutes"] = 0  # never: not the one running
    monkeypatch.setattr(llama_server.db, "get_settings", lambda: encoder)
    server.start(encoder)
    server.wait_ready()
    time.sleep(0.7)
    server.tick()
    assert server.state == "stopping" and server.message == "Stopped after 0.01 minutes idle"

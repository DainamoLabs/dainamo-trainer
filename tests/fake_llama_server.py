"""A stand-in for llama-server (and Dainamo's text-encoder server, which takes fewer of its arguments), for
tests/test_llama_server.py and tests/test_server.py: it takes llama-server's arguments, answers
/health with 503 while it "loads" (FAKE_LLAMA_LOAD seconds), then serves /health, /v1/models and
/v1/chat/completions on --port. With FAKE_LLAMA_FAIL it prints that line and exits with code 1 instead.
--list-devices lists a GPU, or none with FAKE_LLAMA_NO_GPU (a CPU-only build)."""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
if args == ["--list-devices"]:  # a CPU-only build (FAKE_LLAMA_NO_GPU) lists none
    gpu = "  (none)" if os.environ.get("FAKE_LLAMA_NO_GPU") else "  CUDA0: Fake GPU (24000 MiB, 23000 MiB free)"
    print(f"Available devices:\n{gpu}")
    sys.exit(0)
opt = {a: b for a, b in zip(args, args[1:]) if a.startswith("-")}
print(f"build: fake llama-server, args: {' '.join(args)}", flush=True)
if os.environ.get("FAKE_LLAMA_FAIL"):
    print("llama_model_loader: loaded meta data with 30 key-value pairs", flush=True)
    print(os.environ["FAKE_LLAMA_FAIL"], flush=True)
    sys.exit(1)
loaded_at = time.time() + float(os.environ.get("FAKE_LLAMA_LOAD", "0.5"))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        out = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if time.time() < loaded_at:
            self._send(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        elif self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": os.path.basename(opt["--model"])}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        content = body["messages"][-1]["content"]
        images = sum(1 for c in content if c.get("type") == "image_url") if isinstance(content, list) else 0
        print(f"slot launch_slot_: request with {images} image(s)", flush=True)
        reply = f"A caption from {os.path.basename(opt['--model'])} with {images} image(s), ctx {opt.get('--ctx-size', 'none')}."
        self._send(200, {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 50, "completion_tokens": 12}})

    def log_message(self, *a):
        pass


srv = HTTPServer((opt["--host"], int(opt["--port"])), Handler)
print(f"main: server is listening on http://{opt['--host']}:{opt['--port']}", flush=True)
srv.serve_forever()

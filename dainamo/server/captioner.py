"""The local LLM (an OpenAI-compatible endpoint: a server you run, or the llama-server or text encoder Dainamo
runs, see ``llama_server``): auto-captioning datasets and enhancing Sandbox prompts."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..engine.dataset import apply_trigger, list_media, media_kind, read_caption, caption_path
from .llama_server import LlamaError, llama, local_llm, provider


FPS = 24  # clips are trained at 24 fps: "the first N frames" is the first N / 24 seconds
FALLBACK_CODES = (400, 413, 422, 500)  # how servers refuse several images in one message
MAX_TOKENS = 8192  # the most Settings allows
RETRY_ROOM = 4  # a reply cut off at max_tokens is asked for again with this many times the room
CAPTION_TIMEOUT = 300  # seconds a caption may take


def _encode_image(img: Image.Image, max_side: int = 1024) -> str:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def video_frames(path: str, count: int, span: float | None = None) -> tuple[list[Image.Image], list[float]]:
    """``count`` frames spread over the first ``span`` seconds (or the whole clip), and when each was taken."""
    try:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "default=nw=1:nk=1", path], capture_output=True, text=True, timeout=30).stdout)
    except (ValueError, subprocess.SubprocessError, FileNotFoundError):
        dur = 0.0
    span = min(span, dur) if span and dur else span or dur
    frames, times = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(max(1, count)):
            ts = span * (i + 0.5) / max(1, count) if span else 0
            out = os.path.join(tmp, f"{i}.jpg")
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ts:.3f}", "-i", path, "-frames:v", "1", "-y", out],
                               capture_output=True, timeout=60)
            except (subprocess.SubprocessError, FileNotFoundError):
                break  # ffmpeg missing or stuck
            if os.path.exists(out):
                with Image.open(out) as im:
                    frames.append(im.copy())
                times.append(ts)
    return frames, times


def contact_sheet(frames: list[Image.Image], times: list[float], tile: int = 384) -> Image.Image:
    """The frames in one image: a grid in reading order, each numbered with its time."""
    w, h = frames[0].size
    tw, th = max(1, round(w * tile / max(w, h))), max(1, round(h * tile / max(w, h)))
    n, gap = len(frames), 4
    cols = min(range(1, n + 1), key=lambda c: abs(math.log(c * tw / (math.ceil(n / c) * th))))  # near square
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=max(12, th // 11))
    except (TypeError, OSError):  # Pillow without FreeType
        font = ImageFont.load_default()
    for i, (im, t) in enumerate(zip(frames, times)):
        x, y = (i % cols) * (tw + gap), (i // cols) * (th + gap)
        sheet.paste(ImageOps.exif_transpose(im).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS), (x, y))
        label = f"{i + 1}  {t:.1f}s"
        box = draw.textbbox((x + 8, y + 6), label, font=font)
        draw.rectangle((box[0] - 5, box[1] - 4, box[2] + 5, box[3] + 4), fill="black")
        draw.text((x + 8, y + 6), label, fill="white", font=font)
    return sheet


def _server_error(r: httpx.Response) -> str:
    """What an LLM server said was wrong (llama.cpp, Ollama and OpenAI send {"error": {"message"}})."""
    try:
        err = r.json().get("error")
        msg = err.get("message") if isinstance(err, dict) else err
    except (ValueError, AttributeError):
        msg = None
    return " ".join(str(msg or r.text or r.reason_phrase).split())[:300]


def strip_thinking(text: str) -> str:
    """A reply without a reasoning model's thinking: a <think> block, or what comes before a lone </think>
    (Qwen3-style templates open the block in the prompt). Thinking that never finished leaves nothing."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text.split("<think>", 1)[0]


def chat(llm: dict, client: httpx.Client, messages: list[dict], max_tokens: int, temperature: float,
         **extra) -> str:
    """The text of the LLM's reply (``llm``: endpoint, model, api_key), without its thinking. A reply cut off at
    ``max_tokens`` (a reasoning model's thinking can use all of it) is asked for again with ``RETRY_ROOM``
    times the room. Raises httpx.HTTPStatusError with the server's own message, or RuntimeError when the
    reply is empty or still cut off."""
    headers = {"Authorization": f"Bearer {llm['api_key']}"} if llm.get("api_key") else {}
    tries = [max_tokens] + ([min(MAX_TOKENS, max_tokens * RETRY_ROOM)] if max_tokens < MAX_TOKENS else [])
    for room in tries:
        body = {"messages": messages, "max_tokens": room, "temperature": temperature, **extra}
        if llm.get("model"):
            body["model"] = llm["model"]
        r = client.post(llm["endpoint"].rstrip("/") + "/chat/completions", json=body, headers=headers)
        if r.status_code >= 400:
            raise httpx.HTTPStatusError(f"the LLM server answered {r.status_code}: {_server_error(r)}",
                                        request=r.request, response=r)
        data = r.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = strip_thinking(message.get("content") or "").strip()
        if choice.get("finish_reason") != "length":
            if not text:
                raise RuntimeError("the LLM returned no text")
            return text
        used = (data.get("usage") or {}).get("completion_tokens")
        if used is not None and used < room:  # stopped before max_tokens: its context is full
            prompt = (data.get("usage") or {}).get("prompt_tokens")
            raise RuntimeError(f"the reply was cut off after {used} tokens because the LLM's context is full"
                               + (f" (the request took {prompt})" if prompt else "")
                               + ": give the server a larger context, or send fewer frames")
    thinking = not text and (message.get("reasoning_content") or message.get("reasoning") or "<think>" in
                             (message.get("content") or ""))
    raise RuntimeError(f"the reply was cut off at {room} tokens"
                       + (", all of them spent thinking" if thinking else "")
                       + ": raise Max tokens in Settings")


def style_config(captioner: dict, style: dict) -> dict:
    """The caption provider's settings (Settings → captioner) with a caption style's prompt, and its system prompt,
    temperature and max_tokens where it has its own."""
    return {**captioner, "prompt": style["prompt"], "system": style.get("system") or "",
            **{k: style[k] for k in ("temperature", "max_tokens") if style.get(k) is not None}}


def _ask(cfg: dict, client: httpx.Client, prompt: str, images: list[Image.Image], max_side: int = 1024) -> str:
    """The reply to ``prompt`` with ``images``. With a system prompt (``cfg["system"]``) the images come before the
    text, as FLUX 3's prompt upsampler gets them; without one, the text comes first."""
    text = [{"type": "text", "text": prompt}]
    pictures = [{"type": "image_url", "image_url": {"url": _encode_image(im, max_side)}} for im in images]
    if cfg.get("system"):
        messages = [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": pictures + text}]
    else:
        messages = [{"role": "user", "content": text + pictures}]
    reply = chat(cfg, client, messages, int(cfg.get("max_tokens", 600)), float(cfg.get("temperature", 0.2)))
    return " ".join(reply.split())


def caption_file(path: str, cfg: dict, client: httpx.Client, span: float | None = None,
                 transcript: str | None = None) -> str:
    """A caption for an image, or for a clip: frames from its first ``span`` seconds (or all of it), sent
    together, or as one contact sheet for models that take one image (``video_mode`` "sheet", and the
    fallback when a server refuses several images). ``transcript`` is what Whisper heard ("": nothing;
    None: not transcribed)."""
    if media_kind(path) != "video":
        with Image.open(path) as im:
            return _ask(cfg, client, cfg["prompt"], [im.copy()])
    frames, times = video_frames(path, int(cfg.get("video_frames", 6)), span)
    if not frames:
        raise RuntimeError("could not read the clip's frames")
    part = f"the first {span:.1f} seconds of a video clip" if span else "a video clip"
    when = ", ".join(f"{t:.1f}" for t in times)
    after = " Describe it as one continuous clip: what happens, in order, and how the camera moves."
    if transcript:
        stop = "" if transcript[-1] in ".!?…" else "."
        after += (f' Speech heard in it (transcribed automatically, so it may have mistakes): "{transcript}"{stop} '
                  "Include what is said where it fits.")
    elif transcript is not None:
        after += " No speech is heard in it."
    if len(frames) == 1:
        return _ask(cfg, client, f"{cfg['prompt']}\n\nThis is a frame from {part}, at {when} seconds.{after}", frames)
    if cfg.get("video_mode") != "sheet":
        try:
            return _ask(cfg, client, f"{cfg['prompt']}\n\nThese are {len(frames)} frames from {part}, in order, "
                                     f"at {when} seconds.{after}", frames, max_side=512)
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in FALLBACK_CODES:
                raise
    sheet = contact_sheet(frames, times)
    return _ask(cfg, client, f"{cfg['prompt']}\n\nThis image is a grid of {len(frames)} frames from {part}, "
                             f"numbered in reading order (left to right, top to bottom), at {when} seconds.{after}",
                [sheet], max_side=1536)


def _clean_prompt(text: str) -> str:
    """The prompt from a model's reply: no reasoning block, label or wrapping quotes."""
    text = strip_thinking(text).strip()
    text = re.sub(r"^(here is|here's)[^:\n]*:\s*", "", text, flags=re.I)
    text = re.sub(r"^\**\s*(enhanced |improved |rewritten )?(prompt|instruction)\s*\**\s*:\s*\**\s*", "", text,
                  flags=re.I)
    text = " ".join(text.split())
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text.strip("“”").strip()


def enhance_prompt(prompt: str, instruction: str, images: list[str], llm: dict, cfg: dict,
                   seed: int | None = None) -> str:
    """Rewrite ``prompt`` following ``instruction``. Input images go along when there are any; a model
    that can't take images gets the text alone. Raises httpx errors, or RuntimeError for an empty reply."""
    def ask(with_images: bool) -> str:
        user = [{"type": "text", "text": prompt}]
        if with_images:
            for path in images:
                with Image.open(path) as im:
                    user.append({"type": "image_url", "image_url": {"url": _encode_image(im)}})
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": user if with_images else prompt}]
        # the same prompt and seed give the same rewrite, where the server supports it
        return chat(llm, client, messages, int(cfg.get("max_tokens", 400)), float(cfg.get("temperature", 0.7)),
                    **({"seed": int(seed)} if seed is not None else {}))

    with httpx.Client(timeout=180) as client:
        try:
            text = ask(bool(images))
        except httpx.HTTPStatusError:
            if not images:
                raise
            text = ask(False)  # e.g. a text-only model
    text = _clean_prompt(text)
    if not text:
        raise RuntimeError("The LLM returned no text.")
    return text


def test_endpoint(cfg: dict) -> dict:
    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
    r = httpx.get(cfg["endpoint"].rstrip("/") + "/models", headers=headers, timeout=10)
    r.raise_for_status()
    models = [m.get("id") for m in r.json().get("data", [])]
    return {"ok": True, "models": models}


def _start_whisper(cmd: list[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(cmd, **kw)


class CaptionJob:
    """Captions a dataset's files in a thread. With ``speech`` (a Whisper model), clips with sound are
    transcribed first, in a subprocess on a free GPU (or the CPU), and what's said goes into their captions.
    ``clip_frames``: describe (and transcribe) only the first that many frames of each clip; 0 for all.
    ``settings``: the app's, for starting the server Dainamo runs when Settings say so."""

    def __init__(self, dataset: dict, cfg: dict, overwrite: bool, trigger: str, clip_frames: int = 0,
                 speech: str = "", speech_device: str = "auto", settings: dict | None = None):
        self.dataset = dataset
        self.cfg = cfg
        self.settings = settings
        self.overwrite = overwrite
        self.trigger = trigger
        self.span = clip_frames / FPS if clip_frames else None
        self.speech = speech
        self.speech_device = speech_device
        self.total = 0
        self.done = 0
        self.phase = "captioning"  # or "transcribing", or "starting" (the server Dainamo runs)
        self.heard = 0
        self.to_hear = 0
        self.message = ""
        self.errors: list[str] = []
        self.running = True
        self.cancelled = False
        self.current = ""
        self.proc: subprocess.Popen | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def to_json(self) -> dict:
        return {"total": self.total, "done": self.done, "errors": self.errors[-20:], "running": self.running,
                "current": os.path.basename(self.current), "phase": self.phase, "heard": self.heard,
                "to_hear": self.to_hear, "message": self.message}

    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def _run(self):
        try:
            files = [f for f in list_media(self.dataset["path"]) if media_kind(f) != "audio"]  # it can't hear music
            if not self.overwrite:
                files = [f for f in files if not read_caption(f)]
            self.total = len(files)
            heard = self._transcribe([f for f in files if media_kind(f) == "video"]) if self.speech else {}
            if not files or self.cancelled:
                return
            managed = bool(self.settings) and provider(self.settings) != "server"
            self.phase, self.message = "starting" if managed else "captioning", ""
            llm = local_llm(self.settings, self.cfg, self._status, lambda: self.cancelled) if self.settings else \
                contextlib.nullcontext(self.cfg)
            with llm as cfg, httpx.Client(timeout=CAPTION_TIMEOUT) as client:
                self.phase, self.message = "captioning", ""
                for f in files:
                    if self.cancelled:
                        break
                    self.current = f
                    if not self.overwrite and read_caption(f):  # captioned by hand since the run began: kept
                        self.done += 1
                        continue
                    try:
                        cap = apply_trigger(caption_file(f, cfg, client, self.span, heard.get(f)), self.trigger)
                        if not self.overwrite and read_caption(f):  # or while the LLM wrote this one
                            self.done += 1
                            continue
                        with open(caption_path(f), "w", encoding="utf-8") as fh:
                            fh.write(cap)
                    except httpx.ConnectError:
                        self.errors.append(llama.lost() if managed else
                                           f"Couldn't reach the LLM at {cfg['endpoint']}; is it running?")
                        break  # endpoint down; don't spam errors
                    except httpx.TimeoutException:
                        self.errors.append(f"{os.path.basename(f)}: the LLM didn't answer within {CAPTION_TIMEOUT} s")
                    except Exception as e:
                        self.errors.append(f"{os.path.basename(f)}: {e}")
                    self.done += 1
        except LlamaError as e:
            if not self.cancelled:
                self.errors.append(str(e))
        except Exception as e:  # anything else ends the run, said in the UI rather than only the server's console
            traceback.print_exc()
            if not self.cancelled:
                self.errors.append(f"{type(e).__name__}: {e}")
        finally:
            self.running = False
            self.current = ""

    def _status(self, message: str):
        self.message = message

    def _transcribe(self, clips: list[str]) -> dict[str, str]:
        """{clip: what Whisper heard} for the clips with sound; a clip it couldn't do is left out."""
        from ..engine.dataset import probe_video
        from ..engine.transcribe import MODELS
        from . import db, procs
        from .sandbox import free_gpu

        with_sound = []
        for clip in clips:
            try:
                if probe_video(clip)["has_audio"]:
                    with_sound.append(clip)
            except Exception:
                pass  # captioning reports a clip it can't read
        if not with_sound or self.cancelled:
            return {}
        self.phase, self.to_hear = "transcribing", len(with_sound)
        self.message = f"Loading Whisper {self.speech} (a {MODELS[self.speech][1]} download the first time)"
        gpu = None if self.speech_device == "cpu" else free_gpu()
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        log_path = os.path.join(db.DATA_DIR, "transcribe.log")
        heard: dict[str, str] = {}
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n===== {self.dataset['name']}: {len(with_sound)} clip(s), Whisper {self.speech} =====\n")
            log.flush()
            self.proc = _start_whisper(
                [sys.executable, "-m", "dainamo.engine.transcribe", self.speech, f"{self.span or 0:.4f}", *with_sound],
                cwd=root, env=procs.worker_env(gpu), stdout=subprocess.PIPE, stderr=log, text=True,
                encoding="utf-8")
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") == "ready":
                self.message = f"Whisper {self.speech} on " + ("the CPU" if msg.get("device") == "cpu" else f"GPU {gpu}")
            elif "path" in msg:
                self.heard += 1
                if "error" in msg:
                    self.errors.append(f"{os.path.basename(msg['path'])}: speech: {msg['error']}")
                else:
                    heard[msg["path"]] = msg["text"]
        code = self.proc.wait()
        if code != 0 and not self.cancelled:
            self.errors.append(f"Whisper stopped (exit code {code}), so these captions have no speech; "
                               f"see {log_path}")
        return heard


caption_jobs: dict[str, CaptionJob] = {}

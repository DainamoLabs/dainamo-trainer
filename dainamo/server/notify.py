"""Notifications when a training job ends, sent to a webhook: ntfy, Discord, Slack or plain JSON.

Settings (``notifications``): ``url``, ``kind``, ``on_completed`` / ``on_failed`` / ``on_stopped``, and
``link``, the address this app is reached at, for a link to the job in the message."""

from __future__ import annotations

import os
import sys
import threading
import time

import httpx

from . import db
from .queue import job_dir, read_status

KINDS = ("ntfy", "discord", "slack", "json")
EVENTS = {"completed": "on_completed", "failed": "on_failed", "stopped": "on_stopped"}
TITLES = {"completed": "finished", "failed": "failed", "stopped": "stopped", "interrupted": "was interrupted"}
NTFY_TAGS = {"completed": "white_check_mark", "failed": "x", "stopped": "stop_button", "interrupted": "warning"}

last: dict = {}  # the latest delivery: {at, ok, error, title}


def job_ended(job_id: str, status: str, exit_code: int | None = None) -> None:
    """Called by the queue when a job ends; sends in the background if that's enabled for ``status``."""
    cfg = db.get_settings()["notifications"]
    event = "failed" if status == "interrupted" else status
    if not cfg.get("url") or not cfg.get(EVENTS.get(event, ""), False):
        return
    msg = message(job_id, status, exit_code, cfg.get("link", ""))
    threading.Thread(target=_send_logged, args=(cfg, msg), daemon=True).start()


def evaluation_ended(job_id: str, state: str | None) -> None:
    """Called by the queue when a checkpoint evaluation ends (``state``: completed, stopped or failed;
    None when it was cancelled before it started, which isn't sent)."""
    cfg = db.get_settings()["notifications"]
    if not state or not cfg.get("url") or not cfg.get(EVENTS.get(state, ""), False):
        return
    msg = evaluation_message(job_id, state, cfg.get("link", ""))
    threading.Thread(target=_send_logged, args=(cfg, msg), daemon=True).start()


def evaluation_message(job_id: str, state: str, link: str = "") -> dict:
    job = db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,)) or {"id": job_id, "name": job_id}
    ev = read_status(job_id).get("evaluation") or {}
    rec = ev.get("recommended")
    if state == "completed":
        body = f"Recommended: {rec['file']} (step {rec['step']}). {rec['reason']}." if rec else \
            "No checkpoint could be recommended."
    else:
        body = _clip(ev.get("message") or "", 300)
    url = f"{link.rstrip('/')}/#/queue/{job_id}" if link else ""
    return {"title": f"{job['name']}: checkpoint evaluation {TITLES.get(state, state)}", "body": body,
            "url": url, "status": state, "job_id": job_id, "event": f"evaluation_{state}"}


def message(job_id: str, status: str, exit_code: int | None = None, link: str = "") -> dict:
    job = db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,)) or {"id": job_id, "name": job_id}
    st = read_status(job_id)
    lines = []
    step, total = st.get("step"), st.get("total_steps")
    if step:
        lines.append(f"Step {step}/{total}" if total else f"Step {step}")
    if st.get("avg_loss") is not None:
        lines.append(f"avg loss {st['avg_loss']:.4f}")
    if job.get("started") and job.get("finished"):
        lines.append(_duration(job["finished"] - job["started"]))
    body = " · ".join(lines)
    if status in ("failed", "interrupted"):
        reason = _failure(job_id, st, exit_code)
        if reason:
            body = f"{body}\n{reason}" if body else reason
    if status == "completed" and st.get("checkpoints"):
        final = next((c for c in st["checkpoints"] if c.get("final")), st["checkpoints"][-1])
        body += f"\nLoRA: {final['file']}"
    url = f"{link.rstrip('/')}/#/queue/{job_id}" if link else ""
    return {"title": f"{job['name']} {TITLES.get(status, status)}", "body": body.strip(), "url": url,
            "status": status, "job_id": job_id}


def _failure(job_id: str, st: dict, exit_code: int | None) -> str:
    if st.get("phase") == "failed" and st.get("message"):
        return _clip(st["message"], 300)
    if exit_code is not None and exit_code < 0:
        return f"The process was killed (signal {-exit_code}); the system may have run out of memory."
    tail = _log_tail(job_id)
    return _clip(tail, 300) if tail else (f"Exit code {exit_code}" if exit_code is not None else "")


def _log_tail(job_id: str) -> str:
    """The last error-looking line of the job's log."""
    try:
        with open(os.path.join(job_dir(job_id), "log.txt"), "rb") as f:
            f.seek(max(0, os.fstat(f.fileno()).st_size - 16384))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if "Error" in line or "error:" in line or "out of memory" in line:
            return line.strip()
    return ""


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _duration(sec: float) -> str:
    sec = int(max(0, sec))
    h, m = sec // 3600, sec % 3600 // 60
    return f"took {h}h {m}m" if h else f"took {m}m {sec % 60}s" if m else f"took {sec}s"


def send(cfg: dict, msg: dict) -> None:
    """Deliver one message; raises with a readable reason on failure."""
    url, kind = cfg["url"], cfg.get("kind", "ntfy")
    text = msg["body"] or msg["title"]
    with httpx.Client(timeout=10, follow_redirects=True) as c:
        if kind == "ntfy":
            params = {"title": msg["title"], "tags": NTFY_TAGS.get(msg["status"], "")}
            if msg["status"] in ("failed", "interrupted"):
                params["priority"] = "high"
            if msg["url"]:
                params["click"] = msg["url"]
            r = c.post(url, params=params, content=text.encode("utf-8"))
        elif kind == "discord":
            content = f"**{msg['title']}**" + (f"\n{msg['body']}" if msg["body"] else "") + \
                (f"\n{msg['url']}" if msg["url"] else "")
            r = c.post(url, json={"content": content[:2000], "username": "Dainamo Trainer"})
        elif kind == "slack":
            link = f"\n<{msg['url']}|Open the job>" if msg["url"] else ""
            r = c.post(url, json={"text": f"*{msg['title']}*" + (f"\n{msg['body']}" if msg["body"] else "") + link})
        else:
            r = c.post(url, json={"event": msg.get("event", f"job_{msg['status']}"), "title": msg["title"], "message": msg["body"],
                                  "url": msg["url"], "job_id": msg["job_id"], "status": msg["status"]})
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.reason_phrase}: {_clip(r.text, 200)}")


def _send_logged(cfg: dict, msg: dict) -> None:
    try:
        send(cfg, msg)
        last.update(at=time.time(), ok=True, error="", title=msg["title"])
    except Exception as e:
        last.update(at=time.time(), ok=False, error=str(e) or type(e).__name__, title=msg["title"])
        print(f"[notify] couldn't send {msg['title']!r}: {e}", file=sys.stderr)

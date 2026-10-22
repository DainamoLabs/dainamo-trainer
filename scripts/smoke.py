"""Smoke test on real hardware: a tiny training run per model family, end to end, through a Dainamo server of its
own. That server keeps its data in a temporary folder, so your queue, datasets and settings aren't touched.

For each family it makes a two-file dataset (images, video clips with sound, or songs, made with Pillow and
ffmpeg), trains a rank-4 LoRA for a few steps, and checks that:

  - the loss is a real number at every step;
  - Stop saves the training state, and Resume carries on from that step to the end;
  - the saved LoRA is readable, has no NaN or infinite weights, and its weights moved from where they started;
  - a sample renders;
  - the Sandbox loads the LoRA (every layer matches the model) and generates with it.

    .venv/bin/python scripts/smoke.py                      # every family you have a profile (or the files) for
    .venv/bin/python scripts/smoke.py krea2 yue2 --gpu 1
    .venv/bin/python scripts/smoke.py --profile "Krea 2 Turbo"

Each family uses your first profile of it (by name), read from your Dainamo's data folder; a family without one
uses the files Model files finds in your ComfyUI models folder. Samples render at a small size with few steps.
The first load of a big model from a slow disk can take half an hour, so test one family at a time while you
work, and run them all when nothing else needs the GPU. It refuses a GPU that's already in use (``--force``).
Exits 1 if any check fails, keeping the temporary folder (with the server's log and each job's log) to look at.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

import httpx
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTIONS = ["a red circle on a white background", "a blue square on a black background"]
SONG = "synth, test tone, instrumental"


def say(msg: str = "", end: str = "\n"):
    print(msg, end=end, flush=True)


# ---------------------------------------------------------------------------------------- your profiles ---
def your_dainamo(data_dir: str) -> tuple[list[dict], str]:
    """Your profiles ({name, family, data}, by name) and models folder, read from ``data_dir`` without writing."""
    path = os.path.join(data_dir, "dainamo.db")
    if not os.path.exists(path):
        return [], ""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        profiles = [{"name": n, "family": f, "data": json.loads(d or "{}")} for n, f, d in
                    con.execute("SELECT name, family, data FROM profiles ORDER BY name COLLATE NOCASE")]
        row = con.execute("SELECT value FROM settings WHERE key='models'").fetchone()
    finally:
        con.close()
    return profiles, (json.loads(row[0]).get("folder", "") if row else "")


# ------------------------------------------------------------------------------------------------- GPU ---
def pick_gpu(want: int | None, force: bool) -> int:
    """``want``, or the first GPU with nothing on it; refuses one in use (another job would run out of memory)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,name", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        sys.exit("No NVIDIA GPU found (nvidia-smi didn't answer): the smoke test trains on a GPU.")
    gpus = [(int(i), int(used), name.strip()) for i, used, name in (line.split(",", 2) for line in out.splitlines())]
    for i, used, name in gpus:
        if want is None and used < 1024 or i == want:
            if used >= 1024 and not force:
                sys.exit(f"GPU {i} ({name}) has {used} MiB in use: something else is running on it. Pick another "
                         "with --gpu, or use --force to run anyway.")
            say(f"GPU {i}: {name}, {used} MiB in use")
            return i
    sys.exit(f"No GPU {want}" if want is not None else "Every GPU is in use; pick one with --gpu (and --force)")


# ---------------------------------------------------------------------------------------------- server ---
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """A Dainamo server on its own data folder, and its API."""

    def __init__(self, data: str):
        self.data, self.port = data, free_port()
        self.log = open(os.path.join(data, "server.log"), "w")
        env = {k: v for k, v in os.environ.items() if k != "DAINAMO_PASSWORD"}  # no sign-in on this one
        self.proc = subprocess.Popen([sys.executable, "-m", "dainamo", "--port", str(self.port), "--no-browser"],
                                     cwd=ROOT, env=dict(env, DAINAMO_DATA=data), stdout=self.log,
                                     stderr=subprocess.STDOUT)
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=120)
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError(f"The server didn't start; see {self.log.name}")
            try:
                self.http.get("/api/settings").raise_for_status()
                return
            except httpx.HTTPError:
                time.sleep(0.5)
        raise RuntimeError(f"The server didn't answer within 2 minutes; see {self.log.name}")

    def call(self, method: str, url: str, **kw):
        r = self.http.request(method, url, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:300]
            raise RuntimeError(f"{method} {url}: {r.status_code} {detail}")
        return r.json() if "json" in r.headers.get("content-type", "") else r.content

    def get(self, url: str, **kw):
        return self.call("GET", url, **kw)

    def post(self, url: str, body=None):
        return self.call("POST", url, json=body)

    def stop(self):
        """The server, its training processes (which would outlive it) and its Sandbox."""
        try:
            kids = psutil.Process(self.proc.pid).children(recursive=True)
        except psutil.Error:
            kids = []
        for p in [*kids, self.proc]:
            try:
                p.terminate()
            except (psutil.Error, OSError):
                pass
        _, alive = psutil.wait_procs(kids, timeout=20)
        for p in alive:
            p.kill()
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.http.close()
        self.log.close()


# ------------------------------------------------------------------------------------------- datasets ---
def make_media(folder: str, media: list[str]) -> str:
    """Two files of the kind a family trains on, each with a caption."""
    os.makedirs(folder, exist_ok=True)
    if "audio" in media:  # songs: 12 s, stereo, 48 kHz (the health check flags mono and low rates)
        for n, hz in enumerate((220, 330)):
            out = os.path.join(folder, f"song{n}.wav")
            ffmpeg("-f", "lavfi", "-i", f"sine=frequency={hz}:duration=12:sample_rate=48000", "-ac", "2", out)
            with open(out[:-4] + ".txt", "w") as f:
                f.write(SONG)
    elif "video" in media:  # 2 s clips with sound
        for n, (src, hz) in enumerate((("testsrc2", 440), ("rgbtestsrc", 660))):
            out = os.path.join(folder, f"clip{n}.mp4")
            ffmpeg("-f", "lavfi", "-i", f"{src}=size=320x256:rate=24:duration=2", "-f", "lavfi", "-i",
                   f"sine=frequency={hz}:duration=2", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", out)
            with open(out[:-4] + ".txt", "w") as f:
                f.write(f"a test pattern, {CAPTIONS[n]}")
    else:
        from PIL import Image, ImageDraw

        for n, (shape, fg, bg) in enumerate((("circle", "red", "white"), ("square", "blue", "black"))):
            im = Image.new("RGB", (512, 512), bg)
            draw = ImageDraw.Draw(im)
            (draw.ellipse if shape == "circle" else draw.rectangle)((128, 128, 384, 384), fill=fg)
            im.save(os.path.join(folder, f"{shape}.png"))
            with open(os.path.join(folder, f"{shape}.txt"), "w") as f:
                f.write(CAPTIONS[n])
    return folder


def ffmpeg(*args: str):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


# -------------------------------------------------------------------------------------------- a family ---
class Failed(Exception):
    pass


class Family:
    """One family's run: its checks, as (name, ok, detail)."""

    def __init__(self, srv: Server, fam: dict, profile: dict, source: str, gpu: int, steps: int, timeout: float):
        self.srv, self.fam, self.profile, self.source, self.gpu, self.steps = srv, fam, profile, source, gpu, steps
        self.deadline = time.time() + timeout
        self.checks: list[tuple[str, bool, str]] = []
        self.media = fam["media"]
        self.music, self.video = "audio" in self.media, "video" in self.media

    def check(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, ok, detail))
        say(f"  {'✓' if ok else '✗'} {name}" + (f": {detail}" if detail else ""))
        if not ok:
            raise Failed(name)

    def wait(self, what: str, done, every: float = 1.0):
        """Polls ``done()`` until it returns something, printing the job's or Sandbox's messages as they change."""
        last = None
        while time.time() < self.deadline:
            out, msg = done()
            if msg and msg != last:
                say(f"      {msg}")
                last = msg
            if out is not None:
                return out
            time.sleep(every)
        raise Failed(f"timed out waiting for {what}")

    def run(self):
        srv, fam = self.srv, self.fam
        name = self.profile["name"]
        say(f"\n{fam['label']}" + (f": {name}" if name != fam["label"] else "") + f" ({self.source})")
        prof = srv.post("/api/profiles", {"name": self.profile["name"], "family": fam["id"], "data": self.profile["data"]})
        self.div = srv.get("/api/sandbox/defaults", params={"profile_id": prof["id"]})["divisibility"]
        bad = [c for c in srv.post("/api/profiles/validate", {"name": prof["name"], "family": fam["id"],
                                                               "data": prof["data"]}) if c["level"] == "error"]
        self.check("profile files are readable", not bad, "; ".join(f"{c['field']}: {c['message']}" for c in bad))

        folder = make_media(os.path.join(srv.data, "media", fam["id"]), self.media)
        ds = srv.post("/api/datasets", {"name": f"smoke {fam['id']}", "path": folder, "trigger": ""})
        jid = srv.post("/api/jobs", {"name": f"smoke {fam['id']}", "profile_id": prof["id"],
                                     "datasets": [{"id": ds["id"]}], "settings": self.settings(),
                                     "gpu": str(self.gpu)})["id"]
        self.train(jid)
        lora = self.saved_lora(jid)
        self.sample(jid)
        self.sandbox(prof, lora)

    def settings(self) -> dict:
        s = {"steps": self.steps, "rank": 4, "alpha": 4, "save_every": 0, "sample_every": 0, "keep_last": 0,
             "sample_at_start": False, "sample_at_end": True}
        if self.music:
            s.update(clip_seconds=10, sample_seconds=5)
            s["sample_prompts"] = [{"prompt": SONG, "seconds": 5, "seed": 1}]
        else:
            size = self.size(256 if self.video else 512)
            s.update(resolution=256, sample_width=size, sample_height=size)
            card = {"prompt": CAPTIONS[0], "width": size, "height": size, "seed": 1}
            if self.video:
                frames = self.frames()
                s.update(num_frames=frames, sample_frames=frames, auto_frames=False)
                card["frames"] = frames
            s["sample_prompts"] = [card]
        return s

    def size(self, want: int) -> int:
        return max(self.div, want // self.div * self.div)

    def frames(self) -> int:
        """The shortest clip the family's VAE takes, from its frame rule ("17n + 5 (22, 39, …)")."""
        m = re.search(r"\((\d+)", self.fam.get("frame_rule") or "")
        return int(m.group(1)) if m else 9

    # --- training, Stop and Resume ---
    def job(self, jid: str) -> dict:
        return self.srv.get(f"/api/jobs/{jid}")

    def train(self, jid: str):
        # a quarter of the way, polled often: a small model trains these steps in seconds
        t0, stop_at, asked = time.time(), max(1, self.steps // 4), []

        def stopped():
            j = self.job(jid)
            st = j.get("status_detail") or {}
            if j["status"] == "running" and (st.get("step") or 0) >= stop_at and not asked:
                self.srv.post(f"/api/jobs/{jid}/action", {"action": "stop"})
                asked.append(True)
            done = j if j["status"] in ("stopped", "completed", "failed") else None
            return done, st.get("message")

        j = self.wait("the job to stop", stopped, every=0.2)
        if j["status"] == "failed":
            self.check("training runs", False, self.log_tail(jid))
        step = int(j["status_detail"].get("step") or 0)
        if j["status"] == "completed":  # finished before the stop arrived: resume as Extend does instead
            say(f"      (it finished before Stop arrived; extending it to {self.steps + 5} steps to test resuming)")
            self.check("the state is kept for resuming", j.get("resumable", False), f"at step {step}")
            self.srv.post(f"/api/jobs/{jid}/action", {"action": "extend", "steps": self.steps + 5})
            self.steps += 5
        else:
            self.check("Stop saves the state", j.get("resumable", False) and step > 0,
                       f"stopped at step {step} ({time.time() - t0:.0f} s, model loading included)")
            self.srv.post(f"/api/jobs/{jid}/action", {"action": "resume"})

        def finished():
            j = self.job(jid)
            st = j.get("status_detail") or {}
            return (j if j["status"] in ("completed", "failed", "stopped") else None), st.get("message")

        j = self.wait("the job to finish", finished)
        log = self.srv.get(f"/api/jobs/{jid}/log").decode()
        m = re.findall(r"Resumed from step (\d+)", log)
        self.check("Resume carries on", j["status"] == "completed" and bool(m) and int(m[-1]) == step,
                   f"from step {m[-1]} to {j['status_detail'].get('step')}" if m and j["status"] == "completed"
                   else self.log_tail(jid))
        pts = self.srv.get(f"/api/jobs/{jid}/metrics")["points"]
        losses = [p["loss"] for p in pts]
        bad = [p["step"] for p in pts if not math.isfinite(p["loss"])]
        missing = sorted(set(range(1, self.steps + 1)) - {p["step"] for p in pts})
        self.check("the loss is a real number at every step", bool(losses) and not bad and not missing,
                   f"{len(losses)} steps, first {losses[0]:.4f}, last {losses[-1]:.4f}" if losses and not bad
                   and not missing else f"not finite at steps {bad[:10]}, missing steps {missing[:10]}")

    def log_tail(self, jid: str, lines: int = 15) -> str:
        try:
            text = self.srv.get(f"/api/jobs/{jid}/log").decode()
        except RuntimeError as e:
            return str(e)
        return "\n" + "\n".join("        " + line for line in text.strip().splitlines()[-lines:])

    # --- the saved LoRA ---
    def saved_lora(self, jid: str) -> str:
        from safetensors import safe_open

        j = self.job(jid)
        final = [c for c in j["status_detail"].get("checkpoints", []) if c.get("final")]
        if not final:
            self.check("the final LoRA is saved", False, "no final checkpoint")
        path = os.path.join(j["dir"], "output", final[-1]["file"])
        nonfinite, moved, ups, count = [], False, 0, 0
        with safe_open(path, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k).float()
                count += 1
                if not t.isfinite().all():
                    nonfinite.append(k)
                # an "up" (B) matrix starts at zero: any non-zero one means training changed the LoRA
                if re.search(r"lora_up|lora_B|\.up\b", k):
                    ups += 1
                    moved = moved or bool(t.abs().max() > 0)
        size = os.path.getsize(path)
        self.check("the LoRA is readable with finite weights", count > 0 and not nonfinite,
                   f"{final[-1]['file']}, {count} tensors, {size / 1e6:.1f} MB" if not nonfinite
                   else f"NaN or infinite in {nonfinite[:5]}")
        if ups:
            self.check("training changed its weights", moved, "" if moved else "every up matrix is still zero")
        return path

    # --- samples ---
    def sample(self, jid: str):
        samples = self.job(jid)["status_detail"].get("samples") or []
        sizes = [len(self.srv.get(f"/api/jobs/{jid}/file/samples/{s['file']}")) for s in samples]
        self.check("a sample renders", bool(samples) and all(sizes),
                   ", ".join(f"{s['file']} ({n / 1e3:.0f} kB)" for s, n in zip(samples, sizes)) or "none saved")

    # --- the Sandbox ---
    def sandbox(self, prof: dict, lora: str):
        defaults = self.srv.get("/api/sandbox/defaults", params={"profile_id": prof["id"]})
        wf = next((w for w in defaults["workflows"] if not w.get("images")), None)
        if wf is None:
            say("  - Sandbox skipped: this profile's workflows all need images")
            return
        body = {"profile_id": prof["id"], "workflow": wf["id"], "prompt": SONG if self.music else CAPTIONS[0],
                "steps": 4, "cfg": 1.0 if defaults.get("cfg") is None else defaults["cfg"], "seed": 1, "gpu": self.gpu,
                "loras": [{"path": lora, "strength": self.fam.get("lora_strength", 1.0)}]}
        if self.music:
            body.update(seconds=5)
        else:
            size = self.size(256 if self.video else 512)
            body.update(width=size, height=size, frames=self.frames() if self.video else 1)
        rid = self.srv.post("/api/sandbox/generate", body)["id"]

        def done():
            st = self.srv.get("/api/sandbox")
            run = next(r for r in st["runs"] if r["id"] == rid)
            msg = (st.get("session") or {}).get("message")
            return (run if run["status"] in ("done", "failed", "cancelled") else None), msg

        try:
            run = self.wait("the Sandbox", done)
        finally:
            try:
                self.srv.post("/api/sandbox/release")  # the next family trains on this GPU
            except RuntimeError:
                pass
        if run["status"] != "done":
            self.check("the Sandbox generates with the LoRA", False, run.get("message") or run["status"])
        report = run.get("report") or {}
        report = ((json.loads(report) if isinstance(report, str) else report).get("loras") or [{}])[0]
        matched, total = report.get("matched", 0), report.get("total", 0)
        self.check("the Sandbox loads the LoRA", total > 0 and matched == total,
                   f"{matched} of {total} layers matched the model")
        outs = run.get("outputs") or []
        outs = json.loads(outs) if isinstance(outs, str) else outs
        sizes = [len(self.srv.get(f"/api/sandbox/file/{rid}/{o['file']}")) for o in outs]
        self.check("the Sandbox generates with it", bool(outs) and all(sizes),
                   f"{outs[0]['file']} ({sizes[0] / 1e3:.0f} kB) in {run.get('elapsed')} s" if outs else "no output")


# ----------------------------------------------------------------------------------------------- main ---
def choose(srv: Server, families: list[dict], wanted: list[str], names: list[str], yours: list[dict],
           models: str) -> list[tuple[dict, dict, str]]:
    """(family, profile, where it came from) for each family to test."""
    by_id = {f["id"]: f for f in families}
    if names:
        out = []
        for n in names:
            p = next((p for p in yours if p["name"] == n), None)
            if p is None:
                sys.exit(f"No profile named {n!r}. Yours: {', '.join(p['name'] for p in yours) or 'none'}")
            if p["family"] not in by_id:
                sys.exit(f"{n!r} is a {p['family']} profile, a family this Dainamo doesn't have")
            out.append((by_id[p["family"]], p, "your profile"))
        return out
    unknown = [w for w in wanted if w not in by_id]
    if unknown:
        sys.exit(f"Unknown families {unknown}. Families: {', '.join(by_id)}")
    found = {}  # what Model files finds in your models folder, for families without a profile
    listed = []
    if models:
        try:
            srv.call("PUT", "/api/settings", json={"models": {"folder": models}})
            listed = srv.get("/api/models")["families"]
        except RuntimeError as e:
            say(f"Your models folder can't be used: {e}")
    for fam in listed:
        data = {f["key"]: next((x["path"] for x in f["found"] if not x["unsupported"]), "") for f in fam["fields"]}
        if all(data[f["key"]] for f in fam["fields"] if f["required"]):
            found[fam["id"]] = {k: v for k, v in data.items() if v}
    out = []
    for fid in wanted or list(by_id):
        fam = by_id[fid]
        p = next((p for p in yours if p["family"] == fid), None)
        if p:
            out.append((fam, p, "your profile"))
        elif fid in found:
            out.append((fam, {"name": fam["label"], "family": fid, "data": found[fid]}, "files in your models folder"))
        else:
            say(f"{fam['label']}: skipped, no profile and its files aren't in your models folder")
    return out


def quick(profile: dict, fam: dict) -> dict:
    """The profile with few sampling steps, and the field defaults it leaves out."""
    data = {f["key"]: f["default"] for f in fam["profile_fields"] if f.get("default") not in (None, "")}
    data.update(profile["data"])
    if any(f["key"] == "sample_steps" for f in fam["profile_fields"]):
        data["sample_steps"] = 4
    return {**profile, "data": data}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("families", nargs="*", help="family ids, e.g. krea2 yue2 (default: every one you can test)")
    ap.add_argument("--profile", action="append", default=[], help="test this profile of yours (repeatable)")
    ap.add_argument("--gpu", type=int, help="the GPU to train on (default: the first one with nothing on it)")
    ap.add_argument("--force", action="store_true", help="use the GPU even if something else is on it")
    ap.add_argument("--steps", type=int, default=20, help="training steps per family (default 20)")
    ap.add_argument("--timeout", type=float, default=120, help="minutes per family before giving up (default 120)")
    ap.add_argument("--data", default=os.environ.get("DAINAMO_DATA", os.path.join(ROOT, "data")),
                    help="your Dainamo's data folder, for its profiles and models folder (read only)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary data folder even if every check passes")
    args = ap.parse_args()
    if args.steps < 2:
        sys.exit("--steps must be at least 2 (it stops partway and resumes)")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg isn't installed (it makes the video and music datasets)")

    gpu = pick_gpu(args.gpu, args.force)
    yours, models = your_dainamo(args.data)
    tmp = tempfile.mkdtemp(prefix="dainamo-smoke-")
    say(f"Temporary data folder: {tmp}")
    srv = Server(tmp)
    results: list[tuple[str, list, str]] = []
    try:
        srv.call("PUT", "/api/settings", json={"queue": {"mode": "sequential", "gpus": [gpu]}})
        families = srv.get("/api/system")["families"]
        for fam, profile, source in choose(srv, families, args.families, args.profile, yours, models):
            t0 = time.time()
            run = Family(srv, fam, quick(profile, fam), source, gpu, args.steps, args.timeout * 60)
            error = ""
            try:
                run.run()
            except Failed as e:
                error = str(e)
            except (RuntimeError, httpx.HTTPError, subprocess.CalledProcessError, OSError) as e:
                error = str(e)
                say(f"  ✗ {e}")
            say(f"  {time.time() - t0:.0f} s")
            results.append((fam["label"], run.checks, error))
            for j in srv.get("/api/jobs"):  # a failed or timed-out job mustn't hold the GPU for the next family
                if j["status"] in ("queued", "running"):
                    srv.post(f"/api/jobs/{j['id']}/action", {"action": "stop"})
    except KeyboardInterrupt:
        say("\nStopped.")
        results.append(("(interrupted)", [], "interrupted"))
    finally:
        srv.stop()

    say("\nSummary")
    for label, checks, error in results:
        say(f"  {'✗ ' + label + ': ' + error if error else '✓ ' + label + f': {len(checks)} checks passed'}")
    failed = any(error for _, _, error in results)
    if not results:
        say("  Nothing to test.")
    if failed or args.keep:
        say(f"Kept {tmp} (server.log, and each job's log in jobs/<id>/)")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()

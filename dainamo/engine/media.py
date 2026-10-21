"""Writing generated media: PNG for images; MP4 (H.264, plus AAC when there's audio) with a PNG poster
for video; MP3 with a spectrogram poster (and any score) for music. Used by training samples and the Sandbox."""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image


def write_wav(path: str, audio: np.ndarray, sample_rate: int):
    """(channels, samples) float in [-1, 1] -> 16-bit PCM WAV."""
    pcm = (np.clip(audio.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wv:
        wv.setnchannels(pcm.shape[1])
        wv.setsampwidth(2)
        wv.setframerate(int(sample_rate))
        wv.writeframes(pcm.tobytes())


def spectrogram(src: str, out: str, width: int = 1024, height: int = 256):
    """A PNG spectrogram of an audio file (mono, log frequency): music's poster and thumbnail."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-lavfi",
                    f"aformat=channel_layouts=mono,showspectrumpic=s={width}x{height}:legend=0:scale=sqrt:"
                    "fscale=log:color=magma:drange=90", "-frames:v", "1", out],
                   check=True, capture_output=True, timeout=300)


def save_audio(media: dict, out_dir: str, stem: str) -> dict:
    """{audio (C, samples), sample_rate, score?} -> MP3, a spectrogram poster and the score (ABC) as text."""
    out = {"file": stem + ".mp3", "poster": stem + "_poster.png", "kind": "audio"}
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        write_wav(wav_path, media["audio"], media["sample_rate"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "256k",
                        os.path.join(out_dir, out["file"])], check=True, capture_output=True)
        spectrogram(wav_path, os.path.join(out_dir, out["poster"]))
    if media.get("score"):
        out["score"] = stem + ".abc"
        with open(os.path.join(out_dir, out["score"]), "w", encoding="utf-8") as f:
            f.write(media["score"])
    return out


def save_media(media, out_dir: str, stem: str) -> dict:
    """``media``: a PIL image, {frames (T, H, W, 3) uint8, fps, audio (C, samples) | None, sample_rate}, or
    for music {audio, sample_rate, score?}. Returns {file, kind[, poster][, score]} relative to ``out_dir``."""
    if isinstance(media, Image.Image):
        media.save(os.path.join(out_dir, stem + ".png"))
        return {"file": stem + ".png", "kind": "image"}
    if "frames" not in media:
        return save_audio(media, out_dir, stem)
    frames = media["frames"]
    t, h, w, _ = frames.shape
    Image.fromarray(frames[0]).save(os.path.join(out_dir, stem + "_poster.png"))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(media["fps"]), "-i", "-"]
    audio = media.get("audio")
    with tempfile.TemporaryDirectory() as tmp:
        if audio is not None:
            wav_path = os.path.join(tmp, "audio.wav")
            write_wav(wav_path, audio, media["sample_rate"])
            # silence-padded, so the video sets the length: -shortest on an exactly as long soundtrack drops
            # the last frame
            cmd += ["-i", wav_path, "-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                os.path.join(out_dir, stem + ".mp4")]
        subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True)
    return {"file": stem + ".mp4", "poster": stem + "_poster.png", "kind": "video"}

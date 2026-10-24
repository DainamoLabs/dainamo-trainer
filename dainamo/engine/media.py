"""Writing generated media: PNG for images; MP4 (H.264, plus AAC when there's audio) with a PNG poster
for video, whole (``save_media``) or a chunk of frames at a time (``VideoWriter``); MP3 with a spectrogram poster
(and any score) for music. Used by training samples and the Sandbox."""

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
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = None
        if media.get("audio") is not None:
            wav_path = os.path.join(tmp, "audio.wav")
            write_wav(wav_path, media["audio"], media["sample_rate"])
        with VideoWriter(out_dir, stem, media["fps"], audio=wav_path) as video:
            video.write(media["frames"])
            return video.close()


class VideoWriter:
    """An MP4 written as its frames come, so a long clip is never whole in RAM: one ffmpeg, H.264 (crf 18, 4:2:0,
    faststart) at ``fps`` as it's given ("30000/1001" stays exact), with a PNG poster of the first frame.

    ``write`` takes frames (T, H, W, 3) uint8, any number at a time; ``close`` finishes the file and returns
    ``save_media``'s {file, kind: "video", poster}. ``audio``: a file whose first audio stream becomes the sound,
    read by ffmpeg as it goes (re-encoded to AAC, mono or stereo, silence-padded or cut to the video's length).
    As a context manager, an error inside it (cancelling too) stops ffmpeg and removes what was written."""

    def __init__(self, out_dir: str, stem: str, fps, audio: str | None = None):
        self.file, self.poster = stem + ".mp4", stem + "_poster.png"
        self.path, self.poster_path = os.path.join(out_dir, self.file), os.path.join(out_dir, self.poster)
        self.fps, self.audio = str(fps), audio
        self.frames = 0  # written so far
        self._proc = self._err = self._shape = self._result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        elif self._proc is not None and self._result is None:
            self.close()

    def write(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"VideoWriter takes (T, H, W, 3) uint8 frames, not {frames.dtype} {frames.shape}")
        if self._result is not None:
            raise ValueError(f"{self.file} is closed")
        if not len(frames):
            return
        if self._proc is None:
            self._start(frames)
        elif frames.shape[1:] != self._shape:
            raise ValueError(f"{self.file} is {self._shape[1]}×{self._shape[0]}; these frames are "
                             f"{frames.shape[2]}×{frames.shape[1]}")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frames).reshape(-1).data)
        except (BrokenPipeError, ValueError):  # ffmpeg has stopped: say why
            self._proc.wait()
            raise self._failure() from None
        self.frames += len(frames)

    def _start(self, first: np.ndarray) -> None:
        h, w = first.shape[1:3]
        self._shape = first.shape[1:]
        Image.fromarray(first[0]).save(self.poster_path)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", self.fps, "-i", "-"]
        if self.audio is not None:
            # silence-padded, so the video sets the length: -shortest on an exactly as long soundtrack drops
            # the last frame
            cmd += ["-i", self.audio, "-map", "0:v:0", "-map", "1:a:0", "-map_chapters", "-1",
                    "-af", "aformat=channel_layouts=mono|stereo,apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", self.path]
        self._err = tempfile.TemporaryFile()  # not a pipe: nothing reads it until the end
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err)

    def _failure(self) -> ValueError:
        self._err.seek(0)
        why = self._err.read().decode(errors="replace").strip()[-500:]
        self.abort()
        return ValueError(f"ffmpeg couldn't write {self.file}: {why or f'exit code {self._proc.returncode}'}")

    def close(self) -> dict:
        """Finish the file -> {file, kind, poster} (relative to the folder). Again: the same."""
        if self._result is not None:
            return self._result
        if self._proc is None:
            raise ValueError(f"{self.file}: no frames to write")
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        if self._proc.wait() != 0:
            raise self._failure()
        self._err.close()
        self._result = {"file": self.file, "poster": self.poster, "kind": "video"}
        return self._result

    def abort(self) -> None:
        """Stop ffmpeg, and remove the video and poster written so far."""
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self._proc.wait()
        if self._err is not None:
            self._err.close()
        for path in (self.path, self.poster_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

"""Speech-to-text for video clips with Whisper (through transformers), run by Auto-Caption as a subprocess:

    python -m dainamo.engine.transcribe <model> <seconds> <clip>...

It prints one JSON line per clip, {"path", "text"} ("" when nothing is said) or {"path", "error"}, after
{"event": "loading"} and {"event": "ready"} around loading the model. ``seconds`` is how much of each clip
to hear from its start (0: all of it), so a transcript covers the part of the clip that's trained on.

Whisper writes words into silence, music and noise ("you", "thank you for watching"), so a window counts as
speech only when it has sound and Whisper's own no-speech probability is under 0.6, and a transcript that's
only one of those stock phrases is dropped."""

from __future__ import annotations

import json
import re
import sys

# name -> (Hugging Face model, download size)
MODELS = {
    "tiny": ("openai/whisper-tiny", "150 MB"),
    "base": ("openai/whisper-base", "290 MB"),
    "small": ("openai/whisper-small", "970 MB"),
    "medium": ("openai/whisper-medium", "3.1 GB"),
    "large-v3-turbo": ("openai/whisper-large-v3-turbo", "1.6 GB"),
    "large-v3": ("openai/whisper-large-v3", "3.1 GB"),
}
SAMPLE_RATE = 16000
WINDOW = 30.0  # seconds Whisper hears at once
NO_SPEECH = 0.6  # Whisper's own threshold
SILENCE_RMS = 1e-3
STOCK_PHRASES = {"you", "thank you", "thanks", "thanks for watching", "thank you for watching",
                 "thank you so much for watching", "please subscribe", "subtitles by the amaraorg community"}


def is_stock_phrase(text: str) -> bool:
    return re.sub(r"[^a-z ]", "", text.lower()).strip() in STOCK_PHRASES


class Transcriber:
    def __init__(self, model: str):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = MODELS[model][0]
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        from .hf_assets import from_pretrained

        self.processor = from_pretrained(WhisperProcessor, model_id, f"Whisper {model}")
        self.model = from_pretrained(WhisperForConditionalGeneration, model_id, f"Whisper {model}",
                                     dtype=dtype).to(self.device).eval()
        gc = self.model.generation_config
        self.start = gc.decoder_start_token_id
        self.no_speech = gc.no_timestamps_token_id - 1  # <|nocaptions|> / <|nospeech|>, as Whisper finds it

    def window(self, wav) -> str:
        torch = self.torch
        if float((wav ** 2).mean() ** 0.5) < SILENCE_RMS:
            return ""
        feats = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)
        with torch.no_grad():
            first = self.model(feats, decoder_input_ids=torch.tensor([[self.start]], device=self.device)).logits
            if first[0, -1].float().softmax(-1)[self.no_speech].item() > NO_SPEECH:
                return ""
            ids = self.model.generate(feats, task="transcribe")
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return "" if is_stock_phrase(text) else text

    def clip(self, path: str, seconds: float) -> str:
        from .dataset import probe_video, read_audio

        if seconds <= 0:
            seconds = probe_video(path)["duration"]
        wav = read_audio(path, seconds, SAMPLE_RATE, channels=1)[0].numpy()
        step = int(WINDOW * SAMPLE_RATE)
        parts = [self.window(wav[i:i + step]) for i in range(0, max(1, len(wav)), step)]
        return " ".join(p for p in parts if p)


def main():
    model, seconds, paths = sys.argv[1], float(sys.argv[2]), sys.argv[3:]

    def say(**msg):
        print(json.dumps(msg), flush=True)

    say(event="loading")
    t = Transcriber(model)
    say(event="ready", device=t.device.type)
    for path in paths:
        try:
            say(path=path, text=t.clip(path, seconds))
        except Exception as e:  # one bad clip doesn't stop the others
            say(path=path, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

# Music (YuE2)

YuE2-3B (M-A-P) writes whole songs, vocals included, from style tags and lyrics.
Its single checkpoint holds a **composer**, which writes the song's semantic
tokens (and optionally a score), an **acoustic model**, which turns them into
audio, and the VAE and tokenizer.

Dainamo trains a LoRA on the acoustic model only: the composer can't learn from
recordings, because the tokenizer it would need hasn't been released. So a YuE2
LoRA learns how songs sound (style, instruments, timbre, production, mix). The
melody, form and where the words fall still come from the composer.

## Datasets

Songs in WAV, FLAC, MP3, OGG, M4A, Opus or AAC, each with a `.txt` caption
holding its style tags. Lyrics go under a `[Lyrics]` line, marked with YuE2's
section tags:

```
indie folk, female vocals, acoustic guitar, warm, intimate
[Lyrics]
[Verse]
Morning light on the kitchen floor
[Chorus]
...
```

Write music captions by hand: Auto-Caption can't hear. The trigger word is
added to the style tags. **Swap stereo channels** is music's version of flip.
The health check flags songs shorter than 5 s (skipped), longer than 6 minutes
(only the first 6 train), below 44.1 kHz, or mono.

## Clip length

Songs train as clips of the clip length (30 s by default), spread evenly over
the song. Each epoch trains one of a song's clips, so a long song counts no more
than a short one; give a song more weight with a dataset's repeats.

Clips train on the style tags alone. Lyrics are used only with a clip length of
0, which trains whole songs (up to 6 minutes).

## Sampling

Samples and the Sandbox run the whole release pipeline, with 32 midpoint steps
(no sampler choice). The profile's **Score planning** (off, melody or full) has
the composer write a score first; it's off by default, closest to how the LoRA
trains. Sample cards take style tags, lyrics and a length (5 s to 6 minutes).
Songs are saved as 256 kbps MP3s with a spectrogram, plus an `.abc` score when
Score planning is on.

## LoRA strength 2

A YuE2 LoRA acts at about half strength in normal generation, because the
composer's semantic tokens, which training doesn't see, carry much of the sound.
So samples, the Sandbox and strength sweeps default to 2, and the file records
it (`dainamo.recommended_strength`). Use 2 in ComfyUI too.

In testing, a LoRA trained on 12 songs with a telephone effect had fully learned
it by step 1500, yet at strength 1 it barely showed (51/38/68% of energy below
250 Hz, against the base model's 66/70/74%). At strength 2 it came through
(3.1/2.4/7.1%). Prompts without the trigger changed as much, so a YuE2 LoRA's
style applies to every prompt.

![The telephone test's job page: three prompts at each checkpoint, each sample an MP3 with its spectrogram](screenshots/music.png)

**Listen rather than trust the checkpoint picker.** Its validation loss
measures how well the LoRA re-creates the held-out songs, which rises as it
memorises them: in the test above it rose from step 500 on, while the effect was
still being learned until step 1000. Concept bleed is off for music.

## ComfyUI and VRAM

LoRAs use the checkpoint's own weight names, so ComfyUI's LoRA loader applies
them to its YuE2 model.

Training needs about 2.5 GB on 30 s clips and 4.7 GB on whole songs. Sampling
peaks at about 8.5 GB for a 30 s song. YuE2 has no Low VRAM switch.

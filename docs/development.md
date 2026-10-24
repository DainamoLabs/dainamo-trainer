# Development

## Layout

```
dainamo/
  engine/                 training engine (runs as a subprocess per job)
    comfy_io.py           safetensors reader for ComfyUI formats (fp8_scaled, comfy_quant)
    fp8_linear.py         frozen fp8 Linear, dequantized on the fly
    offload.py            layer offloading, Low VRAM streaming, parking models and optimizer state in RAM
    memory.py             VRAM and system RAM projection from file headers
    lora.py               LoRA and LoKr networks + frozen adapters, ComfyUI key formats
    finetune.py           full finetuning and saving whole models
    extract.py            LoRAs extracted from a finetune
    merge.py              LoRAs merged into one, or cut to a smaller rank
    dataset.py            discovery, aspect buckets, caption/trigger handling
    trainer.py            family-agnostic training loop
    ema.py                an exponential moving average of the trained weights, used for saves and samples
    slider.py             concept sliders: their settings and training step
    transcribe.py         Whisper speech-to-text for Auto-Caption (a subprocess)
    textgen.py            text encoders that write captions, and their OpenAI-style server
    hf_assets.py          tokenizers and configs from Hugging Face, pinned to a commit
  assets/                 bundled copies of those for the families whose licenses allow it (see its README)
    families/             one driver per model family
      base.py             driver interface
      krea2/              Krea 2 model + driver
      ltx2/               LTX 2.5: diffusers' LTX-2 classes loaded from ComfyUI files, and the driver
  server/
    app.py                FastAPI app: middleware, routers, the built web UI
    routes/               the API, one module per area (system, profiles, models, datasets, clips, jobs, sandbox, llm, tools)
    validation.py         job settings checks
    db.py                 SQLite store and its schema migrations
    queue.py              job queue: launches training processes onto GPUs
    procs.py              starting, finding and stopping worker processes (Linux and Windows)
    sandbox.py            the Sandbox worker process
    notify.py             webhooks when a job ends
    hardware.py           GPU, CPU and RAM readings
    auth.py               sign-in for remote access
    captioner.py          Auto-Caption and Enhance prompt requests
    llama_server.py       the caption server Dainamo runs (llama-server or textgen)
    dataset_check.py      the dataset health check
    clip_split.py         Split into clips, and the plan the clip editor changes
    media_download.py     Add from URL (yt-dlp)
    model_files.py        Model files: finding model files, and downloading them
  webui.py                whether web/dist is up to date with its source (for the launchers)
scripts/measure_memory.py measures a family's activation memory for the projection
web/                      React + Vite UI (built into web/dist)
docs/                     these documents, and the screenshots in docs/screenshots/
data/                     database, uploaded datasets, job outputs (data/jobs/<name>_<id>/)
```

**Database.** To change the schema, add a step to the end of `MIGRATIONS` in
`dainamo/server/db.py` (SQL or a Python function); never edit a released step.
On startup the database is backed up to `data/dainamo.db.v<N>.bak`, then each
new step runs once, in a transaction.

**Memory internals.** Pinned RAM comes in shared slabs sized per model, since
PyTorch rounds each pinned allocation up to a power of two. Low VRAM moves each
block's saved input to pinned RAM on a side stream and prefetches it one block
ahead of backward. A finetune steps each weight's optimizer inside backward on
an fp32 copy, written back to bf16 with stochastic rounding. Its offloaded
blocks are swapped through `.data`, so the optimizer keeps the same parameters,
and a block that was stepped is copied back to RAM on the side stream when it's
swapped out.

## Adding a model family

Subclass `FamilyDriver` in `dainamo/engine/families/<name>/driver.py`, and
register it in `families/__init__.py`. A driver declares:

- its profile fields and LoRA target layers;
- the tensors that identify each file (`signature_keys`, `signature_shapes`) and
  the files Model files can download (`downloads`: repo, pinned commit, size,
  SHA-256);
- how to load the text encoder and VAE, and how to build the diffusion model
  from its tensor shapes (`build_transformer`; the shared loader reads the
  weights and handles offloading), plus the name of its block list;
- for Low VRAM, a text encoder loaded to `text_encoder_device` and passed
  through `stream_text_encoder` (drop the loaded state dict first, or its
  tensors stay in RAM beside the pinned copies);
- its `MemoryProfile`, measured with `scripts/measure_memory.py`;
- the timestep sampling, the forward pass and sampling.

The queue, UI, caching, LoRA saving and resume work unchanged.

## Tests

```bash
uv pip install --python .venv/bin/python pytest   # once
.venv/bin/python -m pytest tests -v
```

GitHub Actions (`.github/workflows/ci.yml`) runs them, and type-checks and
builds the web UI, on every push to master and every pull request. It has no
GPU, so tests that need one, or real model files, skip themselves.

Tests use a throwaway data folder, never `data/`. Header-only copies of model
files (sparse files of a few kilobytes, from `tests/conftest.py`) stand in where
only the header matters.

| File | Checks |
| --- | --- |
| `test_quant.py` | small ComfyUI-style fp8 and int8/convrot files: loading, forward/backward parity, per-channel scale precision, LoRA gradients. With both Krea 2 Turbo files present, that int8_convrot and fp8_scaled agree layer by layer |
| `test_moss.py` | the MOSS-Image DiT and sampling schedule against OpenMOSS's reference code (fetched from Hugging Face), and the training noise sampler against the report's eq. (4) |
| `test_qwen_image_2.py` | the Qwen-Image 2.1 DiT against the release's diffusers model (text-to-image, edits with one or two references, padded batches), the sampling schedule, reference sizing, batching of edit and text-to-image items |
| `test_ltx2.py` | the LTX 2.5 driver on a tiny random DiT: training loss with and without audio, on images, image-to-video's clean first frame; sampling with every sampler and schedule; frame and sigma schedules, halo attention against diffusers' mask, the log-mel input. With the real files, that every tensor maps onto diffusers' modules |
| `test_minimax.py` | MiniMax H3's packed sequences: per-row timesteps, first-frame and reference rows, audio and no audio, and that the sampler lands on the clean latent given the true velocity |
| `test_yue2.py` | music: lyrics split from captions, songs cut into clips, music job settings and sample cards, padded composer conditions, a training step reaching the LoRA and its ComfyUI names, the midpoint solver, saved songs and the audio health check. With the real checkpoint, the prompt format, the VRAM projection and the VAE's loudness envelope |
| `test_refmod.py` | RefMod files against the pack's format (standalone, older, audio and bundle files), grid refinement, the token cap and strength, making them from images, clips and sound, sheets and the plan of what's made, the Sandbox's worker making one, reference audio in the prompt, generating with them, RefMod jobs end to end |
| `test_offload.py` | block swapping gives exactly the same outputs and LoRA gradients as a model held on the GPU, and a finetune's streamed blocks the same weights; streamed text encoders (Low VRAM); shared pinned slabs; parking models and optimizer state mid-training; saved activations in RAM giving the same loss and gradients |
| `test_memory.py` | the shared loader with offloading, that projected weight sizes match a loaded model exactly, how each phase's VRAM and system RAM add up, Low VRAM's projections |
| `test_finetune.py` | stepping each weight in backward matches a normal optimizer step, stochastic rounding is unbiased, optimizer state saves and loads, saved models keep their source's layout, a finetune job end to end and continued, the same job offloaded giving the same weights, the training adapter applying only while training, the server's finetune rules |
| `test_extract.py` | extracted factors are the best approximation of their rank, a tiny int8 finetune extracted exactly, the Sandbox applying it as the finetune, extract jobs finding their checkpoint, the server's rules |
| `test_merge.py` | an exact merge is the stacked LoRAs, a cut one the best approximation of their sum, kohya files, LoKrs only at a rank, whole changes added up, layers another model has left out, the Sandbox applying a merge as its LoRAs, merge jobs, the Merge & extract routes |
| `test_lokr.py` | LoKr factor splits, a LoKr layer against the full Kronecker product, `lokr_w1`/`lokr_w2` keys saved and loaded, the Sandbox applying LoKr files and explaining the ones it can't, a tiny LoKr job |
| `test_slider.py` | the slider step (base predictions with the LoRA off, the guided target, the strength during backward) and a tiny slider job without a dataset |
| `test_sampling.py` | samplers on Gaussian data with a known solution (Heun and DPM++ 2M converge at second order), DPM++ 2M against k-diffusion, the beta schedule |
| `test_sandbox.py` | Sandbox requests per family and workflow, GPU rules shared with the queue, the LoRA stack, parking models in RAM, Low VRAM's VAE, checkpoint grids, the step counter, making RefMods (checks, the clip VRAM warning, sheets, listing) |
| `test_evaluation.py` | the hold-out split, recommendation rules, deterministic validation loss, bleed normalisation, multi-resolution items and per-dataset settings with their cache keys |
| `test_trainer.py` | a tiny stand-in model on the GPU: a job trains, saves, finishes and continues; Evaluate checkpoints; jobs sharing a dataset's encodings; a job with EMA weights saving the average and carrying it on |
| `test_ema.py` | the EMA's warmup decays, swapping the average in and back (also when the body raises), its saved state |
| `test_masks.py` | loss masks: found beside images only, lined up with the crop and flip, in the cache key and the payload, the weighted loss against plain MSE |
| `test_timesteps.py` | the noise range on every family's timestep mode, Min-SNR and cosine-map weights, none applied at a pinned sigma, the Validator drawing inside the range |
| `test_cache.py` | which model files and settings share cache folders, and that a model file is known by its contents |
| `test_hf_assets.py` | Hugging Face files read from the cache offline, downloaded once at the pinned commit, and a clear error offline without them |
| `test_model_files.py` | the families' download lists, the models folder setting, finding files by their contents, downloads pausing, resuming and checking their checksum, gated and impossible downloads, the Turbo-without-adapter warning |
| `test_dataset_check.py` | each kind of health check problem (loss masks among them), and that a file is read once |
| `test_sources.py` | source datasets: files pairing by name across image types and subfolders, the _refs folder without them, which models take them, the health check's pairing findings, the pairs route, jobs keeping them, the VRAM projection counting them |
| `test_clip_split.py` | Split into clips: planning inside shots, cutting at 24 fps with sound, captions and references, cancelling, each video once per epoch |
| `test_clip_editor.py` | the clip editor: a reviewed split cut as planned, then again only where it changed (captions kept, removed clips gone, deleted ones not brought back), crops, splits from before the editor, replanning, checks on edits, and its frames being the clips' own |
| `test_export.py` | exporting datasets: a folder each in one stored .zip, unsafe and repeated names, captions as they are or with the trigger word (a song's on its style tags), masks and references once each |
| `test_captioner.py` | reading LLM replies: thinking left out, cut-off replies retried, errors reported, the contact sheet fallback |
| `test_llama_server.py` | the caption server against a stand-in (`tests/fake_llama_server.py`): command lines, waiting while loading, GPU choice, failed starts with hints, idle stops, handing over the GPU; the text encoder server's GPU rules |
| `test_textgen.py` | which text encoders can write (from header-only copies), OpenAI messages to chat templates, the server's answers, quantized layers expanded only while there's room. With a GPU, a tiny random Qwen3-VL end to end |
| `test_server.py` | the API: settings, job validation, dataset path sandboxing, the host check, sign-in, notifications against a local webhook, jobs that ended while the server was down, evaluations, continuing and editing jobs, disk use and caches, bulk caption edits, worker processes, Auto-Caption and Enhance prompt through the caption server, the loss curves on Queue cards |
| `test_hardware.py` | GPUs from `nvidia-smi`, CPU names and temperatures on Linux (Intel and AMD sensor drivers) and Windows |
| `test_media_download.py` | Add from URL with a stand-in yt-dlp: the command for each choice, files moved into the dataset safely, playlists, cancelling, updating yt-dlp, the network warning |
| `test_db.py` | schema migrations: fresh databases, pre-versioning ones, backups, failed steps, newer databases refused |
| `test_webui.py` | the launchers rebuild the web UI only when its source changes |

## Smoke test

`scripts/smoke.py` runs a tiny job per model family on your GPU, through a
Dainamo server with its own temporary data folder, so your queue, datasets and
settings aren't touched. Run it before a release, or when a real run looks
wrong and you want to know whether it's the engine or the settings.

```bash
.venv/bin/python scripts/smoke.py                      # every family you have a profile (or the files) for
.venv/bin/python scripts/smoke.py krea2 yue2 --gpu 1
.venv/bin/python scripts/smoke.py --profile "Krea 2 Turbo"
```

For each family it makes a two-file dataset (images, clips with sound, or
songs), trains a rank-4 LoRA for 20 steps, stops it partway and resumes it, and
checks that:

- the loss is a real number at every step;
- Resume carries on from the step it stopped at;
- the LoRA has no NaN or infinite weights, and training changed them;
- a sample renders;
- the Sandbox matches every layer of the LoRA to the model and generates with it.

It uses your first profile of each family, or the files Model files finds in
your models folder. It won't use a GPU that already has something on it
(`--force` overrides that). A first load of a big model from a slow disk can
take half an hour, so test one family at a time while you work, and all of them
when the GPU is free. YuE2 takes about 2 minutes once its file is cached. If a
check fails, it exits 1 and keeps the temporary folder, with the server's log
and each job's log. CI doesn't run it, since CI has no GPU.

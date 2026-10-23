# Profiles and model files

## Profiles

A profile names the files to train with: the diffusion model, text encoder, VAE
and an optional training adapter. Any precision and file name works. The form
reads each file's tensors, so a wrong file is caught before training. Profiles
are grouped by model family, here and in every profile dropdown.

Each file field has a **Download** button, and lists matching files on this
machine when it's empty or the file is wrong. You can save the profile while its
download runs.

**Turbo models** (Krea 2 Turbo, Qwen-Image 2.1 Turbo) need Ostris's training
adapter: training one without it breaks its few-step sampling. A profile whose
diffusion model's file name contains "turbo" and has no adapter gets a warning.
Use **Download** to put the adapter in your `loras` folder. An `owner/repo/file`
path also works, but downloads into the hidden Hugging Face cache, where ComfyUI
doesn't look.

[Merge & extract](merge-and-extract.md), the third tab, merges LoRAs and
extracts them from finetuned models.

## Model files

**Model files**, the Profiles page's second tab, shows what each family needs
and which of those files you have. It looks in the ComfyUI models folder set in
**Settings → Model files** (`diffusion_models` or `unet`, `text_encoders` or
`clip`, `vae`, `loras`, `checkpoints`) and at the files your profiles use. Files
are recognised by their contents, whatever they're called. Until the folder is
set, only your profiles' files are searched; **Use this folder** sets it to the
one they're in.

**Create profile** starts a profile with a family's files filled in.

### Downloads

**Download** gets a file from Hugging Face in a precision Dainamo can read (the
recommended one is marked), into the models folder's subfolder for its kind, so
ComfyUI sees it too. You can choose another folder.

- Files come from a pinned commit of their repo, so a later change to the repo
  can't swap them.
- Free space is checked first, leaving 2 GB spare.
- Downloads run one at a time into a `.part` file, renamed once its size and
  SHA-256 match Hugging Face's. A paused or cut-off download resumes where it
  stopped, even after a restart.
- LTX 2.5 is gated: accept its terms on Hugging Face and sign in on this
  machine (`hf auth login`, or `HF_TOKEN`). Its repo publishes no checksums, so
  its files are checked by size and header only.

New releases appear only once they're added to the family's driver
(`downloads`).

## Files fetched on first use

Most families also need a tokenizer, chat template and configs from their
Hugging Face repo. They're downloaded the first time a job or the Sandbox uses
the family, then read from the Hugging Face cache, so jobs run offline. The same
goes for DINOv2 (concept bleed), Whisper, and a training adapter given as
`owner/repo/file`. LTX 2.5 and YuE2 need none.

Each repo is pinned to a commit (`REVISIONS` in `dainamo/engine/hf_assets.py`),
so an update can't change how captions are read partway through a project.

See [Model families](model-families.md) for what each family needs and the
weight formats Dainamo reads.

# Profiles and model files

The **Models** page has three tabs: **Profiles**, **Model files** and
**Merge & extract**.

## Profiles

A profile names the files to train with: the diffusion model, text encoder, VAE
and an optional training adapter. Any precision and file name works. The form
reads each file's tensors, so a wrong file is caught before training. Profiles
are grouped by model family in every profile dropdown.

**New profile**, on the Profiles tab, starts one. Each profile's card shows its
family and files. **Search** matches profile names, families and file names, so
it finds the profiles that use a file; the family menu shows one family's
profiles. A card's **⋯** duplicates or deletes the profile. Deleting leaves the
model files, and jobs made with it keep their settings.

Each file field has a **Download** button, and lists matching files on this
machine when it's empty or the file is wrong; one from the Hugging Face cache is
tagged **HF cache**. You can save the profile while its download runs.

**Turbo models** (Krea 2 Turbo, Qwen-Image 2.1 Turbo) need Ostris's training
adapter: training one without it breaks its few-step sampling. A profile whose
diffusion model's file name contains "turbo" and has no adapter gets a warning.
Use **Download** to put the adapter in your `loras` folder. An `owner/repo/file`
path also works, but downloads into the hidden Hugging Face cache, where ComfyUI
doesn't look.

[Merge & extract](merge-and-extract.md), the third tab, merges LoRAs and
extracts them from finetuned models.

## Model files

**Model files**, the Models page's second tab, shows what each family needs
and which of those files you have. It searches two places, and the files your
profiles use:

- the ComfyUI models folder: `diffusion_models` or `unet`, `text_encoders` or
  `clip`, `vae`, `loras` and `checkpoints`;
- the Hugging Face cache (`HF_HUB_CACHE`, else the `hub` folder in `HF_HOME`,
  else `~/.cache/huggingface/hub`), each repo at the version its `refs/main`
  names.

Files are recognised by their contents, whatever they're called.

In the Hugging Face cache, a folder of shards (one with
`model.safetensors.index.json`) is one model, named after its repo, such as
`Qwen/Qwen3-4B`. A diffusers pipeline's subfolders are models too
(`owner/repo/text_encoder`), and so is any other `.safetensors` file. What it
finds there is text encoders, VAEs in a layout Dainamo reads, and single-file
models such as ComfyUI's and Black Forest Labs' releases. Diffusion models in
the diffusers layout (a pipeline's `transformer/` folder) aren't found: their
tensors have other names. A repo still downloading is skipped until its shards
are all there. Files from the cache are tagged **Hugging Face cache**, with
their repo, and picking one in a profile puts in its path (the repo folder, for
a folder of shards).

**Model folder**, at the top, is the one place the models folder is set
(Settings links here). Until it's set, only the Hugging Face cache and your
profiles' files are searched, and it offers the folder your profiles' files are
in: **Use this folder** sets it, or **Choose another**. The card also shows the
cache's path, and whether this machine is signed in to Hugging Face.

Each family's card says whether its required files were found. A field with
more than one file found lists them under **View files**; hover a file for its
path and precision. **Search** matches families, fields and file names, and the
status menu shows only the families that are ready, missing files, or not
checked. **Create profile** starts a profile with a family's files filled in,
taking the models folder's file where both places have one. A row no profile
uses is marked optional: FLUX 3 Image Klein's **Prompt upsampler**, for
captions (see [The FLUX 3 style](captions.md#the-flux-3-style)).

The **Upscalers** card lists the files in the models folder's `upscale_models`
and what each one is, and downloads Real-ESRGAN and SeedVR2 into it (see
[Upscaling](upscaling.md)).

### Downloads

The download button beside a field gets a file from Hugging Face in a precision
Dainamo can read (the recommended one is marked), into the models folder's
subfolder for its kind, so ComfyUI sees it too. Downloads never go into the
Hugging Face cache, where ComfyUI doesn't look. You can choose another folder. A
file already in the cache counts as on this machine, so it isn't downloaded
twice.

- Files come from a pinned commit of their repo, so a later change to the repo
  can't swap them.
- Free space is checked first, leaving 2 GB spare.
- Downloads run one at a time into a `.part` file, renamed once its size and
  SHA-256 match Hugging Face's. A paused or cut-off download resumes where it
  stopped, even after a restart.
- Some files are gated (FLUX.2 Klein 9B, FLUX 3 Image Klein, LTX 2.5): accept
  the terms on the repo's Hugging Face page, and sign in on this machine before
  starting Dainamo, with `hf auth login` or by setting `HF_TOKEN`. LTX 2.5's
  repo publishes no checksums, so its files are checked by size and header
  only.

New releases appear only once they're added to the family's driver
(`downloads`).

A text encoder can also be a Hugging Face folder of shards (one with
`model.safetensors.index.json`, or a `models--owner--repo` folder of the
Hugging Face cache). Model files finds the ones in the cache; type the path of
one elsewhere into the profile. FLUX.2 Klein's Qwen3-4B and Qwen3-8B load that
way straight from `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`.

## Files bundled, and files fetched on first use

Most families also need a tokenizer, chat template and configs from their
Hugging Face repo. Krea 2's, MOSS-Image's and Qwen-Image 2.1's ship with the
app (`dainamo/assets/`, about 45 MB), so those families run without a network
from the first job. MiniMax H3's are downloaded the first time a job or the
Sandbox uses it, then read from the Hugging Face cache, so later jobs run
offline: its license doesn't allow bundling them. The same goes for DINOv2
(concept bleed), Whisper, and a training adapter given as `owner/repo/file`.
LTX 2.5 and YuE2 need none.

Each repo is pinned to a commit (`REVISIONS` in `dainamo/engine/hf_assets.py`),
so an update can't change how captions are read partway through a project.

See [Model families](model-families.md) for what each family needs and the
weight formats Dainamo reads.

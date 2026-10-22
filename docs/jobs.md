# Jobs

| Create Job | Queue |
| --- | --- |
| ![Create Job with a dataset card, its health check and the summary](screenshots/create-job.png) | ![The queue with four finished jobs](screenshots/queue.png) |

## Create Job

Pick a profile, the datasets and the training settings, then add the job to the
queue. Each dataset is a card with its own:

- **repeats**: how much it counts against the other datasets;
- **resolutions**: with several, each file trains at each size, bucketed by
  aspect ratio, so the LoRA works across sizes. Files are never upscaled;
- **caption dropout**: the chance a step uses an empty caption, which keeps the
  LoRA working without its trigger words;
- **flip**: also train on mirror images.

Each card runs the dataset's [health check](datasets.md#health-check) for this
job.

**Video models** can mix image and clip datasets (say, stills at 1024 and clips
at 384 and 512). A clip trains its first frames up to the clip length, never
sped up or slowed down. **Auto**, beside the clip length, trains each clip whole
instead, up to 241 frames (10 s). To use more of long videos, see
[Split into clips](datasets.md#split-into-clips-video-models).

VRAM is set by the biggest step: the largest size, and for clips the clip
length. For **Layer offloading**, **Fit to GPU** and **Low VRAM**, see
[Memory](memory.md).

Besides a LoRA, a job can make:

- a LoKr: **Network** under Training ([LoKr](lokr.md));
- a slider, for image models: **Slider** under Train ([Sliders](sliders.md));
- a finetune, for MOSS-Image: **Finetune** under Train
  ([Full finetuning](finetuning.md));
- a RefMod, for MiniMax H3: **RefMod** under Make ([RefMods](refmods.md)).

**Sample prompts** are cards. Width, height, seed and LoRA strength left blank
use the Sampling defaults.

## Queue

Each job's card shows a small loss curve once it has 10 steps, so you can spot a
run going wrong without opening it. **Settings** sets the queue mode (one job at
a time, several on one GPU, or spread across GPUs) and which GPUs to use.

## The job page

The tabs are **Samples**, **Loss**, **Checkpoints** and **Log**. In Samples,
click a sample to enlarge it; the arrow keys step through training
(left/right) and prompts (up/down). Checkpoints has the saved checkpoints with
download links, the checkpoint picker's results and the job's disk use. A
failed job opens on Log.

You can stop a job (it saves its state), resume or restart it, take samples or a
checkpoint now, or duplicate its settings into a new job.

**Edit** opens a job that isn't running in the Create Job form. Before it has
trained, anything can change. After that, it carries on from its last save with
the changes: datasets, clip length, resolution, batch size, learning rate and
schedule, steps, samples and memory settings. The model, network (LoRA or LoKr,
rank, alpha, or a finetune), optimizer and slider can't change, since the saved
state wouldn't fit them. While the checkpoint picker or concept bleed is on,
what it scores can't change either (for the picker: datasets, resolution, clip
length and seed). Changing one of these restarts the job from step 0, after
asking.

**Continue training** gives a finished job more steps, picking up with the same
optimizer state. Its final LoRA so far is kept as `name_002000.safetensors`,
which "keep last N" never deletes. A cosine or linear schedule stretches to the
new length.

**Open in Sandbox**, on an enlarged sample, loads its prompt, seed, size,
sampler and checkpoint into the Sandbox, so Generate re-creates it.

## Shared cache and disk use

Encoded captions and latents are cached in `data/cache/` and shared between
jobs. A job reuses what another encoded with the same VAE or text encoder file
and settings, and encodes only new or changed files and captions. A copy of a
model file elsewhere counts as the same file; another precision or a fine-tune
doesn't.

The Queue page shows each job's disk use and the shared cache's, and clears the
cache of everything no queued or running job uses. A completed job's training
state can be deleted, after which it can't continue.

## Checkpoint picker and concept bleed

Both are off by default; turn them on in section 06 of Create Job. They run at
each save, and at step 0 as a baseline.

**Best checkpoint picker.** Every checkpoint is scored on validation images,
held out of training or taken from a separate dataset, with the same noise and
timesteps each time. The curve turns upward when the LoRA starts to overfit. The
checkpoint with the lowest validation loss is recommended, and "keep last N"
never deletes it. Optionally, it's rendered at several LoRA strengths at the
end.

**Concept bleed detector.** Prompts without the trigger word are rendered by the
base model and each checkpoint with a fixed seed, and compared with DINOv2 image
features (`facebook/dinov2-small`, downloaded once):

- **Bleed**: how far those images moved toward your training images, from 0%
  (no closer than the base model's) to 100% (as close as your own images). Runs
  with no leakage scored 0–9% in testing, and a small copy of the subject pasted
  into an image about 30%. The default limit is 25%; the recommendation skips
  checkpoints above it. Some bleed is expected for style LoRAs.
- **Drift**: how much those images changed at all, where 100% is as much as a
  new seed. High drift means the LoRA alters unrelated images even when your
  concept isn't leaking. A pure colour cast shows up mostly as drift.

**Evaluate checkpoints**, on a finished job's page, runs either check on the
saved checkpoints afterwards, through the queue. The job keeps its outcome and
its results from training. A new evaluation replaces the previous one.

- The picker needs images the LoRA didn't train on. If the job held none out,
  choose a dataset of other pictures of the same subject: scored on its own
  training images, the last checkpoint would always look best.
- Only checkpoints still on disk can be scored. Set **Keep last N saves** to 0
  (keep all) on a job you might evaluate later.

## Notifications

When a job ends, the open app shows a message. Browser notifications work only
on `http://127.0.0.1`, `localhost` or HTTPS, not a plain LAN address.

The server can also post to **ntfy**, **Discord**, **Slack**, or any URL as
**JSON** (`{event, title, message, url, job_id, status}`), for the endings you
choose: finished, failed (or interrupted), stopped. Messages carry the step,
average loss, duration and LoRA file name, or the error. Set the app's address
to link to the job. **Send a test** checks the setup.

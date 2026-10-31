# Jobs

| New job | Queue |
| --- | --- |
| ![New job with a dataset card, its health check and the summary](screenshots/create-job.png) | ![The queue with four finished jobs](screenshots/queue.png) |

## New job

Pick a profile, the datasets and the training settings, then add the job to the
queue. **Add to queue** leaves it there ready to start, until you press
**Start** on it (or **Start all** when several are ready), so you can line up
jobs and check them first; **Add and start** queues it to run at once. The
sidebar sums up the run (its datasets, epochs, saves and file size)
and estimates its memory; when **Add to queue** is greyed out, the line under it
says what's missing.

Each dataset (**Add dataset** for more) has its own:

- **repeats**: how much it counts against the other datasets;
- **resolutions**: with several, each file trains at each size, bucketed by
  aspect ratio, so the LoRA works across sizes. Files are never upscaled;
- **caption dropout**: the chance a step uses an empty caption, which keeps the
  LoRA working without its trigger words;
- **flip**: also train on mirror images;
- **source datasets**, for an edit model (Qwen-Image 2.1) or MiniMax H3's
  ref2va: the images its files pair with by name, the sources of an edit
  ([Edit datasets](datasets.md#edit-datasets-sources-and-results)). Leave it
  empty to train text-to-image;
- **a caption for the files without one** (shown when some have none): for an
  edit model, the instruction ("Make it night") when every pair makes the same
  edit. A file's own caption is used where it has one, and the trigger word is
  added to either. Blank, those files train on the trigger word only.

Each runs the dataset's [health check](datasets.md#health-check) for this job.

**Video models** can mix image and clip datasets (say, stills at 1024 and clips
at 384 and 512). A clip trains its first frames up to the clip length, never
sped up or slowed down. **Auto**, beside the clip length, trains each clip whole
instead, up to 241 frames (10 s). To use more of long videos, see
[Split into clips](datasets.md#split-into-clips-video-models).

VRAM is set by the biggest step: the largest size, and for clips the clip
length. For **Layer offloading**, **Fit to GPU** and **Low VRAM**, in the
sidebar's **Memory estimate**, see [Memory](memory.md).

Besides a LoRA, a job can make:

- a LoKr: **LoKr** under Training method ([LoKr](lokr.md));
- a slider, for image models: **Slider** under Training method
  ([Sliders](sliders.md));
- a finetune, for MOSS-Image and Krea 2: **Finetune** under Training method
  ([Full finetuning](finetuning.md));
- a RefMod, for MiniMax H3: **RefMod** under Training method
  ([RefMods](refmods.md)).

**Sample prompts** are cards. Width, height, seed and LoRA strength left blank
use the Sampling defaults. A video model's card also has **Frames**: blank is
the family's default, shown in the field, and 1 renders a still image. The VRAM
projection counts the largest card. **Likeness score** (on by default for image and
video models) measures how much each sample looks like the training images,
from 0% (the base model's render of the card) to 100% (as alike as the training
images are to each other), and flags a sample that's a near copy of one training
image. See [Likeness](likeness.md).

**Masked loss**, below the datasets (image models), trains each image where its
mask is white ([Masks](datasets.md#masks)). **Outside the mask** is how much the
rest counts: 0 ignores it, and the default 0.1 keeps a little, so the LoRA
doesn't learn just anything there.

**EMA weights**, under Saving, keeps an average of the LoRA's recent weights.
**EMA decay** 0.99 averages about the last 100 steps, 0.999 about the last
1000. Saves, samples, the checkpoint picker and concept bleed use the average,
which is steadier than any single step. Training carries on from the live
weights, and the training state keeps both. A finetune can't use it: an average
of the whole model wouldn't fit beside it.

**Noise range**, in Advanced training settings, limits the noise levels the
steps are drawn from: 0 is the clean image, 1 pure noise. Narrowing it focuses
training: 0.3 to 1 for composition and style, 0 to 0.5 for detail. **Loss
weighting** counts each step by its noise level. **Min-SNR** caps how much the
low-noise steps count (at **Min-SNR gamma**, 5 by default), and counts the
noisiest ones little too. **Cosine map** is SD3's weighting, a gentler lean
toward the middle levels. A slider learns from its prompts, with its own loss:
of these settings, it has EMA weights and the noise range.

## Queue

New jobs sit under **Ready to start** until you start them; a started job goes
behind the jobs already waiting. Merges and LoRA extractions, resumes and
restarts join the queue straight away. Each job's card shows a small loss curve once it has 10 steps, so you can spot a
run going wrong without opening it. **Settings** sets the queue mode (one job at
a time, several on one GPU, or spread across GPUs) and which GPUs to use. Jobs
start in order; one pinned to a GPU that's busy or held by the Sandbox lets the
next in line go first.

**Select** puts a checkbox on each job, to delete several at once, optionally
with their files (LoRAs or finetuned models, samples and training state). A
running job must be stopped first.

## The job page

The tabs are **Samples**, **Loss**, **Checkpoints** and **Log**. In Samples,
click a sample to enlarge it; the arrow keys step through training
(left/right) and prompts (up/down). Each sample carries its
[likeness](likeness.md) to the training images, and the enlarged view can
compare it with any dataset. Loss has the loss curve and, under it, the
likeness curve. Checkpoints has the saved checkpoints with download links, the
checkpoint picker's results and the job's disk use. A failed job opens on Log.

A job's files are in `data/jobs/<name>_<id>/`, named when the job is added;
renaming the job later doesn't move them. Jobs added before folders had names
keep `data/jobs/<id>/`.

When a job completes, its final LoRA is also put in the output folder, `output/`
beside `data/`, as `<name>.safetensors`. Settings › Output folders can point it
somewhere else: ComfyUI's `models/loras`, say, so ComfyUI finds it straight away.
A finetune's model goes in the folder for finetuned models (ComfyUI's
`models/diffusion_models`, if you choose). It's a hardlink when the folder is on
the same disk as the job, so it takes no more space, and a copy otherwise. **To
output** on the Checkpoints tab puts any other checkpoint there too. When a job
completes again (trained on, or made again), the file it put there is replaced.
A file of that name that came from somewhere else is never replaced: the job's
file is called `<name> (2).safetensors` instead. Deleting a job, even with its
files, leaves what it put in the output folder.

On its page you can stop a job (it saves its state), start it when it's ready, cancel it while it waits,
force a stopping one to stop now, resume it from its last save or restart it
from step 0 (after asking), take samples or a checkpoint now, duplicate its
settings into a new job, or delete it. A job with no save to resume from offers
only Restart. A finetune's page also extracts a LoRA from one of its
checkpoints ([Merge and extract](merge-and-extract.md)).

A stopping job finishes the sample or save it's in first. It's forced to stop
only after three minutes in which it writes nothing, so a slow disk doesn't cut
a save short. A job the system kills unasked, such as by running out of memory,
is marked failed and sends the failure notification.

**Edit** opens a job that isn't running in the New job form. Before it has
trained, anything can change. After that, it carries on from its last save with
the changes: datasets, clip length, resolution, batch size, learning rate and
schedule, steps, samples, memory settings, masked loss, EMA weights, the noise
range and loss weighting. EMA turned on part way averages from the weights at
that point. The model, network (LoRA or LoKr, rank, alpha, or a finetune),
optimizer and slider can't change, since the saved state wouldn't fit them.
While the checkpoint picker or concept bleed is on, what it scores can't change
either (for the picker: datasets, resolution, clip length, seed, timestep
sampling, noise range and masked loss). Changing one of these restarts the job
from step 0, after asking. The model is the profile's family, diffusion model
file, training adapter and variant: renaming the profile, or picking another
profile with the same model and other sampling settings, carries on. The job
keeps training with the model files it started with.

**Continue training** gives a finished job more steps, picking up with the same
optimizer state. Its final LoRA so far is kept as `name_002000.safetensors`,
which **Keep last N saves** never deletes. A cosine or linear schedule
stretches to the new length.

**Open in Sandbox**, on an enlarged sample, loads its prompt, seed, size,
sampler and checkpoint into the Sandbox, so Generate re-creates it.

## Shared cache and disk use

Encoded captions and latents are cached in `data/cache/` and shared between
jobs. A job reuses what another encoded with the same VAE or text encoder file
and settings, and encodes only new or changed files and captions. A copy of a
model file elsewhere counts as the same file; another precision or a fine-tune
doesn't.

A file's latents are known by its dataset and where the file is inside it, so
they survive the dataset's folder moving. If you move a dataset's folder, use
**Change folder** in Datasets to say where it is now. A job using it follows it
there when it next runs, resuming with the latents it already cached. Until then,
Resume says which dataset's folder is missing.

Each job's card on the Queue page shows its disk use. Below the jobs, the
Queue shows the total: everything the jobs keep, the cache, and the outputs
(LoRAs, finetuned models and RefMods).
**Manage cache** shows how much of the cache queued or running jobs use, and
clears the rest. A completed job's training state can be deleted on its page,
after which the job can't continue.

## Checkpoint picker and concept bleed

Both are off by default; turn them on in New job's **Checkpoint picker &
concept bleed** section. They run at each save, and at step 0 as a baseline.

**Checkpoint picker.** Every checkpoint is scored on validation images,
held out of training or taken from a separate dataset, with the same noise and
timesteps each time. The curve turns upward when the LoRA starts to overfit. The
checkpoint with the lowest validation loss is recommended, and **Keep last N
saves** never deletes it. Optionally, it's rendered at several LoRA strengths at
the end.

**Concept bleed.** Prompts without the trigger word are rendered by the
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
to link to the job. **Send a test** checks the setup. It reports the reply's
status, not its text, and a webhook URL that redirects isn't followed: the
error names the new address to use instead.

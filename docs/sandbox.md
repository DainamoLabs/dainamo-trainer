# Sandbox

![The Sandbox comparing a LoRA with the base model](screenshots/sandbox.png)

The Sandbox generates with a profile and LoRAs, to test them.

- **Workflow**, grouped into Image, Video and Music: only the ones the
  profile's model supports can be chosen.
- **Size**: a shape and megapixels, where a megapixel is 1024×1024 and sizes
  snap to 64 (1 MP at 9:16 is 768×1344). **Match input** takes the input
  image's shape; **Custom** takes a width and height.
- **LoRAs**: up to 8, each with a strength: any job's checkpoints (the
  recommended one is starred) or any ComfyUI/PEFT or kohya-style LoRA or
  [LoKr](lokr.md) file.
- **Diffusion model**, for each model that can be
  [finetuned](finetuning.md): the profile's, or a finetune job's checkpoint.
- Reference-to-Video with MiniMax H3 also takes [RefMods](refmods.md) and
  reference audio; **Make RefMod…** makes one of images and clips, or a sheet
  of images, without unloading the model. It works with an fl2va profile too,
  but fl2va wasn't trained on references, so it may follow them less closely
  than ref2va.

**Image-to-Image**, for every image model, starts from an input image instead
of noise. The image is cropped to the output's shape (**Match input** keeps all
of it) and noised to the **Strength**, then the steps run from there: low keeps
the picture and changes its look, 0.6 keeps the layout, and 1 ignores it. The
steps are the ones asked, spaced as the model's schedule spaces them below the
strength. It's separate from an edit model's Image Edit, and takes no
references. Enhance prompt describes what the image should become.

**Hi-res fix**, under Generation for images, renders at the size chosen, makes
the result 1.5× or 2× bigger, then renders it again at that size from the
bigger image with the same prompt and seed, so the detail is drawn at full
size. The image is made bigger with an upscaler from `upscale_models`
([Upscaling](upscaling.md)), the one an upscale keeps loaded, or **Resize
only** (Lanczos). **Second pass strength** (0.35 by default) is how much of the
bigger image is redone, and **Second pass steps** (blank: the run's) how many
steps that takes. The first pass is kept too, and shows beside the result with
its size. With **Compare without LoRAs** both renders get the second pass;
**Compare checkpoints** shows only the results. The final size is held to 4096
pixels a side. If the upscaler hasn't room beside the diffusion model (SeedVR2),
the model waits in system RAM while it runs, and SeedVR2 is unloaded before the
second pass. History notes such runs as "i2i 0.6" and "hi-res 2×".

**Upscale**, on an image or video output, makes it bigger with an upscaler
(Real-ESRGAN, SeedVR2 or another file in the models folder's `upscale_models`),
and **Upscale a file**, below History, does the same for a file on this machine
or one you upload. See [Upscaling](upscaling.md).

**Compare without LoRAs** renders each seed again without them. **Compare
checkpoints** renders one prompt and seed with each of a job's chosen
checkpoints (up to 12) at each strength (up to 6, from -10 to 10), optionally
with one without LoRAs.

**Enhance prompt** has the caption provider
([Caption provider](captions.md#caption-provider)) rewrite the prompt in detail
before generating, looking at the input images when the model can see. The
rewrite is reused while the prompt, seed, workflow and images stay the same. Its
instructions are edited in Settings.

**Generation** holds the size, steps, CFG, seed (the button beside it turns on
a new random one each run) and how many to make per run. **Advanced settings** holds
the sampler, scheduler and shift, the GPU and layer offloading; it stays shut
unless you open it, and when shut it warns **Won't fit in VRAM** or **Tight on
VRAM** if the offloading setting needs a look. Ctrl+Enter generates. When
**Generate** is greyed out, the line under it says why.

The latest run shows each seed with the LoRAs beside the render without them, if
compared. **History** keeps every run, newest first, numbered in the order they
were made: click one to see it again, and **Reuse settings** puts its settings
back in the form. An upscale shows there too, with the file it came from beside
the result. A run's delete button (on hover, or always on a touch screen) asks
first, then deletes its outputs from disk.

## GPUs

Models stay loaded between generations. With two or more GPUs, the Sandbox holds
the GPU it runs on, and the queue uses the others until you release it or it's
idle for the time set in Settings. With one GPU, it runs only while no job is
running or queued, and a job ready to start makes it unload after the current
generation.

It has the same layer offloading slider and Low VRAM switch as New job
([Memory](memory.md)), under Advanced settings.

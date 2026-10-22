# Sandbox

![The Sandbox comparing a LoRA with the base model](screenshots/sandbox.png)

The Sandbox generates with a profile and LoRAs, to test them.

- **Workflow**: only the ones the profile's model supports can be chosen.
- **Size**: a shape and megapixels, where a megapixel is 1024×1024 and sizes
  snap to 64 (1 MP at 9:16 is 768×1344). **Match input** takes the input
  image's shape; **Custom** takes a width and height.
- **LoRAs**: up to 8, each with a strength: any job's checkpoints (the
  recommended one is starred) or any ComfyUI/PEFT or kohya-style LoRA or
  [LoKr](lokr.md) file.
- **Diffusion model**, for MOSS-Image: the profile's, or a
  [finetune](finetuning.md) job's checkpoint.
- Reference-to-Video with MiniMax H3 also takes [RefMods](refmods.md) and
  reference audio.

**Compare with base** renders each seed again without the LoRAs. **Compare
checkpoints** renders one prompt and seed with each of a job's chosen
checkpoints (up to 12) at each strength (up to 6), optionally with the base
model.

**Enhance prompt** has the caption LLM ([Caption provider](captions.md#caption-provider))
rewrite the prompt in detail before generating, looking at the input images when
the model can see. The rewrite is reused while the prompt, seed, workflow and
images stay the same. Its instructions are edited in Settings.

## GPUs

Models stay loaded between generations. With two or more GPUs, the Sandbox holds
the GPU it runs on, and the queue uses the others until you release it or it's
idle for the time set in Settings. With one GPU, it runs only while no job is
running or queued, and a job ready to start makes it unload after the current
generation.

It has the same layer offloading slider and Low VRAM switch as Create Job
([Memory](memory.md)).

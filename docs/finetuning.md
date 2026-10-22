# Full finetuning (MOSS-Image)

A finetune trains every weight of the diffusion model instead of an adapter. It
can learn more than a LoRA, which matters when one model should learn many
concepts or a lot of data. Training several concepts together, each with its
own trigger, also avoids the interference of stacking separate LoRAs.

Choose **Finetune** under Train in Create Job (MOSS-Image profiles). It starts at
a learning rate of 1e-5, with Adafactor, keeping the last 2 saves.

Each save is a whole bf16 diffusion model laid out like the profile's file,
which ComfyUI loads from `models/diffusion_models` in place of the original. A
MOSS-Image save is about 10.2 GB, so keep few. The Sandbox's **Diffusion
model** lists a finetune job's checkpoints.

Start from the bf16 release if you can: a quantized file is expanded to bf16 to
train, which works, but trains its rounding in. Resuming or **Continue
training** carries on from the latest save, not the profile's file.

## What's different from a LoRA

To fit in 24 GB, each weight is updated as soon as its gradient is ready, with
stochastic rounding to bf16. So:

- **Optimizer**: Adafactor (about 10 MB of state), or AdamW 8-bit, which needs
  another 10.2 GB, about 24 GB in all at 1024: too much for a 24 GB card.
- No gradient accumulation, no overall gradient clipping, and no offloading.
- Concept bleed, the strength sweep and Evaluate checkpoints don't apply; the
  checkpoint picker does. A profile's training adapter isn't used.

On an RTX 3090, a MOSS-Image finetune at 1024×1024 took 4.6 s a step and peaked
at 13.1 GB.

## LoRA from a finetune

**Extract LoRA**, beside a finetune job's checkpoint, saves what that checkpoint
changed from the profile's model as a LoRA: a much smaller file to use on the
original model, in the Sandbox or ComfyUI, and to combine with other LoRAs.

Choose a rank (the form shows the file size: about 290 MB for MOSS-Image at 64)
and a name. It runs as a job of its own, using about 3 GB of VRAM; the time is
mostly reading the two models. Changes to norms and biases are kept whole
(`.diff`, `.diff_b` keys, which ComfyUI and the Sandbox apply).

The job's page says how much of the change the LoRA keeps. If the result looks
different from the finetune, render the same prompt and seed with both, and
extract again at a higher rank.

Extracting is refused if the original model's file has moved, since the LoRA
would be the change from the wrong model. A warning is logged if the checkpoint
was finetuned from another file.

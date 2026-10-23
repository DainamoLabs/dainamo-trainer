# Full finetuning (MOSS-Image, Krea 2)

A finetune trains every weight of the diffusion model instead of an adapter. It
can learn more than a LoRA, which matters when one model should learn many
concepts or a lot of data. Training several concepts together, each with its
own trigger, also avoids the interference of stacking separate LoRAs.

Choose **Finetune** under Training method in Create Job (MOSS-Image and Krea 2 profiles).
It starts at a learning rate of 1e-5, with Adafactor, keeping the last 2 saves
(Krea 2: the last one).

Each save is a whole bf16 diffusion model laid out like the profile's file,
which ComfyUI loads from `models/diffusion_models` in place of the original. A
MOSS-Image save is about 10.2 GB and a Krea 2 save about 26 GB, so keep few, and
check your free disk space first. The Sandbox's **Diffusion model** lists a
finetune job's checkpoints.

A Turbo model trains with the profile's training adapter, as a LoRA does. The
finetune learns on the de-distilled model, and is saved and sampled without the
adapter, so it keeps Turbo's few-step sampling.

Start from the bf16 release if you can: a quantized file is expanded to bf16 to
train, which works, but trains its rounding in. Resuming or **Continue
training** carries on from the latest save, not the profile's file.

## What's different from a LoRA

To fit in 24 GB, each weight is updated as soon as its gradient is ready, with
stochastic rounding to bf16. So:

- **Optimizer**: Adafactor (10–20 MB of state), or AdamW 8-bit, which keeps
  2 bytes a weight on the GPU: another 10.2 GB for MOSS-Image, too much for a
  24 GB card at 1024, and 26 GB for Krea 2.
- No gradient accumulation and no overall gradient clipping.
- Concept bleed, the strength sweep and Evaluate checkpoints don't apply; the
  checkpoint picker does.

**Layer offloading** works as it does for a LoRA, and Krea 2 needs it: 26 GB in
bf16 doesn't fit on a 24 GB card. An offloaded block is updated while it's on
the GPU for the backward pass, then copied back to system RAM, so it crosses the
PCIe bus three times a step instead of twice. Use **Fit to GPU**. Offloading a
finetune needs gradient checkpointing.

On an RTX 3090, a MOSS-Image finetune at 1024×1024 took 4.6 s a step and peaked
at 13.1 GB. A Krea 2 Turbo finetune at 1024×1024, from the fp8 file with 16 of
its 28 blocks offloaded (Fit to GPU's choice), took about 11.5 s a step on a PCIe
4.0 x4 link, with 14 GB of the model in pinned system RAM. It allocated 17.5 GB
of VRAM at most, so the projection, at 22.7 GB, is on the safe side.

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

To extract from a finetuned model Dainamo didn't make, or to merge the LoRA
with others, use [Merge & extract](merge-and-extract.md).

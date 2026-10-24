# Merge & extract

**Merge & extract**, the Models page's third tab, makes LoRAs from files you
already have. Each one runs in the queue as a job of its own: you see its
progress and log there, it gets a folder in `data/jobs/`, and its LoRA is listed
in the Sandbox with the family's others. Neither loads a model or trains
anything.

Choose the profile first. The LoRAs' layers are checked against its diffusion
model, shown beside it as **Base model file** (read from the file's header
only), and
what's made is listed with its family. Each form's **Output name** is the file's
name, `.safetensors` added; left empty, it's named after its inputs. **How
merging works** and **How extraction works** fold the details away. The button
under each form, **Add to queue**, says what's missing until it can be queued.

## Merge LoRAs

Merging saves what several LoRAs do together as one LoRA, each at its own
strength. Choose a job's LoRAs from the list, or **Browse files** (or **Other
file…**) for others. A negative strength takes a LoRA's change away. The line under each
LoRA says its rank and how many of its layers the profile's model has. Layers
the model doesn't have are left out, and a LoRA that matches none of them is
refused, since it's for another model. Dainamo's LoRAs and LoKrs work, and so do
other trainers' (PEFT `lora_A`/`lora_B` or kohya `lora_unet_…` keys). DoRAs
don't.

**Output rank** sets what's kept:

- **Exact** keeps all of every LoRA. Their ranks add up: a rank 16 and a rank 32
  LoRA merge into a rank 48 one, about as big as the two files together. In the
  Sandbox at strength 1, it gives what the LoRAs give together at their
  strengths.
- **A rank** cuts each layer's merged change to that rank: its best
  approximation of that rank, in a smaller file. The job's page says how much of
  the merged change it keeps, overall and in the layer that keeps least. If the
  result falls short of the LoRAs themselves, merge again at a higher rank.

With one LoRA, merging cuts it to a smaller rank, or bakes its strength into it
(Exact). A LoKr has no rank to add up, so it's merged at a rank. Changes to
norms and biases (from an extracted LoRA) are added up whole.

It takes seconds to a few minutes, a layer at a time on the GPU.

## Extract a LoRA

Extracting saves what a finetuned model changed from the model it was finetuned
from, as a LoRA to use on that model. It's the same form as **Extract LoRA** on
a finetune job's checkpoints
([Full finetuning](finetuning.md#lora-from-a-finetune)), for any finetuned file: choose one of the family's finetune checkpoints, or
**Other file…** for another trainer's.

**Base model file** (optional) is the original model. Leave it empty to use the
profile's diffusion model (for a finetune job's checkpoint, the one its job
trained from). The two have to be the same kind of model, with tensors named the
same way, or extracting is refused. Add the finetune's trigger words if it
has any: they're saved in the LoRA, as a trained LoRA's are.

## How it works

Merging (`engine/merge.py`) sums each layer's change. Where every input is a
LoRA, the sum is itself a LoRA: the down matrices stacked and the up matrices
side by side, each scaled by its alpha / rank and strength. That's the exact
merge. To cut it to a rank, it takes a QR factor of each side and the SVD of
the small matrix between them, so the whole change is never built. A LoKr's
change, or a whole change to a weight, is built whole and cut to the rank, as
extracting does. A layer whose LoRA wouldn't be smaller than its whole change is
saved whole (`.diff`). The inputs are read a layer at a time, so they're never
in memory whole.

The merged file uses the same keys as a trained LoRA
(`diffusion_model.<module>.lora_A/lora_B/alpha`, plus `.diff`/`.diff_b`), with
alpha equal to the rank, so it loads at strength 1. Its metadata lists the
LoRAs it was merged from at their strengths (`dainamo.merged_from`) and their
trigger words.

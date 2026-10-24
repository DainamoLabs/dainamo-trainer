# Likeness

Likeness is how much an image looks like a dataset, as a percentage. Dainamo
measures it on every training sample, on request in the Sandbox, and uses the
same measurement to find a dataset's duplicates.

## How it's measured

Each image goes through DINOv2 (`facebook/dinov2-small`, 88 MB, downloaded
once; the model the concept-bleed check uses too) and comes out as a feature
vector. DINOv2 is self-supervised, so its features describe what's in the
picture and how it's drawn, not what a caption would say: two photos of the same
dog score high, a painting and a photo of a dog less so. A clip stands in as its
first frame.

A dataset is the vectors of its images (a random 2000 of a bigger one), their
mean (the centre) and how alike the images are to each other: the average cosine
of each to the centre, which is what a perfect sample would score. An image's
**similarity** is its cosine to the centre, and its likeness puts that on a
scale:

- **100%** is as alike to the dataset as its own images are to each other.
- **0%** is the **floor**: the base model's render of the same prompt and seed
  when there is one, else an unrelated image (which scores about 0 against a
  dataset's centre).

With the base model's render as the floor, the number says how much of the way
from the plain model to the dataset the LoRA took the image. That's the number
to watch over training: the prompt alone already puts a render some way toward
a dataset of the thing it names, and the floor takes that out.

Beside it, the **closest training image** is the one whose cosine with the image
is highest. At 0.97 or more the image is flagged a **near copy** of it: the LoRA
is reproducing a training image rather than the concept, the usual sign of
overfitting. A sample can score a high likeness without that flag (it looks
like the dataset in general) or carry the flag at a middling likeness (it copies
one image of a varied dataset).

Measured on renders of one prompt at six seeds, and on their copies:

| Images | Cosine |
| --- | --- |
| a copy, re-encoded as JPEG, resized, or brightened | 0.99–1.00 |
| a copy cropped by 10% | 0.97–0.98 |
| a copy cropped by 25%, or in greyscale | 0.94–0.96 |
| a mirrored copy | 0.91–0.94 |
| two renders of one prompt at different seeds | 0.51–0.98 |
| an image against an unrelated dataset's centre | −0.03–0.05 |
| a dataset's own images against its centre | 0.74–0.97 |

Renders of one prompt from a distilled model are very alike (the cat set scored
0.89–0.98 between seeds), so with a dataset of them the near-copy flag can go
off for a legitimate sample, and a mirrored or heavily cropped copy stays under
it. Photos of one subject vary more. The flag is a reason to look at the two
images side by side, not proof.

## Training samples

**Likeness score**, under Sampling in New job, is on by default for image and
video models (not sliders, which have no dataset to look like, nor music). Each
sample card gets its likeness as it's made, shown as a badge on the sample and
in the enlarged view with its closest training image; the **Loss** tab gets a
likeness curve under the loss curve, with the closest any sample came to a
training image as a dashed line, and the band at the top where that counts as a
copy. The log notes each step's likeness.

The floor of a card is its step-0 sample (the weights are the base model's
then). With **Sample before training** off, a LoRA job renders a card with the
LoRA at strength 0 the first time the card is scored; a finetune renders its
cards with the base model before training, since afterwards it can't. Cards
added later to a running job (an edit, or Sample now) get their floor the same
way. The training images' vectors and the floors are kept in the job's
`state/likeness.pt`, so a resumed or continued job scores against the same
floor; a finetune resumed after its state was deleted scores its cards against
an unrelated image instead (the log says so).

What it costs: one DINOv2 pass per sample, the dataset embedded once at the
start (a few seconds for hundreds of images), and without Sample before
training one extra render per card.

## Sandbox

**Compare**, on an image or clip output, picks a dataset and shows the output's
likeness to it, its similarity, and the five training images it's closest to.
With **Compare without LoRAs** on, the render without LoRAs of the same seed is
the floor, so the number is the LoRA's doing; otherwise 0% is an unrelated
image, and the dialog says so. The datasets a job trained on come first in the
list when the comparison started from one of its samples (**Compare with a
dataset**, in a sample's enlarged view).

A dataset's vectors are kept in `data/similarity/<dataset id>.pt` and brought up
to date when its files change, so the first comparison with a dataset embeds it
(a minute for a thousand images on a GPU) and later ones measure only the one
image. The measuring runs in its own process on a GPU no job or Sandbox is
using, else on the CPU (DINOv2-small is small enough).

## Find duplicates

**Find duplicates**, in a dataset's **Tools** menu, lists pairs of images whose
features are at least as alike as the threshold: 0.98 (the default) catches
copies and re-encodes, 0.95 also small crops and brightness changes, 0.90 also
mirrored copies and close variants, where two photos of one subject from the
same angle, or two renders of one prompt, show up too. Each side has a delete
button (with its caption; it asks first). Clips count by their first frame.

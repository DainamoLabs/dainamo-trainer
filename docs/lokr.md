# LoKr

**LoKr**, under Training method in New job, trains a LoKr (LyCORIS's
Kronecker-product adapter) instead of a LoRA. A LoKr can make a full-rank change with fewer
weights than a LoRA.

- **LoKr factor**: the lower it is, the more weights the LoKr trains. Auto uses
  the fewest.
- Rank and alpha don't apply. The summary shows the file size to expect.
- Switching to LoKr doubles the learning rate (2e-4 for the usual 1e-4), and
  switching back halves it, unless you've changed it: a LoKr learns more slowly
  at the same rate.

A LoKr is saved with LyCORIS's `lokr_w1` / `lokr_w2` keys and loads in
ComfyUI's LoRA loader like a LoRA. The Sandbox applies LoKrs too, including
other trainers' (not DoRA, LoHa, or LoKrs for convolutions). Its Sandbox renders
can differ slightly more from the training samples than a LoRA's, from bf16
rounding.

**In testing** (a halftone style, Krea 2 Turbo, 512 px, 800 steps), a
factor-16 LoKr took the same time and memory as a rank-32 LoRA, with a file of
95 MB against 215 MB. At 2e-4 it learned the style about 200 steps later than
the LoRA at 1e-4, but kept a portrait's face that the LoRA washed out. At 4e-4
it behaved like the LoRA.

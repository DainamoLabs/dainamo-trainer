# Memory

## Layer offloading

The **Layer offloading** slider (0–100%), in New job and the Sandbox, keeps
that share of the diffusion model's blocks in pinned system RAM and copies each
to the GPU just before it runs. The output is the same either way. Because the
copies overlap with computation, it often costs no speed: Krea 2 at 1024 px with
50% offloaded trained as fast as without on an RTX 3090, peaking at 12.5 GB
instead of 17.4 GB. Steps slow down when copying a block takes longer than
computing it (small images, fast GPUs).

Offloading saves little in training without gradient checkpointing.

## VRAM projection

Dainamo projects the peak VRAM of each phase of the job (encoding captions and
images, training steps, sampling) and compares the highest with the selected
GPU (with "Auto", the smallest): in New job's **Memory estimate**, with each
phase under **Memory breakdown**, and next to the Sandbox's slider. **Fit to GPU** picks
the lowest offload that fits, with about 0.75 GB to spare.

The projection reads tensor sizes from the model files' headers and adds each
family's measured activation memory at the job's biggest step. On real runs it
came within about 1 GB of nvidia-smi's peak, usually above. It assumes captions
of about 256 tokens and no other programs on the GPU.

Dainamo also projects the most system RAM the job will hold, and warns when
that's more than is free.

## Low VRAM

With every block offloaded, the text encoder sets the least VRAM most jobs need,
since it's loaded whole to encode the captions. The **Low VRAM** switch (off by
default, under the offload slider) streams it from system RAM a layer at a
time instead. The encodings are identical, and since captions are encoded once
and cached, only that step slows down: LTX 2.5's Gemma 4 needs 6.2 GB instead
of 19 GB, at 2.6 s a caption.

It also:

- samples in two steps, so the VAE and the diffusion model aren't on the GPU at
  the same time when there isn't room, at a cost of a few seconds per round of
  samples;
- keeps training's saved activations in system RAM, with only about three on the
  GPU at once. The loss and gradients are exactly the same, and steps take about
  1% longer.

The Sandbox has the same switch. YuE2 has no switch ([Music](music.md)).

### The least VRAM a job needs

Projected with every block offloaded, at 512 px for images and 384 px, 39-frame
clips for video:

| Model | Low VRAM off | Low VRAM on | Set by | System RAM |
| --- | --- | --- | --- | --- |
| Krea 2 | 5.6 GB | 4.6 GB | training steps | 17 GB |
| Qwen-Image 2.1 | 11.4 GB | 3.2 GB | encoding captions | 14 GB |
| MOSS-Image | 10.7 GB | 3.4 GB | encoding captions | 13 GB |
| LTX 2.5 | 19.9 GB | 7.3 GB | encoding captions | 24 GB |
| MiniMax H3 | 19.5 GB | 7.1 GB | encoding clips | 24 GB |

### The longest clip that fits

By the projection, with every block offloaded and Low VRAM on. A job takes clips
of up to 10 s (9.4 s for MiniMax H3).

| GPU | MiniMax H3, 384 px | MiniMax H3, 512 px | LTX 2.5, 384 px | LTX 2.5, 512 px |
| --- | --- | --- | --- | --- |
| 8 GB | 2.3 s | 1.6 s | — | — |
| 10 GB | 6.6 s | 3.8 s | 10 s | 6.4 s |
| 12 GB | 9.4 s | 5.2 s | 10 s | 10 s |
| 16 GB | 9.4 s | 8.7 s | 10 s | 10 s |

On a stand-in 8 GB card (PyTorch limited to 7.2 GB), LTX 2.5 trained on
49-frame clips and MiniMax H3 on 39- and 73-frame clips at 384 px, each peaking
at 6.6–7.5 GB and using 23–25 GB of system RAM.

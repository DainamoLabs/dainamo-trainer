# Upscaling

Dainamo makes images and videos bigger with the upscalers in your models
folder's `upscale_models` folder, where ComfyUI keeps them. You can upscale a
Sandbox output or any file in the [Sandbox](sandbox.md), and a dataset's images
that are smaller than the training resolution with
[Upscale small images](datasets.md#upscale-small-images).

## Upscalers

There are two kinds:

- **Pixel upscalers**: Real-ESRGAN, 4x-UltraSharp and the other single-image
  models [spandrel](https://github.com/chaiNNer-org/spandrel) reads (the ESRGAN
  family, DAT, HAT, SwinIR and more), as `.safetensors` or `.pth`. Each one
  makes a fixed size, such as 4× for RealESRGAN_x4plus. Dainamo runs it at that
  scale, in tiles, then fits the result to the factor you chose, so a 4× model
  asked for 2× is run at 4× and scaled down. A video is upscaled frame by
  frame, so fine detail can flicker from one frame to the next. They're fast
  and need little VRAM.
- **SeedVR2** (ByteDance's one-step diffusion restorer, the 3B model): it makes
  any size directly and restores detail as it goes. It upscales a clip's frames
  together, in passes of as many as fit (41 at 1080p), so they stay consistent
  with each other. Each pass shares its last 8 frames with the next, and over
  those frames the result fades from one pass to the other, so there's no jump
  where a pass ends. It needs its VAE, `seedvr2_ema_vae_fp16.safetensors`, in
  the same folder (or ComfyUI's `vae` folder; a SeedVR2 file in
  `diffusion_models` is found too). It's much slower than a pixel upscaler and
  needs more VRAM: about 12 GB for a 1080p image and 15 to 18 GB for a 1080p
  clip, on top of nothing else, so the Sandbox parks the loaded diffusion model
  in system RAM while it runs.

The **Upscalers** card in [Model files](profiles-and-model-files.md#model-files)
lists the files in `upscale_models` and what each one is, and downloads these:

| File | From | Size | License |
| --- | --- | --- | --- |
| `RealESRGAN_x4plus.safetensors` | [Comfy-Org/Real-ESRGAN](https://huggingface.co/Comfy-Org/Real-ESRGAN) | 67 MB | BSD-3-Clause |
| `seedvr2_3b_fp16.safetensors` | [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) | 6.8 GB | Apache 2.0 |
| `seedvr2_ema_vae_fp16.safetensors` | [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) | 501 MB | Apache 2.0 |

These are ComfyUI's single-file releases. The SeedVR2 file has the text
embedding it was released with built in, so it needs no text encoder. Like
every download, they come from a pinned commit and are checked against their
size and SHA-256. Other upscalers work too: put the file in `upscale_models`
and it's listed. A file Dainamo can't read is listed with the reason.

## In the Sandbox

- **Upscale**, on any image or video output, opens a dialog with the upscaler
  (a pixel upscaler shows its scale, SeedVR2 shows "video, any size") and the
  factor: 2×, 3× or 4×. It shows the size the result will be.
- **Upscale a file**, below History, takes an image or a video on this machine
  (**Browse**), or one sent from your browser (**Upload**). Uploads are kept
  until the server restarts.

An upscale is a run of its own. It waits for the generations queued before it,
runs in the Sandbox's worker beside the model loaded there, and shows in
History as "2× upscale", with the file it came from beside the result. You can
upscale a result again. The upscaler stays loaded for the next upscale until
another one is picked or the GPU is released. A result's longest side can be
at most 8192 pixels (4096 for a video).

A video keeps its frame rate and sound. It's written as it's upscaled, so a
long clip is never held in memory whole. A pixel upscaler takes it a frame at
a time. SeedVR2 takes the whole clip when its frames fit in 8 GB of system RAM
(or a quarter of what's free, if that's less), about 250 frames at 1080p. A
longer clip goes in pieces that share 8 frames and fade into each other the
same way its passes do. If an upscale fails or is cancelled part way, the
unfinished video is removed.

With SeedVR2, the dialog also has **SeedVR2 settings**, folded away:

- **Frames per pass**, for a video: the most frames a pass takes, rounded down
  to 4n + 1 (1 to 1001). Blank takes as many as fit at the size: 41 at 1080p.
  Fewer frames need less VRAM, and a clip has more joins between passes.
- **VAE tile**: **Auto**, or 512, 768 or 1024 px, the size of the tiles
  SeedVR2's VAE works in. Auto tiles only when a frame wouldn't fit in the VRAM
  that's free. A smaller tile needs less VRAM and is slower.
- **Seed** (666 by default): the noise SeedVR2 starts from. The same seed gives
  the same result.

A value out of range is brought into it, and the Sandbox's status line says
what was used.

The Sandbox's [Hi-res fix](sandbox.md) uses the same upscalers inside a
generation: the render is upscaled, then rendered again at the bigger size, so
the model redraws the detail the upscaler made.

## Upscale small images

Training never makes an image bigger, so an image with fewer pixels than the
resolution trains at a smaller size, and the
[health check](datasets.md#health-check)
warns about it. **Upscale small images**, in a dataset's **Tools** menu,
upscales those images:

- It takes the images with fewer pixels than the resolution squared (1024 by
  default, or what you used last) and upscales each to about that many pixels,
  each side by at most the **Largest factor** (4× by default). An image that
  would need more stays smaller. The dialog says how many images it will
  upscale before you start.
- Each result is saved as a PNG with the image's name, and the original is
  moved into an `_originals` folder beside it. Training and the health check
  skip `_originals`, and the health check notes how many images were upscaled.
  Captions (`photo.txt`) stay as they are, so each still belongs to its image.
- An image whose PNG name is taken by another file (`photo.jpg` beside a
  different `photo.png`) is skipped.
- A linked folder is changed in place, the same as an uploaded dataset.
- Clips aren't upscaled here. Use **Upscale a file** in the Sandbox for a clip.

It runs in the Sandbox's worker, after the generations queued before it, and
one dataset at a time. The dataset's page shows its progress. **Cancel** stops
it: the images already done stay upscaled, and the one in progress is left as
it was.

To undo one image, delete its PNG and move the original back from
`_originals`.

## GPUs and VRAM

The Sandbox's [GPU rules](sandbox.md#gpus) apply. With two or more GPUs,
upscaling runs on the Sandbox's GPU (the last free one if the Sandbox isn't
running). With one GPU, it runs only while no training job is running or
queued.

The upscaler loads beside the model the Sandbox holds. If they don't both fit,
the Sandbox moves its diffusion model to system RAM while the upscaler runs.
SeedVR2's weights alone are 6.8 GB, and its working memory grows with the size
and the frames. If it runs out of memory, try a smaller factor, a smaller
SeedVR2 **VAE tile** or fewer **Frames per pass**, or press **Release GPU** in
the Sandbox first so nothing else is loaded beside it.

## Licenses

[spandrel](https://github.com/chaiNNer-org/spandrel), which reads pixel
upscalers, is MIT. Real-ESRGAN's weights are BSD-3-Clause. SeedVR2's code and
weights are Apache 2.0 (see `THIRD_PARTY_NOTICES.md`). Other upscalers have
licenses of their own: check where a file came from before you share what it
makes commercially.

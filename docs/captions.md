# Captions

## Auto-Caption

**Auto-Caption**, on a dataset's page, captions each file with a vision LLM (see
[Caption provider](#caption-provider)). **Add to each caption** writes words at
the start of each caption it makes, unless they're there already; it starts as
the dataset's trigger word, and changing it there doesn't change the dataset's.

- **Caption style** is the prompt sent with each file: a detailed paragraph,
  one sentence, tags, a video style, MOSS-Image's own style (quotes visible
  text exactly and describes its layout, as MOSS-Image was trained), FLUX 3's
  (see [The FLUX 3 style](#the-flux-3-style)), or your own. Styles are edited
  in Settings; under **Advanced**, a style can also have a system prompt and
  its own temperature and max tokens (blank ones are the caption provider's).
- **Clips**: the captioner sees several frames (6 by default) with their
  times, so it can describe motion. By default it describes only the first 39
  frames, the part a job with the default clip length trains on; set the frame
  count to your job's clip length, or describe whole clips. Models that take
  one image get a numbered contact sheet instead.
- **Transcribe speech** runs Whisper (large-v3-turbo is recommended) over each
  clip with sound and puts what's said into the caption. The captioner itself
  can't hear, and Whisper writes down speech only.

Each dataset remembers the settings it was last captioned with.

A caption cut off at **Max tokens** (600 by default, in Settings, or the caption
style's own) is retried with four times the room. A caption that comes back
empty, still cut off, or cut short by a full context is never saved: the file is
listed with the reason, so you can caption just the missing ones (leave
**Overwrite existing captions** off).

Songs are captioned by hand; see [Music](music.md#datasets).

### The FLUX 3 style

FLUX 3 Image Klein was trained on captions written by its prompt upsampler, a
Qwen3.5-9B that Black Forest Labs tuned to turn a short request into the long
description the model works best with. Its captions follow fixed rules: the
medium and style first ("Full-color outdoor photograph, ..."), then the scene,
lighting and camera, then each subject with its colors, materials and position
in the frame, people by what can be seen of them, text quoted exactly, and
about a sixth on the background. A FLUX 3 LoRA trained on captions like these
stays close to what the model knows, and prompts written the same way work
with it.

**FLUX 3 (prompt upsampler style)** asks the caption provider for that
caption. It's the upsampler's own text-to-image system prompt (Apache 2.0,
bundled as `dainamo/assets/flux3_pup_prompts.json`) turned to describing an
image: the framing ("expand the user's short request") becomes "describe the
attached image", the rules that only apply to a request (a target canvas,
deciding what it left open, writing text it needs) are left out or say what the
image shows, and the caption format is kept word for word. The system prompt
goes first, then the image, then "Write the caption for this image.", as the
upsampler gets its reference images. It samples as the upsampler does, at
temperature 0, but stops at 1024 tokens instead of 3584: those leave room for
layout tables, and a caption runs 100 to 350 words. FLUX 3 reads the first 512
tokens of a caption, and the dataset health check flags longer ones.

Any caption provider that sees images can use it: the text encoders below (FLUX
3's own is a Qwen3-VL-4B like Krea 2's, so a FLUX 3 profile's encoder can
caption with nothing else to download), a llama.cpp vision model, or your own
server. The closest captions come from the upsampler itself: download it in
**Models › Model files**, under FLUX 3 Image Klein's **Prompt upsampler**
(optional, for captions; int8 convrot, 9.7 GB, or bf16, 18.8 GB, gated like FLUX
3's other files), and choose it as the caption provider's [text
encoder](#text-encoder). Settings names it "FLUX 3 prompt upsampler". Inside,
the file is laid out like MOSS-Image's encoder, so Dainamo tells it by its name:
keep "upsampler" or "pup" in the file name. No profile uses it, and Model files
doesn't offer it as MOSS-Image's text encoder.

## Caption provider

Auto-Caption and the Sandbox's Enhance prompt use the vision LLM chosen in
**Settings → Caption provider**.

### llama.cpp

Dainamo runs llama.cpp's `llama-server`. Choose the program (blank uses the one
on the PATH), a vision model (`.gguf`, e.g. Qwen2.5-VL or Gemma 3) and its
projector (`mmproj-….gguf`). It starts when needed and stops after 10 idle
minutes or on **Stop**. It listens on 127.0.0.1 from port 8090 up.

- **Runs on**: "A free GPU, else the CPU" takes a GPU no job or Sandbox is
  using, or pick one. Training and the Sandbox come first: they make it stop
  once the current Auto-Caption is done.
- Use a CUDA build of llama.cpp (`…-bin-ubuntu-cuda-…` or `…-bin-win-cuda-…`).
  A CPU-only build runs on the CPU whatever you choose, and Settings says so.
- **Context size** 0 sizes it for the requests (11,264 tokens with the default
  Max tokens), capped at the model's own context.
- **Extra arguments** go to llama-server as written, e.g. `-ngl 20` to keep
  part of the model in system RAM. Dainamo sets the model, projector, host and
  port.

The command line, readiness check and log hints follow
[LLMHub](https://github.com/William-Kruta/LLMHub).

### Text encoder

Some image models' text encoders are whole vision-language models that can
write captions, so no extra download is needed. Choose the encoder's
`.safetensors` file; Settings lists the ones your profiles use and says which
can write. It needs a GPU, and makes way for training like llama-server.

| Encoder file | Model | VRAM | Speed on an RTX 3090 |
|---|---|---|---|
| Krea 2's | Qwen3-VL-4B | 8.3 GB | ~55 tokens/s |
| Qwen-Image 2.1's | Qwen3-VL-8B | 16.3 GB | ~37 tokens/s |
| MOSS-Image's | Qwen3.5-9B | 17.2 GB | ~34 tokens/s |

FLUX 3's prompt upsampler is a Qwen3.5-9B like MOSS-Image's and runs the same
way (see [The FLUX 3 style](#the-flux-3-style)).

An int8 or fp8 file is expanded to bf16 while the GPU keeps 4 GB free, which is
about 3x faster than running it quantized. Installing `flash-linear-attention`
speeds up Qwen3.5.

MiniMax H3's and LTX 2.5's encoders can't write: H3's is cut down to 50 of
Qwen3-VL-32B's 64 layers with no output layer, and LTX's Gemma 4 was retrained
as an encoder and no longer predicts text. Every encoder is tested for this
after it loads, so another retrained one is refused rather than writing
nonsense.

### Your own server

Any OpenAI-compatible endpoint (llama-server with `--mmproj`, Ollama, LM Studio,
vLLM…), with an optional model name and API key. **Test connection** lists its
models.

### When it fails

Settings shows each server's status and log (all of it is in
`data/local-llm.log`). A failed start shows what the server reported, with a
hint for the usual causes: out of memory, an unknown argument, a llama.cpp too
old for the model, or the wrong projector.

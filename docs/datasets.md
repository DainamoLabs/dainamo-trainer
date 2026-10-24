# Datasets

![A dataset with its captions and health check](screenshots/dataset.png)

A dataset is a folder of images, videos or songs. Link an existing folder or
upload files. Each file's caption is a `.txt` file next to it with the same
name. To write captions with a vision LLM, see [Captions](captions.md).

A dataset's page shows its files in one of two views, and remembers the one
you chose:

- **Grid**: big cards, 24 to a page. A song plays on its card. Click a
  caption to edit it in place; Ctrl/Cmd+Enter or clicking away saves, Esc
  cancels. A card's **⋯** enlarges the file, copies its path or deletes it.
- **List**: smaller cards, every file on one page. Click a caption to edit it.

**Search** matches file names and captions, and **Missing captions** shows the
files without one. On an uploaded dataset, **Select files** (or a card's
checkbox) picks files to delete together. A linked folder's files are only
deleted in the folder itself. The **⋯** beside Auto-Caption holds Split into
clips, Upscale small images, Export and Remove dataset.

Click a file to enlarge it. The **Left/Right** arrow keys, or the on-screen
buttons, move between files in the grid's order. You can edit the caption
there: **Save**, or Ctrl/Cmd+Enter. Unsaved edits are saved when you move to
another file or close.

## The Datasets page

Each dataset's card shows what it holds (images, videos, audio or a mix),
whether its files were uploaded or it's a linked folder, how many files have a
caption, and its trigger word and folder. **Search** matches names, trigger
words and folders; the media menu shows only datasets with images, videos or
audio.

## Select, export and delete

The **⋯** on a card exports or deletes that dataset. **Select** on the Datasets
page puts a checkbox on each dataset (the ones the search shows). Then
**Export** downloads the selected datasets as one .zip, with a folder for each.
Each folder holds the dataset's files in their subfolders, each file's caption,
the reference images and masks, and a split's clip list. Nothing is compressed
again, so the download starts at once. **Put the trigger word in each caption**
writes the captions as training reads them: Dainamo adds the trigger word when
it trains, but other trainers read captions as they are. **Delete** removes the
selected datasets. Linked folders are left alone; uploaded files are deleted
only if you tick the box. Esc or **Done** leaves Select.

## Edit datasets: sources and results

An edit LoRA (Qwen-Image 2.1) learns from pairs: an image before the edit, the
image after, and an instruction. Keep them as two datasets:

- **the results**: the images after the edit, each captioned with the edit
  instruction ("Make it night: …"). This is the dataset the job trains;
- **the sources**: the images before the edit, with the same file names
  (`photo.jpg` pairs with `photo.png`; the type doesn't matter). They need no
  captions.

In Create Job, choose the results as the dataset and the sources as its
**Source dataset**. A second or third source dataset adds more images to each
edit. The two folders can be nested either way: sources in a subfolder of the
results are only sources, and results in a subfolder of the sources (`photos/`
and `photos/edited/`) train as results and are never their own source. With
no source dataset, the model trains text-to-image, so the same model trains
plain LoRAs from one dataset. A result without a source of its name trains
text-to-image too, its caption a prompt.

The card says how many files pair (**4 of 6 files pair with a source image**),
its health check lists the files without a source and the sources that pair
with none (usually a name that differs), and **Show pairs** shows each source
beside its result and instruction. The dataset page's health check does the
same: choose the model and a source dataset beside the resolution.

MiniMax H3's ref2va trains with reference images the same way, from a
**Reference dataset** (`clip.mp4` pairs with `clip.png`).

Datasets from before source datasets keep their sources in a `_refs` folder
inside the dataset, named after the file they belong to: `_refs/photo.png`,
`_refs/photo_1.png`, and so on. Those are used when no source dataset is
chosen, and checked and shown the same way.

## Masks

**Masked loss** (in Create Job) trains on part of each image, such as the
subject and not its background. Put a greyscale mask for an image in a `_masks`
folder beside it, with the image's name: `_masks/photo.png` is the mask of
`photo.jpg`. White is where the loss counts, black where it doesn't, and grey
counts partly. A PNG is looked for first, but any image type works.

- A mask is stretched to its image's size, then cropped and flipped with it,
  so a smaller mask still lines up if it has the image's shape. The health
  check warns about a mask whose shape differs from its image's by over 2%.
- The loss is averaged over the mask, so a small mask doesn't make its image
  count for less. Outside it, the loss counts at **Outside the mask** (0.1 by
  default).
- An image without a mask trains as usual, and the checkpoint picker scores
  with the masks too.
- Only image models use masks (Krea 2, MOSS-Image and Qwen-Image 2.1). Video
  and music models train on whole frames and songs.

The `_masks` folder is never trained on itself. The dataset's page shows how
many images have a mask.

## Health check

**Dataset health check**, folded under the dataset's counts until you open it
(**Configure check**), looks for problems for a chosen model and resolution.
Folded, it still shows how the last check went:

- unreadable files;
- images and clips smaller than the resolution. Training never upscales a
  file, so these train smaller; a file is flagged when it would get under 75%
  of the pixels of a large enough file of its shape. [Upscale small
  images](#upscale-small-images) can make the images bigger;
- near-duplicate images;
- missing or empty captions, and two files sharing one caption (`a.png` and
  `a.jpg`);
- captions longer than the model reads (Krea 2 and MiniMax H3: 512 tokens; LTX
  2.5: 1024);
- clips shorter or longer than the clip length (a longer one trains only its
  start);
- notes: captions without the trigger word, caption files with no image, and
  how many images were upscaled (their originals are in `_originals`);
- for a model that trains with source or reference images: files without one,
  and sources that pair with no file;
- masks of another shape than their image (stretched to fit it), and notes:
  how many images have a mask, and masks with no image of their name.

Each dataset in Create Job runs the same check for that job. For songs, see
[Music](music.md#datasets).

## Upscale small images

**Upscale small images**, in the **⋯** menu of a dataset with images, upscales
the images with fewer pixels than the resolution you train at, so they train
at full size. Each result is saved as a PNG with the image's name, the original
is moved into an `_originals` folder beside it (which training skips), and the
captions stay as they are. A linked folder is changed in place. Pick the
upscaler (Real-ESRGAN, SeedVR2 or another file in the models folder's
`upscale_models`), the resolution and the largest factor; the dialog says how
many images it will upscale. It runs in the Sandbox's worker, and the page
shows its progress. See [Upscaling](upscaling.md#upscale-small-images).

## Edit captions

**Edit captions** changes every caption at once: find and replace (plain or a
regular expression), add words at the start or end (skipped where they're
already there), or remove whole words. It previews the changes first, and the
last edit can be undone.

## Split into clips (video models)

A job trains only the start of each clip, up to the clip length (39 frames,
1.6 s, by default for MiniMax H3; with Auto, up to 10 s), so most of a long
video is never used. **Split into clips**, in the **⋯** menu of a dataset with
videos, makes a new dataset with each video cut into clips a job trains on
whole. The original is left as it is.

- **Clip length**: pick a video profile, GPU, resolution, offload and Low VRAM.
  The dialog lists each clip length the model takes (MiniMax H3: 17n + 5
  frames; LTX 2.5: 8n + 1) with its projected VRAM, and picks the longest that
  leaves 0.75 GB free. MiniMax H3 on a 24 GB GPU at the default 40% offload
  fits 141 frames (5.9 s) at 384 px and 73 frames (3 s) at 512 px.
- **Keep each clip inside one shot** (on by default): clips stay a frame clear
  of each scene cut, since a clip that spans a cut teaches the model to cut. A
  shot too short for a clip gives one shorter clip if it's at least 1 s long,
  unless **Keep shots shorter than a clip** is off.
- **Clips per video** (8 by default; 0 for all) caps how many a long video
  gives, spread evenly over it.
- **Each video counts once**: each epoch trains one of a video's clips, picked
  at random, so a long video adds variety rather than weight. The checkpoint
  picker holds out a whole video. A clip added to the dataset later counts on
  its own.
- **Files**: clips are re-encoded at 24 fps (x264 CRF 17, AAC) with the shorter
  side at most 1024 px (or 512, 768, 1280, or as it is), never scaled up.
  Images, sound files and `_refs` are copied.
- **Captions**: each clip gets its video's caption, or none so you can
  Auto-Caption each clip; a long video's caption often doesn't fit every part.

Cancelling a split removes the partial copy.

### Clip editor

**Review first**, in the Split into clips dialog, plans the clips without
cutting them and opens the clip editor. **Edit clips**, on a dataset Split into
clips made, opens it later. Each video of the original is listed with its
clips; pick one to see it:

- **The timeline** shows the video's frames, its shot cuts (dashed lines), its
  clips and where the player is. Drag a clip to move it (a cut pulls its start
  or end a frame clear of it), or one of its ends to change its length (only to
  lengths the model takes). Double-click, or **Add a clip here**, for a new clip
  of the length beside it. **Plan again** replans the video as Split into clips
  does, at that length. Zoom in for long videos.
- **The player** plays the selected clip, looping, with its sound. A browser
  can't play every file (an `.avi`, say); the editor then shows still frames
  from the server.
- **A clip** can start anywhere a frame clear of the video's end, and be
  cropped: choose an aspect (or Free) and drag the box on the video, or its
  corners. **All clips** gives every clip of the video that crop. A clip that
  crosses a shot cut is outlined red, and one overlapping another is noted.
- **First and last frame** are taken from the video as the cut takes them (the
  same seek and resampling to 24 fps), so they're exactly the clip's ends; the
  browser's player is only close.
- Keys: Space plays, ←/→ move the clip a frame (Shift: a second), N adds a clip
  at the player, Delete removes the selected one, Esc deselects.

Changes save as you make them. **Cut** cuts only what changed: new clips, and
clips moved, resized or cropped (which keep their captions), and removes the
clips taken out, with their captions. The rest are left as they are. A clip
deleted from the dataset's page leaves the plan, so cutting again doesn't bring
it back. Videos added to the original since are found and planned when the
editor opens. A split made before the clip editor opens in it too: its clips
are read from the dataset, and its videos looked at again for their shots.

## Add from URL (experimental)

**Add from URL**, on a dataset's page, downloads a video, or only its sound,
into the dataset with [yt-dlp](https://github.com/yt-dlp/yt-dlp), which
supports YouTube and many other sites.

> **Experimental.** A site change can break yt-dlp until it's updated. If
> downloads from a site start failing, click **Update yt-dlp** in the window
> first.
>
> **Download only what you have the right to use.** Copyright and each site's
> terms are your responsibility.

- **Video** comes no taller than **Largest size** (1080p by default), in the
  chosen **Format**:

  | Format | |
  | --- | --- |
  | MP4 (default) | H.264 with AAC where the site has it; plays everywhere. Above 1080p, YouTube has only VP9 or AV1, which MP4 also takes |
  | WebM | VP9 with Opus |
  | MKV | the best the site has (on YouTube, AV1 with Opus); the browser may not preview it |
  | MOV | H.264 with AAC, for video editors; at most the site's best H.264 (1080p on YouTube) |

- **Sound only**: **As the site has it** (default, never converted), Opus, M4A
  (AAC), MP3, WAV, FLAC or OGG (Vorbis). WAV and FLAC lose nothing more, but
  can't restore what the site's compression removed.
- **From** and **To** (`1:30` or `90`) download just that part, cut exactly.
- **It's a playlist** downloads a playlist's first N items (up to 200);
  otherwise a link downloads one item, even a playlist's link.
- **Files** are named after the title and the site's id, such as
  `Big Buck Bunny [aqz-KE-bpKQ].mp4`. Captions are left to you or Auto-Caption.
- YouTube needs a JavaScript runtime: yt-dlp finds [Deno](https://deno.com),
  and Dainamo passes it Node.js or Bun when they're installed. Without one, some
  formats may be missing.

Cancelling keeps the files already added.

**On a network**, anyone signed in from another device can use Add from URL to
make this machine download from any address, including ones on your local
network. The server warns about this when it starts on the network.

# RefMods (MiniMax H3)

A RefMod is a set of references saved as a file, so MiniMax H3's
reference-to-video model (ref2va) can use them without encoding them again:

- **Pictures**: encodes of images and clips of one subject or style, used as
  reference images are.
- **Sound**: the encode of a voice, singing or music, used as reference audio.

The file format is
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)'s:
a RefMod made here loads there (copy it into `models/refmods`), and one made
there loads in the Sandbox.

## Making one

Choose **RefMod** under Make in Create Job (MiniMax H3 profiles, fl2va or
ref2va), then what it's **Of**: *Pictures* or *Sound*. Nothing is trained; the
job loads only a VAE and takes a few minutes or less.

**Pictures.** Every image and clip in the chosen datasets is a reference, in
order. Captions, repeats and `_refs` aren't used.

- **Encoding**: *Compressed* (the default) shrinks each reference to a grid of
  **Grid** cells along its long edge (16), refined over **Refinement steps**
  (500) to match the full encode. *Full* keeps each encode whole, cropped to the
  first reference's shape: the most detail, and the most tokens.
- **Size**: the short edge references are scaled down to (1024; never
  enlarged).
- **Frames per clip**: a clip's first frames at 24 fps, snapped down to 17n + 5.
- **Merge into one reference** (compressed): one grid for every reference,
  instead of one each.
- **Token cap**: the most reference tokens it may hold (5120; 0 for no cap). A
  1024×1024 image in full is 1024 tokens; a 16-cell grid is 64. Over the cap,
  near-duplicate frames go first.
- **Concept** and **Description**: kept in the file, where the pack's nodes
  show them.
- **Save a copy to**: e.g. ComfyUI's `models/refmods`.

**Sound.** It takes the datasets' sound files and clips with sound, in order,
until there are **Length** seconds (10; up to 60), as one sound at 80 tokens a
second. The **Token cap** cuts the end. Its **Concept** is a voice, singing, a
music style, a sound effect, ambience or generic.

The job page previews the result. Editing a made RefMod job, or **Make again**,
makes it again with the changes.

## Using RefMods in the Sandbox

Reference to Video with a ref2va profile takes up to 4 RefMods, beside reference
images or instead of them. Each has a strength from 0 to 1; below 1, it guides
the video less. As with the pack's Apply node, the text encoder doesn't see
picture RefMods.

## Reference audio

For a video, Reference to Video also takes up to 3 sound files (or videos, for
their soundtrack) under **Reference audio**; sound RefMods count among the 3.
Each is cut to the video's length from its start. The model takes a sound
reference only beside a picture one (a reference image or a picture RefMod).

The ComfyUI pack reports that a voice doesn't carry over reliably and music
does better, so expect to try a few seeds.

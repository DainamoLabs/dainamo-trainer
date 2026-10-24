# Bundled Hugging Face files

Tokenizers, chat templates and configs that ComfyUI single-file models don't
carry, copied from the families' Hugging Face repos at the commits pinned in
`dainamo/engine/hf_assets.py` (`REVISIONS`), so those families run without a
network from the first job. `hf_assets.load_assets` reads these first and falls
back to the Hugging Face cache or a download. No model weights are here.

| Folder | Repo and commit | Files | License |
| --- | --- | --- | --- |
| `Qwen--Qwen3-VL-4B-Instruct/` | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec` | tokenizer, generation and model configs | Apache 2.0 (`LICENSE-Apache-2.0.txt`) |
| `OpenMOSS-Team--MOSS-Image-SFT/` | `OpenMOSS-Team/MOSS-Image-SFT` @ `2c795443` | `tokenizer/` (Qwen3.5 tokenizer and chat template), `text_encoder/config.json` | Apache 2.0 (`LICENSE-Apache-2.0.txt`) |
| `Qwen--Qwen-Image-2.1/` | `Qwen/Qwen-Image-2.1` @ `d26bb612` | `processor/` (Qwen3-VL processor), `text_encoder/config.json`, `vae/config.json` | Qwen Research License (`dainamo/engine/families/qwen_image_2/QWEN_RESEARCH_LICENSE.txt`), redistributed under its section 3 with the notice in `NOTICE`. Its `tokenizer.json`, `vocab.json` and `merges.txt` are byte for byte Qwen3-8B's (Apache 2.0) |

MiniMax H3's tokenizer, processor and config aren't bundled: the MiniMax H3
Community License excludes several territories, so the app downloads them from
`MiniMaxAI/MiniMax-H3` on first use, as before.

To update a folder, download the files at the new commit, copy them here with
the same layout, and change the commit in `REVISIONS`.

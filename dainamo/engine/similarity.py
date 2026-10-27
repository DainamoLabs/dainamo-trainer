"""Likeness: how much an image looks like a dataset, measured in DINOv2 feature space.

Every image (a clip stands in as its first frame) becomes a unit vector of DINOv2-small features (the same
``facebook/dinov2-small`` the concept-bleed check uses: self-supervised, so it compares what's in the picture and how
it's drawn rather than what a caption would say). A dataset is a ``Concept``: its images' vectors, their mean (the
centroid) and ``self_sim``, how alike the images are to each other (the mean cosine of each to the centroid).

An image is scored against its **neighbourhood**: the ``neighbours(n)`` training images closest to it, their mean
(the local centre) and how alike they are to it (``target``, the mean cosine of each to the local centre). Its cosine
to the local centre (``sim``) is given as a **likeness**:

    likeness = (sim - floor) / (target - floor)   as a percentage, 0 to 100

100% means the image is as close to that part of the dataset as its own images are, 0% no closer than the floor. The
floor is the base model's render of the same prompt and seed when there is one (a training sample's step-0 render, a
Sandbox run's render without LoRAs), scored against the same neighbourhood: then likeness says how much of the gap
between the plain model and the dataset the LoRA closed, which is the number to watch over training. Without a base
render, or when the base render is already within ``MIN_GAP`` of the target (the prompt alone gets there, and a scale
that narrow reads 0 or 100 on noise), the floor is ``FLOOR``, what an unrelated image scores (about 0, measured: see
``docs/likeness.md``).

The neighbourhood is what makes a varied dataset work: one of Creepers, golems and villages has a centroid that no
single picture is near (its images score 0.47 against it), and a sample of a Creeper rightly moves away from it as it
moves towards the Creepers.

The **nearest** training image (the highest cosine to any single one) tells memorisation from likeness: above
``NEAR_COPY`` the image is a near copy of that training image, however the likeness reads.

``LikenessMeter`` scores a job's training samples as they're made (``trainer.make_samples``), and the module runs as
a program for the server (``python -m dainamo.engine.similarity request.json``): a dataset's concept, cached beside
the data as a .pt and brought up to date when its files change, scored against images, or searched for duplicates.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from . import dataset as ds

DINO_REPO = "facebook/dinov2-small"
# Measured on renders of one prompt at different seeds and on their copies (docs/likeness.md): a copy, re-encoded,
# cropped a little or brightened, scores 0.97 or more against the original, a mirrored one about 0.92; two renders of
# one prompt from a distilled model 0.5 to 0.98; unrelated images score about 0 against a dataset's centroid.
NEAR_COPY = 0.97  # cosine to one training image above which the image is flagged as a near copy of it
FLOOR = 0.0  # what an unrelated image scores against a dataset's centroid: the floor without a base render
MIN_GAP = 0.05  # the least a base render may score below the target for it to be the floor (else FLOOR is)
DUPLICATE = 0.98  # cosine above which two dataset images count as duplicates (copies and re-encodes)
MAX_IMAGES = 2000  # dataset images a concept is built from (a random sample of bigger datasets)
BATCH = 16
FRAME_SIZE = (448, 448)  # a clip's first frame is read at about this size


class Embedder:
    """DINOv2-small image features: unit vectors (N, 384) on the CPU. The model (about 88 MB, downloaded once) loads
    on the first call and stays until ``unload``."""

    def __init__(self, device, what: str = "DINOv2 (for likeness)"):
        self.device, self.what = device, what
        self.model = self.proc = None

    def _load(self):
        from transformers import AutoImageProcessor, AutoModel

        from .hf_assets import from_pretrained

        self.proc = from_pretrained(AutoImageProcessor, DINO_REPO, self.what)
        self.model = from_pretrained(AutoModel, DINO_REPO, self.what).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.zeros(0, 384)
        if self.model is None:
            self._load()
        out = []
        for i in range(0, len(images), BATCH):
            batch = [im.convert("RGB") for im in images[i:i + BATCH]]
            inputs = self.proc(images=batch, return_tensors="pt").to(self.device)
            out.append(F.normalize(self.model(**inputs).pooler_output.float(), dim=-1).cpu())
        return torch.cat(out)

    def unload(self):
        self.model = self.proc = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_image(path: str) -> Image.Image:
    """An image file as RGB (turned as its EXIF says), or a clip's first frame."""
    if os.path.splitext(path)[1].lower() in ds.VIDEO_EXTS:
        return ds.first_frame(path, FRAME_SIZE, 24)
    with Image.open(path) as im:
        return ds.to_rgb(ImageOps.exif_transpose(im))


def media_image(media) -> Image.Image | None:
    """The picture of what a sampler made (``save_media``'s input): the image, or a clip's first frame; None for
    sound."""
    if isinstance(media, Image.Image):
        return media
    if isinstance(media, dict) and "frames" in media:
        frames = media["frames"]
        frame = frames[0]
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        return Image.fromarray(frame)
    return None


def embed_files(embedder: Embedder, paths: list[str], progress=None, check=None) -> tuple[torch.Tensor, list[str]]:
    """``paths`` embedded, in batches, skipping files that can't be read: (vectors, the paths that could)."""
    vecs, kept = [], []
    for i in range(0, len(paths), BATCH):
        if check:
            check()
        images, ok = [], []
        for p in paths[i:i + BATCH]:
            try:
                images.append(load_image(p))
                ok.append(p)
            except (OSError, ValueError, RuntimeError, IndexError, subprocess.SubprocessError):
                continue
        if images:
            vecs.append(embedder(images))
            kept += ok
        if progress:
            progress(min(i + BATCH, len(paths)), len(paths))
    return (torch.cat(vecs) if vecs else torch.zeros(0, 384)), kept


def likeness(sim: float, target: float, floor: float) -> float:
    """``sim`` as a share of the way from ``floor`` to ``target``, in percent, 0 to 100."""
    gap = max(target - floor, 1e-3)
    return round(min(100.0, max(0.0, 100 * (sim - floor) / gap)), 1)


def neighbours(n: int) -> int:
    """How many training images an image is scored against: about the square root of the dataset's size, 3 to 10
    (all of a smaller one)."""
    return min(n, max(3, min(10, round(n ** 0.5))))


@dataclass
class Concept:
    """A dataset's images in feature space: ``emb`` (N, D) unit vectors, one per path, their normalised mean and
    how alike they are (``self_sim``, the mean cosine to the centroid; 1 for a dataset of one image)."""
    paths: list[str]
    emb: torch.Tensor
    centroid: torch.Tensor
    self_sim: float

    @classmethod
    def of(cls, paths: list[str], emb: torch.Tensor) -> "Concept":
        emb = emb.float()
        centroid = F.normalize(emb.mean(0), dim=-1) if len(paths) else torch.zeros(emb.shape[-1])
        self_sim = float((emb @ centroid).mean()) if len(paths) else 0.0
        return cls(list(paths), emb, centroid, self_sim)

    @classmethod
    def build(cls, embedder: Embedder, paths: list[str], limit: int = MAX_IMAGES, seed: int = 0, progress=None,
              check=None) -> "Concept":
        paths = sorted(set(paths))
        if len(paths) > limit:
            paths = sorted(random.Random(seed).sample(paths, limit))
        emb, kept = embed_files(embedder, paths, progress, check)
        return cls.of(kept, emb)

    def score(self, e: torch.Tensor, base: torch.Tensor | None = None, top: int = 1) -> dict:
        """One image's vector against its neighbourhood of the concept: {sim (to the local centre), target (the
        neighbourhood's own, 100%), floor (the base render ``base``'s; None: ``FLOOR``, as there's no base render or
        it's within ``MIN_GAP`` of the target), base_sim (the base render's, used or not), likeness, nearest: {path,
        sim} (the closest training image), top: the ``top`` closest, near_copy}."""
        if not self.paths:
            return {"sim": None, "target": None, "floor": None, "base_sim": None, "likeness": None, "nearest": None,
                    "top": [], "near_copy": False}
        e = e.float()
        sims = self.emb @ e
        order = sims.argsort(descending=True)
        hood = self.emb[order[:neighbours(len(self.paths))]]
        centre = F.normalize(hood.mean(0), dim=-1)
        sim, target = float(e @ centre), float((hood @ centre).mean())
        base_sim = float(base.float() @ centre) if base is not None else None
        floor = base_sim if base_sim is not None and target - base_sim >= MIN_GAP else None
        best = [{"path": self.paths[k], "sim": round(float(sims[k]), 4)} for k in order[:max(1, top)].tolist()]
        return {"sim": round(sim, 4), "target": round(target, 4), "floor": None if floor is None else round(floor, 4),
                "base_sim": None if base_sim is None else round(base_sim, 4),
                "likeness": likeness(sim, target, FLOOR if floor is None else floor),
                "nearest": best[0], "top": best, "near_copy": best[0]["sim"] >= NEAR_COPY}

    def duplicates(self, threshold: float = DUPLICATE) -> list[dict]:
        """Pairs of images at least ``threshold`` alike, the most alike first: [{a, b, sim}] (paths)."""
        if len(self.paths) < 2:
            return []
        sims = self.emb @ self.emb.T
        idx = torch.triu_indices(len(self.paths), len(self.paths), offset=1)
        vals = sims[idx[0], idx[1]]
        hit = (vals >= threshold).nonzero().flatten()
        pairs = [{"a": self.paths[int(idx[0][i])], "b": self.paths[int(idx[1][i])], "sim": round(float(vals[i]), 4)}
                 for i in hit.tolist()]
        return sorted(pairs, key=lambda p: -p["sim"])

    def state(self) -> dict:
        return {"paths": self.paths, "emb": self.emb.half(), "repo": DINO_REPO}

    @classmethod
    def from_state(cls, st: dict) -> "Concept":
        return cls.of(st["paths"], st["emb"].float())


# ----------------------------------------------------------------------------------------------- the cache ---
def _stamp(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{path}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return f"{path}|missing"


def cached_concept(embedder: Embedder, paths: list[str], cache_file: str | None, progress=None, check=None,
                   limit: int = MAX_IMAGES) -> Concept:
    """The concept of ``paths``, from ``cache_file`` when it holds them as they are now; else built, embedding only
    the files that are new or changed (the others' vectors are kept), and saved there."""
    paths = sorted(set(paths))
    if len(paths) > limit:
        paths = sorted(random.Random(0).sample(paths, limit))
    stamps = {p: _stamp(p) for p in paths}
    old: dict[str, torch.Tensor] = {}
    if cache_file and os.path.exists(cache_file):
        try:
            st = torch.load(cache_file)
            if st.get("repo") == DINO_REPO:
                for p, s, e in zip(st["paths"], st["stamps"], st["emb"]):
                    if stamps.get(p) == s:
                        old[p] = e.float()
        except (OSError, RuntimeError, KeyError, ValueError, TypeError):
            old = {}
    todo = [p for p in paths if p not in old]
    emb, kept = embed_files(embedder, todo, progress, check) if todo else (torch.zeros(0, 384), [])
    vecs = dict(old, **dict(zip(kept, emb)))
    final = [p for p in paths if p in vecs]
    concept = Concept.of(final, torch.stack([vecs[p] for p in final]) if final else torch.zeros(0, 384))
    if cache_file and (todo or len(old) != len(final)):
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        torch.save({"repo": DINO_REPO, "paths": final, "stamps": [stamps[p] for p in final],
                    "emb": concept.emb.half()}, cache_file)
    return concept


# ------------------------------------------------------------------------------------------ while training ---
def spec_key(spec: dict) -> str:
    """What makes a sample card's base render: its prompt, seed, size, frames and conditioning images (not the LoRA
    strength: the base render is the same for every strength)."""
    parts = [str(spec.get(k, "")) for k in ("prompt", "seed", "width", "height", "frames", "seconds", "lyrics")]
    parts.append(",".join(spec.get("images") or []))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


class LikenessMeter:
    """Scores a job's samples against its training images as they're made.

    The floor of each card is the base model's render of it: the step-0 sample (the weights are the base model's
    then), else rendered at LoRA strength 0 the first time the card is scored (``render``: a spec -> what the
    sampler made). The concept and the base vectors are kept in ``state_file`` (the job's training state), so a
    resumed job scores against the same floor. A finetune can't render the base model once it has trained
    (``can_render_base`` False): its floors come from step 0 (``prime``, or the step-0 samples), and a card without one
    scores with ``FLOOR``."""

    def __init__(self, device, state_file: str, render, log, can_render_base: bool = True):
        self.state_file, self.render, self.log, self.can_render_base = state_file, render, log, can_render_base
        self.embed = Embedder(device)
        self.concept: Concept | None = None
        self.bases: dict[str, torch.Tensor] = {}

    def setup(self, train_paths: list[str]):
        paths = sorted(set(train_paths))
        if len(paths) > MAX_IMAGES:
            paths = sorted(random.Random(0).sample(paths, MAX_IMAGES))
        saved = None
        if os.path.exists(self.state_file):
            try:
                saved = torch.load(self.state_file)
            except (OSError, RuntimeError, ValueError):
                saved = None
        if saved and saved.get("repo") == DINO_REPO:
            self.bases = {k: v.float() for k, v in (saved.get("bases") or {}).items()}
            if saved.get("paths") == paths:
                self.concept = Concept.from_state(saved)
        if self.concept is None:
            self.log(f"Likeness: embedding {len(paths)} training image(s) with DINOv2")
            self.concept = Concept.build(self.embed, paths)
            self._save()
        k = neighbours(len(self.concept.paths))
        self.log(f"Likeness: each sample is scored against the {k} training image(s) closest to it; at 100% it's as "
                 "alike to them as they are to each other, at 0% no more than the base model's render")

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        torch.save({**self.concept.state(), "bases": {k: v.half() for k, v in self.bases.items()}}, self.state_file)

    def base_for(self, spec: dict, step: int, sample_vec: torch.Tensor | None) -> torch.Tensor | None:
        """The base render's vector for a card: known already, the step-0 sample's own (``sample_vec``), or rendered
        now at strength 0; None when it can't be made."""
        key = spec_key(spec)
        if key in self.bases:
            return self.bases[key]
        vec = None
        if step == 0 and sample_vec is not None:
            vec = sample_vec
        elif self.can_render_base:
            img = media_image(self.render(dict(spec, lora_scale=0.0)))
            if img is not None:
                vec = self.embed([img])[0]
        if vec is not None:
            self.bases[key] = vec
            self._save()
        return vec

    def prime(self, specs: list[dict]):
        """At step 0, when the weights are the base model's: the base vectors of the cards that have none, rendered
        now (a finetune's only chance to render the base model)."""
        todo = [sp for sp in specs if spec_key(sp) not in self.bases]
        if not todo:
            return
        self.log(f"Likeness: rendering {len(todo)} sample card(s) with the base model, as the floor of their likeness")
        for sp in todo:
            img = media_image(self.render(dict(sp, lora_scale=0.0)))
            if img is not None:
                self.bases[spec_key(sp)] = self.embed([img])[0]
        self._save()

    def score(self, spec: dict, media, step: int) -> dict:
        """{likeness, similarity, target, floor, base_sim, nearest: {file, sim}, near_copy} for one sample (see
        ``Concept.score``), or {} for sound."""
        img = media_image(media)
        if img is None or self.concept is None or not self.concept.paths:
            return {}
        e = self.embed([img])[0]
        r = self.concept.score(e, self.base_for(spec, step, e))
        near = r["nearest"]
        return {"likeness": r["likeness"], "similarity": r["sim"], "target": r["target"], "floor": r["floor"],
                "base_sim": r["base_sim"],
                "nearest": {"file": os.path.basename(near["path"]), "path": near["path"], "sim": near["sim"]},
                "near_copy": r["near_copy"]}


def summarise(entries: list[dict]) -> dict | None:
    """One step's samples -> {likeness (mean), nearest (the highest cosine to a training image), near_copy}."""
    scored = [e for e in entries if e.get("likeness") is not None]
    if not scored:
        return None
    return {"likeness": round(sum(e["likeness"] for e in scored) / len(scored), 1),
            "nearest": max(e["nearest"]["sim"] for e in scored), "near_copy": any(e.get("near_copy") for e in scored)}


# -------------------------------------------------------------------------------------------- as a program ---
def run(request: dict, embedder: Embedder | None = None, progress=None) -> dict:
    """The server's requests. ``request``: {concept: {paths, cache}, images: [paths], floors: [path | null] (one
    per image, the base render it's measured from), top: how many nearest images, duplicates: threshold | null}
    -> {count (images in the concept), self_sim, results: [{sim, target, floor, base_sim, likeness, nearest, top,
        near_copy}] (see ``Concept.score``),
        duplicates: [{a, b, sim}]}."""
    embedder = embedder or Embedder("cuda" if torch.cuda.is_available() else "cpu")
    c = request.get("concept") or {}
    concept = cached_concept(embedder, c.get("paths") or [], c.get("cache"), progress)
    out = {"count": len(concept.paths), "self_sim": round(concept.self_sim, 4), "results": [], "duplicates": []}
    images = request.get("images") or []
    floors = request.get("floors") or [None] * len(images)
    for path, floor_path in zip(images, floors):
        e = embedder([load_image(path)])[0]
        base = embedder([load_image(floor_path)])[0] if floor_path else None
        out["results"].append(concept.score(e, base, int(request.get("top") or 1)))
    if request.get("duplicates") is not None:
        out["duplicates"] = concept.duplicates(float(request["duplicates"]))
    return out


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as f:
        request = json.load(f)

    def progress(done, total):
        print(json.dumps({"progress": done, "total": total}), flush=True)

    try:
        out = run(request, progress=progress)
    except Exception as e:  # noqa: BLE001 - reported to the server as the result
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

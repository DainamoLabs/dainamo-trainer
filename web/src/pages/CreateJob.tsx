import { useEffect, useMemo, useRef, useState } from "react";
import { go } from "../App";
import { api, byFamily, type Dataset, type Family, type Gpu, type JobDetail, type Profile, type SystemRam } from "../api";
import { CheckIssues, useDatasetCheck } from "../components/DatasetCheck";
import { EVAL_DEFAULTS, EvalSettings, evalCost } from "../components/EvalSettings";
import { IconPlay, IconPlus, IconSave, IconX } from "../components/icons";
import { ConfirmDialog, ImagesInput, Info, Num, ProfileOptions, usePoll, useToast } from "../components/ui";
import { OffloadControl, gpuTarget, peakOf, streamedFor, useVramProjection } from "../components/Vram";

type S = Record<string, any>;

const BASE_SETTINGS: S = {
  resolution: 1024, rank: 32, alpha: 32, dropout: 0, learning_rate: 1e-4, optimizer: "adamw8bit",
  lr_scheduler: "constant", warmup_steps: 0, steps: 2000, batch_size: 1, grad_accum: 1, max_grad_norm: 1,
  timestep_sampling: "shift", caption_dropout: 0.05, flip: false, seed: 42, weight_decay: 0,
  save_every: 250, keep_last: 4, sample_every: 250, sample_prompts: [], sample_width: 1024, sample_height: 1024,
  sample_seed: 42, sample_at_start: true, sample_at_end: true, gradient_checkpointing: true,
  keep_alpha: true, offload: 0,
  low_vram: false, // stream the text encoder's layers from system RAM too
  // the adapter trained: a LoRA (rank, alpha) or a LoKr (lokr_factor); or the whole model (families with info.finetune)
  network_type: "lora", lokr_factor: 16, finetune_enabled: false,
  // video families
  num_frames: 39, i2v_probability: 0.5, train_audio: true, audio_loss_weight: 1, sample_frames: 39,
  auto_frames: false, // each clip at its own length, up to MAX_CLIP_FRAMES
  // music families: songs train as clips this long (0: whole songs), and samples are this long
  clip_seconds: 30, sample_seconds: 30,
  // optional checkpoint evaluation (off by default)
  ...EVAL_DEFAULTS,
  // concept sliders (families that train them)
  slider_enabled: false, slider_target: "", slider_positive: "", slider_negative: "", slider_strength: 3,
  slider_sample_strengths: "-1, 1", slider_images: 16, slider_preserve: [], slider_keep_layout: true,
  slider_layout_sigma: 0.9,
  // RefMods (families that make them): references encoded once and saved, instead of training
  refmod_enabled: false, refmod_kind: "visual", refmod_mode: "compressed", refmod_size: 1024, refmod_frames: 22,
  refmod_grid: 16, refmod_refine: 500, refmod_merge: false, refmod_max_tokens: 5120, refmod_seconds: 10,
  refmod_concept: "generic", refmod_description: "", refmod_copy_to: "",
};
// the longest a clip trains at (the server's MAX_FRAMES: 10 s at 24 fps)
const MAX_CLIP_FRAMES = 241;
// what switching to a slider sets: it learns one direction, so it needs fewer steps and less rank
const SLIDER_DEFAULTS: S = { steps: 500, rank: 16, alpha: 16, picker_enabled: false };
// a LoKr learns more slowly than a LoRA at the same rate, so it starts at this multiple of the family's rate
const LOKR_LR_SCALE = 2;
// the optimizers whose state fits beside a finetuned model (the server's finetune.OPTIMIZERS)
const FINETUNE_OPTIMIZERS: [string, string][] = [["adafactor", "Adafactor"], ["adamw8bit", "AdamW 8-bit"]];
// what a job makes: a LoRA or a LoKr of its datasets, a slider (always a LoRA; families with info.sliders), a
// finetune of the whole model (families with info.finetune), or a RefMod of them (families with info.refmods),
// which trains nothing
type Kind = "lora" | "lokr" | "slider" | "finetune" | "refmod";
const KINDS: [Kind, string, string][] = [
  ["lora", "LoRA", "Learns the subject or style of your datasets."],
  ["lokr", "LoKr", "Learns the subject or style of your datasets in a Kronecker-product adapter (LyCORIS): a smaller file, which ComfyUI loads like a LoRA. It learns more slowly, so it starts at twice a LoRA's learning rate."],
  ["slider", "Slider", "A LoRA whose strength moves one attribute either way: plus toward one prompt, minus toward another."],
  ["finetune", "Finetune", "Trains every weight of the model itself rather than an adapter beside it: the most it can learn, for many concepts at once or a large dataset. Each save is a new diffusion-model file as big as the model, which ComfyUI loads in place of the original. It needs far more VRAM and disk than a LoRA, and a much lower learning rate."],
  ["refmod", "RefMod", "Your datasets' images and clips, or their sound, encoded once into a reference file, which the reference-to-video model (ref2va) takes as it takes reference images or reference audio, without encoding them again. Nothing is trained: it takes minutes, with only a VAE loaded. ComfyUI loads it with the ComfyUI-MiniMaxH3Mod nodes."],
];
const CONCEPTS: [string, string][] = [
  ["generic", "Generic"], ["identity", "Identity · a person or character"], ["pose_motion", "Pose & motion"],
  ["clothing", "Clothing"], ["background", "Background"], ["style", "Style"],
];
const AUDIO_CONCEPTS: [string, string][] = [
  ["voice", "Voice"], ["singing", "Singing"], ["music_style", "Music style"], ["sound_fx", "Sound effect"],
  ["ambience", "Ambience"], ["generic", "Generic"],
];
const AUDIO_TOKENS_PER_SECOND = 80; // 40 audio latents a second, one token for each in each stereo channel
// MiniMax H3 clips: 17n + 5 frames are 5n + 2 latent frames, and fewer than 5 is the first frame alone
const refFrames = (frames: number) => (frames < 5 ? 1 : Math.floor((frames - 5) / 17) * 5 + 2);

// what the "i" beside a setting says, for the ones whose name doesn't
const INFO = {
  profile: "The model files to train with (the model, its text encoder and VAE), set up on the Profiles page.",
  gpu: "Auto starts the job on whichever of the GPUs chosen in Settings is least busy. Picking one makes the job wait for that GPU.",
  repeats: "How many times each of this dataset's files is used per epoch. Raise it to give a small dataset as much weight as a bigger one.",
  resolution: "The size files train at: each is scaled to about this many pixels squared, keeping its shape, and never enlarged. Select several to train every file at each size, so the LoRA works across sizes.",
  captionDropout: "The chance a step trains on a file from this dataset with an empty caption, which keeps the LoRA working without its trigger words. 0 turns it off.",
  flip: "Also trains on a mirrored copy of each file, doubling the items. Leave it off for text, logos or subjects that aren't symmetrical.",
  swap: "Also trains on each song with its left and right channels swapped, doubling the items.",
  learningRate: "How big each update to the LoRA is. Too high and it overcooks (harsh, distorted samples) or falls apart; too low and it barely learns. The model's default is a good start.",
  optimizer: "How each update is worked out. AdamW 8-bit is the usual choice; AdamW keeps its state in full precision and uses more memory; Adafactor uses less; Prodigy finds its own learning rate, so set Learning rate to 1.",
  rank: "How many weights the LoRA has in each layer. Higher can learn finer detail, but makes a bigger file, uses a little more VRAM and overfits small datasets sooner. 16–32 suits most characters and styles.",
  alpha: "Scales the LoRA's effect by alpha ÷ rank: alpha equal to rank is a scale of 1, half of it halves the effective learning rate. If you change rank, change alpha with it to keep the same behaviour.",
  schedule: "How the learning rate changes over the steps. Constant keeps it the same; Cosine and Linear decay lower it to 0 by the last step, which settles the final checkpoints.",
  warmup: "Steps at the start over which the learning rate rises from 0 to its full value, so the first updates aren't too big. 0 starts at the full rate.",
  batch: "Files per step. A larger batch gives steadier updates but uses more VRAM and makes each step slower; an epoch takes fewer steps.",
  audioWeight: "How much the sound's error counts next to the picture's, for clips with sound. 1 weighs them equally; 0 learns nothing from the sound.",
  trainAudio: "Trains the audio part on clips that have sound. Off, only the picture is trained.",
  gradAccum: "Adds up this many batches before each update, for the steadiness of a bigger batch (batch size × this) without its VRAM. Each step takes this many times as long.",
  timesteps: "Which noise levels training spends its steps on. Each model starts on the option that suits it; Uniform spreads the steps evenly over all noise levels.",
  dropout: "Randomly zeroes this share of the LoRA's input in each layer during training, which can stop it memorising a small dataset. 0 turns it off; 0.05–0.1 is typical.",
  maxGradNorm: "Caps the size of each update, so one unusual batch can't knock training off course. 0 turns the cap off; 1 is the usual value.",
  weightDecay: "Pulls the LoRA's weights slightly toward zero at every update, which can curb overfitting. 0 turns it off.",
  autoFrames: "Trains each clip at its own full length, up to 241 frames (10 s), instead of cutting longer ones to one clip length; none is ever sped up or slowed down. VRAM is set by the longest clip, so long clips need more of it.",
  checkpointing: "Recomputes parts of the model during the backward pass instead of keeping them in VRAM: much less VRAM for somewhat slower steps. Leave it on unless the VRAM projection shows room to spare.",
  keepAlpha: "Images with an alpha channel train as RGBA, so the LoRA keeps their transparency. Off, they're composited over white.",
  sampleAtStart: "Renders the sample prompts before the first update, so they show the base model: a baseline to compare the later samples with.",
  finetuneLr: "How big each update to the model's weights is. A finetune changes the model itself, so it takes a far lower rate than a LoRA (about a tenth); too high and the model forgets what it knew.",
  finetuneOptimizer: "Adafactor keeps almost no state beside the model, so it fits where AdamW 8-bit, which keeps 2 bytes a weight (as much again as the model), may not. Either way each weight is updated as soon as its gradient is ready, so the model's gradients never all exist at once.",
  refmodMode: "Compressed averages each reference down to a small grid, then refines the grid (a few hundred steps on the grid alone) so that, scaled back up, it matches the full encode: few tokens, so many references fit, with less detail. Full keeps each encode as it is, every reference cropped to the first one's shape: the most detail, and the most tokens.",
  refmodTokens: "The reference tokens the RefMod may put into the model, which take VRAM and time at every step like the video's own. Over the cap, near-duplicate frames go first, then the rest are spread evenly. 0: no cap. A 1024 × 1024 image encoded in full is 1024 tokens.",
  refmodMerge: "Refines one grid against every reference at once, so it settles on what they have in common, instead of stacking a grid per reference: the fewest tokens, for one subject seen in several shots.",
  refmodConcept: "What the RefMod is of. It's kept in the file, where ComfyUI's RefMod nodes show it; it doesn't change how it's made.",
  refmodKind: "Pictures encodes the datasets' images and clips with the video VAE: a subject, a look or a motion to take after. Sound encodes their sound with the audio VAE (sound files, and clips' soundtracks): a voice, singing, a style of music or a sound. The model takes a sound reference only beside a picture one (a reference image or a visual RefMod).",
  refmodSeconds: "How much sound the RefMod holds: the datasets' sound files and clips with sound, each from its start, in order, until there's this much. A generation uses as much of it as the video is long (MiniMax H3 makes up to 10 s), from the start, as the released pipeline does; ComfyUI may use more.",
};

// A sample prompt card. Blank fields fall back to the Sampling defaults (kept as strings so they can be blank).
// Music cards have a length (seconds) and lyrics instead of a size.
type Card = {
  prompt: string; width: string; height: string; seed: string; lora_scale: string; frames: string; seconds: string;
  images: string; lyrics: string;
};
const CARD_FIELDS = ["width", "height", "seed", "lora_scale", "frames", "seconds"] as const;
const blankCard = (): Card => ({ prompt: "", width: "", height: "", seed: "", lora_scale: "", frames: "", seconds: "", images: "", lyrics: "" });

function toCards(list: unknown): Card[] {
  if (!Array.isArray(list)) return [];
  return list.map((c: any) => {
    if (typeof c === "string") return { ...blankCard(), prompt: c }; // older jobs: plain prompt strings
    const card = blankCard();
    card.prompt = String(c?.prompt ?? "");
    for (const k of CARD_FIELDS) card[k] = c?.[k] == null ? "" : String(c[k]);
    card.images = Array.isArray(c?.images) ? c.images.join(", ") : String(c?.images ?? "");
    card.lyrics = String(c?.lyrics ?? "");
    return card;
  });
}

function fromCards(cards: Card[]) {
  return cards.filter((c) => c.prompt.trim()).map((c) => {
    const out: Record<string, unknown> = { prompt: c.prompt.trim() };
    for (const k of CARD_FIELDS) if (c[k].trim() !== "") out[k] = Number(c[k]);
    const images = c.images.split(",").map((p) => p.trim()).filter(Boolean);
    if (images.length) out.images = images;
    if (c.lyrics.trim()) out.lyrics = c.lyrics.trim();
    return out;
  });
}

const STORE = "dainamo.lastJob";
export const CLONE_KEY = "dainamo.cloneJob";

function load<T>(k: string): T | null {
  try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : null; } catch { return null; }
}

/** What the form starts from to duplicate or edit a job: its config's names, profile (found by name), settings
 * and datasets. */
export function formOf(cfg: any, profiles: Profile[]) {
  return {
    name: cfg.name, output_name: cfg.output_name || "", profile_id: profiles.find((p) => p.name === cfg.profile_name)?.id,
    settings: cfg.settings,
    // jobs from before per-dataset settings use the job's resolution, flip and dropout for every dataset
    datasets: cfg.datasets.map((d: any) => ({
      id: d.id, repeats: d.repeats, resolution: d.resolution ?? cfg.settings.resolution,
      flip: d.flip ?? cfg.settings.flip, caption_dropout: d.caption_dropout ?? cfg.settings.caption_dropout,
    })),
  };
}

// a job being edited: the form's starting point, its profile's name, the steps it has trained, and its status
type Edit = { id: string; job: ReturnType<typeof formOf> & { gpu: string }; profileName: string; done: number; status: string };

/** The Create Job form filled in with a job's settings, saving them back to the job. */
export function EditJob({ id, families }: { id: string; families: Family[] }) {
  const [edit, setEdit] = useState<Edit | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([api.get<JobDetail>(`/api/jobs/${id}`), api.get<Profile[]>("/api/profiles")]).then(([job, profiles]) =>
      setEdit({ id, job: { ...formOf(job.config, profiles), gpu: job.gpu }, profileName: job.config.profile_name,
        done: job.status_detail?.step || 0, status: job.status }), (e) => setError(e.message));
  }, [id]);
  if (error) return <div className="page"><div className="error-box">{error}</div></div>;
  if (!edit) return <div className="page"><div className="muted">Loading…</div></div>;
  return <CreateJob families={families} edit={edit} />;
}

/** A job's training resolutions: the setting is one number, or a list of them. */
export const resList = (v: unknown): number[] =>
  [...new Set((Array.isArray(v) ? v : [v]).map(Number).filter((n) => n > 0))].sort((a, b) => a - b);

const RES_IMAGE = [512, 768, 1024, 1280, 1536];
const RES_VIDEO = [256, 384, 512, 640, 768];

/** Toggle buttons for one or more training resolutions (at least one stays on). */
function Resolutions(props: { v: unknown; on: (v: number | number[]) => void; video: boolean }) {
  const cur = resList(props.v);
  const opts = [...new Set([...(props.video ? RES_VIDEO : RES_IMAGE), ...cur])].sort((a, b) => a - b);
  const toggle = (r: number) => {
    const next = cur.includes(r) ? cur.filter((x) => x !== r) : [...cur, r].sort((a, b) => a - b);
    if (next.length) props.on(next.length === 1 ? next[0] : next);
  };
  return (
    <div className="field">
      <span>Resolution{cur.length > 1 ? "s" : ""}<Info text={INFO.resolution} /></span>
      <div className="seg res-seg" role="group" aria-label="Training resolutions">
        {opts.map((r) => (
          <button key={r} type="button" className={cur.includes(r) ? "on" : ""} aria-pressed={cur.includes(r)} onClick={() => toggle(r)}>{r}</button>
        ))}
      </div>
    </div>
  );
}

// A dataset in the job: which one, how much it repeats against the others, the sizes it trains at,
// and its own flip and caption dropout
type DsCard = { id: string; repeats: number; resolution: number | number[]; flip: boolean; caption_dropout: number };

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
const fmtSize = (bytes: number) => (bytes < 1e6 ? `${Math.max(1, Math.round(bytes / 1e3))} KB` : `${(bytes / 1e6).toFixed(1)} MB`);

/** The health check of a dataset card's dataset, for this model at the card's resolutions (music: clip length);
 * ``onLongest`` hears its longest clip's frames. */
function CardCheck({ id, resolution, family, frames, seconds, auto, onLongest }: {
  id: string; resolution: unknown; family?: string; frames: number; seconds?: number; auto?: boolean;
  onLongest?: (id: string, frames: number) => void;
}) {
  const { result, loading } = useDatasetCheck(id, { resolution: seconds == null ? resList(resolution).join(",") : "", family, frames, seconds, auto });
  useEffect(() => { if (result) onLongest?.(id, result.longest_clip || 0); }, [result]);
  if (!result) return loading ? <div className="faint small">Checking the dataset…</div> : null;
  return <div style={{ opacity: loading ? 0.5 : 1 }}><CheckIssues result={result} compact /></div>;
}

function Sel(props: { label: string; v: string; on: (s: string) => void; opts: [string, string][]; help?: string; info?: string }) {
  return (
    <label className="field">
      <span>{props.label}{props.info && <Info text={props.info} />}</span>
      <select value={props.v} onChange={(e) => props.on(e.target.value)}>
        {props.opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      {props.help && <span className="help">{props.help}</span>}
    </label>
  );
}

export default function CreateJob({ families, edit }: { families: Family[]; edit?: Edit }) {
  const toast = useToast();
  const [profiles] = usePoll<Profile[]>("/api/profiles", 0);
  const [datasets] = usePoll<Dataset[]>("/api/datasets", 0);
  const [sys] = usePoll<{ gpus: Gpu[]; ram?: SystemRam }>("/api/system", 0);

  const clone = useMemo(() => {
    if (edit) return edit.job;
    const c = load<any>(CLONE_KEY);
    localStorage.removeItem(CLONE_KEY);
    return c;
  }, []);
  const last = clone || load<any>(STORE);

  const [name, setName] = useState(edit ? clone.name : clone ? clone.name + " v2" : "");
  const [outName, setOutName] = useState(edit ? clone.output_name : "");
  const [profileId, setProfileId] = useState<string>(last?.profile_id || "");
  const [gpu, setGpu] = useState<string>(last?.gpu || "auto");
  const [s, setS] = useState<S>(() => {
    const merged = { ...BASE_SETTINGS, ...(last?.settings || {}) };
    if (Array.isArray(merged.sweep_strengths)) merged.sweep_strengths = merged.sweep_strengths.join(", ");
    if (Array.isArray(merged.slider_sample_strengths)) merged.slider_sample_strengths = merged.slider_sample_strengths.join(", ");
    // a duplicated job's resolution is all its datasets' together; new datasets start at the first one's
    const first = clone?.datasets?.[0];
    if (first?.resolution != null) merged.resolution = first.resolution;
    if (first?.flip != null) merged.flip = first.flip;
    if (first?.caption_dropout != null) merged.caption_dropout = first.caption_dropout;
    return merged;
  });
  // s.resolution, s.flip and s.caption_dropout are what a newly added dataset starts with (the model's
  // defaults, or the first dataset of the last job)
  const newCard = (): DsCard => ({ id: "", repeats: 1, resolution: s.resolution, flip: !!s.flip, caption_dropout: Number(s.caption_dropout) || 0 });
  const [dsCards, setDsCards] = useState<DsCard[]>(() => {
    const list: DsCard[] = (Array.isArray(clone?.datasets) ? clone.datasets : []).map((d: any) => ({
      ...newCard(), id: String(d.id), repeats: Math.max(1, Number(d.repeats) || 1), resolution: d.resolution ?? s.resolution,
      ...(d.flip != null ? { flip: !!d.flip } : {}), ...(d.caption_dropout != null ? { caption_dropout: Number(d.caption_dropout) } : {}),
    }));
    return list.length ? list : [newCard()];
  });
  const setDsCard = (i: number, patch: Partial<DsCard>) => setDsCards((cs) => cs.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const [cards, setCards] = useState<Card[]>(() => {
    const c = toCards(last?.settings?.sample_prompts);
    return c.length ? c : [blankCard()];
  });
  const setCard = (i: number, patch: Partial<Card>) => setCards((cs) => cs.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const [busy, setBusy] = useState(false);
  // saving an edit that starts a job that has trained again from step 0: why, to confirm
  const [restartAsk, setRestartAsk] = useState("");
  const set = (k: string) => (v: any) => setS((x) => ({ ...x, [k]: v }));

  const profile = profiles?.find((p) => p.id === profileId);
  const family = families.find((f) => f.id === profile?.family);
  // switching to a profile of another model family applies that family's recommended settings;
  // on first load only an unsupported timestep mode is corrected, so remembered settings survive
  const lastFamily = useRef<string | null>(null);
  useEffect(() => {
    if (!family) return;
    const modes = family.timestep_modes.map((m) => m[0]);
    if (lastFamily.current !== null && lastFamily.current !== family.id) {
      const d = family.training_defaults;
      // the strength sweep brackets the strength the model's LoRAs are used at
      const sweep = [0.6, 0.8, 1, 1.2].map((x) => +(x * (family.lora_strength || 1)).toFixed(2)).join(", ");
      setS((x) => ({ ...x, offload: 0, sweep_strengths: sweep, ...d,
        ...(x.network_type === "lokr" && d.learning_rate != null ? { learning_rate: Number(d.learning_rate) * LOKR_LR_SCALE } : {}),
        ...(x.finetune_enabled && family.finetune ? family.finetune : {}) }));
      const res = family.training_defaults.resolution as DsCard["resolution"] | undefined;
      if (res != null) setDsCards((cs) => cs.map((c) => ({ ...c, resolution: res })));
    } else if (!modes.includes(s.timestep_sampling)) {
      setS((x) => ({ ...x, timestep_sampling: modes[0] }));
    }
    lastFamily.current = family.id;
  }, [family?.id]);
  useEffect(() => {
    if (profiles?.length && !profiles.find((p) => p.id === profileId)) setProfileId(byFamily(profiles, families)[0].profiles[0].id);
  }, [profiles]);

  const isVideo = !!family?.media.includes("video");
  const isAudio = !!family?.media.includes("audio");
  const refmod = !!family?.refmods && !!s.refmod_enabled;
  const finetune = !refmod && !!family?.finetune && !!s.finetune_enabled;
  const slider = !refmod && !finetune && !!family?.sliders && !!s.slider_enabled;
  const lokr = !slider && !finetune && s.network_type === "lokr";
  const kind: Kind = refmod ? "refmod" : finetune ? "finetune" : slider ? "slider" : lokr ? "lokr" : "lora";
  const preserve: string[] = Array.isArray(s.slider_preserve) ? s.slider_preserve : [];
  // a slider keeps either its layout or the attributes listed (the attributes stay put while the layout is chosen)
  const keepLayout = !!s.slider_keep_layout;
  // the learning rate follows the network while it's still the other network's default
  const setKind = (to: Kind) => {
    if (to === kind) return;
    // a RefMod keeps the training settings as they are, for switching back
    if (to === "refmod") return setS((x) => ({ ...x, refmod_enabled: true }));
    const d = family?.training_defaults || {};
    const unslide = slider ? { slider_enabled: false, steps: d.steps ?? BASE_SETTINGS.steps, rank: d.rank ?? BASE_SETTINGS.rank, alpha: d.alpha ?? BASE_SETTINGS.alpha } : {};
    // a finetune starts from the family's finetune settings (its learning rate, optimizer and saves kept)
    if (to === "finetune") return setS((x) => ({ ...x, refmod_enabled: false, finetune_enabled: true, bleed_enabled: false, ...unslide, ...(family?.finetune || {}) }));
    // leaving one goes back to the LoRA's
    const unfine: S = finetune ? { finetune_enabled: false, learning_rate: d.learning_rate ?? BASE_SETTINGS.learning_rate,
      optimizer: d.optimizer ?? BASE_SETTINGS.optimizer, keep_last: d.keep_last ?? BASE_SETTINGS.keep_last } : {};
    const network = to === "lokr" ? "lokr" : "lora";
    const lr = Number(d.learning_rate ?? BASE_SETTINGS.learning_rate);
    const [from, rate] = network === "lokr" ? [lr, lr * LOKR_LR_SCALE] : [lr * LOKR_LR_SCALE, lr];
    setS((x0) => {
      const x = { ...x0, ...unfine };
      return { ...x, refmod_enabled: false, network_type: network, ...((x.network_type !== network || finetune) && x.learning_rate === from ? { learning_rate: rate } : {}),
        ...(to === "slider" ? { slider_enabled: true, ...SLIDER_DEFAULTS } : unslide) };
    });
  };
  // what a dataset has that the model trains on (image models skip clips, music models everything but audio)
  // clips split from one video train once per epoch between them
  const usableFiles = (d: Dataset) => (isAudio ? d.audio || 0 : d.images + (isVideo ? d.video_sources ?? d.videos : 0));
  // the dataset cards that have a dataset chosen, with what's in it
  const chosen = dsCards.flatMap((c) => {
    const d = datasets?.find((x) => x.id === c.id);
    if (!d) return [];
    const files = usableFiles(d);
    // each file is an item per resolution (small images can fill fewer sizes, so this is an upper bound);
    // a song trains once per epoch, as one of its clips picked at random
    return [{ ...c, d, files, items: files * c.repeats * (isAudio ? 1 : resList(c.resolution).length) * (c.flip ? 2 : 1) }];
  });
  // a slider without datasets trains on images the base model renders, at the first card's resolutions
  const rendered = slider && chosen.length === 0;
  const imageCount = chosen.reduce((n, c) => n + c.files * c.repeats, 0);
  const resolutions = rendered ? resList(dsCards[0]?.resolution ?? s.resolution) : resList(chosen.flatMap((c) => resList(c.resolution)));
  const perEpoch = rendered ? (Number(s.slider_images) || 0) * resolutions.length : chosen.reduce((n, c) => n + c.items, 0);
  const sliderReady = !slider || (["slider_target", "slider_positive", "slider_negative"].every((k) => String(s[k] || "").trim())
    && s.slider_positive.trim() !== s.slider_negative.trim());
  // section numbers, after the slider's (or RefMod's) own section when there is one
  const num = (n: number) => String(n + ((slider || refmod) && n > 1 ? 1 : 0)).padStart(2, "0");
  const epochs = perEpoch ? (s.steps * s.batch_size * s.grad_accum) / perEpoch : 0;
  // an Auto clip length trains each clip whole, so the longest one (as the datasets' health checks find it) sets
  // the VRAM; until they're in, the most a clip trains at
  const [longest, setLongest] = useState<Record<string, number>>({});
  const noteLongest = (id: string, frames: number) => setLongest((m) => (m[id] === frames ? m : { ...m, [id]: frames }));
  const autoFrames = !!s.auto_frames;
  const longestKnown = chosen.every((c) => longest[c.id] != null);
  const longestClip = Math.min(MAX_CLIP_FRAMES, Math.max(0, ...chosen.map((c) => longest[c.id] || 0)));
  const clipFrames = autoFrames ? (longestKnown ? Math.max(1, longestClip) : MAX_CLIP_FRAMES) : s.num_frames;
  // turning Auto off keeps the length it showed
  const setAutoFrames = (on: boolean) => setS((x) => ({ ...x, auto_frames: on,
    ...(!on && longestKnown && longestClip ? { num_frames: Math.max(5, longestClip) } : {}) }));
  const missingCaps = chosen.reduce((n, c) => n + c.d.count - c.d.captioned, 0);
  // a RefMod's references: every image and clip of its datasets (one reference when merged), and at most the
  // tokens they'd take if square (a reference of another shape takes fewer), within the cap
  const refImages = chosen.reduce((n, c) => n + c.d.images, 0);
  const refClips = chosen.reduce((n, c) => n + c.d.videos, 0);
  // a sound RefMod: up to its length of the datasets' sound files and clips (the clips that have sound)
  const refSound = refmod && s.refmod_kind === "audio";
  const refSounds = chosen.reduce((n, c) => n + (c.d.audio || 0), 0) + refClips;
  const refCompressed = s.refmod_mode !== "full";
  const perFrame = refCompressed ? (Math.round(Number(s.refmod_grid) / 2) || 1) ** 2 : Math.max(1, Math.round(Number(s.refmod_size) / 32)) ** 2;
  const clipLatents = refFrames(Number(s.refmod_frames) || 1);
  const refMerged = refCompressed && !!s.refmod_merge && refImages + refClips > 1;
  const refTokens = perFrame * (refMerged ? (refClips ? clipLatents : 1) : refImages + refClips * clipLatents);
  const refCap = Number(s.refmod_max_tokens) || 0;
  const refCapped = !!refCap && refTokens > refCap;
  const soundTokens = Math.round((Number(s.refmod_seconds) || 0) * AUDIO_TOKENS_PER_SECOND);
  const soundCapped = !!refCap && soundTokens > refCap;
  const refFits = refSound ? !refCap || refCap >= 2 : !refCap || perFrame <= refCap;
  const setRefKind = (kind: string) => setS((x) => {
    const list = kind === "audio" ? AUDIO_CONCEPTS : CONCEPTS;
    return { ...x, refmod_kind: kind, refmod_concept: list.some(([c]) => c === x.refmod_concept) && x.refmod_concept !== "generic"
      ? x.refmod_concept : list[0][0] };
  });

  const submit = async (restart = false) => {
    // the first dataset's resolution, flip and dropout are remembered as the starting point for new ones
    const first = chosen[0];
    const settings = {
      ...s, sample_prompts: fromCards(cards), slider_enabled: slider, network_type: lokr ? "lokr" : "lora",
      refmod_enabled: refmod, finetune_enabled: finetune, ...(refmod ? { picker_enabled: false, bleed_enabled: false } : {}),
      ...(finetune ? { bleed_enabled: false, sweep_enabled: false, grad_accum: 1, offload: 0 } : {}),
      ...(isAudio ? { bleed_enabled: false } : {}),
      ...(slider ? { picker_enabled: false, ...(keepLayout ? { slider_preserve: [] } : {}) } : {}),
      ...(first ? { resolution: first.resolution, flip: first.flip, caption_dropout: first.caption_dropout }
        : rendered ? { resolution: dsCards[0].resolution } : {}),
    };
    const job = {
      name: name.trim() || `${profile?.name || "job"} ${new Date().toLocaleDateString()}`,
      output_name: outName.trim(),
      profile_id: profileId,
      gpu,
      datasets: chosen.map((c) => ({ id: c.id, repeats: c.repeats, resolution: c.resolution, flip: c.flip, caption_dropout: c.caption_dropout })),
      settings,
    };
    setBusy(true);
    try {
      if (edit) {
        await api.put(`/api/jobs/${edit.id}`, { ...job, restart });
        toast(restart ? "Saved; the job starts again from step 0"
          : refmod && edit.status !== "queued" ? "Saved; it's queued to be made again"
          : edit.done > 0 ? `Changes saved; it carries on from step ${edit.done}` : "Changes saved");
        go("queue", edit.id);
      } else {
        await api.post("/api/jobs", job);
        localStorage.setItem(STORE, JSON.stringify({ profile_id: profileId, gpu, settings }));
        toast("Job added to the queue");
        go("queue");
      }
    } catch (e: any) {
      if (edit && e.status === 409) setRestartAsk(e.message);
      else toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };

  const triggers = chosen.map((c) => c.d.trigger).filter(Boolean);
  // blank seeds walk: Seed + the card's position among the cards that have a prompt (as the trainer does)
  const seedFor = (i: number) => s.sample_seed + cards.slice(0, i).filter((c) => c.prompt.trim()).length;
  const sampler = profile ? `${profile.data.sample_steps ?? "?"} steps at CFG ${profile.data.sample_cfg ?? "?"}` : "";
  // the profile's variant (its field's default when blank), which decides the workflows it has
  const variant = String(profile?.data?.variant || family?.profile_fields.find((f) => f.key === "variant")?.default || "");
  const forVariant = (w: { variants: string[] | null }) => !w.variants || w.variants.includes(variant);
  const imageInput = family?.workflows.find((w) => w.images && forVariant(w))?.images;
  const firstFrame = !!family?.workflows.some((w) => w.id === "i2v" && forVariant(w));
  const lowVram = !!s.low_vram && !!family?.streams_text_encoder;
  // what the VRAM projection depends on
  const vram = useVramProjection(profileId || undefined, {
    resolution: resolutions.length ? resolutions : s.resolution,
    datasets: chosen.map((c) => ({ resolution: c.resolution, video: c.d.videos > 0 })), batch_size: s.batch_size, rank: s.rank, optimizer: s.optimizer,
    network_type: kind === "lokr" ? "lokr" : "lora", lokr_factor: s.lokr_factor, finetune_enabled: finetune,
    gradient_checkpointing: s.gradient_checkpointing, sample_width: s.sample_width, sample_height: s.sample_height,
    low_vram: lowVram,
    // low VRAM samples without the VAE, unless a card's images (a first frame, references) need it
    sample_prompts: lowVram ? fromCards(cards).filter((c) => c.images).map((c) => ({ images: c.images })) : [],
    ...(isVideo ? { num_frames: clipFrames, sample_frames: s.sample_frames } : {}),
    ...(isAudio ? { clip_seconds: s.clip_seconds, sample_seconds: s.sample_seconds } : {}),
  }, "train");
  const target = gpuTarget(sys?.gpus, gpu);
  const vramPeak = !refmod && vram && !vram.error && vram.blocks ? peakOf(vram, finetune ? 0 : streamedFor(vram, s.offload)) : null;
  // what each save of a finetune takes on disk: the whole model, in bf16
  const saveGb = finetune && vram && !vram.error ? (vram.adapter_params * 2) / 1e9 : 0;

  return (
    <div className="page">
      <div className="page-head">
        {edit ? (
          <div style={{ minWidth: 0 }}>
            <div className="small"><a href="#/queue">Queue</a> <span className="faint">/</span> <a href={`#/queue/${edit.id}`}>{edit.job.name}</a> <span className="faint">/</span></div>
            <h1>Edit job</h1>
            <p>{refmod ? (edit.status === "queued" ? "Change its settings before it's made."
                : "Saving makes the RefMod again with your changes, replacing the one it made.")
              : edit.done > 0
              ? `It has trained ${plural(edit.done, "step")}, and carries on from its last save with your changes. Changing its model, network (LoRA or LoKr, rank, alpha, or a finetune), optimizer or slider starts it again from step 0 instead, as does changing what the checkpoint picker or concept bleed check scores while it's on (with the picker on, that includes the datasets, resolution, clip length and seed).`
              : "Change its settings before it trains."}</p>
          </div>
        ) : (
          <div>
            <h1>Create Job</h1>
            <p>{refmod ? "Make a RefMod of your datasets and add it to the queue." : finetune ? "Configure a finetune of the whole model and add it to the queue." : "Configure a LoRA training run and add it to the queue."}</p>
          </div>
        )}
      </div>

      <div className="grid create-layout" style={{ gridTemplateColumns: "minmax(0,1fr) 300px", alignItems: "start" }}>
        <div className="card">
          <div className="section">
            <div className="section-head"><span className="num">01</span><h3>Basics</h3></div>
            <div className="grid cols-2">
              <label className="field"><span>Job name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="my_character_v1" /></label>
              <label className="field"><span>File name <span className="faint">· optional</span></span><input value={outName} onChange={(e) => setOutName(e.target.value)} placeholder="defaults to job name" /></label>
              <label className="field">
                <span>Profile<Info text={INFO.profile} /></span>
                {profiles && profiles.length === 0 ? (
                  <button className="btn" onClick={() => go("profiles")}>Create a profile first →</button>
                ) : (
                  <select value={profileId} onChange={(e) => setProfileId(e.target.value)}>
                    <ProfileOptions profiles={profiles || []} families={families} />
                  </select>
                )}
                {edit && !edit.job.profile_id && <span className="help">This job's profile, {edit.profileName}, is gone; it trains with the one chosen here.</span>}
              </label>
              <Sel label="GPU" info={INFO.gpu} v={gpu} on={setGpu} opts={[["auto", "Auto (queue decides)"], ...(sys?.gpus || []).map((g) => [String(g.index), `GPU ${g.index} · ${g.name.replace("NVIDIA GeForce ", "")}`] as [string, string])]} />
              {family && (
                <div className="field">
                  <span>{family.refmods ? "Make" : "Train"}</span>
                  <div className="seg" role="group" aria-label="What to train">
                    {KINDS.filter(([k]) => (k !== "slider" || family.sliders) && (k !== "refmod" || family.refmods) && (k !== "finetune" || family.finetune)).map(([k, label]) => (
                      <button key={k} type="button" className={kind === k ? "on" : ""} aria-pressed={kind === k} onClick={() => setKind(k)}>{label}</button>
                    ))}
                  </div>
                  <span className="help">{KINDS.find(([k]) => k === kind)?.[2]}</span>
                </div>
              )}
            </div>
          </div>

          {slider && (
            <div className="section">
              <div className="section-head"><span className="num">02</span><h3>Slider</h3><span className="muted small">these prompts replace captions</span></div>
              <div className="stack" style={{ gap: 12 }}>
                <label className="field">
                  <span>Target</span>
                  <input value={s.slider_target} onChange={(e) => set("slider_target")(e.target.value)} placeholder="a photo of a person" />
                  <span className="help">What the slider changes. The LoRA trains on this prompt, so it works best on prompts like it.</span>
                </label>
                <div className="grid cols-2">
                  <label className="field">
                    <span>Positive (+)</span>
                    <input value={s.slider_positive} onChange={(e) => set("slider_positive")(e.target.value)} placeholder="a photo of a person, smiling broadly" />
                    <span className="help">Where positive strengths take the target.</span>
                  </label>
                  <label className="field">
                    <span>Negative (−)</span>
                    <input value={s.slider_negative} onChange={(e) => set("slider_negative")(e.target.value)} placeholder="a photo of a person, frowning" />
                    <span className="help">Where negative strengths take it.</span>
                  </label>
                </div>
                <div className="grid cols-3">
                  <Num label="Guidance strength" v={s.slider_strength} on={set("slider_strength")} step={0.5} min={0.1}
                    help="How far strength 1 moves, in multiples of the difference between the positive and negative prompts." />
                  <label className="field">
                    <span>Sample strengths</span>
                    <input value={s.slider_sample_strengths} onChange={(e) => set("slider_sample_strengths")(e.target.value)} placeholder="-1, 1" />
                    <span className="help">Each sample prompt renders at each of these; step 0 shows the base model.</span>
                  </label>
                  {rendered && (
                    <Num label="Images to render" v={s.slider_images} on={set("slider_images")} min={1}
                      help="With no dataset, the base model renders this many images of the target to train on." />
                  )}
                </div>
                <div className="grid cols-2">
                  <Sel label="Preservation" v={keepLayout ? "layout" : "attributes"} on={(v) => set("slider_keep_layout")(v === "layout")}
                    opts={[["layout", "Keep the layout · expression, lighting, style"], ["attributes", "Keep listed attributes · age, weight, shape"]]} />
                  {keepLayout && (
                    <Num label="Above noise level" v={s.slider_layout_sigma} on={set("slider_layout_sigma")} step={0.05} min={0.05} />
                  )}
                </div>
                <div className="help" style={{ marginTop: -4 }}>
                  {keepLayout
                    ? "For sliders that change a detail. The noisiest steps decide the layout, clothes and background; above this noise level (1 is pure noise) the slider is trained to change nothing, so they stay as they are."
                    : "For sliders that change shape, which the noisiest steps decide. Each step adds one of these attributes, or none, to all three prompts, so the slider learns the change itself rather than what tends to come with it."}
                </div>
                {!keepLayout && (
                  <div className="field">
                    <span>Keep the same <span className="faint">· optional</span></span>
                    <div className="stack" style={{ gap: 6 }}>
                      {preserve.map((p, i) => (
                        <div className="row" key={i}>
                          <input value={p} onChange={(e) => set("slider_preserve")(preserve.map((x, j) => (j === i ? e.target.value : x)))}
                            placeholder={["in a park", "wearing a dark jacket", "an old man"][i % 3]} />
                          <button className="btn ghost icon" title="Remove" aria-label="Remove attribute"
                            onClick={() => set("slider_preserve")(preserve.filter((_, j) => j !== i))}><IconX /></button>
                        </div>
                      ))}
                      {preserve.length < 12 && (
                        <button className="btn add-prompt" onClick={() => set("slider_preserve")([...preserve, ""])}><IconPlus /> Add attribute</button>
                      )}
                    </div>
                    <span className="help">Things the slider shouldn't change: settings, clothes, ages.</span>
                  </div>
                )}
                {!sliderReady && <div className="validation warning">Fill in all three prompts, with different positive and negative ones.</div>}
              </div>
            </div>
          )}

          {refmod && (
            <div className="section">
              <div className="section-head"><span className="num">02</span><h3>RefMod</h3><span className="muted small">references encoded once · nothing trained</span></div>
              <div className="stack" style={{ gap: 12 }}>
                <div className="field">
                  <span>Of<Info text={INFO.refmodKind} /></span>
                  <div className="seg" role="group" aria-label="What the RefMod is of">
                    {([["visual", "Pictures"], ["audio", "Sound"]] as const).map(([k, label]) => (
                      <button key={k} type="button" className={(refSound ? "audio" : "visual") === k ? "on" : ""}
                        aria-pressed={(refSound ? "audio" : "visual") === k} onClick={() => setRefKind(k)}>{label}</button>
                    ))}
                  </div>
                  <span className="help">{refSound
                    ? "The datasets' sound, encoded with the audio VAE: a voice, singing, music or a sound."
                    : "The datasets' images and clips, encoded with the video VAE: a subject, a look or a motion."}</span>
                </div>
                {refSound ? (
                  <div className="grid cols-3">
                    <Num label="Length (seconds)" info={INFO.refmodSeconds} v={s.refmod_seconds} on={set("refmod_seconds")} step={1} min={1}
                      help={`Up to ${soundTokens.toLocaleString()} tokens (${AUDIO_TOKENS_PER_SECOND} a second).`} />
                    <Num label="Token cap" info={INFO.refmodTokens} v={s.refmod_max_tokens} on={set("refmod_max_tokens")} step={512} min={0}
                      help={soundCapped ? `The sound is cut to ${(refCap / AUDIO_TOKENS_PER_SECOND).toFixed(1)} s.` : undefined} />
                  </div>
                ) : <>
                <div className="field">
                  <span>Encoding<Info text={INFO.refmodMode} /></span>
                  <div className="seg" role="group" aria-label="How the references are encoded">
                    {([["compressed", "Compressed"], ["full", "Full"]] as const).map(([m, label]) => (
                      <button key={m} type="button" className={(refCompressed ? "compressed" : "full") === m ? "on" : ""}
                        aria-pressed={(refCompressed ? "compressed" : "full") === m} onClick={() => set("refmod_mode")(m)}>{label}</button>
                    ))}
                  </div>
                  <span className="help">{refCompressed
                    ? "Each reference as a small grid, refined to match its full encode: few tokens, less detail."
                    : "Each reference's encode as it is: the most detail, and the most tokens."}</span>
                </div>
                <div className="grid cols-3">
                  <Num label="Size (short edge, px)" v={s.refmod_size} on={set("refmod_size")} step={32} min={256}
                    help={refCompressed ? "What each reference is encoded at before it's compressed; scaled down to it, never up." : "Scaled down to it, never up, in multiples of 32."} />
                  <Num label="Frames per clip" v={s.refmod_frames} on={set("refmod_frames")} min={1}
                    help={`A clip's first frames at 24 fps, snapped down to ${family?.frame_rule || "a length the model takes"}: ${refFrames(Number(s.refmod_frames) || 1)} latent frame(s). Images are one.`} />
                  <Num label="Token cap" info={INFO.refmodTokens} v={s.refmod_max_tokens} on={set("refmod_max_tokens")} step={512} min={0} />
                  {refCompressed && <>
                    <Num label="Grid (cells, long edge)" v={s.refmod_grid} on={set("refmod_grid")} step={2} min={2}
                      help={`Even; the short edge keeps the first reference's shape. ${s.refmod_grid} is ${perFrame} tokens per square frame.`} />
                    <Num label="Refinement steps" v={s.refmod_refine} on={set("refmod_refine")} step={50} min={0}
                      help="How long each grid is refined to match its full encode. 0: the plain average." />
                    <label className="check" style={{ alignSelf: "end", paddingBottom: 8 }}>
                      <input type="checkbox" checked={!!s.refmod_merge} onChange={(e) => set("refmod_merge")(e.target.checked)} /> Merge into one reference<Info text={INFO.refmodMerge} />
                    </label>
                  </>}
                </div>
                </>}
                <div className="grid cols-2">
                  <Sel label="Concept" info={INFO.refmodConcept} v={s.refmod_concept} on={set("refmod_concept")} opts={refSound ? AUDIO_CONCEPTS : CONCEPTS} />
                  <label className="field">
                    <span>Save a copy to <span className="faint">· optional</span></span>
                    <input value={s.refmod_copy_to} onChange={(e) => set("refmod_copy_to")(e.target.value)} placeholder="/path/to/ComfyUI/models/refmods" />
                    <span className="help">A folder the finished file is copied to, such as ComfyUI's models/refmods.</span>
                  </label>
                </div>
                <label className="field">
                  <span>Description <span className="faint">· optional</span></span>
                  <textarea rows={2} value={s.refmod_description} onChange={(e) => set("refmod_description")(e.target.value)} placeholder="Who or what it is, and how it was made" />
                </label>
                {!refFits && (refSound
                  ? <div className="validation warning">The token cap is below one audio latent (2 tokens): raise it, or set 0 for no cap.</div>
                  : <div className="validation warning">One {refCompressed ? "grid" : "frame"} takes {perFrame} tokens, more than the cap: {refCompressed ? "lower the grid" : "lower the size"}, or raise the cap.</div>)}
                <div className="help">
                  {refSound
                    ? <>Use it in the Sandbox's Reference to video with a ref2va profile, beside a reference image or a visual
                      RefMod (the model takes a sound reference only with a picture one), or in ComfyUI with the
                      ComfyUI-MiniMaxH3Mod nodes. Only the audio VAE is loaded to make it, so the profile can be fl2va or
                      ref2va. Captions aren't used.</>
                    : <>Use it in the Sandbox's Reference to video with a ref2va profile, beside or instead of reference images,
                      or in ComfyUI from models/refmods with the ComfyUI-MiniMaxH3Mod nodes. Only the video VAE is loaded to
                      make it, so the profile can be fl2va or ref2va. Captions aren't used.</>}
                </div>
              </div>
            </div>
          )}

          <div className="section">
            <div className="section-head"><span className="num">{num(2)}</span><h3>Datasets</h3><span className="muted small">{refSound ? "their sound, in order, up to the length" : refmod ? "every image and clip is a reference" : slider ? "optional for a slider · pictures of the target" : "mix several · repeats weight one against the others"}</span></div>
            {datasets && datasets.length === 0 ? (
              <div className="empty" style={{ padding: 20 }}>No datasets yet. <a href="#/datasets">Create one →</a></div>
            ) : (
              <div className="stack" style={{ gap: 10 }}>
                {dsCards.map((c, i) => {
                  const d = datasets?.find((x) => x.id === c.id);
                  const usable = d ? usableFiles(d) : 0;
                  const items = usable * c.repeats * (isAudio ? 1 : resList(c.resolution).length) * (c.flip ? 2 : 1);
                  return (
                    <div className="prompt-card" key={i}>
                      <div className="prompt-card-body">
                        <div className="ds-card-head">
                          <label className="field">
                            <span>Dataset</span>
                            <select value={d ? c.id : ""} onChange={(e) => setDsCard(i, { id: e.target.value })}>
                              <option value="" disabled>Choose a dataset…</option>
                              {datasets?.map((x) => (
                                <option key={x.id} value={x.id} disabled={x.id !== c.id && dsCards.some((o) => o.id === x.id)}>{x.name}</option>
                              ))}
                            </select>
                          </label>
                          {!refmod && (
                            <label className="field">
                              <span>Repeats<Info text={INFO.repeats} /></span>
                              <input type="number" min={1} value={c.repeats} onChange={(e) => setDsCard(i, { repeats: Math.max(1, Number(e.target.value) || 1) })} />
                            </label>
                          )}
                        </div>
                        {d && (
                          <div className="ds-card-info muted small">
                            {[d.images && plural(d.images, "image"),
                              d.videos && plural(d.videos, "clip") + (d.video_sources < d.videos ? ` from ${plural(d.video_sources, "video")}` : ""),
                              d.audio && plural(d.audio, refSound ? "sound file" : "song")].filter(Boolean).join(" · ") || "Empty"}
                            {!refmod && <>{" · "}{d.captioned} captioned</>}{!refmod && usable > 0 && (isAudio
                              ? (s.clip_seconds ? ` · ${plural(items, "song")} per epoch, a clip of each` : ` · ${plural(items, "song")} per epoch`)
                              : ` · ${plural(items, "item")} per epoch`)}
                            {d.trigger && <span className="pill">{d.trigger}</span>}
                          </div>
                        )}
                        {c.id && datasets && !d && <div className="validation warning">This dataset was deleted. Choose another.</div>}
                        {d && !d.exists && <div className="validation warning">Its folder wasn't found: {d.path}</div>}
                        {d && !isVideo && !isAudio && d.videos > 0 && family && (
                          <div className="validation warning">{family.label} trains on images, so {d.videos === 1 ? "its clip is" : `its ${d.videos} clips are`} skipped.</div>
                        )}
                        {d && isAudio && d.images + d.videos > 0 && family && (
                          <div className="validation warning">{family.label} trains on audio, so its images and clips are skipped.</div>
                        )}
                        {d && refSound && d.images > 0 && (
                          <div className="validation warning">A sound RefMod takes sound, so {d.images === 1 ? "its image is" : `its ${d.images} images are`} skipped{d.videos ? " (its clips count if they have sound)" : ""}.</div>
                        )}
                        {d && !isAudio && !refSound && (d.audio || 0) > 0 && family && (
                          <div className="validation warning">{family.label} trains on {isVideo ? "images and clips" : "images"}, so {d.audio === 1 ? "its audio file is" : `its ${d.audio} audio files are`} skipped.</div>
                        )}
                        {!isAudio && !refmod && <Resolutions v={c.resolution} on={(v) => setDsCard(i, { resolution: v })} video={isVideo && (!d || d.videos > 0)} />}
                        {d && !refmod && <CardCheck id={d.id} resolution={c.resolution} family={family?.id} frames={isVideo ? s.num_frames : 0}
                          seconds={isAudio ? Number(s.clip_seconds) || 0 : undefined} auto={isVideo && autoFrames} onLongest={noteLongest} />}
                        {!refmod && <div className="ds-card-opts">
                          {!slider && (
                            <label className="field">
                              <span>Caption dropout<Info text={INFO.captionDropout} /></span>
                              <input type="number" min={0} max={1} step={0.01} value={c.caption_dropout}
                                onChange={(e) => setDsCard(i, { caption_dropout: Math.min(1, Math.max(0, Number(e.target.value) || 0)) })} />
                            </label>
                          )}
                          {isAudio ? (
                            <label className="check">
                              <input type="checkbox" checked={c.flip} onChange={(e) => setDsCard(i, { flip: e.target.checked })} /> Swap stereo channels<Info text={INFO.swap} />
                            </label>
                          ) : (
                            <label className="check">
                              <input type="checkbox" checked={c.flip} onChange={(e) => setDsCard(i, { flip: e.target.checked })} /> Flip horizontally<Info text={INFO.flip} />
                            </label>
                          )}
                        </div>}
                      </div>
                      <button className="btn ghost icon" title="Remove dataset" aria-label="Remove dataset"
                        onClick={() => setDsCards((cs) => (cs.length > 1 ? cs.filter((_, j) => j !== i) : [{ ...cs[0], id: "" }]))}>
                        <IconX />
                      </button>
                    </div>
                  );
                })}
                {(!datasets || dsCards.length < datasets.length) && (
                  <button className="btn add-prompt" onClick={() => setDsCards((cs) => [...cs, newCard()])}><IconPlus /> Add dataset</button>
                )}
              </div>
            )}
            {refSound ? (
              <div className="help" style={{ marginTop: 8 }}>
                The sound files and the clips with sound, in the datasets' order, each from its start, until there
                are {Number(s.refmod_seconds) || 0} s in all, as one sound. Clips without sound and images are skipped.
              </div>
            ) : refmod ? (
              <div className="help" style={{ marginTop: 8 }}>
                Each image and clip is a reference, in the datasets' order. With several, {refCompressed
                  ? "every grid takes the first one's shape" : "every one is cropped to the first one's shape"}. The datasets'
                _refs folders aren't used.
              </div>
            ) : slider ? (
              <div className="help" style={{ marginTop: 8 }}>
                A slider learns from its prompts, so its images only need to show the target; their captions and trigger words aren't used.
                {rendered ? ` With no dataset, the base model first renders ${s.slider_images} images of the target, square at the largest resolution above.` : ""}
              </div>
            ) : isAudio ? (
              <div className="help" style={{ marginTop: 8 }}>
                Songs train as clips of the clip length below; each song's caption is its style tags (genre, instruments,
                mood, voice), with the trigger word added. Caption dropout is the chance a step uses an empty caption, which
                keeps the LoRA working without its trigger words.
              </div>
            ) : (
              <div className="help" style={{ marginTop: 8 }}>
                Files train at each of their dataset's resolutions, bucketed by aspect ratio, so the LoRA works across sizes (smaller ones train faster). Caption dropout is the chance a step uses an empty caption, which keeps the LoRA working without its trigger words.
                {isVideo ? " Images train as stills and clips at the clip length below." : ""} VRAM is set by the largest size{isVideo ? ", and for clips the clip length" : ""}.
              </div>
            )}
          </div>

          {!refmod && <>
          <div className="section">
            <div className="section-head"><span className="num">{num(3)}</span><h3>Training</h3></div>
            <div className="grid cols-3">
              <Num label="Steps" v={s.steps} on={set("steps")} min={1} />
              <Num label="Learning rate" info={finetune ? INFO.finetuneLr : INFO.learningRate} v={s.learning_rate} on={set("learning_rate")} step={finetune ? 0.000001 : 0.00001} />
              {finetune ? (
                <Sel label="Optimizer" info={INFO.finetuneOptimizer} v={FINETUNE_OPTIMIZERS.some(([o]) => o === s.optimizer) ? s.optimizer : "adafactor"}
                  on={set("optimizer")} opts={FINETUNE_OPTIMIZERS} />
              ) : (
                <Sel label="Optimizer" info={INFO.optimizer} v={s.optimizer} on={set("optimizer")} opts={[["adamw8bit", "AdamW 8-bit"], ["adamw", "AdamW"], ["adafactor", "Adafactor"], ["prodigy", "Prodigy (set LR to 1)"]]} />
              )}
              {finetune ? null : lokr ? (
                <Sel label="LoKr factor" v={String(s.lokr_factor)} on={(v) => set("lokr_factor")(Number(v))}
                  opts={[["4", "4 · most weights"], ["8", "8"], ["16", "16"], ["-1", "Auto · fewest weights"]]}
                  help="How each layer is split: the lower the factor, the more weights the LoKr trains." />
              ) : (
                <>
                  <Num label="LoRA rank" info={INFO.rank} v={s.rank} on={set("rank")} min={1} />
                  <Num label="LoRA alpha" info={INFO.alpha} v={s.alpha} on={set("alpha")} min={1} />
                </>
              )}
              <Sel label="LR schedule" info={INFO.schedule} v={s.lr_scheduler} on={set("lr_scheduler")} opts={[["constant", "Constant"], ["cosine", "Cosine"], ["linear", "Linear decay"]]} />
              <Num label="Warmup steps" info={INFO.warmup} v={s.warmup_steps} on={set("warmup_steps")} min={0} />
              <Num label="Batch size" info={INFO.batch} v={s.batch_size} on={set("batch_size")} min={1} />
            </div>
            {isVideo && (
              <div className="grid cols-4" style={{ marginTop: 16 }}>
                <div className="field">
                  <span>Clip length (frames)<Info text={INFO.autoFrames} /></span>
                  <div className="row" style={{ gap: 10 }}>
                    <input type="number" min={5} style={{ flex: 1, minWidth: 0 }} disabled={autoFrames}
                      value={autoFrames ? (longestKnown && longestClip ? longestClip : "") : s.num_frames} placeholder={autoFrames ? "each clip's" : undefined}
                      onChange={(e) => set("num_frames")(e.target.value === "" ? 0 : Number(e.target.value))} />
                    <label className="check"><input type="checkbox" checked={autoFrames} onChange={(e) => setAutoFrames(e.target.checked)} /> Auto</label>
                  </div>
                  <span className="help">{autoFrames
                    ? `Each clip trains whole, up to ${MAX_CLIP_FRAMES} frames, snapped down to ${family?.frame_rule || "a length the model takes"}`
                      + (longestKnown && longestClip ? `; the longest is ${longestClip} (${(longestClip / 24).toFixed(1)} s).` : ".")
                    : `At 24 fps, snapped down to ${family?.frame_rule || "a length the model takes"}. Shorter clips use what they have; longer ones train their start.`}</span>
                </div>
                {firstFrame && (
                  <Num label="First-frame share" v={s.i2v_probability} on={set("i2v_probability")} step={0.05} min={0}
                    help="Share of steps trained with the clip's first frame as a keyframe (image-to-video)." />
                )}
                <Num label="Audio loss weight" info={INFO.audioWeight} v={s.audio_loss_weight} on={set("audio_loss_weight")} step={0.1} min={0} />
                <label className="check" style={{ alignSelf: "end", paddingBottom: 8 }}>
                  <input type="checkbox" checked={s.train_audio} onChange={(e) => set("train_audio")(e.target.checked)} /> Train audio (clips with sound)<Info text={INFO.trainAudio} />
                </label>
              </div>
            )}
            {isAudio && (
              <div className="grid cols-3" style={{ marginTop: 16 }}>
                <Num label="Clip length (seconds)" v={s.clip_seconds} on={set("clip_seconds")} min={0}
                  help="Each song trains as clips this long, spread over it. 0 trains whole songs (up to 6 minutes) with the lyrics in their captions, after a [Lyrics] line." />
              </div>
            )}
            {imageInput?.help && <div className="help" style={{ marginTop: 8 }}>{imageInput.help}</div>}
            <div style={{ marginTop: 16 }}>
              <OffloadControl proj={vram} value={finetune ? 0 : Number(s.offload) || 0} onChange={set("offload")} gpu={target} ram={sys?.ram}
                lowVram={family?.streams_text_encoder ? { on: lowVram, onChange: set("low_vram") } : null} offload={!finetune} />
            </div>
            <details style={{ marginTop: 16 }}>
              <summary className="muted" style={{ cursor: "pointer" }}>Advanced</summary>
              <div className="grid cols-3" style={{ marginTop: 14 }}>
                {!finetune && <Num label="Gradient accumulation" info={INFO.gradAccum} v={s.grad_accum} on={set("grad_accum")} min={1} />}
                <Sel label="Timestep sampling" info={INFO.timesteps} v={s.timestep_sampling} on={set("timestep_sampling")} opts={family?.timestep_modes || [["shift", "Resolution shift"]]} />
                {!finetune && <>
                  <Num label="LoRA dropout" info={INFO.dropout} v={s.dropout} on={set("dropout")} step={0.01} min={0} />
                  <Num label="Max grad norm" info={INFO.maxGradNorm} v={s.max_grad_norm} on={set("max_grad_norm")} step={0.1} min={0} />
                </>}
                <Num label="Weight decay" info={INFO.weightDecay} v={s.weight_decay} on={set("weight_decay")} step={0.001} min={0} />
                <Num label="Seed" v={s.seed} on={set("seed")} />
                <label className="check" style={{ alignSelf: "end", paddingBottom: 8 }}><input type="checkbox" checked={s.gradient_checkpointing} onChange={(e) => set("gradient_checkpointing")(e.target.checked)} /> Gradient checkpointing<Info text={INFO.checkpointing} /></label>
                {finetune && (
                  <div className="help" style={{ gridColumn: "1 / -1" }}>
                    A finetune updates each weight as soon as backward has its gradient, so it takes one batch per step (no
                    gradient accumulation) and has no overall gradient clipping; Adafactor caps its own updates.
                  </div>
                )}
                {family?.id === "qwen_image_2" && (
                  <label className="check" style={{ alignSelf: "end", paddingBottom: 8 }}>
                    <input type="checkbox" checked={s.keep_alpha} onChange={(e) => set("keep_alpha")(e.target.checked)} /> Keep transparency (RGBA)<Info text={INFO.keepAlpha} />
                  </label>
                )}
              </div>
            </details>
          </div>

          <div className="section">
            <div className="section-head"><span className="num">{num(4)}</span><h3>Saving</h3></div>
            <div className="grid cols-3">
              <Num label="Save every (steps)" v={s.save_every} on={set("save_every")} min={0} help="0 = only at the end" />
              <Num label="Keep last N saves" v={s.keep_last} on={set("keep_last")} min={0} help="0 = keep all" />
            </div>
            {finetune && saveGb > 0 && (
              <div className="help" style={{ marginTop: 8 }}>
                Each save is the whole model: about {saveGb.toFixed(1)} GB, so {s.keep_last
                  ? `the last ${s.keep_last} and the final one take about ${((Number(s.keep_last) + 1) * saveGb).toFixed(0)} GB`
                  : `all ${s.save_every ? Math.floor((s.steps - 1) / s.save_every) + 1 : 1} take about ${((s.save_every ? Math.floor((s.steps - 1) / s.save_every) + 1 : 1) * saveGb).toFixed(0)} GB`}.
                Stopping the job saves one too, which it carries on from.
              </div>
            )}
          </div>

          <div className="section">
            <div className="section-head"><span className="num">{num(5)}</span><h3>Sampling</h3></div>
            <div className="grid cols-4">
              <Num label="Sample every (steps)" v={s.sample_every} on={set("sample_every")} min={0} help="0 = no samples during training" />
              {isAudio ? (
                <Num label="Length (seconds)" v={s.sample_seconds} on={set("sample_seconds")} min={5} />
              ) : <>
                <Num label="Width" v={s.sample_width} on={set("sample_width")} step={16} />
                <Num label="Height" v={s.sample_height} on={set("sample_height")} step={16} />
              </>}
              <Num label="Seed" v={s.sample_seed} on={set("sample_seed")} />
              {isVideo && <Num label="Frames" v={s.sample_frames} on={set("sample_frames")} min={1} help="1 renders a still image." />}
            </div>
            <div className="row wrap" style={{ gap: 22, marginTop: 12 }}>
              <label className="check"><input type="checkbox" checked={s.sample_at_start} onChange={(e) => set("sample_at_start")(e.target.checked)} /> Sample before training (step 0)<Info text={INFO.sampleAtStart} /></label>
              <label className="check"><input type="checkbox" checked={s.sample_at_end} onChange={(e) => set("sample_at_end")(e.target.checked)} /> Sample at the end</label>
            </div>
            {profile && (
              <div className="help" style={{ marginTop: 10 }}>
                Sampling uses {sampler} from the <a href="#/profiles">{profile.name}</a> profile
                {profile.data.training_adapter ? ", with the training adapter switched off" : ""}.
              </div>
            )}

            <div className="section-title" style={{ marginTop: 20, marginBottom: 10 }}>Sample prompts ({cards.filter((c) => c.prompt.trim()).length})</div>
            <div className="stack" style={{ gap: 10 }}>
              {cards.map((c, i) => (
                <div className="prompt-card" key={i}>
                  <div className="prompt-card-body">
                    <label className="field">
                      <span>{isAudio ? "Style" : "Prompt"}</span>
                      <textarea rows={2} value={c.prompt} onChange={(e) => setCard(i, { prompt: e.target.value })}
                        placeholder={slider ? s.slider_target || "a photo of a person"
                          : isAudio ? `${triggers[0] || "ohwx"}, indie pop, upbeat, electric guitar, female vocals, English`
                          : `${triggers[0] || "ohwx"}, portrait photo in soft window light`} />
                    </label>
                    {isAudio && (
                      <label className="field">
                        <span>Lyrics <span className="faint">· optional</span></span>
                        <textarea rows={3} value={c.lyrics} onChange={(e) => setCard(i, { lyrics: e.target.value })}
                          placeholder={"[Verse]\nWalking down the street tonight\n\n[Chorus]\n…"} />
                      </label>
                    )}
                    <div className="prompt-card-fields">
                      {([
                        ...(isAudio ? [["seconds", "Length (s)", `${s.sample_seconds}`, 1]] as const
                          : [["width", "Width", `${s.sample_width}`, 16], ["height", "Height", `${s.sample_height}`, 16]] as const),
                        ["seed", "Seed", `${seedFor(i)}`, 1],
                        ...(slider || finetune ? [] : [["lora_scale", "LoRA strength", (family?.lora_strength ?? 1).toFixed(1), 0.05]] as const),
                        ...(isVideo ? [["frames", "Frames", `${s.sample_frames}`, 1]] as const : []),
                      ] as const).map(([k, label, def, step]) => (
                        <label className="field" key={k}>
                          <span>{label}</span>
                          <input type="number" step={step} value={c[k]} placeholder={`${def} (default)`}
                            onChange={(e) => setCard(i, { [k]: e.target.value } as Partial<Card>)} />
                        </label>
                      ))}
                    </div>
                    {imageInput && (
                      <label className="field" onClick={(e) => e.preventDefault()}>
                        <span>{imageInput.label} <span className="faint">· optional</span></span>
                        <ImagesInput value={c.images} onChange={(v) => setCard(i, { images: v })} multiple={imageInput.max > 1}
                          placeholder={imageInput.max > 1 ? "comma-separated image paths" : "image path"} />
                      </label>
                    )}
                  </div>
                  <button className="btn ghost icon" title="Remove prompt" aria-label="Remove prompt"
                    onClick={() => setCards((cs) => (cs.length > 1 ? cs.filter((_, j) => j !== i) : [blankCard()]))}>
                    <IconX />
                  </button>
                </div>
              ))}
              <button className="btn add-prompt" onClick={() => setCards((cs) => [...cs, blankCard()])}><IconPlus /> Add prompt</button>
            </div>
            <div className="help" style={{ marginTop: 8 }}>
              Blank fields use the defaults above. A blank seed counts up from Seed so each prompt differs; {slider
                ? "each prompt renders at every sample strength with the same seed, so they compare side by side."
                : finetune ? "the samples before training (step 0) show the base model for comparison."
                : "LoRA strength 0 shows the base model for comparison."}
              {isAudio && ` Songs without lyrics are instrumental; section tags like [Verse] and [Chorus] shape them. ${family?.label} LoRAs train
                without the composer's tokens that every song is made from, so they act at about half strength in use: samples render at
                ${family?.lora_strength ?? 1} unless a card says otherwise, and that's the strength to use them at, in ComfyUI too.`}
            </div>
          </div>

          <div className="section">
            <div className="section-head"><span className="num">{num(6)}</span><h3>{slider ? "Concept bleed" : isAudio || finetune ? "Checkpoint picker" : "Checkpoint picker & concept bleed"}</h3><span className="muted small">optional</span></div>
            <div className="stack" style={{ gap: 12 }}>
              <EvalSettings s={s} set={set} datasets={datasets} triggers={triggers} holdout="choose" imageCount={imageCount} picker={!slider} bleed={!isAudio && !finetune} sweep={!finetune} />
              {isAudio && s.picker_enabled && (
                <div className="help">
                  For music, the validation loss mostly measures how well the LoRA re-creates the held-out songs from their tags; it
                  rises once it starts memorising its songs, which can be before their sound is fully learned. Listen to the samples too.
                </div>
              )}
              {(s.picker_enabled || s.bleed_enabled) && (
                <div className="help">
                  Each saved checkpoint costs about {evalCost(s, s.val_dataset ? "N" : s.val_holdout)}, using the profile's sampler for images.
                  {!s.save_every && <span className="validation warning" style={{ marginTop: 6 }}>Save every is 0, so only the final LoRA will be evaluated.</span>}
                </div>
              )}
            </div>
          </div>
          </>}
        </div>

        <div className="card card-pad summary">
          <h2>Summary</h2>
          <div className="kv"><span>Model</span><span>{family?.label || "—"}</span></div>
          <div className="kv"><span>Profile</span><span className="ellipsis">{profile?.name || "—"}</span></div>
          {refSound ? <>
            <div className="kv"><span>Make</span><span>RefMod · sound</span></div>
            <div className="kv"><span>Dataset{chosen.length === 1 ? "" : "s"}</span><span>{chosen.length || "—"}</span></div>
            <div className="kv"><span>Sources</span><span>{refSounds ? [chosen.reduce((n, c) => n + (c.d.audio || 0), 0) && plural(chosen.reduce((n, c) => n + (c.d.audio || 0), 0), "sound file"),
              refClips && plural(refClips, "clip")].filter(Boolean).join(" · ") : "—"}</span></div>
            <div className="kv"><span>Length</span><span>≤ {soundCapped ? (refCap / AUDIO_TOKENS_PER_SECOND).toFixed(1) : s.refmod_seconds} s</span></div>
            <div className="kv"><span>Tokens</span><span>≤ {(soundCapped ? refCap : soundTokens).toLocaleString()}{soundCapped ? " (capped)" : ""}</span></div>
            <div className="kv"><span>File size</span><span>≤ {fmtSize((soundCapped ? refCap : soundTokens) * 32 * 2)}</span></div>
            <div className="kv"><span>Concept</span><span>{AUDIO_CONCEPTS.find(([c]) => c === s.refmod_concept)?.[1] || s.refmod_concept}</span></div>
          </> : refmod ? <>
            <div className="kv"><span>Make</span><span>RefMod · {refCompressed ? "compressed" : "full"}</span></div>
            <div className="kv"><span>Dataset{chosen.length === 1 ? "" : "s"}</span><span>{chosen.length || "—"}</span></div>
            <div className="kv"><span>References</span><span>{refImages + refClips
              ? [refImages && plural(refImages, "image"), refClips && plural(refClips, "clip")].filter(Boolean).join(" · ") + (refMerged ? " · merged" : "")
              : "—"}</span></div>
            <div className="kv"><span>{refCompressed ? "Grid" : "Size"}</span><span>{refCompressed ? `${s.refmod_grid} cells, long edge` : `${s.refmod_size} px short edge`}</span></div>
            <div className="kv"><span>Tokens</span><span>{refImages + refClips ? `≤ ${(refCapped ? refCap : refTokens).toLocaleString()}${refCapped ? " (capped)" : ""}` : "—"}</span></div>
            <div className="kv"><span>File size</span><span>{refImages + refClips ? `≤ ${fmtSize((refCapped ? refCap : refTokens) * 4 * 24 * 2)}` : "—"}</span></div>
            <div className="kv"><span>Concept</span><span>{CONCEPTS.find(([c]) => c === s.refmod_concept)?.[1].split(" · ")[0] || s.refmod_concept}</span></div>
          </> : <>
          <div className="kv"><span>Training adapter</span><span>{profile?.data?.training_adapter ? (finetune ? "not used" : "on") : "off"}</span></div>
          {slider && <div className="kv"><span>Train</span><span>Slider</span></div>}
          <div className="kv"><span>{finetune ? "Train" : "Network"}</span><span>{finetune
            ? `Whole model${vram && !vram.error && vram.adapter_params ? ` · ${(vram.adapter_params / 1e9).toFixed(2)}B weights` : ""}`
            : lokr ? `LoKr · factor ${s.lokr_factor === -1 ? "auto" : s.lokr_factor}` : `LoRA · rank ${s.rank}`}</span></div>
          {vram && !vram.error && vram.adapter_params > 0 && (
            <div className="kv"><span>File size</span><span>≈ {finetune ? `${saveGb.toFixed(1)} GB` : `${Math.round((vram.adapter_params * 2) / 1e6)} MB`}</span></div>
          )}
          {slider && <div className="kv"><span>Keeps</span><span>{keepLayout ? `layout above ${s.slider_layout_sigma}`
            : preserve.filter((p) => p.trim()).length ? plural(preserve.filter((p) => p.trim()).length, "attribute") : "—"}</span></div>}
          <div className="kv"><span>Dataset{chosen.length === 1 ? "" : "s"}</span><span>{chosen.length || (rendered ? `${s.slider_images} rendered images` : "—")}</span></div>
          {isAudio ? <>
            <div className="kv"><span>Clips</span><span>{Number(s.clip_seconds) ? `${s.clip_seconds} s` : "whole songs"}</span></div>
            <div className="kv"><span>Songs</span><span>{perEpoch || "—"}</span></div>
          </> : <>
            <div className="kv"><span>Resolution{resolutions.length > 1 ? "s" : ""}</span><span>{resolutions.join(" · ") || "—"}</span></div>
            <div className="kv"><span>Items per epoch</span><span>{perEpoch || "—"}</span></div>
            <div className="kv"><span>Epochs</span><span>{epochs ? epochs.toFixed(1) : "—"}</span></div>
          </>}
          <div className="kv"><span>Steps</span><span>{s.steps}</span></div>
          <div className="kv"><span>Checkpoints</span><span>{s.save_every ? Math.floor((s.steps - 1) / s.save_every) + 1 : 1}</span></div>
          {!slider && <div className="kv"><span>Checkpoint picker</span><span>{s.picker_enabled ? "on" : "off"}</span></div>}
          {!isAudio && <div className="kv"><span>Concept bleed</span><span>{s.bleed_enabled ? "on" : "off"}</span></div>}
          <div className="kv"><span>Projected VRAM</span><span className={vramPeak && target && vramPeak.total > target.totalGb ? "vram-peak over" : ""}>
            {vramPeak ? `≈ ${vramPeak.total.toFixed(1)} GB` : "—"}{s.offload && !finetune ? ` · ${s.offload}% offloaded` : ""}{lowVram ? " · low VRAM" : ""}</span></div>
          </>}
          {!slider && !refmod && missingCaps > 0 && <div className="validation warning" style={{ marginTop: 10 }}>{missingCaps} file(s) have no caption — they'll train on the trigger word only.</div>}
          <button className="btn primary" style={{ width: "100%", marginTop: 16, padding: "11px 14px" }} disabled={busy || !profileId || (chosen.length === 0 && !slider) || !sliderReady || (refmod && !refFits)} onClick={() => submit()}>
            {edit ? <><IconSave /> Save changes</> : <><IconPlay /> Add to queue</>}
          </button>
          {edit && <button className="btn ghost" style={{ width: "100%", marginTop: 8 }} onClick={() => go("queue", edit.id)}>Cancel</button>}
          {chosen.length === 0 && !slider && <div className="help" style={{ marginTop: 8, textAlign: "center" }}>Choose at least one dataset</div>}
          {!sliderReady && <div className="help" style={{ marginTop: 8, textAlign: "center" }}>Fill in the slider's prompts</div>}
        </div>
        {restartAsk && (
          <ConfirmDialog title="Start again from step 0?" danger confirmLabel="Save and start again" message={restartAsk}
            onClose={() => setRestartAsk("")} onConfirm={() => submit(true)} />
        )}
      </div>
    </div>
  );
}

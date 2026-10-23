export type ProfileField = {
  key: string;
  label: string;
  kind: "path" | "text" | "number" | "bool" | "select";
  required: boolean;
  help: string;
  default: unknown;
  options: string[] | null;
  hint: string; // path fields: the model the file must be, e.g. "Qwen3-VL-4B"
};

export type Family = {
  id: string;
  label: string;
  media: string[];
  profile_fields: ProfileField[];
  training_defaults: Record<string, unknown>;
  timestep_modes: [string, string][];
  workflows: FamilyWorkflow[];
  frame_rule: string; // video families: the clip lengths the VAE takes
  sliders: boolean; // trains concept sliders
  refmods: boolean; // makes RefMods: references encoded once and saved (MiniMax H3)
  streams_text_encoder: boolean; // "Low VRAM" streams its text encoder from system RAM (not YuE2)
  lora_strength: number; // the strength its LoRAs are meant to be used at (YuE2: 2)
  // trains as a full finetune too: the settings one starts from (learning rate, optimizer, saves kept); null: it can't
  finetune: Record<string, unknown> | null;
  // trains with per-file reference images (an edit's source images) from source datasets paired by file name
  sources: FamilySources | null;
};

// label: one source dataset ("Source dataset"); noun: one of its images ("source image"); variants: the profile
// variants that train with them (null: every one); without: what a file without one trains as
export type FamilySources = { label: string; noun: string; max: number; variants: string[] | null; without: string; help: string };

export type WorkflowImages = { label: string; min: number; max: number; help: string };
export type MediaKind = "image" | "video" | "audio";
export type Workflow = { id: string; label: string; output: MediaKind };
// a workflow a family supports; variants limits it to profiles with that variant (null = any)
// refmods: it takes RefMods as references, as well as or instead of images
// ``audio``: reference sound files it takes for a video (with audio RefMods, at most ``max`` in all)
export type FamilyWorkflow = Workflow & {
  images: WorkflowImages | null; variants: string[] | null; refmods: boolean;
  audio: { label: string; max: number; help: string } | null;
};

export type Gpu = { index: number; name: string; mem_used: number; mem_total: number; util: number; temp: number };
export type SystemRam = { total_gb: number; available_gb: number; used_gb: number };
// the CPU's temperature is null where it isn't reported (Windows, or a Linux machine without the sensor driver)
export type Cpu = { name: string; percent: number; temp: number | null; cores: number; threads: number };
export type Hardware = { gpus: Gpu[]; cpu: Cpu; ram: SystemRam };

export type Profile = { id: string; name: string; family: string; data: Record<string, unknown>; updated: number };

/** Profiles under their families, in the families' order (a family this server no longer has comes last, under its
 *  id), each family's by name as the server sends them. */
export function byFamily(profiles: Profile[], families: Family[]) {
  const ids = new Set([...families.map((f) => f.id), ...profiles.map((p) => p.family)]);
  return [...ids]
    .map((id) => ({ id, label: families.find((f) => f.id === id)?.label || id, profiles: profiles.filter((p) => p.family === id) }))
    .filter((g) => g.profiles.length);
}

export type Dataset = {
  id: string;
  name: string;
  path: string;
  managed: number;
  trigger: string;
  count: number;
  images: number;
  videos: number;
  audio: number;
  captioned: number;
  bytes: number; // its images, videos and sound files
  video_sources: number; // what its videos count as per epoch: clips split from one video count once
  split: { source: { id: string; name: string }; changes: number } | null; // made by Split into clips: changes not cut yet
  exists: boolean;
  cover: string | null;
  caption_options: CaptionOptions | null; // the Auto-Caption choices it was last captioned with
};
// clip_frames: describe the first N frames of clips (0: all); speech: a Whisper model ("" = off)
export type CaptionOptions = { style: string; clip_frames: number; speech: string };
export type WhisperModel = { id: string; size: string };

export type MediaItem = {
  path: string; name: string; kind: MediaKind; size: number | null; caption: string; has_caption: boolean;
  cut_from: { source: string; start: number } | null; // a clip from Split into clips: its video, and where
};

export type Progress = {
  phase: string | null;
  step: number | null;
  total_steps: number | null;
  loss: number | null;
  avg_loss: number | null;
  lr: number | null;
  it_per_sec: number | null;
  eta_seconds: number | null;
  message: string | null;
  vram_gb: number | null;
};

export type Sample = {
  step: number; index: number; prompt: string; file: string;
  width?: number; height?: number; seed?: number; lora_scale?: number; // absent on jobs from before prompt cards
  kind?: MediaKind; poster?: string; frames?: number; images?: string[];
  seconds?: number; lyrics?: string; score?: string; // music: the song's length and lyrics, and its score (ABC) file
};
export type Checkpoint = { step: number; file: string; final: boolean };

export type SandboxParams = {
  workflow: string; prompt: string; negative: string; width: number; height: number; frames: number;
  seconds?: number; lyrics?: string; // music
  steps: number; cfg: number; sampler: string; scheduler: string; shift: number | null; seed: number; count: number;
  loras: { path: string; strength: number }[]; images: string[]; compare_base: boolean; offload?: number;
  model?: string | null; // a finetuned diffusion model in place of the profile's
  refmods?: { path: string; strength: number }[];
  audio?: string[]; // reference sound files
  low_vram?: boolean; // the text encoder streamed from system RAM
  source_prompt?: string; // what was typed, when ``prompt`` is the enhanced version
  grid?: SandboxGrid | null;
};
// compare checkpoints: each checkpoint at each strength (and the base model, if asked), per seed
export type SandboxGrid = {
  job_id: string; job_name: string; checkpoints: { path: string; label: string; step: number | null }[];
  strengths: number[]; base: boolean;
};
export type SandboxOutput = {
  file: string; kind: MediaKind; poster?: string; score?: string; seed: number; variant: "lora" | "base" | "grid";
  checkpoint?: number; strength?: number; // a grid image: indexes into the grid's checkpoints and strengths
};
export type SandboxRun = {
  id: string; created: number; started: number | null; status: "queued" | "running" | "done" | "failed" | "cancelled";
  n?: number; // its number: Run 1 is the first made
  profile_id: string; profile_name: string; family: string; workflow: string; gpu: number; params: SandboxParams;
  outputs: SandboxOutput[]; report: { loras: { label: string; matched: number; total: number }[] | null } | null;
  message: string | null; elapsed: number | null;
};
// model calls done of total, and the step counter: render ``image`` of ``images``, its ``step`` of ``steps``,
// seconds per step and the seconds elapsed and left (null until the first step is timed)
export type SandboxProgress = {
  done: number; total: number; image?: number; images?: number; step?: number; steps?: number;
  sec_per_step?: number | null; elapsed?: number; eta?: number | null;
};
export type SandboxSession = {
  state: "off" | "starting" | "ready" | "busy" | "stopping"; gpu: number | null; message: string;
  progress: SandboxProgress | null; current: string | null; pending: string[]; yielding: boolean;
  idle_seconds: number | null; refmod: SandboxRefMod | null;
};
// the RefMod the Sandbox is making, or made last
export type SandboxRefMod = {
  id: string; name: string; state: "queued" | "running" | "done" | "failed" | "cancelled"; message: string;
  made: MadeRefMod | null;
};
export type MadeRefMod = {
  name: string; file: string; path: string; kind: string; mode: "full" | "compressed"; references: number; sheet: boolean;
  tokens: number; bytes: number; frames: number; width: number; height: number; copied_to: string | null;
  preview: { file: string; frame: number }[];
};
// what making a RefMod will do (POST /api/sandbox/refmod/check): each reference at its encode size, the tokens (and
// how many the cap keeps), the sheet's layout, and the VRAM the largest encode takes beside what's free
export type RefModCheck = {
  name: string; refs: { kind: "image" | "video"; width: number; height: number; frames: number }[];
  tokens: number; per_frame: number; cap: number; kept: number; sheet: { columns: number; rows: number } | null;
  vram: { gpu: number; need_gb: number; free_gb: number; loaded: boolean } | null; warning: string | null; error: string | null;
};
export type SandboxStatus = {
  mode: "reserve" | "idle"; gpu_count: number; training_gpus: number[]; blocked: string | null; session: SandboxSession;
  samplers: Record<string, string>; schedulers: Record<string, string>; idle_minutes: number; runs: SandboxRun[];
};
export type SandboxDefaults = {
  steps: number; cfg: number; negative: string; width: number; height: number; frames: number; seconds?: number;
  supports_samplers: boolean; divisibility: number; offload: number; workflows: FamilyWorkflow[];
};
// a RefMod a job or the Sandbox made (GET /api/sandbox/refmods): a Sandbox one has no job_id, and job_name is its name
export type RefModOption = {
  job_id: string | null; job_name: string; file: string; path: string; kind: string; mode: "full" | "compressed";
  frames: number; seconds?: number | null; tokens: number; references: number; sheet?: boolean; created: number;
};
export type LoraOption = {
  job_id: string; job_name: string; job_status: string; file: string; path: string; step: number | null;
  final: boolean; recommended: boolean;
};
// a finetune job's checkpoint: a whole diffusion model (GET /api/sandbox/models)
export type ModelOption = Omit<LoraOption, "recommended">;

export type Probe = { prompt: string; file: string; bleed: number; drift?: number };
export type Eval = { step: number; file: string | null; val_loss?: number; bleed?: number; drift?: number; probes?: Probe[] };
export type Recommendation = { step: number; file: string; reason: string };
export type Sweep = {
  step: number; file: string; strengths: number[];
  images: { strength: number; index: number; prompt: string; file: string; kind?: string; poster?: string }[];
};

// a checkpoint evaluation of a finished job (Evaluate checkpoints), apart from the evaluation done in training
export type Evaluation = {
  state: "running" | "completed" | "stopped" | "failed"; message?: string; settings: Record<string, any>;
  evals: Eval[]; recommended: Recommendation | null; sweep: Sweep | null;
  started_at: number; finished_at?: number; vram_gb?: number | null;
};

// bytes a job's folder uses: LoRA checkpoints (output), samples, cached captions and latents, training state
export type DiskUse = { output: number; samples: number; cache: number; state: number; other: number; total: number };
// a job's: also the bytes of the shared cache's files it uses (encoded captions and latents; not in its total)
export type JobDiskUse = DiskUse & { shared: number };

export type Job = {
  id: string;
  name: string;
  status: "queued" | "running" | "stopping" | "completed" | "failed" | "stopped" | "interrupted";
  family: string;
  media: string[]; // its model's: "image", "video" or "audio"
  profile_name: string;
  gpu: string;
  assigned_gpu: number | null;
  created: number;
  started: number | null;
  finished: number | null;
  progress: Progress;
  last_sample: Sample | null;
  loss_curve: number[]; // its loss so far, smoothed, in at most 60 values evenly spaced over its steps
  recommended: Recommendation | null;
  datasets: string[];
  evaluating: boolean; // queued or running for a checkpoint evaluation, not training
  slider: boolean; // a concept slider; without datasets it trains on images it renders
  network: "lora" | "lokr" | "full"; // the adapter it trains, or "full": a finetune of the whole model
  refmod: boolean; // makes a RefMod of its datasets instead of training
  extract: boolean; // extracts a LoRA from a finetune's checkpoint instead of training
  merge: boolean; // merges LoRAs into one instead of training
};

// the RefMod a job made: its file, shape (latent frames of height × width cells), the reference tokens it puts into
// the model, and a preview of some of its frames (each decoded on its own)
// an audio RefMod (``kind`` audio) has ``seconds`` of sound and ``audio``, its sound decoded, and no frames
export type RefModMade = {
  file: string; kind: "image" | "video" | "audio"; mode: "full" | "compressed"; references: number; frames: number;
  height: number; width: number; tokens: number; bytes: number; copied_to: string | null;
  preview: { file: string; frame: number }[];
  seconds?: number; concept?: string; audio?: { file: string; poster: string };
};

// the LoRA an extract job made from a finetune: how many matrices became LoRAs and how many tensors were kept whole
// (or hadn't changed), the share of the change's squared size it keeps (overall, and in the layer keeping least),
// how far the finetune moved the weights (relative), and the files it compared
export type ExtractMade = {
  file: string; rank: number; layers: number; whole: number; unchanged: number; missing: number; kept: number;
  worst: { name: string; kept: number } | null; change: number; bytes: number; source: string; base: string;
};

// the LoRA a merge job made: layers saved as LoRAs (``cut`` of them to the rank; rank 0: exact) and kept whole, the
// share of the merged change's squared size it keeps (overall, and in the layer keeping least), and what it merged
export type MergeMade = {
  file: string; rank: number; layers: number; cut: number; whole: number; kept: number; bytes: number;
  worst: { name: string; kept: number } | null;
  inputs: { file: string; strength: number; matched: number; layers: number }[];
};

export type JobDetail = Job & {
  config: any;
  dir: string;
  resumable: boolean; // its training state is kept, so it can train on from its last step
  status_detail: Progress & {
    samples?: Sample[]; checkpoints?: Checkpoint[];
    evals?: Eval[]; recommended?: Recommendation | null; sweep?: Sweep | null;
    evaluation?: Evaluation | null;
    refmod?: RefModMade;
    extract?: ExtractMade;
    merge?: MergeMade;
  };
};

export type NotifyKind = "ntfy" | "discord" | "slack" | "json";
export type CaptionStyle = { id: string; name: string; prompt: string };
export type Settings = {
  queue: { mode: "sequential" | "parallel_single" | "parallel_multi"; gpus: number[]; max_per_gpu: number };
  sandbox: { idle_minutes: number };
  notifications: {
    url: string; kind: NotifyKind; link: string; on_completed: boolean; on_failed: boolean; on_stopped: boolean;
  };
  enhancer: { image: string; edit: string; video: string; max_tokens: number; temperature: number };
  caption_styles: { default: string; styles: CaptionStyle[] };
  captioner: {
    endpoint: string;
    model: string;
    api_key: string;
    max_tokens: number;
    temperature: number;
    video_frames: number;
    video_mode: "frames" | "sheet";
    /** Who answers Auto-Caption and Enhance prompt: a server the user runs (``endpoint``), or one Dainamo runs. */
    provider: CaptionProvider;
  };
  /** The llama-server Dainamo runs for the "llama" provider. */
  llama: {
    binary: string; model: string; mmproj: string; gpu: string; ctx_size: number; extra_args: string; idle_minutes: number;
  };
  /** The text encoder Dainamo runs for the "encoder" provider. */
  text_encoder: { path: string; gpu: string; idle_minutes: number };
  speech: { model: string; device: "auto" | "cpu" };
  /** ComfyUI's models folder: the model manager finds files in it and downloads into its subfolders. */
  models: { folder: string };
};

/** A file the model manager can download for a profile field (a family driver's ``downloads``). */
export type ModelDownloadOption = {
  id: string; field: string; repo: string; path: string; name: string; size: number; label: string;
  folder: string; // the models folder's subfolder it goes in: diffusion_models, text_encoders, vae, loras, checkpoints
  gated: boolean; recommended: boolean; variant: string; note: string; url: string;
  have: string | null; // where it is on this machine (the file of that name and size)
  task: string | null; // its unfinished download
};
/** A file on this machine that a profile field can use. */
export type FoundModel = {
  path: string; name: string; size: number; where: string; quant: string[]; unsupported: string[];
  download: string | null; // the catalog file it is
};
export type ModelField = { key: string; label: string; hint: string; required: boolean; downloads: ModelDownloadOption[]; found: FoundModel[] };
export type ModelFamily = { id: string; label: string; fields: ModelField[] };
export type ModelTask = {
  id: string; download: string; family: string; family_label: string; field: string; repo: string; name: string;
  label: string; dest: string; size: number; done: number; speed: number; gated: boolean; error: string;
  state: "queued" | "checking" | "downloading" | "verifying" | "done" | "failed" | "cancelled" | "interrupted";
};
export type ModelsOverview = {
  folder: string; folder_ok: boolean; folders: Record<string, string>; token: boolean;
  families: ModelFamily[]; tasks: ModelTask[];
};

export type CaptionProvider = "server" | "llama" | "encoder";

/** The local LLM server Dainamo runs (GET /api/llm): llama-server or a text encoder. */
export type LlmStatus = {
  state: "off" | "starting" | "ready" | "stopping" | "failed"; message: string; error: string; hint: string;
  gpu: number | null; device: string | null; endpoint: string | null; pid: number | null; model: string;
  command: string[]; users: number; load_seconds: number | null; loading_seconds: number | null;
  idle_seconds: number | null; auto_ctx: number; on_path: string; provider: "llama" | "encoder"; what: string;
};

/** A text encoder file: which one it is, or why it can't write text. */
export type EncoderInfo = {
  path: string; ok: boolean; name?: string; whose?: string; vision?: boolean; error?: string; profiles?: string[];
};

/** Fired when the server asks for sign-in (remote access with a password); the app shows the sign-in page. */
export const SIGNED_OUT = "dainamo:signed-out";
export type AuthStatus = { required: boolean; signed_in: boolean };

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const r = await fetch(url, init);
  if (r.status === 401 && !url.startsWith("/api/auth")) window.dispatchEvent(new Event(SIGNED_OUT));
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not json */
    }
    throw Object.assign(new Error(msg), { status: r.status });
  }
  const ct = r.headers.get("content-type") || "";
  return (ct.includes("json") ? r.json() : r.text()) as Promise<T>;
}

export const api = {
  get: <T>(url: string) => req<T>("GET", url),
  post: <T>(url: string, body?: unknown) => req<T>("POST", url, body),
  put: <T>(url: string, body?: unknown) => req<T>("PUT", url, body),
  del: <T>(url: string) => req<T>("DELETE", url),
};

export const q = encodeURIComponent;

/** A clock-style duration: 0:09, 1:05, 1:02:03. */
export function fmtClock(sec: number): string {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

export function fmtDuration(sec: number | null | undefined): string {
  if (sec == null || !isFinite(sec)) return "—";
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

export function fmtTime(ts: number | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// VRAM projection (POST /api/estimate; see dainamo/engine/memory.py). Sizes in GB (GiB).
// ram: what the phase holds in system RAM besides the process and the diffusion model's share, which is its
// streamed blocks (model_ram "streamed", the default with the model on the GPU), all of it ("all") or none
export type VramPhase = {
  id: string; label: string; model: boolean; parts: [string, number][];
  ram?: [string, number][]; model_ram?: "all" | "streamed" | null;
};
export type VramProjection = {
  blocks: number; block_gb: number[]; resident_gb: number; inflight_gb: number; context_gb: number;
  fragmentation: number; model_gb: number; process_ram_gb: number; phases: VramPhase[]; notes: string[]; error?: string;
  adapter_params: number; // the trained LoRA's or LoKr's (a finetune's: the model's; training only), saved at 2 bytes each
};

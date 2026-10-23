import { useEffect, useMemo, useRef, useState } from "react";
import {
  api, byFamily, fmtClock, fmtDuration, fmtTime, type Family, type Gpu, type LoraOption, type ModelOption, type Profile, type RefModOption, type SandboxDefaults,
  type SandboxOutput, type SandboxParams, type SandboxProgress, type SandboxRun, type SandboxStatus, type SystemRam,
  type MediaKind, type Workflow,
} from "../api";
import { IconPlayFill, IconPlus, IconRefresh, IconRight, IconSparkle, IconStop, IconTrash, IconX } from "../components/icons";
import { MakeRefModDialog, NEW_REFMOD, type RefModDraft } from "../components/MakeRefMod";
import { AudioPlayer, Empty, ImagesInput, Info, Lightbox, PathInput, ProfileOptions, StatusPill, usePoll, useToast } from "../components/ui";
import { OffloadControl, gpuTarget, offloadFit, useVramProjection } from "../components/Vram";

type LoraRow = { path: string; strength: number; custom: boolean };
// compare checkpoints: a job's checkpoints (paths) at some strengths
type GridForm = { on: boolean; job_id: string; checkpoints: string[]; strengths: string; base: boolean };
type SizeMode = "shape" | "custom" | "match";
// the latest enhanced prompt, and what it was made from: reused until one of those changes
type Enhanced = { prompt: string; seed: number; workflow: string; images: string; text: string };
type Form = {
  profile_id: string; workflow: string; prompt: string; negative: string; width: number; height: number; frames: number;
  seconds: number; lyrics: string; // music
  steps: number; cfg: number; sampler: string; scheduler: string; shift: string; seed: number; random_seed: boolean;
  count: number; loras: LoraRow[]; images: string; compare_base: boolean; gpu: number | null;
  model: string; model_custom: boolean; // a finetuned diffusion model in place of the profile's ("": the profile's)
  refmods: LoraRow[]; // RefMods as references (workflows that take them), each at a strength 0–1
  audio: string[]; // reference sound files (workflows that take them, for a video)
  offload: number | null; // % of the diffusion model's blocks in system RAM (null: the model's default)
  low_vram: boolean; // the text encoder's layers streamed from system RAM too
  size_mode: SizeMode; shape: string; mp: number; // width/height are used as typed only in "custom"
  enhance: boolean; enhanced: Enhanced | null;
  grid: GridForm;
};

const STORE = "dainamo.sandbox.form";
const ADVANCED = "dainamo.sandbox.advanced"; // Advanced settings left open
const SESSION: Record<string, string> = { off: "Model unloaded", starting: "Loading model", ready: "Model loaded", busy: "Generating", stopping: "Unloading" };
// a training sample to re-create, left by "Open in Sandbox" on a job's samples (GET /api/jobs/{id}/sandbox)
export const SANDBOX_PRESET = "dainamo.sandboxPreset";
type Preset = {
  profile_id: string; workflow: string; prompt: string; negative: string; width: number; height: number; frames: number;
  steps: number; cfg: number; seed: number; loras: { path: string; strength: number }[]; images: string[];
  offload: number | null; low_vram?: boolean; notes: string[]; job_name: string; step: number; seconds?: number; lyrics?: string;
  model?: string | null; // a finetune's checkpoint
};

function readPreset(): Preset | null {
  try {
    const v = localStorage.getItem(SANDBOX_PRESET);
    return v ? JSON.parse(v) : null;
  } catch {
    return null;
  }
}

/** The form set up to re-create a training sample: training samples use Euler and the model's own schedule. */
function presetForm(base: Form, p: Preset, video: boolean): Form {
  return {
    ...base, profile_id: p.profile_id, workflow: p.workflow, prompt: p.prompt, negative: p.negative,
    ...sizeState(p.width, p.height, video), frames: p.frames, steps: p.steps, cfg: p.cfg,
    ...(p.seconds != null ? { seconds: p.seconds, lyrics: p.lyrics || "" } : {}),
    sampler: "euler", scheduler: "default", shift: "", seed: p.seed, random_seed: false, count: 1, compare_base: false,
    loras: p.loras.map((l) => ({ ...l, custom: false })), images: p.images.join(", "), offload: p.offload ?? null,
    low_vram: !!p.low_vram, enhance: false, enhanced: null, refmods: [], audio: [], model: p.model || "", model_custom: false,
  };
}
const BLANK: Form = {
  profile_id: "", workflow: "t2i", prompt: "", negative: "", width: 1024, height: 1024, frames: 1, seconds: 30, lyrics: "",
  steps: 20, cfg: 1,
  sampler: "euler", scheduler: "default", shift: "", seed: 42, random_seed: true, count: 1, loras: [], images: "",
  refmods: [], audio: [], compare_base: false, gpu: null, offload: null, low_vram: false, size_mode: "shape", shape: "1:1", mp: 1, enhance: false, enhanced: null,
  model: "", model_custom: false,
  grid: { on: false, job_id: "", checkpoints: [], strengths: "1.0", base: false },
};

const parseStrengths = (s: string) => s.replace(/,/g, " ").split(/\s+/).filter(Boolean).map(Number).filter((n) => Number.isFinite(n));

// Sizes by shape and megapixels (a megapixel is 1024×1024 pixels, as ComfyUI counts it), snapped to 64:
// 1 MP at 9:16 is 768×1344.
const MP = 1024 * 1024;
const RATIOS: Record<string, number> = { "1:1": 1, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16, "5:3": 5 / 3, "3:5": 3 / 5 };
const IMAGE_SHAPES = ["1:1", "4:3", "3:4", "16:9", "9:16"];
const VIDEO_SHAPES = ["1:1", "5:3", "3:5", "16:9", "9:16"];
const ROUND_MP = [0.25, 0.35, 0.5, 1, 1.5, 2]; // what a size is described as, when one of these gives it exactly
const MAX_SIDE = 4096; // the server's limit

export function sizeFor(ratio: number, mp: number): [number, number] {
  const snap = (v: number) => Math.max(128, Math.round(v / 64) * 64);
  return [snap(Math.sqrt(mp * MP * ratio)), snap(Math.sqrt((mp * MP) / ratio))];
}

const nearest = <T,>(xs: T[], cost: (x: T) => number) => xs.reduce((a, b) => (cost(b) < cost(a) ? b : a));

/** The size fields for a width × height: the shape and megapixels that give exactly it, else "custom". */
function sizeState(w: number, h: number, video: boolean): Pick<Form, "size_mode" | "shape" | "mp" | "width" | "height"> {
  const shape = nearest(video ? VIDEO_SHAPES : IMAGE_SHAPES, (s) => Math.abs(Math.log(w / h / RATIOS[s])));
  const fits = (m: number) => sizeFor(RATIOS[shape], m).join() === `${w},${h}`;
  const mp = ROUND_MP.find(fits) ?? Math.round((w * h) / MP * 100) / 100;
  return { size_mode: fits(mp) ? "shape" : "custom", shape, mp, width: w, height: h };
}

const imageList = (s: string) => s.split(",").map((t) => t.trim()).filter(Boolean);

function loadForm(): Form {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) || "{}");
    const form = { ...BLANK, ...saved };
    // remembered from before sizes by shape: keep the size it had
    if (saved.width && !saved.size_mode) Object.assign(form, sizeState(saved.width, saved.height, false));
    return form;
  } catch {
    return BLANK;
  }
}

function saveForm(f: Form) {
  try {
    localStorage.setItem(STORE, JSON.stringify(f));
  } catch {
    /* private mode: the form just isn't remembered */
  }
}

const fileUrl = (run: SandboxRun, name: string) => `/api/sandbox/file/${run.id}/${encodeURIComponent(name)}`;
const base = (p: string) => p.split("/").pop() || p;

// the workflow menu's groups, by what each workflow makes
const WORKFLOW_GROUPS: [MediaKind, string][] = [["image", "Image"], ["video", "Video"], ["audio", "Music"]];

// reference audio: sound files, or videos for their soundtrack
const AUDIO_EXTS = ".mp3,.wav,.flac,.ogg,.m4a,.opus,.aac,.mp4,.mov,.mkv,.webm";

// what the "i" beside a setting says, for the ones whose name doesn't
const INFO = {
  profile: "The model files to generate with (the model, its text encoder and VAE), set up on the Profiles page.",
  workflow: "What to make: from a prompt alone, an edit of an input image, a video from a first frame, and so on. Only the ones this profile's model supports can be chosen.",
  enhance: "Before generating, the caption provider set in Settings rewrites your prompt into a more detailed one, with an instruction for this workflow. The rewrite is kept and used again while the prompt, seed and inputs stay the same.",
  model: "The diffusion model to generate with: the profile's, or a finetune job's checkpoint (a whole model trained from it) in its place, with the profile's text encoder and VAE. Switching reloads only the diffusion model.",
  loras: "LoRAs applied while generating, each at its own strength: 1 is full strength, 0 is off, and a negative strength pushes the other way (as with a slider). Pick a job's checkpoint (★ is the one the checkpoint picker recommended) or any LoRA file.",
  megapixels: "How big to make it, at the shape chosen above. A megapixel is 1024×1024 pixels, and each side is rounded to a multiple of 64. Bigger takes more VRAM and time.",
  compareBase: "Also renders each seed without the LoRAs, beside the images with them, to show what the LoRAs change.",
  refmods: "References saved by a RefMod job (or made in ComfyUI with ComfyUI-MiniMaxH3Mod), put in after the reference images without encoding anything again. Strength 1 is the reference as it is; lower blends it toward a blurred copy of itself, so it guides the video less. The text encoder doesn't see picture ones, as with the ComfyUI pack's Apply node. Sound ones go in as reference audio (for a video, beside a reference image or a picture RefMod), labelled <Audio 1>, <Audio 2>, … in the prompt as the released pipeline does.",
  audio: "Each goes in as clean audio ahead of the video's own, after the reference images and RefMods, and is labelled <Audio 1>, <Audio 2>, … ahead of the prompt (after any sound RefMods' labels), as the released pipeline presents them. A video file's soundtrack works too. The ComfyUI pack reports that a voice doesn't carry over reliably; music has done better.",
  strengths: "Each checkpoint renders at each of these LoRA strengths, comma-separated: one row of the grid per strength.",
  cfg: "How strongly the image follows the prompt and steers away from the negative prompt. Above 1 runs the model twice a step, so it's about twice as slow; distilled (turbo) models are made for 1.",
  tokenCfg: "How strongly the composer follows the style and lyrics as it writes the song, token by token. 1 turns it off; the model's release uses 1.01.",
  negative: "What to steer away from. It's only used while CFG is above 1.",
  sampler: "How each step moves from noise toward the image. Euler is fast and dependable; Heun is more accurate but runs the model twice a step; DPM++ 2M is often cleaner in fewer steps at Euler's speed.",
  scheduler: "How the noise levels are spaced across the steps. Model default is the model's own schedule; Simple spaces them evenly; Beta puts more steps at the start and the end.",
  shift: "Moves the steps toward the noisy end, where the composition is decided: higher suits bigger images. Blank uses the model's own shift.",
  prompt: "What to make, or for an edit the change to make. Ctrl+Enter generates.",
  style: "The song's genre, instruments, mood and voice, as comma-separated tags, such as: indie pop, upbeat, electric guitar, female vocals, English. Ctrl+Enter generates.",
  lyrics: "Blank makes an instrumental. Section tags like [Intro], [Verse], [Chorus] and [Bridge] lay out the song, each followed by its lines. Score planning is the profile's setting.",
  compareCheckpoints: "Renders the same prompt and seed with each checkpoint of a job at each strength, as a grid. The LoRAs above apply to every image.",
  generation: "How big or long to make it, how many steps, how closely to follow the prompt, and the seed.",
  seconds: "The song's length, up to 6 minutes.",
  count: "Each makes its own, the seeds counting up from the seed.",
  advanced: "The sampler and its schedule, the GPU, and how much of the model to keep in system RAM while generating.",
};

function Num(props: { label: string; v: number; on: (n: number) => void; step?: number; min?: number; max?: number; disabled?: boolean; help?: string; info?: string }) {
  return (
    <label className="field">
      <span>{props.label}{props.info && <Info text={props.info} />}</span>
      <input type="number" value={Number.isFinite(props.v) ? props.v : ""} step={props.step} min={props.min} max={props.max}
        disabled={props.disabled} onChange={(e) => props.on(e.target.value === "" ? NaN : Number(e.target.value))} />
      {props.help && <span className="help">{props.help}</span>}
    </label>
  );
}

/** A checkpoint grid's outputs: per seed, a column per checkpoint and a row per strength, the base model first. */
type Open = (src: string, video: boolean, audio?: { poster?: string; score?: string }) => void;
// a song's poster and score as URLs, for the lightbox's player
const audioOf = (run: SandboxRun, o: SandboxOutput) => o.kind === "audio"
  ? { poster: o.poster ? fileUrl(run, o.poster) : undefined, score: o.score ? fileUrl(run, o.score) : undefined } : undefined;

function GridView({ run, onOpen }: { run: SandboxRun; onOpen: Open }) {
  const g = run.params.grid!;
  const seeds = Array.from({ length: run.params.count }, (_, i) => run.params.seed + i);
  const find = (seed: number, variant: string, c?: number, s?: number) =>
    (run.outputs || []).find((o) => o.seed === seed && o.variant === variant && (variant !== "grid" || (o.checkpoint === c && o.strength === s)));
  const cell = (o: SandboxOutput | undefined, key: string, style?: React.CSSProperties) => (
    <div key={key} className="sbx-cell" style={style}>
      {o ? (
        <img src={fileUrl(run, o.poster || o.file)} alt="" className={o.kind === "audio" ? "audio-thumb" : undefined}
          onClick={() => onOpen(fileUrl(run, o.file), o.kind === "video", audioOf(run, o))} />
      ) : <div className="sbx-wait">…</div>}
    </div>
  );
  const cols = g.checkpoints.length + (g.base ? 1 : 0);
  return (
    <div className="stack" style={{ gap: 18 }}>
      {seeds.map((seed) => (
        <div key={seed}>
          {seeds.length > 1 && <div className="muted small" style={{ marginBottom: 6 }}>Seed {seed}</div>}
          <div className="sbx-grid-wrap">
            <div className="sbx-grid" style={{ gridTemplateColumns: `auto repeat(${cols}, minmax(96px, 1fr))` }}>
              <div />
              {g.base && <div className="sbx-grid-head">Base model</div>}
              {g.checkpoints.map((c) => <div key={c.path} className="sbx-grid-head" title={c.path}>{c.label}</div>)}
              {g.strengths.map((st, si) => [
                <div key={`l${si}`} className="sbx-grid-label">LoRA {st}</div>,
                g.base && si === 0 ? cell(find(seed, "base"), "base", { gridRow: `span ${g.strengths.length}` }) : null,
                ...g.checkpoints.map((_, ci) => cell(find(seed, "grid", ci, si), `${ci}:${si}`)),
              ])}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/** One run's outputs, per seed: with the LoRAs and (comparing) the base model side by side, each under its name and
 * over what made it (``meta``, the run's settings, then its seed and LoRAs). */
function RunView({ run, meta, onOpen }: { run: SandboxRun; meta: string; onOpen: Open }) {
  if (run.params.grid) return <GridView run={run} onOpen={onOpen} />;
  const bySeed = useMemo(() => {
    const m = new Map<number, Partial<Record<"lora" | "base", SandboxOutput>>>();
    for (const o of run.outputs || []) m.set(o.seed, { ...(m.get(o.seed) || {}), [o.variant]: o });
    return [...m.entries()];
  }, [run.outputs]);
  const compare = run.params.compare_base;
  const loras = run.params.loras;
  const song = run.params.seconds != null;
  const panel = (o: SandboxOutput | undefined, variant: "lora" | "base", seed: number) => {
    const label = variant === "base" ? "Base model" : loras.length ? (loras.length > 1 ? "With LoRAs" : "With LoRA") : "Result";
    const withLoras = variant === "lora" && loras.length > 0;
    return (
      <div className="sbx-out" key={variant}>
        <div className="sbx-out-head">
          <strong>{label}</strong>
          {withLoras && <span className="faint ellipsis" title={loras.map((l) => `${base(l.path)} @ ${l.strength}`).join("\n")}>{loras.map((l) => base(l.path)).join(", ")}</span>}
        </div>
        {!o ? <div className={"sbx-wait" + (song ? " song" : "")}>{label}…</div>
          : o.kind === "audio" ? <>
            {o.poster && <img className="sbx-poster" src={fileUrl(run, o.poster)} alt="Spectrogram" onClick={() => onOpen(fileUrl(run, o.file), false, audioOf(run, o))} />}
            <AudioPlayer src={fileUrl(run, o.file)} download={run.n ? `run${run.n}_${o.file}` : o.file} />
          </> : o.kind === "video" ? (
            <video src={fileUrl(run, o.file)} poster={o.poster ? fileUrl(run, o.poster) : undefined} controls loop preload="metadata" />
          ) : (
            <img src={fileUrl(run, o.file)} alt="" onClick={() => onOpen(fileUrl(run, o.file), false)} />
          )}
        <div className="sbx-out-meta">
          <div>{meta}</div>
          <div>Seed {o?.seed ?? seed}{withLoras ? (loras.length === 1 ? ` · LoRA strength ${loras[0].strength}` : ` · ${loras.length} LoRAs`) : variant === "base" ? " · without LoRAs" : ""}</div>
        </div>
      </div>
    );
  };
  return (
    <div className="stack" style={{ gap: 14 }}>
      {bySeed.map(([seed, o]) => (
        <div key={seed} className={"sbx-pair" + (compare ? " two" : "")}>
          {panel(o.lora, "lora", seed)}
          {compare && panel(o.base, "base", seed)}
        </div>
      ))}
    </div>
  );
}

/** Under the progress bar: "Image 2/4 · Step 5/20 · 9.0 s/step" and "0:45 / 3:00" (elapsed / estimated total). */
function StepCounter({ p }: { p: SandboxProgress }) {
  if (!p.steps) return null;
  const rate = p.sec_per_step;
  const parts = [
    (p.images ?? 1) > 1 ? `Image ${p.image}/${p.images}` : "",
    `Step ${p.step ?? 0}/${p.steps}`,
    rate ? (rate >= 1 ? `${rate.toFixed(1)} s/step` : `${(1 / rate).toFixed(1)} steps/s`) : "",
  ].filter(Boolean);
  return (
    <div className="sbx-counter">
      <span>{parts.join(" · ")}</span>
      {p.elapsed != null && (
        <span title={p.eta != null ? "Elapsed / estimated total" : "Elapsed"}>
          {fmtClock(p.elapsed)}{p.eta != null ? ` / ${fmtClock(p.elapsed + p.eta)}` : ""}
        </span>
      )}
    </div>
  );
}

export default function Sandbox({ families, workflows, gpus, ram }: {
  families: Family[]; workflows: Workflow[]; gpus: Gpu[]; ram?: SystemRam;
}) {
  const toast = useToast();
  const [profiles] = usePoll<Profile[]>("/api/profiles", 0);
  const [status, reload] = usePoll<SandboxStatus>("/api/sandbox", 1500);
  const [preset] = useState(readPreset);
  const [form, setFormState] = useState<Form>(() => preset
    ? presetForm(loadForm(), preset, workflows.find((w) => w.id === preset.workflow)?.output === "video") : loadForm());
  const [fromSample, setFromSample] = useState(preset);
  const [defaults, setDefaults] = useState<SandboxDefaults | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [light, setLight] = useState<{ src: string; video: boolean; audio?: { poster?: string; score?: string } } | null>(null);
  const [busy, setBusy] = useState(false);
  const [enhancing, setEnhancing] = useState(false);
  const [imgSize, setImgSize] = useState<{ path: string; width: number; height: number } | null>(null);
  const applied = useRef<string | null>(preset ? preset.profile_id : null); // a preset's profile keeps its settings
  useEffect(() => {
    if (!preset) return;
    localStorage.removeItem(SANDBOX_PRESET);
    saveForm(form);
  }, []);

  const setForm = (patch: Partial<Form>) => setFormState((f) => { const n = { ...f, ...patch }; saveForm(n); return n; });
  const profile = profiles?.find((p) => p.id === form.profile_id);
  const family = families.find((f) => f.id === profile?.family);
  const [loraOptions] = usePoll<LoraOption[]>(family ? `/api/sandbox/loras?family=${family.id}` : null, 0, [family?.id]);
  const [refmodOptions, reloadRefmods] = usePoll<RefModOption[]>(family?.refmods ? `/api/sandbox/refmods?family=${family.id}` : null, 0, [family?.id]);
  // Make RefMod: its dialog's settings (kept between makes), whether it's open, and the make this page asked for
  const [refmodDraft, setRefmodDraft] = useState<RefModDraft>(NEW_REFMOD);
  const [refmodOpen, setRefmodOpen] = useState(false);
  const [makingId, setMakingId] = useState<string | null>(null);
  const handledMake = useRef<string | null>(null);
  const [modelOptions] = usePoll<ModelOption[]>(family?.finetune ? `/api/sandbox/models?family=${family.id}` : null, 0, [family?.id]);

  // pick a profile, then load its defaults; switching profiles applies them (and a workflow it supports)
  useEffect(() => {
    if (profiles && !profile && profiles.length) setForm({ profile_id: byFamily(profiles, families)[0].profiles[0].id });
  }, [profiles]);
  useEffect(() => {
    if (!form.profile_id) return;
    api.get<SandboxDefaults>(`/api/sandbox/defaults?profile_id=${form.profile_id}`).then((d) => {
      setDefaults(d);
      if (applied.current === null && d.workflows.some((w) => w.id === form.workflow)) {
        applied.current = form.profile_id;  // first load: keep the remembered settings
        return;
      }
      if (applied.current !== form.profile_id) {
        applied.current = form.profile_id;
        const keep = d.workflows.some((w) => w.id === form.workflow);
        const workflow = keep ? form.workflow : d.workflows[0]?.id || "t2i";
        setForm({
          steps: d.steps, cfg: d.cfg, negative: d.negative, frames: d.frames, seconds: d.seconds ?? 30, workflow, loras: [], images: "", refmods: [], audio: [],
          model: "", model_custom: false,
          sampler: "euler", scheduler: "default", shift: "", offload: d.offload,
          ...sizeState(d.width, d.height, d.workflows.find((w) => w.id === workflow)?.output === "video"),
        });
      }
    }, (e) => toast(e.message, true));
  }, [form.profile_id]);

  const supported = defaults?.workflows || [];
  const wf = supported.find((w) => w.id === form.workflow);
  const video = wf?.output === "video";
  const music = wf?.output === "audio";
  const strength = family?.lora_strength ?? 1; // the strength a new LoRA row starts at
  const session = status?.session;
  const reserve = status?.mode === "reserve";
  const training = new Set(status?.training_gpus || []);
  const gpu = form.gpu ?? (reserve ? (session?.gpu ?? [...gpus].reverse().find((g) => !training.has(g.index))?.index ?? null) : 0);
  const blocked = reserve ? (gpu != null && training.has(gpu) ? `GPU ${gpu} is training` : null) : status?.blocked;
  const runs = status?.runs || [];
  const current = runs.find((r) => r.id === selected) || runs[0];
  const active = runs.filter((r) => r.status === "queued" || r.status === "running");
  const progress = session?.progress;
  const making = makingId && session?.refmod?.id === makingId ? session.refmod : null;
  const makingNow = making?.state === "queued" || making?.state === "running";
  const offload = form.offload ?? defaults?.offload ?? 0;
  const lowVram = !!form.low_vram && !!family?.streams_text_encoder;
  const [advanced, setAdvanced] = useState(() => {
    try { return localStorage.getItem(ADVANCED) === "1"; } catch { return false; }
  });
  const openAdvanced = (on: boolean) => {
    setAdvanced(on);
    try { localStorage.setItem(ADVANCED, on ? "1" : "0"); } catch { /* not kept: it starts closed next time */ }
  };

  // size: a shape (or the input image's) at some megapixels, or typed in
  const shapes = video ? VIDEO_SHAPES : IMAGE_SHAPES;
  const images = wf?.images ? imageList(form.images) : [];
  const formRefmods = form.refmods || [];
  const setRefmod = (i: number, patch: Partial<LoraRow>) => setForm({ refmods: formRefmods.map((l, j) => (j === i ? { ...l, ...patch } : l)) });
  // the RefMods that go in: chosen, at a strength above 0
  const refmods = wf?.refmods ? formRefmods.filter((l) => l.path.trim() && l.strength > 0) : [];
  // reference sound files: for a video only
  const formAudio = form.audio || [];
  const audioRefs = wf?.audio && video && form.frames > 1 ? formAudio.map((a) => a.trim()).filter(Boolean) : [];
  // the RefMods menu: the Sandbox's, then the jobs'
  const refmodGroups: [string, RefModOption[]][] = [["Made in the Sandbox", (refmodOptions || []).filter((o) => !o.job_id)],
    ["Made by RefMod jobs", (refmodOptions || []).filter((o) => o.job_id)]];
  // the dialog opens on its settings (the reference images, if it has no files yet), or on a make still going
  const openMakeRefmod = () => {
    if (making && !makingNow) setMakingId(null);
    if (!refmodDraft.paths.some((p) => p.trim())) setRefmodDraft({ ...refmodDraft, paths: images.length ? images : [""] });
    setRefmodOpen(true);
  };
  // a RefMod this page asked for, once made: listed, and added to the RefMods if there's room
  useEffect(() => {
    if (!making || handledMake.current === making.id || (making.state !== "done" && making.state !== "failed")) return;
    handledMake.current = making.id;
    if (making.state === "failed" || !making.made) {
      if (!refmodOpen) toast(`Couldn't make the RefMod: ${making.message}`, true);
      return;
    }
    const made = making.made;
    reloadRefmods();
    const rows = formRefmods.filter((l) => l.path.trim());
    const room = !!wf?.refmods && rows.length < 4 && !rows.some((l) => l.path === made.path);
    if (room) setForm({ refmods: [...rows, { path: made.path, strength: 1, custom: false }] });
    if (!refmodOpen) toast(`Made ${made.name}${room ? " and added it to RefMods" : ""}`);
  }, [making?.id, making?.state]); // eslint-disable-line react-hooks/exhaustive-deps
  const mode: SizeMode = form.size_mode === "match" && !wf?.images ? "shape" : form.size_mode;
  useEffect(() => {
    const path = images[0];
    if (mode !== "match" || !path || imgSize?.path === path) return;
    const t = setTimeout(() => {
      api.get<{ width: number; height: number }>(`/api/sandbox/image-size?path=${encodeURIComponent(path)}`)
        .then((d) => setImgSize({ path, ...d }), () => setImgSize(null));
    }, 300);
    return () => clearTimeout(t);
  }, [mode, images[0]]);
  const matched = mode === "match" && imgSize && imgSize.path === images[0] ? imgSize : null;
  const ratio = matched ? matched.width / matched.height : RATIOS[form.shape] ?? 1;
  const [width, height] = mode === "custom" ? [form.width, form.height] : sizeFor(ratio, form.mp);
  const mpOk = mode === "custom" || (form.mp > 0 && Math.max(width, height) <= MAX_SIDE);

  const grid = { ...BLANK.grid, ...form.grid };
  const gridStrengths = parseStrengths(grid.strengths);
  const gridImages = grid.on ? (grid.checkpoints.length * gridStrengths.length + (grid.base ? 1 : 0)) * form.count : 0;
  const vram = useVramProjection(form.profile_id || undefined, {
    width, height, frames: video ? form.frames : 1, ...(music ? { seconds: form.seconds } : {}),
    ...(form.model.trim() ? { model: form.model.trim() } : {}), // a finetuned model is bf16: often bigger than the profile's
    // a grid applies one of its checkpoints at a time, next to the LoRAs
    loras: [...form.loras.map((l) => l.path.trim()), ...(grid.on ? grid.checkpoints.slice(0, 1) : [])].filter(Boolean),
    low_vram: lowVram,
    images: lowVram ? images : [], // low VRAM keeps the VAE in system RAM while sampling, unless images need it
  }, "sample");

  // the enhanced prompt is reused while the prompt, seed, workflow and input images stay the same
  const enhancedFor = (seed: number) => {
    const e = form.enhanced;
    return form.enhance && e && e.prompt === form.prompt.trim() && e.seed === seed && e.workflow === form.workflow
      && e.images === images.join(", ") ? e : null;
  };
  const reusable = form.random_seed ? null : enhancedFor(form.seed);

  const generate = async () => {
    setBusy(true);
    try {
      let prompt = form.prompt;
      let seed = form.random_seed ? -1 : form.seed;
      if (form.enhance && !music) {
        if (seed < 0) seed = Math.floor(Math.random() * 2 ** 31); // the rewrite is made for this seed
        let e = enhancedFor(seed);
        if (!e) {
          setEnhancing(true);
          const r = await api.post<{ prompt: string }>("/api/sandbox/enhance", {
            profile_id: form.profile_id, workflow: form.workflow, prompt: form.prompt, seed, images,
          });
          e = { prompt: form.prompt.trim(), seed, workflow: form.workflow, images: images.join(", "), text: r.prompt };
          setForm({ enhanced: e });
          setEnhancing(false);
        }
        prompt = e.text;
      }
      const body = {
        profile_id: form.profile_id, workflow: form.workflow, prompt, negative: form.negative,
        source_prompt: form.enhance && !music ? form.prompt : "",
        width, height, frames: video ? form.frames : 1, steps: form.steps, cfg: form.cfg,
        ...(music ? { seconds: form.seconds, lyrics: form.lyrics } : {}),
        sampler: form.sampler, scheduler: form.scheduler, shift: form.shift.trim() === "" ? null : Number(form.shift),
        seed, count: form.count, compare_base: form.compare_base && !grid.on,
        loras: form.loras.filter((l) => l.path.trim()).map((l) => ({ path: l.path, strength: l.strength })),
        model: form.model.trim(), images, gpu, offload, low_vram: lowVram,
        ...(wf?.refmods ? { refmods: refmods.map((l) => ({ path: l.path, strength: l.strength })) } : {}),
        ...(wf?.audio ? { audio: audioRefs } : {}),
        grid: grid.on ? { job_id: grid.job_id, checkpoints: grid.checkpoints, strengths: gridStrengths, base: grid.base } : null,
      };
      const run = await api.post<SandboxRun>("/api/sandbox/generate", body);
      setSelected(run.id);
      if (form.random_seed) setForm({ seed: run.params.seed });
      reload();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
      setEnhancing(false);
    }
  };

  const reuse = (r: SandboxRun) => {
    const p: SandboxParams = r.params;
    const src = p.source_prompt;
    applied.current = r.profile_id;
    setForm({
      profile_id: r.profile_id, workflow: p.workflow, prompt: src || p.prompt, negative: p.negative,
      ...sizeState(p.width, p.height, workflows.find((w) => w.id === p.workflow)?.output === "video"),
      frames: p.frames, steps: p.steps, cfg: p.cfg, sampler: p.sampler, scheduler: p.scheduler,
      ...(p.seconds != null ? { seconds: p.seconds, lyrics: p.lyrics || "" } : {}),
      shift: p.shift == null ? "" : String(p.shift), seed: p.seed, random_seed: false, count: p.count,
      compare_base: p.compare_base, images: p.images.join(", "), offload: p.offload ?? null, low_vram: !!p.low_vram,
      loras: p.loras.map((l) => ({ ...l, custom: !(loraOptions || []).some((o) => o.path === l.path) })),
      model: p.model || "", model_custom: !!p.model && !(modelOptions || []).some((o) => o.path === p.model),
      refmods: (p.refmods || []).map((l) => ({ ...l, custom: !(refmodOptions || []).some((o) => o.path === l.path) })),
      audio: p.audio || [],
      // an enhanced run comes back with its rewrite, so generating again reuses it
      enhance: !!src,
      enhanced: src ? { prompt: src.trim(), seed: p.seed, workflow: p.workflow, images: p.images.join(", "), text: p.prompt } : null,
      grid: p.grid ? { on: true, job_id: p.grid.job_id, checkpoints: p.grid.checkpoints.map((c) => c.path), strengths: p.grid.strengths.join(", "), base: p.grid.base }
        : { ...grid, on: false },
    });
    toast("Settings loaded");
  };

  // a run's number: from the server, or (from before runs had one) its place in the list
  const runNumber = (r: SandboxRun) => r.n ?? runs.length - runs.indexOf(r);
  const runMeta = (r: SandboxRun) => {
    const p = r.params;
    const song = p.seconds != null;
    return [
      r.profile_name, p.images.length ? workflows.find((w) => w.id === r.workflow)?.label : "",
      song ? `${p.seconds} s${p.lyrics ? " with lyrics" : ""}` : `${p.width}×${p.height}`, p.frames > 1 ? `${p.frames} frames` : "",
      `${p.steps} steps`, song ? "" : status?.samplers[p.sampler] || p.sampler,
      p.scheduler !== "default" ? status?.schedulers[p.scheduler] : "", p.shift != null ? `shift ${p.shift}` : "",
      p.cfg !== 1 ? `CFG ${p.cfg}` : "",
      p.grid ? `${p.grid.job_name || "checkpoints"}: ${p.grid.checkpoints.length} × ${p.grid.strengths.length} strength${p.grid.strengths.length === 1 ? "" : "s"}` : "",
    ].filter(Boolean).join(" · ");
  };
  const setLora = (i: number, patch: Partial<LoraRow>) => setForm({ loras: form.loras.map((l, j) => (j === i ? { ...l, ...patch } : l)) });
  const jobs = useMemo(() => {
    const m = new Map<string, LoraOption[]>();
    for (const o of loraOptions || []) m.set(o.job_name, [...(m.get(o.job_name) || []), o]);
    return [...m.entries()];
  }, [loraOptions]);
  const modelJobs = useMemo(() => {
    const m = new Map<string, ModelOption[]>();
    for (const o of modelOptions || []) m.set(o.job_name, [...(m.get(o.job_name) || []), o]);
    return [...m.entries()];
  }, [modelOptions]);
  // a job's checkpoints for the grid, oldest first (the final one last)
  const gridJobs = useMemo(() => {
    const m = new Map<string, { name: string; opts: LoraOption[] }>();
    for (const o of loraOptions || []) {
      const e = m.get(o.job_id) || { name: o.job_name, opts: [] };
      e.opts.push(o);
      m.set(o.job_id, e);
    }
    for (const e of m.values()) e.opts.sort((a, b) => Number(a.final) - Number(b.final) || (a.step ?? 0) - (b.step ?? 0));
    return m;
  }, [loraOptions]);
  const setGrid = (patch: Partial<GridForm>) => setForm({ grid: { ...grid, ...patch } });
  const pickGridJob = (id: string) => setGrid({ job_id: id, checkpoints: (gridJobs.get(id)?.opts || []).slice(-12).map((o) => o.path) });
  const fit = offloadFit(vram, offload, gpuTarget(gpus, gpu));
  const gridOk = !grid.on || (grid.checkpoints.length > 0 && gridStrengths.length > 0 && gridImages <= 64);
  const canGo = !!profile && !!wf && form.prompt.trim() !== "" && !blocked && !busy && gridOk && mpOk;

  return (
    <div className="page wide">
      <div className="page-head">
        <div>
          <h1>Sandbox</h1>
          <p>Test your LoRAs and compare results.</p>
        </div>
      </div>

      <div className="sbx">
        <div className="card card-pad stack sbx-controls">
          {fromSample && (
            <div className="sbx-from">
              <div>
                <strong>Step {fromSample.step} sample of {fromSample.job_name}.</strong> Same prompt, seed, size, sampler and LoRA
                as in training; press Generate to re-create it. {fromSample.notes.join(" ")}
              </div>
              <button className="btn ghost icon" aria-label="Dismiss" onClick={() => setFromSample(null)}><IconX /></button>
            </div>
          )}
          <div className="grid cols-2">
            <label className="field">
              <span>Profile<Info text={INFO.profile} /></span>
              <select value={form.profile_id} onChange={(e) => setForm({ profile_id: e.target.value })}>
                <ProfileOptions profiles={profiles || []} families={families} />
              </select>
            </label>
            <label className="field">
              <span>Workflow<Info text={INFO.workflow} /></span>
              <select value={form.workflow} onChange={(e) => setForm({ workflow: e.target.value, images: "" })}>
                {WORKFLOW_GROUPS.map(([kind, label]) => (
                  <optgroup key={kind} label={label}>
                    {workflows.filter((w) => w.output === kind).map((w) => {
                      const ok = supported.some((s) => s.id === w.id);
                      return <option key={w.id} value={w.id} disabled={!ok}>{w.label}{ok ? "" : ` — not for ${family?.label || "this model"}`}</option>;
                    })}
                  </optgroup>
                ))}
              </select>
            </label>
          </div>

          <label className="field">
            <span>{music ? "Style" : "Prompt"}<Info text={music ? INFO.style : INFO.prompt} /></span>
            <textarea rows={music ? 2 : 4} value={form.prompt}
              placeholder={form.workflow === "edit" ? "The edit instruction, e.g. Make it night"
                : music ? "Genre, instruments, mood and voice: indie pop, upbeat, electric guitar, female vocals, English" : "Describe what to generate"}
              onChange={(e) => setForm({ prompt: e.target.value })}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGo) generate(); }} />
          </label>
          {music && (
            <label className="field">
              <span>Lyrics<Info text={INFO.lyrics} /></span>
              <textarea rows={2} value={form.lyrics} placeholder="Leave blank for instrumental music"
                onChange={(e) => setForm({ lyrics: e.target.value })}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGo) generate(); }} />
            </label>
          )}
          {!music && <div className="field sbx-enhance">
            <label className="check">
              <input type="checkbox" checked={form.enhance} onChange={(e) => setForm({ enhance: e.target.checked, enhanced: null })} />
              Enhance prompt with the local LLM<Info text={INFO.enhance} />
            </label>
            {form.enhance && (reusable ? (
              <div className="sbx-enhanced">
                <div className="faint small">Enhanced for seed {reusable.seed}; used again while the prompt and seed stay the same</div>
                <div>{reusable.text}</div>
              </div>
            ) : (
              <span className="help">
                When you press Generate, the LLM set in Settings rewrites the prompt{form.random_seed ? " for each new seed" : ""}.
                {form.enhanced ? " The prompt, seed or input changed since the last rewrite." : ""}
              </span>
            ))}
          </div>}

          {wf?.images && (
            <label className="field" onClick={(e) => e.preventDefault()}>
              <span>{wf.images.label} <span className="faint">· {refmods.length ? `0–${wf.images.max} with RefMods`
                : wf.images.min === wf.images.max ? wf.images.min : `${wf.images.min}–${wf.images.max}`}</span></span>
              <ImagesInput value={form.images} onChange={(v) => setForm({ images: v })} multiple={wf.images.max > 1}
                placeholder={wf.images.max > 1 ? "comma-separated image paths" : "image path"} />
            </label>
          )}

          {wf?.refmods && (
            <div className="field">
              <span>RefMods <span className="faint">· optional</span><Info text={INFO.refmods} /></span>
              <div className="stack" style={{ gap: 8 }}>
                {formRefmods.map((l, i) => (
                  <div key={i} className="sbx-lora">
                    {l.custom || (l.path && refmodOptions && !refmodOptions.some((o) => o.path === l.path)) ? (
                      <PathInput value={l.path} onChange={(v) => setRefmod(i, { path: v })} exts=".safetensors" placeholder="/path/to/refmod.safetensors" />
                    ) : (
                      <select value={l.path} onChange={(e) => e.target.value === "__custom" ? setRefmod(i, { custom: true, path: "" }) : setRefmod(i, { path: e.target.value })}>
                        <option value="">Choose a RefMod…</option>
                        {refmodGroups.map(([label, opts]) => opts.length > 0 && (
                          <optgroup key={label} label={label}>
                            {opts.map((o) => (
                              <option key={o.path} value={o.path}>{o.job_name} · {o.kind === "audio" ? `sound, ${o.seconds} s` : `${o.mode}${o.sheet ? ", sheet" : ""}`} · {o.tokens.toLocaleString()} tokens</option>
                            ))}
                          </optgroup>
                        ))}
                        <option value="__custom">Other file…</option>
                      </select>
                    )}
                    <input type="number" step={0.05} min={0} max={1} value={l.strength} title="Strength (0–1)"
                      onChange={(e) => setRefmod(i, { strength: Math.min(1, Math.max(0, Number(e.target.value) || 0)) })} />
                    <button className="btn ghost icon" title="Remove" onClick={() => setForm({ refmods: formRefmods.filter((_, j) => j !== i) })}><IconX /></button>
                  </div>
                ))}
                <div className="row">
                  <button className="btn" onClick={() => setForm({ refmods: [...formRefmods, { path: "", strength: 1, custom: false }] })} disabled={formRefmods.length >= 4}>
                    <IconPlus /> Add RefMod
                  </button>
                  <button className="btn" onClick={openMakeRefmod}><IconSparkle /> {makingNow ? "Making RefMod…" : "Make RefMod…"}</button>
                  {refmodOptions && refmodOptions.length === 0 && <span className="faint small">None made yet.</span>}
                </div>
              </div>
            </div>
          )}

          {wf?.audio && video && (
            <div className="field">
              <span>{wf.audio.label} <span className="faint">· optional, up to {wf.audio.max} with sound RefMods</span><Info text={`${wf.audio.help} ${INFO.audio}`} /></span>
              <div className="stack" style={{ gap: 8 }}>
                {formAudio.map((a, i) => (
                  <div key={i} className="sbx-lora no-strength">
                    <PathInput value={a} onChange={(v) => setForm({ audio: formAudio.map((x, j) => (j === i ? v : x)) })}
                      exts={AUDIO_EXTS} placeholder="/path/to/voice.wav (or a video, for its sound)" />
                    <button className="btn ghost icon" title="Remove" onClick={() => setForm({ audio: formAudio.filter((_, j) => j !== i) })}><IconX /></button>
                  </div>
                ))}
                <div className="row">
                  <button className="btn" onClick={() => setForm({ audio: [...formAudio, ""] })} disabled={formAudio.length >= wf.audio.max}>
                    <IconPlus /> Add audio
                  </button>
                  {formAudio.length > 0 && form.frames <= 1 && <span className="faint small">Reference audio needs a video: set more than one frame.</span>}
                </div>
              </div>
            </div>
          )}

          {(family?.finetune || form.model) && (
            <div className="field">
              <span>Diffusion model<Info text={INFO.model} /></span>
              {form.model_custom || (form.model && modelOptions && !modelOptions.some((o) => o.path === form.model)) ? (
                <div className="sbx-lora no-strength">
                  <PathInput value={form.model} onChange={(v) => setForm({ model: v })} exts=".safetensors" placeholder="/path/to/finetuned_model.safetensors" />
                  <button className="btn ghost icon" title="Use the profile's" onClick={() => setForm({ model: "", model_custom: false })}><IconX /></button>
                </div>
              ) : (
                <select value={form.model} onChange={(e) => e.target.value === "__custom" ? setForm({ model: "", model_custom: true }) : setForm({ model: e.target.value })}>
                  <option value="">The profile's · {base(String(profile?.data?.diffusion_model || ""))}</option>
                  {modelJobs.map(([name, opts]) => (
                    <optgroup key={name} label={`${name} · finetune`}>
                      {opts.map((o) => <option key={o.path} value={o.path}>{o.final ? "final" : `step ${o.step}`} · {o.file}</option>)}
                    </optgroup>
                  ))}
                  <option value="__custom">Other file…</option>
                </select>
              )}
            </div>
          )}

          <div className="field">
            <span>LoRAs<Info text={INFO.loras} /></span>
            <div className="stack" style={{ gap: 8 }}>
              {form.loras.map((l, i) => (
                <div key={i} className="sbx-lora">
                  {l.custom || (l.path && loraOptions && !loraOptions.some((o) => o.path === l.path)) ? (
                    <PathInput value={l.path} onChange={(v) => setLora(i, { path: v })} exts=".safetensors" placeholder="/path/to/lora.safetensors" />
                  ) : (
                    <select value={l.path} onChange={(e) => e.target.value === "__custom" ? setLora(i, { custom: true, path: "" }) : setLora(i, { path: e.target.value })}>
                      <option value="">Choose a checkpoint…</option>
                      {jobs.map(([name, opts]) => (
                        <optgroup key={name} label={name}>
                          {opts.map((o) => <option key={o.path} value={o.path}>{o.final ? "final" : `step ${o.step}`}{o.recommended ? " ★ recommended" : ""} · {o.file}</option>)}
                        </optgroup>
                      ))}
                      <option value="__custom">Other file…</option>
                    </select>
                  )}
                  <input type="number" step={0.05} value={l.strength} title="Strength" onChange={(e) => setLora(i, { strength: Number(e.target.value) })} />
                  <button className="btn ghost icon" title="Remove" onClick={() => setForm({ loras: form.loras.filter((_, j) => j !== i) })}><IconX /></button>
                </div>
              ))}
              <div className="row">
                <button className="btn" onClick={() => setForm({ loras: [...form.loras, { path: "", strength, custom: false }] })} disabled={form.loras.length >= 8}
                  title={strength !== 1 ? `${family?.label} LoRAs are used at about strength ${strength}` : undefined}>
                  <IconPlus /> Add LoRA
                </button>
              </div>
            </div>
          </div>

          <div className="sbx-checks">
            <label className="check" title={grid.on ? "Compare checkpoints has its own: Include the base model" : undefined}>
              <input type="checkbox" checked={form.compare_base && !grid.on} disabled={!form.loras.length || grid.on} onChange={(e) => setForm({ compare_base: e.target.checked })} />
              Compare with base model<Info text={INFO.compareBase} />
            </label>
            <label className="check">
              <input type="checkbox" checked={grid.on} onChange={(e) => {
                const on = e.target.checked;
                if (on && !grid.job_id && gridJobs.size) {
                  const [id, e0] = [...gridJobs.entries()][0];
                  setGrid({ on, job_id: id, checkpoints: e0.opts.slice(-12).map((o) => o.path) });
                } else setGrid({ on });
              }} />
              Compare checkpoints<Info text={INFO.compareCheckpoints} />
            </label>
          </div>
          {grid.on && (
          <div className="sbx-sub">
            {gridJobs.size === 0 ? (
              <div className="help">No {family?.label || ""} job has checkpoints yet.</div>
            ) : (
              <div className="stack" style={{ gap: 10 }}>
                <label className="field">
                  <span>Job</span>
                  <select value={grid.job_id} onChange={(e) => pickGridJob(e.target.value)}>
                    {!gridJobs.has(grid.job_id) && <option value="">Choose a job…</option>}
                    {[...gridJobs.entries()].map(([id, e]) => <option key={id} value={id}>{e.name} ({e.opts.length})</option>)}
                  </select>
                </label>
                <div className="field">
                  <span>Checkpoints <span className="faint">· {grid.checkpoints.length} of up to 12</span></span>
                  <div className="sbx-ckpts">
                    {(gridJobs.get(grid.job_id)?.opts || []).map((o) => {
                      const on = grid.checkpoints.includes(o.path);
                      return (
                        <button key={o.path} type="button" className={"chip" + (on ? " on" : "")} title={o.file}
                          disabled={!on && grid.checkpoints.length >= 12}
                          onClick={() => setGrid({ checkpoints: on ? grid.checkpoints.filter((p) => p !== o.path) : [...grid.checkpoints, o.path] })}>
                          {o.final ? "final" : `step ${o.step}`}{o.recommended ? " ★" : ""}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="grid cols-2">
                  <label className="field">
                    <span>Strengths<Info text={INFO.strengths} /></span>
                    <input value={grid.strengths} onChange={(e) => setGrid({ strengths: e.target.value })} placeholder="0.6, 0.8, 1.0" />
                  </label>
                  <label className="check" style={{ alignSelf: "end", paddingBottom: 9 }}>
                    <input type="checkbox" checked={grid.base} onChange={(e) => setGrid({ base: e.target.checked })} /> Include the base model
                  </label>
                </div>
                <span className={gridImages > 64 ? "validation error" : "help"}>
                  {gridImages} image{gridImages === 1 ? "" : "s"}{form.count > 1 ? ` (${form.count} seeds)` : ""}{gridImages > 64 ? "; keep a grid to 64" : ""}
                  {progress?.sec_per_step && gridImages ? ` · about ${fmtDuration(gridImages * form.steps * progress.sec_per_step)} at the current speed` : ""}
                </span>
              </div>
            )}
          </div>
          )}

          <div className="sbx-section">
            <h3>Generation<Info text={INFO.generation} /></h3>
            {!music && <div className="field">
              <span>Size</span>
              <div className="seg sbx-shapes" role="group" aria-label="Shape">
                {shapes.map((label) => (
                  <button key={label} type="button" className={mode === "shape" && form.shape === label ? "on" : ""}
                    title={form.mp > 0 ? sizeFor(RATIOS[label], form.mp).join("×") : undefined} onClick={() => setForm({ size_mode: "shape", shape: label })}>{label}</button>
                ))}
                {wf?.images && (
                  <button type="button" className={mode === "match" ? "on" : ""} title="The shape of the (first) input image"
                    onClick={() => setForm({ size_mode: "match" })}>Match input</button>
                )}
                <button type="button" className={mode === "custom" ? "on" : ""} title="Type the width and height"
                  onClick={() => setForm({ size_mode: "custom", width, height })}>Custom</button>
              </div>
              {mode === "custom" ? (
                <div className="sbx-size">
                  <input type="number" aria-label="Width" value={Number.isFinite(form.width) ? form.width : ""} step={defaults?.divisibility || 16} min={128}
                    onChange={(e) => setForm({ width: e.target.value === "" ? NaN : Number(e.target.value) })} />
                  <span className="faint">×</span>
                  <input type="number" aria-label="Height" value={Number.isFinite(form.height) ? form.height : ""} step={defaults?.divisibility || 16} min={128}
                    onChange={(e) => setForm({ height: e.target.value === "" ? NaN : Number(e.target.value) })} />
                  <button type="button" className="btn icon" title="Swap width and height" aria-label="Swap width and height"
                    onClick={() => setForm({ width: form.height, height: form.width })}>⇄</button>
                </div>
              ) : null}
              {mode !== "shape" && (
                <span className="help">
                  {mode === "custom" ? `Sizes are rounded down to multiples of ${defaults?.divisibility || 16}.`
                    : matched ? `Same shape as ${base(matched.path)} (${matched.width}×${matched.height}).`
                      : `Choose an input image above to match its shape; until then this is ${form.shape}.`}
                </span>
              )}
            </div>}
            {!music && (mode !== "custom" || video) && (
              <div className="grid cols-2">
                {mode !== "custom" && (
                  <Num label="Megapixels" info={INFO.megapixels} v={form.mp} on={(n) => setForm({ mp: n })} step={0.05} min={0.05}
                    help={!(form.mp > 0) ? "Above 0, such as 1 or 0.75"
                      : Math.max(width, height) > MAX_SIDE ? `Too big: a side can be at most ${MAX_SIDE} pixels`
                        : `${width} × ${height} · ${((width * height) / MP).toFixed(2)} MP`} />
                )}
                {video && <Num label="Frames" v={form.frames} on={(n) => setForm({ frames: n })} min={1} max={241} help="24 fps" />}
              </div>
            )}
            <div className="grid cols-2">
              {music && <Num label="Duration (seconds)" info={INFO.seconds} v={form.seconds} on={(n) => setForm({ seconds: n })} min={5} max={360} />}
              <Num label="Steps" v={form.steps} on={(n) => setForm({ steps: n })} min={1} max={200} />
              <Num label={music ? "Token CFG (1 = off)" : "CFG (1 = off)"} info={music ? INFO.tokenCfg : INFO.cfg} v={form.cfg} on={(n) => setForm({ cfg: n })} step={music ? 0.01 : 0.5} min={0}
                disabled={defaults ? !defaults.supports_samplers && !music : false} />
              <label className="field">
                <span>Seed</span>
                <div className="sbx-seed">
                  <input type="number" value={form.seed} disabled={form.random_seed} onChange={(e) => setForm({ seed: Number(e.target.value) })} />
                  <button className={"btn icon" + (form.random_seed ? " primary" : "")} title={form.random_seed ? "A random seed each time: on" : "A random seed each time: off"}
                    aria-pressed={form.random_seed} onClick={() => setForm({ random_seed: !form.random_seed })}><IconRefresh /></button>
                </div>
              </label>
              <Num label={music ? "Songs per run" : video ? "Videos per run" : "Images per run"} info={INFO.count} v={form.count} on={(n) => setForm({ count: n })} min={1} max={8} />
            </div>
            {form.cfg > 1 && defaults?.supports_samplers && (
              <label className="field">
                <span>Negative prompt<Info text={INFO.negative} /></span>
                <input value={form.negative} onChange={(e) => setForm({ negative: e.target.value })} />
              </label>
            )}
          </div>

          <div className="sbx-advanced">
            <div className="sbx-adv-head">
              <button type="button" className="sbx-adv-toggle" aria-expanded={advanced} onClick={() => openAdvanced(!advanced)}>
                <IconRight /> Advanced settings
              </button>
              <Info text={INFO.advanced} />
              {!advanced && fit === "over" && <span className="pill failed" title="Open Advanced settings and raise Layer offloading">Won't fit in VRAM</span>}
              {!advanced && fit === "tight" && <span className="pill warn" title="Close to the GPU's memory: see Layer offloading">Tight on VRAM</span>}
            </div>
            {advanced && (
              <div className="stack sbx-adv-body">
                {defaults?.supports_samplers === false ? (
                  <div className="help">{music
                    ? "The composer writes the song token by token (Token CFG guides it; the release uses 1.01 with planning off), then the steps above render it with the release's midpoint sampler; sampler and scheduler don't apply."
                    : "This model uses its released sampler (joint video + audio); sampler and scheduler don't apply."}</div>
                ) : <>
                  <div className="grid cols-2">
                    <label className="field">
                      <span>Sampler<Info text={INFO.sampler} /></span>
                      <select value={form.sampler} onChange={(e) => setForm({ sampler: e.target.value })}>
                        {Object.entries(status?.samplers || {}).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                      </select>
                    </label>
                    <label className="field">
                      <span>Scheduler<Info text={INFO.scheduler} /></span>
                      <select value={form.scheduler} onChange={(e) => setForm({ scheduler: e.target.value })}>
                        {Object.entries(status?.schedulers || {}).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                      </select>
                    </label>
                  </div>
                  <div className="grid cols-2">
                    <label className="field">
                      <span>Shift<Info text={INFO.shift} /></span>
                      <input value={form.shift} placeholder="model's own" onChange={(e) => setForm({ shift: e.target.value })} />
                    </label>
                  </div>
                </>}
                {reserve && (
                  <label className="field">
                    <span>GPU</span>
                    <select value={gpu ?? ""} onChange={(e) => setForm({ gpu: Number(e.target.value) })}>
                      {gpus.map((g) => <option key={g.index} value={g.index} disabled={training.has(g.index)}>GPU {g.index}{training.has(g.index) ? " — training" : ""}</option>)}
                    </select>
                    <span className="help">The Sandbox holds this GPU; training uses the others until you release it.</span>
                  </label>
                )}
                <OffloadControl proj={vram} value={offload} onChange={(n) => setForm({ offload: n })} gpu={gpuTarget(gpus, gpu)} compact ram={ram}
                  lowVram={family?.streams_text_encoder ? { on: lowVram, onChange: (on) => setForm({ low_vram: on }) } : null} />
              </div>
            )}
          </div>

          {blocked && <div className="error-box">{blocked}</div>}
          <div className="sbx-go">
            <button className="btn primary lg" disabled={!canGo} onClick={generate} title="Ctrl+Enter">
              <IconPlayFill /> {enhancing ? "Enhancing prompt…" : `Generate ${music ? "audio" : video ? (form.count > 1 ? "videos" : "video") : form.count > 1 || grid.on ? "images" : "image"}`}
            </button>
            {active.length > 0 && (
              <button className="btn lg" onClick={() => active.forEach((r) => api.post(`/api/sandbox/runs/${r.id}/cancel`).then(reload))}><IconStop /> Cancel{active.length > 1 ? ` ${active.length}` : ""}</button>
            )}
          </div>
        </div>

        <div className="stack sbx-output">
          <div className="card sbx-session">
            <div className="sbx-session-row">
              <span className={`sbx-dot ${session?.state || "off"}`} aria-hidden="true" />
              <strong>{SESSION[session?.state || "off"]}</strong>
              <span className="sbx-session-sep" aria-hidden="true" />
              <span className="muted ellipsis">
                {!session || session.state === "off" ? "Loads when generation starts"
                  : `GPU ${session.gpu}` + (session.state === "ready" ? (status?.idle_minutes ? ` · unloads after ${status.idle_minutes} min idle` : "")
                    : session.message ? ` · ${session.message}` : "")}
              </span>
              <span className="spacer" />
              {session && session.state !== "off" && (
                <button className="btn sm" onClick={() => api.post("/api/sandbox/release").then(reload)} title="Unload the models and free the GPU">Release GPU</button>
              )}
            </div>
            {progress && progress.total > 0 && <>
              <div className="progress" style={{ marginTop: 12 }}><div style={{ width: `${(100 * progress.done) / progress.total}%` }} /></div>
              <StepCounter p={progress} />
            </>}
          </div>

          {current ? (
            <div className="card card-pad">
              <div className="card-head">
                <StatusPill status={current.status === "done" ? "completed" : current.status} />
                <h3>{current === runs[0] ? "Latest result" : `Run ${runNumber(current)}`}</h3>
                {current.params.source_prompt && <span className="pill">enhanced</span>}
                <span className="spacer" />
                {current.elapsed ? <span className="faint small" title="How long it took">{fmtDuration(current.elapsed)}</span> : null}
                <button className="btn" onClick={() => reuse(current)}>Reuse settings</button>
              </div>
              <p className="sbx-prompt">{current.params.source_prompt || current.params.prompt}</p>
              {current.params.source_prompt && (
                <details className="sbx-enhanced-run">
                  <summary className="muted small">Enhanced prompt</summary>
                  <div className="small">{current.params.prompt}</div>
                </details>
              )}
              {current.params.model && (
                <div className="muted small sbx-detail">Model: <span className="mono">{base(current.params.model)}</span></div>
              )}
              {current.params.images.length > 0 && (
                <div className="muted small sbx-detail">Images: <span className="mono">{current.params.images.map(base).join(", ")}</span></div>
              )}
              {(current.params.refmods || []).length > 0 && (
                <div className="muted small sbx-detail">RefMods: <span className="mono">{current.params.refmods!.map((r) => `${base(r.path)}${r.strength !== 1 ? ` @ ${r.strength}` : ""}`).join(", ")}</span></div>
              )}
              {(current.params.audio || []).length > 0 && (
                <div className="muted small sbx-detail">Reference audio: <span className="mono">{current.params.audio!.map(base).join(", ")}</span></div>
              )}
              {current.status === "failed" && <div className="error-box" style={{ marginBottom: 12 }}>{current.message}</div>}
              {current.report?.loras?.some((l) => l.matched < l.total) && (
                <div className="help" style={{ marginBottom: 10 }}>
                  {current.report.loras.filter((l) => l.matched < l.total).map((l) => `${l.label}: ${l.matched} of ${l.total} layers matched this model`).join(" · ")}
                </div>
              )}
              <RunView run={current} meta={runMeta(current)} onOpen={(src, v, audio) => setLight({ src, video: v, audio })} />
            </div>
          ) : (
            <Empty icon={<IconSparkle />} title="No generations yet">Pick a profile, write a prompt and press Generate.</Empty>
          )}

          {runs.length > 0 && (
            <div className="card card-pad">
              <div className="card-head"><h3>History</h3><span className="count-pill">{runs.length}</span></div>
              <div className="sbx-history">
                {runs.map((r) => {
                  const first = r.outputs?.[0];
                  const thumb = first ? fileUrl(r, first.poster || first.file) : null;
                  const p = r.params;
                  const note = r.status !== "done" ? r.status.charAt(0).toUpperCase() + r.status.slice(1)
                    : p.seconds != null ? fmtClock(p.seconds) : p.frames > 1 ? `${p.frames} frames` : `${p.width}×${p.height}`;
                  return (
                    <div key={r.id} className={"sbx-thumb" + (r.id === current?.id ? " on" : "")} role="button" tabIndex={0}
                      aria-pressed={r.id === current?.id} onClick={() => setSelected(r.id)} onKeyDown={(e) => e.key === "Enter" && setSelected(r.id)}
                      title={`${p.source_prompt || p.prompt}\n${fmtTime(r.created)}`}>
                      <div className="sbx-thumb-img">
                        {thumb ? <img src={thumb} alt="" loading="lazy" className={first?.kind === "audio" ? "song" : undefined} /> : <div className="sbx-wait">…</div>}
                        {r.status !== "queued" && r.status !== "running" && (
                          <button className="btn ghost icon danger" title="Delete" onClick={(e) => { e.stopPropagation(); api.del(`/api/sandbox/runs/${r.id}`).then(reload, (er) => toast(er.message, true)); }}><IconTrash /></button>
                        )}
                      </div>
                      <div className="sbx-thumb-name">Run {runNumber(r)}</div>
                      <div className={"sbx-thumb-note" + (r.status === "failed" ? " err" : "")}>{note}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
      {light && <Lightbox src={light.src} video={light.video} audio={light.audio} onClose={() => setLight(null)} />}
      {refmodOpen && (
        <MakeRefModDialog draft={refmodDraft} setDraft={setRefmodDraft} profileId={form.profile_id} gpu={gpu} making={making}
          onMake={setMakingId} onBack={() => setMakingId(null)} onClose={() => setRefmodOpen(false)} />
      )}
    </div>
  );
}

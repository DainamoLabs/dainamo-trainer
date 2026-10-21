import { useEffect, useMemo, useRef, useState } from "react";
import {
  api, fmtClock, fmtDuration, fmtTime, type Family, type Gpu, type LoraOption, type Profile, type SandboxDefaults,
  type SandboxOutput, type SandboxParams, type SandboxProgress, type SandboxRun, type SandboxStatus, type SystemRam,
  type Workflow,
} from "../api";
import { IconPlay, IconPlus, IconRefresh, IconStop, IconTrash, IconX } from "../components/icons";
import { AudioCard, ImagesInput, Lightbox, PathInput, StatusPill, usePoll, useToast } from "../components/ui";
import { OffloadControl, gpuTarget, useVramProjection } from "../components/Vram";

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
  offload: number | null; // % of the diffusion model's blocks in system RAM (null: the model's default)
  low_vram: boolean; // the text encoder's layers streamed from system RAM too
  size_mode: SizeMode; shape: string; mp: number; // width/height are used as typed only in "custom"
  enhance: boolean; enhanced: Enhanced | null;
  grid: GridForm;
};

const STORE = "dainamo.sandbox.form";
// a training sample to re-create, left by "Open in Sandbox" on a job's samples (GET /api/jobs/{id}/sandbox)
export const SANDBOX_PRESET = "dainamo.sandboxPreset";
type Preset = {
  profile_id: string; workflow: string; prompt: string; negative: string; width: number; height: number; frames: number;
  steps: number; cfg: number; seed: number; loras: { path: string; strength: number }[]; images: string[];
  offload: number | null; low_vram?: boolean; notes: string[]; job_name: string; step: number; seconds?: number; lyrics?: string;
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
    low_vram: !!p.low_vram, enhance: false, enhanced: null,
  };
}
const BLANK: Form = {
  profile_id: "", workflow: "t2i", prompt: "", negative: "", width: 1024, height: 1024, frames: 1, seconds: 30, lyrics: "",
  steps: 20, cfg: 1,
  sampler: "euler", scheduler: "default", shift: "", seed: 42, random_seed: true, count: 1, loras: [], images: "",
  compare_base: false, gpu: null, offload: null, low_vram: false, size_mode: "shape", shape: "1:1", mp: 1, enhance: false, enhanced: null,
  grid: { on: false, job_id: "", checkpoints: [], strengths: "1.0", base: false },
};

const parseStrengths = (s: string) => s.replace(/,/g, " ").split(/\s+/).filter(Boolean).map(Number).filter((n) => Number.isFinite(n));

// Sizes by shape and megapixels (a megapixel is 1024×1024 pixels, as ComfyUI counts it), snapped to 64:
// 1 MP at 9:16 is 768×1344.
const MP = 1024 * 1024;
const RATIOS: Record<string, number> = { "1:1": 1, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16, "5:3": 5 / 3, "3:5": 3 / 5 };
const IMAGE_SHAPES = ["1:1", "4:3", "3:4", "16:9", "9:16"];
const VIDEO_SHAPES = ["1:1", "5:3", "3:5", "16:9", "9:16"];
const IMAGE_MP = [0.5, 1, 1.5, 2];
const VIDEO_MP = [0.25, 0.35, 0.5];

export function sizeFor(ratio: number, mp: number): [number, number] {
  const snap = (v: number) => Math.max(128, Math.round(v / 64) * 64);
  return [snap(Math.sqrt(mp * MP * ratio)), snap(Math.sqrt((mp * MP) / ratio))];
}

const nearest = <T,>(xs: T[], cost: (x: T) => number) => xs.reduce((a, b) => (cost(b) < cost(a) ? b : a));

/** The size fields for a width × height: the shape and megapixels that give exactly it, else "custom". */
function sizeState(w: number, h: number, video: boolean): Pick<Form, "size_mode" | "shape" | "mp" | "width" | "height"> {
  const shape = nearest(video ? VIDEO_SHAPES : IMAGE_SHAPES, (s) => Math.abs(Math.log(w / h / RATIOS[s])));
  const mp = nearest(video ? VIDEO_MP : IMAGE_MP, (m) => Math.abs(Math.log((w * h) / MP / m)));
  const [sw, sh] = sizeFor(RATIOS[shape], mp);
  return { size_mode: sw === w && sh === h ? "shape" : "custom", shape, mp, width: w, height: h };
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

function Num(props: { label: string; v: number; on: (n: number) => void; step?: number; min?: number; max?: number; disabled?: boolean; help?: string }) {
  return (
    <label className="field">
      <span>{props.label}</span>
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

/** One run's outputs: LoRA and base side by side per seed. */
function RunView({ run, onOpen }: { run: SandboxRun; onOpen: Open }) {
  if (run.params.grid) return <GridView run={run} onOpen={onOpen} />;
  const bySeed = useMemo(() => {
    const m = new Map<number, Partial<Record<"lora" | "base", SandboxOutput>>>();
    for (const o of run.outputs || []) m.set(o.seed, { ...(m.get(o.seed) || {}), [o.variant]: o });
    return [...m.entries()];
  }, [run.outputs]);
  const compare = run.params.compare_base;
  const media = (o: SandboxOutput | undefined, label: string) => o ? (
    <figure>
      {o.kind === "audio" ? (
        <AudioCard src={fileUrl(run, o.file)} {...audioOf(run, o)} />
      ) : o.kind === "video" ? (
        <video src={fileUrl(run, o.file)} poster={o.poster ? fileUrl(run, o.poster) : undefined} controls loop preload="metadata" />
      ) : (
        <img src={fileUrl(run, o.file)} alt="" onClick={() => onOpen(fileUrl(run, o.file), false)} />
      )}
      <figcaption>{label} · seed {o.seed}</figcaption>
    </figure>
  ) : <figure className="pending"><div className="sbx-wait">{label}…</div></figure>;
  return (
    <div className="stack" style={{ gap: 14 }}>
      {bySeed.map(([seed, o]) => (
        <div key={seed} className={"sbx-pair" + (compare ? " two" : "")}>
          {media(o.lora, run.params.loras.length ? "With LoRA" : "Result")}
          {compare && media(o.base, "Base model")}
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

  // pick a profile, then load its defaults; switching profiles applies them (and a workflow it supports)
  useEffect(() => {
    if (profiles && !profile && profiles.length) setForm({ profile_id: profiles[0].id });
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
          steps: d.steps, cfg: d.cfg, negative: d.negative, frames: d.frames, seconds: d.seconds ?? 30, workflow, loras: [], images: "",
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
  const offload = form.offload ?? defaults?.offload ?? 0;
  const lowVram = !!form.low_vram && !!family?.streams_text_encoder;

  // size: a shape (or the input image's) at some megapixels, or typed in
  const shapes = video ? VIDEO_SHAPES : IMAGE_SHAPES;
  const mps = video ? VIDEO_MP : IMAGE_MP;
  const images = wf?.images ? imageList(form.images) : [];
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

  const grid = { ...BLANK.grid, ...form.grid };
  const gridStrengths = parseStrengths(grid.strengths);
  const gridImages = grid.on ? (grid.checkpoints.length * gridStrengths.length + (grid.base ? 1 : 0)) * form.count : 0;
  const vram = useVramProjection(form.profile_id || undefined, {
    width, height, frames: video ? form.frames : 1, ...(music ? { seconds: form.seconds } : {}),
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
        images, gpu, offload, low_vram: lowVram,
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
      // an enhanced run comes back with its rewrite, so generating again reuses it
      enhance: !!src,
      enhanced: src ? { prompt: src.trim(), seed: p.seed, workflow: p.workflow, images: p.images.join(", "), text: p.prompt } : null,
      grid: p.grid ? { on: true, job_id: p.grid.job_id, checkpoints: p.grid.checkpoints.map((c) => c.path), strengths: p.grid.strengths.join(", "), base: p.grid.base }
        : { ...grid, on: false },
    });
    toast("Settings loaded");
  };

  const setLora = (i: number, patch: Partial<LoraRow>) => setForm({ loras: form.loras.map((l, j) => (j === i ? { ...l, ...patch } : l)) });
  const jobs = useMemo(() => {
    const m = new Map<string, LoraOption[]>();
    for (const o of loraOptions || []) m.set(o.job_name, [...(m.get(o.job_name) || []), o]);
    return [...m.entries()];
  }, [loraOptions]);
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
  const gridOk = !grid.on || (grid.checkpoints.length > 0 && gridStrengths.length > 0 && gridImages <= 64);
  const canGo = !!profile && !!wf && form.prompt.trim() !== "" && !blocked && !busy && gridOk;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Sandbox</h1>
          <p>Try a profile with your LoRAs: same engine and samplers as training, models kept loaded between generations.</p>
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
              <span>Profile</span>
              <select value={form.profile_id} onChange={(e) => setForm({ profile_id: e.target.value })}>
                {(profiles || []).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Workflow</span>
              <select value={form.workflow} onChange={(e) => setForm({ workflow: e.target.value, images: "" })}>
                {workflows.map((w) => {
                  const ok = supported.some((s) => s.id === w.id);
                  return <option key={w.id} value={w.id} disabled={!ok}>{w.label}{ok ? "" : ` — not for ${family?.label || "this model"}`}</option>;
                })}
              </select>
            </label>
          </div>

          <label className="field">
            <span>{music ? "Style" : "Prompt"}</span>
            <textarea rows={music ? 2 : 4} value={form.prompt}
              placeholder={form.workflow === "edit" ? "The edit instruction, e.g. Make it night"
                : music ? "Genre, instruments, mood and voice: indie pop, upbeat, electric guitar, female vocals, English" : "Describe what to generate"}
              onChange={(e) => setForm({ prompt: e.target.value })}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGo) generate(); }} />
          </label>
          {music && (
            <label className="field">
              <span>Lyrics <span className="faint">· blank for an instrumental</span></span>
              <textarea rows={7} value={form.lyrics} placeholder={"[Verse]\nWalking down the street tonight\nNeon signs are shining bright\n\n[Chorus]\n…"}
                onChange={(e) => setForm({ lyrics: e.target.value })}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGo) generate(); }} />
              <span className="help">Section tags like [Intro], [Verse], [Chorus] and [Bridge] lay out the song. Score planning is the profile's setting.</span>
            </label>
          )}
          {!music && <div className="field sbx-enhance">
            <label className="check">
              <input type="checkbox" checked={form.enhance} onChange={(e) => setForm({ enhance: e.target.checked, enhanced: null })} />
              Enhance prompt with the local LLM
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
              <span>{wf.images.label} <span className="faint">· {wf.images.min === wf.images.max ? wf.images.min : `${wf.images.min}–${wf.images.max}`}</span></span>
              <ImagesInput value={form.images} onChange={(v) => setForm({ images: v })} multiple={wf.images.max > 1}
                placeholder={wf.images.max > 1 ? "comma-separated image paths" : "image path"} />
            </label>
          )}

          <div className="field">
            <span>LoRAs</span>
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
                {!grid.on && (
                  <label className="check" title="Also render each seed without the LoRAs">
                    <input type="checkbox" checked={form.compare_base} disabled={!form.loras.length} onChange={(e) => setForm({ compare_base: e.target.checked })} /> Compare with base
                  </label>
                )}
              </div>
            </div>
          </div>

          <div className={"toggle-card" + (grid.on ? " on" : "")}>
            <label className="check toggle-head">
              <input type="checkbox" checked={grid.on} onChange={(e) => {
                const on = e.target.checked;
                if (on && !grid.job_id && gridJobs.size) {
                  const [id, e0] = [...gridJobs.entries()][0];
                  setGrid({ on, job_id: id, checkpoints: e0.opts.slice(-12).map((o) => o.path) });
                } else setGrid({ on });
              }} />
              <span><strong>Compare checkpoints</strong>
                <span className="help">The same prompt and seed with each checkpoint at each strength, as a grid. The LoRAs above apply to every image.</span></span>
            </label>
            {grid.on && (gridJobs.size === 0 ? (
              <div className="help" style={{ marginTop: 10 }}>No {family?.label || ""} job has checkpoints yet.</div>
            ) : (
              <div className="stack" style={{ gap: 10, marginTop: 12 }}>
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
                    <span>Strengths</span>
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
            ))}
          </div>

          {music ? (
            <div className="grid cols-2">
              <Num label="Length (seconds)" v={form.seconds} on={(n) => setForm({ seconds: n })} min={5} max={360} help="Up to 6 minutes" />
            </div>
          ) : <div className="field">
            <span>Size</span>
            <div className="seg sbx-shapes" role="group" aria-label="Shape">
              {shapes.map((label) => (
                <button key={label} type="button" className={mode === "shape" && form.shape === label ? "on" : ""}
                  title={sizeFor(RATIOS[label], form.mp).join("×")} onClick={() => setForm({ size_mode: "shape", shape: label })}>{label}</button>
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
            ) : (
              <div className="sbx-mp">
                <span className="faint small">Megapixels</span>
                <div className="seg" role="group" aria-label="Megapixels">
                  {mps.map((m) => (
                    <button key={m} type="button" className={form.mp === m ? "on" : ""} title={sizeFor(ratio, m).join("×")}
                      onClick={() => setForm({ mp: m })}>{m}</button>
                  ))}
                </div>
                <span className="sbx-dims">{width} × {height} <span className="faint">· {((width * height) / MP).toFixed(2)} MP</span></span>
              </div>
            )}
            <span className="help">
              {mode === "custom" ? `Sizes are rounded down to multiples of ${defaults?.divisibility || 16}.`
                : mode === "match" ? (matched ? `Same shape as ${base(matched.path)} (${matched.width}×${matched.height}).`
                  : `Choose an input image above to match its shape; until then this is ${form.shape}.`)
                : "A megapixel is 1024×1024 pixels; sizes snap to multiples of 64."}
            </span>
          </div>}
          {video && (
            <div className="grid cols-2">
              <Num label="Frames" v={form.frames} on={(n) => setForm({ frames: n })} min={1} max={241} help="24 fps" />
            </div>
          )}

          <div className="grid sbx-steps">
            <Num label="Steps" v={form.steps} on={(n) => setForm({ steps: n })} min={1} max={200} />
            <Num label={music ? "Token CFG (1 = off)" : "CFG (1 = off)"} v={form.cfg} on={(n) => setForm({ cfg: n })} step={music ? 0.01 : 0.5} min={0}
              disabled={defaults ? !defaults.supports_samplers && !music : false} />
            <label className="field">
              <span>Seed</span>
              <div className="row" style={{ gap: 6 }}>
                <input type="number" value={form.seed} disabled={form.random_seed} onChange={(e) => setForm({ seed: Number(e.target.value) })} />
                <button className={"btn icon" + (form.random_seed ? " primary" : "")} title="Random seed each time" onClick={() => setForm({ random_seed: !form.random_seed })}><IconRefresh /></button>
              </div>
            </label>
          </div>
          {form.cfg > 1 && defaults?.supports_samplers && (
            <label className="field">
              <span>Negative prompt</span>
              <input value={form.negative} onChange={(e) => setForm({ negative: e.target.value })} />
            </label>
          )}

          {defaults?.supports_samplers === false ? (
            <div className="help">{music
              ? "The composer writes the song token by token (Token CFG guides it; the release uses 1.01 with planning off), then the steps above render it with the release's midpoint sampler; sampler and scheduler don't apply."
              : "This model uses its released sampler (joint video + audio); sampler and scheduler don't apply."}</div>
          ) : (
            <div className="grid cols-2">
              <label className="field">
                <span>Sampler</span>
                <select value={form.sampler} onChange={(e) => setForm({ sampler: e.target.value })}>
                  {Object.entries(status?.samplers || {}).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Scheduler</span>
                <select value={form.scheduler} onChange={(e) => setForm({ scheduler: e.target.value })}>
                  {Object.entries(status?.schedulers || {}).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                </select>
              </label>
            </div>
          )}

          <div className="grid cols-2">
            {defaults?.supports_samplers !== false && (
              <label className="field">
                <span>Shift</span>
                <input value={form.shift} placeholder="model's own" onChange={(e) => setForm({ shift: e.target.value })} />
              </label>
            )}
            <Num label={music ? "Songs per run" : "Images per run"} v={form.count} on={(n) => setForm({ count: n })} min={1} max={8} help="Seeds count up from the seed" />
          </div>
          <div className="grid cols-2">
            {reserve ? (
              <label className="field">
                <span>GPU</span>
                <select value={gpu ?? ""} onChange={(e) => setForm({ gpu: Number(e.target.value) })}>
                  {gpus.map((g) => <option key={g.index} value={g.index} disabled={training.has(g.index)}>GPU {g.index}{training.has(g.index) ? " — training" : ""}</option>)}
                </select>
                <span className="help">The Sandbox holds this GPU; training uses the others until you release it.</span>
              </label>
            ) : <div />}
          </div>
          <OffloadControl proj={vram} value={offload} onChange={(n) => setForm({ offload: n })} gpu={gpuTarget(gpus, gpu)} compact ram={ram}
            lowVram={family?.streams_text_encoder ? { on: lowVram, onChange: (on) => setForm({ low_vram: on }) } : null} />

          {blocked && <div className="error-box">{blocked}</div>}
          <div className="row">
            <button className="btn primary" disabled={!canGo} onClick={generate}><IconPlay /> {enhancing ? "Enhancing prompt…" : "Generate"}</button>
            {active.length > 0 && (
              <button className="btn" onClick={() => active.forEach((r) => api.post(`/api/sandbox/runs/${r.id}/cancel`).then(reload))}><IconStop /> Cancel{active.length > 1 ? ` ${active.length}` : ""}</button>
            )}
            <div className="spacer" />
            <span className="faint small">Ctrl+Enter</span>
          </div>
        </div>

        <div className="stack sbx-output">
          <div className="card card-pad sbx-session">
            <div className="row">
              <StatusPill status={session?.state === "busy" ? "running" : session?.state === "off" || !session ? "stopped" : session.state === "ready" ? "completed" : "queued"} />
              <span className="muted small">
                {!session || session.state === "off" ? "Nothing loaded" :
                  `GPU ${session.gpu} · ${session.message || session.state}` +
                  (session.state === "ready" && session.idle_seconds != null && status?.idle_minutes ? ` · releases after ${status.idle_minutes} min idle` : "")}
              </span>
              <div className="spacer" />
              {session && session.state !== "off" && (
                <button className="btn sm" onClick={() => api.post("/api/sandbox/release").then(reload)} title="Unload the models and free the GPU">Release GPU</button>
              )}
            </div>
            {progress && progress.total > 0 && <>
              <div className="progress" style={{ marginTop: 10 }}><div style={{ width: `${(100 * progress.done) / progress.total}%` }} /></div>
              <StepCounter p={progress} />
            </>}
          </div>

          {current ? (
            <div className="card card-pad">
              <div className="row" style={{ marginBottom: 12 }}>
                <StatusPill status={current.status === "done" ? "completed" : current.status} />
                <span className="ellipsis" style={{ fontWeight: 600 }} title={current.params.source_prompt || current.params.prompt}>{current.params.source_prompt || current.params.prompt}</span>
                {current.params.source_prompt && <span className="pill">enhanced</span>}
                <div className="spacer" />
                <button className="btn sm" onClick={() => reuse(current)}>Reuse settings</button>
              </div>
              <div className="muted small" style={{ marginBottom: 12 }}>
                {current.profile_name} · {workflows.find((w) => w.id === current.workflow)?.label} · {current.params.seconds != null
                  ? `${current.params.seconds} s${current.params.lyrics ? " · with lyrics" : ""}` : `${current.params.width}×${current.params.height}`}
                {current.params.frames > 1 ? ` · ${current.params.frames}f` : ""} · {current.params.steps} steps · {status?.samplers[current.params.sampler] || current.params.sampler}
                {current.params.scheduler !== "default" ? ` · ${status?.schedulers[current.params.scheduler]}` : ""}
                {current.params.shift != null ? ` · shift ${current.params.shift}` : ""}{current.params.cfg !== 1 ? ` · CFG ${current.params.cfg}` : ""}
                {current.params.loras.map((l) => ` · ${base(l.path)} @ ${l.strength}`).join("")}
                {current.params.grid ? ` · ${current.params.grid.job_name || "checkpoints"}: ${current.params.grid.checkpoints.length} × ${current.params.grid.strengths.length} strength${current.params.grid.strengths.length === 1 ? "" : "s"}` : ""}
                {current.elapsed ? ` · ${fmtDuration(current.elapsed)}` : ""}
              </div>
              {current.params.source_prompt && (
                <details className="sbx-enhanced-run">
                  <summary className="muted small">Enhanced prompt</summary>
                  <div className="small">{current.params.prompt}</div>
                </details>
              )}
              {current.params.images.length > 0 && (
                <div className="muted small" style={{ marginBottom: 12 }}>Images: <span className="mono">{current.params.images.map(base).join(", ")}</span></div>
              )}
              {current.status === "failed" && <div className="error-box">{current.message}</div>}
              {current.report?.loras?.some((l) => l.matched < l.total) && (
                <div className="help" style={{ marginBottom: 10 }}>
                  {current.report.loras.filter((l) => l.matched < l.total).map((l) => `${l.label}: ${l.matched} of ${l.total} layers matched this model`).join(" · ")}
                </div>
              )}
              <RunView run={current} onOpen={(src, v, audio) => setLight({ src, video: v, audio })} />
            </div>
          ) : (
            <div className="empty"><strong>No generations yet</strong>Pick a profile, write a prompt and press Generate.</div>
          )}

          {runs.length > 0 && (
            <div className="card card-pad">
              <div className="row" style={{ marginBottom: 10 }}><strong>History</strong><span className="faint small">{runs.length}</span></div>
              <div className="sbx-history">
                {runs.map((r) => {
                  const first = r.outputs?.[0];
                  const thumb = first ? fileUrl(r, first.poster || first.file) : null;
                  return (
                    <div key={r.id} className={"sbx-thumb" + (r.id === current?.id ? " on" : "")} onClick={() => setSelected(r.id)} title={`${r.params.source_prompt || r.params.prompt}\n${fmtTime(r.created)}`}>
                      {thumb ? <img src={thumb} alt="" loading="lazy" /> : <div className="sbx-wait">{r.status}</div>}
                      {r.status !== "queued" && r.status !== "running" && (
                        <button className="btn ghost icon danger" title="Delete" onClick={(e) => { e.stopPropagation(); api.del(`/api/sandbox/runs/${r.id}`).then(reload, (er) => toast(er.message, true)); }}><IconTrash /></button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
      {light && <Lightbox src={light.src} video={light.video} audio={light.audio} onClose={() => setLight(null)} />}
    </div>
  );
}

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { go } from "../App";
import { api, type Sample } from "../api";
import { SANDBOX_PRESET } from "../pages/Sandbox";
import { Lightbox, useToast } from "./ui";

type View = "grid" | "steps";
type Size = "s" | "m" | "l";
const CELL: Record<Size, number> = { s: 110, m: 170, l: 260 };

function load<T extends string>(key: string, fallback: T, allowed: readonly T[]): T {
  try {
    const v = localStorage.getItem(key) as T | null;
    return v && allowed.includes(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* not remembered in private mode */
  }
}

/** Samples as rows (one per sample prompt card) by columns (one per step, oldest first). */
function useTable(samples: Sample[]) {
  return useMemo(() => {
    const cells = new Map<string, Sample>(); // "index:step" -> the latest sample there
    const rows = new Map<number, Sample>(); // index -> its latest sample (for the label)
    for (const s of samples) {
      cells.set(`${s.index}:${s.step}`, s);
      rows.set(s.index, s);
    }
    const steps = [...new Set(samples.map((s) => s.step))].sort((a, b) => a - b);
    const indexes = [...rows.keys()].sort((a, b) => a - b);
    return { cells, rows, steps, indexes, at: (index: number, step: number) => cells.get(`${index}:${step}`) };
  }, [samples]);
}

const stepLabel = (step: number) => (step === 0 ? "Step 0 · before training" : `Step ${step}`);

/** A sample's settings (its size unless ``size`` is false); the LoRA strength when it isn't 1, and always on a
 * slider's (its cards render at several). */
function meta(s: Sample, slider = false, size = true) {
  if (s.width == null) return "";
  const strength = slider ? ` · slider ${s.lora_scale! > 0 ? "+" : ""}${s.lora_scale}` : s.lora_scale !== 1 ? ` · LoRA ${s.lora_scale}` : "";
  if (s.kind === "audio") return `${s.seconds} s${s.lyrics ? " · with lyrics" : ""} · seed ${s.seed}${strength}`;
  const dims = size ? `${s.width}×${s.height}${s.frames && s.frames > 1 ? ` · ${s.frames}f` : ""} · ` : s.frames && s.frames > 1 ? `${s.frames}f · ` : "";
  return `${dims}seed ${s.seed}${strength}`;
}

/** Load a training sample's settings (and that step's checkpoint) into the Sandbox and go there. */
function useOpenInSandbox(jobId: string) {
  const toast = useToast();
  return async (s: Sample) => {
    try {
      const preset = await api.get(`/api/jobs/${jobId}/sandbox?step=${s.step}&index=${s.index}`);
      localStorage.setItem(SANDBOX_PRESET, JSON.stringify(preset));
      go("sandbox");
    } catch (e: any) {
      toast(e.message, true);
    }
  };
}

/** A sample full size, in the shared lightbox: left and right walk the steps, up and down the prompts. */
function Viewer(props: { url: (f: string) => string; table: ReturnType<typeof useTable>; at: { index: number; step: number };
  onMove: (at: { index: number; step: number }) => void; onClose: () => void; onSandbox: (s: Sample) => void; slider: boolean }) {
  const { table, at } = props;
  const s = table.at(at.index, at.step);
  const si = table.steps.indexOf(at.step);
  const ri = table.indexes.indexOf(at.index);
  const move = (dr: number, dc: number) => {
    // skip cells with no sample (e.g. a prompt added by "Sample now" later)
    for (let r = ri + dr, c = si + dc; r >= 0 && r < table.indexes.length && c >= 0 && c < table.steps.length; r += dr, c += dc) {
      if (table.at(table.indexes[r], table.steps[c])) return props.onMove({ index: table.indexes[r], step: table.steps[c] });
    }
  };
  if (!s) return null;
  return (
    <Lightbox key={s.file} src={props.url(s.file)} video={s.kind === "video"} onClose={props.onClose}
      audio={s.kind === "audio" ? { poster: s.poster ? props.url(s.poster) : undefined, score: s.score ? props.url(s.score) : undefined, title: s.lyrics || undefined } : undefined}
      nav={{ at: si, count: table.steps.length, go: (d) => move(0, d), noun: "step" }}
      navY={{ at: ri, count: table.indexes.length, go: (d) => move(d, 0), noun: "prompt" }}
      info={{ name: stepLabel(s.step), caption: s.prompt, details: [meta(s, props.slider, false)].filter(Boolean),
        actions: s.width != null && (
          <button className="btn sm" title="Re-create this sample in the Sandbox: same prompt, seed, size, sampler and this step's checkpoint"
            onClick={() => props.onSandbox(s)}>Open in Sandbox</button>
        ) }} />
  );
}

/** A job's samples: a prompt-by-step grid (to see a LoRA develop) or grouped by step. */
export function SamplesPanel(props: { jobId: string; samples: Sample[]; slider?: boolean }) {
  const slider = !!props.slider;
  const [view, setView] = useState<View>(() => load("dainamo.samples.view", "grid", ["grid", "steps"] as const));
  const [size, setSize] = useState<Size>(() => load("dainamo.samples.size", "m", ["s", "m", "l"] as const));
  const [open, setOpen] = useState<{ index: number; step: number } | null>(null);
  const table = useTable(props.samples);
  const url = (f: string) => `/api/jobs/${props.jobId}/file/samples/${f}`;
  const openInSandbox = useOpenInSandbox(props.jobId);
  const scroller = useRef<HTMLDivElement>(null);
  const atEnd = useRef(true);

  // keep the newest step in view as columns arrive, unless the grid was scrolled back
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && atEnd.current) el.scrollLeft = el.scrollWidth;
  }, [table.steps.length, view, size]);

  const pick = (v: View) => { setView(v); save("dainamo.samples.view", v); };
  const pickSize = (v: Size) => { setSize(v); save("dainamo.samples.size", v); };
  const thumb = (s: Sample) => (
    <img src={url(s.kind === "video" || s.kind === "audio" ? s.poster || s.file : s.file)} alt={s.prompt} loading="lazy"
      className={s.kind === "audio" ? "audio-thumb" : undefined} title={s.kind === "audio" ? "Play" : undefined}
      style={s.kind === "audio" ? undefined : s.width && s.height ? { aspectRatio: `${s.width} / ${s.height}` } : undefined}
      onClick={() => setOpen({ index: s.index, step: s.step })} />
  );

  return (
    <div className="card card-pad samples-card">
      <div className="row samples-bar">
        <h2 style={{ margin: 0 }}>Samples</h2>
        {table.steps.length > 0 && <span className="faint small">{table.indexes.length} prompt{table.indexes.length === 1 ? "" : "s"} · {table.steps.length} step{table.steps.length === 1 ? "" : "s"}</span>}
        <span className="spacer" />
        {view === "grid" && table.steps.length > 0 && (
          <div className="seg" role="group" aria-label="Thumbnail size">
            {(["s", "m", "l"] as const).map((v) => (
              <button key={v} className={size === v ? "on" : ""} onClick={() => pickSize(v)}>{v.toUpperCase()}</button>
            ))}
          </div>
        )}
        <div className="seg" role="group" aria-label="Layout">
          <button className={view === "grid" ? "on" : ""} onClick={() => pick("grid")}>Grid</button>
          <button className={view === "steps" ? "on" : ""} onClick={() => pick("steps")}>By step</button>
        </div>
      </div>

      {table.steps.length === 0 && <div className="muted small">Samples appear here at the configured interval, or use Sample now while the job runs.</div>}

      {table.steps.length > 0 && view === "grid" && (
        <div className="sample-table-wrap" ref={scroller}
          onScroll={(e) => { const el = e.currentTarget; atEnd.current = el.scrollWidth - el.scrollLeft - el.clientWidth < 8; }}>
          <div className="sample-table" style={{ gridTemplateColumns: `var(--sg-label) repeat(${table.steps.length}, ${CELL[size]}px)` }}>
            <div className="sg-corner" />
            {table.steps.map((st) => <div key={st} className="sg-step">{st === 0 ? "Step 0 (base)" : `Step ${st}`}</div>)}
            {table.indexes.map((index) => {
              const row = table.rows.get(index)!;
              return [
                <div key={`l${index}`} className="sg-label">
                  <div className="sg-prompt" title={row.prompt}>{row.prompt}</div>
                  <div className="faint">{meta(row, slider)}</div>
                </div>,
                ...table.steps.map((st) => {
                  const s = table.at(index, st);
                  return <div key={`${index}:${st}`} className="sg-cell">{s ? thumb(s) : <span className="sg-missing" />}</div>;
                }),
              ];
            })}
          </div>
        </div>
      )}

      {table.steps.length > 0 && view === "steps" && [...table.steps].reverse().map((st) => (
        <div key={st} style={{ marginBottom: 16 }}>
          <div className="muted small" style={{ marginBottom: 6 }}>{stepLabel(st)}</div>
          <div className="samples-grid">
            {table.indexes.map((index) => table.at(index, st)).filter((s): s is Sample => !!s).map((s) => (
              <figure key={s.file}>
                {thumb(s)}
                <figcaption className="ellipsis" title={s.prompt}>{s.prompt}</figcaption>
                {s.width != null && <figcaption className="faint">{meta(s, slider)}</figcaption>}
              </figure>
            ))}
          </div>
        </div>
      ))}

      {open && <Viewer url={url} table={table} at={open} onMove={setOpen} onClose={() => setOpen(null)} onSandbox={openInSandbox} slider={slider} />}
    </div>
  );
}

import { useEffect, useState } from "react";
import { api, type Gpu, type SystemRam, type VramProjection } from "../api";
import { Info } from "./ui";

// Mirrors dainamo/engine/memory.py: blocks streamed at an offload %, and the GB of each phase.
export const streamedFor = (p: VramProjection, pct: number) =>
  Math.max(0, Math.min(p.blocks, Math.floor((pct / 100) * p.blocks + 0.5)));

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

function weightsGb(p: VramProjection, streamed: number) {
  return p.resident_gb + sum(p.block_gb.slice(0, p.blocks - streamed)) + (streamed ? p.inflight_gb : 0);
}

type PhaseTotal = { id: string; label: string; total: number; segments: [string, number][] };

export function phaseTotals(p: VramProjection, streamed: number): PhaseTotal[] {
  return p.phases.map((ph) => {
    const segments: [string, number][] = [...(ph.model ? [["Model weights", weightsGb(p, streamed)] as [string, number]] : []), ...ph.parts];
    const allocated = sum(segments.map(([, v]) => v));
    const overhead = allocated * p.fragmentation + p.context_gb;
    return { id: ph.id, label: ph.label, total: allocated + overhead, segments: [...segments, ["CUDA context + slack", overhead]] };
  });
}

export function peakOf(p: VramProjection, streamed: number): PhaseTotal | null {
  const all = phaseTotals(p, streamed);
  return all.length ? all.reduce((a, b) => (b.total > a.total ? b : a)) : null;
}

/** System RAM of the phase that holds the most (memory.ram_totals): the process, the diffusion model's share
 * and what the phase parks or streams there. */
export function ramPeakOf(p: VramProjection, streamed: number): { label: string; total: number } | null {
  const all = p.phases.map((ph) => {
    const held = ph.model_ram !== undefined ? ph.model_ram : ph.model ? "streamed" : null;
    const model = held === "all" ? p.model_gb : held === "streamed" ? sum(p.block_gb.slice(p.blocks - streamed)) : 0;
    return { label: ph.label, total: (p.process_ram_gb ?? 0) + model + sum((ph.ram ?? []).map(([, v]) => v)) };
  });
  return all.length ? all.reduce((a, b) => (b.total > a.total ? b : a)) : null;
}

/** Debounced projection for a profile and the settings it depends on. */
export function useVramProjection(profileId: string | undefined, settings: Record<string, unknown>, kind: "train" | "sample") {
  const [proj, setProj] = useState<VramProjection | null>(null);
  const key = JSON.stringify(settings);
  useEffect(() => {
    if (!profileId) { setProj(null); return; }
    let live = true;
    const t = setTimeout(() => {
      api.post<VramProjection>("/api/estimate", { profile_id: profileId, settings, kind })
        .then((r) => live && setProj(r))
        .catch((e) => live && setProj({ error: String(e.message || e) } as VramProjection));
    }, 350);
    return () => { live = false; clearTimeout(t); };
  }, [profileId, key, kind]);
  return proj;
}

export type GpuTarget = { label: string; totalGb: number } | null;

/** The GPU a job or generation is compared against: the chosen one, else the smallest. */
export function gpuTarget(gpus: Gpu[] | undefined, chosen: string | number | null | undefined): GpuTarget {
  if (!gpus?.length) return null;
  const pick = chosen === "auto" || chosen == null || chosen === ""
    ? gpus.reduce((a, b) => (b.mem_total < a.mem_total ? b : a))
    : gpus.find((g) => g.index === Number(chosen));
  if (!pick) return null;
  const name = pick.name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "");
  const auto = chosen === "auto" || chosen == null || chosen === "";
  return { label: `GPU ${pick.index} · ${name}${auto && gpus.length > 1 ? " (the smallest)" : ""}`, totalGb: pick.mem_total / 1024 };
}

const MARGIN_GB = 0.75; // headroom "Fit to GPU" leaves for estimate error and other processes
const COLORS = ["var(--accent)", "#c0703a", "#8e6be8", "#3f9fd8", "#3ccf6e", "#d0d0d0"];

/** The "Low VRAM" switch: the text encoder's layers stream from system RAM too, and so do training's saved
 * activations. Off by default. */
export type LowVram = { on: boolean; onChange: (on: boolean) => void };

/** Offload slider (and the Low VRAM switch, for models that have it) with the projected VRAM of each phase
 * against the target GPU. */
export function OffloadControl(props: {
  proj: VramProjection | null; value: number; onChange: (pct: number) => void; gpu: GpuTarget; compact?: boolean;
  lowVram?: LowVram | null; ram?: SystemRam | null;
}) {
  const { proj, value, onChange, gpu, lowVram, ram } = props;
  const valid = proj && !proj.error && proj.blocks > 0;
  const streamed = valid ? streamedFor(proj, value) : 0;
  const phases = valid ? phaseTotals(proj, streamed) : [];
  const peak = valid ? peakOf(proj, streamed) : null;
  const ramGb = valid ? sum(proj.block_gb.slice(proj.blocks - streamed)) : 0;
  const ramPeak = valid ? ramPeakOf(proj, streamed) : null;
  const ramOver = !!ram && !!ramPeak && ramPeak.total > ram.available_gb;
  const total = gpu?.totalGb ?? 0;
  const state = !peak || !total ? "" : peak.total > total ? "over" : peak.total > total - MARGIN_GB ? "tight" : "fits";

  const fit = () => {
    if (!valid || !total) return;
    for (let pct = 0; pct <= 100; pct++) {
      const p = peakOf(proj, streamedFor(proj, pct));
      if (p && p.total <= total - MARGIN_GB) { onChange(pct); return; }
    }
    onChange(100);
  };
  const fullPeak = valid ? peakOf(proj, proj.blocks) : null;
  const fitsAtAll = valid && total ? (fullPeak?.total ?? 0) <= total : true;
  // fully offloaded, only encoding captions (the whole text encoder on the GPU) is still over
  const textBound = !!lowVram && !lowVram.on && fullPeak?.id === "captions";
  // or the training step is, with its saved activations on the GPU
  const stepBound = !!lowVram && !lowVram.on && fullPeak?.id === "training";
  const training = !!proj?.phases?.some((ph) => ph.id === "training");
  const scale = Math.max(total, peak?.total ?? 0) * 1.04 || 1;

  return (
    <div className={"vram" + (props.compact ? " compact" : "")}>
      <div className="vram-head">
        <span className="vram-title">Layer offloading<Info text="Keeps this share of the diffusion model's blocks in system RAM and moves each to the GPU only while it runs: less VRAM, slower steps. Fit to GPU picks the lowest share that fits." /></span>
        <span className="vram-value">
          {value}%{valid && <span className="muted"> · {streamed} of {proj.blocks} blocks in system RAM</span>}
        </span>
      </div>
      <input type="range" min={0} max={100} step={1} value={value} onChange={(e) => onChange(Number(e.target.value))}
        aria-label="Share of the diffusion model's blocks kept in system RAM" />
      {lowVram && (
        <label className="switch-row">
          <span className="switch">
            <input type="checkbox" role="switch" checked={lowVram.on} onChange={(e) => lowVram.onChange(e.target.checked)} /><i />
          </span>
          <span>Low VRAM <span className="muted">· stream the text encoder{training ? " and saved activations" : ""} from system RAM too</span></span>
        </label>
      )}
      {proj?.error && <div className="validation warning">{proj.error}</div>}
      {!proj && <div className="help">Projecting VRAM…</div>}
      {valid && peak && (
        <>
          <div className="vram-bar" title={`${peak.label}: ${peak.total.toFixed(1)} GB`}>
            {peak.segments.filter(([, v]) => v > 0.005).map(([label, v], i) => (
              <span key={label} style={{ width: `${(v / scale) * 100}%`, background: COLORS[i % COLORS.length] }} title={`${label}: ${v.toFixed(2)} GB`} />
            ))}
            {total > 0 && <i className="vram-limit" style={{ left: `${(total / scale) * 100}%` }} title={`${gpu?.label}: ${total.toFixed(1)} GB`} />}
          </div>
          <div className="vram-summary">
            <span className={"vram-peak " + state}>
              ≈ {peak.total.toFixed(1)} GB peak{total ? ` of ${total.toFixed(1)} GB` : ""}
              {state === "over" ? " — won't fit" : state === "tight" ? " — tight" : ""}
            </span>
            {total > 0 && <button className="btn sm" type="button" onClick={fit} title="Lowest offload that fits, with a little headroom">Fit to GPU</button>}
          </div>
          {!props.compact && (
            <div className="vram-legend">
              {peak.segments.filter(([, v]) => v > 0.005).map(([label, v], i) => (
                <span key={label}><i style={{ background: COLORS[i % COLORS.length] }} />{label} {v.toFixed(1)}</span>
              ))}
            </div>
          )}
          <div className="help">
            {gpu ? `Compared with ${gpu.label}. ` : ""}
            {phases.length > 1 && <>By phase: {phases.map((ph) => `${ph.label.toLowerCase()} ${ph.total.toFixed(1)} GB`).join(" · ")}. </>}
            {streamed > 0 && <>Offloaded blocks take {ramGb.toFixed(1)} GB of pinned system RAM and are copied to the GPU again each step, alongside the computation; steps slow down only when the copies take longer than the compute. </>}
            {!fitsAtAll && (textBound
              ? <span className="validation error" style={{ display: "inline" }}>Even fully offloaded this won't fit: encoding the text takes {fullPeak?.total.toFixed(1)} GB with the whole text encoder on the GPU. Turn on Low VRAM to stream it. </span>
              : stepBound
                ? <span className="validation error" style={{ display: "inline" }}>Even fully offloaded this won't fit: a training step takes {fullPeak?.total.toFixed(1)} GB. Turn on Low VRAM to keep its saved activations in system RAM, or lower the resolution, batch size or frame count. </span>
                : <span className="validation error" style={{ display: "inline" }}>Even fully offloaded this won't fit; lower the resolution, batch size or frame count. </span>)}
          </div>
          {ramPeak && (
            <div className="help">
              System RAM ≈ {ramPeak.total.toFixed(1)} GB at most ({ramPeak.label.toLowerCase()})
              {ram ? `, with ${ram.available_gb.toFixed(1)} GB of ${ram.total_gb.toFixed(0)} GB free now` : ""}.
              {ramOver && <span className="validation warning" style={{ display: "inline" }}> That's more than is free: close other programs{streamed > 0 ? " or offload less" : ""}, or the system may start swapping or stop the job.</span>}
            </div>
          )}
          {proj.notes.map((n) => <div key={n} className="help">{n}</div>)}
        </>
      )}
    </div>
  );
}

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

/** Whether the peak fits the GPU at this offload: "over", "tight" (within the margin), "fits", or "" (not known). */
export function offloadFit(proj: VramProjection | null, value: number, gpu: GpuTarget): "over" | "tight" | "fits" | "" {
  const peak = proj && !proj.error && proj.blocks > 0 ? peakOf(proj, streamedFor(proj, value)) : null;
  const total = gpu?.totalGb ?? 0;
  return !peak || !total ? "" : peak.total > total ? "over" : peak.total > total - MARGIN_GB ? "tight" : "fits";
}
const COLORS = ["var(--accent)", "#c0703a", "#8e6be8", "#3f9fd8", "#3ccf6e", "#d0d0d0"];

/** The "Low VRAM" switch: the text encoder's layers stream from system RAM too, and so do training's saved
 * activations. Off by default. */
export type LowVram = { on: boolean; onChange: (on: boolean) => void };

type OffloadProps = {
  proj: VramProjection | null; value: number; onChange: (pct: number) => void; gpu: GpuTarget;
  lowVram?: LowVram | null; ram?: SystemRam | null;
  finetune?: boolean; // offloaded blocks are trained as they stream, and copied back to RAM after
};

/** What the offload controls show: the peak at this offload, against the GPU and system RAM, and ``fit``, which sets
 * the lowest offload that fits with a little headroom. */
function useOffload({ proj, value, onChange, gpu, lowVram, ram }: OffloadProps) {
  const valid = !!proj && !proj.error && proj.blocks > 0;
  const streamed = valid ? streamedFor(proj, value) : 0;
  const peak = valid ? peakOf(proj, streamed) : null;
  const ramPeak = valid ? ramPeakOf(proj, streamed) : null;
  const total = gpu?.totalGb ?? 0;
  const fit = () => {
    if (!valid || !total) return;
    for (let pct = 0; pct <= 100; pct++) {
      const p = peakOf(proj, streamedFor(proj, pct));
      if (p && p.total <= total - MARGIN_GB) { onChange(pct); return; }
    }
    onChange(100);
  };
  const fullPeak = valid ? peakOf(proj, proj.blocks) : null;
  return {
    valid, streamed, peak, ramPeak, total, fit, fullPeak,
    phases: valid ? phaseTotals(proj, streamed) : [],
    ramGb: valid ? sum(proj.block_gb.slice(proj.blocks - streamed)) : 0,
    ramOver: !!ram && !!ramPeak && ramPeak.total > ram.available_gb,
    state: offloadFit(proj, value, gpu),
    fitsAtAll: valid && total ? (fullPeak?.total ?? 0) <= total : true,
    // fully offloaded, only encoding captions (the whole text encoder on the GPU) is still over
    textBound: !!lowVram && !lowVram.on && fullPeak?.id === "captions",
    // or the training step is, with its saved activations on the GPU
    stepBound: !!lowVram && !lowVram.on && fullPeak?.id === "training",
    training: !!proj?.phases?.some((ph) => ph.id === "training"),
  };
}

/** Why it won't fit even fully offloaded, and what would help. */
function FitError({ o, finetune }: { o: ReturnType<typeof useOffload>; finetune?: boolean }) {
  if (o.fitsAtAll) return null;
  const lead = "Even fully offloaded this won't fit";
  return o.textBound
    ? <span className="validation error" style={{ display: "inline" }}>{lead}: encoding the text takes {o.fullPeak?.total.toFixed(1)} GB with the whole text encoder on the GPU. Turn on Low VRAM to stream it. </span>
    : o.stepBound
      ? <span className="validation error" style={{ display: "inline" }}>{lead}: a training step takes {o.fullPeak?.total.toFixed(1)} GB. Turn on Low VRAM to keep its saved activations in system RAM, or lower the resolution, batch size or frame count{finetune ? " (or use Adafactor)" : ""}. </span>
      : <span className="validation error" style={{ display: "inline" }}>{lead}; lower the resolution, batch size or frame count{finetune ? " (or use Adafactor)" : ""}. </span>;
}

/** The peak phase as a bar of what takes the memory, against the GPU, with a legend. */
function PeakBar({ o, gpu, legend = true }: { o: ReturnType<typeof useOffload>; gpu: GpuTarget; legend?: boolean }) {
  if (!o.peak) return null;
  const scale = Math.max(o.total, o.peak.total) * 1.04 || 1;
  const parts = o.peak.segments.filter(([, v]) => v > 0.005);
  return <>
    <div className="vram-bar" title={`${o.peak.label}: ${o.peak.total.toFixed(1)} GB`}>
      {parts.map(([label, v], i) => (
        <span key={label} style={{ width: `${(v / scale) * 100}%`, background: COLORS[i % COLORS.length] }} title={`${label}: ${v.toFixed(2)} GB`} />
      ))}
      {o.total > 0 && <i className="vram-limit" style={{ left: `${(o.total / scale) * 100}%` }} title={`${gpu?.label}: ${o.total.toFixed(1)} GB`} />}
    </div>
    {legend && (
      <div className="vram-legend">
        {parts.map(([label, v], i) => <span key={label}><i style={{ background: COLORS[i % COLORS.length] }} />{label} {v.toFixed(1)}</span>)}
      </div>
    )}
  </>;
}

/** The projection's details: each phase, what offloaded blocks cost, system RAM, and the projection's notes. */
function VramNotes({ o, proj, gpu, ram, finetune, compared = true }: {
  o: ReturnType<typeof useOffload>; proj: VramProjection; gpu: GpuTarget; ram?: SystemRam | null; finetune?: boolean; compared?: boolean;
}) {
  return <>
    <div className="help">
      {compared && gpu ? `Compared with ${gpu.label}. ` : ""}
      {o.phases.length > 1 && <>By phase: {o.phases.map((ph) => `${ph.label.toLowerCase()} ${ph.total.toFixed(1)} GB`).join(" · ")}. </>}
      {o.streamed > 0 && <>Offloaded blocks take {o.ramGb.toFixed(1)} GB of pinned system RAM and are copied to the GPU again each step{finetune ? ", and back once they're updated" : ""}, alongside the computation; steps slow down only when the copies take longer than the compute. </>}
      {compared && <FitError o={o} finetune={finetune} />}
    </div>
    {o.ramPeak && (
      <div className="help">
        System RAM ≈ {o.ramPeak.total.toFixed(1)} GB at most ({o.ramPeak.label.toLowerCase()})
        {ram ? `, with ${ram.available_gb.toFixed(1)} GB of ${ram.total_gb.toFixed(0)} GB free now` : ""}.
        {compared && o.ramOver && <RamWarning streamed={o.streamed} />}
      </div>
    )}
    {proj.notes.map((n) => <div key={n} className="help">{n}</div>)}
  </>;
}

const RamWarning = ({ streamed }: { streamed: number }) => (
  <span className="validation warning" style={{ display: "inline" }}> That's more than is free: close other programs{streamed > 0 ? " or offload less" : ""}, or the system may start swapping or stop the job.</span>
);

const OFFLOAD_INFO = "Keeps this share of the diffusion model's blocks in system RAM and moves each to the GPU only while it runs: less VRAM, slower steps. Fit to GPU picks the lowest share that fits.";

/** Offload slider (and the Low VRAM switch, for models that have it) with the projected VRAM of each phase
 * against the target GPU. */
export function OffloadControl(props: OffloadProps & { compact?: boolean }) {
  const { proj, value, onChange, gpu, lowVram, ram, finetune } = props;
  const o = useOffload(props);
  return (
    <div className={"vram" + (props.compact ? " compact" : "")}>
      <div className="vram-head">
        <span className="vram-title">Layer offloading<Info text={OFFLOAD_INFO} /></span>
        <span className="vram-value">
          {value}%{o.valid && <span className="muted"> · {o.streamed} of {proj!.blocks} blocks in system RAM</span>}
        </span>
      </div>
      <input type="range" min={0} max={100} step={1} value={value} onChange={(e) => onChange(Number(e.target.value))}
        aria-label="Share of the diffusion model's blocks kept in system RAM" />
      {lowVram && (
        <label className="switch-row">
          <span className="switch">
            <input type="checkbox" role="switch" checked={lowVram.on} onChange={(e) => lowVram.onChange(e.target.checked)} /><i />
          </span>
          <span>Low VRAM <span className="muted">· stream the text encoder{o.training ? " and saved activations" : ""} from system RAM too</span></span>
        </label>
      )}
      {proj?.error && <div className="validation warning">{proj.error}</div>}
      {!proj && <div className="help">Projecting VRAM…</div>}
      {o.valid && o.peak && (
        <>
          <PeakBar o={o} gpu={gpu} legend={!props.compact} />
          <div className="vram-summary">
            <span className={"vram-peak " + o.state}>
              ≈ {o.peak.total.toFixed(1)} GB peak{o.total ? ` of ${o.total.toFixed(1)} GB` : ""}
              {o.state === "over" ? " — won't fit" : o.state === "tight" ? " — tight" : ""}
            </span>
            {o.total > 0 && <button className="btn sm" type="button" onClick={o.fit} title="Lowest offload that fits, with a little headroom">Fit to GPU</button>}
          </div>
          <VramNotes o={o} proj={proj!} gpu={gpu} ram={ram} finetune={finetune} />
        </>
      )}
    </div>
  );
}

/** Create Job's memory card: the projected peak against the GPU, the offload slider, Low VRAM and Fit to GPU, with
 * the breakdown folded away (what won't fit, or won't fit in system RAM, stays in view). */
export function MemoryEstimate(props: OffloadProps & { noProfile?: boolean }) {
  const { proj, value, onChange, gpu, lowVram, ram, finetune } = props;
  const o = useOffload(props);
  const pct = o.peak && o.total ? Math.round((o.peak.total / o.total) * 100) : null;
  return (
    <div className="card mem-card">
      <h2 className="side-title">Memory estimate<Info text="The most VRAM the job is projected to use at once, from the model files and the settings, against the GPU it's compared with. The breakdown shows what takes it in each phase." /></h2>
      {props.noProfile ? <div className="help">Choose a profile to see what the job needs.</div>
        : proj?.error ? <div className="validation warning">{proj.error}</div>
        : !proj ? <div className="help">Projecting VRAM…</div>
        : o.peak && <>
          <div className={"mem-big " + o.state}>≈ {o.peak.total.toFixed(1)}<span>{o.total ? ` / ${Number(o.total.toFixed(1))} GB` : " GB"}</span></div>
          <div className="muted small">Estimated peak{gpu ? ` · ${gpu.label}` : ""}{o.state === "over" ? " · won't fit" : o.state === "tight" ? " · tight" : ""}</div>
          {pct != null && (
            <div className="mem-meter">
              <div className={"progress " + o.state}><div style={{ width: `${Math.min(100, pct)}%` }} /></div>
              <span>{pct}%</span>
            </div>
          )}
        </>}
      <div className="mem-offload">
        <div className="mem-row"><span className="vram-title">Layer offloading<Info text={OFFLOAD_INFO} /></span><span className="vram-value">{value}%</span></div>
        <input type="range" min={0} max={100} step={1} value={value} onChange={(e) => onChange(Number(e.target.value))}
          aria-label="Share of the diffusion model's blocks kept in system RAM" />
        {o.valid && <div className="faint small" style={{ textAlign: "right" }}>{o.streamed} / {proj!.blocks} blocks</div>}
      </div>
      {(lowVram || (o.valid && o.total > 0)) && (
        <div className="mem-actions">
          {lowVram ? (
            <label className="mem-low">
              <span className="switch-row">
                <span className="switch">
                  <input type="checkbox" role="switch" checked={lowVram.on} onChange={(e) => lowVram.onChange(e.target.checked)} /><i />
                </span>
                <span>Low VRAM</span>
              </span>
              <span className="muted small">Stream the text encoder{o.training ? " and saved activations" : ""} from system RAM too.</span>
            </label>
          ) : <span />}
          {o.valid && o.total > 0 && <button className="btn outline" type="button" onClick={o.fit} title="Lowest offload that fits, with a little headroom">Fit to GPU</button>}
        </div>
      )}
      {o.valid && !o.fitsAtAll && <div className="help"><FitError o={o} finetune={finetune} /></div>}
      {o.valid && o.ramOver && <div className="help"><RamWarning streamed={o.streamed} /></div>}
      {o.valid && o.peak && (
        <details className="mem-breakdown">
          <summary>Memory breakdown</summary>
          <div className="stack" style={{ gap: 8, marginTop: 10 }}>
            <div className="muted small">{o.peak.label}, the largest phase{gpu ? `, against ${gpu.label}` : ""}:</div>
            <PeakBar o={o} gpu={gpu} />
            <VramNotes o={o} proj={proj!} gpu={gpu} ram={ram} finetune={finetune} compared={false} />
          </div>
        </details>
      )}
    </div>
  );
}

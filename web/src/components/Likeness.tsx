import { useEffect, useMemo, useRef, useState } from "react";
import { api, q, type Dataset, type LikenessCompare, type LikenessPoint, type Duplicates } from "../api";
import { IconTrash } from "./icons";
import { ConfirmDialog, Modal, useToast } from "./ui";

/** Likeness (``engine.similarity``): how much an image looks like a dataset, from the cosine of DINOv2 features.
 * 100% is as alike to the dataset as its own images are to each other; 0% is the base model's render of the same
 * prompt (a training sample's step 0, a Sandbox run's render without LoRAs) or, without one, an unrelated image. */

/** What a sample card or an output carries once scored. */
export type Scored = {
  likeness?: number | null; similarity?: number; floor?: number | null;
  nearest?: { file: string; path?: string; sim: number }; near_copy?: boolean;
};

export const LIKENESS_INFO = "How much each sample looks like the training images, from 0% (no more than the base model's render of the same prompt at step 0) to 100% (as alike to them as they are to each other), measured with DINOv2 image features. It rises as the LoRA learns the concept; a sample flagged as a near copy reproduces one training image, the sign of overfitting. Costs one DINOv2 pass per sample (the model is 88 MB, downloaded once) and, without Sample before training, one render of each card with the base model.";

const DATASET_KEY = "dainamo.likeness.dataset"; // the dataset compared with last
const remembered = () => { try { return localStorage.getItem(DATASET_KEY) || ""; } catch { return ""; } };
const remember = (v: string) => { try { localStorage.setItem(DATASET_KEY, v); } catch { /* not kept */ } };

export const pct = (v: number) => `${Math.round(v)}%`;

/** "Likeness 82% · closest training image cat_03.jpg (0.71)", with the near-copy warning. */
export function likenessText(s: Scored): string {
  if (s.likeness == null) return "";
  const near = s.nearest ? ` · closest training image ${s.nearest.file} (${s.nearest.sim.toFixed(2)})` : "";
  return `Likeness ${pct(s.likeness)}${near}${s.near_copy ? " · near copy" : ""}`;
}

/** A small "82%" on a sample; red with "copy" when it's a near copy of a training image. */
export function LikenessBadge({ s }: { s: Scored }) {
  if (s.likeness == null) return null;
  return (
    <span className={"likeness" + (s.near_copy ? " copy" : "")} title={likenessText(s)}>
      {pct(s.likeness)}{s.near_copy ? " · copy" : ""}
    </span>
  );
}

/** Likeness over training (the mean of the step's samples), with the closest any sample came to one training image
 * on the same 0–100 scale (cosine × 100); the band at the top is where that counts as a copy. */
export function LikenessChart({ points, totalSteps, nearCopy = 0.97 }: { points: LikenessPoint[]; totalSteps: number; nearCopy?: number }) {
  const W = 760, H = 200, L = 46, R = 12, T = 12, B = 26;
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const pts = useMemo(() => [...points].sort((a, b) => a.step - b.step), [points]);
  const xMax = Math.max(totalSteps || 1, pts.at(-1)?.step || 1);
  const x = (s: number) => L + ((W - L - R) * s) / xMax;
  const y = (v: number) => T + (H - T - B) * (1 - Math.min(100, Math.max(0, v)) / 100);
  const path = (f: (p: LikenessPoint) => number) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(f(p)).toFixed(1)}`).join("");
  const onMove = (e: React.MouseEvent) => {
    if (!ref.current || !pts.length) return;
    const r = ref.current.getBoundingClientRect();
    const step = ((e.clientX - r.left) / r.width * W - L) / (W - L - R) * xMax;
    let best = 0;
    for (let i = 0; i < pts.length; i++) if (Math.abs(pts[i].step - step) < Math.abs(pts[best].step - step)) best = i;
    setHover(best);
  };
  if (!pts.length) return <div className="empty" style={{ padding: 30 }}>Likeness appears with the first samples.</div>;
  const hp = hover != null ? pts[hover] : null;
  const xt = Array.from({ length: 5 }, (_, i) => Math.round((xMax * i) / 4));
  return (<>
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <rect x={L} y={y(100)} width={W - L - R} height={y(nearCopy * 100) - y(100)} fill="rgba(240,82,82,0.08)" />
      {[0, 25, 50, 75, 100].map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="#222" />
          <text x={L - 8} y={y(t) + 4} fill="#6b6b6b" fontSize="10.5" textAnchor="end">{t}%</text>
        </g>
      ))}
      {xt.map((s) => <text key={s} x={x(s)} y={H - 8} fill="#6b6b6b" fontSize="10.5" textAnchor="middle">{s}</text>)}
      <path d={path((p) => p.nearest * 100)} fill="none" stroke="rgba(240,82,82,0.7)" strokeWidth="1.5" strokeDasharray="4 3" />
      <path d={path((p) => p.likeness)} fill="none" stroke="#ff8c00" strokeWidth="2" />
      {pts.map((p) => <circle key={p.step} cx={x(p.step)} cy={y(p.likeness)} r="3" fill="#ff8c00" />)}
      {hp && (
        <g>
          <line x1={x(hp.step)} x2={x(hp.step)} y1={T} y2={H - B} stroke="#444" strokeDasharray="3 3" />
          <circle cx={x(hp.step)} cy={y(hp.likeness)} r="4" fill="#ff8c00" stroke="#000" strokeWidth="2" />
          <text x={Math.min(x(hp.step) + 8, W - 260)} y={H - B - 8} fill="#ededed" fontSize="11.5">
            step {hp.step} · likeness {pct(hp.likeness)} · closest training image {hp.nearest.toFixed(2)}{hp.near_copy ? " · near copy" : ""}
          </text>
        </g>
      )}
    </svg>
    <div className="row faint small" style={{ gap: 18, justifyContent: "flex-end", marginTop: 4 }}>
      <span><span style={{ display: "inline-block", width: 18, borderTop: "2px solid #ff8c00", verticalAlign: "middle", marginRight: 6 }} />likeness (mean of the step's samples)</span>
      <span><span style={{ display: "inline-block", width: 18, borderTop: "2px dashed rgba(240,82,82,0.8)", verticalAlign: "middle", marginRight: 6 }} />closest training image (cosine × 100)</span>
    </div>
  </>);
}

/** What to compare: a Sandbox run's output, a job's training sample, or a file on this machine. */
export type CompareTarget = {
  source?: { run_id: string; file: string }; sample?: { job_id: string; file: string }; upload?: string;
  label: string; preview?: string | null; datasetIds?: string[]; // the datasets to offer first (a job's own)
};

/** Compare an image with a dataset: pick the dataset, then see its likeness and the training images it's closest to. */
export function CompareDialog({ target, onClose }: { target: CompareTarget; onClose: () => void }) {
  const toast = useToast();
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [picked, setPicked] = useState(() => target.datasetIds?.[0] || remembered());
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<LikenessCompare | null>(null);
  useEffect(() => {
    api.get<Dataset[]>("/api/datasets").then((ds) => {
      const usable = ds.filter((d) => d.exists && d.images + d.videos > 0);
      setDatasets(usable);
      if (!usable.some((d) => d.id === picked)) setPicked(usable[0]?.id || "");
    }, (e) => toast(e.message, true));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const dataset = datasets?.find((d) => d.id === picked);
  const go = async () => {
    if (!picked) return;
    setBusy(true);
    try {
      const body = { dataset_id: picked, ...(target.source ? { source: target.source } : target.sample ? { sample: target.sample } : { upload: target.upload }) };
      setResult(await api.post<LikenessCompare>("/api/similarity/compare", body));
      remember(picked);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  // the datasets a job trained on first, then the rest
  const ordered = datasets ? [...datasets].sort((a, b) => Number(target.datasetIds?.includes(b.id) ?? false) - Number(target.datasetIds?.includes(a.id) ?? false)) : [];
  const r = result;
  return (
    <Modal title="Compare with a dataset" onClose={onClose} wide footer={<button className="btn ghost" onClick={onClose}>Close</button>}>
      <div className="stack">
        <div className="row wrap" style={{ gap: 14, alignItems: "flex-end" }}>
          {target.preview && <img src={target.preview} alt="" style={{ height: 96, borderRadius: 8, border: "1px solid var(--border)" }} />}
          <div className="stack" style={{ gap: 4, flex: 1, minWidth: 220 }}>
            <div className="ellipsis" title={target.label}>{target.label}</div>
            <label className="field">
              <span>Dataset</span>
              <select value={picked} onChange={(e) => { setPicked(e.target.value); setResult(null); }} disabled={!datasets}>
                {!datasets && <option value="">Loading…</option>}
                {datasets && !datasets.length && <option value="">No dataset with images</option>}
                {ordered.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.images + d.videos} {d.videos ? "files" : "images"}</option>)}
              </select>
            </label>
          </div>
          <button className="btn primary" onClick={go} disabled={!picked || busy}>{busy ? "Measuring…" : r ? "Measure again" : "Compare"}</button>
        </div>
        {busy && <div className="muted small">Embedding the dataset's images the first time takes a moment; after that only this image is measured.</div>}
        {r && dataset && (
          <div className="stack" style={{ gap: 14 }}>
            <div className="row wrap" style={{ gap: 18, alignItems: "baseline" }}>
              <span className={"like-big" + (r.near_copy ? " copy" : "")}>{pct(r.likeness)}</span>
              <span className="muted">likeness to <strong>{r.dataset.name}</strong>
                {r.near_copy ? <> · <span style={{ color: "var(--danger)" }}>a near copy of one training image</span></> : ""}</span>
            </div>
            <div className="muted small" style={{ lineHeight: 1.5 }}>
              Similarity to the dataset's centre {r.similarity.toFixed(2)}; its own {r.dataset.count} images score {r.dataset.self_sim.toFixed(2)} (100%).{" "}
              {r.floor
                ? <>0% is the same seed rendered without LoRAs ({r.floor.similarity.toFixed(2)}): the number says how much of the way from the base model to the dataset this image went.</>
                : <>0% is an unrelated image. Compare an output that was rendered with and without LoRAs (<em>Compare without LoRAs</em>), or a training sample after step 0, and 0% becomes the base model's own render.</>}
            </div>
            <div>
              <div className="muted small" style={{ marginBottom: 6 }}>Closest training images (cosine; at {r.near_copy_at.toFixed(2)} or more the image is a near copy)</div>
              <div className="like-near">
                {r.nearest.map((n) => (
                  <figure key={n.path} style={{ margin: 0 }}>
                    <img src={n.url} alt={n.name} title={n.name} loading="lazy" />
                    <figcaption className="ellipsis" title={n.name}><strong style={n.sim >= r.near_copy_at ? { color: "var(--danger)" } : undefined}>{n.sim.toFixed(2)}</strong> · {n.name}</figcaption>
                  </figure>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}

const THRESHOLDS: [number, string][] = [[0.98, "Copies and re-encodes (0.98)"], [0.95, "Also small crops and brightness changes (0.95)"], [0.9, "Also mirrored copies and close variants (0.90)"]];

/** A dataset's duplicate images: pairs at least as alike as the threshold, each side deletable. */
export function DuplicatesDialog({ dataset, onClose, onChanged }: { dataset: Dataset; onClose: () => void; onChanged: () => void }) {
  const toast = useToast();
  const [threshold, setThreshold] = useState(0.98);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Duplicates | null>(null);
  const [gone, setGone] = useState<Set<string>>(new Set());
  const [confirm, setConfirm] = useState<{ path: string; name: string } | null>(null);
  const find = async (t = threshold) => {
    setBusy(true);
    try {
      setResult(await api.get<Duplicates>(`/api/datasets/${dataset.id}/duplicates?threshold=${t}`));
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => { find(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const remove = async (path: string, name: string) => {
    try {
      await api.del(`/api/media?path=${q(path)}`);
      setGone((g) => new Set(g).add(path));
      toast(`Deleted ${name}`);
      onChanged();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  const pairs = (result?.pairs || []).filter((p) => !gone.has(p.a.path) && !gone.has(p.b.path));
  const side = (f: { path: string; name: string; url: string }) => (
    <figure style={{ margin: 0, minWidth: 0 }}>
      <img src={f.url} alt={f.name} loading="lazy" />
      <figcaption className="row" style={{ gap: 6, marginTop: 4 }}>
        <span className="ellipsis small" title={f.name} style={{ flex: 1 }}>{f.name}</span>
        <button className="btn ghost sm danger" title={`Delete ${f.name} (and its caption)`} onClick={() => setConfirm(f)}><IconTrash /></button>
      </figcaption>
    </figure>
  );
  return (
    <Modal title={`Duplicates in ${dataset.name}`} onClose={onClose} wide footer={<button className="btn ghost" onClick={onClose}>Close</button>}>
      <div className="stack">
        <div className="row wrap" style={{ gap: 12, alignItems: "flex-end" }}>
          <label className="field" style={{ flex: 1, minWidth: 260 }}>
            <span>How alike counts as a duplicate</span>
            <select value={threshold} onChange={(e) => { const t = Number(e.target.value); setThreshold(t); find(t); }} disabled={busy}>
              {THRESHOLDS.map(([t, label]) => <option key={t} value={t}>{label}</option>)}
            </select>
          </label>
          <button className="btn" onClick={() => find()} disabled={busy}>{busy ? "Looking…" : "Look again"}</button>
        </div>
        <div className="muted small">Pairs of images whose DINOv2 features are at least this alike. Two photos of one subject from the same angle, or two renders of one prompt, can score 0.95 too: look before deleting.</div>
        {busy && !result && <div className="muted">Embedding {dataset.images + dataset.videos} files the first time takes a moment…</div>}
        {result && !pairs.length && <div className="muted">No duplicates among {result.count} images at this threshold.</div>}
        {pairs.length > 0 && <div className="muted small">{pairs.length} pair{pairs.length === 1 ? "" : "s"} among {result!.count} images</div>}
        {pairs.map((p) => (
          <div key={p.a.path + p.b.path} className="dup-pair">
            {side(p.a)}
            {side(p.b)}
            <div className="stack" style={{ alignItems: "center", gap: 2 }}>
              <strong>{p.sim.toFixed(3)}</strong>
              <span className="faint small">alike</span>
            </div>
          </div>
        ))}
      </div>
      {confirm && <ConfirmDialog title="Delete this image?" danger confirmLabel="Delete"
        message={<>{confirm.name} and its caption are deleted from the dataset's folder. This can't be undone.</>}
        onConfirm={() => remove(confirm.path, confirm.name)} onClose={() => setConfirm(null)} />}
    </Modal>
  );
}

import { useState } from "react";
import { fmtTime, type Eval, type JobDetail } from "../api";
import { IconDownload } from "./icons";
import { AudioCard } from "./ui";

// display bands; the job's own bleed limit decides what the recommendation skips
const LOW_BLEED = 10;
const DRIFT_BANDS = [25, 60];

type Level = "low" | "moderate" | "high";

function bleedLevel(b: number, limit: number): Level {
  if (b < Math.min(LOW_BLEED, limit)) return "low";
  return b <= limit ? "moderate" : "high";
}

function driftLevel(d: number): Level {
  return d < DRIFT_BANDS[0] ? "low" : d < DRIFT_BANDS[1] ? "moderate" : "high";
}

/** Checkpoint picker and concept bleed results: from training, from a checkpoint evaluation run
 * afterwards (Evaluate checkpoints), or both, with a switch between them. */
export function EvalPanel({ job, onOpen }: { job: JobDetail; onOpen: (src: string) => void }) {
  const [view, setView] = useState<"after" | "training">("after");
  const trained = job.config?.settings || {};
  const st = job.status_detail || ({} as JobDetail["status_detail"]);
  const after = st.evaluation || null;
  const pending = job.config?.evaluate; // a checkpoint evaluation is queued or running
  const inTraining = !!(trained.picker_enabled || trained.bleed_enabled);
  const source = after && (view === "after" || !inTraining) ? "after" : inTraining ? "training" : null;
  if (!source && !pending) return null;
  const cfg = source === "after" ? { ...trained, ...after!.settings } : trained;
  const picker = !!cfg.picker_enabled;
  const bleed = !!cfg.bleed_enabled;
  const result = source === "after" ? after! : { evals: st.evals, recommended: st.recommended, sweep: st.sweep };
  const evals: Eval[] = [...(result.evals || [])].sort((a, b) => a.step - b.step);
  const rec = result.recommended;
  const sweep = result.sweep;
  const limit = Number(cfg.bleed_max ?? 25);
  const available = new Set((st.checkpoints || []).map((c) => c.file));
  const url = (kind: string, f: string) => `/api/jobs/${job.id}/file/${kind}/${f}`;

  const losses = evals.map((e) => e.val_loss).filter((v): v is number => v != null);
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const bestLoss = losses.length ? lo : null;
  const probes = (evals[0]?.probes || []).map((p) => p.prompt);
  const title = !source ? "Checkpoint evaluation" : picker && bleed ? "Checkpoint picker & concept bleed" : picker ? "Checkpoint picker" : "Concept bleed";
  const queued = pending && job.status === "queued";

  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <div className="row wrap" style={{ marginBottom: 12, gap: 10 }}>
        <h2 style={{ margin: 0 }}>{title}</h2>
        <span className="spacer" />
        {after && inTraining && (
          <div className="seg" role="group" aria-label="Which evaluation">
            <button className={source === "after" ? "on" : ""} onClick={() => setView("after")}>After training</button>
            <button className={source === "training" ? "on" : ""} onClick={() => setView("training")}>During training</button>
          </div>
        )}
      </div>
      {queued && <div className="muted small" style={{ marginBottom: 12 }}>A checkpoint evaluation is queued.{after ? " Below are the results of the last one." : ""}</div>}
      {source === "after" && !queued && (
        after!.state === "running" ? <div className="muted small" style={{ marginBottom: 12 }}>Evaluating: {st.message || "starting"}…</div>
          : after!.state === "failed" ? <div className="error-box small" style={{ marginBottom: 12 }}>The checkpoint evaluation failed: {after!.message}</div>
          : after!.state === "stopped" ? <div className="muted small" style={{ marginBottom: 12 }}>The checkpoint evaluation was stopped; below are the checkpoints it got to.</div>
          : <div className="muted small" style={{ marginBottom: 12 }}>Evaluated after training, {fmtTime(after!.finished_at || after!.started_at)}{after!.vram_gb ? ` · peak VRAM ${after!.vram_gb} GB` : ""}.</div>
      )}
      {!source ? null : rec ? (
        <div className="rec-banner">
          <div style={{ minWidth: 0 }}>
            <div><span className="rec-star">★</span> Recommended: <span className="mono">{rec.file}</span> <span className="muted">· step {rec.step}</span></div>
            <div className="muted small">{rec.reason}.</div>
          </div>
          {available.has(rec.file) && <a className="btn sm primary" href={url("output", rec.file)} download><IconDownload /> Download</a>}
        </div>
      ) : (
        <div className="muted small" style={{ marginBottom: 12 }}>A recommendation appears once the first saved checkpoint has been evaluated.</div>
      )}

      {evals.length > 0 && (
        <div className="eval-table">
          <div className="eval-row head">
            <span>Step</span>
            {picker && <span>Validation loss</span>}
            {bleed && <span>Concept bleed</span>}
            {bleed && <span>Drift</span>}
            <span>Checkpoint</span>
          </div>
          {evals.map((e) => {
            const isRec = rec && e.file === rec.file;
            return (
              <div className={"eval-row" + (isRec ? " rec" : "")} key={e.step}>
                <span className="mono">{e.step}{isRec && <span className="rec-star"> ★</span>}</span>
                {picker && (
                  <span className="eval-loss">
                    <span className="mono">{e.val_loss?.toFixed(5) ?? "—"}</span>
                    {e.val_loss != null && (
                      <span className="eval-bar"><span style={{ width: `${hi > lo ? 8 + (92 * (e.val_loss - lo)) / (hi - lo) : 50}%` }}
                        className={e.val_loss === bestLoss ? "best" : ""} /></span>
                    )}
                  </span>
                )}
                {bleed && (
                  <span>
                    {e.bleed != null ? <span className={"pill bleed-" + bleedLevel(e.bleed, limit)}>{e.bleed.toFixed(0)}% · {bleedLevel(e.bleed, limit)}</span> : "—"}
                  </span>
                )}
                {bleed && (
                  <span className={e.drift != null ? "bleed-text bleed-" + driftLevel(e.drift) : ""}>
                    {e.drift != null ? `${e.drift.toFixed(0)}%` : "—"}
                  </span>
                )}
                <span className="ellipsis">
                  {e.file == null ? <span className="faint">base model</span>
                    : available.has(e.file) ? <a href={url("output", e.file)} download className="mono small">{e.file}</a>
                    : <span className="faint small" title="Removed by 'Keep last N saves'">{e.file} (deleted)</span>}
                </span>
              </div>
            );
          })}
        </div>
      )}
      {source && <div className="help" style={{ marginTop: 8 }}>
        {picker && "Lower validation loss is better; if it rises while training loss keeps falling, the LoRA has started to overfit. "}
        {bleed && "Concept bleed: how much images from prompts without your trigger word moved toward your training images. Drift: how much they changed at all versus the base model (100% = as much as a new seed); high drift means the LoRA is altering unrelated images even when your concept isn't leaking."}
      </div>}

      {bleed && probes.length > 0 && (
        <div style={{ marginTop: 18 }}>
          <div className="section-title" style={{ marginBottom: 8 }}>Probe prompts (no trigger word)</div>
          {probes.map((prompt, j) => (
            <div key={j} style={{ marginBottom: 14 }}>
              <div className="small muted ellipsis" style={{ marginBottom: 6 }} title={prompt}>{prompt}</div>
              <div className="probe-strip">
                {evals.map((e) => {
                  const p = e.probes?.[j];
                  if (!p) return null;
                  return (
                    <figure key={e.step}>
                      <img src={url("samples", p.file)} alt="" loading="lazy" onClick={() => onOpen(url("samples", p.file))} />
                      <figcaption>{e.step === 0 && e.file == null ? "base model" : `step ${e.step}`}</figcaption>
                      {!(e.step === 0 && e.file == null) && (
                        <figcaption className="probe-scores">
                          <span className={"bleed-text bleed-" + bleedLevel(p.bleed, limit)}>bleed {p.bleed.toFixed(0)}%</span>
                          {p.drift != null && <span className={"bleed-text bleed-" + driftLevel(p.drift)}>drift {p.drift.toFixed(0)}%</span>}
                        </figcaption>
                      )}
                    </figure>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {sweep && sweep.images.length > 0 && (
        <div style={{ marginTop: 18 }}>
          <div className="section-title" style={{ marginBottom: 8 }}>Strength sweep · step {sweep.step}</div>
          <div className="sweep-grid" style={{ gridTemplateColumns: `repeat(${sweep.strengths.length}, minmax(0, 180px))` }}>
            {sweep.strengths.map((s) => <div key={s} className="small muted" style={{ textAlign: "center" }}>LoRA {s}</div>)}
            {[...sweep.images].sort((a, b) => a.index - b.index || a.strength - b.strength).map((im) => (
              im.kind === "audio" ? (
                <AudioCard key={im.file} src={url("samples", im.file)} poster={im.poster ? url("samples", im.poster) : undefined} />
              ) : im.kind === "video" ? (
                <video key={im.file} src={url("samples", im.file)} poster={im.poster ? url("samples", im.poster) : undefined} controls loop preload="metadata" />
              ) : (
                <img key={im.file} src={url("samples", im.file)} alt={im.prompt} title={im.prompt} loading="lazy" onClick={() => onOpen(url("samples", im.file))} />
              )
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

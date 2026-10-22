import { useEffect, useMemo, useState } from "react";
import type { Tab } from "../App";
import { fmtDuration, type Gpu, type Job } from "../api";
import { FINISHED, PHASES } from "../pages/Queue";
import { Temp } from "./Hardware";
import { IconAlert, IconImages, IconLayers, IconPlus, IconQueue, IconSettings, IconSparkle } from "./icons";
import { smooth, useMetrics } from "./LossChart";

const PAGES: { id: Tab; label: string; icon: React.FC<React.SVGProps<SVGSVGElement>> }[] = [
  { id: "queue", label: "Queue", icon: IconQueue },
  { id: "sandbox", label: "Sandbox", icon: IconSparkle },
  { id: "datasets", label: "Datasets", icon: IconImages },
  { id: "profiles", label: "Profiles", icon: IconLayers },
];

// the failed job whose warning was put away by opening its page
const SEEN_KEY = "dainamo.failed-seen";

// editing a job is part of the Queue
const pageOf = (tab: Tab) => (tab === "edit" ? "queue" : tab);

function ago(ts: number) {
  const s = Date.now() / 1000 - ts;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 2 * 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}

/** What a running job is doing (unless training, which its percentage says), its step of how many and the time left
 * (while training), and how far along it is (while training or extracting, the phases that count steps). */
function progressOf(j: Job) {
  const p = j.progress;
  const stopping = j.status === "stopping";
  const phase = stopping ? "Stopping" : j.evaluating ? "Evaluating checkpoints" : PHASES[p.phase || ""] || p.phase || "Starting";
  const counted = !j.evaluating && (p.phase === "training" || p.phase === "extracting") && !!p.total_steps;
  const pct = counted ? Math.min(100, (100 * (p.step || 0)) / p.total_steps!) : null;
  const left = !stopping && p.phase === "training" && p.eta_seconds != null ? fmtDuration(p.eta_seconds) : null;
  const steps = counted ? `${(p.step || 0).toLocaleString()} / ${p.total_steps!.toLocaleString()}` : null;
  const training = !stopping && !j.evaluating && p.phase === "training";
  return { pct, left, meta: [!training && phase, steps, left && `${left} left`].filter(Boolean).join(" · ") };
}

/** The job's loss so far, smoothed, as a small line. */
function Spark({ id }: { id: string }) {
  const points = useMetrics(id, 10000);
  const d = useMemo(() => {
    const ps = smooth(points.filter((p) => isFinite(p.loss)), 0.95);
    if (ps.length < 10) return "";
    const n = 32;
    const ys = Array.from({ length: n }, (_, i) => ps[Math.round(((ps.length - 1) * i) / (n - 1))].loss);
    const lo = Math.min(...ys), hi = Math.max(...ys);
    return ys.map((v, i) => `${i ? "L" : "M"}${((64 * i) / (n - 1)).toFixed(1)},${(2 + 16 * (1 - (v - lo) / (hi - lo || 1))).toFixed(1)}`).join("");
  }, [points]);
  if (!d) return null;
  return <svg className="tb-spark" viewBox="0 0 64 20" aria-hidden="true"><path d={d} /></svg>;
}

function Ring({ pct }: { pct: number }) {
  const c = 2 * Math.PI * 10;
  return (
    <svg className="tb-ring-pct" viewBox="0 0 26 26" aria-hidden="true">
      <circle cx="13" cy="13" r="10" />
      <circle cx="13" cy="13" r="10" strokeDasharray={`${(c * pct) / 100} ${c}`} transform="rotate(-90 13 13)" />
    </svg>
  );
}

/** The pages on the left; on the right, what the trainer is doing (the running job, or the last one if it failed, or
 * idle), each GPU's memory and temperature, Settings and New job. On a phone the pages move to ``TabBar``. */
export function TopBar({ tab, arg, jobs, gpus, waiting }: { tab: Tab; arg: string; jobs: Job[] | null; gpus: Gpu[]; waiting: number }) {
  const page = pageOf(tab);
  const running = (jobs || []).filter((j) => j.status === "running" || j.status === "stopping").sort((a, b) => (a.started || 0) - (b.started || 0));
  const last = (jobs || []).filter((j) => FINISHED.includes(j.status) && j.finished).sort((a, b) => b.finished! - a.finished!)[0];
  const broke = last && (last.status === "failed" || last.status === "interrupted") ? last : null;
  const [seen, setSeen] = useState(() => {
    try { return localStorage.getItem(SEEN_KEY) || ""; } catch { return ""; }
  });
  useEffect(() => {
    if (!broke || page !== "queue" || arg !== broke.id || seen === broke.id) return;
    setSeen(broke.id);
    try { localStorage.setItem(SEEN_KEY, broke.id); } catch { /* not kept: it shows again after a reload */ }
  }, [broke?.id, page, arg, seen]);
  const job = running[0];
  const prog = job && progressOf(job);
  const failed = !job && broke && seen !== broke.id ? broke : null;
  const failedAt = failed && `${failed.status === "failed" ? "Failed" : "Interrupted"}${failed.progress.step ? ` at step ${failed.progress.step.toLocaleString()}` : ""}`;

  return (
    <header className="topbar">
      <a className="tb-brand" href="#/queue" aria-label="Dainamo Trainer: the Queue">
        <img src="/logo.svg" alt="" />
        <span className="tb-word">Dainamo<span>Trainer</span></span>
      </a>
      <span className="tb-sep" aria-hidden="true" />
      <nav className="tb-nav" aria-label="Pages">
        {PAGES.map((p) => (
          <a key={p.id} href={`#/${p.id}`} className={page === p.id ? "active" : ""} aria-current={page === p.id ? "page" : undefined}>
            {p.label}
            {p.id === "queue" && waiting > 0 && <span className="tb-count" title={`${waiting} waiting`}>{waiting}</span>}
          </a>
        ))}
      </nav>

      <div className="tb-right">
        <div className="tb-activity">
          {job && prog && (
            <a className="tb-chip tb-job" href={`#/queue/${encodeURIComponent(job.id)}`} title={`${job.name} · ${prog.meta}`}>
              <span className="tb-live" aria-hidden="true" />
              <span className="tb-text"><span className="tb-name">{job.name}</span><span className="tb-meta">{prog.meta}</span></span>
              {job.progress.phase === "training" && !job.evaluating && <Spark id={job.id} />}
              {prog.pct != null && <span className="tb-pct">{Math.floor(prog.pct)}%</span>}
              {prog.pct != null && <span className="tb-line" aria-hidden="true"><span style={{ width: `${prog.pct}%` }} /></span>}
            </a>
          )}
          {running.length > 1 && <a className="tb-chip tb-more" href="#/queue"><b>+{running.length - 1}</b> running</a>}
          {failed && (
            <a className="tb-chip err" href={`#/queue/${encodeURIComponent(failed.id)}/log`} title={failed.progress.message || undefined}>
              <IconAlert />
              <span className="tb-text"><span className="tb-name">{failed.name}</span><span className="tb-meta">{failedAt}</span></span>
              <span className="tb-err-link">View log</span>
            </a>
          )}
          {!job && !failed && (
            <div className="tb-chip">
              <span className="tb-idle" aria-hidden="true" />
              <span className="tb-text">
                <span className="tb-name">Idle</span>
                <span className="tb-meta">{waiting ? `${waiting} waiting` : last?.finished ? `Last job finished ${ago(last.finished)}` : "Nothing queued"}</span>
              </span>
            </div>
          )}
          {gpus.length > 0 && (
            <div className="tb-gpus">
              {gpus.map((g) => {
                const pct = g.mem_total ? (100 * g.mem_used) / g.mem_total : 0;
                return (
                  <div className="tb-gpu" key={g.index}
                    title={`GPU ${g.index} · ${g.name.replace(/^NVIDIA (GeForce )?/, "")} · ${g.util}% busy · ${(g.mem_used / 1024).toFixed(1)} of ${(g.mem_total / 1024).toFixed(0)} GB`}>
                    <span className="tb-gpu-top"><span>GPU {g.index}</span><span className="tb-temp"><Temp c={g.temp} /></span></span>
                    <span className="tb-gpu-mem">
                      <span className={`tb-meter${pct > 10 ? " on" : ""}`}><span style={{ width: `${pct}%` }} /></span>
                      {(g.mem_used / 1024).toFixed(1)} GB
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {job && prog ? (
          <a className="tb-mini" href={`#/queue/${encodeURIComponent(job.id)}`} aria-label={`${job.name}: ${prog.meta}`}>
            {prog.pct != null ? <Ring pct={prog.pct} /> : <span className="tb-live" />}
            {prog.pct != null && <span className="tb-pct">{Math.floor(prog.pct)}%</span>}
            {prog.left && <span className="tb-meta">{prog.left}</span>}
            {running.length > 1 && <span className="tb-meta">+{running.length - 1}</span>}
          </a>
        ) : failed ? (
          <a className="tb-mini err" href={`#/queue/${encodeURIComponent(failed.id)}/log`} aria-label={`${failed.name}: ${failedAt}`}>
            <IconAlert /> {failed.status === "failed" ? "Failed" : "Interrupted"}
          </a>
        ) : null}

        <span className="tb-sep" aria-hidden="true" />
        <a className={`tb-icon${tab === "settings" ? " active" : ""}`} href="#/settings" aria-label="Settings" title="Settings"
          aria-current={tab === "settings" ? "page" : undefined}><IconSettings /></a>
        <a className="btn primary tb-new" href="#/create" aria-current={tab === "create" ? "page" : undefined}><IconPlus /><span className="tb-new-label">New job</span></a>
      </div>
    </header>
  );
}

/** On a phone, the pages and New job, along the bottom. */
export function TabBar({ tab, waiting }: { tab: Tab; waiting: number }) {
  const page = pageOf(tab);
  const link = (p: (typeof PAGES)[number]) => (
    <a key={p.id} href={`#/${p.id}`} className={page === p.id ? "active" : ""} aria-current={page === p.id ? "page" : undefined}>
      <span className="tb-tab-icon"><p.icon />{p.id === "queue" && waiting > 0 && <span className="tb-count">{waiting}</span>}</span>
      {p.label}
    </a>
  );
  return (
    <nav className="tb-tabs" aria-label="Pages">
      {PAGES.slice(0, 2).map(link)}
      <a href="#/create" className={`tb-tab-new${tab === "create" ? " active" : ""}`} aria-current={tab === "create" ? "page" : undefined}>
        <span className="tb-tab-icon"><IconPlus /></span>New job
      </a>
      {PAGES.slice(2).map(link)}
    </nav>
  );
}

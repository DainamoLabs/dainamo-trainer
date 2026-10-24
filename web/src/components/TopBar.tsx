import { useEffect, useState } from "react";
import type { Tab } from "../App";
import { fmtLeft, type BackgroundTask, type Gpu, type Job, type SystemRam } from "../api";
import { FINISHED, PHASES } from "../pages/Queue";
import { Temp } from "./Hardware";
import { IconAlert, IconDownload, IconImages, IconLayers, IconPlus, IconQueue, IconScissors, IconSettings, IconSparkle, IconUpscale } from "./icons";
import { Spark } from "./LossChart";

const PAGES: { id: Tab; label: string; icon: React.FC<React.SVGProps<SVGSVGElement>> }[] = [
  { id: "queue", label: "Queue", icon: IconQueue },
  { id: "sandbox", label: "Sandbox", icon: IconSparkle },
  { id: "datasets", label: "Datasets", icon: IconImages },
  { id: "profiles", label: "Models", icon: IconLayers },
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
 * (while training), and how far along it is (while training, extracting or merging, the phases that count steps). */
function progressOf(j: Job) {
  const p = j.progress;
  const stopping = j.status === "stopping";
  const phase = stopping ? "Stopping" : j.evaluating ? "Evaluating checkpoints" : PHASES[p.phase || ""] || p.phase || "Starting";
  const counted = !j.evaluating && ["training", "extracting", "merging"].includes(p.phase || "") && !!p.total_steps;
  const pct = counted ? Math.min(100, (100 * (p.step || 0)) / p.total_steps!) : null;
  const left = !stopping && p.phase === "training" ? fmtLeft(p.eta_seconds) : "";
  const steps = counted ? `${(p.step || 0).toLocaleString()} / ${p.total_steps!.toLocaleString()}` : null;
  const training = !stopping && !j.evaluating && p.phase === "training";
  return { pct, left, meta: [!training && phase, steps, left].filter(Boolean).join(" · ") };
}

const TASK_ICON: Record<BackgroundTask["kind"], React.ReactNode> = {
  caption: <IconSparkle />, split: <IconScissors />, download: <IconDownload />, upscale: <IconUpscale />,
};
const taskPct = (t: BackgroundTask) => (t.total ? Math.min(100, (100 * t.done) / t.total) : null);

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
 * idle), a dataset task going (captioning, a split, a download, an upscale: its page isn't the only place it shows),
 * each GPU's memory and temperature, Settings and New job. On a phone the pages move to ``TabBar``. */
export function TopBar({ tab, arg, jobs, gpus, ram, waiting, tasks }: {
  tab: Tab; arg: string; jobs: Job[] | null; gpus: Gpu[]; ram?: SystemRam; waiting: number; tasks: BackgroundTask[];
}) {
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
  const task = tasks[0];
  const taskMeta = task && [task.dataset, task.total ? `${task.done.toLocaleString()} / ${task.total.toLocaleString()}` : ""].filter(Boolean).join(" · ");

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
              {job.progress.phase === "training" && !job.evaluating && <Spark values={job.loss_curve} className="tb-spark" />}
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
          {task && (
            <a className="tb-chip tb-task" href={`#/datasets/${encodeURIComponent(task.dataset_id)}`} title={`${task.label} · ${taskMeta}`}>
              <span className="tb-task-icon" aria-hidden="true">{TASK_ICON[task.kind]}</span>
              <span className="tb-text"><span className="tb-name">{task.label}</span><span className="tb-meta">{taskMeta}</span></span>
              {tasks.length > 1 && <span className="tb-meta">+{tasks.length - 1}</span>}
              {taskPct(task) != null && <span className="tb-line" aria-hidden="true"><span style={{ width: `${taskPct(task)}%` }} /></span>}
            </a>
          )}
          {!job && !failed && !task && (
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
              {ram && ram.total_gb > 0 && (  // system RAM: what offloading, Low VRAM and the Sandbox's parked models live in
                <div className="tb-gpu tb-ram"
                  title={`System RAM · ${ram.used_gb.toFixed(1)} of ${ram.total_gb.toFixed(0)} GB in use · ${ram.available_gb.toFixed(1)} GB available`}>
                  <span className="tb-gpu-top"><span>RAM</span><span className="tb-temp faint">{ram.total_gb.toFixed(0)} GB</span></span>
                  <span className="tb-gpu-mem">
                    <span className={`tb-meter${ram.used_gb / ram.total_gb > 0.1 ? " on" : ""}`}><span style={{ width: `${(100 * ram.used_gb) / ram.total_gb}%` }} /></span>
                    {ram.used_gb.toFixed(1)} GB
                  </span>
                </div>
              )}
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
        ) : task ? (
          <a className="tb-mini tb-mini-task" href={`#/datasets/${encodeURIComponent(task.dataset_id)}`} aria-label={`${task.label}: ${taskMeta}`}>
            {taskPct(task) != null ? <Ring pct={taskPct(task)!} /> : <span className="tb-task-icon">{TASK_ICON[task.kind]}</span>}
            <span className="tb-meta">{task.label}</span>
          </a>
        ) : null}

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

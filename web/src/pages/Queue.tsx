import { useEffect, useRef, useState } from "react";
import { go } from "../App";
import { api, fmtDuration, fmtTime, type DiskUse, type Job, type JobDetail } from "../api";
import { EvalPanel } from "../components/EvalPanel";
import { EvaluateDialog } from "../components/EvalSettings";
import { HardwareCards } from "../components/Hardware";
import { LossChart } from "../components/LossChart";
import { IconCamera, IconCopy, IconDown, IconDownload, IconImages, IconPlay, IconRefresh, IconSave, IconStop, IconTarget, IconTrash, IconUp } from "../components/icons";
import { ConfirmDialog, Lightbox, Modal, StatusPill, fmtSize, usePoll, useToast } from "../components/ui";
import { SamplesPanel } from "../components/Samples";
import { Terminal } from "../components/Terminal";
import { CLONE_KEY } from "./CreateJob";

const PHASES: Record<string, string> = {
  starting: "Starting", preparing: "Scanning datasets", caching_text: "Encoding captions", caching_latents: "Encoding training files",
  loading: "Loading model", rendering: "Rendering training images", training: "Training", sampling: "Sampling", evaluating: "Evaluating checkpoint", completed: "Completed", stopped: "Stopped", failed: "Failed",
};

function useJobAction() {
  const toast = useToast();
  return async (job: { id: string; name: string; evaluating?: boolean }, action: string, after?: () => void) => {
    if (action === "restart" && !confirm(`Restart "${job.name}" from step 0? Its checkpoints and samples will be deleted.`)) return;
    try {
      await api.post(`/api/jobs/${job.id}/action`, { action });
      const msg: Record<string, string> = { stop: "evaluating" in job && job.evaluating ? "Stopping the checkpoint evaluation" : "Stopping — saving state…", sample: "Sampling requested", save: "Checkpoint requested", resume: "Re-queued (resumes from last save)", restart: "Restarted" };
      if (msg[action]) toast(msg[action]);
      after?.();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
}

async function cloneJob(id: string) {
  const cfg = await api.get<any>(`/api/jobs/${id}/config`);
  const profiles = await api.get<any[]>("/api/profiles");
  const prof = profiles.find((p) => p.name === cfg.profile_name);
  localStorage.setItem(CLONE_KEY, JSON.stringify({
    name: cfg.name, profile_id: prof?.id, settings: cfg.settings, gpu: "auto",
    // jobs from before per-dataset settings use the job's resolution, flip and dropout for every dataset
    datasets: cfg.datasets.map((d: any) => ({
      id: d.id, repeats: d.repeats, resolution: d.resolution ?? cfg.settings.resolution,
      flip: d.flip ?? cfg.settings.flip, caption_dropout: d.caption_dropout ?? cfg.settings.caption_dropout,
    })),
  }));
  go("create");
}

const FINISHED = ["completed", "stopped", "failed", "interrupted"];

/** What a job's folder holds on disk, with clearing what can be made again (or isn't needed any more). */
function DiskCard({ job }: { job: JobDetail }) {
  const toast = useToast();
  const [disk, reload] = usePoll<DiskUse>(`/api/jobs/${job.id}/disk`, 0, [job.status]);
  const [confirm, setConfirm] = useState<"cache" | "state" | null>(null);
  if (!disk) return null;
  const finished = FINISHED.includes(job.status) && !job.config?.evaluate;
  const clean = async (part: "cache" | "state") => {
    try {
      const r = await api.post<{ freed: number; kept: string[] }>("/api/disk/clean", { parts: [part], jobs: [job.id] });
      toast(r.kept.length ? r.kept.join("; ") : `Freed ${fmtSize(r.freed)}`, !!r.kept.length);
      reload();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  const rows: [string, number, React.ReactNode][] = [
    ["LoRA checkpoints", disk.output, null],
    ["Samples", disk.samples, null],
    ["Cached captions and latents", disk.cache, finished && disk.cache > 0 && <button className="btn sm" onClick={() => setConfirm("cache")}>Clear</button>],
    ["Training state (to resume or continue)", disk.state,
      job.status === "completed" && finished && disk.state > 0 && <button className="btn sm" onClick={() => setConfirm("state")}>Delete</button>],
    ["Log, metrics and status", disk.other, null],
  ];
  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <div className="row" style={{ marginBottom: 6 }}><h2 style={{ margin: 0 }}>Disk use</h2><span className="spacer" /><span className="muted">{fmtSize(disk.total)}</span></div>
      {rows.map(([label, size, action]) => (
        <div className="kv" key={label}><span>{label}</span><span className="row" style={{ gap: 10 }}>{fmtSize(size)}{action}</span></div>
      ))}
      {confirm === "cache" && (
        <ConfirmDialog title="Clear the cache" confirmLabel={`Free ${fmtSize(disk.cache)}`} onClose={() => setConfirm(null)} onConfirm={() => clean("cache")}
          message="The encoded captions and images are deleted. If this job continues or is evaluated, they're encoded again first, which loads the text encoder and VAE." />
      )}
      {confirm === "state" && (
        <ConfirmDialog title="Delete the training state" danger confirmLabel={`Free ${fmtSize(disk.state)}`} onClose={() => setConfirm(null)} onConfirm={() => clean("state")}
          message="The optimizer state saved for resuming is deleted, so this job can't continue training afterwards. Its LoRA checkpoints and samples stay." />
      )}
    </div>
  );
}

function JobRow({ j, reload, queuedIndex, queuedCount, size }: { j: Job; reload: () => void; queuedIndex: number; queuedCount: number; size?: number }) {
  const act = useJobAction();
  const toast = useToast();
  const p = j.progress;
  const pct = p.total_steps ? (100 * (p.step || 0)) / p.total_steps : 0;
  const active = j.status === "running" || j.status === "stopping";
  const [confirmDel, setConfirmDel] = useState(false);
  const remove = async (files: boolean) => {
    try { await api.del(`/api/jobs/${j.id}?delete_files=${files}`); reload(); } catch (e: any) { toast(e.message, true); }
  };
  return (
    <div className="card job-card">
      <div className="job-thumb" style={j.last_sample ? { backgroundImage: `url(/api/jobs/${j.id}/file/samples/${j.last_sample.poster || j.last_sample.file})`, cursor: "pointer" } : {}} onClick={() => go("queue", j.id)}>
        {!j.last_sample && <IconImages width={26} />}
      </div>
      <div style={{ minWidth: 0 }}>
        <div className="row">
          <a className="job-title ellipsis" href={`#/queue/${j.id}`} style={{ color: "var(--text)" }}>{j.name}</a>
          <StatusPill status={j.status} />
        </div>
        <div className="job-meta">
          <span>{j.profile_name}</span>
          {j.slider && <span>slider</span>}
          {j.network === "lokr" && <span>LoKr</span>}
          {j.datasets.length > 0 && <span>{j.datasets.join(", ")}</span>}
          {j.assigned_gpu != null && active && <span>GPU {j.assigned_gpu}</span>}
          {j.gpu !== "auto" && !active && <span>pinned GPU {j.gpu}</span>}
          <span>{j.evaluating ? (j.status === "queued" ? "checkpoint evaluation queued" : "evaluating checkpoints")
            : j.status === "queued" ? `queued ${fmtTime(j.created)}` : j.finished ? `finished ${fmtTime(j.finished)}` : j.started ? `started ${fmtTime(j.started)}` : ""}</span>
          {j.recommended && <span className="rec-meta">★ step {j.recommended.step} recommended</span>}
          {size ? <span title="On disk">{fmtSize(size)}</span> : null}
        </div>
        {active && j.evaluating && (
          <div className="small" style={{ marginTop: 10 }}>
            Checkpoint evaluation · {PHASES[p.phase || ""] || p.phase}
            {p.message && <div className="faint" style={{ marginTop: 4 }}>{p.message}</div>}
          </div>
        )}
        {!(active && j.evaluating) && (active || (p.step ?? 0) > 0) && (
          <div style={{ marginTop: 10 }}>
            <div className="row small" style={{ marginBottom: 5 }}>
              <span>{active ? PHASES[p.phase || ""] || p.phase : `Step ${p.step}`}{active && p.phase === "training" ? ` · step ${p.step} / ${p.total_steps}` : ""}</span>
              <span className="spacer" />
              {p.avg_loss != null && <span className="muted">loss {p.avg_loss.toFixed(4)}</span>}
              {active && p.it_per_sec != null && <span className="muted">{(1 / p.it_per_sec).toFixed(1)} s/it</span>}
              {active && p.eta_seconds != null && p.phase === "training" && <span className="muted">ETA {fmtDuration(p.eta_seconds)}</span>}
            </div>
            <div className="progress"><div style={{ width: `${pct}%` }} /></div>
            {active && p.phase !== "training" && p.message && <div className="faint small" style={{ marginTop: 4 }}>{p.message}</div>}
          </div>
        )}
        {j.status === "failed" && p.message && <div className="error-box small" style={{ marginTop: 8 }}>{p.message}</div>}
      </div>
      <div className="row">
        {j.status === "queued" && <>
          <button className="btn icon" title="Move up" disabled={queuedIndex === 0} onClick={() => act(j, "up", reload)}><IconUp /></button>
          <button className="btn icon" title="Move down" disabled={queuedIndex === queuedCount - 1} onClick={() => act(j, "down", reload)}><IconDown /></button>
        </>}
        {(j.status === "running" || j.status === "queued") && <button className="btn" onClick={() => act(j, "stop", reload)}><IconStop /> {j.status === "queued" ? "Cancel" : "Stop"}</button>}
        {j.status === "stopping" && <button className="btn danger" onClick={() => act(j, "stop", reload)}>Force stop</button>}
        {["stopped", "failed", "interrupted"].includes(j.status) && <button className="btn primary" onClick={() => act(j, "resume", reload)}><IconPlay /> Resume</button>}
        {!active && j.status !== "queued" && <button className="btn icon" title="Duplicate settings into a new job" onClick={() => cloneJob(j.id)}><IconCopy /></button>}
        {!active && <button className="btn icon danger" title="Delete" onClick={() => setConfirmDel(true)}><IconTrash /></button>}
      </div>
      {confirmDel && (
        <ConfirmDialog title="Delete job" danger confirmLabel="Delete job" onClose={() => setConfirmDel(false)} onConfirm={remove}
          message={<>Remove <strong style={{ color: "var(--text)" }}>{j.name}</strong> from the queue?</>}
          checkbox="Also delete its LoRA checkpoints and samples from disk" />
      )}
    </div>
  );
}

const SCHEDULES: Record<string, string> = { cosine: "cosine", linear: "linear decay" };

/** Train a finished job on to a new total step count, from its last step. */
function ContinueDialog({ job, onClose, onQueued }: { job: JobDetail; onClose: () => void; onQueued: () => void }) {
  const toast = useToast();
  const st = job.status_detail;
  const done = st.step || 0;
  const [steps, setSteps] = useState(done + Math.max(100, Math.round(done / 4 / 50) * 50));
  const [busy, setBusy] = useState(false);
  const final = (st.checkpoints || []).find((c) => c.final);
  const kept = final ? final.file.replace(/\.safetensors$/, `_${String(final.step).padStart(6, "0")}.safetensors`) : "";
  const schedule = SCHEDULES[job.config?.settings?.lr_scheduler];
  const start = async () => {
    setBusy(true);
    try {
      await api.post(`/api/jobs/${job.id}/action`, { action: "extend", steps });
      toast(`Queued to train on to step ${steps}`);
      onQueued();
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Continue training" onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy || !(steps > done)} onClick={start}><IconPlay /> Continue training</button>
      </>
    }>
      <div className="stack" style={{ gap: 12 }}>
        <label className="field">
          <span>Train to step</span>
          <input type="number" min={done + 1} step={50} value={Number.isFinite(steps) ? steps : ""} autoFocus
            onChange={(e) => setSteps(e.target.value === "" ? NaN : Number(e.target.value))} />
          <span className="help">{steps > done ? `${steps - done} more steps, ` : ""}from step {done} with the same settings, optimizer state and cached captions and images. Samples, saves and any checkpoint picker carry on at the same intervals.</span>
        </label>
        {final && (
          <div className="muted small">The current final LoRA is kept as <span className="mono">{kept}</span>, and "keep last N" never deletes it. The new end saves as <span className="mono">{final.file}</span>.</div>
        )}
        {schedule && (
          <div className="validation warning">The {schedule} learning-rate schedule stretches to the new length, so the learning rate picks up a little from where it ended before decaying again.</div>
        )}
      </div>
    </Modal>
  );
}

function JobDetailView({ id }: { id: string }) {
  const act = useJobAction();
  const [job, reload] = usePoll<JobDetail>(`/api/jobs/${id}`, 2500);
  const [metrics, setMetrics] = useState<{ step: number; loss: number }[]>([]);
  const [light, setLight] = useState<string | null>(null);
  const [evaluate, setEvaluate] = useState(false);
  const [extend, setExtend] = useState(false);
  const isVideoFile = (f: string) => f.endsWith(".mp4");
  const lastStep = useRef(0);

  useEffect(() => {
    let live = true;
    lastStep.current = 0;
    setMetrics([]);
    type M = { points: { step: number; loss: number }[]; last: number };
    const pull = (): Promise<void> => api.get<M>(`/api/jobs/${id}/metrics?after=${lastStep.current}`).then(({ points, last }) => {
      if (!live) return;
      if (last < lastStep.current) {
        // the job was restarted and its metrics rewritten from step 1
        lastStep.current = 0;
        setMetrics([]);
        return pull();
      }
      if (!points.length) return;
      lastStep.current = points[points.length - 1].step;
      setMetrics((xs) => [...xs.filter((x) => x.step < points[0].step), ...points]);
    });
    pull();
    const t = setInterval(pull, 3000);
    return () => { live = false; clearInterval(t); };
  }, [id]);

  if (!job) return <div className="page"><div className="muted">Loading…</div></div>;
  const st = job.status_detail || ({} as JobDetail["status_detail"]);
  const total = st.total_steps || job.config?.settings?.steps || 0;
  const running = job.status === "running";
  const evaluating = !!job.config?.evaluate; // queued or running for a checkpoint evaluation
  const slider = !!job.config?.settings?.slider_enabled;
  const jobPath = `${job.dir}/output/`;

  return (
    <div className="page">
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="small"><a href="#/queue">Queue</a> <span className="faint">/</span></div>
          <div className="row"><h1 className="ellipsis">{job.name}</h1><StatusPill status={job.status} /></div>
          <p>{[job.config?.profile_name, slider && `slider: ${job.config?.settings?.slider_target}`,
            job.config?.settings?.network_type === "lokr" && "LoKr",
            job.config?.datasets?.map((d: any) => d.name).join(", ")].filter(Boolean).join(" · ")} {job.assigned_gpu != null && `· GPU ${job.assigned_gpu}`}</p>
        </div>
        <div className="row">
          {running && !evaluating && <>
            <button className="btn" onClick={() => act(job, "sample")}><IconCamera /> Sample now</button>
            <button className="btn" onClick={() => act(job, "save")}><IconSave /> Save now</button>
            <button className="btn" onClick={() => act(job, "stop", reload)}><IconStop /> Stop</button>
          </>}
          {evaluating && ["queued", "running"].includes(job.status) && (
            <button className="btn" onClick={() => act({ ...job, evaluating }, "stop", reload)}><IconStop /> {job.status === "queued" ? "Cancel evaluation" : "Stop evaluation"}</button>
          )}
          {["stopped", "failed", "interrupted"].includes(job.status) && <button className="btn primary" onClick={() => act(job, "resume", reload)}><IconPlay /> Resume</button>}
          {job.status === "completed" && !evaluating && job.resumable && (
            <button className="btn" title="Train this job on for more steps, from where it finished" onClick={() => setExtend(true)}><IconPlay /> Continue training</button>
          )}
          {!running && !evaluating && !["queued", "stopping"].includes(job.status) && (st.checkpoints || []).length > 0 && (
            <button className="btn" title="Score the saved checkpoints with the checkpoint picker and/or concept bleed, and recommend one" onClick={() => setEvaluate(true)}><IconTarget /> Evaluate checkpoints</button>
          )}
          {!running && job.status !== "queued" && job.status !== "stopping" && <button className="btn" onClick={() => act(job, "restart", reload)}><IconRefresh /> Restart</button>}
          {!running && <button className="btn" onClick={() => cloneJob(job.id)}><IconCopy /> Duplicate</button>}
        </div>
      </div>

      {job.status === "failed" && st.message && <div className="error-box" style={{ marginBottom: 16 }}>{st.message}</div>}

      <div className="stats" style={{ marginBottom: 16 }}>
        <div className="stat"><div className="k">Phase</div><div className="v" style={{ fontSize: 15 }}>{PHASES[st.phase || ""] || st.phase || "Queued"}</div></div>
        <div className="stat"><div className="k">Step</div><div className="v">{st.step ?? 0}<span className="faint" style={{ fontSize: 13 }}> / {total}</span></div></div>
        <div className="stat"><div className="k">Avg loss</div><div className="v">{st.avg_loss?.toFixed(4) ?? "—"}</div></div>
        <div className="stat"><div className="k">Speed</div><div className="v">{st.it_per_sec ? `${(1 / st.it_per_sec).toFixed(2)}s` : "—"}<span className="faint" style={{ fontSize: 13 }}> /it</span></div></div>
        <div className="stat"><div className="k">ETA</div><div className="v">{running ? fmtDuration(st.eta_seconds) : "—"}</div></div>
        <div className="stat"><div className="k">Peak VRAM</div><div className="v">{st.vram_gb ? `${st.vram_gb} GB` : "—"}</div></div>
        <div className="stat"><div className="k">LR</div><div className="v" style={{ fontSize: 15 }}>{st.lr?.toExponential(2) ?? "—"}</div></div>
      </div>
      <div className="progress" style={{ marginBottom: 16 }}><div style={{ width: `${total ? (100 * (st.step || 0)) / total : 0}%` }} /></div>

      <SamplesPanel jobId={job.id} samples={st.samples || []} slider={slider} />

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <h2>Loss</h2>
        <LossChart points={metrics} totalSteps={total} />
      </div>

      <Terminal jobId={job.id} live={running || job.status === "stopping"}
        alt={{ label: "Config", text: JSON.stringify(job.config, null, 2) }} />

      <EvalPanel job={job} onOpen={setLight} />

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <h2>Checkpoints</h2>
        {(st.checkpoints || []).length === 0 && <div className="muted small">None yet.</div>}
        {[...(st.checkpoints || [])].reverse().map((c) => (
          <div className="kv" key={c.file}>
            <span className="mono ellipsis">{c.file}</span>
            <span className="row" style={{ gap: 8 }}>
              <span className="faint small">step {c.step}</span>
              {c.final && <span className="pill completed">final</span>}
              <a className="btn sm" href={`/api/jobs/${job.id}/file/output/${c.file}`} download><IconDownload /></a>
            </span>
          </div>
        ))}
        <div className="help" style={{ marginTop: 10 }}>Saved in <span className="mono">{jobPath}</span> — ComfyUI-compatible; copy into <span className="mono">models/loras</span>.</div>
      </div>

      <DiskCard job={job} />

      {light && <Lightbox src={light} video={isVideoFile(light)} onClose={() => setLight(null)}
        audio={light.endsWith(".mp3") ? { poster: light.replace(/\.mp3$/, "_poster.png") } : undefined} />}
      {evaluate && <EvaluateDialog job={job} onClose={() => setEvaluate(false)} onQueued={reload} />}
      {extend && <ContinueDialog job={job} onClose={() => setExtend(false)} onQueued={reload} />}
    </div>
  );
}

export default function Queue({ jobId }: { jobId: string }) {
  const [jobs, reload] = usePoll<Job[]>("/api/jobs", 2500);
  if (jobId) return <JobDetailView id={jobId} />;
  return <QueueList jobs={jobs} reload={reload} />;
}

function QueueList({ jobs, reload }: { jobs: Job[] | null; reload: () => void }) {
  const toast = useToast();
  const [disk, reloadDisk] = usePoll<{ jobs: Record<string, DiskUse>; total: DiskUse }>("/api/disk", 0, [jobs?.length]);
  const [confirmClean, setConfirmClean] = useState(false);
  // what clearing would free: the caches of finished jobs
  const clearable = (jobs || []).filter((j) => FINISHED.includes(j.status) && !j.evaluating)
    .reduce((n, j) => n + (disk?.jobs[j.id]?.cache || 0), 0);
  const clean = async () => {
    try {
      const r = await api.post<{ freed: number; jobs: number }>("/api/disk/clean", { parts: ["cache"] });
      toast(`Freed ${fmtSize(r.freed)} from ${r.jobs} job${r.jobs === 1 ? "" : "s"}`);
      reloadDisk();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  const running = jobs?.filter((j) => j.status === "running" || j.status === "stopping") || [];
  const queued = jobs?.filter((j) => j.status === "queued") || [];
  const done = (jobs?.filter((j) => !["running", "stopping", "queued"].includes(j.status)) || []).sort((a, b) => (b.finished || b.created) - (a.finished || a.created));
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Queue</h1>
          <p>Running, queued and finished training jobs. Queue mode and GPUs are set in <a href="#/settings">Settings</a>.</p>
        </div>
        <button className="btn primary" onClick={() => go("create")}>New job</button>
      </div>
      <HardwareCards />
      {disk && disk.total.total > 0 && (
        <div className="row wrap small queue-disk">
          <span className="muted">Jobs use {fmtSize(disk.total.total)} on disk: {fmtSize(disk.total.output)} of LoRAs, {fmtSize(disk.total.cache)} of cached captions and latents, {fmtSize(disk.total.state)} of training state.</span>
          {clearable > 0 && <button className="btn sm" onClick={() => setConfirmClean(true)}>Clear caches of finished jobs</button>}
        </div>
      )}
      {confirmClean && (
        <ConfirmDialog title="Clear caches" confirmLabel={`Free ${fmtSize(clearable)}`} onClose={() => setConfirmClean(false)} onConfirm={clean}
          message="The encoded captions and images of finished jobs are deleted. A job that continues or is evaluated encodes them again first. Checkpoints, samples and training state stay." />
      )}
      {jobs && jobs.length === 0 && <div className="empty"><strong>Nothing in the queue</strong><a href="#/create">Create a job →</a></div>}
      {[["In progress", running], ["Queued", queued], ["Finished", done]].map(([title, list]) =>
        (list as Job[]).length > 0 && (
          <div key={title as string} style={{ marginBottom: 26 }}>
            <div className="section-title">{title as string} · {(list as Job[]).length}</div>
            <div className="stack" style={{ gap: 10 }}>
              {(list as Job[]).map((j, i) => <JobRow key={j.id} j={j} reload={reload} queuedIndex={i} queuedCount={(list as Job[]).length} size={disk?.jobs[j.id]?.total} />)}
            </div>
          </div>
        ),
      )}
    </div>
  );
}

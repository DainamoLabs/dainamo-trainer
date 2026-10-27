import { useEffect, useState } from "react";
import { go } from "../App";
import {
  api, fmtDuration, fmtLeft, fmtSpeed, fmtTime, type Checkpoint, type DiskUse, type ExtractMade, type Family, type Job, type JobDetail, type JobDiskUse,
  type MergeMade, type Profile, type RefModMade,
} from "../api";
import { EvalPanel } from "../components/EvalPanel";
import { EvaluateDialog } from "../components/EvalSettings";
import { ExtractLora } from "../components/ExtractLora";
import { LikenessChart } from "../components/Likeness";
import { LossChart, Spark, useMetrics } from "../components/LossChart";
import { IconCamera, IconCheck, IconCopy, IconDatabase, IconDown, IconDownload, IconDrive, IconEdit, IconFile, IconImages, IconLayers, IconPlay, IconPlus, IconRefresh, IconRight, IconSave, IconServer, IconStop, IconTarget, IconTrash, IconUp } from "../components/icons";
import { AudioCard, ConfirmDialog, Empty, Lightbox, LoadError, Modal, SelectBar, StatusPill, fmtSize, someNames, usePoll, useSelection, useToast, type Selection } from "../components/ui";
import { SamplesPanel } from "../components/Samples";
import { Terminal } from "../components/Terminal";
import { CLONE_KEY, formOf, jobProfile } from "./CreateJob";

export const PHASES: Record<string, string> = {
  starting: "Starting", preparing: "Scanning datasets", caching_text: "Encoding captions", caching_latents: "Encoding training files",
  loading: "Loading model", rendering: "Rendering training images", training: "Training", sampling: "Sampling", evaluating: "Evaluating checkpoints", completed: "Completed", stopped: "Stopped", failed: "Failed",
  encoding: "Making the RefMod", extracting: "Extracting the LoRA", merging: "Merging the LoRAs",
};

type Acting = { id: string; name: string; status?: string; evaluating?: boolean; refmod?: boolean; extract?: boolean; merge?: boolean };

/** A job's actions (stop, resume, restart, sample, save, up, down), each toasted, and the dialog that confirms a
 * restart, which deletes what the job made: render ``dialog`` beside the buttons. A RefMod job (``refmod``), an
 * extracted LoRA's (``extract``) or a merged one's (``merge``) is made again from the start by "restart". */
function useJobAction(): [(job: Acting, action: string, after?: () => void) => void, React.ReactNode] {
  const toast = useToast();
  const [asking, setAsking] = useState<{ job: Acting; after?: () => void } | null>(null);
  const send = async (job: Acting, action: string, after?: () => void) => {
    try {
      await api.post(`/api/jobs/${job.id}/action`, { action });
      const again = job.refmod ? "Queued to be made again" : job.extract ? "Queued to be extracted again" : job.merge ? "Queued to be merged again" : "";
      const msg: Record<string, string> = {
        stop: job.evaluating ? (job.status === "queued" ? "Checkpoint evaluation cancelled" : "Stopping the checkpoint evaluation")
          : job.status === "queued" ? "Cancelled" : job.status === "stopping" ? "Forcing it to stop" : again ? "Stopping" : "Stopping: saving its state first",
        start: "Started: it's queued for a free GPU",
        sample: "Sample requested", save: "Save requested", resume: again || "Queued to resume from its last save",
        restart: again || "Queued to train again from step 0",
      };
      if (msg[action]) toast(msg[action]);
      after?.();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  const act = (job: Acting, action: string, after?: () => void) => (action === "restart" ? setAsking({ job, after }) : send(job, action, after));
  const j = asking?.job;
  const dialog = j && (
    <ConfirmDialog danger onClose={() => setAsking(null)} onConfirm={() => send(j, "restart", asking!.after)}
      title={j.refmod ? "Make the RefMod again" : j.extract ? "Extract the LoRA again" : j.merge ? "Merge the LoRAs again" : "Restart from step 0"}
      confirmLabel={j.refmod ? "Make again" : j.extract ? "Extract again" : j.merge ? "Merge again" : "Restart"}
      message={<><strong style={{ color: "var(--text)" }}>{j.name}</strong>{j.refmod ? "'s RefMod and its preview are deleted, and it's made again from the start."
        : j.extract || j.merge ? "'s LoRA is deleted, and it's made again from the start."
        : "'s checkpoints and samples are deleted, and it trains again from step 0."}</>} />
  );
  return [act, dialog];
}

async function cloneJob(id: string) {
  const cfg = await api.get<any>(`/api/jobs/${id}/config`);
  const profiles = await api.get<any[]>("/api/profiles");
  localStorage.setItem(CLONE_KEY, JSON.stringify({ ...formOf(cfg, profiles), gpu: "auto" }));
  go("create");
}

export const FINISHED = ["completed", "stopped", "failed", "interrupted"];

/** What a job's folder holds on disk, with clearing what can be made again (or isn't needed any more). */
function DiskCard({ job }: { job: JobDetail }) {
  const toast = useToast();
  // while it runs, the shared cache's files it uses grow as it encodes them
  const [disk, reload] = usePoll<JobDiskUse>(`/api/jobs/${job.id}/disk`, job.status === "running" ? 10000 : 0, [job.status]);
  const [confirm, setConfirm] = useState<"cache" | "state" | null>(null);
  if (!disk) return null;
  const finished = FINISHED.includes(job.status) && !job.config?.evaluate;
  const refmod = !!job.config?.settings?.refmod_enabled;
  const finetune = !!job.config?.settings?.finetune_enabled;
  const extract = !!job.config?.settings?.extract_enabled;
  const merge = !!job.config?.settings?.merge_enabled;
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
    [refmod ? "RefMod" : finetune ? "Finetuned models" : extract ? "Extracted LoRA" : merge ? "Merged LoRA" : "LoRA checkpoints", disk.output, null],
    ...(extract || merge ? [] : [[refmod ? "Preview" : "Samples", disk.samples, null] as [string, number, React.ReactNode]]),
    // a job's own cache is from before caches were shared between jobs
    ...(disk.cache > 0 ? [["Cached captions and latents (its own, from before they were shared)", disk.cache,
      finished && <button className="btn sm" onClick={() => setConfirm("cache")}>Clear</button>] as [string, number, React.ReactNode]] : []),
    ...((refmod || extract || merge) && !disk.state ? [] : [["Training state (to resume or continue)", disk.state,
      job.status === "completed" && finished && disk.state > 0 && <button className="btn sm" onClick={() => setConfirm("state")}>Delete</button>] as [string, number, React.ReactNode]]),
    ["Log, metrics and status", disk.other, null],
  ];
  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <div className="row" style={{ marginBottom: 6 }}><h2 style={{ margin: 0 }}>Disk use</h2><span className="spacer" /><span className="muted">{fmtSize(disk.total)}</span></div>
      {rows.map(([label, size, action]) => (
        <div className="kv" key={label}><span>{label}</span><span className="row" style={{ gap: 10 }}>{fmtSize(size)}{action}</span></div>
      ))}
      {disk.shared > 0 && (
        <div className="help" style={{ marginTop: 10 }}>
          It also uses {fmtSize(disk.shared)} of encoded captions and latents in the cache jobs share (not counted above). Any
          job encoding the same files the same way reuses them; Manage cache on the Queue page clears what no queued or running job uses.
        </div>
      )}
      {confirm === "cache" && (
        <ConfirmDialog title="Clear the cache" confirmLabel={`Free ${fmtSize(disk.cache)}`} onClose={() => setConfirm(null)} onConfirm={() => clean("cache")}
          message="Its own encoded captions and images are deleted. Jobs now use the shared cache instead, so if this job continues or is evaluated, they're encoded there first (unless another job already has)." />
      )}
      {confirm === "state" && (
        <ConfirmDialog title="Delete the training state" danger confirmLabel={`Free ${fmtSize(disk.state)}`} onClose={() => setConfirm(null)} onConfirm={() => clean("state")}
          message="The optimizer state saved for resuming is deleted, so this job can't continue training afterwards. Its LoRA checkpoints and samples stay." />
      )}
    </div>
  );
}

function JobRow({ j, reload, queuedIndex, queuedCount, size, sel }: { j: Job; reload: () => void; queuedIndex: number; queuedCount: number; size?: number; sel: Selection }) {
  const [act, confirmRestart] = useJobAction();
  const toast = useToast();
  const p = j.progress;
  const pct = p.total_steps ? (100 * (p.step || 0)) / p.total_steps : 0;
  const active = j.status === "running" || j.status === "stopping";
  const once = j.refmod || j.extract || j.merge; // makes one file: made again from the start, not resumed
  const [confirmDel, setConfirmDel] = useState(false);
  const remove = async (files: boolean) => {
    try { await api.del(`/api/jobs/${j.id}?delete_files=${files}`); reload(); } catch (e: any) { toast(e.message, true); }
  };
  // picking jobs to delete: the card picks it; a running one can't be deleted
  const picked = sel.on && sel.picked.has(j.id);
  return (
    <div className={"card job-card" + (sel.on ? " selecting" : "") + (picked ? " picked" : "") + (sel.on && active ? " locked" : "")}
      onClick={sel.on && !active ? () => sel.toggle(j.id) : undefined}>
      {sel.on && <input type="checkbox" className="pick" checked={picked} disabled={active} aria-label={`Select ${j.name}`}
        title={active ? "Stop it before deleting it" : undefined} onClick={(e) => e.stopPropagation()} onChange={() => sel.toggle(j.id)} />}
      <div className="job-thumb" style={j.last_sample ? { backgroundImage: `url(/api/jobs/${j.id}/file/samples/${j.last_sample.poster || j.last_sample.file})`, cursor: "pointer" } : {}}
        onClick={sel.on ? undefined : () => go("queue", j.id)}>
        {!j.last_sample && <IconImages width={26} />}
      </div>
      <div style={{ minWidth: 0 }}>
        <div className="row">
          {sel.on ? <span className="job-title ellipsis">{j.name}</span>
            : <a className="job-title ellipsis" href={`#/queue/${j.id}`} style={{ color: "var(--text)" }}>{j.name}</a>}
          <StatusPill status={j.status} />
        </div>
        <div className="job-meta">
          <span>{j.profile_name}</span>
          {j.slider && <span>slider</span>}
          {j.refmod && <span>RefMod</span>}
          {j.extract && <span>LoRA from a finetune</span>}
          {j.merge && <span>merged LoRA</span>}
          {j.network === "lokr" && !j.refmod && <span>LoKr</span>}
          {j.network === "full" && <span>finetune</span>}
          {j.datasets.length > 0 && <span>{j.datasets.join(", ")}</span>}
          {j.assigned_gpu != null && active && <span>GPU {j.assigned_gpu}</span>}
          {j.gpu !== "auto" && !active && <span>pinned GPU {j.gpu}</span>}
          <span>{j.evaluating ? (j.status === "queued" ? "checkpoint evaluation queued" : "evaluating checkpoints")
            : j.status === "queued" ? `queued ${fmtTime(j.created)}` : j.status === "ready" ? `added ${fmtTime(j.created)}` : j.finished ? `finished ${fmtTime(j.finished)}` : j.started ? `started ${fmtTime(j.started)}` : ""}</span>
          {j.recommended && <span className="rec-meta">★ step {j.recommended.step} recommended</span>}
          {size ? <span title="On disk">{fmtSize(size)}</span> : null}
        </div>
        {active && j.evaluating && (
          <div className="small" style={{ marginTop: 10 }}>
            Checkpoint evaluation · {PHASES[p.phase || ""] || p.phase}
            {p.message && <div className="faint" style={{ marginTop: 4 }}>{p.message}</div>}
          </div>
        )}
        {!(active && j.evaluating) && (active || ((p.step ?? 0) > 0 && !j.extract && !j.merge)) && (
          <div style={{ marginTop: 10 }}>
            <div className="row small" style={{ marginBottom: 5 }}>
              <span>{active ? PHASES[p.phase || ""] || p.phase : `Step ${p.step}`}{active && p.phase === "training" ? ` · step ${p.step} / ${p.total_steps}` : ""}</span>
              <span className="spacer" />
              <Spark values={j.loss_curve} className="job-spark" />
              {p.avg_loss != null && <span className="muted">loss {p.avg_loss.toFixed(4)}</span>}
              {active && p.it_per_sec != null && <span className="muted">{fmtSpeed(1 / p.it_per_sec)}</span>}
              {active && p.eta_seconds != null && p.phase === "training" && <span className="muted">{fmtLeft(p.eta_seconds)}</span>}
            </div>
            <div className="progress"><div style={{ width: `${pct}%` }} /></div>
            {active && p.phase !== "training" && p.message && <div className="faint small" style={{ marginTop: 4 }}>{p.message}</div>}
          </div>
        )}
        {j.status === "failed" && p.message && <div className="error-box small" style={{ marginTop: 8 }}>{p.message}</div>}
      </div>
      {!sel.on && <div className="row">
        {j.status === "queued" && <>
          <button className="btn icon" title="Move up" disabled={queuedIndex === 0} onClick={() => act(j, "up", reload)}><IconUp /></button>
          <button className="btn icon" title="Move down" disabled={queuedIndex === queuedCount - 1} onClick={() => act(j, "down", reload)}><IconDown /></button>
        </>}
        {j.status === "ready" && <button className="btn primary" title="Queue it to run when a GPU is free" onClick={() => act(j, "start", reload)}><IconPlay /> Start</button>}
        {(j.status === "running" || j.status === "queued") && <button className="btn" onClick={() => act(j, "stop", reload)}><IconStop /> {j.status === "queued" ? "Cancel" : "Stop"}</button>}
        {j.status === "stopping" && <button className="btn danger" onClick={() => act(j, "stop", reload)}>Force stop</button>}
        {["stopped", "failed", "interrupted"].includes(j.status) && (once
          ? <button className="btn primary" onClick={() => act(j, "restart", reload)}><IconRefresh /> {j.refmod ? "Make again" : j.extract ? "Extract again" : "Merge again"}</button>
          : j.resumable ? <button className="btn primary" onClick={() => act(j, "resume", reload)}><IconPlay /> Resume</button>
          : <button className="btn primary" title="It saved no training state to resume from" onClick={() => act(j, "restart", reload)}><IconRefresh /> Restart</button>)}
        {!active && j.status !== "queued" && j.status !== "ready" && !j.extract && !j.merge && <button className="btn icon" title="Duplicate settings into a new job" onClick={() => cloneJob(j.id)}><IconCopy /></button>}
        {!active && <button className="btn icon danger" title="Delete" onClick={() => setConfirmDel(true)}><IconTrash /></button>}
      </div>}
      {confirmDel && (
        <ConfirmDialog title="Delete job" danger confirmLabel="Delete job" onClose={() => setConfirmDel(false)} onConfirm={remove}
          message={<>Remove <strong style={{ color: "var(--text)" }}>{j.name}</strong> from the queue?</>}
          checkbox={j.refmod ? "Also delete its RefMod and preview from disk" : j.extract || j.merge ? "Also delete its LoRA from disk"
            : j.network === "full" ? "Also delete its finetuned models and samples from disk"
            : "Also delete its LoRA checkpoints and samples from disk"} />
      )}
      {confirmRestart}
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
          <div className="muted small">The current final {job.config?.settings?.finetune_enabled ? "model" : "LoRA"} is kept as <span className="mono">{kept}</span>, and "Keep last N saves" never deletes it. The new end saves as <span className="mono">{final.file}</span>.</div>
        )}
        {schedule && (
          <div className="validation warning">The {schedule} learning-rate schedule stretches to the new length, so the learning rate picks up a little from where it ended before decaying again.</div>
        )}
      </div>
    </Modal>
  );
}

const CONCEPT_LABELS: Record<string, string> = {
  generic: "generic", identity: "identity", pose_motion: "pose & motion", clothing: "clothing", background: "background", style: "style",
  voice: "voice", singing: "singing", music_style: "music style", sound_fx: "sound effect", ambience: "ambience",
};

/** What a RefMod job made: its file and shape, where it was copied, and a preview of some of its frames (or, for
 * an audio one, its sound decoded). */
function RefModCard({ job, made, onOpen }: { job: JobDetail; made?: RefModMade; onOpen: (src: string) => void }) {
  const s = job.config?.settings || {};
  // the preview's files keep their names when it's made again
  const v = `?v=${(job.status_detail as any)?.started_at || ""}`;
  const url = (f: string) => `/api/jobs/${job.id}/file/samples/${f}${v}`;
  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <h2>RefMod</h2>
      {!made ? (
        <div className="muted small">{job.status === "queued" ? "Not made yet." : job.status === "running" ? "Being made…" : "Not made."}</div>
      ) : <>
        <div className="kv">
          <span className="mono ellipsis">{made.file}</span>
          <span className="row" style={{ gap: 8 }}>
            <span className="faint small">{fmtSize(made.bytes)}</span>
            <a className="btn sm" href={`/api/jobs/${job.id}/file/output/${made.file}`} download><IconDownload /></a>
          </span>
        </div>
        {made.kind === "audio" ? <>
          <div className="kv"><span>Of</span><span>sound{s.refmod_concept && s.refmod_concept !== "generic" ? ` · ${CONCEPT_LABELS[s.refmod_concept] || s.refmod_concept}` : ""}</span></div>
          <div className="kv"><span>Sources</span><span>{made.references}</span></div>
          <div className="kv"><span>Length</span><span>{made.seconds} s</span></div>
        </> : <>
          <div className="kv"><span>Encoding</span><span>{made.mode}{s.refmod_merge && made.mode === "compressed" && made.references > 1 ? ", merged" : ""}{s.refmod_concept && s.refmod_concept !== "generic" ? ` · ${CONCEPT_LABELS[s.refmod_concept] || s.refmod_concept}` : ""}</span></div>
          <div className="kv"><span>References</span><span>{made.references}</span></div>
          <div className="kv"><span>Latent</span><span>{made.frames} frame{made.frames === 1 ? "" : "s"} of {made.width} × {made.height} cells ({made.width * 16} × {made.height * 16} px)</span></div>
        </>}
        <div className="kv"><span>Reference tokens</span><span>{made.tokens.toLocaleString()}</span></div>
        {made.copied_to && <div className="kv"><span>Copied to</span><span className="mono ellipsis">{made.copied_to}</span></div>}
        {made.audio && <>
          <div className="section-title" style={{ marginTop: 16, marginBottom: 10 }}>Listen</div>
          <AudioCard src={url(made.audio.file)} poster={url(made.audio.poster)} />
          <div className="help" style={{ marginTop: 8 }}>Its latents decoded by the audio VAE: what the model is given to take after.</div>
        </>}
        {made.preview.length > 0 && <>
          <div className="section-title" style={{ marginTop: 16, marginBottom: 10 }}>Preview</div>
          <div className="samples-grid">
            {made.preview.map((p) => (
              <figure key={p.file}>
                <img src={url(p.file)} alt={`Frame ${p.frame + 1}`} loading="lazy" onClick={() => onOpen(url(p.file))} />
                <figcaption>frame {p.frame + 1} of {made.frames}</figcaption>
              </figure>
            ))}
          </div>
          <div className="help" style={{ marginTop: 8 }}>
            {made.mode === "compressed" ? "Each frame decoded from its grid, so it shows the coarse picture the model gets, not the detail of the references. " : ""}
            {made.frames > 1 ? "Frames are decoded one at a time, so those within a clip look rougher than its first." : ""}
          </div>
        </>}
      </>}
      <div className="help" style={{ marginTop: 10 }}>
        Saved in <span className="mono">{job.dir}/output/</span>. Use it in the Sandbox's Reference to video with a ref2va profile
        {made?.kind === "audio" ? " (beside a reference image or a visual RefMod)" : ""}, or in
        ComfyUI: copy it into <span className="mono">models/refmods</span> and load it with the ComfyUI-MiniMaxH3Mod nodes.
      </div>
    </div>
  );
}

const pct = (x: number) => `${(100 * x).toFixed(x >= 0.9995 ? 0 : 1)}%`;
const basename = (p: string) => String(p || "").split(/[\\/]/).pop() || "";

/** What an extract job made: its LoRA, the finetune checkpoint and original model it's the difference of, and how
 * much of that change it keeps. */
function ExtractCard({ job, made }: { job: JobDetail; made?: ExtractMade }) {
  const s = job.config?.settings || {};
  const from = s.extract_from || {};
  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <h2>LoRA</h2>
      {!made ? (
        <div className="muted small" style={{ marginBottom: 8 }}>{job.status === "queued" ? "Not extracted yet." : job.status === "running" ? "Being extracted…" : "Not extracted."}</div>
      ) : <>
        <div className="kv">
          <span className="mono ellipsis">{made.file}</span>
          <span className="row" style={{ gap: 8 }}>
            <span className="faint small">{fmtSize(made.bytes)}</span>
            <a className="btn sm" href={`/api/jobs/${job.id}/file/output/${made.file}`} download><IconDownload /></a>
          </span>
        </div>
        <div className="kv"><span>Kept</span><span>{pct(made.kept)} of the change{made.worst ? ` · least in ${made.worst.name}: ${pct(made.worst.kept)}` : ""}</span></div>
        <div className="kv"><span>Layers</span><span>{made.layers} as rank {made.rank} LoRAs, {made.whole} kept whole{made.unchanged ? `, ${made.unchanged} unchanged` : ""}</span></div>
        <div className="kv"><span>Change</span><span>the finetune moved the weights by {(100 * made.change).toFixed(2)}%</span></div>
      </>}
      <div className="kv"><span>Finetuned</span><span className="ellipsis">{from.job_id && <><a href={`#/queue/${from.job_id}`}>{from.job_name}</a>{" · "}</>}<span className="mono">{made?.source || from.file || basename(s.extract_source)}</span>{from.step ? ` (step ${from.step})` : ""}</span></div>
      <div className="kv"><span>Original</span><span className="mono ellipsis">{made?.base || basename(s.extract_base)}</span></div>
      <div className="help" style={{ marginTop: 10 }}>
        What the finetune changed, as a LoRA: each weight matrix's change at rank {s.extract_rank} (its largest part), and the
        changes to norms and biases whole. Kept is how much of the change it holds, by squared size; the rest is spread too
        thin for this rank. To see whether that's enough, render the same prompt and seed with it on the original model and
        with the finetuned model in the Sandbox, and if it falls short, extract again at a higher rank. Saved in{" "}
        <span className="mono">{job.dir}/output/</span>; it's in the Sandbox's LoRAs, and ComfyUI loads it from{" "}
        <span className="mono">models/loras</span> on the original model.
      </div>
    </div>
  );
}

/** What a merge job made: its LoRA, the LoRAs it merged at their strengths, and how much of their change it keeps. */
function MergeCard({ job, made }: { job: JobDetail; made?: MergeMade }) {
  const s = job.config?.settings || {};
  const inputs: { path: string; strength: number }[] = s.merge_loras || [];
  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <h2>LoRA</h2>
      {!made ? (
        <div className="muted small" style={{ marginBottom: 8 }}>{job.status === "queued" ? "Not merged yet." : job.status === "running" ? "Being merged…" : "Not merged."}</div>
      ) : <>
        <div className="kv">
          <span className="mono ellipsis">{made.file}</span>
          <span className="row" style={{ gap: 8 }}>
            <span className="faint small">{fmtSize(made.bytes)}</span>
            <a className="btn sm" href={`/api/jobs/${job.id}/file/output/${made.file}`} download><IconDownload /></a>
          </span>
        </div>
        <div className="kv"><span>Kept</span><span>{pct(made.kept)} of the merged change{made.worst ? ` · least in ${made.worst.name}: ${pct(made.worst.kept)}` : ""}</span></div>
        <div className="kv"><span>Layers</span><span>{made.layers} as LoRAs{made.rank ? ` (${made.cut} cut to rank ${made.rank})` : ""}, {made.whole} kept whole</span></div>
      </>}
      {inputs.map((l, i) => {
        const m = made?.inputs[i];
        return (
          <div className="kv" key={i}>
            <span>{i === 0 ? "Merged" : ""}</span>
            <span className="ellipsis"><span className="mono">{basename(l.path)}</span> × {l.strength}
              {m && m.matched < m.layers ? <span className="faint"> · {m.matched} of its {m.layers} layers</span> : null}</span>
          </div>
        );
      })}
      <div className="help" style={{ marginTop: 10 }}>
        {s.merge_rank
          ? <>What the LoRAs do together, each layer's merged change cut to rank {s.merge_rank} (its largest part). Kept is how much of
            the change it holds, by squared size. Compare it in the Sandbox with the LoRAs themselves at these strengths, and if it
            falls short, merge again at a higher rank.</>
          : <>What the LoRAs do together, exactly: their ranks added up. In the Sandbox it gives what they give together at these strengths.</>}
        {" "}Saved in <span className="mono">{job.dir}/output/</span>; it's in the Sandbox's LoRAs, and ComfyUI loads it from{" "}
        <span className="mono">models/loras</span>; use it at strength 1.
      </div>
    </div>
  );
}

// a training job's page shows one of these below its status, chosen by the tabs (and kept in the URL)
type View = "samples" | "loss" | "checkpoints" | "log";
const VIEWS: [View, string][] = [["log", "Log"], ["samples", "Samples"], ["loss", "Loss"], ["checkpoints", "Checkpoints"]];
const isView = (v?: string): v is View => VIEWS.some(([k]) => k === v);

function JobDetailView({ id, view, families }: { id: string; view?: string; families: Family[] }) {
  const toast = useToast();
  const [act, confirmRestart] = useJobAction();
  const [deleting, setDeleting] = useState(false);
  const [picked, setPicked] = useState<View | null>(isView(view) ? view : null);
  useEffect(() => { if (isView(view)) setPicked(view); }, [view]);
  const [job, reload] = usePoll<JobDetail>(`/api/jobs/${id}`, 2500);
  const metrics = useMetrics(id);
  const [light, setLight] = useState<string | null>(null);
  const [evaluate, setEvaluate] = useState(false);
  const [extend, setExtend] = useState(false);
  const [extractFrom, setExtractFrom] = useState<Checkpoint | null>(null);
  // the profile the job trained with, for Extract LoRA's other files (gone: only its checkpoints)
  const [profiles, , profilesErr] = usePoll<Profile[]>(extractFrom ? "/api/profiles" : null, 0);
  const isVideoFile = (f: string) => f.endsWith(".mp4");

  if (!job) return <div className="page"><div className="muted">Loading…</div></div>;
  const st = job.status_detail || ({} as JobDetail["status_detail"]);
  // the job's own step count (an edited or continued job shows its new one before it starts again)
  const total = job.config?.settings?.steps || st.total_steps || 0;
  const running = job.status === "running";
  const evaluating = !!job.config?.evaluate; // queued or running for a checkpoint evaluation
  const slider = !!job.config?.settings?.slider_enabled;
  const refmod = !!job.config?.settings?.refmod_enabled; // makes a RefMod: nothing is trained
  const finetune = !!job.config?.settings?.finetune_enabled; // trains the whole model: each checkpoint is one
  const extract = !!job.config?.settings?.extract_enabled; // extracts a LoRA from a finetune: nothing is trained
  const merge = !!job.config?.settings?.merge_enabled; // merges LoRAs into one: nothing is trained
  const once = refmod || extract || merge; // makes one file, made again rather than resumed, edited or continued
  const made = st.refmod;
  const jobPath = `${job.dir}/output/`;
  const checkpoints = st.checkpoints || [];
  // until a tab is picked: a job opens on its log, which shows what it's doing now (or why it failed);
  // a completed one opens on its samples, the result
  const tab: View = picked || (job.status === "completed" ? "samples" : "log");
  const pickTab = (v: View) => {
    setPicked(v);
    // in place: Back still leaves the job page rather than stepping through its tabs
    history.replaceState(null, "", `#/queue/${encodeURIComponent(id)}/${v}`);
  };
  const active = running || job.status === "stopping";
  const remove = async (files: boolean) => {
    try {
      await api.del(`/api/jobs/${job.id}?delete_files=${files}`);
      toast(`Deleted ${job.name}`);
      go("queue");
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  const family = families.find((f) => f.id === job.family);

  return (
    <div className="page">
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="small"><a href="#/queue">Queue</a> <span className="faint">/</span></div>
          <div className="row"><h1 className="ellipsis">{job.name}</h1><StatusPill status={job.status} /></div>
          <p>{[job.config?.profile_name, slider && `slider: ${job.config?.settings?.slider_target}`, refmod && "RefMod",
            !once && !finetune && job.config?.settings?.network_type === "lokr" && "LoKr", finetune && "finetune",
            extract && `LoRA from a finetune, rank ${job.config?.settings?.extract_rank}`,
            merge && `merged LoRA, ${job.config?.settings?.merge_rank ? `rank ${job.config.settings.merge_rank}` : "exact"}`,
            job.config?.datasets?.map((d: any) => d.name).join(", ")].filter(Boolean).join(" · ")} {job.assigned_gpu != null && `· GPU ${job.assigned_gpu}`}</p>
        </div>
        <div className="row">
          {running && !evaluating && <>
            {!once && <>
              <button className="btn" onClick={() => act(job, "sample")}><IconCamera /> Sample now</button>
              <button className="btn" onClick={() => act(job, "save")}><IconSave /> Save now</button>
            </>}
            <button className="btn" onClick={() => act({ ...job, refmod, extract, merge }, "stop", reload)}><IconStop /> Stop</button>
          </>}
          {job.status === "ready" && <button className="btn primary" title="Queue it to run when a GPU is free" onClick={() => act(job, "start", reload)}><IconPlay /> Start</button>}
          {job.status === "queued" && !evaluating && <button className="btn" onClick={() => act(job, "stop", reload)}><IconStop /> Cancel</button>}
          {job.status === "stopping" && <button className="btn danger" title="Stop it now, without waiting for it to save" onClick={() => act(job, "stop", reload)}>Force stop</button>}
          {evaluating && ["queued", "running"].includes(job.status) && (
            <button className="btn" onClick={() => act({ ...job, evaluating }, "stop", reload)}><IconStop /> {job.status === "queued" ? "Cancel evaluation" : "Stop evaluation"}</button>
          )}
          {!once && ["stopped", "failed", "interrupted"].includes(job.status) && job.resumable && (
            <button className="btn primary" onClick={() => act(job, "resume", reload)}><IconPlay /> Resume</button>
          )}
          {!once && job.status === "completed" && !evaluating && job.resumable && (
            <button className="btn" title="Train this job on for more steps, from where it finished" onClick={() => setExtend(true)}><IconPlay /> Continue training</button>
          )}
          {!once && !finetune && !running && !evaluating && !["queued", "stopping"].includes(job.status) && (st.checkpoints || []).length > 0 && (
            <button className="btn" title="Score the saved checkpoints with the checkpoint picker and/or concept bleed, and recommend one" onClick={() => setEvaluate(true)}><IconTarget /> Evaluate checkpoints</button>
          )}
          {!running && job.status !== "queued" && job.status !== "stopping" && (once
            ? <button className={`btn${job.status === "completed" ? "" : " primary"}`} onClick={() => act({ ...job, refmod, extract, merge }, "restart", reload)}><IconRefresh /> {refmod ? "Make again" : extract ? "Extract again" : "Merge again"}</button>
            // nothing saved to resume from: Restart is what's left
            : <button className={`btn${job.status !== "completed" && !job.resumable ? " primary" : ""}`} onClick={() => act(job, "restart", reload)}><IconRefresh /> Restart</button>)}
          {!running && job.status !== "stopping" && !evaluating && !extract && !merge && (
            <button className="btn" title="Change this job's settings" onClick={() => go("edit", job.id)}><IconEdit /> Edit</button>
          )}
          {!running && !extract && !merge && <button className="btn" onClick={() => cloneJob(job.id)}><IconCopy /> Duplicate</button>}
          {!active && <button className="btn icon danger" title="Delete" aria-label="Delete job" onClick={() => setDeleting(true)}><IconTrash /></button>}
        </div>
      </div>

      {job.status === "failed" && st.message && <div className="error-box" style={{ marginBottom: 16 }}>{st.message}</div>}

      {refmod ? <>
        <div className="stats" style={{ marginBottom: 16 }}>
          <div className="stat"><div className="k">Phase</div><div className="v" style={{ fontSize: 15 }}>{PHASES[st.phase || ""] || st.phase || "Queued"}</div></div>
          {job.config?.settings?.refmod_kind === "audio"
            ? <div className="stat"><div className="k">Of</div><div className="v" style={{ fontSize: 15 }}>Sound</div></div>
            : <div className="stat"><div className="k">Encoding</div><div className="v" style={{ fontSize: 15 }}>{job.config?.settings?.refmod_mode === "full" ? "Full" : "Compressed"}</div></div>}
          {job.config?.settings?.refmod_kind === "audio"
            ? <div className="stat"><div className="k">Length</div><div className="v">{made?.seconds != null ? `${made.seconds} s` : "—"}</div></div>
            : <div className="stat"><div className="k">References</div><div className="v">{made?.references ?? "—"}</div></div>}
          <div className="stat"><div className="k">Tokens</div><div className="v">{made ? made.tokens.toLocaleString() : "—"}</div></div>
          <div className="stat"><div className="k">File size</div><div className="v" style={{ fontSize: 15 }}>{made ? fmtSize(made.bytes) : "—"}</div></div>
          <div className="stat"><div className="k">Peak VRAM</div><div className="v">{st.vram_gb ? `${st.vram_gb} GB` : "—"}</div></div>
        </div>
        {running && st.message && <div className="faint small" style={{ marginBottom: 16 }}>{st.message}</div>}
        <RefModCard job={job} made={made} onOpen={setLight} />
        <Terminal jobId={job.id} live={running || job.status === "stopping"}
          alt={{ label: "Config", text: JSON.stringify(job.config, null, 2) }} />
      </> : extract ? <>
        <div className="stats" style={{ marginBottom: 16 }}>
          <div className="stat"><div className="k">Phase</div><div className="v" style={{ fontSize: 15 }}>{PHASES[st.phase || ""] || st.phase || "Queued"}</div></div>
          <div className="stat"><div className="k">Rank</div><div className="v">{job.config?.settings?.extract_rank}</div></div>
          <div className="stat"><div className="k">Kept</div><div className="v">{st.extract ? pct(st.extract.kept) : "—"}</div></div>
          <div className="stat"><div className="k">File size</div><div className="v" style={{ fontSize: 15 }}>{st.extract ? fmtSize(st.extract.bytes) : "—"}</div></div>
          <div className="stat"><div className="k">Peak VRAM</div><div className="v">{st.vram_gb ? `${st.vram_gb} GB` : "—"}</div></div>
        </div>
        {running && <div className="progress" style={{ marginBottom: 8 }}><div style={{ width: `${st.total_steps ? (100 * (st.step || 0)) / st.total_steps : 0}%` }} /></div>}
        {running && st.message && <div className="faint small" style={{ marginBottom: 16 }}>{st.message}</div>}
        <ExtractCard job={job} made={st.extract} />
        <Terminal jobId={job.id} live={running || job.status === "stopping"}
          alt={{ label: "Config", text: JSON.stringify(job.config, null, 2) }} />
      </> : merge ? <>
        <div className="stats" style={{ marginBottom: 16 }}>
          <div className="stat"><div className="k">Phase</div><div className="v" style={{ fontSize: 15 }}>{PHASES[st.phase || ""] || st.phase || "Queued"}</div></div>
          <div className="stat"><div className="k">Rank</div><div className="v" style={job.config?.settings?.merge_rank ? undefined : { fontSize: 15 }}>{job.config?.settings?.merge_rank || "Exact"}</div></div>
          <div className="stat"><div className="k">Kept</div><div className="v">{st.merge ? pct(st.merge.kept) : "—"}</div></div>
          <div className="stat"><div className="k">File size</div><div className="v" style={{ fontSize: 15 }}>{st.merge ? fmtSize(st.merge.bytes) : "—"}</div></div>
          <div className="stat"><div className="k">Peak VRAM</div><div className="v">{st.vram_gb ? `${st.vram_gb} GB` : "—"}</div></div>
        </div>
        {running && <div className="progress" style={{ marginBottom: 8 }}><div style={{ width: `${st.total_steps ? (100 * (st.step || 0)) / st.total_steps : 0}%` }} /></div>}
        {running && st.message && <div className="faint small" style={{ marginBottom: 16 }}>{st.message}</div>}
        <MergeCard job={job} made={st.merge} />
        <Terminal jobId={job.id} live={running || job.status === "stopping"}
          alt={{ label: "Config", text: JSON.stringify(job.config, null, 2) }} />
      </> : <>
      <div className="stats" style={{ marginBottom: 16 }}>
        <div className="stat"><div className="k">Phase</div><div className="v" style={{ fontSize: 15 }}>{PHASES[st.phase || ""] || st.phase || "Queued"}</div></div>
        <div className="stat"><div className="k">Step</div><div className="v">{st.step ?? 0}<span className="faint" style={{ fontSize: 13 }}> / {total}</span></div></div>
        <div className="stat"><div className="k">Avg loss</div><div className="v">{st.avg_loss?.toFixed(4) ?? "—"}</div></div>
        <div className="stat"><div className="k">Speed</div><div className="v">{st.it_per_sec ? fmtSpeed(1 / st.it_per_sec) : "—"}</div></div>
        <div className="stat"><div className="k">Time left</div><div className="v">{running ? fmtDuration(st.eta_seconds) : "—"}</div></div>
        <div className="stat"><div className="k">Peak VRAM</div><div className="v">{st.vram_gb ? `${st.vram_gb} GB` : "—"}</div></div>
        <div className="stat"><div className="k">LR</div><div className="v" style={{ fontSize: 15 }}>{st.lr?.toExponential(2) ?? "—"}</div></div>
      </div>
      <div className="progress" style={{ marginBottom: running && st.phase !== "training" && st.message ? 6 : 16 }}><div style={{ width: `${total ? (100 * (st.step || 0)) / total : 0}%` }} /></div>
      {running && st.phase !== "training" && st.message && <div className="faint small ellipsis" style={{ marginBottom: 16 }}>{st.message}</div>}

      <div className="view-tabs" role="tablist" aria-label="Job details">
        {VIEWS.map(([v, label]) => (
          <button key={v} role="tab" aria-selected={tab === v} className={tab === v ? "on" : ""} onClick={() => pickTab(v)}>
            {label}
            {v === "checkpoints" && checkpoints.length > 0 && <span className="count">{checkpoints.length}</span>}
            {v === "checkpoints" && evaluating && <span className="dot" title="A checkpoint evaluation is queued or running" />}
            {v === "log" && (running || job.status === "stopping") && <span className="dot live" title="Running" />}
            {v === "log" && job.status === "failed" && <span className="dot err" title="The job failed" />}
          </button>
        ))}
      </div>

      {tab === "samples" && <SamplesPanel jobId={job.id} samples={st.samples || []} slider={slider} datasets={(job.config?.datasets || []).map((d: { id: string }) => d.id)} />}

      {tab === "loss" && (
        <div className="card card-pad" style={{ marginBottom: 16 }}>
          <h2>Loss</h2>
          <LossChart points={metrics} totalSteps={total} />
        </div>
      )}
      {tab === "loss" && !!st.likeness?.length && (
        <div className="card card-pad" style={{ marginBottom: 16 }}>
          <h2>Likeness</h2>
          <p className="muted small" style={{ marginTop: -6 }}>How much the samples look like the training images: 0% is the base model's render of the same prompt, 100% as alike as the training images are to each other. The dashed line is the closest any sample came to one training image; in the band at the top it's a near copy.</p>
          <LikenessChart points={st.likeness} totalSteps={total} />
        </div>
      )}

      {tab === "log" && <Terminal jobId={job.id} live={running || job.status === "stopping"}
        alt={{ label: "Config", text: JSON.stringify(job.config, null, 2) }} />}

      {tab === "checkpoints" && <>
        <EvalPanel job={job} onOpen={setLight} />

        <div className="card card-pad" style={{ marginBottom: 16 }}>
          <h2>Checkpoints</h2>
          {checkpoints.length === 0 && <div className="muted small">None yet.</div>}
          {[...checkpoints].reverse().map((c) => (
            <div className="kv" key={c.file}>
              <span className="mono ellipsis">{c.file}</span>
              <span className="row" style={{ gap: 8 }}>
                <span className="faint small">step {c.step}</span>
                {job.recommended?.file === c.file && <span className="rec-star" title="Recommended by the checkpoint evaluation">★</span>}
                {c.final && <span className="pill completed">final</span>}
                {finetune && <button className="btn sm" title="Save what this checkpoint changed as a LoRA" onClick={() => setExtractFrom(c)}><IconLayers /> Extract LoRA</button>}
                <a className="btn sm" href={`/api/jobs/${job.id}/file/output/${c.file}`} download><IconDownload /></a>
              </span>
            </div>
          ))}
          {finetune ? (
            <div className="help" style={{ marginTop: 10 }}>Saved in <span className="mono">{jobPath}</span>. Each is a whole
              diffusion model in bf16, laid out like <span className="mono">{String(job.config?.profile?.diffusion_model || "").split(/[\\/]/).pop()}</span>: copy
              it into ComfyUI's <span className="mono">models/diffusion_models</span> and load it in place of the original, or try
              it in the Sandbox (Finetuned model). Extract LoRA saves what a checkpoint changed as a LoRA, a much smaller file
              to use on the original model.</div>
          ) : (
            <div className="help" style={{ marginTop: 10 }}>Saved in <span className="mono">{jobPath}</span>, in ComfyUI's format: copy them into <span className="mono">models/loras</span>.</div>
          )}
        </div>

        <DiskCard job={job} />
      </>}
      </>}

      {once && <DiskCard job={job} />}

      {light && <Lightbox src={light} video={isVideoFile(light)} onClose={() => setLight(null)}
        audio={light.endsWith(".mp3") ? { poster: light.replace(/\.mp3$/, "_poster.png") } : undefined} />}
      {evaluate && <EvaluateDialog job={job} onClose={() => setEvaluate(false)} onQueued={() => { reload(); pickTab("checkpoints"); }} />}
      {extend && <ContinueDialog job={job} onClose={() => setExtend(false)} onQueued={reload} />}
      {extractFrom && family && (profiles || profilesErr) && (
        <ExtractLora family={family} profile={(profiles && jobProfile(job.config, profiles)) || null}
          start={{ jobId: job.id, file: extractFrom.file }} dialog={{ onClose: () => setExtractFrom(null) }} />
      )}
      {deleting && (
        <ConfirmDialog title="Delete job" danger confirmLabel="Delete job" onClose={() => setDeleting(false)} onConfirm={remove}
          message={<>Remove <strong style={{ color: "var(--text)" }}>{job.name}</strong> from the queue?</>}
          checkbox={refmod ? "Also delete its RefMod and preview from disk" : extract || merge ? "Also delete its LoRA from disk"
            : finetune ? "Also delete its finetuned models and samples from disk" : "Also delete its LoRA checkpoints and samples from disk"} />
      )}
      {confirmRestart}
    </div>
  );
}

export default function Queue({ jobId, view, families }: { jobId: string; view?: string; families: Family[] }) {
  const [jobs, reload, err] = usePoll<Job[]>(jobId ? null : "/api/jobs", 2500, [jobId]);
  if (jobId) return <JobDetailView key={jobId} id={jobId} view={view} families={families} />;
  return <QueueList jobs={jobs} reload={reload} error={err} />;
}

type SharedCache = { total: number; in_use: number };
type QueueDisk = { jobs: Record<string, DiskUse>; total: DiskUse; shared_cache: SharedCache };

/** The cache of encoded captions and latents: how much there is, how much queued and running jobs use, and clearing
 * the rest (with finished jobs' own caches, from before the cache was shared). */
function CacheDialog({ disk, jobs, onClose, onCleared }: { disk: QueueDisk; jobs: Job[]; onClose: () => void; onCleared: () => void }) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const shared = disk.shared_cache;
  const own = jobs.filter((j) => FINISHED.includes(j.status) && !j.evaluating).reduce((n, j) => n + (disk.jobs[j.id]?.cache || 0), 0);
  const clearable = shared.total - shared.in_use + own;
  const clear = async () => {
    setBusy(true);
    try {
      const r = await api.post<{ freed: number }>("/api/disk/clean", { parts: ["cache"] });
      toast(`Freed ${fmtSize(r.freed)}`);
      onCleared();
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Cache" onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Close</button>
        <button className="btn primary" autoFocus disabled={busy || clearable <= 0} onClick={clear}>
          <IconTrash /> {clearable > 0 ? `Clear ${fmtSize(clearable)}` : "Nothing to clear"}
        </button>
      </>
    }>
      <div className="stack" style={{ gap: 12 }}>
        <p className="muted" style={{ margin: 0 }}>
          Captions and training files are encoded once and kept here. Jobs share them: a job encoding the same files the same
          way reuses them instead of encoding them again.
        </p>
        <div>
          <div className="kv"><span>Cached captions and latents</span><span>{fmtSize(disk.total.cache + shared.total)}</span></div>
          <div className="kv"><span>Used by queued or running jobs</span><span>{fmtSize(shared.in_use)}</span></div>
          <div className="kv"><span>Can be cleared</span><span>{fmtSize(clearable)}</span></div>
        </div>
        <p className="help" style={{ margin: 0 }}>
          Clearing deletes what no queued or running job uses; a job that needs it again encodes it again first. LoRAs,
          samples and training state stay. Training state ({fmtSize(disk.total.state)} now) is what a job resumes or
          continues from; a completed job's can be deleted on its page, under Checkpoints.
        </p>
      </div>
    </Modal>
  );
}

function QueueList({ jobs, reload, error }: { jobs: Job[] | null; reload: () => void; error: string | null }) {
  const toast = useToast();
  const [disk, reloadDisk] = usePoll<QueueDisk>("/api/disk", 0, [jobs?.length]);
  const [manage, setManage] = useState(false);
  const shared = disk?.shared_cache || { total: 0, in_use: 0 };
  const running = jobs?.filter((j) => j.status === "running" || j.status === "stopping") || [];
  const queued = jobs?.filter((j) => j.status === "queued") || [];
  // added and waiting for Start, oldest first (Start all queues them in that order)
  const ready = (jobs?.filter((j) => j.status === "ready") || []).sort((a, b) => a.created - b.created);
  const [starting, setStarting] = useState(false);
  const startAll = async () => {
    setStarting(true);
    let started = 0;
    const failed: string[] = [];
    for (const j of ready) {
      try {
        await api.post(`/api/jobs/${j.id}/action`, { action: "start" });
        started++;
      } catch (e: any) {
        failed.push(`${j.name}: ${e.message}`);
      }
    }
    toast(`Started ${started} job${started === 1 ? "" : "s"}${failed.length ? `; couldn't start ${failed.join("; ")}` : ""}`, failed.length > 0);
    setStarting(false);
    reload();
  };
  // Select: several jobs deleted at once (not running ones)
  const sel = useSelection();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const deletable = (jobs || []).filter((j) => !running.includes(j)).map((j) => j.id);
  const chosen = (jobs || []).filter((j) => sel.picked.has(j.id) && deletable.includes(j.id));
  const removeMany = async (files: boolean) => {
    setDeleting(true);
    let deleted = 0;
    const failed: string[] = [];
    for (const j of chosen) {
      try {
        await api.del(`/api/jobs/${j.id}?delete_files=${files}`);
        deleted++;
      } catch (e: any) {
        failed.push(`${j.name}: ${e.message}`);
      }
    }
    toast(`Deleted ${deleted} job${deleted === 1 ? "" : "s"}${failed.length ? `; couldn't delete ${failed.join("; ")}` : ""}`, failed.length > 0);
    setDeleting(false);
    sel.stop();
    reload();
    reloadDisk();
  };
  const queuedChosen = chosen.filter((j) => j.status === "queued").length;
  const done = (jobs?.filter((j) => !["running", "stopping", "queued", "ready"].includes(j.status)) || []).sort((a, b) => (b.finished || b.created) - (a.finished || a.created));
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Queue{!!jobs?.length && <span className="count-pill">{jobs.length}</span>}</h1>
          <p>Training, RefMods, merges and extracted LoRAs, run one at a time per GPU in this order.</p>
        </div>
        <div className="row">
          {!!jobs?.length && !sel.on && <button className="btn lg" onClick={sel.start}><IconCheck /> Select</button>}
        </div>
      </div>
      {error && !jobs && <LoadError what="the jobs" error={error} onRetry={reload} />}
      {jobs && jobs.length === 0 && (
        <Empty icon={<IconServer />} title="No jobs yet" action={<a className="btn primary lg" href="#/create"><IconPlus /> New job</a>}>
          Make one with New job: it waits here until you press Start, then for a free GPU, and its samples, loss
          and log show on its page.
        </Empty>
      )}
      {sel.on && (
        <SelectBar sel={sel} ids={deletable} noun={["job", "jobs"]}
          info={<span className="muted small">{fmtSize(chosen.reduce((n, j) => n + (disk?.jobs[j.id]?.total || 0), 0))} on disk</span>}>
          <button className="btn danger" onClick={() => setConfirmDelete(true)} disabled={!chosen.length || deleting}><IconTrash /> {deleting ? "Deleting…" : "Delete"}</button>
        </SelectBar>
      )}
      {confirmDelete && (
        <ConfirmDialog title={`Delete ${chosen.length} job${chosen.length === 1 ? "" : "s"}`} danger confirmLabel="Delete"
          onClose={() => setConfirmDelete(false)} onConfirm={removeMany}
          message={<>Remove <strong style={{ color: "var(--text)" }}>{someNames(chosen.map((j) => j.name))}</strong> from the queue?
            {queuedChosen > 0 && ` ${queuedChosen === chosen.length ? (queuedChosen === 1 ? "It's" : "They're") : `${queuedChosen} of them ${queuedChosen === 1 ? "is" : "are"}`} queued, and won't run.`}</>}
          checkbox={`Also delete their files from disk (${fmtSize(chosen.reduce((n, j) => n + (disk?.jobs[j.id]?.total || 0), 0))}): LoRAs or finetuned models, samples and training state`} />
      )}
      {[["In progress", running], ["Ready to start", ready], ["Queued", queued], ["Finished", done]].map(([title, list]) =>
        (list as Job[]).length > 0 && (
          <div key={title as string} className="job-group">
            <div className="section-title row">
              <span>{title as string} · {(list as Job[]).length}</span>
              {list === ready && ready.length > 1 && !sel.on && <>
                <span className="spacer" />
                <button className="btn sm primary" disabled={starting} onClick={startAll}><IconPlay /> {starting ? "Starting…" : "Start all"}</button>
              </>}
            </div>
            <div className="stack" style={{ gap: 10 }}>
              {(list as Job[]).map((j, i) => <JobRow key={j.id} j={j} reload={reload} queuedIndex={i} queuedCount={(list as Job[]).length} size={disk?.jobs[j.id]?.total} sel={sel} />)}
            </div>
          </div>
        ),
      )}
      {disk && (
        <div className="card disk-bar">
          <span className="disk-item" title="Everything the jobs keep on disk: LoRAs and finetuned models, samples, training state, logs, and the cache">
            <IconDrive /><span>Storage</span><b>{fmtSize(disk.total.total + shared.total)}</b>
          </span>
          <span className="disk-item" title="Encoded captions and latents, shared between jobs that encode the same files the same way">
            <IconDatabase /><span>Cache</span><b>{fmtSize(disk.total.cache + shared.total)}</b>
          </span>
          <span className="disk-item" title="What the jobs made: LoRAs, finetuned models and RefMods">
            <IconFile /><span>Outputs</span><b>{fmtSize(disk.total.output)}</b>
          </span>
          <span className="spacer" />
          <button className="btn" onClick={() => setManage(true)}>Manage cache <IconRight /></button>
        </div>
      )}
      {manage && disk && <CacheDialog disk={disk} jobs={jobs || []} onClose={() => setManage(false)} onCleared={reloadDisk} />}
    </div>
  );
}

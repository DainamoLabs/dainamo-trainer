import { useEffect, useState } from "react";
import { go } from "../App";
import { api, fmtDuration, fmtTime, q, type Checkpoint, type DiskUse, type ExtractMade, type Job, type JobDetail, type JobDiskUse, type MergeMade, type RefModMade } from "../api";
import { EvalPanel } from "../components/EvalPanel";
import { EvaluateDialog } from "../components/EvalSettings";
import { HardwareCards } from "../components/Hardware";
import { LossChart, Spark, useMetrics } from "../components/LossChart";
import { IconCamera, IconCopy, IconDown, IconDownload, IconEdit, IconImages, IconLayers, IconPlay, IconRefresh, IconSave, IconStop, IconTarget, IconTrash, IconUp } from "../components/icons";
import { AudioCard, ConfirmDialog, Lightbox, Modal, StatusPill, fmtSize, usePoll, useToast } from "../components/ui";
import { SamplesPanel } from "../components/Samples";
import { Terminal } from "../components/Terminal";
import { CLONE_KEY, formOf } from "./CreateJob";

export const PHASES: Record<string, string> = {
  starting: "Starting", preparing: "Scanning datasets", caching_text: "Encoding captions", caching_latents: "Encoding training files",
  loading: "Loading model", rendering: "Rendering training images", training: "Training", sampling: "Sampling", evaluating: "Evaluating checkpoint", completed: "Completed", stopped: "Stopped", failed: "Failed",
  encoding: "Making the RefMod", extracting: "Extracting the LoRA", merging: "Merging the LoRAs",
};

function useJobAction() {
  const toast = useToast();
  // a RefMod job (``refmod``), an extracted LoRA's (``extract``) or a merged one's (``merge``) is made again from the
  // start: "restart" asks about the file it made
  return async (job: { id: string; name: string; evaluating?: boolean; refmod?: boolean; extract?: boolean; merge?: boolean }, action: string, after?: () => void) => {
    if (action === "restart" && !confirm(job.refmod ? `Make "${job.name}" again? The RefMod it made and its preview will be deleted.`
      : job.extract ? `Extract "${job.name}" again? The LoRA it made will be deleted.`
      : job.merge ? `Merge "${job.name}" again? The LoRA it made will be deleted.`
      : `Restart "${job.name}" from step 0? Its checkpoints and samples will be deleted.`)) return;
    try {
      await api.post(`/api/jobs/${job.id}/action`, { action });
      const again = job.refmod ? "Queued to be made again" : job.extract ? "Queued to be extracted again" : job.merge ? "Queued to be merged again" : "";
      const msg: Record<string, string> = { stop: job.evaluating ? "Stopping the checkpoint evaluation" : again ? "Stopping" : "Stopping — saving state…", sample: "Sampling requested", save: "Checkpoint requested", resume: again || "Re-queued (resumes from last save)", restart: again || "Restarted" };
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
          job encoding the same files the same way reuses them; Clear caches on the Queue page frees what no queued or running job uses.
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
          {j.refmod && <span>RefMod</span>}
          {j.extract && <span>LoRA from a finetune</span>}
          {j.merge && <span>merged LoRA</span>}
          {j.network === "lokr" && !j.refmod && <span>LoKr</span>}
          {j.network === "full" && <span>finetune</span>}
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
        {!(active && j.evaluating) && (active || ((p.step ?? 0) > 0 && !j.extract && !j.merge)) && (
          <div style={{ marginTop: 10 }}>
            <div className="row small" style={{ marginBottom: 5 }}>
              <span>{active ? PHASES[p.phase || ""] || p.phase : `Step ${p.step}`}{active && p.phase === "training" ? ` · step ${p.step} / ${p.total_steps}` : ""}</span>
              <span className="spacer" />
              <Spark values={j.loss_curve} className="job-spark" />
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
        {["stopped", "failed", "interrupted"].includes(j.status) && <button className="btn primary" onClick={() => act(j, "resume", reload)}><IconPlay /> {j.refmod ? "Make again" : j.extract ? "Extract again" : j.merge ? "Merge again" : "Resume"}</button>}
        {!active && j.status !== "queued" && !j.extract && !j.merge && <button className="btn icon" title="Duplicate settings into a new job" onClick={() => cloneJob(j.id)}><IconCopy /></button>}
        {!active && <button className="btn icon danger" title="Delete" onClick={() => setConfirmDel(true)}><IconTrash /></button>}
      </div>
      {confirmDel && (
        <ConfirmDialog title="Delete job" danger confirmLabel="Delete job" onClose={() => setConfirmDel(false)} onConfirm={remove}
          message={<>Remove <strong style={{ color: "var(--text)" }}>{j.name}</strong> from the queue?</>}
          checkbox={j.refmod ? "Also delete its RefMod and preview from disk" : j.extract || j.merge ? "Also delete its LoRA from disk"
            : j.network === "full" ? "Also delete its finetuned models and samples from disk"
            : "Also delete its LoRA checkpoints and samples from disk"} />
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
          <div className="muted small">The current final {job.config?.settings?.finetune_enabled ? "model" : "LoRA"} is kept as <span className="mono">{kept}</span>, and "keep last N" never deletes it. The new end saves as <span className="mono">{final.file}</span>.</div>
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

const EXTRACT_RANKS = [16, 32, 64, 128, 256, 512];
const pct = (x: number) => `${(100 * x).toFixed(x >= 0.9995 ? 0 : 1)}%`;
const basename = (p: string) => String(p || "").split(/[\\/]/).pop() || "";

/** Extract a LoRA from one of a finetune's checkpoints: a job of its own, queued, so its LoRA is listed with it. */
function ExtractDialog({ job, checkpoint, onClose }: { job: JobDetail; checkpoint: Checkpoint; onClose: () => void }) {
  const toast = useToast();
  const [rank, setRank] = useState(128);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  type Plan = { sizes: Record<string, number>; base: string; base_found: boolean };
  const [plan, , err] = usePoll<Plan>(`/api/jobs/${job.id}/extract?file=${q(checkpoint.file)}`, 0, [checkpoint.file]);
  const auto = `${checkpoint.file.replace(/\.safetensors$/, "")}_r${rank}`;
  const start = async () => {
    setBusy(true);
    try {
      const made = await api.post<Job>(`/api/jobs/${job.id}/extract`, { file: checkpoint.file, rank, name: name.trim() });
      toast("Queued to extract the LoRA");
      onClose();
      go("queue", made.id);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Extract a LoRA" onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy || (!!plan && !plan.base_found)} onClick={start}><IconLayers /> Extract</button>
      </>
    }>
      <div className="stack" style={{ gap: 12 }}>
        <div className="muted small">
          What <span className="mono">{checkpoint.file}</span> (step {checkpoint.step}) changed from the model it was finetuned
          from, <span className="mono">{plan ? basename(plan.base) : "…"}</span>, saved as a LoRA to use on that model.
        </div>
        {err && <div className="error-box small">{err}</div>}
        {plan && !plan.base_found && <div className="error-box small">{plan.base || "Its profile's diffusion model"} isn't there any more, and the LoRA is the change from it.</div>}
        <label className="field">
          <span>Rank</span>
          <select value={rank} onChange={(e) => setRank(Number(e.target.value))}>
            {EXTRACT_RANKS.map((r) => <option key={r} value={r}>{r}{plan ? ` · about ${fmtSize(plan.sizes[r])}` : ""}</option>)}
          </select>
          <span className="help">Each weight matrix's change is kept at this rank: its largest part. A higher rank keeps more of
            what the finetune learned, in a bigger file. Changes to norms and biases are kept whole at any rank (they're small).
            The job's page says how much of the change the LoRA keeps.</span>
        </label>
        <label className="field">
          <span>Name</span>
          <input value={name} placeholder={auto} onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="help">It runs in the queue and needs little VRAM: the two models are compared a block at a time, which
          takes a few minutes when a model is on a slow disk.</div>
      </div>
    </Modal>
  );
}

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
const VIEWS: [View, string][] = [["samples", "Samples"], ["loss", "Loss"], ["checkpoints", "Checkpoints"], ["log", "Log"]];
const isView = (v?: string): v is View => VIEWS.some(([k]) => k === v);

function JobDetailView({ id, view }: { id: string; view?: string }) {
  const act = useJobAction();
  const [picked, setPicked] = useState<View | null>(isView(view) ? view : null);
  useEffect(() => { if (isView(view)) setPicked(view); }, [view]);
  const [job, reload] = usePoll<JobDetail>(`/api/jobs/${id}`, 2500);
  const metrics = useMetrics(id);
  const [light, setLight] = useState<string | null>(null);
  const [evaluate, setEvaluate] = useState(false);
  const [extend, setExtend] = useState(false);
  const [extractFrom, setExtractFrom] = useState<Checkpoint | null>(null);
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
  // until a tab is picked: a failed job opens on its log, which says why
  const tab: View = picked || (["failed", "interrupted"].includes(job.status) ? "log" : "samples");
  const pickTab = (v: View) => {
    setPicked(v);
    // in place: Back still leaves the job page rather than stepping through its tabs
    history.replaceState(null, "", `#/queue/${encodeURIComponent(id)}/${v}`);
  };

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
          {evaluating && ["queued", "running"].includes(job.status) && (
            <button className="btn" onClick={() => act({ ...job, evaluating }, "stop", reload)}><IconStop /> {job.status === "queued" ? "Cancel evaluation" : "Stop evaluation"}</button>
          )}
          {!once && ["stopped", "failed", "interrupted"].includes(job.status) && <button className="btn primary" onClick={() => act(job, "resume", reload)}><IconPlay /> Resume</button>}
          {!once && job.status === "completed" && !evaluating && job.resumable && (
            <button className="btn" title="Train this job on for more steps, from where it finished" onClick={() => setExtend(true)}><IconPlay /> Continue training</button>
          )}
          {!once && !finetune && !running && !evaluating && !["queued", "stopping"].includes(job.status) && (st.checkpoints || []).length > 0 && (
            <button className="btn" title="Score the saved checkpoints with the checkpoint picker and/or concept bleed, and recommend one" onClick={() => setEvaluate(true)}><IconTarget /> Evaluate checkpoints</button>
          )}
          {!running && job.status !== "queued" && job.status !== "stopping" && (once
            ? <button className={`btn${job.status === "completed" ? "" : " primary"}`} onClick={() => act({ ...job, refmod, extract, merge }, "restart", reload)}><IconRefresh /> {refmod ? "Make again" : extract ? "Extract again" : "Merge again"}</button>
            : <button className="btn" onClick={() => act(job, "restart", reload)}><IconRefresh /> Restart</button>)}
          {!running && job.status !== "stopping" && !evaluating && !extract && !merge && (
            <button className="btn" title="Change this job's settings" onClick={() => go("edit", job.id)}><IconEdit /> Edit</button>
          )}
          {!running && !extract && !merge && <button className="btn" onClick={() => cloneJob(job.id)}><IconCopy /> Duplicate</button>}
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
        <div className="stat"><div className="k">Speed</div><div className="v">{st.it_per_sec ? `${(1 / st.it_per_sec).toFixed(2)}s` : "—"}<span className="faint" style={{ fontSize: 13 }}> /it</span></div></div>
        <div className="stat"><div className="k">ETA</div><div className="v">{running ? fmtDuration(st.eta_seconds) : "—"}</div></div>
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

      {tab === "samples" && <SamplesPanel jobId={job.id} samples={st.samples || []} slider={slider} />}

      {tab === "loss" && (
        <div className="card card-pad" style={{ marginBottom: 16 }}>
          <h2>Loss</h2>
          <LossChart points={metrics} totalSteps={total} />
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
            <div className="help" style={{ marginTop: 10 }}>Saved in <span className="mono">{jobPath}</span> — ComfyUI-compatible; copy into <span className="mono">models/loras</span>.</div>
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
      {extractFrom && <ExtractDialog job={job} checkpoint={extractFrom} onClose={() => setExtractFrom(null)} />}
    </div>
  );
}

export default function Queue({ jobId, view }: { jobId: string; view?: string }) {
  const [jobs, reload] = usePoll<Job[]>("/api/jobs", 2500);
  if (jobId) return <JobDetailView key={jobId} id={jobId} view={view} />;
  return <QueueList jobs={jobs} reload={reload} />;
}

function QueueList({ jobs, reload }: { jobs: Job[] | null; reload: () => void }) {
  const toast = useToast();
  type SharedCache = { total: number; in_use: number };
  const [disk, reloadDisk] = usePoll<{ jobs: Record<string, DiskUse>; total: DiskUse; shared_cache: SharedCache }>("/api/disk", 0, [jobs?.length]);
  const [confirmClean, setConfirmClean] = useState(false);
  const shared = disk?.shared_cache || { total: 0, in_use: 0 };
  // what clearing would free: the shared cache but what queued and running jobs use, and finished jobs' own caches
  const clearable = shared.total - shared.in_use + (jobs || []).filter((j) => FINISHED.includes(j.status) && !j.evaluating)
    .reduce((n, j) => n + (disk?.jobs[j.id]?.cache || 0), 0);
  const clean = async () => {
    try {
      const r = await api.post<{ freed: number; jobs: number }>("/api/disk/clean", { parts: ["cache"] });
      toast(`Freed ${fmtSize(r.freed)}`);
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
      </div>
      <HardwareCards />
      {disk && disk.total.total + shared.total > 0 && (
        <div className="row wrap small queue-disk">
          <span className="muted">Jobs use {fmtSize(disk.total.total + shared.total)} on disk: {fmtSize(disk.total.output)} of LoRAs, {fmtSize(shared.total + disk.total.cache)} of
            cached captions and latents (shared between jobs that encode them the same way), {fmtSize(disk.total.state)} of training state.</span>
          {clearable > 0 && <button className="btn sm" onClick={() => setConfirmClean(true)}>Clear caches</button>}
        </div>
      )}
      {confirmClean && (
        <ConfirmDialog title="Clear caches" confirmLabel={`Free ${fmtSize(clearable)}`} onClose={() => setConfirmClean(false)} onConfirm={clean}
          message="The encoded captions and images that no queued or running job is using are deleted. A job that needs them again encodes them again first. Checkpoints, samples and training state stay." />
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

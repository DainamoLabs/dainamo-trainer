import { useEffect, useMemo, useState } from "react";
import { go } from "../App";
import { api, q, type Family, type Job, type ModelOption, type Profile } from "../api";
import { IconClock, IconQueueAdd, IconRight, IconUpload, IconX } from "./icons";
import { Modal, PathInput, fmtSize, usePoll, useToast } from "./ui";

const EXTRACT_RANKS = [16, 32, 64, 128, 256, 512];
export const fileBase = (p: string) => String(p || "").split(/[\\/]/).pop() || "";
export const stem = (p: string) => fileBase(p).replace(/\.safetensors$/, "");
export const outName = (name: string) => name.trim().replace(/\.safetensors$/i, ""); // the field shows the extension after it

/** A form's output name: a file name, the .safetensors after it shown, not typed. */
export function NameField({ value, placeholder, onChange }: { value: string; placeholder: string; onChange: (v: string) => void }) {
  return (
    <label className="field">
      <span>Output name</span>
      <div className="input-suffix">
        <input value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
        <span>.safetensors</span>
      </div>
    </label>
  );
}

/** How a form's job works, folded away under its fields. */
export function How({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="me-how">
      <summary><IconRight />{title}</summary>
      <div className="me-how-body">{children}</div>
    </details>
  );
}

/** The bottom of a form that queues a job: Add to queue, and why it can't be pressed yet. */
export function Submit({ busy, why, onClick }: { busy: boolean; why: string; onClick: () => void }) {
  return (
    <div className="me-submit">
      <div className="me-queued"><IconClock />Runs as a queued job</div>
      <button className="btn primary lg" disabled={busy || !!why} onClick={onClick}><IconQueueAdd /> Add to queue</button>
      {why && <div className="help">{why}</div>}
    </div>
  );
}

/** ``value``, once it has stopped changing for ``ms`` (paths typed a letter at a time aren't checked each time). */
export function useSettled<T>(value: T, ms = 400): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [JSON.stringify(value)]);
  return v;
}

/** Options grouped by job, newest job first, for a <select>. */
export function byJob<T extends { job_name: string }>(opts: T[] | null): [string, T[]][] {
  const out = new Map<string, T[]>();
  for (const o of opts || []) out.set(o.job_name, [...(out.get(o.job_name) || []), o]);
  return [...out];
}

type ExtractPlan = { sizes: Record<string, number>; base: string; base_found: boolean; base_error?: string };

/** Extract a LoRA: what a finetuned model changed from its base model, saved as a LoRA, queued as a job of its own.
 * The one form, on Models' Merge & extract (``profile``: the profile picked there) and in a finetune job's page, as
 * a dialog (``dialog``) opened on one of its checkpoints (``start``). A finetune job's checkpoint is extracted as
 * the job's own (linked to it, from the model its profile had, with its trigger words); any other file, or one with
 * a base model set, against ``profile``'s model. Without a ``profile`` (the job's is gone), only its checkpoints. */
export function ExtractLora({ family, profile, start, dialog }: {
  family: Family; profile: Profile | null; start?: { jobId: string; file: string }; dialog?: { onClose: () => void };
}) {
  const toast = useToast();
  const [models] = usePoll<ModelOption[]>(family.finetune ? `/api/sandbox/models?family=${family.id}` : null, 0, [family.id]);
  const [tuned, setTuned] = useState("");
  const [custom, setCustom] = useState(!family.finetune);
  const [original, setOriginal] = useState("");
  const [rank, setRank] = useState(128);
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [busy, setBusy] = useState(false);
  // the checkpoint it was opened on, once the list of them has loaded
  useEffect(() => {
    const m = start && models?.find((o) => o.job_id === start.jobId && o.file === start.file);
    if (m && !tuned) setTuned(m.path);
  }, [models]);
  const settled = useSettled({ tuned: tuned.trim(), original: original.trim() });
  const jobs = useMemo(() => byJob(models), [models]);
  // a finetune job's checkpoint, extracted as its job's own (linked to it, with its trigger words)
  const ckpt = !settled.original ? models?.find((m) => m.path === settled.tuned) : undefined;
  const planUrl = !settled.tuned ? null
    : ckpt ? `/api/jobs/${ckpt.job_id}/extract?file=${q(ckpt.file)}`
    : profile ? `/api/tools/extract/sizes?profile_id=${profile.id}&path=${q(settled.tuned)}&base=${q(settled.original)}` : null;
  // fresh: the check of the paths as they are now, none once the finetuned model is cleared
  const [plan, , err] = usePoll<ExtractPlan>(planUrl, 0, [], true);
  const auto = tuned.trim() ? `${stem(tuned.trim())}_r${rank}` : "extracted";
  const why = !tuned.trim() ? "Choose a finetuned model to continue."
    : !ckpt && !profile && settled.tuned === tuned.trim() ? "Its job's profile is gone: choose one of the finetune's checkpoints."
    : err && settled.tuned ? "Fix the problem above to continue."
    : !plan || settled.tuned !== tuned.trim() || settled.original !== original.trim() ? "Checking the model…"
    : !plan.base_found ? "Fix the base model to continue."
    : "";
  const baseError = plan && !plan.base_found
    ? plan.base_error || `${plan.base || "Its profile's diffusion model"} isn't there any more, and the LoRA is the change from it.` : "";

  const queue = async () => {
    setBusy(true);
    try {
      const made = ckpt
        ? await api.post<Job>(`/api/jobs/${ckpt.job_id}/extract`, { file: ckpt.file, rank, name: outName(name) })
        : await api.post<Job>("/api/tools/extract", { profile_id: profile!.id, tuned: tuned.trim(), base: original.trim(), rank, name: outName(name), trigger });
      toast("Queued to extract the LoRA");
      dialog?.onClose();
      go("queue", made.id);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };

  const fields = <>
    <div className="field">
      <span>Finetuned model</span>
      {custom || (tuned && models && !models.some((m) => m.path === tuned)) ? (
        <div className="sbx-lora no-strength">
          <PathInput value={tuned} onChange={setTuned} exts=".safetensors" placeholder="/path/to/finetuned_model.safetensors" />
          {family.finetune && <button className="btn ghost icon" title="Choose a finetune's checkpoint" aria-label="Choose a finetune's checkpoint"
            onClick={() => { setTuned(""); setCustom(false); }}><IconX /></button>}
        </div>
      ) : (
        <select value={tuned} aria-label="Finetuned model" onChange={(e) => e.target.value === "__custom" ? (setTuned(""), setCustom(true)) : setTuned(e.target.value)}>
          <option value="">Choose a finetune's checkpoint…</option>
          {jobs.map(([job, opts]) => (
            <optgroup key={job} label={`${job} · finetune`}>
              {opts.map((o) => <option key={o.path} value={o.path}>{o.final ? "final" : `step ${o.step}`} · {o.file}</option>)}
            </optgroup>
          ))}
          {profile && <option value="__custom">Other file…</option>}
        </select>
      )}
    </div>
    {profile && (
      <label className="field" onClick={(e) => e.preventDefault()}>
        <span>Base model file <span className="faint">· optional</span></span>
        <PathInput value={original} onChange={setOriginal} exts=".safetensors" placeholder={fileBase(String(profile.data?.diffusion_model || ""))} />
        <span className="help">The model it was finetuned from, of the same kind and layout. Blank: {ckpt ? "the one its job trained from" : "the profile's"}.</span>
      </label>
    )}
    {err && settled.tuned && <div className="error-box small">{err}</div>}
    {baseError && <div className="error-box small">{baseError}</div>}
    <div className="me-pair">
      <label className="field">
        <span>Output rank</span>
        <select value={rank} onChange={(e) => setRank(Number(e.target.value))}>
          {EXTRACT_RANKS.map((r) => <option key={r} value={r}>{r}{plan?.sizes?.[r] ? ` · about ${fmtSize(plan.sizes[r])}` : ""}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Trigger words <span className="faint">· optional</span></span>
        {ckpt
          ? <input disabled placeholder="Its job's" title="A finetune job's checkpoint gets its job's trigger words" />
          : <input value={trigger} onChange={(e) => setTrigger(e.target.value)} />}
      </label>
    </div>
    <span className="help me-pair-help">Higher ranks keep more of what the finetune learned, in a bigger file.</span>
    <NameField value={name} placeholder={auto} onChange={setName} />
    <How title="How extraction works">
      <p>Each weight matrix's change is kept at the rank: its largest part. Changes to norms and biases are kept whole (they're
        small). The base model has to be the one the finetune started from, the same kind of model laid out the same way.
        The job's page says how much of the change the LoRA keeps.</p>
      <p>A finetune job's checkpoint is extracted as its job's own: linked to the job, with its trigger words. Trigger words
        given here are saved in the LoRA, as a trained one's are.</p>
      <p>It runs in the queue and needs little VRAM: the two models are compared a block at a time, which takes minutes, longer
        from a slow disk.</p>
    </How>
  </>;

  if (dialog) {
    return (
      <Modal title="Extract a LoRA" onClose={dialog.onClose} footer={<>
        {why && <span className="help modal-why">{why}</span>}
        <button className="btn ghost" onClick={dialog.onClose}>Cancel</button>
        <button className="btn primary" disabled={busy || !!why} onClick={queue}><IconQueueAdd /> Add to queue</button>
      </>}>
        <div className="stack me-dialog" style={{ gap: 14 }}>
          <p className="muted small" style={{ margin: 0 }}>Save what a finetuned model changed from its base model as a LoRA to use on that model.</p>
          {fields}
        </div>
      </Modal>
    );
  }
  return (
    <div className="card me-card">
      <div className="me-head">
        <IconUpload />
        <div>
          <h3>Extract a LoRA</h3>
          <p>Save what a finetuned model changed from its base model as a LoRA.</p>
        </div>
      </div>
      {fields}
      <Submit busy={busy} why={why} onClick={queue} />
    </div>
  );
}

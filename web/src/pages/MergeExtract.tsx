import { useEffect, useMemo, useState } from "react";
import { api, q, type Family, type Job, type LoraOption, type ModelOption, type Profile } from "../api";
import { go } from "../App";
import { IconLayers, IconPlus, IconX } from "../components/icons";
import { Info, PathInput, ProfileOptions, fmtSize, usePoll, useToast } from "../components/ui";

const MERGE_RANKS = [4, 8, 16, 32, 64, 128, 256, 512];
const EXTRACT_RANKS = [16, 32, 64, 128, 256, 512];
const PROFILE_KEY = "dainamo.tools.profile";
const base = (p: string) => String(p || "").split(/[\\/]/).pop() || "";
const stem = (p: string) => base(p).replace(/\.safetensors$/, "");

/** ``value``, once it has stopped changing for ``ms`` (paths typed a letter at a time aren't checked each time). */
function useSettled<T>(value: T, ms = 400): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [JSON.stringify(value)]);
  return v;
}

/** Options grouped by job, newest job first, for a <select>. */
function byJob<T extends { job_name: string }>(opts: T[] | null): [string, T[]][] {
  const out = new Map<string, T[]>();
  for (const o of opts || []) out.set(o.job_name, [...(out.get(o.job_name) || []), o]);
  return [...out];
}

/** Merge & extract (a tab of Profiles): LoRAs merged into one, or a LoRA extracted from a finetuned model, for one
 * profile's model. Each is queued as a job of its own, so it shows its progress there and its LoRA is listed with it,
 * in the Sandbox too. */
export default function MergeExtract({ families }: { families: Family[] }) {
  const [profiles] = usePoll<Profile[]>("/api/profiles", 0);
  const [pid, setPid] = useState(() => { try { return localStorage.getItem(PROFILE_KEY) || ""; } catch { return ""; } });
  const profile = profiles?.find((p) => p.id === pid) || profiles?.[0];
  const family = families.find((f) => f.id === profile?.family);
  const pick = (id: string) => {
    setPid(id);
    try { localStorage.setItem(PROFILE_KEY, id); } catch { /* not kept */ }
  };

  if (profiles && !profiles.length) {
    return <div className="empty"><strong>No profiles yet</strong>Merging and extracting work on a profile's model: create one first.</div>;
  }
  return (
    <div className="stack" style={{ gap: 16 }}>
      <label className="field" style={{ maxWidth: 520 }}>
        <span>Profile</span>
        <select value={profile?.id || ""} onChange={(e) => pick(e.target.value)}>
          {profiles && <ProfileOptions profiles={profiles} families={families} />}
        </select>
        <span className="help">The model the LoRAs are for: their layers are checked against its diffusion model,
          {" "}<span className="mono">{base(String(profile?.data?.diffusion_model || "")) || "…"}</span>, and what's made is listed with its family's LoRAs.</span>
      </label>
      {profile && family && (
        <div className="grid cols-2" style={{ alignItems: "start" }}>
          <MergeForm key={`m${profile.id}`} profile={profile} family={family} />
          <ExtractForm key={`e${profile.id}`} profile={profile} family={family} />
        </div>
      )}
    </div>
  );
}

type Input = { path: string; strength: number; custom: boolean };
type Planned = {
  file: string; path: string; bytes: number; layers: number; matched: number; kinds: string[]; ranks: number[]; trigger: string;
};
type MergePlan = { inputs: Planned[]; sizes: Record<string, number | null>; lokr: boolean };

function MergeForm({ profile, family }: { profile: Profile; family: Family }) {
  const toast = useToast();
  const [options] = usePoll<LoraOption[]>(`/api/sandbox/loras?family=${family.id}`, 0, [family.id]);
  const [inputs, setInputs] = useState<Input[]>([{ path: "", strength: 1, custom: false }, { path: "", strength: 1, custom: false }]);
  const [rank, setRank] = useState(0);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [plan, setPlan] = useState<MergePlan | null>(null);
  const [err, setErr] = useState("");
  const paths = useSettled(inputs.map((i) => i.path.trim()).filter(Boolean));
  const jobs = useMemo(() => byJob(options), [options]);

  useEffect(() => {
    let live = true;
    if (!paths.length) { setPlan(null); setErr(""); return; }
    api.post<MergePlan>("/api/tools/merge/check", { profile_id: profile.id, loras: paths }).then(
      (p) => { if (live) { setPlan(p); setErr(""); } },
      (e) => { if (live) { setPlan(null); setErr(String(e.message || e)); } });
    return () => { live = false; };
  }, [JSON.stringify(paths), profile.id]);
  // a LoKr has no rank to stack: it's merged at one
  useEffect(() => { if (plan?.lokr && rank === 0) setRank(64); }, [plan?.lokr]);

  const set = (i: number, patch: Partial<Input>) => setInputs(inputs.map((x, j) => (j === i ? { ...x, ...patch } : x)));
  // what the check found in each file (in the order they were sent)
  const info = (path: string) => (plan && plan.inputs.length === paths.length ? plan.inputs[paths.indexOf(path)] : undefined);
  const chosen = inputs.filter((i) => i.path.trim());
  const auto = chosen.length > 1 ? chosen.map((i) => stem(i.path)).join("+") + (rank ? `_r${rank}` : "")
    : chosen.length ? `${stem(chosen[0].path)}${rank ? `_r${rank}` : "_copy"}` : "merged";
  const size = (r: number) => (plan && plan.sizes[String(r)] != null ? ` · about ${fmtSize(plan.sizes[String(r)] as number)}` : "");
  const unmatched = plan?.inputs.filter((p) => !p.matched) || [];
  const ready = !!plan && !err && !unmatched.length && chosen.length === plan.inputs.length && chosen.every((i) => i.strength !== 0);

  const start = async () => {
    setBusy(true);
    try {
      const made = await api.post<Job>("/api/tools/merge", {
        profile_id: profile.id, rank, name: name.trim(),
        loras: chosen.map((i) => ({ path: i.path.trim(), strength: i.strength })),
      });
      toast("Queued to merge the LoRAs");
      go("queue", made.id);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card card-pad">
      <h2>Merge LoRAs</h2>
      <p className="muted small" style={{ marginTop: 0 }}>What several LoRAs do together, each at its strength, as one LoRA. With one, it's that
        LoRA at a smaller rank (or baked at its strength).</p>
      <div className="stack" style={{ gap: 12 }}>
        <div className="field">
          <span>LoRAs<Info text="LoRAs and LoKrs, Dainamo's or other trainers' (PEFT or kohya keys). Each one's change is added at its strength: a negative strength takes it away. The line under each says how many of its layers this profile's model has; layers it doesn't have are left out." /></span>
          <div className="stack" style={{ gap: 8 }}>
            {inputs.map((l, i) => {
              const p = l.path.trim() ? info(l.path.trim()) : undefined;
              return (
                <div key={i} className="stack" style={{ gap: 3 }}>
                  <div className="sbx-lora">
                    {l.custom || (l.path && options && !options.some((o) => o.path === l.path)) ? (
                      <PathInput value={l.path} onChange={(v) => set(i, { path: v })} exts=".safetensors" placeholder="/path/to/lora.safetensors" />
                    ) : (
                      <select value={l.path} onChange={(e) => e.target.value === "__custom" ? set(i, { custom: true, path: "" }) : set(i, { path: e.target.value })}>
                        <option value="">Choose a LoRA…</option>
                        {jobs.map(([job, opts]) => (
                          <optgroup key={job} label={job}>
                            {opts.map((o) => <option key={o.path} value={o.path}>{o.final ? "final" : `step ${o.step}`}{o.recommended ? " ★ recommended" : ""} · {o.file}</option>)}
                          </optgroup>
                        ))}
                        <option value="__custom">Other file…</option>
                      </select>
                    )}
                    <input type="number" step={0.05} value={l.strength} title="Strength" onChange={(e) => set(i, { strength: Number(e.target.value) })} />
                    <button className="btn ghost icon" title="Remove" disabled={inputs.length <= 1} onClick={() => setInputs(inputs.filter((_, j) => j !== i))}><IconX /></button>
                  </div>
                  {p && (
                    <div className={p.matched ? "faint small" : "validation error"}>
                      {!p.matched ? "Matches no layers of this model: it may be for another model, or name its layers another way (diffusers')."
                        : <>{p.kinds.includes("lokr") ? "LoKr" : p.ranks.length ? `rank ${p.ranks[0] === p.ranks[1] ? p.ranks[0] : p.ranks.join("–")}` : "whole changes"}
                          {" · "}{p.matched === p.layers ? `${p.matched} layers` : `${p.matched} of its ${p.layers} layers (the rest aren't in this model)`}
                          {" · "}{fmtSize(p.bytes)}{p.trigger ? <> · trigger <span className="mono">{p.trigger}</span></> : null}</>}
                    </div>
                  )}
                </div>
              );
            })}
            <div className="row">
              <button className="btn" onClick={() => setInputs([...inputs, { path: "", strength: 1, custom: false }])} disabled={inputs.length >= 8}><IconPlus /> Add LoRA</button>
              {options && options.length === 0 && <span className="faint small">No {family.label} LoRAs trained yet: choose Other file….</span>}
            </div>
          </div>
        </div>
        {err && <div className="error-box small">{err}</div>}
        <label className="field">
          <span>Rank</span>
          <select value={rank} onChange={(e) => setRank(Number(e.target.value))}>
            <option value={0} disabled={!!plan?.lokr}>Exact · every rank kept{plan?.lokr ? " (not with a LoKr)" : size(0)}</option>
            {MERGE_RANKS.map((r) => <option key={r} value={r}>{r}{size(r)}</option>)}
          </select>
          <span className="help">Exact keeps all of each LoRA: its ranks add up, so the file is about as big as the LoRAs together. A rank
            keeps each layer's merged change at that rank, its largest part, in a smaller file; the job's page says how much of
            the change that keeps. A LoKr is merged at a rank.</span>
        </label>
        <label className="field">
          <span>Name</span>
          <input value={name} placeholder={auto} onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="row">
          <button className="btn primary" disabled={busy || !ready} onClick={start}><IconLayers /> Merge</button>
          <span className="faint small">Runs in the queue, a layer at a time; it takes seconds to a few minutes.</span>
        </div>
      </div>
    </div>
  );
}

type ExtractPlan = { sizes: Record<string, number>; base: string; base_found: boolean; base_error: string };

function ExtractForm({ profile, family }: { profile: Profile; family: Family }) {
  const toast = useToast();
  const [models] = usePoll<ModelOption[]>(family.finetune ? `/api/sandbox/models?family=${family.id}` : null, 0, [family.id]);
  const [tuned, setTuned] = useState("");
  const [custom, setCustom] = useState(!family.finetune);
  const [original, setOriginal] = useState("");
  const [rank, setRank] = useState(128);
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [busy, setBusy] = useState(false);
  const settled = useSettled({ tuned: tuned.trim(), original: original.trim() });
  const [plan, , err] = usePoll<ExtractPlan>(settled.tuned
    ? `/api/tools/extract/sizes?profile_id=${profile.id}&path=${q(settled.tuned)}&base=${q(settled.original)}` : null, 0);
  const jobs = useMemo(() => byJob(models), [models]);
  // a finetune job's checkpoint, extracted as its job's Extract LoRA does (linked to it, with its trigger words)
  const ckpt = !original.trim() ? models?.find((m) => m.path === tuned) : undefined;
  const auto = tuned.trim() ? `${stem(tuned.trim())}_r${rank}` : "extracted";

  const start = async () => {
    setBusy(true);
    try {
      const made = ckpt
        ? await api.post<Job>(`/api/jobs/${ckpt.job_id}/extract`, { file: ckpt.file, rank, name: name.trim() })
        : await api.post<Job>("/api/tools/extract", { profile_id: profile.id, tuned: tuned.trim(), base: original.trim(), rank, name: name.trim(), trigger });
      toast("Queued to extract the LoRA");
      go("queue", made.id);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card card-pad">
      <h2>Extract a LoRA</h2>
      <p className="muted small" style={{ marginTop: 0 }}>What a finetuned model changed from the model it was finetuned from, as a LoRA to use on
        that model: Dainamo's finetunes, or another trainer's.</p>
      <div className="stack" style={{ gap: 12 }}>
        <div className="field">
          <span>Finetuned model</span>
          {custom || (tuned && models && !models.some((m) => m.path === tuned)) ? (
            <div className="sbx-lora no-strength">
              <PathInput value={tuned} onChange={setTuned} exts=".safetensors" placeholder="/path/to/finetuned_model.safetensors" />
              {family.finetune && <button className="btn ghost icon" title="Choose a finetune's checkpoint" onClick={() => { setTuned(""); setCustom(false); }}><IconX /></button>}
            </div>
          ) : (
            <select value={tuned} onChange={(e) => e.target.value === "__custom" ? (setTuned(""), setCustom(true)) : setTuned(e.target.value)}>
              <option value="">Choose a finetune's checkpoint…</option>
              {jobs.map(([job, opts]) => (
                <optgroup key={job} label={`${job} · finetune`}>
                  {opts.map((o) => <option key={o.path} value={o.path}>{o.final ? "final" : `step ${o.step}`} · {o.file}</option>)}
                </optgroup>
              ))}
              <option value="__custom">Other file…</option>
            </select>
          )}
        </div>
        <label className="field" onClick={(e) => e.preventDefault()}>
          <span>Finetuned from <span className="faint">· optional</span></span>
          <PathInput value={original} onChange={setOriginal} exts=".safetensors" placeholder={`the profile's: ${base(String(profile.data?.diffusion_model || ""))}`} />
          <span className="help">The model it was finetuned from, which the LoRA is the change from, and which it's used on. The two have to be the same
            kind of model, laid out the same way.</span>
        </label>
        {err && settled.tuned && <div className="error-box small">{err}</div>}
        {plan && !plan.base_found && <div className="error-box small">{plan.base_error}</div>}
        <label className="field">
          <span>Rank</span>
          <select value={rank} onChange={(e) => setRank(Number(e.target.value))}>
            {EXTRACT_RANKS.map((r) => <option key={r} value={r}>{r}{plan ? ` · about ${fmtSize(plan.sizes[r])}` : ""}</option>)}
          </select>
          <span className="help">Each weight matrix's change is kept at this rank: its largest part. A higher rank keeps more of what the
            finetune learned, in a bigger file. Changes to norms and biases are kept whole (they're small).</span>
        </label>
        {!ckpt && (
          <label className="field">
            <span>Trigger words <span className="faint">· optional</span></span>
            <input value={trigger} placeholder="saved in the LoRA, as a trained one's are" onChange={(e) => setTrigger(e.target.value)} />
          </label>
        )}
        <label className="field">
          <span>Name</span>
          <input value={name} placeholder={auto} onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="row">
          <button className="btn primary" disabled={busy || !tuned.trim() || !plan || !plan.base_found || !!err} onClick={start}><IconLayers /> Extract</button>
          <span className="faint small">Runs in the queue, a block at a time: minutes, longer from a slow disk.</span>
        </div>
      </div>
    </div>
  );
}

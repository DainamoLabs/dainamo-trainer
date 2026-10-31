import { useEffect, useState } from "react";
import { go } from "../App";
import { api, fmtTime, type Dataset } from "../api";
import { IconDownload, IconUpload } from "./icons";
import { Modal, PathInput, TaskProgress, fmtSize, usePoll, useToast } from "./ui";

/** The one export or import running on the server (GET /api/transfer). */
type Transfer = {
  state: "idle" | "running" | "done" | "failed" | "cancelled"; kind?: "export" | "import"; label?: string; done?: number; total?: number;
  path?: string; job_id?: string | null; message?: string; started?: number;
};
// what exporting a job copies (GET /api/jobs/{id}/export-job)
type ExportSize = {
  job: number; cache: number; total: number; status: string;
  datasets: { id: string; name: string; roles: string[]; exists: boolean; bytes: number; files: number }[];
};
// how a profile here compares with the exported job's model files (by their contents)
type ProfileMatch = {
  id: string; name: string; same_model: boolean; identical: boolean; same_variant: boolean;
  fields: { key: string; label: string; exported: string; here: string; missing: boolean; same: boolean }[];
};
// an exported job, read for Import (POST /api/import-job/inspect)
type Seen = {
  path: string; name: string; family: string; family_label: string; status: string; step: number; steps: number | null;
  trained: boolean; exported: number; profiles: ProfileMatch[]; recommended: string | null;
  datasets: { id: string; name: string; trigger: string; roles: string[]; included: boolean; files: number; bytes: number; here: { id: string; name: string; path: string } | null }[];
  sizes: { job: number; cache: number };
};

const EXPORT_DIR = "dainamo.exportDir";
const remembered = (k: string) => { try { return localStorage.getItem(k) || ""; } catch { return ""; } };
const roles = (r: string[]) => r.filter((x) => x !== "training").map((x) => (x === "source" ? "source images" : x)).join(", ");

/** The transfer this dialog started (``begin`` it with what the start returned), polled while it runs. */
function useTransfer() {
  const [started, setStarted] = useState<number | null>(null);
  const [st] = usePoll<Transfer>(started ? "/api/transfer" : null, 700, [started]);
  const t = st && st.started === started ? st : null;
  return { t, running: !!started && (!t || t.state === "running"), begin: (s: Transfer) => setStarted(s.started ?? null) };
}

function Progress({ t, verb }: { t: Transfer | null; verb: string }) {
  const toast = useToast();
  const done = t?.done || 0, total = t?.total || 0;
  return (
    <TaskProgress label={verb} running pct={total ? (100 * done) / total : null}
      detail={total ? `${fmtSize(done)} of ${fmtSize(total)}` : undefined}
      onCancel={() => api.post("/api/transfer/cancel", {}).catch((e) => toast(e.message, true))} />
  );
}

/** Export: the job copied into a new folder in one the user picks, with its datasets (unless left out) and the cache
 * it used, to add on another machine with Import job. */
export function ExportJobDialog({ job, onClose }: { job: { id: string; name: string }; onClose: () => void }) {
  const [path, setPath] = useState(() => remembered(EXPORT_DIR));
  const [withData, setWithData] = useState(true);
  const [size, , sizeErr] = usePoll<ExportSize>(`/api/jobs/${job.id}/export-job`, 0);
  const { t, running, begin } = useTransfer();
  const [err, setErr] = useState("");
  const data = size?.datasets.filter((d) => d.exists).reduce((n, d) => n + d.bytes, 0) ?? 0;
  const total = size ? size.job + size.cache + (withData ? data : 0) : 0;
  const start = async () => {
    setErr("");
    try {
      begin(await api.post<Transfer>(`/api/jobs/${job.id}/export-job`, { path, datasets: withData }));
      try { localStorage.setItem(EXPORT_DIR, path); } catch { /* (only remembered) */ }
    } catch (e: any) {
      setErr(e.message);
    }
  };
  const finished = t && t.state !== "running";
  return (
    <Modal title={`Export ${job.name}`} onClose={running ? () => {} : onClose} footer={running ? null : finished ? (
      <button className="btn primary" onClick={onClose}>Close</button>
    ) : <>
      <button className="btn ghost" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={!path.trim() || !size} onClick={start}><IconDownload /> Export</button>
    </>}>
      <div className="stack">
        <p className="muted" style={{ margin: 0 }}>Copies the job into a new folder in the one you choose: its checkpoints, samples,
          training state and log, the encoded cache it used, and its datasets. On another machine, add it with <strong style={{ color: "var(--text)" }}>Import job</strong> on
          the Queue page. The model files aren't copied: there, you choose a profile with the same models.</p>
        <label className="field" onClick={(e) => e.preventDefault()}>
          <span>Save in</span>
          <PathInput mode="dir" value={path} onChange={setPath} placeholder="A folder, e.g. on a USB drive" />
        </label>
        {sizeErr && <div className="error-box small">{sizeErr}</div>}
        {size && (
          <div className="transfer-parts">
            <div><span>The job <span className="faint">· checkpoints, samples, training state, log</span></span><b>{fmtSize(size.job)}</b></div>
            <div><span>Encoded cache <span className="faint">· saves encoding again there</span></span><b>{size.cache ? fmtSize(size.cache) : "none"}</b></div>
            {size.datasets.length > 0 && (
              <label className="check">
                <input type="checkbox" checked={withData} onChange={(e) => setWithData(e.target.checked)} /> Include the datasets' files
              </label>
            )}
            {size.datasets.map((d) => (
              <div key={d.id} className={withData && d.exists ? "" : "faint"}>
                <span>{d.name}{d.roles.some((r) => r !== "training") && <span className="faint"> · {roles(d.roles)}</span>}
                  {!d.exists && <span className="faint"> · its folder isn't there: left out</span>}</span>
                <b>{d.exists ? fmtSize(d.bytes) : "—"}</b>
              </div>
            ))}
            <div className="transfer-total"><span>Total</span><b>{fmtSize(total)}</b></div>
          </div>
        )}
        {!withData && <span className="help">Leave them out when the other machine has them: you pick them there when importing.</span>}
        {err && <div className="error-box small">{err}</div>}
        {running && <Progress t={t} verb="Exporting" />}
        {t?.state === "done" && <div className="ok-box small">Exported to <span className="mono">{t.path}</span></div>}
        {(t?.state === "failed" || t?.state === "cancelled") && <div className="error-box small">{t.state === "cancelled" ? "Cancelled: nothing was kept" : t.message}</div>}
      </div>
    </Modal>
  );
}

/** Import: a job exported from another machine (or this one) added to the Queue, with a profile here for its models
 * and a dataset here (or the export's copy) for each of its datasets. */
export function ImportJobDialog({ onClose, onImported }: { onClose: () => void; onImported: () => void }) {
  const toast = useToast();
  const [path, setPath] = useState("");
  const [seen, setSeen] = useState<Seen | null>(null);
  const [reading, setReading] = useState(false);
  const [name, setName] = useState("");
  const [profileId, setProfileId] = useState("");
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [restart, setRestart] = useState(false);
  const [err, setErr] = useState("");
  const [here] = usePoll<Dataset[]>(seen ? "/api/datasets" : null, 0);
  const { t, running, begin } = useTransfer();

  const read = async () => {
    setErr("");
    setReading(true);
    try {
      const s = await api.post<Seen>("/api/import-job/inspect", { path });
      setSeen(s);
      setName(s.name);
      setProfileId(s.recommended || "");
      setChoices(Object.fromEntries(s.datasets.map((d) => [d.id, d.here ? d.here.id : d.included ? "export" : ""])));
      setRestart(false);
    } catch (e: any) {
      setSeen(null);
      setErr(e.message);
    } finally {
      setReading(false);
    }
  };
  useEffect(() => {
    if (t?.state === "done" && t.job_id) {
      toast(`Imported ${name}`);
      onImported();
      onClose();
      go("queue", t.job_id);
    }
  }, [t?.state]);

  const profile = seen?.profiles.find((p) => p.id === profileId);
  const mustRestart = !!seen?.trained && !!profile && !profile.same_model;
  const reEncode = !!seen?.trained && !!profile?.same_model && !profile.identical;
  const ready = !!seen && !!name.trim() && !!profile && seen.datasets.every((d) => choices[d.id]) && (!mustRestart || restart);
  const start = async () => {
    setErr("");
    try {
      begin(await api.post<Transfer>("/api/import-job", { path: seen!.path, name, profile_id: profileId, datasets: choices, restart: mustRestart && restart }));
    } catch (e: any) {
      setErr(e.message);
    }
  };
  const copying = seen ? seen.sizes.job + seen.sizes.cache + seen.datasets.filter((d) => choices[d.id] === "export").reduce((n, d) => n + d.bytes, 0) : 0;
  const failed = t?.state === "failed" || t?.state === "cancelled";
  return (
    <Modal title="Import job" wide onClose={running ? () => {} : onClose} footer={running ? null : <>
      <button className="btn ghost" onClick={onClose}>{failed ? "Close" : "Cancel"}</button>
      <button className="btn primary" disabled={!ready} onClick={start}><IconUpload /> Import</button>
    </>}>
      <div className="stack">
        <p className="muted" style={{ margin: 0 }}>Adds a job that Export copied to a folder, on another machine or this one.</p>
        <div className="field">
          <span>Exported folder</span>
          <PathInput mode="dir" value={path} onChange={(v) => { setPath(v); setSeen(null); }}
            extra={<button className="btn" disabled={!path.trim() || reading} onClick={read}>{reading ? "Reading…" : "Read"}</button>} />
        </div>
        {seen && <>
          <div className="muted small">
            {seen.family_label} · {seen.trained ? `${seen.status} at step ${seen.step}${seen.steps ? ` of ${seen.steps}` : ""}` : "not trained yet"}
            {seen.exported ? ` · exported ${fmtTime(seen.exported)}` : ""}
          </div>
          <label className="field">
            <span>Job name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </label>

          <div className="field">
            <span>Profile</span>
            {seen.profiles.length === 0 ? (
              <div className="error-box small">There's no {seen.family_label} profile here. Make one in Profiles (Models can download its
                files), then import.</div>
            ) : <>
              <select value={profileId} onChange={(e) => { setProfileId(e.target.value); setRestart(false); }}>
                {seen.profiles.map((p) => <option key={p.id} value={p.id}>{p.name}{p.identical ? " · the same model files" : p.same_model ? " · the same diffusion model" : ""}</option>)}
              </select>
              {profile && (
                <div className="transfer-parts">
                  {profile.fields.map((f) => (
                    <div key={f.key}>
                      <span>{f.label}</span>
                      {f.same ? <b className="check-ok">the same file</b>
                        : <span className="warn-text" title={`Exported: ${f.exported || "none"}\nHere: ${f.here || "none"}`}>
                          {!f.here ? "none here" : f.missing ? `${f.here} isn't there` : !f.exported ? "the job had none" : `another file (${f.here})`}
                        </span>}
                    </div>
                  ))}
                  {!profile.same_variant && <div><span>Variant</span><span className="warn-text">another variant</span></div>}
                </div>
              )}
              {mustRestart && (
                <div className="warn-box small stack" style={{ gap: 8 }}>
                  <span>The job trained on another diffusion model{profile?.fields.find((f) => f.key === "diffusion_model")?.exported
                    ? ` (${profile.fields.find((f) => f.key === "diffusion_model")!.exported})` : ""}. Its LoRA can't carry on with this one: it can only start again
                    from step 0.</span>
                  <label className="check"><input type="checkbox" checked={restart} onChange={(e) => setRestart(e.target.checked)} /> Start again from step 0
                    (its checkpoints, samples and training state aren't imported)</label>
                </div>
              )}
              {reEncode && <span className="help">The text encoder or VAE is another file, so its training files are encoded again; it carries on from step {seen.step}.</span>}
            </>}
          </div>

          {seen.datasets.length > 0 && (
            <div className="field">
              <span>Datasets</span>
              <div className="transfer-parts">
                {seen.datasets.map((d) => (
                  <div key={d.id} className="transfer-dataset">
                    <span className="ellipsis">{d.name}{d.roles.some((r) => r !== "training") && <span className="faint"> · {roles(d.roles)}</span>}</span>
                    <select value={choices[d.id] || ""} onChange={(e) => setChoices((c) => ({ ...c, [d.id]: e.target.value }))}>
                      {!choices[d.id] && <option value="">Choose…</option>}
                      {d.included && <option value="export">Add the export's copy ({d.files} files, {fmtSize(d.bytes)})</option>}
                      {(here || []).map((h) => <option key={h.id} value={h.id}>Use {h.name}{h.id === d.here?.id ? " (the same dataset)" : ""}</option>)}
                    </select>
                  </div>
                ))}
              </div>
              {seen.datasets.some((d) => !d.included) && <span className="help">The export left some datasets' files out: choose a dataset here for those.</span>}
            </div>
          )}
          <span className="help">About {fmtSize(copying)} is copied into Dainamo's data folder.</span>
        </>}
        {err && <div className="error-box small">{err}</div>}
        {running && <Progress t={t} verb="Importing" />}
        {failed && <div className="error-box small">{t!.state === "cancelled" ? "Cancelled: nothing was kept" : t!.message}</div>}
      </div>
    </Modal>
  );
}

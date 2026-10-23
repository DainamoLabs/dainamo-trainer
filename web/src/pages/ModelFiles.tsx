import { useEffect, useState } from "react";
import { api, type Family, type ModelFamily, type ModelField, type ModelsOverview, type Profile } from "../api";
import { IconAlert, IconCheck, IconDown, IconDownload, IconInfo, IconMinusCircle, IconSearch } from "../components/icons";
import { ACTIVE, DownloadDialog, TaskRow } from "../components/ModelDownload";
import { FileBrowser, usePoll, useToast } from "../components/ui";

/** The files a profile of this family would use: the first usable one found per field, and the variant the
 *  diffusion model is for. */
export function profileFrom(fam: ModelFamily, info: Family | undefined): Partial<Profile> {
  const data: Record<string, unknown> = {};
  info?.profile_fields.forEach((x) => { if (x.default != null) data[x.key] = x.default; });
  for (const f of fam.fields) {
    const pick = f.found.find((x) => !x.unsupported.length);
    if (pick) data[f.key] = pick.path;
  }
  const dm = fam.fields.find((f) => f.key === "diffusion_model");
  const opt = dm?.downloads.find((o) => o.have && o.have === data.diffusion_model);
  if (opt?.variant && info?.profile_fields.some((x) => x.key === "variant")) data.variant = opt.variant;
  return { name: fam.label.replace(/ \(.*\)$/, ""), family: fam.id, data };
}

const usable = (f: ModelField) => f.found.filter((x) => !x.unsupported.length);

/** A found file's name; where it is, and its precision, on hover. */
function FoundFile({ x }: { x: ModelField["found"][number] }) {
  return <span className="mf-found mono ellipsis" title={[x.path, x.quant.join(", ")].filter(Boolean).join(" · ")}>{x.name.replace(/\.safetensors$/, "")}</span>;
}

function FieldRow({ f, overview, searched, onDownload }: { f: ModelField; overview: ModelsOverview; searched: boolean; onDownload: () => void }) {
  const files = usable(f);
  const [open, setOpen] = useState(false); // the list of files found, where there's more than one
  const task = overview.tasks.find((t) => ACTIVE.includes(t.state) && f.downloads.some((o) => o.id === t.download));
  const pct = task ? Math.floor((100 * task.done) / (task.size || 1)) : 0;
  return (
    <div className="mf-field">
      <div className="mf-label">
        <div className="mf-name">{f.label.replace(" (CLIP)", "")}</div>
        <div className="mf-hint">{!f.required && <span className="mf-optional">Optional · </span>}{f.hint}</div>
      </div>
      <div className="mf-files">
        {task ? (
          <div className="mf-state">
            <span className="muted">Downloading · {pct}%</span>
            <div className="bar mf-bar"><div style={{ width: `${pct}%` }} /></div>
          </div>
        ) : files.length === 1 ? (
          <div className="mf-state ok"><IconCheck /><FoundFile x={files[0]} /></div>
        ) : files.length > 1 ? (
          <>
            <div className="mf-state ok"><IconCheck /><span>{files.length} files found</span></div>
            <button className="mf-more" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? "Hide files" : "View files"} <IconDown /></button>
            {open && <div className="mf-list">{files.map((x) => <FoundFile key={x.path} x={x} />)}</div>}
          </>
        ) : searched && f.required ? (
          <div className="mf-state warn"><IconAlert /><span>Not found</span></div>
        ) : (
          <div className="mf-state muted" title={searched ? undefined : "There's no models folder to look in yet"}><IconMinusCircle /><span>{searched ? "None found" : "Not checked"}</span></div>
        )}
      </div>
      {f.downloads.length > 0
        ? <button className="btn icon mf-dl" onClick={onDownload} title="Download a file for this from Hugging Face" aria-label={`Download a ${f.label.replace(" (CLIP)", "")}`}><IconDownload /></button>
        : <span />}
    </div>
  );
}

/** The ComfyUI models folder Dainamo looks in: set here (saved straight away, unlike on the Settings page, which saves
 * everything at once), offering the folder the profiles' files are in. */
function FolderCard({ ov, suggested, onSaved }: { ov: ModelsOverview; suggested: string; onSaved: () => void }) {
  const toast = useToast();
  const [draft, setDraft] = useState<string | null>(null); // null: not edited, so the folder (or the suggestion)
  const [browsing, setBrowsing] = useState(false);
  const [saving, setSaving] = useState(false);
  const value = draft ?? (ov.folder || suggested);
  const save = async (folder: string) => {
    setSaving(true);
    try {
      await api.put("/api/settings", { models: { folder } });
      toast("Models folder set");
      setDraft(null);
      onSaved();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setSaving(false);
    }
  };
  const changed = !!value.trim() && value.trim() !== ov.folder;
  const [badge, tone] = !ov.folder ? ["Not configured", "warn"] : !ov.folder_ok ? ["Folder not found", "error"] : ["Configured", "ok"];
  return (
    <div className="card mf-folder">
      <div className="mf-folder-top">
        <div style={{ minWidth: 0 }}>
          <div className="mf-folder-title">
            <h3>Model folder</h3>
            <span className={`mf-badge ${tone}`}>{tone === "ok" ? <IconCheck /> : <IconAlert />}{badge}</span>
          </div>
          <p className="muted">
            {!ov.folder ? "Only the files your profiles use have been checked."
              : !ov.folder_ok ? "This folder isn't there, so only the files your profiles use have been checked."
              : "Its diffusion_models (or unet), text_encoders (or clip), vae, loras and checkpoints folders are searched."}
          </p>
        </div>
        <a className="mf-hf" href="#/settings" title={ov.token ? "Signed in to Hugging Face (Settings)" : "Sign in to Hugging Face in Settings"}>
          <i className={ov.token ? "on" : ""} />{ov.token ? "Hugging Face connected" : <>Hugging Face not connected<span className="faint"> · needed only for LTX 2.5</span></>}
        </a>
      </div>
      <div className="mf-folder-row">
        <input className="mono" value={value} placeholder="/path/to/ComfyUI/models" aria-label="ComfyUI models folder"
          onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === "Enter" && changed && !saving && save(value.trim())} />
        {(changed || !ov.folder) && <button className="btn primary lg" disabled={!changed || saving} onClick={() => save(value.trim())}>Use this folder</button>}
        <button className="btn lg outline" onClick={() => setBrowsing(true)}>{value.trim() ? "Choose another" : "Browse"}</button>
      </div>
      {!ov.folder && suggested && value.trim() === suggested && <p className="help">Your profiles use files in this folder.</p>}
      {browsing && (
        <FileBrowser start={value.trim() || "~"} mode="dir" onClose={() => setBrowsing(false)} onPick={(p) => { setBrowsing(false); save(p); }} />
      )}
    </div>
  );
}

type Status = "all" | "ready" | "missing" | "unchecked";
const STATUSES: [Status, string][] = [["all", "All statuses"], ["ready", "Required files found"], ["missing", "Missing files"], ["unchecked", "Not checked"]];

/** Profiles → Model files: what each family needs, what's on this machine, and downloads of the rest. */
export default function ModelFiles({ families, onCreate }: { families: Family[]; onCreate: (p: Partial<Profile>) => void }) {
  const [ov, reload, err] = usePoll<ModelsOverview>("/api/models", 0);
  const [dialog, setDialog] = useState<{ family: string; field: string } | null>(null);
  const [suggested, setSuggested] = useState<string[]>([]); // models folders the profiles' files are in
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<Status>("all");
  const active = !!ov?.tasks.some((t) => ACTIVE.includes(t.state));
  useEffect(() => {
    if (!active) return;
    const t = setInterval(reload, 1500);
    return () => clearInterval(t);
  }, [active, reload]);
  useEffect(() => {
    if (ov && !ov.folder) api.get<{ folders: string[] }>("/api/models/suggest").then((r) => setSuggested(r.folders), () => {});
  }, [ov?.folder]);

  if (err && !ov) return <div className="error-box">{err.includes("404") ? "The Dainamo server is older than this page: restart it, then reload." : err}</div>;
  if (!ov) return <div className="muted">Looking for model files…</div>;
  const searched = ov.folder_ok; // otherwise only the files profiles use were looked at
  const cards = ov.families.map((fam) => {
    const needed = fam.fields.filter((f) => f.required);
    const have = needed.filter((f) => usable(f).length).length;
    const state: Status = have === needed.length ? "ready" : searched ? "missing" : "unchecked";
    return { fam, needed: needed.length, have, state };
  });
  const words = search.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = cards.filter((c) => (status === "all" || c.state === status) && words.every((w) =>
    [c.fam.label, ...c.fam.fields.flatMap((f) => [f.label, f.hint, ...usable(f).map((x) => x.name)])].join(" ").toLowerCase().includes(w)));
  return (
    <div className="stack" style={{ gap: 16 }}>
      <FolderCard ov={ov} suggested={suggested[0] || ""} onSaved={reload} />

      {ov.tasks.length > 0 && (
        <div className="card card-pad stack" style={{ gap: 12 }}>
          <h3 className="mf-section">Downloads</h3>
          {ov.tasks.map((t) => <TaskRow key={t.id} t={t} onChange={reload} />)}
        </div>
      )}

      <div className="ds-toolbar pf-toolbar">
        <label className="search">
          <IconSearch />
          <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search models…" aria-label="Search models"
            onKeyDown={(e) => e.key === "Escape" && search && (e.stopPropagation(), setSearch(""))} />
        </label>
        <select value={status} onChange={(e) => setStatus(e.target.value as Status)} aria-label="Status">
          {STATUSES.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
        </select>
      </div>
      {!shown.length && (
        <div className="empty">
          <strong>No models match</strong>
          <button className="btn ghost" onClick={() => { setSearch(""); setStatus("all"); }}>Clear the search{status !== "all" ? " and filter" : ""}</button>
        </div>
      )}

      <div className="mf-grid">
        {shown.map(({ fam, needed, have, state }) => (
          <div className="card mf-card" key={fam.id}>
            <div className="mf-card-head">
              <h3>{fam.label}</h3>
              {state === "ready" ? <span className="mf-badge ok"><IconCheck />Required files found</span>
                : state === "missing" ? <span className="mf-badge warn"><IconAlert />{needed - have} missing</span>
                : <span className="mf-badge">Not checked</span>}
            </div>
            <p className={"mf-sub " + (state === "missing" ? "warn-text" : "muted")}>
              {state === "ready" ? "Everything it needs is here" : state === "missing" ? `${have} of the ${needed} files it needs` : "Not searched yet"}
            </p>
            {fam.fields.map((f) => (
              <FieldRow key={f.key} f={f} overview={ov} searched={searched} onDownload={() => setDialog({ family: fam.id, field: f.key })} />
            ))}
            <div className="mf-foot">
              <button className="btn mf-create" disabled={state !== "ready"}
                title={state === "ready" ? undefined : searched ? "Download the missing files first" : "Set your models folder first"}
                onClick={() => onCreate(profileFrom(fam, families.find((f) => f.id === fam.id)))}>Create profile</button>
            </div>
          </div>
        ))}
      </div>
      <div className="card ds-note">
        <IconInfo />
        <span>Files are recognised by what's in them, whatever they're called. Downloads come from Hugging Face, from the commit
          Dainamo was checked against, and each is checked against its published size and checksum.</span>
      </div>

      {dialog && <DownloadDialog family={dialog.family} field={dialog.field} onClose={() => setDialog(null)} onStarted={reload} />}
    </div>
  );
}

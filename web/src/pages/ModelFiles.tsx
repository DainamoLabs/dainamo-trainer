import { useEffect, useState } from "react";
import { api, type Family, type ModelFamily, type ModelField, type ModelsOverview, type Profile } from "../api";
import { IconAlert, IconCheck, IconDown, IconDownload, IconInfo, IconMinusCircle, IconSearch } from "../components/icons";
import { ACTIVE, DownloadDialog, TaskRow } from "../components/ModelDownload";
import { FileBrowser, usePoll, useToast } from "../components/ui";

/** The files a profile of this family would use: the first usable one found per field, and the variant the
 *  diffusion model is for. Where files are for one of the family's models (FLUX.2 Klein's 4B or 9B), the others
 *  are the diffusion model's model's: the first diffusion model whose model has a file for every field, else the
 *  first. */
export function profileFrom(fam: ModelFamily, info: Family | undefined): Partial<Profile> {
  const data: Record<string, unknown> = {};
  info?.profile_fields.forEach((x) => { if (x.default != null) data[x.key] = x.default; });
  const fields = fam.fields.filter((f) => f.profile !== false); // not e.g. FLUX 3's prompt upsampler (for captions)
  const pick = (f: ModelField, model: string | null | undefined) =>
    usable(f).find((x) => !model || !x.model || x.model === model);
  const dm = fam.fields.find((f) => f.key === "diffusion_model");
  const dms = dm ? usable(dm) : [];
  const model = (dms.find((d) => fields.every((f) => !usable(f).length || pick(f, d.model))) || dms[0])?.model;
  for (const f of fields) {
    const x = pick(f, model);
    if (x) data[f.key] = x.path;
  }
  const opt = dm?.downloads.find((o) => o.have && o.have === data.diffusion_model);
  if (opt?.variant && info?.profile_fields.some((x) => x.key === "variant")) data.variant = opt.variant;
  return { name: fam.label.replace(/ \(.*\)$/, ""), family: fam.id, data };
}

const usable = (f: ModelField) => f.found.filter((x) => !x.unsupported.length);

/** What the server says beyond ``ModelsOverview``: where each file was found ("models": the models folder or a
 *  profile; "hf_cache": the Hugging Face cache, with its "owner/repo"), and the cache's folder. */
type Found = ModelField["found"][number] & { source?: "models" | "hf_cache"; repo?: string };
type Overview = ModelsOverview & { hf_cache?: string; hf_cache_ok?: boolean };

/** A found file's name (a Hugging Face cache's also tagged, with the repo it's in); where it is, and its precision,
 *  on hover. The cache's are named as on the hub: "owner/repo" for a folder of shards, "owner/repo/vae/ae.safetensors"
 *  for a file, shown as its name over the folder it's in. */
function FoundFile({ x }: { x: Found }) {
  const title = [x.path, x.quant.join(", ")].filter(Boolean).join(" · ");
  if (x.source !== "hf_cache") return <span className="mf-found mono ellipsis" title={title}>{x.name.replace(/\.safetensors$/, "")}</span>;
  const file = x.name.endsWith(".safetensors");
  const cut = x.name.lastIndexOf("/");
  return (
    <span className="mf-found" title={title} style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 3 }}>
      <span className="mono ellipsis" style={{ maxWidth: "100%" }}>{file ? x.name.slice(cut + 1).replace(/\.safetensors$/, "") : x.name}</span>
      {/* the repo goes under the tag where there's no room beside it */}
      <span className="row wrap" style={{ gap: "3px 6px", maxWidth: "100%" }}>
        <span className="pill" style={{ flex: "none" }}>Hugging Face cache</span>
        {file && cut > 0 && <span className="faint small ellipsis" style={{ maxWidth: "100%" }}>{x.name.slice(0, cut)}</span>}
      </span>
    </span>
  );
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

/** The ComfyUI models folder Dainamo looks in and downloads into: the one place it's set (Settings links here), saved
 * straight away, offering the folder the profiles' files are in. The Hugging Face cache is searched as well. */
function FolderCard({ ov, suggested, onSaved }: { ov: Overview; suggested: string; onSaved: () => void }) {
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
  const others = ov.hf_cache ? "the Hugging Face cache and the files your profiles use" : "the files your profiles use";
  // the families with a download Hugging Face wants a sign-in for
  const gated = ov.families.filter((f) => f.fields.some((x) => x.downloads.some((o) => o.gated))).map((f) => f.label.replace(/ \(.*\)$/, ""));
  return (
    <div className="card mf-folder">
      <div className="mf-folder-top">
        <div style={{ minWidth: 0 }}>
          <div className="mf-folder-title">
            <h3>Model folder</h3>
            <span className={`mf-badge ${tone}`}>{tone === "ok" ? <IconCheck /> : <IconAlert />}{badge}</span>
          </div>
          <p className="muted">
            {!ov.folder ? `Only ${others} have been checked.`
              : !ov.folder_ok ? `This folder isn't there, so only ${others} have been checked.`
              : "Its diffusion_models (or unet), text_encoders (or clip), vae, loras and checkpoints folders are searched, and downloads go there."}
          </p>
          {ov.hf_cache && (
            <p className="muted" style={{ overflowWrap: "anywhere" }} title="Text encoders, VAEs and single model files are found there; diffusion models in the diffusers layout aren't">
              Also searched: <span className="mono">{ov.hf_cache}</span> (the Hugging Face cache{ov.hf_cache_ok ? "" : ", not there yet"})
            </p>
          )}
        </div>
        {/* the dot stays by the first line when the advice wraps */}
        <span className="mf-hf" style={{ alignItems: "flex-start" }} title={ov.token ? "A Hugging Face sign-in was found on this machine"
          : `Gated downloads need one${gated.length ? ` (some ${gated.join(", ")} files)` : ""}: accept the model's terms on Hugging Face, then sign in`}>
          <i className={ov.token ? "on" : ""} style={{ marginTop: 5 }} />{ov.token ? "Hugging Face connected"
            : <span>Hugging Face not connected<span className="faint"> · for gated downloads, run <code>hf auth login</code> (or set HF_TOKEN) before starting Dainamo</span></span>}
        </span>
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

const usableUpscaler = (m: ModelsOverview["upscalers"]["models"][number]) => m.kind !== "unknown" && !m.error;

/** The upscalers in upscale_models (for the Sandbox's Upscale and a dataset's Upscale small images), what each is,
 * and downloads of Real-ESRGAN and SeedVR2. */
function UpscalersCard({ ov, onDownload }: { ov: ModelsOverview; onDownload: () => void }) {
  const up = ov.upscalers;
  const field = up.fields[0];
  const task = ov.tasks.find((t) => ACTIVE.includes(t.state) && field?.downloads.some((o) => o.id === t.download));
  const pct = task ? Math.floor((100 * task.done) / (task.size || 1)) : 0;
  const ready = up.models.filter(usableUpscaler);
  return (
    <div className="card mf-card">
      <div className="mf-card-head">
        <h3>Upscalers</h3>
        {ready.length ? <span className="mf-badge ok"><IconCheck />{ready.length} ready</span>
          : <span className="mf-badge">{up.folder_ok ? "None found" : "Not checked"}</span>}
      </div>
      <p className="mf-sub muted">For Upscale in the Sandbox and a dataset's Upscale small images. They're read from the models folder's
        upscale_models, as in ComfyUI.</p>
      {up.models.map((m) => (
        <div className="mf-field mf-up" key={m.path}>
          <div className="mf-label">
            <div className="mf-name ellipsis" title={m.path}>{m.name}</div>
            <div className="mf-hint">{m.kind === "seedvr2" ? "SeedVR2 · any size, keeps video frames consistent"
              : m.kind === "pixel" ? `Pixel upscaler · ${m.scale}×, fitted to the size asked for` : "Not an upscaler Dainamo reads"}</div>
          </div>
          <div className="mf-files">
            {usableUpscaler(m) ? <div className="mf-state ok"><IconCheck /><span>Ready</span></div>
              : <div className="mf-state warn"><IconAlert /><span>{m.error || "Unknown file"}</span></div>}
          </div>
          <span />
        </div>
      ))}
      {!up.models.length && (
        <div className="mf-field mf-up">
          <div className="mf-label"><div className="mf-name">No upscaler</div>
            <div className="mf-hint">{up.folder_ok ? "upscale_models has none" : ov.folder ? "There's no upscale_models folder yet" : "Set your models folder first"}</div></div>
          <span /><span />
        </div>
      )}
      {up.error && <div className="validation warning">{up.error}</div>}
      {task && (
        <div className="mf-state" style={{ paddingTop: 8 }}>
          <span className="muted">Downloading {task.name} · {pct}%</span>
          <div className="bar mf-bar"><div style={{ width: `${pct}%` }} /></div>
        </div>
      )}
      <div className="mf-foot">
        <button className="btn mf-create" onClick={onDownload} disabled={!field}><IconDownload /> Download an upscaler</button>
      </div>
    </div>
  );
}

type Status = "all" | "ready" | "missing" | "unchecked";
const STATUSES: [Status, string][] = [["all", "All statuses"], ["ready", "Required files found"], ["missing", "Missing files"], ["unchecked", "Not checked"]];

/** Model files: what each family needs, what's on this machine (the models folder and the Hugging Face cache), and
 *  downloads of the rest. */
export default function ModelFiles({ families, onCreate }: { families: Family[]; onCreate: (p: Partial<Profile>) => void }) {
  const [ov, reload, err] = usePoll<Overview>("/api/models", 0);
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
  // the Upscalers card belongs to no family: it shows unless a status is picked, or the search leaves it out
  const upscalers = !!ov.upscalers && status === "all" && words.every((w) =>
    ["upscalers upscale", ...ov.upscalers.models.map((m) => `${m.name} ${m.kind}`)].join(" ").toLowerCase().includes(w));
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
      {!shown.length && !upscalers && (
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
        {upscalers && <UpscalersCard ov={ov} onDownload={() => setDialog({ family: ov.upscalers.id, field: ov.upscalers.fields[0]?.key || "" })} />}
      </div>
      <div className="card ds-note">
        <IconInfo />
        <span>Files are recognised by what's in them, whatever they're called, in the models folder and the Hugging Face cache.
          Downloads come from Hugging Face, from the commit Dainamo was checked against, into the models folder, and each is
          checked against its published size and checksum.</span>
      </div>

      {dialog && <DownloadDialog family={dialog.family} field={dialog.field} onClose={() => setDialog(null)} onStarted={reload} />}
    </div>
  );
}

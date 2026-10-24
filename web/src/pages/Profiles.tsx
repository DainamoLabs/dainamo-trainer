import { useEffect, useState } from "react";
import { api, byFamily, type Family, type ModelsOverview, type Profile, type ProfileField } from "../api";
import { go } from "../App";
import { IconCopy, IconDownload, IconEdit, IconLayers, IconPlus, IconSearch, IconTrash } from "../components/icons";
import { DownloadDialog } from "../components/ModelDownload";
import { ConfirmDialog, Empty, LoadError, Menu, Modal, PathInput, familyBadge, usePoll, useToast } from "../components/ui";
import MergeExtract from "./MergeExtract";
import ModelFiles from "./ModelFiles";

type Validation = { field: string; level: string; message: string };

// where ComfyUI keeps each kind of file
const FOLDERS: Record<string, string> = {
  diffusion_model: "models/diffusion_models", text_encoder: "models/text_encoders (or clip)", vae: "models/vae",
  audio_vae: "models/vae", training_adapter: "models/loras, or a Hugging Face path",
};

/** What a family's profile needs, so the right files can be found before browsing. */
function Needs({ fam }: { fam: Family }) {
  const files = fam.profile_fields.filter((f) => f.kind === "path");
  return (
    <div className="needs">
      <div className="needs-title">{fam.label} needs</div>
      {files.map((f) => (
        <div className="needs-row" key={f.key}>
          <span className="muted">{f.label.replace(" (CLIP)", "")}</span>
          <span>{f.hint || "—"}{!f.required && <span className="faint"> · optional</span>}</span>
          <span className="faint mono">{FOLDERS[f.key] || ""}</span>
        </div>
      ))}
      <div className="help">Any file name and precision works (bf16, fp8_scaled, int8_convrot, nvfp4). Each file is checked when you pick it.</div>
    </div>
  );
}

function defaultsFor(f: Family | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  f?.profile_fields.forEach((x) => { if (x.default != null) out[x.key] = x.default; });
  return out;
}

function ProfileEditor(props: { families: Family[]; initial: Partial<Profile> | null; onClose: () => void; onSaved: () => void }) {
  const toast = useToast();
  const [name, setName] = useState(props.initial?.name || "");
  const [family, setFamily] = useState(props.initial?.family || props.families[0]?.id || "");
  const fam = props.families.find((f) => f.id === family);
  const [data, setData] = useState<Record<string, unknown>>(props.initial?.data || defaultsFor(fam));
  const [checks, setChecks] = useState<Validation[]>([]);
  const [busy, setBusy] = useState(false);
  // the files on this machine and the ones to download, for each path field (a server older than this page has none)
  const [models, reloadModels] = usePoll<ModelsOverview>("/api/models", 0);
  const [download, setDownload] = useState<string | null>(null); // the field whose Download dialog is open
  const fileFields = models?.families.find((f) => f.id === family)?.fields || [];
  // a file being downloaded is checked again every few seconds, to show how far it's got
  const downloading = checks.some((c) => c.level === "warning" && /^(Downloading|Waiting)/.test(c.message));

  useEffect(() => {
    const check = () => api.post<Validation[]>("/api/profiles/validate", { name, family, data }).then(setChecks, () => {});
    const t = setTimeout(check, 350);
    const again = downloading ? setInterval(check, 3000) : undefined;
    return () => { clearTimeout(t); clearInterval(again); };
  }, [family, JSON.stringify(data), downloading]);

  const set = (k: string, v: unknown) => setData((d) => ({ ...d, [k]: v }));
  const save = async () => {
    setBusy(true);
    try {
      const body = { name: name.trim(), family, data };
      if (props.initial?.id) await api.put(`/api/profiles/${props.initial.id}`, body);
      else await api.post("/api/profiles", body);
      toast(`Saved profile "${name}"`);
      props.onSaved();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  const errors = checks.filter((c) => c.level === "error").length;

  return (
    <Modal
      title={props.initial?.id ? `Edit profile` : "New profile"}
      onClose={props.onClose}
      footer={
        <>
          {errors > 0 && <span className="validation error" style={{ marginRight: "auto" }}>{errors} problem{errors > 1 ? "s" : ""}: you can still save</span>}
          <button className="btn ghost" onClick={props.onClose}>Cancel</button>
          <button className="btn primary" disabled={!name.trim() || busy} onClick={save}>Save profile</button>
        </>
      }
    >
      <div className="stack profile-editor">
        <div className="grid cols-2">
          <label className="field">
            <span>Profile name</span>
            <input autoFocus value={name} placeholder="My model" onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="field">
            <span>Model family</span>
            <select value={family} onChange={(e) => { setFamily(e.target.value); setData(defaultsFor(props.families.find((f) => f.id === e.target.value))); }}>
              {props.families.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
            </select>
          </label>
        </div>
        {fam && <Needs fam={fam} />}
        {fam?.profile_fields.map((f) => {
          const v = checks.find((c) => c.field === f.key);
          const mf = fileFields.find((x) => x.key === f.key);
          const value = String(data[f.key] ?? "");
          // files on this machine it could be, offered while the field is empty or its file is wrong
          const found = (mf?.found || []).filter((x) => !x.unsupported.length && x.path !== value).slice(0, 4);
          return (
            <label className="field" key={f.key} onClick={(e) => f.kind === "path" && e.preventDefault()}>
              <span>{f.label}{!f.required && <span className="faint"> · optional</span>}{f.hint && <span className="field-hint">{f.hint}</span>}</span>
              {f.kind === "path" ? (
                <PathInput value={value} onChange={(x) => set(f.key, x)} exts=".safetensors,.sft,.gguf" validation={v}
                  placeholder={f.hint ? `${f.hint} (.safetensors)` : undefined}
                  extra={mf && mf.downloads.length > 0 && (
                    <button type="button" className="btn" title="Download it from Hugging Face" onClick={() => setDownload(f.key)}><IconDownload /> Download</button>
                  )} />
              ) : f.kind === "select" ? (
                <select value={String(data[f.key] ?? f.default ?? "")} onChange={(e) => set(f.key, e.target.value)}>
                  {(f.options || []).map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              ) : f.kind === "number" ? (
                <input type="number" value={String(data[f.key] ?? "")} onChange={(e) => set(f.key, e.target.value === "" ? "" : Number(e.target.value))} />
              ) : (
                <input value={String(data[f.key] ?? "")} onChange={(e) => set(f.key, e.target.value)} />
              )}
              {f.kind === "path" && found.length > 0 && (!value.trim() || v?.level === "error") && (
                <div className="row wrap" style={{ gap: 6 }}>
                  <span className="help">On this machine:</span>
                  {found.map((x) => (
                    <button key={x.path} type="button" className="chip" title={x.source === "hf_cache" ? `${x.path}\nIn the Hugging Face cache, from ${x.repo}` : x.path}
                      onClick={() => set(f.key, x.path)}>
                      {/* the cache's are named as on the hub: a file by its own name, a folder of shards as "owner/repo" */}
                      {(x.source === "hf_cache" && x.name.endsWith(".safetensors") ? x.name.split("/").pop()! : x.name).replace(/\.safetensors$/, "")}
                      {x.source === "hf_cache" && <span className="pill chip-tag">HF cache</span>}
                    </button>
                  ))}
                </div>
              )}
              {f.help && <span className="help">{f.help}</span>}
            </label>
          );
        })}
      </div>
      {download && (
        <DownloadDialog family={family} field={download} variant={data.variant ? String(data.variant) : undefined}
          onClose={() => setDownload(null)}
          onUse={(path) => set(download, path)}
          onStarted={(t) => { set(t.field, t.dest); reloadModels(); }} />
      )}
    </Modal>
  );
}

const FAMILY_FILTER = "dainamo.profiles.family";

/** A setting as a card shows it: a file's name, "None" for no file, a plain word capitalized ("off" reads "Off"). */
function shownValue(f: ProfileField, v: unknown) {
  const s = String(v ?? "").trim();
  if (!s) return "None";
  if (f.kind === "path") return s.split(/[\\/]/).pop() || s;
  return /^[a-z]+$/.test(s) ? s[0].toUpperCase() + s.slice(1) : s;
}

function ProfileCard({ p, fam, onEdit, onDuplicate, onDelete }: {
  p: Profile; fam?: Family; onEdit: () => void; onDuplicate: () => void; onDelete: () => void;
}) {
  return (
    <div className="card pf-card">
      <div className="pf-head">
        <div className="pf-icon"><IconLayers /></div>
        <div className="pf-title">
          <h3 className="ellipsis" title={p.name}>{p.name}</h3>
          <span className="pf-family" title={fam?.label || p.family}>{familyBadge(fam?.label || p.family)}</span>
        </div>
        <button className="btn pf-edit" onClick={onEdit}><IconEdit /> Edit</button>
        <Menu boxed label={`More for ${p.name}`} items={[
          { label: "Duplicate", icon: <IconCopy />, onClick: onDuplicate },
          { label: "Delete", icon: <IconTrash />, danger: true, onClick: onDelete },
        ]} />
      </div>
      {fam ? fam.profile_fields.filter((f) => f.kind === "path" || f.kind === "select").map((f) => {
        const v = p.data[f.key] ?? f.default;
        return (
          <div className="pf-row" key={f.key}>
            <span>{f.label.replace(" (CLIP)", "")}</span>
            <span className="mono ellipsis" title={String(v ?? "")}>{shownValue(f, v)}</span>
          </div>
        );
      }) : <p className="help">This version of Dainamo doesn't train {p.family}.</p>}
    </div>
  );
}

export default function Profiles({ families, view }: { families: Family[]; view: string }) {
  view = ["files", "merge"].includes(view) ? view : "";
  const toast = useToast();
  const [profiles, reload, loadError] = usePoll<Profile[]>("/api/profiles", 0);
  const [editing, setEditing] = useState<Partial<Profile> | null | undefined>(undefined);
  const [deleting, setDeleting] = useState<Profile | null>(null);
  const [search, setSearch] = useState("");
  const [family, setFamilyState] = useState(() => {
    try {
      return localStorage.getItem(FAMILY_FILTER) || "";
    } catch {
      return "";
    }
  });
  const setFamily = (f: string) => {
    setFamilyState(f);
    try {
      localStorage.setItem(FAMILY_FILTER, f);
    } catch { /* not kept */ }
  };

  const remove = async (p: Profile) => {
    setDeleting(null);
    try {
      await api.del(`/api/profiles/${p.id}`);
      toast(`Deleted profile "${p.name}"`);
    } catch (e: any) {
      toast(e.message, true);
    }
    reload();
  };

  // in family order, so a family's profiles sit together
  const groups = byFamily(profiles || [], families);
  const fam = groups.some((g) => g.id === family) ? family : ""; // a remembered family with no profiles left shows all
  const words = search.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = groups.filter((g) => !fam || g.id === fam).flatMap((g) => g.profiles).filter((p) => {
    // a profile's name, its family, and its files' names: which profiles use this file?
    const files = Object.values(p.data).map((v) => String(v ?? "").split(/[\\/]/).pop());
    return words.every((w) => [p.name, groups.find((g) => g.id === p.family)?.label, ...files].join(" ").toLowerCase().includes(w));
  });

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Models</h1>
          <p>Profiles are the model files a job trains with. Model files finds and downloads them; Merge &amp; extract makes LoRAs from LoRAs and finetunes.</p>
        </div>
        {view === "" && <button className="btn primary lg" onClick={() => setEditing(null)} disabled={!families.length}><IconPlus /> New profile</button>}
      </div>

      <div className="view-tabs" role="tablist" aria-label="Models">
        {([["", "Profiles"], ["files", "Model files"], ["merge", "Merge & extract"]] as const).map(([v, label]) => (
          <button key={v} role="tab" aria-selected={view === v} className={view === v ? "on" : ""} onClick={() => go("profiles", v)}>
            {label}
            {v === "" && profiles && profiles.length > 0 && <span className="count">{profiles.length}</span>}
          </button>
        ))}
      </div>

      {view === "files" && <ModelFiles families={families} onCreate={setEditing} />}
      {view === "merge" && <MergeExtract families={families} />}

      {view === "" && loadError && !profiles && <LoadError what="the profiles" error={loadError} onRetry={reload} />}
      {view === "" && profiles && profiles.length === 0 && (
        <Empty icon={<IconLayers width={44} height={44} />} title="No profiles yet" action={
          <div className="row" style={{ justifyContent: "center" }}>
            <button className="btn primary lg" onClick={() => setEditing(null)} disabled={!families.length}><IconPlus /> New profile</button>
            <button className="btn lg" onClick={() => go("profiles", "files")}>Model files</button>
          </div>
        }>
          A profile is a model setup: the diffusion model, text encoder and VAE files to train with. ComfyUI model files
          work directly. Model files shows which you have, and downloads the rest.
        </Empty>
      )}

      {view === "" && !!profiles?.length && (
        <>
          <div className="ds-toolbar pf-toolbar">
            <label className="search">
              <IconSearch />
              <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search profiles…" aria-label="Search profiles"
                onKeyDown={(e) => e.key === "Escape" && search && (e.stopPropagation(), setSearch(""))} />
            </label>
            <select value={fam} onChange={(e) => setFamily(e.target.value)} aria-label="Model family">
              <option value="">All model families</option>
              {groups.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
            </select>
          </div>
          {!shown.length && (
            <div className="empty">
              <strong>No profiles match</strong>
              <button className="btn ghost" onClick={() => { setSearch(""); setFamily(""); }}>Clear the search{fam ? " and filter" : ""}</button>
            </div>
          )}
          <div className="grid cols-2 pf-grid">
            {shown.map((p) => (
              <ProfileCard key={p.id} p={p} fam={families.find((f) => f.id === p.family)}
                onEdit={() => setEditing(p)}
                onDuplicate={() => setEditing({ name: p.name + " copy", family: p.family, data: { ...p.data } })}
                onDelete={() => setDeleting(p)} />
            ))}
          </div>
        </>
      )}

      {editing !== undefined && (
        <ProfileEditor families={families} initial={editing} onClose={() => setEditing(undefined)} onSaved={() => { setEditing(undefined); reload(); }} />
      )}
      {deleting && (
        <ConfirmDialog title={`Delete ${deleting.name}`} danger confirmLabel="Delete" onClose={() => setDeleting(null)} onConfirm={() => remove(deleting)}
          message="Delete this profile? Its model files aren't touched, and jobs made with it keep their settings." />
      )}
    </div>
  );
}

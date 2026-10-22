import { useEffect, useState } from "react";
import { api, byFamily, type Family, type ModelsOverview, type Profile } from "../api";
import { go } from "../App";
import { IconCopy, IconDownload, IconEdit, IconLayers, IconPlus, IconTrash } from "../components/icons";
import { DownloadDialog } from "../components/ModelDownload";
import { Modal, PathInput, usePoll, useToast } from "../components/ui";
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
          {errors > 0 && <span className="validation error" style={{ marginRight: "auto" }}>{errors} problem{errors > 1 ? "s" : ""} — you can still save</span>}
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
                    <button key={x.path} type="button" className="chip" title={x.path} onClick={() => set(f.key, x.path)}>
                      {x.name.replace(/\.safetensors$/, "")}
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

export default function Profiles({ families, view }: { families: Family[]; view: string }) {
  const toast = useToast();
  const [profiles, reload] = usePoll<Profile[]>("/api/profiles", 0);
  const [editing, setEditing] = useState<Partial<Profile> | null | undefined>(undefined);

  const remove = async (p: Profile) => {
    if (!confirm(`Delete profile "${p.name}"? Model files are not touched.`)) return;
    await api.del(`/api/profiles/${p.id}`);
    toast("Profile deleted");
    reload();
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Profiles</h1>
          <p>A profile is a model setup: which diffusion model, text encoder and VAE files to train with. ComfyUI model files work directly.</p>
        </div>
        <button className="btn primary" onClick={() => setEditing(null)} disabled={!families.length}><IconPlus /> New profile</button>
      </div>

      <div className="view-tabs" role="tablist" aria-label="Profiles">
        {([["", "Profiles"], ["files", "Model files"]] as const).map(([v, label]) => (
          <button key={v} role="tab" aria-selected={view === v} className={view === v ? "on" : ""} onClick={() => go("profiles", v)}>
            {label}
            {v === "" && profiles && profiles.length > 0 && <span className="count">{profiles.length}</span>}
          </button>
        ))}
      </div>

      {view === "files" && <ModelFiles families={families} onCreate={setEditing} />}

      {view !== "files" && profiles && profiles.length === 0 && (
        <div className="empty">
          <strong>No profiles yet</strong>
          Create one to point Dainamo at your model files, e.g. the ones in your ComfyUI models folder. Model files shows which
          you have, and downloads the rest.
          <div className="row" style={{ marginTop: 14, justifyContent: "center" }}>
            <button className="btn primary" onClick={() => setEditing(null)}><IconPlus /> New profile</button>
            <button className="btn" onClick={() => go("profiles", "files")}>Model files</button>
          </div>
        </div>
      )}

      {view !== "files" && byFamily(profiles || [], families).map((g) => (
        <section className="pf-group" key={g.id}>
          <div className="pf-group-head">
            <h2>{g.label}</h2>
            <span className="count">{g.profiles.length}</span>
          </div>
          <div className="grid cols-2">
            {g.profiles.map((p) => {
              const fam = families.find((f) => f.id === p.family);
              return (
                <div className="card card-pad" key={p.id}>
                  <div className="row" style={{ marginBottom: 12 }}>
                    <div className="ds-cover" style={{ width: 40, height: 40, color: "var(--accent)" }}><IconLayers width={20} /></div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: 15 }} className="ellipsis">{p.name}</div>
                    </div>
                    <div className="spacer" />
                    <button className="btn ghost icon" title="Duplicate" onClick={() => setEditing({ name: p.name + " copy", family: p.family, data: { ...p.data } })}><IconCopy /></button>
                    <button className="btn ghost icon" title="Edit" onClick={() => setEditing(p)}><IconEdit /></button>
                    <button className="btn ghost icon danger" title="Delete" onClick={() => remove(p)}><IconTrash /></button>
                  </div>
                  {fam?.profile_fields.filter((f) => f.kind === "path" || f.kind === "select").map((f) => (
                    <div className="kv" key={f.key}>
                      <span>{f.label}</span>
                      <span className="mono ellipsis" title={String(p.data[f.key] || "")} style={{ maxWidth: "70%" }}>
                        {String(p.data[f.key] || "—").split("/").pop()}
                      </span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </section>
      ))}

      {editing !== undefined && (
        <ProfileEditor families={families} initial={editing} onClose={() => setEditing(undefined)} onSaved={() => { setEditing(undefined); reload(); }} />
      )}
    </div>
  );
}

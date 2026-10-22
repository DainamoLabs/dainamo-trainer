import { useEffect, useState } from "react";
import { api, type Family, type ModelFamily, type ModelField, type ModelsOverview, type Profile } from "../api";
import { IconAlert, IconCheck, IconDownload } from "../components/icons";
import { ACTIVE, DownloadDialog, TaskRow } from "../components/ModelDownload";
import { usePoll, useToast } from "../components/ui";

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

function FieldRow({ f, overview, searched, onDownload }: { f: ModelField; overview: ModelsOverview; searched: boolean; onDownload: () => void }) {
  const usable = f.found.filter((x) => !x.unsupported.length);
  const task = overview.tasks.find((t) => ACTIVE.includes(t.state) && f.downloads.some((o) => o.id === t.download));
  return (
    <div className="mf-field">
      <div className="mf-label">
        {f.label.replace(" (CLIP)", "")}{!f.required && <span className="faint"> · optional</span>}
        <div className="faint small">{f.hint}</div>
      </div>
      <div className="mf-files">
        {usable.slice(0, 2).map((x) => (
          <div className="mf-file" key={x.path} title={x.where}>
            <IconCheck width={13} height={13} />
            <span className="mono ellipsis">{x.name.replace(/\.safetensors$/, "")}</span>
            {/* the precision, unless the name says it */}
            {x.quant.some((l) => !x.name.toLowerCase().replace(/[-_]/g, " ").includes(l)) && (
              <span className="faint" style={{ flex: "none" }}>{x.quant.join(", ")}</span>
            )}
          </div>
        ))}
        {usable.length > 2 && <div className="faint small">+{usable.length - 2} more</div>}
        {!usable.length && !task && (searched
          ? (
            <div className={"small " + (f.required ? "validation warning" : "muted")}>
              {f.required && <IconAlert width={13} height={13} />}{f.required ? "Not found" : "None found"}
            </div>
          )
          : <div className="small faint" title="There's no models folder to look in yet">Not searched</div>)}
        {task && (
          <div className="mf-file">
            <span className="muted small">Downloading {task.name} · {Math.floor((100 * task.done) / (task.size || 1))}%</span>
          </div>
        )}
      </div>
      {f.downloads.length > 0
        ? <button className="btn sm" onClick={onDownload} title="Download a file for this from Hugging Face"><IconDownload /> Download</button>
        : <span />}
    </div>
  );
}

/** Profiles → Model files: what each family needs, what's on this machine, and downloads of the rest. */
export default function ModelFiles({ families, onCreate }: { families: Family[]; onCreate: (p: Partial<Profile>) => void }) {
  const [ov, reload, err] = usePoll<ModelsOverview>("/api/models", 0);
  const [dialog, setDialog] = useState<{ family: string; field: string } | null>(null);
  const [suggested, setSuggested] = useState<string[]>([]); // models folders the profiles' files are in
  const [saving, setSaving] = useState(false);
  const toast = useToast();
  const active = !!ov?.tasks.some((t) => ACTIVE.includes(t.state));
  useEffect(() => {
    if (!active) return;
    const t = setInterval(reload, 1500);
    return () => clearInterval(t);
  }, [active, reload]);
  useEffect(() => {
    if (ov && !ov.folder) api.get<{ folders: string[] }>("/api/models/suggest").then((r) => setSuggested(r.folders), () => {});
  }, [ov?.folder]);

  // saved straight away, unlike on the Settings page, which saves everything at once
  const setFolder = async (folder: string) => {
    setSaving(true);
    try {
      await api.put("/api/settings", { models: { folder } });
      toast("Models folder set");
      reload();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setSaving(false);
    }
  };

  if (err && !ov) return <div className="error-box">{err.includes("404") ? "The Dainamo server is older than this page: restart it, then reload." : err}</div>;
  if (!ov) return <div className="muted">Looking for model files…</div>;
  const searched = ov.folder_ok; // otherwise only the files profiles use were looked at
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="mf-head">
        <div style={{ minWidth: 0, flex: "1 1 320px" }}>
          <div className="muted small">ComfyUI models folder</div>
          {ov.folder
            ? <div className="mono ellipsis" title={ov.folder}>{ov.folder}{!ov.folder_ok && <span className="validation error"> · not found</span>}</div>
            : <div className="validation warning"><IconAlert width={13} height={13} /> Not set, so only the files your profiles use were looked for</div>}
          {!ov.folder && suggested[0] && (
            <div className="small muted" style={{ overflowWrap: "anywhere" }}>Your profiles use files in <span className="mono">{suggested[0]}</span></div>
          )}
        </div>
        <div className="small muted" style={{ flex: "1 1 220px" }}>
          Hugging Face: {ov.token ? "signed in" : "not signed in (needed only for LTX 2.5)"}
        </div>
        <div className="row" style={{ gap: 8 }}>
          {!ov.folder && suggested[0] && <button className="btn sm primary" disabled={saving} onClick={() => setFolder(suggested[0])}>Use this folder</button>}
          <a className="btn sm" href="#/settings">{ov.folder ? "Change in Settings" : suggested[0] ? "Choose another" : "Set it in Settings"}</a>
        </div>
      </div>

      {ov.tasks.length > 0 && (
        <div className="card card-pad stack" style={{ gap: 12 }}>
          <h2 style={{ margin: 0 }}>Downloads</h2>
          {ov.tasks.map((t) => <TaskRow key={t.id} t={t} onChange={reload} />)}
        </div>
      )}

      <div className="grid cols-2 mf-grid">
        {ov.families.map((fam) => {
          const needed = fam.fields.filter((f) => f.required);
          const have = needed.filter((f) => f.found.some((x) => !x.unsupported.length)).length;
          const all = have === needed.length;
          return (
            <div className="card card-pad" key={fam.id}>
              <div className="row" style={{ marginBottom: 6 }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 15 }} className="ellipsis">{fam.label}</div>
                  <div className={"small " + (all || !searched ? "muted" : "validation warning")} style={{ display: "block" }}>
                    {all ? "Everything it needs is here" : searched ? `${have} of the ${needed.length} files it needs` : "Not searched yet"}
                  </div>
                </div>
                <div className="spacer" />
                <button className="btn sm" disabled={!all} title={all ? undefined : searched ? "Download the missing files first" : "Set your models folder first"}
                  onClick={() => onCreate(profileFrom(fam, families.find((f) => f.id === fam.id)))}>Create profile</button>
              </div>
              {fam.fields.map((f) => (
                <FieldRow key={f.key} f={f} overview={ov} searched={searched} onDownload={() => setDialog({ family: fam.id, field: f.key })} />
              ))}
            </div>
          );
        })}
      </div>
      <p className="help">
        Files are recognised by what's in them, whatever they're called, in the models folder's diffusion_models (or unet),
        text_encoders (or clip), vae, loras and checkpoints folders, along with the files your profiles use. Downloads come from
        Hugging Face, from the commit Dainamo was checked against, and each is checked against its published size and checksum.
      </p>

      {dialog && <DownloadDialog family={dialog.family} field={dialog.field} onClose={() => setDialog(null)} onStarted={reload} />}
    </div>
  );
}

import { useState } from "react";
import { api, fmtLeft, q, type ModelDownloadOption, type ModelsOverview, type ModelTask } from "../api";
import { IconDownload } from "./icons";
import { ConfirmDialog, fmtSize, Modal, PathInput, TaskProgress, usePoll, useToast } from "./ui";

export const ACTIVE: ModelTask["state"][] = ["queued", "checking", "downloading", "verifying"];
const SPARE = 2 * 1024 ** 3; // what the server leaves free on a disk

type Space = { free: number | null; exists: boolean; in_models: boolean };

/** Pick one of the files a family's field can use and download it from Hugging Face, into the models folder's
 *  subfolder for its kind (set in Model files) or a folder chosen here. ``onUse`` offers the copy already on this machine.
 *  ``variant`` and ``model`` (the profile's, e.g. FLUX.2 Klein's "9B") leave out the files for others. */
export function DownloadDialog(props: {
  family: string; field: string; variant?: string; model?: string; onClose: () => void;
  onStarted?: (t: ModelTask) => void; onUse?: (path: string) => void;
}) {
  const toast = useToast();
  const [ov] = usePoll<ModelsOverview>("/api/models", 0);
  const fam = props.family === ov?.upscalers?.id ? ov.upscalers : ov?.families.find((f) => f.id === props.family);
  const fld = fam?.fields.find((f) => f.key === props.field);
  const opts = (fld?.downloads || []).filter((o) => (!props.variant || !o.variant || o.variant === props.variant)
    && (!props.model || !o.model || o.model === props.model));
  const [sel, setSel] = useState("");
  const cur: ModelDownloadOption | undefined = opts.find((o) => o.id === sel) || opts.find((o) => o.recommended) || opts[0];
  const [dest, setDest] = useState<string | null>(null); // null: the chosen file's folder in the models folder
  const folder = dest ?? (cur && ov ? ov.folders[cur.folder] || "" : "");
  const [space] = usePoll<Space>(folder.trim() ? `/api/models/space?folder=${q(folder.trim())}` : null, 0);
  const [busy, setBusy] = useState(false);
  const short = !!cur && space?.free != null && space.free - cur.size < SPARE;

  const start = async () => {
    if (!cur) return;
    setBusy(true);
    try {
      const t = await api.post<ModelTask>("/api/models/downloads", { id: cur.id, folder: folder.trim() });
      toast(`Downloading ${t.name}`);
      props.onStarted?.(t);
      props.onClose();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title={fam && fld ? (fam === ov?.upscalers ? "Download an upscaler" : `${fld.label.replace(" (CLIP)", "")} for ${fam.label.replace(/ \(.*\)$/, "")}`) : "Download"}
      onClose={props.onClose}
      footer={
        <>
          <button className="btn ghost" onClick={props.onClose}>Cancel</button>
          {cur?.have ? (
            props.onUse
              ? <button className="btn primary" onClick={() => { props.onUse!(cur.have!); props.onClose(); }}>Use this file</button>
              : <button className="btn primary" disabled>Already on this machine</button>
          ) : cur?.task ? (
            <button className="btn primary" disabled>Already downloading</button>
          ) : (
            <button className="btn primary" disabled={!cur || !folder.trim() || busy || short} onClick={start}>
              <IconDownload /> Download{cur ? ` · ${fmtSize(cur.size)}` : ""}
            </button>
          )}
        </>
      }
    >
      {!ov ? <div className="muted">Loading…</div> : !opts.length ? (
        <div className="muted">Dainamo doesn't know where to download this one from. Browse to the file instead.</div>
      ) : (
        <div className="stack">
          <div className="stack" style={{ gap: 6 }} role="radiogroup" aria-label="File">
            {opts.map((o) => (
              <label key={o.id} className={"ds-pick mf-opt" + (o.id === cur?.id ? " on" : "")}>
                <input type="radio" name="model-file" checked={o.id === cur?.id} onChange={() => setSel(o.id)} style={{ accentColor: "var(--accent)" }} />
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600 }}>{o.label}</div>
                  <div className="faint small mono ellipsis" title={o.name}>{o.name}</div>
                </div>
                <div className="mf-opt-side">
                  <span className="small">{fmtSize(o.size)}</span>
                  {o.have ? <span className="pill completed">On this machine</span>
                    : o.task ? <span className="pill queued">Downloading</span>
                    : o.recommended ? <span className="pill accent">Recommended</span> : null}
                </div>
              </label>
            ))}
          </div>
          {cur && (cur.have ? (
            <div className="help" style={{ overflowWrap: "anywhere" }}>It's at <span className="mono">{cur.have}</span></div>
          ) : (
            <div className="field">
              <span>Save to</span>
              <PathInput mode="dir" value={folder} onChange={setDest} placeholder="A folder for the file" />
              <span className="help">
                {space?.free != null && <span className={short ? "validation error" : ""}>{fmtSize(space.free)} free on that disk{short ? `: not enough for ${fmtSize(cur.size)} (Dainamo leaves 2 GB spare)` : ""}. </span>}
                {!ov.folder
                  ? <>Set your ComfyUI models folder in <a href="#/profiles/files">Model files</a> and this is filled in with the right subfolder. </>
                  : space && !space.in_models
                    ? <span className="validation warning">Outside your models folder: ComfyUI won't see the file there unless its extra_model_paths.yaml lists it. </span>
                    : null}
                From <a href={cur.url} target="_blank" rel="noreferrer">{cur.repo}</a> on Hugging Face, checked once it's downloaded.
              </span>
            </div>
          ))}
          {cur?.gated && !cur.have && (
            <div className={"validation " + (ov.token ? "ok" : "warning")} style={{ display: "block", lineHeight: 1.5 }}>
              {cur.repo} needs a Hugging Face account that has accepted its terms at{" "}
              <a href={cur.url} target="_blank" rel="noreferrer">its page</a>.{" "}
              {ov.token ? "A Hugging Face sign-in was found on this machine." : <>Sign in on this machine first with <code>hf auth login</code> (or set HF_TOKEN).</>}
            </div>
          )}
          {cur?.note && <div className="help">{cur.note}</div>}
        </div>
      )}
    </Modal>
  );
}

const STATE_LABEL: Partial<Record<ModelTask["state"], string>> = {
  queued: "Waiting for the download before it", checking: "Checking the part downloaded before",
  verifying: "Checking the file", done: "Downloaded",
};

/** A download: its progress, and Pause (it resumes from where it stopped) / Resume / Remove. */
export function TaskRow({ t, onChange }: { t: ModelTask; onChange: () => void }) {
  const toast = useToast();
  const [asking, setAsking] = useState(false);
  const active = ACTIVE.includes(t.state);
  const act = async (what: "cancel" | "resume" | "remove") => {
    try {
      if (what === "remove") await api.del(`/api/models/downloads/${t.id}`);
      else await api.post(`/api/models/downloads/${t.id}/${what}`);
    } catch (e: any) {
      toast(e.message, true);
    }
    onChange();
  };
  // a partial download is deleted by Remove: asked first
  const remove = () => (t.state !== "done" && t.done > 0 ? setAsking(true) : act("remove"));
  const left = t.speed > 0 ? fmtLeft((t.size - t.done) / t.speed) : "";
  const status = t.state === "downloading"
    ? `${fmtSize(t.done)} of ${fmtSize(t.size)}${t.speed > 0 ? ` · ${fmtSize(t.speed)}/s · ${left}` : ""}`
    : t.state === "failed" ? t.error
    : t.state === "cancelled" || t.state === "interrupted"
      ? `Paused at ${fmtSize(t.done)} of ${fmtSize(t.size)}${t.state === "interrupted" ? " when the server stopped" : ""}`
      : STATE_LABEL[t.state];
  return (
    <div className="mf-task">
      <TaskProgress running={active} bar={t.state !== "done" && t.state !== "failed"}
        pct={t.state === "queued" || t.state === "verifying" || t.state === "checking" ? null : t.size ? (100 * t.done) / t.size : 0}
        label={<span className="mf-task-name" title={t.dest}>{t.name}</span>}
        detail={<span className={t.state === "failed" ? "validation error" : undefined}>{t.family_label.replace(/ \(.*\)$/, "")} · {t.label} · {status}</span>}
        onCancel={() => act("cancel")} cancelLabel="Pause"
        actions={!active && <>
          {t.state !== "done" && <button className="btn sm" onClick={() => act("resume")}>Resume</button>}
          <button className="btn sm ghost" onClick={remove}>{t.state === "done" ? "Clear" : "Remove"}</button>
        </>} />
      {asking && (
        <ConfirmDialog title="Remove the download" danger confirmLabel="Remove" onClose={() => setAsking(false)} onConfirm={() => act("remove")}
          message={`The ${fmtSize(t.done)} of ${t.name} downloaded so far is deleted, and the download starts over if you download it again.`} />
      )}
    </div>
  );
}

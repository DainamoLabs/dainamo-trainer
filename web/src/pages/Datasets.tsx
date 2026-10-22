import { useEffect, useRef, useState } from "react";
import { go } from "../App";
import { api, byFamily, fmtTime, q, type Dataset, type Family, type Gpu, type MediaItem, type Profile, type Settings, type WhisperModel } from "../api";
import { CheckIssues, useDatasetCheck } from "../components/DatasetCheck";
import { IconDownload, IconEdit, IconImages, IconPlus, IconScissors, IconSparkle, IconTrash, IconUpload, IconX } from "../components/icons";
import { ConfirmDialog, Info, Lightbox, Modal, PathInput, ProfileOptions, fmtSize, usePoll, useToast } from "../components/ui";

type SplitStatus = {
  running: boolean; total: number; done: number; clips: number; current?: string; message?: string; errors: string[];
  cancelled?: boolean; target: { id: string; name: string } | null;
};
type ClipFit = { frames: number | null; fps: number; lengths: { frames: number; seconds: number; peak_gb: number }[]; error?: string };

type DownloadStatus = {
  running: boolean; phase?: "starting" | "downloading" | "processing" | ""; title?: string; step?: string; item?: number; items?: number;
  done_bytes?: number; total_bytes?: number; saved: string[]; count: number; errors: string[]; cancelled: boolean;
};
type YtdlpInfo = { installed: boolean; version: string | null; current: boolean; minimum: string; bundled: boolean; network: boolean };

type CapStatus = {
  running: boolean; total: number; done: number; errors: string[]; current?: string;
  phase?: "starting" | "transcribing" | "captioning"; heard?: number; to_hear?: number; message?: string;
};

function NewDataset({ onClose, onCreated }: { onClose: () => void; onCreated: (d: Dataset) => void }) {
  const toast = useToast();
  const [name, setName] = useState("");
  const [source, setSource] = useState<"upload" | "folder">("folder");
  const [path, setPath] = useState("");
  const [trigger, setTrigger] = useState("");
  const create = async () => {
    try {
      const d = await api.post<Dataset>("/api/datasets", { name, path: source === "folder" ? path : "", trigger });
      onCreated(d);
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  return (
    <Modal title="New dataset" onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={!name.trim() || (source === "folder" && !path.trim())} onClick={create}>Create dataset</button>
      </>
    }>
      <div className="stack">
        <label className="field">
          <span>Name</span>
          <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="my_character" />
        </label>
        <div className="field">
          <span className="muted small" style={{ fontWeight: 500 }}>Source</span>
          <div className="seg" style={{ alignSelf: "flex-start" }}>
            <button className={source === "folder" ? "on" : ""} onClick={() => setSource("folder")}>Existing folder</button>
            <button className={source === "upload" ? "on" : ""} onClick={() => setSource("upload")}>Upload files</button>
          </div>
          <span className="help">
            {source === "folder"
              ? "Images/videos are used in place; captions are saved as .txt files next to each one."
              : "Dainamo creates a folder in its data directory; you'll upload files on the next screen."}
          </span>
        </div>
        {source === "folder" && (
          <label className="field" onClick={(e) => e.preventDefault()}>
            <span>Folder</span>
            <PathInput mode="dir" value={path} onChange={setPath} placeholder="/path/to/images" />
          </label>
        )}
        <label className="field">
          <span>Trigger word <span className="faint">· optional</span></span>
          <input value={trigger} onChange={(e) => setTrigger(e.target.value)} placeholder="e.g. ohwx" />
          <span className="help">Added to the start of every caption (auto-captioning and at training time) if missing.</span>
        </label>
      </div>
    </Modal>
  );
}

function CaptionCell({ item, onSaved }: { item: MediaItem; onSaved: (c: string) => void }) {
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(item.caption);
  useEffect(() => setVal(item.caption), [item.caption]);
  const save = async () => {
    setEditing(false);
    if (val.trim() === item.caption) return;
    try {
      await api.put("/api/caption", { path: item.path, caption: val });
      onSaved(val.trim());
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  if (editing)
    return (
      <textarea autoFocus value={val} onChange={(e) => setVal(e.target.value)} onBlur={save}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) save(); if (e.key === "Escape") { setVal(item.caption); setEditing(false); } }}
        style={{ border: 0, borderRadius: 0, minHeight: 110, fontSize: 12 }} />
    );
  return (
    <div className={"media-cap" + (item.caption ? "" : " none")} onClick={() => setEditing(true)} title="Click to edit">
      {item.caption || "No caption — click to write one"}
    </div>
  );
}

const CHECK_FAMILY = "dainamo.check.family";

/** The health check, for a chosen model and resolution; it runs when asked, then again as they change. */
function HealthCheck({ id, families, onOpen }: { id: string; families: Family[]; onOpen: (path: string) => void }) {
  const [familyId, setFamilyId] = useState(() => { try { return localStorage.getItem(CHECK_FAMILY) || ""; } catch { return ""; } });
  const family = families.find((f) => f.id === familyId) || families[0];
  const [resolution, setResolution] = useState("");
  const [seconds, setSeconds] = useState("");
  const [on, setOn] = useState(false);
  const music = !!family?.media.includes("audio");
  const { result, error, loading, recheck } = useDatasetCheck(on ? id : null, {
    resolution: music ? "" : resolution, family: family?.id, seconds: music && seconds.trim() !== "" ? Number(seconds) : undefined });
  const pick = (v: string) => { setFamilyId(v); try { localStorage.setItem(CHECK_FAMILY, v); } catch { /* not remembered */ } };
  const counts = result ? (["error", "warning", "info"] as const).map((l) => [l, result.issues.filter((i) => i.level === l).length] as const) : [];
  const words = { error: ["error", "errors"], warning: ["warning", "warnings"], info: ["note", "notes"] };
  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <div className="row wrap health-head">
        <h2 style={{ margin: 0 }}>Health check</h2>
        <span className="muted small">for</span>
        <select value={family?.id || ""} onChange={(e) => pick(e.target.value)} aria-label="Model">
          {families.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
        </select>
        {music ? <>
          <span className="muted small">in clips of</span>
          <input type="number" min={0} value={seconds} onChange={(e) => setSeconds(e.target.value)} aria-label="Clip length" style={{ width: 80 }}
            placeholder={String(family?.training_defaults?.clip_seconds ?? 30)} title="Clip length in seconds (0: whole songs)" />
          <span className="muted small">s</span>
        </> : <>
          <span className="muted small">at</span>
          <input value={resolution} onChange={(e) => setResolution(e.target.value)} aria-label="Resolution" style={{ width: 110 }}
            placeholder={String(family?.training_defaults?.resolution ?? 1024)} title="Training resolution (several: 512, 1024)" />
        </>}
        <span className="spacer" />
        {result && !loading && (
          <span className="small muted">{counts.filter(([, n]) => n).map(([l, n]) => `${n} ${words[l][n === 1 ? 0 : 1]}`).join(" · ") || "All good"}</span>
        )}
        <button className="btn sm" disabled={loading} onClick={() => (on ? recheck() : setOn(true))}>{loading ? "Checking…" : on ? "Check again" : "Check dataset"}</button>
      </div>
      {!on && <div className="help" style={{ marginTop: 8 }}>{music
        ? "Looks for unreadable files, songs too short to use, low sample rates, mono files, missing or empty captions, and captions shared by two files."
        : "Looks for unreadable files, images too small for the resolution, near-duplicates, missing, empty or over-long captions, and captions shared by two files."}</div>}
      {error && <div className="error-box small" style={{ marginTop: 10 }}>{error}</div>}
      {result && <div style={{ marginTop: 12, opacity: loading ? 0.5 : 1 }}><CheckIssues result={result} onOpen={onOpen} /></div>}
    </div>
  );
}

function DatasetDetail({ id, families }: { id: string; families: Family[] }) {
  const toast = useToast();
  const [ds, reloadDs] = usePoll<Dataset[]>("/api/datasets", 0);
  const d = ds?.find((x) => x.id === id);
  const [items, setItems] = useState<MediaItem[] | null>(null);
  const [filter, setFilter] = useState<"all" | "missing">("all");
  // the file in the enlarged view, and the files previous and next step through: the grid's, as it was when it opened
  const [light, setLight] = useState<{ path: string; order: string[] } | null>(null);
  const [capOpen, setCapOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [trigger, setTrigger] = useState("");
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [cap, , ] = usePoll<CapStatus>(`/api/datasets/${id}/autocaption`, 1500);
  const wasRunning = useRef(false);
  const [splitOpen, setSplitOpen] = useState(false);
  const [split, reloadSplit] = usePoll<SplitStatus>(`/api/datasets/${id}/split`, 1500);
  const wasSplitting = useRef(false);
  useEffect(() => {
    if (split?.running) wasSplitting.current = true;
    else if (wasSplitting.current) { wasSplitting.current = false; reloadDs(); }
  }, [split?.running]);
  const [dlOpen, setDlOpen] = useState(false);
  const [dl, reloadDl] = usePoll<DownloadStatus>(`/api/datasets/${id}/download`, 1500);
  const wasDownloading = useRef(false);
  useEffect(() => {
    if (dl?.running) wasDownloading.current = true;
    else if (wasDownloading.current) { wasDownloading.current = false; reloadDs(); }
    if (dl?.count) loadItems(); // a playlist's files show as they arrive
  }, [dl?.running, dl?.count]);

  const loadItems = () => api.get<MediaItem[]>(`/api/datasets/${id}/items`).then(setItems);
  useEffect(() => { loadItems(); }, [id]);
  useEffect(() => { if (d) setTrigger(d.trigger); }, [d?.trigger]);
  useEffect(() => {
    if (cap?.running) wasRunning.current = true;
    else if (wasRunning.current) { wasRunning.current = false; loadItems(); reloadDs(); }
    if (cap?.running && cap.done % 5 === 0) loadItems();
  }, [cap?.running, cap?.done]);

  const upload = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (!arr.length) return;
    let saved = 0;
    for (let i = 0; i < arr.length; i += 20) {
      setUploading(`Uploading ${Math.min(i + 20, arr.length)} / ${arr.length}`);
      const fd = new FormData();
      arr.slice(i, i + 20).forEach((f) => fd.append("files", f));
      try {
        saved += (await api.post<{ saved: number }>(`/api/datasets/${id}/upload`, fd)).saved;
      } catch (e: any) {
        toast(e.message, true);
      }
    }
    setUploading(null);
    toast(`Uploaded ${saved} file${saved === 1 ? "" : "s"}`);
    loadItems();
    reloadDs();
  };

  const saveTrigger = async () => {
    if (!d || trigger === d.trigger) return;
    await api.put(`/api/datasets/${id}`, { name: d.name, trigger });
    toast("Trigger word saved");
    reloadDs();
  };

  const removeFile = async (it: MediaItem) => {
    if (!confirm(`Delete ${it.name} and its caption from disk?`)) return;
    try {
      await api.del(`/api/media?path=${q(it.path)}`);
      loadItems();
    } catch (e: any) {
      toast(e.message, true);
    }
  };

  const [confirmRemove, setConfirmRemove] = useState(false);
  const removeDataset = async (files: boolean) => {
    await api.del(`/api/datasets/${id}?delete_files=${files}`);
    go("datasets");
  };

  if (ds && !d) return <div className="page"><div className="empty"><strong>Dataset not found</strong><a href="#/datasets">Back to datasets</a></div></div>;
  const shown = (items || []).filter((i) => filter === "all" || !i.caption);
  const open = (path: string) => {
    const from = shown.some((x) => x.path === path) ? shown : items || []; // the health check can open a filtered-out one
    setLight({ path, order: from.map((x) => x.path) });
  };
  const lit = light && items?.find((x) => x.path === light.path);
  const captionSaved = (path: string, c: string) =>
    setItems((xs) => xs!.map((x) => (x.path === path ? { ...x, caption: c, has_caption: true } : x)));
  const missing = (items || []).filter((i) => !i.caption).length;
  const onlyAudio = !!items?.length && items.every((i) => i.kind === "audio");

  return (
    <div className="page"
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setOver(false); }}
      onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files); }}>
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="small"><a href="#/datasets">Datasets</a> <span className="faint">/</span></div>
          <h1>{d?.name}</h1>
          <p className="mono ellipsis">{d?.path}</p>
        </div>
        <div className="row">
          <button className="btn danger" onClick={() => setConfirmRemove(true)}><IconTrash /> Remove</button>
          <button className="btn" onClick={() => fileRef.current?.click()}><IconUpload /> Upload</button>
          <button className="btn" onClick={() => setDlOpen(true)} disabled={dl?.running}
            title="Download a video, or only its sound, from a web page into this dataset (experimental)"><IconDownload /> Add from URL</button>
          <button className="btn" onClick={() => setEditOpen(true)} disabled={!items?.length} title="Find and replace, add or remove words in every caption"><IconEdit /> Edit captions</button>
          {items?.some((i) => i.kind === "video") && (
            <button className="btn" onClick={() => setSplitOpen(true)} disabled={split?.running}
              title="Make a copy of this dataset with each video cut into clips a video model trains on whole">
              <IconScissors /> Split into clips</button>
          )}
          <button className="btn primary" onClick={() => setCapOpen(true)} disabled={!items?.length || onlyAudio}
            title={onlyAudio ? "Auto-Caption describes images and clips; write each song's style tags yourself (lyrics go after a [Lyrics] line)" : undefined}>
            <IconSparkle /> Auto-Caption</button>
          <input ref={fileRef} type="file" multiple hidden accept="image/*,video/*,audio/*,.txt" onChange={(e) => e.target.files && upload(e.target.files)} />
        </div>
      </div>

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <div className="row wrap" style={{ gap: 18 }}>
          <label className="field" style={{ width: 260 }}>
            <span>Trigger word</span>
            <input value={trigger} onChange={(e) => setTrigger(e.target.value)} onBlur={saveTrigger} onKeyDown={(e) => e.key === "Enter" && saveTrigger()} placeholder="none" />
          </label>
          <div className="stat"><div className="k">Files</div><div className="v">{items?.length ?? "…"}</div></div>
          <div className="stat"><div className="k">Captioned</div><div className="v">{items ? items.length - missing : "…"}</div></div>
          <div className="spacer" />
          <div className="seg">
            <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>All</button>
            <button className={filter === "missing" ? "on" : ""} onClick={() => setFilter("missing")}>Missing captions ({missing})</button>
          </div>
        </div>
        {cap && (cap.running || cap.errors.length > 0) && (
          <div style={{ marginTop: 14 }}>
            <div className="row small" style={{ marginBottom: 6 }}>
              <IconSparkle width={14} style={{ color: "var(--accent)" }} />
              {!cap.running ? <span>Captioning finished</span> : cap.phase === "transcribing"
                ? <span>Transcribing speech {cap.heard} / {cap.to_hear} <span className="faint">{cap.message}</span></span>
                : cap.phase === "starting" ? <span>{cap.message || "Starting the caption model"}</span>
                : <span>Captioning {cap.done} / {cap.total} <span className="faint">{cap.current}</span></span>}
              <span className="spacer" />
              {cap.running && <button className="btn sm" onClick={() => api.del(`/api/datasets/${id}/autocaption`)}><IconX /> Cancel</button>}
            </div>
            {cap.running && <div className="progress"><div style={{ width: `${cap.phase === "transcribing"
              ? (100 * (cap.heard || 0)) / Math.max(1, cap.to_hear || 0) : (100 * cap.done) / Math.max(1, cap.total)}%` }} /></div>}
            {cap.errors.length > 0 && <div className="error-box" style={{ marginTop: 8 }}>{cap.errors.slice(-3).map((e) => <div key={e}>{e}</div>)}</div>}
          </div>
        )}
        {split?.target && (split.running || split.done > 0) && (
          <div style={{ marginTop: 14 }}>
            <div className="row small" style={{ marginBottom: 6 }}>
              <IconScissors width={14} style={{ color: "var(--accent)" }} />
              {split.running
                ? <span>Splitting {split.done} / {split.total} files · {split.clips} clips <span className="faint">{split.current}{split.message ? ` · ${split.message}` : ""}</span></span>
                : split.cancelled ? <span>Split cancelled; the partial copy was removed</span>
                : <span>Split finished: {split.clips} clips in <a href={`#/datasets/${split.target.id}`}>{split.target.name}</a></span>}
              <span className="spacer" />
              {split.running && <button className="btn sm" onClick={() => api.del(`/api/datasets/${id}/split`).then(reloadSplit)}><IconX /> Cancel</button>}
            </div>
            {split.running && <div className="progress"><div style={{ width: `${(100 * split.done) / Math.max(1, split.total)}%` }} /></div>}
            {split.errors.length > 0 && <div className="error-box" style={{ marginTop: 8 }}>{split.errors.slice(-5).map((e) => <div key={e}>{e}</div>)}</div>}
          </div>
        )}
        {dl && (dl.running || dl.count > 0 || dl.errors.length > 0 || dl.cancelled) && (
          <div style={{ marginTop: 14 }}>
            <div className="row small" style={{ marginBottom: 6 }}>
              <IconDownload width={14} style={{ color: "var(--accent)" }} />
              {dl.running ? (
                <span>{dl.phase === "processing" ? dl.step : dl.phase === "downloading" ? "Downloading" : "Starting the download"}
                  {(dl.items || 0) > 1 ? ` ${dl.item} / ${dl.items}` : ""}{" "}
                  <span className="faint">{dl.title}{dl.phase === "downloading" && dl.done_bytes
                    ? ` · ${mb(dl.done_bytes)}${dl.total_bytes ? ` / ${mb(dl.total_bytes)}` : ""}` : ""}</span></span>
              ) : dl.cancelled ? <span>Download cancelled{dl.count ? `; the ${dl.count} file${dl.count === 1 ? "" : "s"} already added stay` : ""}</span>
                : <span>{dl.count ? `Added ${dl.count} file${dl.count === 1 ? "" : "s"} from the web` : "Nothing was downloaded"}
                  {dl.count === 1 && <span className="faint"> · {dl.saved[0]}</span>}</span>}
              <span className="spacer" />
              {dl.running && <button className="btn sm" onClick={() => api.del(`/api/datasets/${id}/download`).then(reloadDl)}><IconX /> Cancel</button>}
            </div>
            {dl.running && (dl.total_bytes
              ? <div className="progress"><div style={{ width: `${(100 * ((dl.item || 1) - 1 + (dl.done_bytes || 0) / dl.total_bytes)) / Math.max(1, dl.items || 1)}%` }} /></div>
              : <div className="progress busy"><div /></div>)}
            {dl.errors.length > 0 && <div className="error-box" style={{ marginTop: 8 }}>{dl.errors.slice(-5).map((e) => <div key={e}>{e}</div>)}</div>}
          </div>
        )}
      </div>

      <HealthCheck id={id} families={families} onOpen={(path) => { if (items?.some((x) => x.path === path)) open(path); }} />

      {(over || uploading) && (
        <div className={"dropzone over"} style={{ marginBottom: 16 }}>{uploading || "Drop images, videos or audio to upload"}</div>
      )}

      {items && items.length === 0 && (
        <div className="dropzone" onClick={() => fileRef.current?.click()}>
          <IconUpload width={28} height={28} /><br />
          <strong style={{ color: "var(--text)" }}>No images, videos or audio yet</strong><br />
          Drag files here or click to upload. Matching .txt caption files are picked up too (for a song: its style
          tags, then any lyrics after a [Lyrics] line).
        </div>
      )}

      <div className="media-grid">
        {shown.map((it) => (
          <div className="media-card" key={it.path}>
            <div className={"media-thumb" + (it.kind === "audio" ? " audio" : "")} style={{ backgroundImage: `url(/api/thumb?path=${q(it.path)})` }} onClick={() => open(it.path)}>
              {it.kind === "video" && <span className="badge">VIDEO</span>}
              {it.kind === "audio" && <span className="badge">AUDIO</span>}
              {!it.caption && <span className="nocap" title="No caption" />}
            </div>
            <CaptionCell item={it} onSaved={(c) => captionSaved(it.path, c)} />
            <div className="row small" style={{ padding: "0 10px 8px" }}>
              <span className="faint ellipsis" title={it.cut_from ? `${it.name}: cut from ${it.cut_from.source} at ${fmtClock(it.cut_from.start)}` : it.name}>{it.name}</span>
              <span className="spacer" />
              {d?.managed ? <button className="btn ghost icon sm" title="Delete file" onClick={() => removeFile(it)}><IconTrash /></button> : null}
            </div>
          </div>
        ))}
      </div>

      {light && lit && <Lightbox src={`/api/media?path=${q(lit.path)}`} video={lit.kind === "video"} onClose={() => setLight(null)}
        audio={lit.kind === "audio" ? { poster: `/api/thumb?size=1024&path=${q(lit.path)}` } : undefined}
        nav={{ at: light.order.indexOf(lit.path), count: light.order.length,
          go: (s) => setLight((l) => l && { ...l, path: l.order[l.order.indexOf(l.path) + s] ?? l.path }) }}
        info={{ name: lit.name, caption: lit.caption, details: [
          ...(lit.size != null ? [fmtSize(lit.size)] : []),
          ...(lit.cut_from ? [`cut from ${lit.cut_from.source} at ${fmtClock(lit.cut_from.start)}`] : []),
        ], onCaption: async (c) => {
          await api.put("/api/caption", { path: lit.path, caption: c });
          captionSaved(lit.path, c.trim());
        } }} />}
      {confirmRemove && d && (
        <ConfirmDialog title="Remove dataset" danger confirmLabel="Remove" onClose={() => setConfirmRemove(false)} onConfirm={removeDataset}
          message={d.managed ? <>Remove <strong style={{ color: "var(--text)" }}>{d.name}</strong> from Dainamo?</>
            : <>Remove <strong style={{ color: "var(--text)" }}>{d.name}</strong> from Dainamo? The folder <span className="mono">{d.path}</span> and its files are left untouched.</>}
          checkbox={d.managed ? "Also delete the uploaded files and captions" : undefined} />
      )}
      {capOpen && d && <AutoCaptionModal dataset={d} missing={missing} hasVideo={!!items?.some((i) => i.kind === "video")}
        onClose={() => { setCapOpen(false); reloadDs(); }} />}
      {editOpen && d && <EditCaptionsModal dataset={d} onClose={() => setEditOpen(false)} onChanged={() => { loadItems(); reloadDs(); }} />}
      {splitOpen && d && <SplitModal dataset={d} families={families} onClose={() => setSplitOpen(false)} onStarted={reloadSplit} />}
      {dlOpen && d && <DownloadModal dataset={d} onClose={() => setDlOpen(false)} onStarted={reloadDl} />}
    </div>
  );
}

const fmtClock = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const mb = (bytes: number) => `${(bytes / 1024 ** 2).toFixed(1)} MB`;

const HEIGHTS: [number, string][] = [[360, "360p"], [480, "480p"], [720, "720p"], [1080, "1080p"], [1440, "1440p"], [2160, "2160p (4K)"], [0, "The best there is"]];
// the formats a download can come as (media_download.VIDEO_FORMATS and AUDIO_FORMATS); the first is the default
const VIDEO_FORMATS: [string, string][] = [["mp4", "MP4 · H.264 where offered"], ["webm", "WebM · VP9"], ["mkv", "MKV · the best the site has"],
  ["mov", "MOV · H.264, for video editors"]];
const AUDIO_FORMATS: [string, string][] = [["best", "As the site has it"], ["opus", "Opus (.opus)"], ["m4a", "M4A · AAC (.m4a)"], ["mp3", "MP3 (.mp3)"],
  ["wav", "WAV · uncompressed (.wav)"], ["flac", "FLAC · lossless (.flac)"], ["vorbis", "OGG · Vorbis (.ogg)"]];
const FORMAT_INFO = {
  video: "MP4 plays everywhere and is the safe choice. WebM and MKV keep the site's newer codecs (VP9, AV1), smaller for the same quality, "
    + "though the browser may not preview an MKV. MOV is H.264 for video editors, so at most the site's best H.264 (1080p on YouTube). "
    + "A part chosen with From and To is re-encoded to cut it exactly.",
  audio: "As the site has it keeps the site's own stream (often Opus or AAC), so nothing more is lost; Opus and M4A are copied too when "
    + "the site has them. MP3 and OGG are converted and lose a little more. WAV and FLAC lose nothing more but are several times bigger, "
    + "and can't bring back what the site's compression removed.",
};

/** "1:02:03.5", "1:30" or "75" -> seconds; "" -> null; NaN when it isn't a time. */
const parseClock = (s: string): number | null => {
  const t = s.trim();
  if (!t) return null;
  if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(t)) return NaN;
  return t.split(":").reduce((a, p) => a * 60 + Number(p), 0);
};

/** Add from URL (experimental): yt-dlp downloads a video, or only its sound, into the dataset. */
function DownloadModal({ dataset, onClose, onStarted }: { dataset: Dataset; onClose: () => void; onStarted: () => void }) {
  const toast = useToast();
  const [info, setInfo] = useState<YtdlpInfo | null>(null);
  const [updating, setUpdating] = useState(false);
  const [url, setUrl] = useState("");
  const [kind, setKind] = useState<"video" | "audio">("video");
  const [height, setHeight] = useState(1080);
  const [videoFormat, setVideoFormat] = useState(VIDEO_FORMATS[0][0]);
  const [audioFormat, setAudioFormat] = useState(AUDIO_FORMATS[0][0]);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [playlist, setPlaylist] = useState(false);
  const [items, setItems] = useState(10);
  useEffect(() => { api.get<YtdlpInfo>("/api/ytdlp").then(setInfo, (e) => toast(e.message, true)); }, []);
  const start = parseClock(from), end = parseClock(to);
  const partOk = playlist || (!Number.isNaN(start) && !Number.isNaN(end) && (end == null || end > (start ?? 0)));
  const ok = /^https?:\/\/\S+$/i.test(url.trim()) && partOk && (!playlist || (items >= 1 && items <= 200)) && !!info?.current && !updating;
  const update = async () => {
    setUpdating(true);
    try {
      const i = await api.post<YtdlpInfo>("/api/ytdlp/update");
      setInfo(i);
      toast(`yt-dlp ${i.version} is installed`);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setUpdating(false);
    }
  };
  const begin = async () => {
    try {
      await api.post(`/api/datasets/${dataset.id}/download`, {
        url: url.trim(), kind, max_height: height, format: kind === "audio" ? audioFormat : videoFormat, start: playlist ? null : start, end: playlist ? null : end, playlist: playlist ? items : 0,
      });
      toast("Download started");
      onStarted();
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  return (
    <Modal title={<>Add from URL<span className="pill warn">Experimental</span></>} onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" onClick={begin} disabled={!ok}><IconDownload /> Download</button>
      </>
    }>
      <div className="stack">
        <label className="field">
          <span>Web address</span>
          <input autoFocus value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://…"
            onKeyDown={(e) => e.key === "Enter" && ok && begin()} />
        </label>
        <div className="field">
          <span>Download<Info text="Sound only keeps just the audio, for music datasets." /></span>
          <div className="seg" style={{ alignSelf: "flex-start" }}>
            <button type="button" className={kind === "video" ? "on" : ""} onClick={() => setKind("video")}>Video</button>
            <button type="button" className={kind === "audio" ? "on" : ""} onClick={() => setKind("audio")}>Sound only</button>
          </div>
        </div>
        <div className="grid cols-2">
          <label className="field">
            <span>Format<Info text={FORMAT_INFO[kind]} /></span>
            {kind === "audio"
              ? <select value={audioFormat} onChange={(e) => setAudioFormat(e.target.value)}>{AUDIO_FORMATS.map(([f, label]) => <option key={f} value={f}>{label}</option>)}</select>
              : <select value={videoFormat} onChange={(e) => setVideoFormat(e.target.value)}>{VIDEO_FORMATS.map(([f, label]) => <option key={f} value={f}>{label}</option>)}</select>}
          </label>
          {kind === "video" && (
            <label className="field">
              <span>Largest size<Info text="Training scales every file down to its resolution, so more than 1080p mostly costs disk space and download time." /></span>
              <select value={height} onChange={(e) => setHeight(Number(e.target.value))}>
                {HEIGHTS.map(([h, label]) => <option key={h} value={h}>{label}</option>)}
              </select>
            </label>
          )}
        </div>
        {!playlist && (
          <div>
            <div className="grid cols-2">
              <label className="field"><span>From</span><input value={from} onChange={(e) => setFrom(e.target.value)} placeholder="the start" /></label>
              <label className="field"><span>To</span><input value={to} onChange={(e) => setTo(e.target.value)} placeholder="the end" /></label>
            </div>
            <div className={partOk ? "help" : "validation warning"} style={{ marginTop: 6 }}>
              {partOk ? "Only that part, as m:ss or seconds (1:30 or 90). Leave both empty for all of it."
                : "Times look like 1:30 or 90, and the end comes after the start."}
            </div>
          </div>
        )}
        <div>
          <label className="check">
            <input type="checkbox" checked={playlist} onChange={(e) => setPlaylist(e.target.checked)} /> It's a playlist: download several items
          </label>
          {playlist && (
            <label className="field" style={{ marginTop: 10, maxWidth: 220 }}>
              <span>Items, from the start</span>
              <input type="number" min={1} max={200} value={Number.isFinite(items) ? items : ""} onChange={(e) => setItems(e.target.value === "" ? NaN : Number(e.target.value))} />
            </label>
          )}
        </div>
        <div className="dl-notes">
          {info && !info.installed ? (
            <p className="warn">yt-dlp, which does the downloading, isn't installed.{" "}
              <button className="btn sm" disabled={updating} onClick={update}>{updating ? "Installing…" : "Install yt-dlp"}</button></p>
          ) : info && !info.current ? (
            <p className="warn">This yt-dlp ({info.version}{info.bundled ? "" : ", the one on your PATH"}) is older than Dainamo needs
              ({info.minimum} or newer).{" "}
              <button className="btn sm" disabled={updating} onClick={update}>{updating ? "Updating…" : info.bundled ? "Update yt-dlp" : "Install it in Dainamo"}</button></p>
          ) : (
            <p>Experimental. Downloads use yt-dlp{info?.version ? ` ${info.version}` : ""}{info && !info.bundled ? " (the one on your PATH)" : ""}.
              Websites change often, and a change can break downloads until yt-dlp is updated.{" "}
              {info && <button className="btn sm" disabled={updating} onClick={update}>{updating ? "Updating…" : info.bundled ? "Update yt-dlp" : "Install it in Dainamo"}</button>}</p>
          )}
          <p>Download only what you have the right to use: copyright and each site's terms are your responsibility.</p>
          {info?.network && (
            <p className="warn">This Dainamo is reachable from other machines, so anyone who can use it can make this machine download
              from any web address, including ones on your local network.</p>
          )}
        </div>
      </div>
    </Modal>
  );
}
const SPLIT_SIZES = [256, 320, 384, 448, 512, 576, 640, 768, 1024];

/** Split into clips: a copy of the dataset with each video cut into clips of a length that fits the GPU,
 *  inside one shot each and spread over the video. Each video then trains once per epoch. */
function SplitModal({ dataset, families, onClose, onStarted }: { dataset: Dataset; families: Family[]; onClose: () => void; onStarted: () => void }) {
  const toast = useToast();
  const videoFamilies = families.filter((f) => f.media.includes("video"));
  const [profiles, setProfiles] = useState<Profile[] | null>(null);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [profileId, setProfileId] = useState("");
  const [gpu, setGpu] = useState(-1);
  const [resolution, setResolution] = useState(384);
  const [offload, setOffload] = useState(0);
  const [lowVram, setLowVram] = useState(false);
  const [fit, setFit] = useState<ClipFit | null>(null);
  const [frames, setFrames] = useState(0);
  const [name, setName] = useState(`${dataset.name} clips`);
  const [maxClips, setMaxClips] = useState(8);
  const [scenes, setScenes] = useState(true);
  const [keepShort, setKeepShort] = useState(true);
  const [maxSide, setMaxSide] = useState(1024);
  const [captions, setCaptions] = useState<"copy" | "none">("copy");
  const profile = profiles?.find((p) => p.id === profileId);
  const family = videoFamilies.find((f) => f.id === profile?.family);
  const pickProfile = (p: Profile) => {
    const fam = videoFamilies.find((f) => f.id === p.family);
    setProfileId(p.id);
    setResolution(Number(fam?.training_defaults.resolution) || 384);
    setOffload(Number(fam?.training_defaults.offload) || 0);
  };
  useEffect(() => {
    api.get<Profile[]>("/api/profiles").then((ps) => {
      const video = ps.filter((p) => videoFamilies.some((f) => f.id === p.family));
      setProfiles(video);
      if (video[0]) pickProfile(byFamily(video, videoFamilies)[0].profiles[0]);
    });
    api.get<{ gpus: Gpu[] }>("/api/system").then((sys) => {
      setGpus(sys.gpus);
      if (sys.gpus.length) setGpu(sys.gpus.reduce((a, b) => (b.mem_total < a.mem_total ? b : a)).index);
    });
  }, []);
  const gpuGb = (gpus.find((g) => g.index === gpu)?.mem_total ?? 0) / 1024;
  useEffect(() => {
    if (!profileId || !gpuGb) return;
    let live = true;
    setFit(null);
    api.post<ClipFit>(`/api/datasets/${dataset.id}/split/fit`, { profile_id: profileId, resolution, offload, low_vram: lowVram, gpu_gb: gpuGb })
      .then((f) => {
        if (!live) return;
        setFit(f);
        setFrames(f.frames ?? f.lengths?.[0]?.frames ?? 0);
      })
      .catch((e) => live && setFit({ frames: null, fps: 24, lengths: [], error: e.message }));
    return () => { live = false; };
  }, [profileId, resolution, offload, lowVram, gpuGb]);
  const chosen = fit?.lengths.find((l) => l.frames === frames);
  const start = async () => {
    try {
      await api.post(`/api/datasets/${dataset.id}/split`, {
        name, family: family!.id, frames, max_clips: maxClips, scenes, keep_short: keepShort, max_side: maxSide, captions,
      });
      toast("Splitting started");
      onStarted();
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  return (
    <Modal title="Split into clips" onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" onClick={start} disabled={!family || !(frames > 1) || !name.trim()}><IconScissors /> Split</button>
      </>
    }>
      <div className="stack">
        <p className="muted" style={{ margin: 0 }}>
          Makes a new dataset with each video cut into clips of one length, inside one shot each and spread over the
          video. This dataset stays as it is. Images and sound files are copied as they are. Each video then trains
          once per epoch, as one of its clips picked at random, so a long video counts no more than a short one.
        </p>
        <label className="field">
          <span>New dataset name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        {profiles && !profiles.length && <div className="validation warning">Add a profile for a video model first: the clip length comes from its VRAM projection.</div>}
        <div className="grid cols-2">
          <label className="field">
            <span>Profile</span>
            <select value={profileId} onChange={(e) => pickProfile(profiles!.find((p) => p.id === e.target.value)!)} disabled={!profiles?.length}>
              <ProfileOptions profiles={profiles || []} families={videoFamilies} />
            </select>
          </label>
          <label className="field">
            <span>GPU</span>
            <select value={gpu} onChange={(e) => setGpu(Number(e.target.value))} disabled={!gpus.length}>
              {gpus.map((g) => <option key={g.index} value={g.index}>GPU {g.index} · {g.name} · {(g.mem_total / 1024).toFixed(0)} GB</option>)}
            </select>
          </label>
          <label className="field">
            <span>Training resolution</span>
            <select value={resolution} onChange={(e) => setResolution(Number(e.target.value))}>
              {[...new Set([...SPLIT_SIZES, resolution])].sort((a, b) => a - b).map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Offload <span className="faint">· % of blocks in RAM</span></span>
            <input type="number" min={0} max={100} step={5} value={offload} onChange={(e) => setOffload(Math.min(100, Math.max(0, Number(e.target.value) || 0)))} />
          </label>
        </div>
        {family?.streams_text_encoder && (
          <label className="switch-row">
            <span className="switch"><input type="checkbox" role="switch" checked={lowVram} onChange={(e) => setLowVram(e.target.checked)} /><i /></span>
            <span>Low VRAM <span className="muted">· the job streams its text encoder and saved activations from system RAM too</span></span>
          </label>
        )}
        <label className="field">
          <span>Clip length</span>
          {fit?.error || (fit && !fit.lengths.length) ? (
            <>
              <input type="number" min={2} value={frames || ""} onChange={(e) => setFrames(Math.round(Number(e.target.value)))} />
              <span className="validation warning">{fit.error || "No projection"}; enter the frames yourself ({family?.frame_rule}).</span>
            </>
          ) : (
            <>
              <select value={frames} onChange={(e) => setFrames(Number(e.target.value))} disabled={!fit}>
                {!fit && <option value={0}>Projecting VRAM…</option>}
                {fit?.lengths.map((l) => (
                  <option key={l.frames} value={l.frames}>
                    {l.frames} frames · {l.seconds.toFixed(1)} s · {l.peak_gb.toFixed(1)} GB{l.frames === fit.frames ? " · longest that fits" : l.peak_gb > gpuGb - 0.75 ? " · won't fit" : ""}
                  </option>
                ))}
              </select>
              <span className="help">
                {fit && fit.frames == null ? "Even the shortest clip won't fit at this resolution and offload; lower the resolution or raise the offload. " : ""}
                {chosen ? `Projected for a job on this GPU (${gpuGb.toFixed(0)} GB) at ${resolution} px with ${offload}% offload${lowVram && family?.streams_text_encoder ? ", Low VRAM" : ""} and the model's other defaults. ` : ""}
                Train the copy with this clip length or a shorter one.
              </span>
            </>
          )}
        </label>
        <div className="grid cols-2">
          <label className="field">
            <span>Clips per video <span className="faint">· 0 for all</span></span>
            <input type="number" min={0} max={1000} value={maxClips} onChange={(e) => setMaxClips(Math.max(0, Math.round(Number(e.target.value) || 0)))} />
            <span className="help">Spread over the video when it has room for more.</span>
          </label>
          <label className="field">
            <span>Scale down to</span>
            <select value={maxSide} onChange={(e) => setMaxSide(Number(e.target.value))}>
              {[512, 768, 1024, 1280].map((v) => <option key={v} value={v}>{v} px on the shorter side</option>)}
              <option value={0}>Keep the size</option>
            </select>
            <span className="help">Never scaled up.</span>
          </label>
        </div>
        <label className="check"><input type="checkbox" checked={scenes} onChange={(e) => setScenes(e.target.checked)} /> Keep each clip inside one shot</label>
        <span className="help" style={{ marginTop: -8 }}>Finds the cuts in each video first: a clip that spans a cut teaches the model to cut.</span>
        <label className="check"><input type="checkbox" checked={keepShort} onChange={(e) => setKeepShort(e.target.checked)} /> Keep shots shorter than a clip, as shorter clips (from 1 s)</label>
        <div className="field">
          <span>Captions</span>
          <label className="check"><input type="radio" checked={captions === "copy"} onChange={() => setCaptions("copy")} /> Give each clip its video's caption</label>
          <label className="check"><input type="radio" checked={captions === "none"} onChange={() => setCaptions("none")} /> Leave the clips uncaptioned, to Auto-Caption each one</label>
          <span className="help">A video's caption may not fit every part of it; Auto-Caption describes each clip on its own.</span>
        </div>
      </div>
    </Modal>
  );
}

type BulkOp = "replace" | "add" | "remove";
type BulkResult = { changed: number; total: number; applied: boolean; examples: { name: string; before: string | null; after: string }[] };
type UndoInfo = { available: boolean; at?: number; what?: string; count?: number };

/** One edit to every caption: find and replace, add words, or remove words, previewed before it's applied. */
function EditCaptionsModal({ dataset, onClose, onChanged }: { dataset: Dataset; onClose: () => void; onChanged: () => void }) {
  const toast = useToast();
  const [op, setOp] = useState<BulkOp>("replace");
  const [f, setF] = useState({ find: "", replace: "", text: dataset.trigger, where: "start", regex: false, match_case: false, uncaptioned: false });
  const [preview, setPreview] = useState<BulkResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0); // preview again after the captions change
  const [undo, reloadUndo] = usePoll<UndoInfo>(`/api/datasets/${dataset.id}/captions/undo`, 0);
  const body = { op, ...f };
  const ready = op === "replace" ? f.find !== "" : f.text.trim() !== "";
  useEffect(() => {
    if (!ready) { setPreview(null); setError(null); return; }
    const t = setTimeout(() => {
      api.post<BulkResult>(`/api/datasets/${dataset.id}/captions/bulk`, body)
        .then((r) => { setPreview(r); setError(null); }, (e) => { setPreview(null); setError(e.message); });
    }, 300);
    return () => clearTimeout(t);
  }, [JSON.stringify(body), refresh]);
  const set = (patch: Partial<typeof f>) => setF((x) => ({ ...x, ...patch }));

  const apply = async () => {
    setBusy(true);
    try {
      const r = await api.post<BulkResult>(`/api/datasets/${dataset.id}/captions/bulk`, { ...body, apply: true });
      toast(`Changed ${r.changed} caption${r.changed === 1 ? "" : "s"}`);
      onChanged();
      reloadUndo();
      setRefresh((n) => n + 1);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  const revert = async () => {
    try {
      const r = await api.post<{ restored: number }>(`/api/datasets/${dataset.id}/captions/undo`);
      toast(`Put back ${r.restored} caption${r.restored === 1 ? "" : "s"}`);
      onChanged();
      reloadUndo();
      setRefresh((n) => n + 1);
    } catch (e: any) {
      toast(e.message, true);
    }
  };

  return (
    <Modal title="Edit captions" wide onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Close</button>
        <button className="btn primary" disabled={busy || !preview?.changed} onClick={apply}>
          {preview?.changed ? `Change ${preview.changed} caption${preview.changed === 1 ? "" : "s"}` : "Nothing to change"}
        </button>
      </>
    }>
      <div className="stack" style={{ gap: 14 }}>
        {undo?.available && (
          <div className="row small caption-undo">
            <span className="muted">Last edit: {undo.what}, {undo.count} caption{undo.count === 1 ? "" : "s"} · {fmtTime(undo.at!)}</span>
            <span className="spacer" />
            <button className="btn sm" onClick={revert}>Undo it</button>
          </div>
        )}
        <div className="seg" role="group" aria-label="Edit">
          <button className={op === "replace" ? "on" : ""} onClick={() => setOp("replace")}>Find and replace</button>
          <button className={op === "add" ? "on" : ""} onClick={() => setOp("add")}>Add words</button>
          <button className={op === "remove" ? "on" : ""} onClick={() => setOp("remove")}>Remove words</button>
        </div>
        {op === "replace" ? (
          <>
            <div className="grid cols-2">
              <label className="field"><span>Find</span><input value={f.find} onChange={(e) => set({ find: e.target.value })} autoFocus /></label>
              <label className="field"><span>Replace with</span><input value={f.replace} onChange={(e) => set({ replace: e.target.value })} placeholder="nothing" /></label>
            </div>
            <div className="row" style={{ gap: 18 }}>
              <label className="check"><input type="checkbox" checked={f.match_case} onChange={(e) => set({ match_case: e.target.checked })} /> Match case</label>
              <label className="check"><input type="checkbox" checked={f.regex} onChange={(e) => set({ regex: e.target.checked })} /> Regular expression</label>
            </div>
          </>
        ) : (
          <>
            <div className="grid cols-2">
              <label className="field"><span>{op === "add" ? "Words to add" : "Words to remove"}</span>
                <input value={f.text} onChange={(e) => set({ text: e.target.value })} placeholder={dataset.trigger || "e.g. ohwx"} autoFocus /></label>
              {op === "add" && (
                <label className="field"><span>Where</span>
                  <select value={f.where} onChange={(e) => set({ where: e.target.value })}>
                    <option value="start">At the start</option>
                    <option value="end">At the end</option>
                  </select>
                </label>
              )}
            </div>
            <div className="row" style={{ gap: 18 }}>
              <label className="check"><input type="checkbox" checked={f.match_case} onChange={(e) => set({ match_case: e.target.checked })} /> Match case</label>
              {op === "add" && <label className="check"><input type="checkbox" checked={f.uncaptioned} onChange={(e) => set({ uncaptioned: e.target.checked })} /> Also caption files that have none</label>}
            </div>
            <div className="help">
              {op === "add" ? "Joined with a comma. Captions that already contain the words are left alone."
                : "Whole words only, so “ohwx” doesn't touch “ohwx2”; the commas and spaces left behind are tidied."}
              {dataset.trigger && op === "add" ? ` The trigger word “${dataset.trigger}” is also added when training to captions without it, so you don't need to add it here.` : ""}
            </div>
          </>
        )}
        {error && <div className="error-box small">{error}</div>}
        {preview && (
          <div className="stack" style={{ gap: 8 }}>
            <div className="muted small">{preview.changed ? `Changes ${preview.changed} of ${preview.total} captions${preview.examples.length < preview.changed ? `; the first ${preview.examples.length}:` : ":"}` : `No caption would change (${preview.total} files).`}</div>
            {preview.examples.map((x) => (
              <div key={x.name} className="caption-diff">
                <div className="faint small ellipsis">{x.name}</div>
                <div className="caption-before">{x.before ?? <em>no caption</em>}</div>
                <div className="caption-after">{x.after || <em>empty</em>}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Modal>
  );
}

const FPS = 24; // clips are trained at 24 fps

function AutoCaptionModal({ dataset, missing, hasVideo, onClose }: { dataset: Dataset; missing: number; hasVideo: boolean; onClose: () => void }) {
  const toast = useToast();
  const last = dataset.caption_options; // what this dataset used last time
  const [overwrite, setOverwrite] = useState(false);
  const [trigger, setTrigger] = useState(dataset.trigger);
  const [test, setTest] = useState<{ ok: boolean; models?: string[]; error?: string; message?: string; managed?: boolean } | null>(null);
  const [styles, setStyles] = useState<Settings["caption_styles"] | null>(null);
  const [style, setStyle] = useState("");
  const [whole, setWhole] = useState(last ? last.clip_frames === 0 : false);
  const [clipFrames, setClipFrames] = useState(last?.clip_frames || 39);
  const [speechOn, setSpeechOn] = useState(!!last?.speech);
  const [speech, setSpeech] = useState(last?.speech || "");
  const [whisper, setWhisper] = useState<WhisperModel[]>([]);
  useEffect(() => {
    api.post<typeof test>("/api/captioner/test", {}).then(setTest);
    api.get<Settings>("/api/settings").then((st) => {
      setStyles(st.caption_styles);
      const known = st.caption_styles.styles.some((x) => x.id === last?.style);
      setStyle(known ? last!.style : st.caption_styles.default);
      setSpeech((m) => m || st.speech.model);
    });
    if (hasVideo) api.get<{ whisper_models: WhisperModel[] }>("/api/system").then((sys) => setWhisper(sys.whisper_models));
  }, []);
  const chosen = styles?.styles.find((x) => x.id === style);
  const start = async () => {
    try {
      await api.post(`/api/datasets/${dataset.id}/autocaption`, {
        overwrite, trigger, style: style || null,
        ...(hasVideo ? { clip_frames: whole ? 0 : clipFrames, speech: speechOn ? speech : "" } : {}),
      });
      toast("Auto-captioning started");
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  return (
    <Modal title="Auto-Caption" onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" onClick={start} disabled={(!overwrite && missing === 0) || (hasVideo && !whole && !(clipFrames >= 1))}><IconSparkle /> Start captioning</button>
      </>
    }>
      <div className="stack">
        <div className={"validation " + (test == null ? "warning" : test.ok ? "ok" : "error")} style={{ fontSize: 13 }}>
          {test == null ? "Checking captioning server…" : test.ok
            ? test.message || `Connected to captioning server${test.models?.length ? ` · ${test.models.join(", ")}` : ""}`
            : test.managed ? test.error : `Can't reach captioning server: ${test.error}. Set the endpoint in Settings.`}
        </div>
        <p className="muted" style={{ margin: 0 }}>
          Each image, or frames from each clip, goes to the caption provider chosen in <a href="#/settings">Settings</a>.
        </p>
        <label className="field">
          <span>Caption style</span>
          <select value={style} onChange={(e) => setStyle(e.target.value)} disabled={!styles}>
            {styles?.styles.map((x) => <option key={x.id} value={x.id}>{x.name}{x.id === styles.default ? " (default)" : ""}</option>)}
          </select>
          {chosen && <span className="caption-prompt">{chosen.prompt}</span>}
          <span className="help">
            {last && chosen?.id === last.style ? "The style this dataset was captioned with last time. " : ""}
            Add or edit styles in <a href="#/settings">Settings</a>.
          </span>
        </label>
        {hasVideo && (
          <div className="field">
            <span>Clips</span>
            <div className="caption-clips">
              <label className="check">
                <input type="radio" checked={!whole} onChange={() => setWhole(false)} /> Describe the first
                <input type="number" min={1} max={241} value={clipFrames} className="caption-frames" disabled={whole}
                  onChange={(e) => setClipFrames(Math.round(Number(e.target.value)))} aria-label="Frames to describe" />
                frames <span className="faint">({(clipFrames / FPS).toFixed(1)} s)</span>
              </label>
              <label className="check"><input type="radio" checked={whole} onChange={() => setWhole(true)} /> Describe whole clips</label>
              <span className="help">Training uses the start of each clip, up to the job's clip length (39 frames by default), so describe that part and the caption matches what's trained.</span>
              <label className="check">
                <input type="checkbox" checked={speechOn} onChange={(e) => setSpeechOn(e.target.checked)} /> Transcribe speech with Whisper
              </label>
              {speechOn && (
                <select value={speech} onChange={(e) => setSpeech(e.target.value)} aria-label="Whisper model">
                  {whisper.map((m) => <option key={m.id} value={m.id}>{m.id} · {m.size}{m.id === "large-v3-turbo" ? " · recommended" : ""}</option>)}
                </select>
              )}
              <span className="help">
                {speechOn
                  ? "Clips with sound are transcribed first, over the same span, and what's said goes into the caption. It runs on a free GPU, else the CPU, and the model downloads the first time. Whisper writes down speech only: not music or sound effects."
                  : "The captioner only sees frames, so it can't tell what's said in a clip."}
              </span>
            </div>
          </div>
        )}
        <label className="check"><input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
          Overwrite existing captions <span className="faint">({overwrite ? dataset.count : missing} file{(overwrite ? dataset.count : missing) === 1 ? "" : "s"} will be captioned)</span>
        </label>
        <label className="field">
          <span>Trigger word <span className="faint">· optional</span></span>
          <input value={trigger} onChange={(e) => setTrigger(e.target.value)} placeholder="none" />
          <span className="help">Prepended as "trigger, caption…" when not already present.</span>
        </label>
      </div>
    </Modal>
  );
}

export default function Datasets({ datasetId, families }: { datasetId: string; families: Family[] }) {
  const [datasets, reload] = usePoll<Dataset[]>("/api/datasets", 0, [datasetId]);
  const [creating, setCreating] = useState(false);
  if (datasetId) return <DatasetDetail key={datasetId} id={datasetId} families={families} />; // fresh state per dataset
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Datasets</h1>
          <p>Upload images, videos or music, or link an existing folder. Captions live next to each file as .txt.</p>
        </div>
        <button className="btn primary" onClick={() => setCreating(true)}><IconPlus /> New dataset</button>
      </div>
      {datasets && datasets.length === 0 && (
        <div className="empty">
          <strong>No datasets yet</strong>
          Link a folder of images or upload some to get started.
          <div style={{ marginTop: 14 }}><button className="btn primary" onClick={() => setCreating(true)}><IconPlus /> New dataset</button></div>
        </div>
      )}
      <div className="grid cols-3">
        {datasets?.map((d) => (
          <div className="card ds-card" key={d.id} onClick={() => go("datasets", d.id)}>
            <div className="ds-cover" style={d.cover ? { backgroundImage: `url(/api/thumb?size=160&path=${q(d.cover)})` } : {}}>
              {!d.cover && <IconImages width={22} />}
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 600 }} className="ellipsis">{d.name}</div>
              <div className="muted small">{d.count} files · {d.captioned} captioned</div>
              <div className="faint small ellipsis mono" title={d.path}>{d.exists ? (d.managed ? "uploaded" : d.path) : "⚠ folder missing"}</div>
              {d.trigger && <span className="pill" style={{ marginTop: 5 }}>{d.trigger}</span>}
            </div>
          </div>
        ))}
      </div>
      {creating && <NewDataset onClose={() => setCreating(false)} onCreated={(d) => { setCreating(false); reload(); go("datasets", d.id); }} />}
    </div>
  );
}

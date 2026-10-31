import { useEffect, useRef, useState } from "react";
import { go } from "../App";
import {
  api, byFamily, fmtClock, fmtTime, q, type Dataset, type DatasetUpscale, type Family, type Gpu, type Job, type MediaItem, type Profile, type Settings,
  type SystemRam, type WhisperModel,
} from "../api";
import { CheckIssues, PairSummary, useDatasetCheck } from "../components/DatasetCheck";
import { PairsDialog } from "../components/Pairs";
import { DuplicatesDialog } from "../components/Likeness";
import { DatasetUpscaleModal, DatasetUpscaleTask } from "../components/Upscale";
import {
  IconAlert, IconArrowRight, IconCheck, IconCheckCircle, IconCopy, IconDown, IconDownload, IconEdit, IconFile, IconFilm, IconFolder, IconGrid,
  IconImages, IconInfo, IconLayers, IconLeft, IconLink, IconList, IconPause, IconPlayFill, IconPlus, IconRight, IconScissors, IconSearch, IconSparkle, IconTarget,
  IconTools, IconTrash, IconUpload, IconUpscale, IconWave,
} from "../components/icons";
import {
  ConfirmDialog, Empty, Info, Lightbox, LoadError, Menu, Modal, PathInput, ProfileOptions, SelectBar, TaskProgress, fmtSize, someNames, usePoll, useSelection,
  useToast, useToastErrors, type Selection,
} from "../components/ui";
import { OffloadControl, gpuTarget, useVramProjection } from "../components/Vram";
import ClipEditor from "./ClipEditor";

type SplitStatus = {
  running: boolean; mode?: "cut" | "plan"; fresh?: boolean; total: number; done: number; clips: number; removed?: number; current?: string; message?: string; errors: string[];
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

/** A file's caption: a click edits it; Ctrl+Enter or clicking away saves, Escape
 * cancels. ``grid``: the grid view's two lines, and a boxed editor. While it's being edited, the page's polling
 * doesn't replace what's typed (a caption written meanwhile, say by Auto-Caption, is noted; saving keeps the typed
 * one). */
function CaptionCell({ item, onSaved, grid }: {
  item: MediaItem; onSaved: (c: string) => void; grid?: boolean;
}) {
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(item.caption);
  // the caption as it was when editing began: a different one arriving meanwhile was written by something else
  const [began, setBegan] = useState(item.caption);
  useEffect(() => { if (!editing) setVal(item.caption); }, [item.caption, editing]);
  const edit = () => { setVal(item.caption); setBegan(item.caption); setEditing(true); };
  const changed = editing && item.caption !== began;
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
      <>
        <textarea autoFocus value={val} onChange={(e) => setVal(e.target.value)} onBlur={save} className={grid ? "file-cap-edit" : undefined}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) save(); if (e.key === "Escape") { e.stopPropagation(); setVal(item.caption); setEditing(false); } }}
          style={grid ? undefined : { border: 0, borderRadius: 0, minHeight: 110, fontSize: 12 }} />
        {changed && <span className="help warn">This caption was rewritten while you typed. Saving keeps yours; Escape keeps the new one.</span>}
      </>
    );
  return (
    <div className={(grid ? "file-cap" : "media-cap") + (item.caption ? "" : " none")} onClick={edit} title="Click to edit">
      {item.caption || "No caption: click to write one"}
    </div>
  );
}

const CHECK_FAMILY = "dainamo.check.family";
const CHECK_OPEN = "dainamo.check.open"; // the health check left open

/** The health check, for a chosen model and resolution; it runs when asked, then again as they change. Folded, its
 * bar says how the last check went. For a model that trains with reference images (an edit's source images), it
 * also checks how the files pair with a chosen source dataset's images, or else the _refs folder's. */
function HealthCheck({ id, name, datasets, families, onOpen }: {
  id: string; name: string; datasets: Dataset[] | null; families: Family[]; onOpen: (path: string) => void;
}) {
  const [familyId, setFamilyId] = useState(() => { try { return localStorage.getItem(CHECK_FAMILY) || ""; } catch { return ""; } });
  const family = families.find((f) => f.id === familyId) || families[0];
  const [resolution, setResolution] = useState("");
  const [seconds, setSeconds] = useState("");
  const [on, setOn] = useState(false);
  const [open, setOpenState] = useState(() => { try { return localStorage.getItem(CHECK_OPEN) === "1"; } catch { return false; } });
  const setOpen = (o: boolean) => { setOpenState(o); try { localStorage.setItem(CHECK_OPEN, o ? "1" : "0"); } catch { /* not remembered */ } };
  const music = !!family?.media.includes("audio");
  const pairing = family?.sources || null;
  const [source, setSource] = useState("");
  const [showPairs, setShowPairs] = useState(false);
  const { result, error, loading, recheck } = useDatasetCheck(on ? id : null, {
    resolution: music ? "" : resolution, family: family?.id, seconds: music && seconds.trim() !== "" ? Number(seconds) : undefined,
    references: !!pairing, sources: pairing && source ? [source] : [] });
  const pick = (v: string) => { setFamilyId(v); try { localStorage.setItem(CHECK_FAMILY, v); } catch { /* not remembered */ } };
  const counts = result ? (["error", "warning", "info"] as const).map((l) => [l, result.issues.filter((i) => i.level === l).length] as const) : [];
  const words = { error: ["error", "errors"], warning: ["warning", "warnings"], info: ["note", "notes"] };
  const n = Object.fromEntries(counts) as Record<string, number>;
  const state = loading ? "busy" : error ? "error" : !result ? "" : n.error ? "error" : n.warning ? "warning" : "ok";
  const summary = loading ? "Checking…" : error ? "The check failed"
    : result ? counts.filter(([, k]) => k).map(([l, k]) => `${k} ${words[l][k === 1 ? 0 : 1]}`).join(" · ") || "All good" : "";
  return (
    <div className={"card health" + (open ? " open" : "")}>
      <div className="health-bar">
        <button className="health-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          <IconDown className="health-chev" />
          <span className={"health-state " + state}>{state === "ok" ? <IconCheckCircle /> : state === "error" || state === "warning" ? <IconAlert /> : <IconInfo />}</span>
          <span className="health-title">Dataset health check</span>
          {summary && <span className={"health-summary " + state}>{summary}</span>}
        </button>
        {open
          ? <button className="btn" disabled={loading} onClick={() => (on ? recheck() : setOn(true))}>{loading ? "Checking…" : on ? "Check again" : "Check dataset"}</button>
          : <button className="btn" onClick={() => setOpen(true)}>Configure check</button>}
      </div>
      {open && (
        <div className="health-body">
          <div className="row wrap health-head">
            <span className="muted small">Check for</span>
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
            {pairing && <>
              <span className="muted small">with {pairing.label.toLowerCase()}</span>
              <select value={source} onChange={(e) => setSource(e.target.value)} aria-label={pairing.label} style={{ maxWidth: 220 }}>
                <option value="">None (the _refs folder)</option>
                {datasets?.filter((x) => x.id !== id).map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
              </select>
            </>}
          </div>
          {!on && <div className="help" style={{ marginTop: 8 }}>{music
            ? "Looks for unreadable files, songs too short to use, low sample rates, mono files, missing or empty captions, and captions shared by two files."
            : "Looks for unreadable files, images too small for the resolution, near-duplicates, missing, empty or over-long captions, and captions shared by two files"
              + (pairing ? `, and which files pair with a ${pairing.noun} (the ${pairing.label.toLowerCase()} chosen, or the _refs folder).` : ".")}</div>}
          {error && <div className="error-box small" style={{ marginTop: 10 }}>{error}</div>}
          {result && (
            <div className="stack" style={{ gap: 10, marginTop: 12, opacity: loading ? 0.5 : 1 }}>
              {pairing && result.pairs && <PairSummary pairs={result.pairs} info={pairing} onShow={() => setShowPairs(true)} />}
              <CheckIssues result={result} onOpen={onOpen} />
            </div>
          )}
          {showPairs && pairing && <PairsDialog id={id} name={name} sources={source ? [source] : []} video={!!family?.media.includes("video")}
            info={pairing} onClose={() => setShowPairs(false)} />}
        </div>
      )}
    </div>
  );
}

const PAGE = 24; // files on a page of the grid view
const LAYOUT = "dainamo.dataset.layout";
type Layout = "grid" | "list";

/** How far the song playing in a card is: a line along the bottom of its spectrogram. */
function PlayBar({ audio }: { audio: HTMLAudioElement }) {
  const [at, setAt] = useState(0);
  useEffect(() => {
    const tick = () => setAt(audio.duration ? audio.currentTime / audio.duration : 0);
    audio.addEventListener("timeupdate", tick);
    return () => audio.removeEventListener("timeupdate", tick);
  }, [audio]);
  return <div className="file-progress" style={{ width: `${100 * at}%` }} />;
}

/** A file in the grid view: its picture (a song's spectrogram, which plays here), name, caption and whether it has
 * one (a click on the caption edits it); ⋯ enlarges it, copies its path or deletes it. ``song``: it's the song playing (true) or paused (false).
 * ``sel``: it can be picked (to delete), its box shown on hover. */
// a file's thumbnail, which the browser keeps a day: by its size too, so a file replaced by an upscaled one shows anew
const thumbUrl = (it: MediaItem) => `/api/thumb?path=${q(it.path)}&v=${it.size ?? 0}`;

function FileCard({ it, audio, song, onPlay, onOpen, onSaved, onDelete, sel }: {
  it: MediaItem; audio: HTMLAudioElement | null; song: boolean | null; onPlay: () => void; onOpen: () => void; onSaved: (c: string) => void;
  onDelete?: () => void; sel?: Selection;
}) {
  const toast = useToast();
  const picked = !!sel?.picked.has(it.path);
  const copy = () => navigator.clipboard.writeText(it.path).then(() => toast("Path copied"), () => toast("Couldn't copy the path", true));
  return (
    <div className={"file-card" + (sel?.on ? " selecting" : "") + (picked ? " picked" : "")}>
      <div className={"file-thumb " + it.kind} style={{ backgroundImage: `url(${thumbUrl(it)})` }}
        onClick={() => (sel?.on ? sel.toggle(it.path) : onOpen())}>
        {it.kind !== "image" && !sel?.on && (
          <button className="file-play" aria-label={song ? `Pause ${it.name}` : `Play ${it.name}`}
            onClick={(e) => { e.stopPropagation(); if (it.kind === "audio") onPlay(); else onOpen(); }}>{song ? <IconPause /> : <IconPlayFill />}</button>
        )}
        {it.kind === "video" && <span className="badge">VIDEO</span>}
        {sel && <input type="checkbox" className="file-pick" checked={picked} aria-label={`Select ${it.name}`}
          onClick={(e) => e.stopPropagation()} onChange={() => { if (!sel.on) sel.start(); sel.toggle(it.path); }} />}
        {song != null && audio && <PlayBar audio={audio} />}
      </div>
      <div className="file-body">
        <div className="file-name ellipsis" title={it.cut_from ? `${it.name}: cut from ${it.cut_from.source} at ${fmtClock(it.cut_from.start)}` : it.name}>{it.name}</div>
        <CaptionCell item={it} onSaved={onSaved} grid />
        <div className="file-foot">
          {it.caption ? <span className="file-state ok"><IconCheckCircle /> Captioned</span> : <span className="file-state none"><IconAlert /> No caption</span>}
          <span className="spacer" />
          <Menu label={`More for ${it.name}`} items={[
            { label: "Enlarge", icon: <IconImages />, onClick: onOpen },
            { label: "Copy path", icon: <IconCopy />, onClick: copy },
            ...(onDelete ? [{ label: "Delete", icon: <IconTrash />, onClick: onDelete, danger: true }] : []),
          ]} />
        </div>
      </div>
    </div>
  );
}

/** Pages 1 … 4 5 6 … 20: the first and last, and those next to this one. */
function Pager({ page, pages, onPage }: { page: number; pages: number; onPage: (p: number) => void }) {
  const lo = Math.max(2, Math.min(page - 1, pages - 3));
  const hi = Math.min(pages - 1, Math.max(page + 1, 4));
  const nums: (number | string)[] = [1, ...(lo > 2 ? ["gap-lo"] : [])];
  for (let p = lo; p <= hi; p++) nums.push(p);
  nums.push(...(hi < pages - 1 ? ["gap-hi"] : []), ...(pages > 1 ? [pages] : []));
  return (
    <nav className="pager" aria-label="Pages">
      <button onClick={() => onPage(page - 1)} disabled={page <= 1} aria-label="Previous page"><IconLeft /></button>
      {nums.map((n) => typeof n === "number"
        ? <button key={n} className={n === page ? "on" : ""} aria-current={n === page ? "page" : undefined} onClick={() => onPage(n)}>{n}</button>
        : <span key={n} className="pager-gap">…</span>)}
      <button onClick={() => onPage(page + 1)} disabled={page >= pages} aria-label="Next page"><IconRight /></button>
    </nav>
  );
}

/** What a dataset holds, for the badge by its name. */
function datasetKind(d: Dataset): { label: string; kind: string; icon: React.ReactNode } | null {
  const kinds = (["images", "videos", "audio"] as const).filter((k) => d[k] > 0);
  if (kinds.length !== 1) return kinds.length ? { label: "Mixed dataset", kind: "mixed", icon: <IconLayers /> } : null;
  return {
    images: { label: "Image dataset", kind: "images", icon: <IconImages /> },
    videos: { label: "Video dataset", kind: "videos", icon: <IconFilm /> },
    audio: { label: "Audio dataset", kind: "audio", icon: <IconWave /> },
  }[kinds[0]];
}

function DatasetDetail({ id, families }: { id: string; families: Family[] }) {
  const toast = useToast();
  const [ds, reloadDs] = usePoll<Dataset[]>("/api/datasets", 0);
  const d = ds?.find((x) => x.id === id);
  const [items, setItems] = useState<MediaItem[] | null>(null);
  const [filter, setFilter] = useState<"all" | "missing">("all");
  const [search, setSearch] = useState("");
  const [layout, setLayoutState] = useState<Layout>(() => { try { return localStorage.getItem(LAYOUT) === "list" ? "list" : "grid"; } catch { return "grid"; } });
  const [page, setPage] = useState(1);
  const toolbar = useRef<HTMLDivElement>(null);
  // the song playing (or paused) in its grid card
  const player = useRef<HTMLAudioElement>(null);
  const [song, setSong] = useState<{ path: string; on: boolean } | null>(null);
  // Select files (or ⋯ on one): deleted together
  const sel = useSelection();
  const [deleting, setDeleting] = useState<string[] | null>(null);
  const [exporting, setExporting] = useState(false);
  // the file in the enlarged view, and the files previous and next step through: the grid's, as it was when it opened
  const [light, setLight] = useState<{ path: string; order: string[] } | null>(null);
  const [capOpen, setCapOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [trigger, setTrigger] = useState("");
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  // caption files uploaded for files already here replace the captions those have (off: theirs are kept)
  const [replaceCaptions, setReplaceCaptions] = useState(false);
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
  // Upscale small images: the dataset's last upscale, its dialog, and a finished one dismissed
  const [upOpen, setUpOpen] = useState(false);
  const [dupOpen, setDupOpen] = useState(false); // Find duplicates
  const [up, reloadUp] = usePoll<DatasetUpscale>(`/api/datasets/${id}/upscale`, 1500);
  const [upHidden, setUpHidden] = useState<string | null>(null);
  const wasUpscaling = useRef(false);
  const upActive = up?.state === "queued" || up?.state === "running";
  useEffect(() => {
    if (upActive) wasUpscaling.current = true;
    else if (wasUpscaling.current) { wasUpscaling.current = false; loadItems(); reloadDs(); }
    if (up?.state === "running" && up.done) loadItems(); // each image shows as it's replaced
  }, [up?.state, up?.done]);
  const [dlOpen, setDlOpen] = useState(false);
  const [dl, reloadDl] = usePoll<DownloadStatus>(`/api/datasets/${id}/download`, 1500);
  const wasDownloading = useRef(false);
  useEffect(() => {
    if (dl?.running) wasDownloading.current = true;
    else if (wasDownloading.current) { wasDownloading.current = false; reloadDs(); }
    if (dl?.count) loadItems(); // a playlist's files show as they arrive
  }, [dl?.running, dl?.count]);

  const [itemsError, setItemsError] = useState<string | null>(null);
  const loadItems = () => api.get<MediaItem[]>(`/api/datasets/${id}/items`).then((x) => { setItems(x); setItemsError(null); }, (e) => setItemsError(e.message));
  useEffect(() => { loadItems(); }, [id]);
  useEffect(() => { if (d) setTrigger(d.trigger); }, [d?.trigger]);
  useEffect(() => {
    if (cap?.running) wasRunning.current = true;
    else if (wasRunning.current) { wasRunning.current = false; loadItems(); reloadDs(); }
    if (cap?.running && cap.done % 5 === 0) loadItems();
  }, [cap?.running, cap?.done]);
  useEffect(() => setPage(1), [search, filter]);

  const upload = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (!arr.length) return;
    // a file and its caption (a.png, a.txt) go in the same request, so a name taken in the dataset renames both
    const stem = (f: File) => f.name.replace(/\.[^.]*$/, "").toLowerCase();
    const groups = new Map<string, File[]>();
    for (const f of arr) groups.set(stem(f), [...(groups.get(stem(f)) || []), f]);
    const batches: File[][] = [];
    for (const g of groups.values()) {
      const last = batches[batches.length - 1];
      if (last && last.length + g.length <= 20) last.push(...g);
      else batches.push([...g]);
    }
    let saved = 0, sent = 0, failed = 0, firstError = "";
    const renamed: string[][] = [];
    const kept: string[] = [];
    for (const batch of batches) {
      sent += batch.length;
      setUploading(`Uploading ${sent} / ${arr.length}`);
      const fd = new FormData();
      batch.forEach((f) => fd.append("files", f));
      fd.append("overwrite_captions", replaceCaptions ? "true" : "false");
      try {
        const r = await api.post<{ saved: number; renamed?: string[][]; captions_kept?: string[] }>(`/api/datasets/${id}/upload`, fd);
        saved += r.saved;
        renamed.push(...(r.renamed || []));
        kept.push(...(r.captions_kept || []));
      } catch (e: any) {
        failed += batch.length;
        firstError ||= e.message;
      }
    }
    setUploading(null);
    const parts = [saved || !kept.length ? `Uploaded ${saved} file${saved === 1 ? "" : "s"}` : "Nothing was uploaded"];
    if (renamed.length) parts.push(renamed.length === 1 ? `${renamed[0][0]} was taken, so it's ${renamed[0][1]}`
      : `${renamed.length} names were taken, so those files have a number added`);
    if (kept.length) parts.push(`${kept.length === 1 ? `${kept[0]} already has a caption, so it was kept`
      : `${kept.length} files already have captions, so theirs were kept`} (tick "Uploaded captions replace existing ones" to replace them)`);
    if (failed) parts.push(`${failed} failed: ${firstError}`);
    toast(parts.join("; "), failed > 0);
    loadItems();
    reloadDs();
  };

  const saveTrigger = async () => {
    if (!d || trigger === d.trigger) return;
    try {
      await api.put(`/api/datasets/${id}`, { name: d.name, trigger });
      toast("Trigger word saved");
      reloadDs();
    } catch (e: any) {
      toast(e.message, true);
    }
  };

  const removeFiles = async () => {
    const paths = deleting || [];
    let removed = 0;
    const failed: string[] = [];
    for (const path of paths) {
      try {
        await api.del(`/api/media?path=${q(path)}`);
        removed++;
      } catch (e: any) {
        failed.push(e.message);
      }
    }
    if (paths.length > 1 || failed.length) toast(`Deleted ${removed} file${removed === 1 ? "" : "s"}${failed.length ? `; ${failed[0]}` : ""}`, failed.length > 0);
    if (song && paths.includes(song.path)) player.current?.pause();
    sel.stop();
    loadItems();
    reloadDs();
  };

  const [confirmRemove, setConfirmRemove] = useState(false);
  const [moving, setMoving] = useState(false);

  const setLayout = (l: Layout) => {
    setLayoutState(l);
    player.current?.pause();
    try { localStorage.setItem(LAYOUT, l); } catch { /* not remembered */ }
  };
  const play = (path: string) => {
    const a = player.current;
    if (!a) return;
    if (song?.path === path && !a.paused) return a.pause();
    document.querySelectorAll("audio").forEach((x) => x !== a && x.pause());
    if (song?.path !== path) a.src = `/api/media?path=${q(path)}`;
    setSong({ path, on: true });
    a.play().catch(() => setSong(null));
  };
  const goPage = (p: number) => {
    setPage(p);
    toolbar.current?.scrollIntoView({ block: "nearest" });
  };

  if (ds && !d) return <div className="page"><div className="empty"><strong>Dataset not found</strong><a href="#/datasets">Back to datasets</a></div></div>;
  const words = search.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = (items || []).filter((i) => (filter === "all" || !i.caption) && words.every((w) => `${i.name} ${i.caption}`.toLowerCase().includes(w)));
  const pages = Math.max(1, Math.ceil(shown.length / PAGE));
  const at = Math.min(page, pages);
  const paged = layout === "grid" ? shown.slice((at - 1) * PAGE, at * PAGE) : shown;
  const open = (path: string) => {
    player.current?.pause();
    const from = shown.some((x) => x.path === path) ? shown : items || []; // the health check can open a filtered-out one
    setLight({ path, order: from.map((x) => x.path) });
  };
  const closeLight = () => {
    // the grid shows the page of the file last enlarged
    const i = light ? shown.findIndex((x) => x.path === light.path) : -1;
    if (layout === "grid" && i >= 0) setPage(Math.floor(i / PAGE) + 1);
    setLight(null);
  };
  const lit = light && items?.find((x) => x.path === light.path);
  const captionSaved = (path: string, c: string) =>
    setItems((xs) => xs!.map((x) => (x.path === path ? { ...x, caption: c, has_caption: true } : x)));
  const missing = (items || []).filter((i) => !i.caption).length;
  const captioned = (items?.length || 0) - missing;
  const done = !!items?.length && !missing;
  const onlyAudio = !!items?.length && items.every((i) => i.kind === "audio");
  const kind = d && datasetKind(d);
  const picked = shown.filter((i) => sel.picked.has(i.path));
  const canDelete = !!d?.managed; // a linked folder's files are only deleted in the folder itself

  return (
    <div className="page wide"
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setOver(false); }}
      onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files); }}>
      <div className="page-head dd-head">
        <div style={{ minWidth: 0 }}>
          <div className="crumbs"><a href="#/datasets">Datasets</a><span className="faint">/</span><span>{d?.name}</span></div>
          <div className="dd-title">
            <h1 className="ellipsis">{d?.name}</h1>
            {kind && <span className={"ds-kind lg " + kind.kind}>{kind.icon}{kind.label}</span>}
          </div>
          <p className="ellipsis" title={d?.path}>{d?.path}</p>
        </div>
        <div className="row wrap dd-actions">
          <button className="btn lg" onClick={() => fileRef.current?.click()}><IconUpload /> Upload</button>
          <button className="btn lg" onClick={() => setDlOpen(true)} disabled={dl?.running}
            title="Download a video, or only its sound, from a web page into this dataset (experimental)"><IconLink /> Add from URL</button>
          <button className="btn lg" onClick={() => setEditOpen(true)} disabled={!items?.length} title="Find and replace, add or remove words in every caption"><IconEdit /> Edit captions</button>
          <Menu label="Tools" text="Tools" icon={<IconTools />} items={[
            ...(items?.some((i) => i.kind === "video") && !d?.split ? [{ label: "Split into clips", icon: <IconScissors />, onClick: () => setSplitOpen(true),
              disabled: !!split?.running, title: "Make a copy of this dataset with each video cut into clips a video model trains on whole" }] : []),
            ...(d?.split ? [{ label: "Edit clips", icon: <IconScissors />, onClick: () => go("datasets", id, "clips"),
              title: `Move, resize, crop, add or remove the clips cut from ${d.split.source.name}'s videos` }] : []),
            ...(items?.some((i) => i.kind === "image") ? [{ label: "Upscale small images", icon: <IconUpscale />, onClick: () => setUpOpen(true),
              disabled: upActive || !!up?.elsewhere,
              title: up?.elsewhere ? `${up.elsewhere.name} is being upscaled: one dataset at a time`
                : "Make the images smaller than the training resolution bigger, keeping the originals in _originals" }] : []),
            ...((items?.filter((i) => i.kind !== "audio").length ?? 0) >= 2 ? [{ label: "Find duplicates", icon: <IconCopy />, onClick: () => setDupOpen(true),
              title: "Images and clips that look alike (copies, re-encodes, crops), to delete the extras" }] : []),
            { label: "Export", icon: <IconDownload />, onClick: () => setExporting(true), disabled: !items?.length, title: "Download it as a .zip" },
            ...(d && !d.managed ? [{ label: "Change folder", icon: <IconFolder />, onClick: () => setMoving(true), title: "Moved its folder? Say where it is now" }] : []),
            { label: "Delete dataset", icon: <IconTrash />, onClick: () => setConfirmRemove(true), danger: true, sep: true },
          ]} />
          <button className="btn primary lg" onClick={() => setCapOpen(true)} disabled={!items?.length || onlyAudio}
            title={onlyAudio ? "Auto-Caption describes images and clips; write each song's style tags yourself (lyrics go after a [Lyrics] line)" : undefined}>
            <IconSparkle /> Auto-Caption</button>
          <input ref={fileRef} type="file" multiple hidden accept="image/*,video/*,audio/*,.txt"
            onChange={(e) => { if (e.target.files) upload(e.target.files); e.target.value = ""; /* so picking the same files again uploads them again */ }} />
        </div>
      </div>

      {d && !d.exists && (
        <div className="error-box row" style={{ marginBottom: 16 }}>
          <span>Its folder isn't there any more: <span className="mono">{d.path}</span>.{!d.managed && " If you moved it, say where it is now."}</span>
          {!d.managed && <><span className="spacer" /><button className="btn sm" onClick={() => setMoving(true)}><IconFolder /> Change folder</button></>}
        </div>
      )}

      <div className="card dd-stats">
        <div className="dd-stats-row">
          <label className="field dd-trigger">
            <span>Trigger word</span>
            <span className="input-icon">
              <input value={trigger} onChange={(e) => setTrigger(e.target.value)} onBlur={saveTrigger} onKeyDown={(e) => e.key === "Enter" && saveTrigger()} placeholder="none" />
              <IconEdit />
            </span>
          </label>
          <div className="dd-stat">
            <IconFile />
            <div><b>{items?.length ?? "…"}</b><span>file{items?.length === 1 ? "" : "s"}</span></div>
          </div>
          {!!d?.masks && (
            <div className="dd-stat" title="Images with a loss mask in a _masks folder beside them, for Masked loss (Create Job)">
              <IconTarget />
              <div><b>{d.masks}</b><span>mask{d.masks === 1 ? "" : "s"}</span></div>
            </div>
          )}
          <div className="dd-stat">
            <span className={"dd-cap-icon" + (done ? " ok" : missing && captioned ? " part" : "")}>{done ? <IconCheckCircle /> : <IconAlert />}</span>
            <div><b>{items ? `${captioned} / ${items.length}` : "…"}</b><span>captioned</span></div>
            <div className={"bar dd-bar" + (done ? " ok" : "")}><div style={{ width: `${items?.length ? (100 * captioned) / items.length : 0}%` }} /></div>
          </div>
          <label className="check dd-replace" title="Caption files you upload for files already in the dataset replace their captions. Off: a file's caption is kept and the uploaded one skipped.">
            <input type="checkbox" checked={replaceCaptions} onChange={(e) => setReplaceCaptions(e.target.checked)} />
            Uploaded captions replace existing ones
          </label>
        </div>
        {cap && (cap.running || cap.errors.length > 0) && (
          <TaskProgress className="dd-task" icon={<IconSparkle />} running={cap.running} errors={cap.errors.slice(-3)}
            label={!cap.running ? "Captioning finished" : cap.phase === "transcribing" ? "Transcribing speech"
              : cap.phase === "starting" ? cap.message || "Starting the caption model" : "Captioning"}
            done={cap.phase === "transcribing" ? cap.heard : cap.phase === "starting" ? undefined : cap.done}
            total={cap.phase === "transcribing" ? cap.to_hear : cap.phase === "starting" ? undefined : cap.total}
            detail={cap.running && cap.phase !== "starting" ? (cap.phase === "transcribing" ? cap.message : cap.current) : undefined}
            onCancel={() => api.del(`/api/datasets/${id}/autocaption`)} />
        )}
        {d?.split && d.split.changes > 0 && (
          <div className="validation warning dd-task">
            {d.split.changes} change{d.split.changes === 1 ? "" : "s"} to the clips {d.split.changes === 1 ? "isn't" : "aren't"} cut yet:
            <a href={`#/datasets/${id}/clips`}>open the clip editor</a>
          </div>
        )}
        {split?.target && (split.running || split.done > 0) && (
          <TaskProgress className="dd-task" icon={<IconScissors />} running={split.running} errors={split.errors}
            label={split.running ? (split.mode === "plan" ? "Planning clips in" : "Splitting")
              : split.mode === "plan" ? <>Clips planned for <a href={`#/datasets/${split.target.id}/clips`}>{split.target.name}</a>: review them in the clip editor, then cut them</>
              : split.cancelled ? (split.fresh ? "Split cancelled; the partial copy was removed" : "Stopped; the clips cut so far stay")
              : <>Split finished: {split.clips} clip{split.clips === 1 ? "" : "s"} cut{split.removed ? `, ${split.removed} removed` : ""} in <a href={`#/datasets/${split.target.id}`}>{split.target.name}</a></>}
            done={split.done} total={split.total}
            detail={split.running && <>{split.mode === "plan" ? "files" : `files · ${split.clips} clips`} {split.current}{split.message ? ` · ${split.message}` : ""}
              {split.mode === "plan" && <> · <a href={`#/datasets/${split.target.id}/clips`}>open the clip editor</a></>}</>}
            onCancel={() => api.del(`/api/datasets/${id}/split`).then(reloadSplit)} />
        )}
        {up && up.state !== "none" && up.id !== upHidden && (
          <DatasetUpscaleTask up={up} onCancel={() => api.post(`/api/datasets/${id}/upscale/cancel`).then(reloadUp)}
            onDismiss={() => setUpHidden(up.id || null)} />
        )}
        {dl && (dl.running || dl.count > 0 || dl.errors.length > 0 || dl.cancelled) && (
          <TaskProgress className="dd-task" icon={<IconDownload />} running={dl.running} errors={dl.errors}
            label={dl.running ? (dl.phase === "processing" ? dl.step : dl.phase === "downloading" ? "Downloading" : "Starting the download")
              : dl.cancelled ? `Download cancelled${dl.count ? `; the ${dl.count} file${dl.count === 1 ? "" : "s"} already added stay` : ""}`
              : <>{dl.count ? `Added ${dl.count} file${dl.count === 1 ? "" : "s"} from the web` : "Nothing was downloaded"}
                {dl.count === 1 && <span className="faint"> · {dl.saved[0]}</span>}</>}
            done={(dl.items || 0) > 1 ? dl.item : undefined} total={(dl.items || 0) > 1 ? dl.items : undefined}
            pct={dl.total_bytes ? (100 * ((dl.item || 1) - 1 + (dl.done_bytes || 0) / dl.total_bytes)) / Math.max(1, dl.items || 1) : null}
            detail={dl.running && `${dl.title || ""}${dl.phase === "downloading" && dl.done_bytes ? ` · ${fmtSize(dl.done_bytes)}${dl.total_bytes ? ` of ${fmtSize(dl.total_bytes)}` : ""}` : ""}`}
            onCancel={() => api.del(`/api/datasets/${id}/download`).then(reloadDl)} />
        )}
      </div>

      <HealthCheck id={id} name={d?.name || ""} datasets={ds} families={families} onOpen={(path) => { if (items?.some((x) => x.path === path)) open(path); }} />

      {(over || uploading) && (
        <div className={"dropzone over"} style={{ marginBottom: 16 }}>{uploading || "Drop images, videos or audio to upload"}</div>
      )}

      {itemsError && !items && <LoadError what="the dataset's files" error={itemsError} onRetry={loadItems} />}
      {items && items.length === 0 && (
        <div className="dropzone" onClick={() => fileRef.current?.click()}>
          <IconUpload width={28} height={28} /><br />
          <strong style={{ color: "var(--text)" }}>No images, videos or audio yet</strong><br />
          Drag files here or click to upload. Matching .txt caption files are picked up too (for a song: its style
          tags, then any lyrics after a [Lyrics] line).
        </div>
      )}

      {!!items?.length && (
        <div className="dd-toolbar" ref={toolbar}>
          <label className="search">
            <IconSearch />
            <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search files or captions…" aria-label="Search files or captions"
              onKeyDown={(e) => e.key === "Escape" && search && (e.stopPropagation(), setSearch(""))} />
          </label>
          <div className="seg dd-filter">
            <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>All <span>{items.length}</span></button>
            <button className={filter === "missing" ? "on" : ""} onClick={() => setFilter("missing")}>Missing captions <span>{missing}</span></button>
          </div>
          <span className="spacer" />
          <div className="view-toggle" role="group" aria-label="View">
            <button aria-pressed={layout === "grid"} title="Grid: bigger cards, a page at a time" onClick={() => setLayout("grid")}><IconGrid /></button>
            <button aria-pressed={layout === "list"} title="List: every file on one page" onClick={() => setLayout("list")}><IconList /></button>
          </div>
          {canDelete && !sel.on && <button className="btn lg" onClick={sel.start} title="Pick files to delete together"><IconCheck /> Select</button>}
        </div>
      )}
      {sel.on && (
        <SelectBar sel={sel} ids={shown.map((i) => i.path)} noun={["file", "files"]}
          info={<span className="muted small">{fmtSize(picked.reduce((n, i) => n + (i.size || 0), 0))}</span>}>
          <button className="btn danger" onClick={() => setDeleting(picked.map((i) => i.path))} disabled={!picked.length}><IconTrash /> Delete</button>
        </SelectBar>
      )}
      {!!items?.length && !shown.length && (
        <div className="empty">
          <strong>{filter === "missing" && !words.length ? "Every file has a caption" : "No files match"}</strong>
          {!!words.length && <button className="btn ghost" onClick={() => setSearch("")}>Clear the search</button>}
        </div>
      )}

      {layout === "grid" ? (
        <div className="file-grid">
          {paged.map((it) => (
            <FileCard key={it.path} it={it} audio={player.current} song={song?.path === it.path ? song.on : null} onPlay={() => play(it.path)} onOpen={() => open(it.path)}
              onSaved={(c) => captionSaved(it.path, c)} onDelete={canDelete ? () => setDeleting([it.path]) : undefined} sel={canDelete ? sel : undefined} />
          ))}
        </div>
      ) : (
        <div className="media-grid">
          {paged.map((it) => {
            const on = sel.picked.has(it.path);
            return (
              <div className={"media-card" + (on ? " picked" : "")} key={it.path}>
                <div className={"media-thumb" + (it.kind === "audio" ? " audio" : "")} style={{ backgroundImage: `url(${thumbUrl(it)})` }}
                  onClick={() => (sel.on ? sel.toggle(it.path) : open(it.path))}>
                  {it.kind === "video" && <span className="badge">VIDEO</span>}
                  {it.kind === "audio" && <span className="badge">AUDIO</span>}
                  {sel.on ? <input type="checkbox" className="pick media-pick" checked={on} aria-label={`Select ${it.name}`} onClick={(e) => e.stopPropagation()} onChange={() => sel.toggle(it.path)} />
                    : !it.caption && <span className="nocap" title="No caption" />}
                </div>
                <CaptionCell item={it} onSaved={(c) => captionSaved(it.path, c)} />
                <div className="row small" style={{ padding: "0 10px 8px" }}>
                  <span className="faint ellipsis" title={it.cut_from ? `${it.name}: cut from ${it.cut_from.source} at ${fmtClock(it.cut_from.start)}` : it.name}>{it.name}</span>
                  <span className="spacer" />
                  {canDelete ? <button className="btn ghost icon sm" title="Delete file" onClick={() => setDeleting([it.path])}><IconTrash /></button> : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
      {layout === "grid" && shown.length > 0 && (
        <div className="pager-row">
          <span className="muted">Showing {(at - 1) * PAGE + 1}–{Math.min(at * PAGE, shown.length)} of {shown.length}</span>
          {pages > 1 && <Pager page={at} pages={pages} onPage={goPage} />}
        </div>
      )}
      <audio ref={player} hidden onEnded={() => setSong(null)}
        onPlay={() => setSong((x) => x && { ...x, on: true })} onPause={() => setSong((x) => x && { ...x, on: !player.current?.paused })} />

      {light && lit && <Lightbox src={`/api/media?path=${q(lit.path)}`} video={lit.kind === "video"} onClose={closeLight}
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
      {deleting && (
        <ConfirmDialog title={deleting.length === 1 ? "Delete file" : `Delete ${deleting.length} files`} danger confirmLabel="Delete"
          onClose={() => setDeleting(null)} onConfirm={removeFiles}
          message={<>Delete <strong style={{ color: "var(--text)" }}>{deleting.length === 1 ? items?.find((i) => i.path === deleting[0])?.name : `${deleting.length} files`}</strong>{" "}
            and {deleting.length === 1 ? "its caption" : "their captions"} from disk? This can't be undone.</>} />
      )}
      {exporting && d && <ExportDialog datasets={[d]} onClose={() => setExporting(false)} />}
      {moving && d && <ChangeFolder dataset={d} onClose={() => setMoving(false)} onMoved={() => { reloadDs(); loadItems(); }} />}
      {confirmRemove && d && <DeleteDatasets datasets={[d]} onClose={() => setConfirmRemove(false)} onDeleted={() => go("datasets")} />}
      {capOpen && d && <AutoCaptionModal dataset={d} missing={missing} hasVideo={!!items?.some((i) => i.kind === "video")}
        onClose={() => { setCapOpen(false); reloadDs(); }} />}
      {editOpen && d && <EditCaptionsModal dataset={d} onClose={() => setEditOpen(false)} onChanged={() => { loadItems(); reloadDs(); }} />}
      {splitOpen && d && <SplitModal dataset={d} families={families} onClose={() => setSplitOpen(false)} onStarted={reloadSplit} />}
      {dlOpen && d && <DownloadModal dataset={d} onClose={() => setDlOpen(false)} onStarted={reloadDl} />}
      {upOpen && d && <DatasetUpscaleModal dataset={d} last={up?.last ?? null} onClose={() => setUpOpen(false)}
        onStarted={() => { setUpHidden(null); reloadUp(); }} />}
      {dupOpen && d && <DuplicatesDialog dataset={d} onClose={() => setDupOpen(false)} onChanged={() => { loadItems(); reloadDs(); }} />}
    </div>
  );
}

/** Delete datasets (one, from its page or its card, or those picked with Select): from Dainamo, and with the box
 * ticked their uploaded files too. A queued or running job that uses one is named, since deleting its files makes it
 * fail. The one dialog for it. */
/** Point a linked dataset at the folder it's in now, after it was moved; jobs using it follow it there (and keep
 * its cached latents) when they next run. */
function ChangeFolder({ dataset, onClose, onMoved }: { dataset: Dataset; onClose: () => void; onMoved: () => void }) {
  const toast = useToast();
  const [path, setPath] = useState(dataset.path);
  const [busy, setBusy] = useState(false);
  const save = async () => {
    setBusy(true);
    try {
      await api.put(`/api/datasets/${dataset.id}`, { name: dataset.name, trigger: dataset.trigger, path });
      toast(`${dataset.name} is now in ${path}`);
      onMoved();
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Change folder" onClose={onClose} footer={<>
      <button className="btn ghost" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={busy || !path.trim() || path.trim() === dataset.path} onClick={save}><IconFolder /> Use this folder</button>
    </>}>
      <div className="stack">
        <p className="muted" style={{ margin: 0 }}>Moved <strong style={{ color: "var(--text)" }}>{dataset.name}</strong>'s folder? Choose
          where it is now. Jobs using it carry on from there, and the latents they cached for its files are still used.</p>
        <PathInput mode="dir" value={path} onChange={setPath} />
      </div>
    </Modal>
  );
}

function DeleteDatasets({ datasets, onClose, onDeleted }: { datasets: Dataset[]; onClose: () => void; onDeleted: () => void }) {
  const toast = useToast();
  const [jobs] = usePoll<Job[]>("/api/jobs", 0);
  const uploaded = datasets.filter((d) => d.managed);
  const names = new Set(datasets.map((d) => d.name));
  const using = (jobs || []).filter((j) => ["ready", "queued", "running", "stopping"].includes(j.status) && j.datasets.some((n) => names.has(n)));
  const remove = async (files: boolean) => {
    let removed = 0;
    const failed: string[] = [];
    for (const d of datasets) {
      try {
        await api.del(`/api/datasets/${d.id}?delete_files=${files}`);
        removed++;
      } catch (e: any) {
        failed.push(datasets.length > 1 ? `${d.name}: ${e.message}` : e.message);
      }
    }
    toast(failed.length ? `${removed ? `Deleted ${removed}; c` : "C"}ouldn't delete ${failed.join("; ")}`
      : `Deleted ${datasets.length === 1 ? datasets[0].name : `${removed} datasets`}`, failed.length > 0);
    if (removed) onDeleted();
  };
  const one = datasets.length === 1;
  return (
    <ConfirmDialog title={one ? "Delete dataset" : `Delete ${datasets.length} datasets`} danger confirmLabel="Delete" onClose={onClose} onConfirm={remove}
      message={<>
        Remove <strong style={{ color: "var(--text)" }}>{someNames(datasets.map((d) => d.name))}</strong> from Dainamo?
        {uploaded.length < datasets.length && (one
          ? <> The folder <span className="mono">{datasets[0].path}</span> and its files are left as they are.</>
          : " Linked folders and their files are left as they are.")}
        {using.length > 0 && <div className="validation warning" style={{ marginTop: 10 }}>
          {someNames(using.map((j) => j.name))} {using.length === 1 ? "is" : "are"} queued or running with {one ? "it" : "some of them"}:
          deleting the files makes {using.length === 1 ? "that job" : "those jobs"} fail.</div>}
      </>}
      checkbox={uploaded.length ? `Also delete the uploaded files and captions${uploaded.length < datasets.length ? ` of ${someNames(uploaded.map((d) => d.name))}` : ""}` : undefined} />
  );
}

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
  const attempt = useToastErrors();
  const [profiles, setProfiles] = useState<Profile[] | null>(null);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [ram, setRam] = useState<SystemRam | null>(null);
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
    attempt(api.get<Profile[]>("/api/profiles"), "Couldn't load the profiles").then((ps) => {
      if (!ps) return;
      const video = ps.filter((p) => videoFamilies.some((f) => f.id === p.family));
      setProfiles(video);
      if (video[0]) pickProfile(byFamily(video, videoFamilies)[0].profiles[0]);
    });
    attempt(api.get<{ gpus: Gpu[]; ram?: SystemRam }>("/api/system"), "Couldn't load the GPUs").then((sys) => {
      if (!sys) return;
      setGpus(sys.gpus);
      setRam(sys.ram || null);
      if (sys.gpus.length) setGpu(sys.gpus.reduce((a, b) => ((b.mem_total ?? Infinity) < (a.mem_total ?? Infinity) ? b : a)).index);
    });
  }, []);
  const gpuGb = (gpus.find((g) => g.index === gpu)?.mem_total ?? 0) / 1024;
  useEffect(() => {
    if (!profileId || !gpuGb) return;
    let live = true;
    setFit(null);
    api.post<Partial<ClipFit>>(`/api/datasets/${dataset.id}/split/fit`, { profile_id: profileId, resolution, offload, low_vram: lowVram, gpu_gb: gpuGb })
      .then((answer) => {
        if (!live) return;
        // a projection that failed is only {error}: no lengths to pick from
        const f: ClipFit = { frames: null, fps: 24, ...answer, lengths: answer.lengths || [] };
        setFit(f);
        setFrames(f.frames ?? f.lengths[0]?.frames ?? 0);
      })
      .catch((e) => live && setFit({ frames: null, fps: 24, lengths: [], error: e.message }));
    return () => { live = false; };
  }, [profileId, resolution, offload, lowVram, gpuGb]);
  const chosen = fit?.lengths.find((l) => l.frames === frames);
  // the job's projection at the chosen clip length, for the shared offload control (as the split's fit is reckoned)
  const proj = useVramProjection(profileId || undefined, {
    ...(family?.training_defaults || {}), resolution, offload, low_vram: lowVram, num_frames: frames || 1,
    datasets: [{ resolution, video: true }],
  }, "train");
  const start = async (review: boolean) => {
    try {
      const job = await api.post<SplitStatus>(`/api/datasets/${dataset.id}/split`, {
        name, family: family!.id, frames, max_clips: maxClips, scenes, keep_short: keepShort, max_side: maxSide, captions, review,
      });
      onStarted();
      onClose();
      if (review) go("datasets", job.target!.id, "clips");
      else toast("Splitting started");
    } catch (e: any) {
      toast(e.message, true);
    }
  };
  return (
    <Modal title="Split into clips" onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn" onClick={() => start(true)} disabled={!family || !(frames > 1) || !name.trim()}
          title="Plan the clips, then move, resize, crop, add or remove them in the clip editor before they're cut">Review first</button>
        <button className="btn primary" onClick={() => start(false)} disabled={!family || !(frames > 1) || !name.trim()}><IconScissors /> Split</button>
      </>
    }>
      <div className="stack">
        <p className="muted" style={{ margin: 0 }}>
          Makes a new dataset with each video cut into clips of one length, inside one shot each and spread over the
          video. This dataset stays as it is. Images and sound files are copied as they are. Each video then trains
          once per epoch, as one of its clips picked at random, so a long video counts no more than a short one.
          <strong style={{ color: "var(--text)" }}> Review first</strong> opens the clip editor before anything is cut;
          the clips can be changed there later too.
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
              {gpus.map((g) => <option key={g.index} value={g.index}>GPU {g.index} · {g.name} · {g.mem_total != null ? `${(g.mem_total / 1024).toFixed(0)} GB` : "memory not reported"}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Training resolution</span>
            <select value={resolution} onChange={(e) => setResolution(Number(e.target.value))}>
              {[...new Set([...SPLIT_SIZES, resolution])].sort((a, b) => a - b).map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </label>
        </div>
        <OffloadControl proj={proj} value={offload} onChange={setOffload} gpu={gpuTarget(gpus, gpu)} ram={ram} compact
          lowVram={family?.streams_text_encoder ? { on: lowVram, onChange: setLowVram } : null} />
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
  const attempt = useToastErrors();
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
    api.post<typeof test>("/api/captioner/test", {}).then(setTest, (e) => setTest({ ok: false, error: e.message }));
    attempt(api.get<Settings>("/api/settings"), "Couldn't load the caption styles").then((st) => {
      if (!st) return;
      setStyles(st.caption_styles);
      const known = st.caption_styles.styles.some((x) => x.id === last?.style);
      setStyle(known ? last!.style : st.caption_styles.default);
      setSpeech((m) => m || st.speech.model);
    });
    if (hasVideo) attempt(api.get<{ whisper_models: WhisperModel[] }>("/api/system"), "Couldn't load the Whisper models").then((sys) => sys && setWhisper(sys.whisper_models));
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
          {/* a system prompt (FLUX 3's style) is the style's substance: it shows first, the prompt after it */}
          {chosen && <span className="caption-prompt" style={{ whiteSpace: "pre-wrap" }}>{chosen.system ? `${chosen.system}\n\n${chosen.prompt}` : chosen.prompt}</span>}
          <span className="help">
            {last && chosen?.id === last.style ? "The style this dataset was captioned with last time. " : ""}
            {chosen && (chosen.temperature != null || chosen.max_tokens != null) ? `It has its own settings (${[
              chosen.temperature != null && `temperature ${chosen.temperature}`, chosen.max_tokens != null && `${chosen.max_tokens} tokens at most`,
            ].filter(Boolean).join(", ")}). ` : ""}
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
          <span>Add to each caption <span className="faint">· optional</span></span>
          <input value={trigger} onChange={(e) => setTrigger(e.target.value)} placeholder="nothing" />
          <span className="help">Written at the start of each caption made now, as "words, caption…", unless it's there already. It starts
            as the dataset's trigger word; changing it here doesn't change that.</span>
        </label>
      </div>
    </Modal>
  );
}

/** Several datasets as one .zip (``/api/export/datasets``): a folder each, with captions, reference images and masks. */
function ExportDialog({ datasets, onClose }: { datasets: Dataset[]; onClose: () => void }) {
  const toast = useToast();
  const triggers = datasets.filter((d) => d.trigger);
  const [withTrigger, setWithTrigger] = useState(triggers.length > 0);
  const files = datasets.reduce((n, d) => n + d.count, 0);
  const download = () => {
    const a = document.createElement("a");
    a.href = `/api/export/datasets?ids=${datasets.map((d) => d.id).join(",")}&trigger=${withTrigger}`;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
    toast("The download has started");
    onClose();
  };
  return (
    <Modal title={datasets.length === 1 ? `Export ${datasets[0].name}` : `Export ${datasets.length} datasets`} onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" autoFocus onClick={download}><IconDownload /> Download .zip</button>
      </>
    }>
      <div className="stack">
        <p className="muted" style={{ margin: 0 }}>
          A .zip with {datasets.length === 1 ? "the dataset's folder" : "a folder for each dataset"}: its {files} file{files === 1 ? "" : "s"}{" "}
          ({fmtSize(datasets.reduce((n, d) => n + d.bytes, 0))}) in their subfolders, each with its caption .txt beside it, plus
          reference images, loss masks and a split's clip list. Nothing is compressed again, so the download starts right away.
        </p>
        {triggers.length > 0 && (
          <div className="field">
            <label className="check"><input type="checkbox" checked={withTrigger} onChange={(e) => setWithTrigger(e.target.checked)} /> Put the trigger word in each caption</label>
            <span className="help">Dainamo adds {someNames([...new Set(triggers.map((d) => `"${d.trigger}"`))])} to the captions when it trains; another trainer reads
              them as they are. A file without a caption gets one with just the trigger word. Off: the captions as they are on disk.</span>
          </div>
        )}
      </div>
    </Modal>
  );
}

const MEDIA: [Media, string][] = [["all", "All media"], ["images", "Images"], ["videos", "Videos"], ["audio", "Audio"]];
type Media = "all" | "images" | "videos" | "audio";
const MEDIA_FILTER = "dainamo.datasets.media";

/** What a dataset holds, for its badge: "Images", "Images & videos", "Mixed" (all three), or nothing while it's empty. */
function mediaBadge(d: Dataset): { label: string; kind: string } | null {
  const kinds = (["images", "videos", "audio"] as const).filter((k) => d[k] > 0);
  const name = { images: "Images", videos: "Videos", audio: "Audio" };
  if (kinds.length === 1) return { label: name[kinds[0]], kind: kinds[0] };
  if (kinds.length === 2) return { label: `${name[kinds[0]]} & ${name[kinds[1]].toLowerCase()}`, kind: "mixed" };
  return kinds.length ? { label: "Mixed", kind: "mixed" } : null;
}

/** A long path's last folders, "…/loras/set_a", so the folder's own name shows. */
const tailPath = (p: string, max = 44) => {
  if (p.length <= max) return p;
  const sep = p.includes("\\") ? "\\" : "/"; // Windows'
  const parts = p.split(sep).filter(Boolean);
  let out = parts.pop() || p;
  while (parts.length && out.length + parts[parts.length - 1].length + 1 <= max - 2) out = `${parts.pop()}${sep}${out}`;
  return `…${sep}${out}`;
};

/** A dataset in the list: its cover, name, what it holds and where it came from, how much is captioned, its trigger
 * word and (linked) folder. A click opens it, or picks it while selecting; ⋯ exports or deletes it. */
function DatasetCard({ d, sel, onExport, onMove, onDelete }: { d: Dataset; sel: Selection; onExport: () => void; onMove: () => void; onDelete: () => void }) {
  const picked = sel.picked.has(d.id);
  const badge = mediaBadge(d);
  const done = d.count > 0 && d.captioned === d.count;
  const status = !d.exists ? <span className="danger-text">Its folder is missing</span>
    : !d.count ? <span className="muted">No files yet</span>
    : done ? <span className="ok-text">All files captioned</span>
    : !d.captioned ? <span className="muted">No captions yet</span>
    : <span className="muted">{d.count - d.captioned} without a caption</span>;
  return (
    <div className={"card ds-card" + (sel.on ? " selecting" : "") + (picked ? " picked" : "")}
      onClick={() => (sel.on ? sel.toggle(d.id) : go("datasets", d.id))}>
      <div className="ds-card-top">
        <div className="ds-cover" style={d.cover ? { backgroundImage: `url(/api/thumb?size=192&path=${q(d.cover)})` } : {}}>
          {!d.cover && <IconImages width={26} />}
        </div>
        <div className="ds-title">
          <div className="ds-name">
            <h3 className="ellipsis" title={d.name}>{d.name}</h3>
            {badge && <span className={"ds-kind " + badge.kind}>{badge.label}</span>}
          </div>
          {d.split ? <span className="ds-source" title={`Clips cut from ${d.split.source.name}'s videos`}><IconScissors /><span className="ellipsis">Clips of {d.split.source.name}</span></span>
            : d.managed ? <span className="ds-source"><IconUpload /> Uploaded</span>
            : <span className="ds-source"><IconFolder /> Linked folder</span>}
        </div>
        {sel.on
          ? <input type="checkbox" className="pick" checked={picked} aria-label={`Select ${d.name}`} onClick={(e) => e.stopPropagation()} onChange={() => sel.toggle(d.id)} />
          : <Menu label={`More for ${d.name}`} items={[
            { label: "Export", icon: <IconDownload />, onClick: onExport, disabled: !d.exists, title: d.exists ? "Download it as a .zip" : "Its folder is missing" },
            ...(d.managed ? [] : [{ label: "Change folder", icon: <IconFolder />, onClick: onMove, title: "Moved its folder? Say where it is now" }]),
            { label: "Delete", icon: <IconTrash />, onClick: onDelete, danger: true },
          ]} />}
      </div>
      <div className="ds-progress">
        <div className="ds-progress-row"><span>{d.count} file{d.count === 1 ? "" : "s"}</span><span>{d.captioned} / {d.count} captioned</span></div>
        <div className={"bar" + (done ? " ok" : "")}><div style={{ width: `${d.count ? (100 * d.captioned) / d.count : 0}%` }} /></div>
        <div className="ds-status">{status}</div>
      </div>
      {(d.trigger || !d.managed) && (
        <div className="ds-facts">
          {d.trigger && <div><div className="ds-fact">Trigger word</div><code className="ds-chip">{d.trigger}</code></div>}
          {!d.managed && <div><div className="ds-fact">Folder</div><code className="ds-chip ellipsis" title={d.path}>{tailPath(d.path)}</code></div>}
        </div>
      )}
      <div className="ds-foot">
        <a className="btn ds-open" href={`#/datasets/${encodeURIComponent(d.id)}`} tabIndex={sel.on ? -1 : undefined}
          onClick={(e) => { e.stopPropagation(); if (sel.on) { e.preventDefault(); sel.toggle(d.id); } }}>Open dataset <IconArrowRight /></a>
      </div>
    </div>
  );
}

export default function Datasets({ datasetId, view, families }: { datasetId: string; view: string; families: Family[] }) {
  const [datasets, reload, loadError] = usePoll<Dataset[]>(datasetId ? null : "/api/datasets", 0, [datasetId]);
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState("");
  const [media, setMediaState] = useState<Media>(() => {
    try {
      const m = localStorage.getItem(MEDIA_FILTER);
      return MEDIA.find(([v]) => v === m)?.[0] || "all";
    } catch {
      return "all";
    }
  });
  const setMedia = (m: Media) => {
    setMediaState(m);
    try {
      localStorage.setItem(MEDIA_FILTER, m);
    } catch { /* not kept */ }
  };
  // Select (or ⋯ on one): exported or deleted together
  const sel = useSelection();
  const [dialog, setDialog] = useState<{ kind: "export" | "move" | "delete"; ids: string[] } | null>(null);
  if (datasetId && view === "clips") return <ClipEditor key={datasetId} id={datasetId} />;
  if (datasetId) return <DatasetDetail key={datasetId} id={datasetId} families={families} />; // fresh state per dataset
  const words = search.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = (datasets || []).filter((d) => (media === "all" || d[media] > 0)
    && words.every((w) => `${d.name} ${d.trigger} ${d.managed ? "" : d.path}`.toLowerCase().includes(w)));
  const picked = shown.filter((d) => sel.picked.has(d.id)); // what Select acts on: only those in view
  const chosen = (datasets || []).filter((d) => dialog?.ids.includes(d.id));
  const filtered = !!words.length || media !== "all";
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Datasets{!!datasets?.length && <span className="count-pill">{datasets.length}</span>}</h1>
          <p>Organize images, videos and music for training.</p>
        </div>
        <div className="row">
          {!!datasets?.length && !sel.on && <button className="btn lg" onClick={sel.start}><IconCheck /> Select</button>}
          <button className="btn primary lg" onClick={() => setCreating(true)}><IconPlus /> New dataset</button>
        </div>
      </div>
      {!!datasets?.length && (
        <div className="ds-toolbar">
          <label className="search">
            <IconSearch />
            <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search datasets…" aria-label="Search datasets"
              onKeyDown={(e) => e.key === "Escape" && search && (e.stopPropagation(), setSearch(""))} />
          </label>
          <select value={media} onChange={(e) => setMedia(e.target.value as Media)} aria-label="Media">
            {MEDIA.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
          </select>
        </div>
      )}
      {sel.on && (
        <SelectBar sel={sel} ids={shown.map((d) => d.id)} noun={["dataset", "datasets"]}
          info={<span className="muted small">{fmtSize(picked.reduce((n, d) => n + d.bytes, 0))}</span>}>
          <button className="btn" onClick={() => setDialog({ kind: "export", ids: picked.map((d) => d.id) })} disabled={!picked.length || picked.some((d) => !d.exists)}
            title={picked.some((d) => !d.exists) ? "A selected dataset's folder is missing" : "Download them as a .zip"}><IconDownload /> Export</button>
          <button className="btn danger" onClick={() => setDialog({ kind: "delete", ids: picked.map((d) => d.id) })} disabled={!picked.length}><IconTrash /> Delete</button>
        </SelectBar>
      )}
      {dialog?.kind === "export" && <ExportDialog datasets={chosen} onClose={() => setDialog(null)} />}
      {dialog?.kind === "move" && chosen[0] && <ChangeFolder dataset={chosen[0]} onClose={() => setDialog(null)} onMoved={reload} />}
      {dialog?.kind === "delete" && <DeleteDatasets datasets={chosen} onClose={() => setDialog(null)} onDeleted={() => { sel.stop(); reload(); }} />}
      {loadError && !datasets && <LoadError what="the datasets" error={loadError} onRetry={reload} />}
      {datasets && datasets.length === 0 && (
        <Empty icon={<IconImages />} title="No datasets yet" action={<button className="btn primary lg" onClick={() => setCreating(true)}><IconPlus /> New dataset</button>}>
          Upload images, videos or music, or link a folder you already have.
        </Empty>
      )}
      {!!datasets?.length && !shown.length && (
        <div className="empty">
          <strong>No datasets match</strong>
          <button className="btn ghost" onClick={() => { setSearch(""); setMedia("all"); }}>Clear the search{media !== "all" ? " and filter" : ""}</button>
        </div>
      )}
      <div className="grid cols-2 ds-grid">
        {shown.map((d) => (
          <DatasetCard key={d.id} d={d} sel={sel} onExport={() => setDialog({ kind: "export", ids: [d.id] })}
            onMove={() => setDialog({ kind: "move", ids: [d.id] })} onDelete={() => setDialog({ kind: "delete", ids: [d.id] })} />
        ))}
      </div>
      {!!datasets?.length && !filtered && (
        <>
          <div className="card ds-note"><IconInfo /> Captions are stored beside each file as .txt files.</div>
          <p className="muted small" style={{ margin: "12px 0 0" }}>Add files by uploading media or linking an existing folder.</p>
        </>
      )}
      {creating && <NewDataset onClose={() => setCreating(false)} onCreated={(d) => { setCreating(false); reload(); go("datasets", d.id); }} />}
    </div>
  );
}

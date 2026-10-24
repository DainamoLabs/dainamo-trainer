import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, byFamily, fmtClock, q, type Family, type Profile } from "../api";
import { IconCheck, IconDownload, IconFile, IconFolder, IconAlert, IconLeft, IconMore, IconMute, IconPause, IconPlayFill, IconRight, IconUp, IconVolume, IconX } from "./icons";

/** A profile <select>'s options, under a heading per family. */
export function ProfileOptions({ profiles, families }: { profiles: Profile[]; families: Family[] }) {
  return (
    <>
      {byFamily(profiles, families).map((g) => (
        <optgroup key={g.id} label={g.label}>
          {g.profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </optgroup>
      ))}
    </>
  );
}

// ---- modal ---------------------------------------------------------------
const ARROWS: Record<string, number> = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 };

/** A dialog. It takes the focus when it opens (a field or button in it with autoFocus keeps it) and gives it back
 * when it closes; the arrow keys move between its footer's buttons and Enter presses the one in focus. Keys pressed
 * in it stay in it, so the page behind (a lightbox's arrows, the clip editor's keys) doesn't act on them too. */
export function Modal(props: { title: React.ReactNode; onClose: () => void; children: React.ReactNode; footer?: React.ReactNode; wide?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const [opener] = useState(() => document.activeElement as HTMLElement | null); // before an autoFocus in it moves it
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && props.onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props.onClose]);
  useEffect(() => {
    if (box.current && !box.current.contains(document.activeElement)) box.current.focus();
    return () => { if (opener && document.contains(opener)) opener.focus(); };
  }, []);
  const onKey = (e: React.KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === "Escape") return props.onClose();
    const by = ARROWS[e.key];
    const at = document.activeElement;
    const buttons = [...box.current!.querySelectorAll<HTMLButtonElement>(".modal-foot button:not(:disabled)")];
    const i = buttons.indexOf(at as HTMLButtonElement);
    if (!by || !buttons.length || (i < 0 && at !== box.current) || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    buttons[i < 0 ? (by > 0 ? 0 : buttons.length - 1) : (i + by + buttons.length) % buttons.length].focus();
  };
  return (
    <div className="modal-bg" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div className={"modal" + (props.wide ? " wide" : "")} ref={box} tabIndex={-1} role="dialog" aria-modal="true" onKeyDown={onKey}>
        <div className="modal-head">
          <h3>{props.title}</h3>
          <div className="spacer" />
          <button className="btn ghost icon" onClick={props.onClose} aria-label="Close"><IconX /></button>
        </div>
        <div className="modal-body">{props.children}</div>
        {props.footer && <div className="modal-foot">{props.footer}</div>}
      </div>
    </div>
  );
}

// ---- toast ---------------------------------------------------------------
type ToastT = { msg: string; err?: boolean; id: number };
const ToastCtx = createContext<(msg: string, err?: boolean) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toast, setToast] = useState<ToastT | null>(null);
  const show = useCallback((msg: string, err?: boolean) => {
    const id = Date.now();
    setToast({ msg, err, id });
    setTimeout(() => setToast((t) => (t && t.id === id ? null : t)), err ? 6000 : 3000);
  }, []);
  return (
    <ToastCtx.Provider value={show}>
      {children}
      {toast && <div className={"toast" + (toast.err ? " err" : "")}>{toast.msg}</div>}
    </ToastCtx.Provider>
  );
}

// ---- polling -------------------------------------------------------------
/** ``url``'s answer, fetched now and every ``ms`` (0: once, and again when ``deps`` change), a reload, and the last
 * error. Nothing is fetched while ``url`` is null. The last answer stays while a new url loads, unless ``fresh``: then
 * the answer and error are only ever the current url's (none while it's null or loading). */
export function usePoll<T>(url: string | null, ms: number, deps: unknown[] = [], fresh = false): [T | null, () => void, string | null] {
  const [data, setData] = useState<{ url: string; v: T } | null>(null);
  const [err, setErr] = useState<{ url: string; msg: string } | null>(null);
  const tick = useRef(0);
  const load = useCallback(() => {
    if (!url) return;
    const my = ++tick.current;
    api.get<T>(url).then(
      (d) => { if (my === tick.current) { setData({ url, v: d }); setErr(null); } },
      (e) => my === tick.current && setErr({ url, msg: String(e.message || e) }),
    );
  }, [url]);
  useEffect(() => {
    load();
    if (!ms) return;
    const t = setInterval(load, ms);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, ms, ...deps]);
  const mine = (x: { url: string } | null) => !!x && (!fresh || x.url === url);
  return [mine(data) ? data!.v : null, load, mine(err) ? err!.msg : null];
}

// ---- file browser ----------------------------------------------------------
type FsList = { path: string; parent: string; dirs: string[]; files: { name: string; size: number }[]; sep: string; roots: string[] };

export function fmtSize(n: number) {
  if (n > 1e9) return (n / 1e9).toFixed(1) + " GB";
  if (n > 1e6) return (n / 1e6).toFixed(1) + " MB";
  return Math.round(n / 1e3) + " KB";
}

export function FileBrowser(props: { start: string; mode: "file" | "dir"; exts?: string; onPick: (p: string) => void; onClose: () => void }) {
  const [list, setList] = useState<FsList | null>(null);
  const [path, setPath] = useState(props.start || "~");
  const [typed, setTyped] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.get<FsList>(`/api/fs?path=${q(path)}&exts=${q(props.exts || "")}`).then(
      (l) => { setList(l); setTyped(l.path); setErr(null); setSel(null); },
      (e) => setErr(e.message),
    );
  }, [path, props.exts]);

  const join = (n: string) => (/[\\/]$/.test(list!.path) ? list!.path + n : list!.path + (list!.sep || "/") + n);
  return (
    <Modal
      title={props.mode === "dir" ? "Choose folder" : "Choose file"}
      onClose={props.onClose}
      footer={
        <>
          <button className="btn ghost" onClick={props.onClose}>Cancel</button>
          {props.mode === "dir" ? (
            <button className="btn primary" disabled={!list} onClick={() => props.onPick(sel ? join(sel) : list!.path)}>
              Use {sel ? "selected" : "this"} folder
            </button>
          ) : (
            <button className="btn primary" disabled={!sel} onClick={() => sel && props.onPick(join(sel))}>Select file</button>
          )}
        </>
      }
    >
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="btn icon" title="Up" disabled={!list} onClick={() => list && setPath(list.parent)}><IconUp /></button>
        <input className="mono" value={typed} onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => e.key === "Enter" && setPath(typed)} />
      </div>
      {!!list?.roots?.length && (
        <div className="row wrap" style={{ gap: 6, marginBottom: 10 }}>
          {list.roots.map((r) => (
            <button key={r} className={"btn sm" + (list.path.toUpperCase().startsWith(r.toUpperCase()) ? " primary" : "")} onClick={() => setPath(r)}>{r.replace(/\\$/, "")}</button>
          ))}
        </div>
      )}
      {err && <div className="error-box" style={{ marginBottom: 10 }}>{err}</div>}
      <div className="fs-list">
        {list?.dirs.map((d) => (
          <div key={d} className={"fs-item dir" + (props.mode === "dir" && sel === d ? " sel" : "")}
            onClick={() => (props.mode === "dir" ? setSel(d) : setPath(join(d)))}
            onDoubleClick={() => setPath(join(d))}>
            <IconFolder /> <span className="ellipsis">{d}</span>
          </div>
        ))}
        {props.mode === "file" && list?.files.map((f) => (
          <div key={f.name} className={"fs-item" + (sel === f.name ? " sel" : "")} onClick={() => setSel(f.name)}
            onDoubleClick={() => props.onPick(join(f.name))}>
            <IconFile /> <span className="ellipsis">{f.name}</span><span className="spacer" /><span className="faint small">{fmtSize(f.size)}</span>
          </div>
        ))}
        {list && list.dirs.length === 0 && (props.mode === "dir" || list.files.length === 0) && (
          <div className="fs-item faint">Empty</div>
        )}
      </div>
      <p className="help" style={{ marginTop: 8 }}>
        {props.mode === "dir" ? "Click to select, double-click to open." : "Double-click a folder to open it, a file to choose it."}
      </p>
    </Modal>
  );
}

export function PathInput(props: {
  value: string; onChange: (v: string) => void; mode?: "file" | "dir"; exts?: string; placeholder?: string;
  validation?: { level: string; message: string };
  extra?: React.ReactNode; // after Browse, e.g. a Download button
}) {
  const [open, setOpen] = useState(false);
  const v = props.validation;
  return (
    <div className="stack" style={{ gap: 5 }}>
      <div className="row">
        <input className="mono" value={props.value} placeholder={props.placeholder} onChange={(e) => props.onChange(e.target.value)} />
        <button type="button" className="btn" onClick={() => setOpen(true)}>Browse</button>
        {props.extra}
      </div>
      {v && (
        <div className={"validation " + v.level}>
          {v.level === "ok" ? <IconCheck width={13} height={13} /> : <IconAlert width={13} height={13} />} {v.message}
        </div>
      )}
      {open && (
        <FileBrowser
          start={props.value ? (props.mode === "dir" ? props.value : props.value.replace(/[\\/][^\\/]*$/, "") || "/") : "~"}
          mode={props.mode || "file"}
          exts={props.exts}
          onClose={() => setOpen(false)}
          onPick={(p) => { props.onChange(p); setOpen(false); }}
        />
      )}
    </div>
  );
}

/** Comma-separated image paths with a Browse button that appends one. */
export function ImagesInput(props: { value: string; onChange: (v: string) => void; multiple: boolean; placeholder: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="row">
      <input className="mono" value={props.value} placeholder={props.placeholder} onChange={(e) => props.onChange(e.target.value)} />
      <button type="button" className="btn" onClick={() => setOpen(true)}>Browse</button>
      {open && (
        <FileBrowser start="~" mode="file" exts=".png,.jpg,.jpeg,.webp" onClose={() => setOpen(false)}
          onPick={(p) => { props.onChange(props.multiple && props.value.trim() ? `${props.value.trim()}, ${p}` : p); setOpen(false); }} />
      )}
    </div>
  );
}

const TIP_WIDTH = 280;

/** An "i" beside a setting's name that explains it, on hover, keyboard focus or a tap (phones can't hover). The
 * explanation is placed on the page itself, above the "i" (below it near the top), so no card or scroll area
 * cuts it off, and it follows the "i" when the page scrolls (focusing an "i" off screen scrolls to it). A click
 * on it doesn't reach the label it's in. */
export function Info({ text }: { text: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  const [at, setAt] = useState<{ left: number; top: number; width: number; below: boolean } | null>(null);
  useEffect(() => {
    if (!open) return;
    const place = () => {
      const r = ref.current?.getBoundingClientRect();
      if (!r) return;
      const width = Math.min(TIP_WIDTH, window.innerWidth - 16);
      const left = Math.min(Math.max(8, r.left + r.width / 2 - width / 2), window.innerWidth - width - 8);
      const below = r.top < 160;
      setAt({ left, top: below ? r.bottom + 8 : r.top - 8, width, below });
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open]);
  const show = () => setOpen(true);
  const hide = () => setOpen(false);
  return (
    <span ref={ref} className="info-i" tabIndex={0} role="button" aria-label={text}
      onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide} onKeyDown={(e) => e.key === "Escape" && hide()}
      onPointerUp={(e) => e.pointerType !== "mouse" && show()} // a tap; tapping elsewhere blurs it
      onClick={(e) => { e.preventDefault(); e.stopPropagation(); show(); }}>
      i
      {open && at && createPortal(
        <div className={"info-tip" + (at.below ? " below" : "")} role="tooltip" style={{ left: at.left, top: at.top, width: at.width }}>{text}</div>,
        document.body)}
    </span>
  );
}

export function Num(props: { label: string; v: number; on: (n: number) => void; step?: number; min?: number; max?: number; help?: string; info?: string }) {
  return (
    <label className="field">
      <span>{props.label}{props.info && <Info text={props.info} />}</span>
      <input type="number" value={props.v} step={props.step ?? 1} min={props.min} max={props.max} onChange={(e) => props.on(e.target.value === "" ? 0 : Number(e.target.value))} />
      {props.help && <span className="help">{props.help}</span>}
    </label>
  );
}

export function StatusPill({ status }: { status: string }) {
  return <span className={"pill " + status}><span className="dot" />{status.charAt(0).toUpperCase() + status.slice(1)}</span>;
}

/** A song: its spectrogram (``poster``) above a player, and a link to its score (ABC) when it has one. */
export function AudioCard({ src, poster, score, autoPlay, title }: {
  src: string; poster?: string; score?: string; autoPlay?: boolean; title?: React.ReactNode;
}) {
  return (
    <div className="audio-card" onClick={(e) => e.stopPropagation()}>
      {title && <div className="audio-title">{title}</div>}
      {poster && <img src={poster} alt="Spectrogram" loading="lazy" />}
      <audio src={src} controls autoPlay={autoPlay} preload={autoPlay ? "auto" : "none"} />
      {score && <a className="small" href={score} target="_blank" rel="noreferrer">Score (ABC notation)</a>}
    </div>
  );
}

/** A song's player: play, where it is of how long, seeking, mute and (with ``download``, the file's name) download.
 * Starting one pauses any other playing on the page. */
export function AudioPlayer({ src, download }: { src: string; download?: string }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [at, setAt] = useState(0);
  const [length, setLength] = useState(0);
  const [muted, setMuted] = useState(false);
  const toggle = () => {
    const a = ref.current;
    if (!a) return;
    if (a.paused) {
      document.querySelectorAll("audio").forEach((o) => o !== a && o.pause());
      a.play().catch(() => setPlaying(false));
    } else a.pause();
  };
  const pct = length ? (100 * at) / length : 0;
  return (
    <div className="player" onClick={(e) => e.stopPropagation()}>
      <audio ref={ref} src={src} preload="metadata" muted={muted} onLoadedMetadata={(e) => setLength(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setAt(e.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
      <button type="button" className="player-play" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>{playing ? <IconPause /> : <IconPlayFill />}</button>
      <span className="player-time">{fmtClock(at)} / {fmtClock(length)}</span>
      <input type="range" className="player-seek" min={0} max={length || 1} step={0.1} value={at} aria-label="Position"
        style={{ "--p": `${pct}%` } as React.CSSProperties}
        onChange={(e) => { const t = Number(e.target.value); if (ref.current) ref.current.currentTime = t; setAt(t); }} />
      <button type="button" className="player-btn" onClick={() => setMuted(!muted)} aria-label={muted ? "Unmute" : "Mute"} title={muted ? "Unmute" : "Mute"}>
        {muted ? <IconMute /> : <IconVolume />}
      </button>
      {download && <a className="player-btn" href={src} download={download} aria-label="Download" title="Download"><IconDownload /></a>}
    </div>
  );
}

// the aspect ratios worth naming, to say which one an image or video is (or is near)
const ASPECTS = [[1, 1], [5, 4], [4, 3], [3, 2], [16, 9], [2, 1], [21, 9], [3, 1]].flatMap(([a, b]) => a === b ? [[a, b]] : [[a, b], [b, a]]);

function aspect(w: number, h: number): string | null {
  for (const [a, b] of ASPECTS) {
    if (a * h === b * w) return `${a}:${b}`;
    if (Math.abs((w / h) / (a / b) - 1) < 0.015) return `≈ ${a}:${b}`;
  }
  return null;
}

/** Full-screen view of an image, a video (``video``) or a song (``audio``: its poster and score). With ``info`` (a
 * dataset file's), a panel at the bottom shows its name, its size in pixels and its length (as the browser reads
 * them), ``details`` and its caption, which ``info.onCaption`` makes editable. ``nav`` adds previous and next (the
 * arrow keys too) through a list of files, ``at`` of ``count``. An edited caption is saved before moving on or
 * closing; if that fails, the view stays. */
export function Lightbox({ src, video, audio, info, nav, onClose }: {
  src: string; video?: boolean; audio?: { poster?: string; score?: string; title?: React.ReactNode };
  info?: { name: string; caption: string; details?: React.ReactNode[]; onCaption?: (caption: string) => Promise<void> };
  nav?: { at: number; count: number; go: (step: -1 | 1) => void };
  onClose: () => void;
}) {
  const toast = useToast();
  const [shape, setShape] = useState<{ src: string; facts: string[] } | null>(null);
  const [draft, setDraft] = useState(info?.caption ?? "");
  useEffect(() => setDraft(info?.caption ?? ""), [src, info?.caption]);
  const dirty = !!info?.onCaption && draft.trim() !== info.caption;
  const busy = useRef(false);
  // save an edited caption, then do ``then`` (move or close)
  const leave = async (then?: () => void) => {
    if (busy.current) return;
    busy.current = true;
    try {
      if (dirty) await info!.onCaption!(draft);
      then?.();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      busy.current = false;
    }
  };
  const step = (s: -1 | 1) => {
    if (nav && nav.at + s >= 0 && nav.at + s < nav.count) leave(() => nav.go(s));
  };
  const keys = useRef<(e: KeyboardEvent) => void>(() => {});
  keys.current = (e) => {
    if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return; // typing a caption
    if (e.key === "Escape") leave(onClose);
    else if (nav && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault(); // not a seek in a focused video
      step(e.key === "ArrowLeft" ? -1 : 1);
    }
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const measure = (w: number, h: number, sec?: number) => {
    if (!w || !h) return;
    const r = aspect(w, h);
    setShape({ src, facts: [`${w} × ${h} px`, ...(r ? [r] : []),
      sec != null && isFinite(sec) ? fmtClock(sec) : `${((w * h) / 1e6).toFixed(2)} MP`] });
  };
  const media = audio ? <AudioCard src={src} autoPlay {...audio} />
    : video ? <video src={src} controls autoPlay loop onClick={(e) => e.stopPropagation()}
        onLoadedMetadata={(e) => measure(e.currentTarget.videoWidth, e.currentTarget.videoHeight, e.currentTarget.duration)} />
    : <img src={src} alt="" onLoad={(e) => measure(e.currentTarget.naturalWidth, e.currentTarget.naturalHeight)} />;
  if (!info) return <div className="lightbox" onClick={onClose}>{media}</div>;
  const facts = [...(shape?.src === src ? shape.facts : []), ...(info.details || [])];
  const arrow = (s: -1 | 1) => nav && (
    <button className={"lightbox-nav " + (s < 0 ? "prev" : "next")} title={s < 0 ? "Previous (←)" : "Next (→)"}
      disabled={nav.at + s < 0 || nav.at + s >= nav.count} onClick={(e) => { e.stopPropagation(); step(s); }}>
      {s < 0 ? <IconLeft /> : <IconRight />}
    </button>
  );
  return (
    <div className="lightbox with-info" onClick={() => leave(onClose)}>
      <div className="lightbox-media">{arrow(-1)}{media}{arrow(1)}</div>
      <div className="lightbox-info" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-facts">
          {nav && <span className="lightbox-pos">{nav.at + 1} / {nav.count}</span>}
          <strong className="ellipsis" title={info.name}>{info.name}</strong>
          {facts.map((f, i) => <span key={i}>{f}</span>)}
        </div>
        {info.onCaption ? (
          <div className="lightbox-edit">
            <textarea value={draft} rows={3} placeholder="No caption: write one" onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); leave(); }
                if (e.key === "Escape") e.currentTarget.blur(); // then Escape again closes
              }} />
            <div className="row small">
              <span className={"faint" + (dirty ? "" : " key-hint")}>{dirty ? "Edited: saved when you move on, or Ctrl+Enter" : nav ? "← → for the other files" : ""}</span>
              <span className="spacer" />
              {dirty && <button className="btn sm" onClick={() => setDraft(info.caption)}>Undo</button>}
              <button className="btn sm primary" disabled={!dirty} onClick={() => leave()}>Save</button>
            </div>
          </div>
        ) : (
          <div className="lightbox-caption">{info.caption || <span className="faint">No caption</span>}</div>
        )}
      </div>
    </div>
  );
}

export function ConfirmDialog(props: {
  title: string; message: React.ReactNode; confirmLabel: string; danger?: boolean;
  checkbox?: string; onConfirm: (checked: boolean) => void; onClose: () => void;
}) {
  const [checked, setChecked] = useState(false);
  return (
    <Modal title={props.title} onClose={props.onClose} footer={
      <>
        {/* in focus to start with: Cancel where confirming deletes something, so a stray Enter doesn't */}
        <button className="btn ghost" autoFocus={props.danger} onClick={props.onClose}>Cancel</button>
        <button className={"btn " + (props.danger ? "danger" : "primary")} autoFocus={!props.danger} onClick={() => { props.onConfirm(checked); props.onClose(); }}>{props.confirmLabel}</button>
      </>
    }>
      <div className="stack">
        <div className="muted">{props.message}</div>
        {props.checkbox && <label className="check"><input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />{props.checkbox}</label>}
      </div>
    </Modal>
  );
}

/** A list with nothing in it yet: an icon, what's missing, what to do about it, and the button that does it. */
export function Empty({ icon, title, children, action }: { icon: React.ReactNode; title: string; children?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="empty empty-lg">
      <div className="empty-icon" aria-hidden="true">{icon}</div>
      <strong>{title}</strong>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

/** Picking several cards at once: Select starts it; Done (or Esc) ends it and forgets what was picked. Esc in a dialog
 * or an enlarged file closes that, not this. */
export function useSelection() {
  const [on, setOn] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const stop = () => { setOn(false); setPicked(new Set()); };
  const toggle = (id: string) => setPicked((s) => {
    const next = new Set(s);
    if (!next.delete(id)) next.add(id);
    return next;
  });
  useEffect(() => {
    if (!on) return;
    const key = (e: KeyboardEvent) => { if (e.key === "Escape" && !document.querySelector(".modal-bg, .lightbox")) stop(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [on]);
  return { on, start: () => setOn(true), stop, picked, toggle, set: (ids: string[]) => setPicked(new Set(ids)) };
}
export type Selection = ReturnType<typeof useSelection>;

/** The bar while picking: how many are picked (of ``ids``, those that can be), all or none, and what to do. */
export function SelectBar({ sel, ids, noun, info, children }: { sel: Selection; ids: string[]; noun: [string, string]; info?: React.ReactNode; children: React.ReactNode }) {
  const n = ids.filter((id) => sel.picked.has(id)).length;
  const all = ids.length > 0 && n === ids.length;
  return (
    <div className="select-bar">
      <label className="check" title={all ? "Select none" : "Select all"}>
        <input type="checkbox" checked={all} ref={(el) => { if (el) el.indeterminate = n > 0 && !all; }} onChange={() => sel.set(all ? [] : ids)} disabled={!ids.length} />
        <span>{n ? `${n} ${n === 1 ? noun[0] : noun[1]} selected` : `Select ${noun[1]}`}</span>
      </label>
      {n > 0 && info}
      <span className="spacer" />
      {children}
      <button className="btn ghost" onClick={sel.stop}>Done</button>
    </div>
  );
}

export type MenuItem = { label: string; icon?: React.ReactNode; onClick: () => void; danger?: boolean; disabled?: boolean; title?: string };

/** A "⋯" button and the actions it opens: Up and Down move between them, Escape or a click elsewhere closes it. Clicks
 * in it don't reach what it's in (a card that opens on a click). */
export function Menu({ label, items, boxed }: { label: string; items: MenuItem[]; boxed?: boolean }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const options = () => [...(box.current?.querySelectorAll<HTMLButtonElement>(".menu-list button:not(:disabled)") || [])];
  useEffect(() => {
    if (!open) return;
    options()[0]?.focus();
    const away = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);
  const onKey = (e: React.KeyboardEvent) => {
    if (!open) return;
    if (e.key === "Escape" || e.key === "Tab") {
      if (e.key === "Escape") e.preventDefault();
      e.stopPropagation();
      setOpen(false);
      button.current?.focus();
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const all = options();
      const i = all.indexOf(document.activeElement as HTMLButtonElement);
      all[(i + (e.key === "ArrowDown" ? 1 : all.length - 1)) % all.length]?.focus();
    }
  };
  return (
    <div className="menu" ref={box} onKeyDown={onKey} onClick={(e) => e.stopPropagation()}>
      <button ref={button} className={(boxed ? "btn icon lg menu-btn boxed" : "btn ghost icon menu-btn") + (open ? " on" : "")} aria-label={label} title={label} aria-haspopup="menu" aria-expanded={open}
        onClick={() => setOpen(!open)}><IconMore /></button>
      {open && (
        <div className="menu-list" role="menu">
          {items.map((it) => (
            <button key={it.label} role="menuitem" className={it.danger ? "danger" : ""} disabled={it.disabled} title={it.title}
              onClick={() => { button.current?.focus(); setOpen(false); it.onClick(); }}>{it.icon}{it.label}</button>
          ))}
        </div>
      )}
    </div>
  );
}

/** "a, b and 3 more" */
/** A family's name on a badge: "MiniMax H3 (video + audio)" reads "MiniMax H3 · Video + audio". */
export function familyBadge(label: string) {
  return label.replace(/\s*\((.+)\)$/, (_, s: string) => ` · ${s[0].toUpperCase()}${s.slice(1)}`);
}

export function someNames(names: string[], max = 4): string {
  if (names.length <= max) return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : names[0] || "";
  return `${names.slice(0, max).join(", ")} and ${names.length - max} more`;
}

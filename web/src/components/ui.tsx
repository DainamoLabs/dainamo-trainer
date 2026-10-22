import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, fmtClock, q } from "../api";
import { IconCheck, IconFile, IconFolder, IconAlert, IconUp, IconX } from "./icons";

// ---- modal ---------------------------------------------------------------
export function Modal(props: { title: React.ReactNode; onClose: () => void; children: React.ReactNode; footer?: React.ReactNode; wide?: boolean }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && props.onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props.onClose]);
  return (
    <div className="modal-bg" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div className={"modal" + (props.wide ? " wide" : "")}>
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
export function usePoll<T>(url: string | null, ms: number, deps: unknown[] = []): [T | null, () => void, string | null] {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const tick = useRef(0);
  const load = useCallback(() => {
    if (!url) return;
    const my = ++tick.current;
    api.get<T>(url).then(
      (d) => { if (my === tick.current) { setData(d); setErr(null); } },
      (e) => my === tick.current && setErr(String(e.message || e)),
    );
  }, [url]);
  useEffect(() => {
    load();
    if (!ms) return;
    const t = setInterval(load, ms);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, ms, ...deps]);
  return [data, load, err];
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
}) {
  const [open, setOpen] = useState(false);
  const v = props.validation;
  return (
    <div className="stack" style={{ gap: 5 }}>
      <div className="row">
        <input className="mono" value={props.value} placeholder={props.placeholder} onChange={(e) => props.onChange(e.target.value)} />
        <button type="button" className="btn" onClick={() => setOpen(true)}>Browse</button>
      </div>
      {v && (
        <div className={"validation " + v.level}>
          {v.level === "ok" ? <IconCheck width={13} height={13} /> : <IconAlert width={13} height={13} />} {v.message}
        </div>
      )}
      {open && (
        <FileBrowser
          start={props.value ? props.value.replace(/[\\/][^\\/]*$/, "") || "/" : "~"}
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

export function Num(props: { label: string; v: number; on: (n: number) => void; step?: number; min?: number; help?: string; info?: string }) {
  return (
    <label className="field">
      <span>{props.label}{props.info && <Info text={props.info} />}</span>
      <input type="number" value={props.v} step={props.step ?? 1} min={props.min} onChange={(e) => props.on(e.target.value === "" ? 0 : Number(e.target.value))} />
      {props.help && <span className="help">{props.help}</span>}
    </label>
  );
}

export function StatusPill({ status }: { status: string }) {
  return <span className={"pill " + status}><span className="dot" />{status}</span>;
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
 * them), ``details`` and its caption. */
export function Lightbox({ src, video, audio, info, onClose }: {
  src: string; video?: boolean; audio?: { poster?: string; score?: string; title?: React.ReactNode };
  info?: { name: string; caption: string; details?: React.ReactNode[] }; onClose: () => void;
}) {
  const [shape, setShape] = useState<{ src: string; facts: string[] } | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
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
  return (
    <div className="lightbox with-info" onClick={onClose}>
      <div className="lightbox-media">{media}</div>
      <div className="lightbox-info" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-facts">
          <strong className="ellipsis" title={info.name}>{info.name}</strong>
          {facts.map((f, i) => <span key={i}>{f}</span>)}
        </div>
        <div className="lightbox-caption">{info.caption || <span className="faint">No caption</span>}</div>
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
        <button className="btn ghost" onClick={props.onClose}>Cancel</button>
        <button className={"btn " + (props.danger ? "danger" : "primary")} onClick={() => { props.onConfirm(checked); props.onClose(); }}>{props.confirmLabel}</button>
      </>
    }>
      <div className="stack">
        <div className="muted">{props.message}</div>
        {props.checkbox && <label className="check"><input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />{props.checkbox}</label>}
      </div>
    </Modal>
  );
}

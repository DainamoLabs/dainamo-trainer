import { useEffect, useRef, useState } from "react";
import {
  api, q, type Dataset, type DatasetUpscale, type SandboxOutput, type SandboxRun, type UpscaleParams, type UpscalePlan, type Upscaler,
  type Upscalers,
} from "../api";
import { IconFilm, IconImages, IconUpload, IconUpscale, IconX } from "./icons";
import { Info, Modal, PathInput, usePoll, useToast } from "./ui";

/** What to upscale: a Sandbox run's output, or a file on this machine (``upload``), with its size when known. */
export type UpscaleTarget = {
  source?: { run_id: string; file: string }; upload?: string;
  name: string; kind: "image" | "video"; width?: number; height?: number; preview?: string | null;
};

const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
const VIDEO_EXTS = [".mp4", ".webm", ".mov", ".mkv", ".avi"];
export const UPSCALE_EXTS = [...IMAGE_EXTS, ...VIDEO_EXTS].join(",");
const FACTORS = [2, 3, 4];
const MAX_FACTORS = [2, 3, 4, 6, 8];
const MAX_SIDE = { image: 8192, video: 4096 }; // the server's limits on an upscaled file's longest side
const MODEL_KEY = "dainamo.upscale.model"; // the upscaler picked last
const FACTOR_KEY = "dainamo.upscale.factor";

export const UPSCALE_INFO = {
  model: "Upscalers are files in your models folder's upscale_models, as in ComfyUI (Model files downloads them). "
    + "A fixed-scale model such as Real-ESRGAN makes its own size (4× for x4plus), and the result is then fitted to the factor chosen here. "
    + "SeedVR2 makes any size directly, and takes a clip whole, so its frames stay consistent with each other.",
  resolution: "The size you train at. Images with fewer pixels than this squared are upscaled to about that many; larger ones are left as they are.",
  maxFactor: "No image is made more than this many times bigger. One that would need more stays below the resolution.",
};

const kindOf = (p: string): "image" | "video" | null => {
  const ext = p.slice(p.lastIndexOf(".")).toLowerCase();
  return IMAGE_EXTS.includes(ext) ? "image" : VIDEO_EXTS.includes(ext) ? "video" : null;
};
const base = (p: string) => p.split(/[\\/]/).pop() || p;
const remembered = (key: string) => { try { return localStorage.getItem(key) || ""; } catch { return ""; } };
const remember = (key: string, v: string) => { try { localStorage.setItem(key, v); } catch { /* not kept */ } };

/** An upscaler's name and what it does: "RealESRGAN_x4plus · 4×", "seedvr2_3b_fp16 · video, any size". */
export function upscalerLabel(m: Upscaler) {
  if (m.kind === "seedvr2") return `${m.name} · video, any size`;
  if (m.kind === "pixel") return `${m.name} · ${m.scale}×`;
  return m.name;
}
const usable = (m: Upscaler) => m.kind !== "unknown" && !m.error;

/** The upscaler to start on: the one picked last, else SeedVR2 for a clip and a pixel upscaler for an image. */
function pickModel(models: Upscaler[], kind: "image" | "video", wanted?: string) {
  const ok = models.filter(usable);
  return ok.find((m) => m.path === wanted) || ok.find((m) => m.path === remembered(MODEL_KEY))
    || ok.find((m) => m.kind === (kind === "video" ? "seedvr2" : "pixel")) || ok[0];
}

/** The upscalers menu: the usable ones, then the others, greyed with the reason. */
function UpscalerSelect({ models, value, onChange }: { models: Upscaler[]; value: string; onChange: (path: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {models.filter(usable).map((m) => <option key={m.path} value={m.path}>{upscalerLabel(m)}</option>)}
      {models.filter((m) => !usable(m)).map((m) => (
        <option key={m.path} value={m.path} disabled>{m.name} · {m.error || "not an upscaler Dainamo reads"}</option>
      ))}
    </select>
  );
}

function NoUpscaler({ list }: { list: Upscalers }) {
  return (
    <div className="validation warning" style={{ display: "block", lineHeight: 1.5 }}>
      {list.folder
        ? <>No upscaler in <span className="mono">{list.folder}</span> yet. Download Real-ESRGAN or SeedVR2 in <a href="#/profiles/files">Model files</a>, or put one there.</>
        : <>Set your ComfyUI models folder in <a href="#/profiles/files">Model files</a> first: upscalers are read from its upscale_models folder.</>}
      {list.error && <div>{list.error}</div>}
    </div>
  );
}

/** Upscale a run's output, or a file, as a Sandbox run: the upscaler and how many times bigger. */
export function UpscaleDialog({ target, gpu, onClose, onQueued }: {
  target: UpscaleTarget; gpu: number | null; onClose: () => void; onQueued: (run: SandboxRun) => void;
}) {
  const toast = useToast();
  const [list] = usePoll<Upscalers>("/api/upscalers", 0);
  const models = list?.models || [];
  const [picked, setPicked] = useState("");
  const cur = pickModel(models, target.kind, picked);
  const [factor, setFactor] = useState(() => FACTORS.includes(Number(remembered(FACTOR_KEY))) ? Number(remembered(FACTOR_KEY)) : 2);
  const [size, setSize] = useState(target.width && target.height ? { width: target.width, height: target.height } : null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (size || target.kind !== "image" || !target.upload) return;
    api.get<{ width: number; height: number }>(`/api/sandbox/image-size?path=${q(target.upload)}`).then(setSize, () => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const out = size ? [Math.round(size.width * factor), Math.round(size.height * factor)] : null;
  const tooBig = !!out && Math.max(...out) > MAX_SIDE[target.kind];

  const go = async () => {
    if (!cur) return;
    setBusy(true);
    try {
      const run = await api.post<SandboxRun>("/api/sandbox/upscale", {
        model: cur.path, factor, gpu, ...(target.source ? { source: target.source } : { upload: target.upload }),
      });
      remember(MODEL_KEY, cur.path);
      remember(FACTOR_KEY, String(factor));
      onQueued(run);
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title={`Upscale ${target.kind === "video" ? "a video" : "an image"}`} onClose={onClose} footer={<>
      <button className="btn ghost" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={!cur || busy || tooBig} onClick={go}><IconUpscale /> {busy ? "Starting…" : "Upscale"}</button>
    </>}>
      {!list ? <div className="muted">Looking for upscalers…</div> : (
        <div className="stack">
          <div className="up-target">
            {target.preview ? <img src={target.preview} alt="" /> : <span className="up-thumb">{target.kind === "video" ? <IconFilm /> : <IconImages />}</span>}
            <div style={{ minWidth: 0 }}>
              <div className="ellipsis" style={{ fontWeight: 600 }} title={target.upload || target.name}>{target.name}</div>
              <div className="muted small">{size ? `${size.width}×${size.height}` : target.kind === "video" ? "Video" : "Image"}</div>
            </div>
          </div>
          {!models.some(usable) ? <NoUpscaler list={list} /> : <>
            <label className="field">
              <span>Upscaler<Info text={UPSCALE_INFO.model} /></span>
              <UpscalerSelect models={models} value={cur?.path || ""} onChange={setPicked} />
            </label>
            <div className="field">
              <span>Factor</span>
              <div className="seg" role="group" aria-label="Factor" style={{ alignSelf: "flex-start" }}>
                {FACTORS.map((f) => <button key={f} type="button" className={factor === f ? "on" : ""} aria-pressed={factor === f} onClick={() => setFactor(f)}>{f}×</button>)}
              </div>
              {out && <span className={tooBig ? "validation error" : "help"}>
                {size!.width}×{size!.height} → {out[0]}×{out[1]}{tooBig ? `: a side can be at most ${MAX_SIDE[target.kind]} pixels` : ""}
              </span>}
            </div>
            <p className="help" style={{ margin: 0 }}>
              {cur?.kind === "seedvr2"
                ? `SeedVR2 makes the size asked for directly${target.kind === "video" ? ", and upscales the clip whole, so its frames stay consistent" : ""}. It's slower than a pixel upscaler and needs more VRAM.`
                : cur ? `${cur.name} makes ${cur.scale}× the size, and the result is fitted to ${factor}×.${target.kind === "video"
                  ? " It works frame by frame, so fine detail can flicker; SeedVR2 keeps a clip's frames consistent." : ""}` : ""}
              {" "}It runs in the Sandbox after the generations before it, beside the model it holds, and shows in History.
            </p>
          </>}
        </div>
      )}
    </Modal>
  );
}

/** The Sandbox's "Upscale a file": an image or a video on this machine, or one uploaded from this browser. */
export function UpscaleFileCard({ onUpscale }: { onUpscale: (t: UpscaleTarget) => void }) {
  const toast = useToast();
  const [path, setPath] = useState("");
  const [sent, setSent] = useState<{ path: string; url: string } | null>(null); // the last file uploaded
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const upload = async (file: File) => {
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await api.post<{ path: string; url: string }>("/api/sandbox/upload", fd);
      setSent(r);
      setPath(r.path);
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };
  const p = path.trim();
  const kind = kindOf(p);
  return (
    <div className="card card-pad stack" style={{ gap: 10 }}>
      <div className="card-head" style={{ marginBottom: 0 }}><IconUpscale width={16} style={{ color: "var(--accent)" }} /><h3>Upscale a file</h3></div>
      <p className="help" style={{ margin: 0 }}>An image or a video from this machine, or uploaded from this browser, made bigger with an upscaler from Model files.</p>
      <PathInput value={path} onChange={setPath} exts={UPSCALE_EXTS} placeholder="/path/to/image.png or clip.mp4"
        extra={<button type="button" className="btn" disabled={uploading} onClick={() => fileRef.current?.click()}><IconUpload /> {uploading ? "Uploading…" : "Upload"}</button>} />
      <input ref={fileRef} type="file" hidden accept="image/*,video/*" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
      {p && !kind && <span className="validation error">Choose an image or a video ({UPSCALE_EXTS.replace(/,/g, ", ")})</span>}
      <div className="row">
        <button className="btn primary" disabled={!kind} onClick={() => onUpscale({
          upload: p, name: base(p), kind: kind!, preview: sent?.path === p && kind === "image" ? sent.url : null,
        })}><IconUpscale /> Upscale…</button>
      </div>
    </div>
  );
}

const fileUrl = (run: SandboxRun, name: string) => `/api/sandbox/file/${run.id}/${encodeURIComponent(name)}`;
export const upscaleParams = (run: SandboxRun): UpscaleParams | null => (run.workflow === "upscale" ? run.params as unknown as UpscaleParams : null);

/** An upscale run's one-line summary: "RealESRGAN_x4plus · 2× · Run 12's output". */
export function upscaleMeta(p: UpscaleParams) {
  return [p.model_name, `${p.factor}×`, p.source ? `Run ${p.source.n}'s output` : p.input.name].join(" · ");
}

/** An upscale run: what it was made from beside the result, each with its size. */
export function UpscaleRunView({ run, onOpen, onUpscale }: {
  run: SandboxRun; onOpen: (src: string, video: boolean) => void; onUpscale: (o: SandboxOutput) => void;
}) {
  const p = upscaleParams(run)!;
  const inp = p.input;
  const [srcOk, setSrcOk] = useState(true); // an uploaded file is gone once the server restarts
  const media = (src: string, video: boolean, poster?: string | null, onError?: () => void) => video
    ? <video src={src} poster={poster || undefined} controls loop preload="metadata" onError={onError} />
    : <img src={src} alt="" onClick={() => onOpen(src, false)} onError={onError} />;
  const outputs = run.outputs || [];
  return (
    <div className="stack" style={{ gap: 14 }}>
      {(outputs.length ? outputs : [null]).map((o, i) => (
        <div key={o?.file || i} className={"sbx-pair" + (inp.url && srcOk ? " two" : "")}>
          {inp.url && srcOk && (
            <div className="sbx-out">
              <div className="sbx-out-head"><strong>Before</strong><span className="faint ellipsis" title={inp.path}>{inp.name}</span></div>
              {media(inp.url, inp.kind === "video", inp.poster, () => setSrcOk(false))}
              <div className="sbx-out-meta">{inp.width}×{inp.height}{inp.frames ? ` · ${inp.frames} frames` : ""}</div>
            </div>
          )}
          <div className="sbx-out">
            <div className="sbx-out-head">
              <strong>Upscaled</strong>
              {o && (o.kind === "image" || o.kind === "video") && (
                <button className="btn ghost sm sbx-out-act" onClick={() => onUpscale(o)} title="Upscale this again"><IconUpscale /> Upscale</button>
              )}
            </div>
            {!o ? <div className="sbx-wait">{run.status === "queued" || run.status === "running" ? "Upscaling…" : "Nothing was made"}</div>
              : media(fileUrl(run, o.file), o.kind === "video", o.poster ? fileUrl(run, o.poster) : null)}
            <div className="sbx-out-meta">
              {o?.width ? `${o.width}×${o.height}` : `${p.width}×${p.height}`} · {p.factor}× with {p.model_name}
              {!inp.url && <div className="faint" style={{ overflowWrap: "anywhere" }}>From {inp.path}</div>}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/** Upscale small images: a dataset's images under the training resolution upscaled to it, in the Sandbox's worker,
 *  each original kept in an _originals folder beside it. */
export function DatasetUpscaleModal({ dataset, last, onClose, onStarted }: {
  dataset: Dataset; last: DatasetUpscale["last"]; onClose: () => void; onStarted: () => void;
}) {
  const toast = useToast();
  const [list] = usePoll<Upscalers>("/api/upscalers", 0);
  const models = list?.models || [];
  const [picked, setPicked] = useState("");
  const cur = pickModel(models, "image", picked || last?.model);
  const [resolution, setResolution] = useState(last?.resolution ?? 1024);
  const [maxFactor, setMaxFactor] = useState(last?.max_factor ?? 4);
  const resOk = Number.isInteger(resolution) && resolution >= 128 && resolution <= 4096;
  const [plan] = usePoll<UpscalePlan>(resOk ? `/api/datasets/${dataset.id}/upscale/plan?resolution=${resolution}&max_factor=${maxFactor}` : null, 0, [], true);
  const [busy, setBusy] = useState(false);
  const start = async () => {
    if (!cur) return;
    setBusy(true);
    try {
      await api.post(`/api/datasets/${dataset.id}/upscale`, { model: cur.path, resolution, max_factor: maxFactor });
      remember(MODEL_KEY, cur.path);
      onStarted();
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };
  const n = plan?.small ?? 0;
  return (
    <Modal title="Upscale small images" onClose={onClose} footer={<>
      <button className="btn ghost" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={!cur || !resOk || !n || busy} onClick={start}>
        <IconUpscale /> {busy ? "Starting…" : n ? `Upscale ${n} image${n === 1 ? "" : "s"}` : "Upscale"}
      </button>
    </>}>
      {!list ? <div className="muted">Looking for upscalers…</div> : (
        <div className="stack">
          <p className="help" style={{ margin: 0 }}>Training never makes an image bigger, so one smaller than the resolution trains smaller.
            This upscales the images under the resolution to about its size (each side by at most the largest factor), saves each as a PNG
            of the same name, and leaves the captions as they are. Clips aren't upscaled here: use Upscale a file in the Sandbox.</p>
          {!models.some(usable) ? <NoUpscaler list={list} /> : (
            <label className="field">
              <span>Upscaler<Info text={UPSCALE_INFO.model} /></span>
              <UpscalerSelect models={models} value={cur?.path || ""} onChange={setPicked} />
            </label>
          )}
          <div className="grid cols-2">
            <label className="field">
              <span>Resolution<Info text={UPSCALE_INFO.resolution} /></span>
              <input type="number" value={Number.isFinite(resolution) ? resolution : ""} min={128} max={4096} step={64}
                onChange={(e) => setResolution(e.target.value === "" ? NaN : Number(e.target.value))} />
            </label>
            <label className="field">
              <span>Largest factor<Info text={UPSCALE_INFO.maxFactor} /></span>
              <select value={maxFactor} onChange={(e) => setMaxFactor(Number(e.target.value))}>
                {[...new Set([...MAX_FACTORS, maxFactor])].sort((a, b) => a - b).map((f) => <option key={f} value={f}>{f}×</option>)}
              </select>
            </label>
          </div>
          {!resOk ? <span className="validation error">A resolution of 128–4096</span> : plan && (
            <div className={n ? "muted small" : "validation warning"} style={{ lineHeight: 1.5 }}>
              {n ? `${n} of the ${plan.images} images ${n === 1 ? "is" : "are"} under ${resolution}×${resolution} pixels and will be upscaled`
                : `None of the ${plan.images} images is under ${resolution}×${resolution} pixels`}
              {plan.smallest && n > 0 ? ` (the smallest is ${plan.smallest.width}×${plan.smallest.height})` : ""}.
              {plan.capped > 0 && ` ${plan.capped} need${plan.capped === 1 ? "s" : ""} more than ${maxFactor}×: ${plan.capped === 1 ? "it's" : "they're"} upscaled ${maxFactor}× and ${plan.capped === 1 ? "stays" : "stay"} smaller.`}
              {plan.clash.length > 0 && ` ${plan.clash.length} ${plan.clash.length === 1 ? "is" : "are"} skipped, as another file has the name the PNG would get (${plan.clash.slice(0, 3).join(", ")}${plan.clash.length > 3 ? "…" : ""}).`}
              {plan.unreadable > 0 && ` ${plan.unreadable} can't be read.`}
            </div>
          )}
          <div className={dataset.managed ? "muted small" : "validation warning"} style={{ display: "block", lineHeight: 1.5 }}>
            Each original is moved to an <span className="mono">_originals</span> folder beside it, which training skips, so you can put it back.
            {!dataset.managed && <> This dataset is a linked folder: the files are changed in <span className="mono">{dataset.path}</span> itself.</>}
          </div>
          {cur?.kind === "seedvr2" && <p className="help" style={{ margin: 0 }}>SeedVR2 takes longer per image than a pixel upscaler.</p>}
          <p className="help" style={{ margin: 0 }}>It runs in the Sandbox, after the generations queued before it, one dataset at a time.</p>
        </div>
      )}
    </Modal>
  );
}

/** A dataset's upscale on its page: progress with Cancel, then what was done. */
export function DatasetUpscaleTask({ up, onCancel, onDismiss }: { up: DatasetUpscale; onCancel: () => void; onDismiss: () => void }) {
  const active = up.state === "queued" || up.state === "running";
  const made = up.made;
  return (
    <div className="dd-task">
      <div className="row small" style={{ marginBottom: 6 }}>
        <IconUpscale width={14} style={{ color: "var(--accent)" }} />
        {up.state === "queued" ? <span>Upscaling waits for the Sandbox's generations before it</span>
          : up.state === "running" ? <span>Upscaling {up.done} / {up.total} images <span className="faint">{up.message}</span></span>
          : up.state === "done" ? <span>Upscaled {made?.upscaled ?? up.done} image{(made?.upscaled ?? up.done) === 1 ? "" : "s"}{made?.skipped ? `, ${made.skipped} skipped` : ""}
            <span className="faint"> · the originals are in _originals</span></span>
          : up.state === "cancelled" ? <span>Upscaling cancelled{up.done ? `; the ${up.done} image${up.done === 1 ? "" : "s"} already upscaled stay so` : ""}</span>
          : <span className="validation error">Upscaling failed: {up.message}</span>}
        <span className="spacer" />
        {active ? <button className="btn sm" onClick={onCancel}><IconX /> Cancel</button>
          : <button className="btn ghost icon sm" aria-label="Dismiss" title="Dismiss" onClick={onDismiss}><IconX /></button>}
      </div>
      {active && (up.state === "running" && up.total
        ? <div className="progress"><div style={{ width: `${(100 * up.done) / Math.max(1, up.total)}%` }} /></div>
        : <div className="progress busy"><div /></div>)}
      {!!made?.errors.length && <div className="error-box" style={{ marginTop: 8 }}>{made.errors.slice(-5).map((e) => <div key={e}>{e}</div>)}</div>}
    </div>
  );
}

import { useEffect, useRef, useState, type MutableRefObject, type RefObject } from "react";
import { api, fmtClock, q } from "../api";
import { IconLeft, IconPlay, IconPlus, IconRefresh, IconRight, IconScissors, IconStop, IconTrash } from "../components/icons";
import { ConfirmDialog, TaskProgress, useToast } from "../components/ui";

type Crop = [number, number, number, number]; // x, y, width, height in the video's pixels
type Clip = { id: number; start: number; frames: number; crop: Crop | null };
type PlanVideo = {
  rel: string; path: string | null; looked: boolean; stem: string | null; duration: number | null; fps: number | null; width: number | null;
  height: number | null; has_audio: boolean; cuts: number[]; clips: Clip[]; next: number; note: string;
  cut: Record<string, { start: number; frames: number; crop: Crop | null }>; // how each clip was cut, by number
};
type SplitJob = {
  running: boolean; mode: "cut" | "plan"; total: number; done: number; clips: number; removed: number; current: string;
  message: string; errors: string[]; cancelled: boolean;
};
type Plan = {
  id: string; name: string; source: { id: string; name: string }; family: string; family_label: string; fps: number;
  frames: number; edge: number; lengths: number[];
  settings: { max_clips: number; scenes: boolean; keep_short: boolean; max_side: number; captions: "copy" | "none" };
  videos: PlanVideo[]; changes: { cut: number; remove: number }; job: SplitJob | null;
};
type Saved = { video: PlanVideo; changes: Plan["changes"] };

const MIN_CROP = 64;
const ASPECTS: [string, number | null][] = [["Free", null], ["16:9", 16 / 9], ["9:16", 9 / 16], ["1:1", 1], ["4:3", 4 / 3], ["3:4", 3 / 4]];
const ZOOMS = [1, 2, 4, 8, 16, 32, 64];

const fmtT = (s: number) => {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
};
/** "1:04.5", "64.5" or "1:02:03" -> seconds. */
const parseT = (s: string): number | null => {
  const parts = s.trim().split(":");
  if (!parts.every((p) => /^\d*\.?\d+$/.test(p)) || parts.length > 3) return null;
  return parts.reduce((t, p) => t * 60 + Number(p), 0);
};
const r3 = (t: number) => Math.round(t * 1000) / 1000;
const ev = (n: number) => Math.round(n / 2) * 2;
const sameCrop = (a: Crop | null, b: Crop | null) => (a && b ? a.every((x, i) => x === b[i]) : !a && !b);

type ClipState = "cut" | "changed" | "new";
function stateOf(v: PlanVideo, c: Clip): ClipState {
  const was = v.cut[String(c.id)];
  if (!was) return "new";
  return Math.abs(was.start - c.start) < 5e-4 && was.frames === c.frames && sameCrop(was.crop, c.crop) ? "cut" : "changed";
}
/** Clips to cut (new or changed) and cut clips to remove. */
function pendingOf(v: PlanVideo): number {
  const ids = new Set(v.clips.map((c) => String(c.id)));
  return v.clips.filter((c) => stateOf(v, c) !== "cut").length + Object.keys(v.cut).filter((id) => !ids.has(id)).length;
}
/** The latest a clip of ``frames`` starts: a frame clear of the file's end, as Split into clips keeps it. */
const latest = (v: PlanVideo, frames: number, fps: number, edge: number) => Math.max(0, (v.duration || 0) - (frames + edge) / fps);
/** ``t`` on one of the video's own frames. */
const snapT = (t: number, v: PlanVideo) => r3(Math.round(t * (v.fps || 30)) / (v.fps || 30));
const place = (v: PlanVideo, t: number, frames: number, fps: number, edge: number) => r3(Math.min(Math.max(0, snapT(t, v)), latest(v, frames, fps, edge)));
const fitting = (v: PlanVideo, start: number, lengths: number[], fps: number, edge: number) =>
  lengths.filter((l) => start + (l + edge) / fps <= (v.duration || 0) + 1e-3);
const nearest = (ls: number[], want: number) => ls.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a), ls[0]);
/** Shot cuts a clip shows both sides of: a cut from its second frame to its last. */
const crossed = (v: PlanVideo, c: Clip, fps: number) => v.cuts.filter((t) => t > c.start + 1e-3 && t <= c.start + (c.frames - 1) / fps + 1e-3);
const overlapping = (v: PlanVideo, c: Clip, fps: number) =>
  v.clips.filter((o) => o.id !== c.id && o.start < c.start + c.frames / fps - 1e-3 && c.start < o.start + o.frames / fps - 1e-3);
const clipFile = (stem: string, c: Clip) => `${stem}_${String(c.id).padStart(3, "0")}.mp4`;
const aspectOf = (crop: Crop | null) => {
  if (!crop) return "";
  const r = crop[2] / crop[3];
  return ASPECTS.find(([, a]) => a && Math.abs(r / a - 1) < 0.02)?.[0] || "Free";
};
/** The largest crop of ``aspect`` (null: 80% of the frame) centered where ``around`` is, inside the frame. */
function cropOf(v: PlanVideo, aspect: number | null, around: Crop | null): Crop {
  const W = v.width || 0, H = v.height || 0;
  const [cx, cy] = around ? [around[0] + around[2] / 2, around[1] + around[3] / 2] : [W / 2, H / 2];
  let w = aspect ? Math.min(W, H * aspect) : W * 0.8, h = aspect ? w / aspect : H * 0.8;
  w = Math.min(W, ev(w)); h = Math.min(H, ev(h));
  const x = ev(Math.min(Math.max(0, cx - w / 2), W - w)), y = ev(Math.min(Math.max(0, cy - h / 2), H - h));
  return [Math.min(x, W - w), Math.min(y, H - h), w, h];
}
const pct = (c: Crop, v: PlanVideo) => ({
  left: `${(100 * c[0]) / (v.width || 1)}%`, top: `${(100 * c[1]) / (v.height || 1)}%`,
  width: `${(100 * c[2]) / (v.width || 1)}%`, height: `${(100 * c[3]) / (v.height || 1)}%`,
});
/** The frame outside a crop, darkened: four strips around it. */
function Shade({ crop, v }: { crop: Crop; v: PlanVideo }) {
  const W = v.width || 1, H = v.height || 1;
  const [x, y, w, h] = [crop[0] / W, crop[1] / H, crop[2] / W, crop[3] / H].map((f) => f * 100);
  return (
    <>
      <div className="ce-shade" style={{ left: 0, top: 0, width: "100%", height: `${y}%` }} />
      <div className="ce-shade" style={{ left: 0, top: `${y + h}%`, width: "100%", bottom: 0 }} />
      <div className="ce-shade" style={{ left: 0, top: `${y}%`, width: `${x}%`, height: `${h}%` }} />
      <div className="ce-shade" style={{ left: `${x + w}%`, top: `${y}%`, right: 0, height: `${h}%` }} />
    </>
  );
}
const frameUrl = (v: PlanVideo, start: number, n: number, fps: number, size = 480) =>
  `/api/clips/frame?path=${q(v.path || "")}&start=${start.toFixed(3)}&n=${n}&fps=${fps}&size=${size}`;

function useSettled<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [JSON.stringify(value)]);
  return v;
}

/** The clip editor: a split dataset's clips, video by video, moved, resized, cropped, added and removed, then cut
 *  again (only the clips that changed). Opened from the split dataset, or from Split into clips' Review first. */
export default function ClipEditor({ id }: { id: string }) {
  const toast = useToast();
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rel, setRel] = useState("");
  const [selId, setSelId] = useState<number | null>(null);
  const [newLen, setNewLen] = useState(0);
  // a video's edits not saved yet: rels with changes the server hasn't answered, the save waiting for each, and a
  // count per rel so an answer to an older save doesn't undo a newer edit
  const dirty = useRef(new Set<string>());
  const waiting = useRef(new Map<string, { timer: number; clips: Clip[]; n: number }>());
  const edits = useRef(new Map<string, number>());
  const inflight = useRef(new Set<Promise<unknown>>());
  const askedToLook = useRef(false);
  const [replanning, setReplanning] = useState<string | null>(null); // the video whose Plan again is being confirmed

  const load = () => api.get<Plan>(`/api/datasets/${id}/clips`).then((p) => {
    setError(null);
    setPlan((old) => !old ? p : { ...p, videos: p.videos.map((v) => {
      const o = old.videos.find((x) => x.rel === v.rel);
      return o && dirty.current.has(v.rel) ? { ...v, clips: o.clips, next: o.next } : v;
    }) });
  }, (e) => setError(e.message));
  useEffect(() => { load(); }, [id]);

  // the first video to show, the length new clips get, and videos the plan doesn't know yet looked at
  useEffect(() => {
    if (!plan) return;
    if (!rel && plan.videos.length) {
      const first = plan.videos.find((v) => v.path && v.looked) || plan.videos[0];
      setRel(first.rel);
      setSelId(first.clips[0]?.id ?? null);
    }
    if (!newLen && plan.lengths.length) setNewLen(plan.lengths.includes(plan.frames) ? plan.frames : plan.lengths[0]);
    if (!askedToLook.current && !plan.job?.running && plan.videos.some((v) => v.path && !v.looked)) {
      askedToLook.current = true;
      api.post<SplitJob>(`/api/datasets/${id}/clips/look`).then((job) => setPlan((p) => p && { ...p, job }), (e) => toast(e.message, true));
    }
  }, [plan]);

  const running = !!plan?.job?.running;
  const wasRunning = useRef<SplitJob | null>(null);
  useEffect(() => {
    if (!running) return;
    const t = setInterval(load, 1000);
    return () => clearInterval(t);
  }, [running]);
  useEffect(() => {
    const job = plan?.job;
    if (job?.running) wasRunning.current = job;
    else if (wasRunning.current && job) {
      wasRunning.current = null;
      if (job.mode === "cut" && !job.cancelled) {
        toast(`Cut ${job.clips} clip${job.clips === 1 ? "" : "s"}${job.removed ? `, removed ${job.removed}` : ""}${job.errors.length ? `; ${job.errors.length} failed` : ""}`, job.errors.length > 0);
      }
    }
  }, [plan?.job?.running]);

  const send = (r: string, clips: Clip[], n: number) => {
    waiting.current.delete(r);
    const p = api.put<Saved>(`/api/datasets/${id}/clips`, { rel: r, clips }).then((res) => {
      if (edits.current.get(r) !== n) return; // edited again since: that save's answer counts
      dirty.current.delete(r);
      setPlan((pl) => pl && { ...pl, changes: res.changes, videos: pl.videos.map((v) => (v.rel === r ? res.video : v)) });
    }, (e) => {
      toast(e.message, true);
      if (edits.current.get(r) === n) { dirty.current.delete(r); load(); }
    });
    inflight.current.add(p);
    p.finally(() => inflight.current.delete(p));
  };
  const flush = async () => {
    for (const [r, w] of [...waiting.current]) { clearTimeout(w.timer); send(r, w.clips, w.n); }
    await Promise.all([...inflight.current]);
  };
  useEffect(() => () => { flush(); }, []); // leaving the page saves what's waiting

  /** A video's clips changed; saved shortly after unless ``save`` is false (a drag that isn't over). */
  const setClips = (r: string, clips: Clip[], save = true) => {
    clips = [...clips].sort((a, b) => a.start - b.start || a.id - b.id);
    setPlan((p) => p && { ...p, videos: p.videos.map((v) => (v.rel === r ? { ...v, clips, next: Math.max(v.next, ...clips.map((c) => c.id + 1)) } : v)) });
    dirty.current.add(r);
    if (!save) return;
    const n = (edits.current.get(r) || 0) + 1;
    edits.current.set(r, n);
    clearTimeout(waiting.current.get(r)?.timer);
    waiting.current.set(r, { timer: window.setTimeout(() => send(r, clips, n), 350), clips, n });
  };

  const replan = async (r: string) => {
    clearTimeout(waiting.current.get(r)?.timer);
    waiting.current.delete(r);
    edits.current.set(r, (edits.current.get(r) || 0) + 1);
    try {
      await flush(); // a save still on its way would land after the new plan and put the old clips back
      const res = await api.post<Saved>(`/api/datasets/${id}/clips/replan`, { rel: r, frames: newLen });
      dirty.current.delete(r);
      setPlan((pl) => pl && { ...pl, changes: res.changes, videos: pl.videos.map((v) => (v.rel === r ? res.video : v)) });
      setSelId(res.video.clips[0]?.id ?? null);
    } catch (e: any) {
      toast(e.message, true);
    }
  };

  const cut = async () => {
    try {
      await flush();
      const job = await api.post<SplitJob>(`/api/datasets/${id}/clips/cut`);
      setPlan((p) => p && { ...p, job });
    } catch (e: any) {
      toast(e.message, true);
    }
  };

  if (error && !plan) return (
    <div className="page stack">
      <div className="small"><a href="#/datasets">Datasets</a> <span className="faint">/</span> <a href={`#/datasets/${id}`}>Dataset</a></div>
      <div className="error-box">{error}</div>
    </div>
  );
  if (!plan) return <div className="page"><p className="muted">Loading the clips…</p></div>;

  const v = plan.videos.find((x) => x.rel === rel) || null;
  const pending = plan.videos.reduce((n, x) => n + pendingOf(x), 0);
  const job = plan.job;
  const cutting = !!job?.running && job.mode === "cut";
  const pick = (r: string) => {
    setRel(r);
    setSelId(plan.videos.find((x) => x.rel === r)?.clips[0]?.id ?? null);
  };
  const side = plan.settings.max_side ? `the shorter side at most ${plan.settings.max_side} px` : "the size kept";

  return (
    <div className="page ce-page">
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="small"><a href="#/datasets">Datasets</a> <span className="faint">/</span> <a href={`#/datasets/${id}`}>{plan.name}</a> <span className="faint">/</span></div>
          <h1>Clip editor</h1>
          <p>Clips of <a href={`#/datasets/${plan.source.id}`}>{plan.source.name}</a> for {plan.family_label}: {plan.fps} fps, {side}.
            {" "}{plan.settings.captions === "copy" ? "A new clip gets its video's caption" : "New clips are left uncaptioned"}; a changed one keeps its own.</p>
        </div>
        <div className="row">
          <button className="btn primary" onClick={cut} disabled={!pending || running}
            title={running ? "Wait for the videos to be looked at" : "Cut the new and changed clips, and remove the ones taken out"}>
            <IconScissors /> {pending ? `Cut ${pending} change${pending === 1 ? "" : "s"}` : "All cut"}</button>
        </div>
      </div>

      {job && (job.running || job.errors.length > 0) && (
        <TaskProgress className="card card-pad ce-task" icon={<IconScissors />} running={job.running} errors={job.errors}
          label={job.running ? (job.mode === "cut" ? "Cutting" : "Looking at the videos")
            : job.mode === "cut" ? `Cut ${job.clips} clip${job.clips === 1 ? "" : "s"}${job.removed ? `, removed ${job.removed}` : ""}` : "Looked at the videos"}
          done={job.done} total={job.total}
          detail={job.running && `files${job.mode === "cut" ? ` · ${job.clips} cut` : ""} ${job.current}${job.message ? ` · ${job.message}` : ""}`}
          onCancel={() => api.del(`/api/datasets/${id}/clips/job`).then(load)} />
      )}

      {!plan.videos.length && <div className="empty"><strong>No videos</strong>{plan.source.name} has no videos to cut clips from.</div>}
      <div className="ce">
        <div className="ce-list" role="listbox" aria-label="Videos">
          {plan.videos.map((x) => {
            const n = pendingOf(x);
            return (
              <button key={x.rel} role="option" aria-selected={x.rel === rel} className={"ce-item" + (x.rel === rel ? " on" : "")} onClick={() => pick(x.rel)}>
                <span className="ce-item-thumb" style={x.path ? { backgroundImage: `url(/api/thumb?size=160&path=${q(x.path)})` } : {}} />
                <span className="ce-item-text">
                  <span className="ellipsis" title={x.rel}>{x.rel}</span>
                  <span className="faint small">
                    {!x.path ? "Gone from the source" : !x.looked ? (running ? "Looking…" : "Not looked at") : `${x.clips.length} clip${x.clips.length === 1 ? "" : "s"} · ${fmtClock(x.duration || 0)}`}
                    {n > 0 && <span style={{ color: "var(--accent)" }}> · {n} to cut</span>}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
        {v && <VideoEditor key={v.rel} plan={plan} v={v} selId={selId} onSelect={setSelId} locked={cutting || !v.looked || !v.path}
          onClips={(clips, save) => setClips(v.rel, clips, save)} newLen={newLen} setNewLen={setNewLen} onReplan={() => setReplanning(v.rel)} />}
      </div>
      {replanning && (
        <ConfirmDialog title="Plan the clips again" danger confirmLabel="Plan again" onClose={() => setReplanning(null)} onConfirm={() => replan(replanning)}
          message={`This video's clips are replaced by new ones of ${newLen} frames each; any already cut are removed when you cut.`} />
      )}
    </div>
  );
}

/** One video: its player (with the selected clip's crop), its timeline, and the selected clip's settings. */
function VideoEditor({ plan, v, selId, onSelect, locked, onClips, newLen, setNewLen, onReplan }: {
  plan: Plan; v: PlanVideo; selId: number | null; onSelect: (id: number | null) => void; locked: boolean;
  onClips: (clips: Clip[], save: boolean) => void; newLen: number; setNewLen: (n: number) => void; onReplan: () => void;
}) {
  const toast = useToast();
  const { fps, edge, lengths } = plan;
  const video = useRef<HTMLVideoElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const clock = useRef<HTMLSpanElement>(null);
  const pps = useRef(1);
  const range = useRef<{ start: number; end: number } | null>(null);
  const [unplayable, setUnplayable] = useState(false);
  const [still, setStill] = useState(0); // where the timeline was clicked, for a video the browser can't play
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [muted, setMuted] = useState(false);
  const [zoom, setZoom] = useState(1);
  const clip = v.clips.find((c) => c.id === selId) || null;
  const [aspect, setAspect] = useState(aspectOf(clip?.crop || null));
  useEffect(() => setAspect(aspectOf(clip?.crop || null)), [selId]);
  useEffect(() => { if (selId != null && !clip) onSelect(null); }, [clip, selId]);

  // the playhead and clock follow the video without re-rendering; a clip playing loops (or stops) at its end
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const el = video.current;
      const t = el && !unplayable ? el.currentTime : still;
      const r = range.current;
      if (el && r && !el.paused && t >= r.end) {
        if (loop) el.currentTime = r.start;
        else { el.pause(); range.current = null; }
      }
      if (head.current) head.current.style.transform = `translateX(${t * pps.current}px)`;
      if (clock.current) clock.current.textContent = fmtT(t);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [loop, unplayable, still]);

  const seek = (t: number) => {
    range.current = null;
    if (video.current && !unplayable) {
      video.current.pause();
      video.current.currentTime = t;
    } else setStill(t);
  };
  useEffect(() => { if (clip) seek(clip.start); }, [selId]);
  const play = () => {
    const el = video.current;
    if (!el || unplayable) return;
    if (!el.paused) { el.pause(); return; }
    if (clip) {
      range.current = { start: clip.start, end: clip.start + clip.frames / fps };
      el.currentTime = clip.start;
    }
    el.play().catch(() => {});
  };
  const now = () => (video.current && !unplayable ? video.current.currentTime : still);

  const change = (c: Clip, save = true) => onClips(v.clips.map((x) => (x.id === c.id ? c : x)), save);
  const remove = (c: Clip) => {
    const i = v.clips.findIndex((x) => x.id === c.id);
    const rest = v.clips.filter((x) => x.id !== c.id);
    onClips(rest, true);
    onSelect(rest[Math.min(i, rest.length - 1)]?.id ?? null);
  };
  /** A clip of the new-clip length at ``t``: inside the shot there if it fits there, and inside the video. */
  const add = (t: number) => {
    if (!fitting(v, 0, [newLen], fps, edge).length) { toast(`This video is too short for a clip of ${newLen} frames`, true); return; }
    const d = newLen / fps;
    const lo = Math.max(0, ...v.cuts.filter((c) => c <= t).map((c) => c + edge / fps));
    const hi = Math.min((v.duration || 0) - edge / fps, ...v.cuts.filter((c) => c > t).map((c) => c - edge / fps));
    const start = place(v, hi - lo >= d ? Math.min(Math.max(t, lo), hi - d) : t, newLen, fps, edge);
    const id = Math.max(v.next, ...v.clips.map((c) => c.id + 1));
    onClips([...v.clips, { id, start, frames: newLen, crop: null }], true);
    onSelect(id);
  };
  const nudge = (c: Clip, by: number) => change({ ...c, start: place(v, c.start + by, c.frames, fps, edge) });

  const keys = useRef<(e: KeyboardEvent) => void>(() => {});
  keys.current = (e) => {
    const el = e.target as HTMLElement;
    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); play(); }
    else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && clip && !locked) {
      e.preventDefault();
      nudge(clip, (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 1 : 1 / (v.fps || 30)));
    } else if ((e.key === "Delete" || e.key === "Backspace") && clip && !locked) { e.preventDefault(); remove(clip); }
    else if (e.key === "Escape") onSelect(null);
    else if ((e.key === "n" || e.key === "N") && !locked) add(now());
  };
  useEffect(() => {
    const on = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  if (!v.path) return <div className="ce-main"><div className="empty"><strong>{v.rel} is gone</strong>It's no longer in {plan.source.name}, so its clips can't be changed. The ones cut stay in this dataset.</div></div>;
  if (!v.looked) return <div className="ce-main"><div className="empty"><strong>Looking at {v.rel}</strong>Finding its length{plan.settings.scenes ? " and shots" : ""}…</div></div>;
  const W = v.width || 16, H = v.height || 9;

  return (
    <div className="ce-main">
      <div className="ce-stage">
        <div className="ce-frame" style={{ aspectRatio: `${W} / ${H}`, width: `min(100%, calc(48vh * ${W / H}))` }}>
          {unplayable
            ? <img src={frameUrl(v, still, 0, fps, 960)} alt="" />
            : <video ref={video} src={`/api/media?path=${q(v.path)}`} preload="auto" playsInline muted={muted}
                onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onError={() => setUnplayable(true)} onClick={play} />}
          {clip?.crop && <Shade crop={clip.crop} v={v} />}
          {clip?.crop && <CropBox v={v} crop={clip.crop} aspect={ASPECTS.find(([k]) => k === aspect)?.[1] ?? null} locked={locked}
            onChange={(crop, save) => change({ ...clip, crop }, save)} />}
        </div>
      </div>
      {unplayable && <div className="validation warning">This browser can't play {v.rel}, so the frames shown are from the server: click the timeline to see a moment.</div>}

      <div className="ce-bar">
        <button className="btn sm" onClick={play} disabled={unplayable}>{playing ? <><IconStop /> Pause</> : <><IconPlay /> {clip ? `Play clip ${clip.id}` : "Play"}</>}</button>
        <label className="check small"><input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} /> Loop the clip</label>
        {v.has_audio && <label className="check small"><input type="checkbox" checked={!muted} onChange={(e) => setMuted(!e.target.checked)} /> Sound</label>}
        <span className="ce-time"><span ref={clock}>0:00.00</span> / {fmtT(v.duration || 0)}</span>
        <span className="spacer" />
        <span className="small muted">Zoom</span>
        <div className="seg ce-zoom">
          <button onClick={() => setZoom(ZOOMS[Math.max(0, ZOOMS.indexOf(zoom) - 1)])} disabled={zoom === 1} aria-label="Zoom out">−</button>
          <button className="on" disabled>{zoom}×</button>
          <button onClick={() => setZoom(ZOOMS[Math.min(ZOOMS.length - 1, ZOOMS.indexOf(zoom) + 1)])} disabled={zoom === ZOOMS[ZOOMS.length - 1]} aria-label="Zoom in">+</button>
        </div>
      </div>

      <Timeline v={v} fps={fps} edge={edge} lengths={lengths} selId={selId} locked={locked} zoom={zoom} head={head} pps={pps}
        focus={clip ? clip.start + clip.frames / fps / 2 : now()} onSelect={onSelect} onSeek={seek} onAdd={add}
        onChange={(clips, save) => onClips(clips, save)} />

      <div className="row wrap ce-tools">
        <button className="btn sm" onClick={() => add(now())} disabled={locked}><IconPlus /> Add a clip here</button>
        <label className="row small" style={{ gap: 6 }}>
          <span className="muted">of</span>
          <select className="ce-len" value={newLen} onChange={(e) => setNewLen(Number(e.target.value))}>
            {lengths.map((l) => <option key={l} value={l}>{l} frames · {(l / fps).toFixed(1)} s{l === plan.frames ? " · the split's" : ""}</option>)}
          </select>
        </label>
        <button className="btn sm ghost" onClick={onReplan} disabled={locked} title="Plan this video's clips again as Split into clips does, at this length"><IconRefresh /> Plan again</button>
        <span className="spacer" />
        <span className="help key-hint">Drag a clip to move it, its ends to change its length; double-click to add one. Space plays · ←/→ a frame (Shift: a second) · N adds · Delete removes</span>
      </div>
      {v.note && !v.clips.length && <div className="validation warning">{v.note[0].toUpperCase() + v.note.slice(1)}: add a shorter clip, or none.</div>}

      {clip && <ClipPanel plan={plan} v={v} clip={clip} locked={locked} aspect={aspect} onChange={change} onRemove={() => remove(clip)}
        onNudge={(by) => nudge(clip, by)}
        onAspect={(a) => {
          setAspect(a);
          const keep = a === "Free" && clip.crop; // a free crop starts as the crop there is
          change({ ...clip, crop: !a ? null : keep ? clip.crop : cropOf(v, ASPECTS.find(([k]) => k === a)?.[1] ?? null, clip.crop) });
        }}
        onCropAll={() => onClips(v.clips.map((c) => ({ ...c, crop: clip.crop })), true)} />}
    </div>
  );
}

/** The video's frames along time, its shot cuts, its clips (dragged to move, by their ends to change length), and
 *  where the player is. */
function Timeline({ v, fps, edge, lengths, selId, locked, zoom, head, pps, focus, onSelect, onSeek, onAdd, onChange }: {
  v: PlanVideo; fps: number; edge: number; lengths: number[]; selId: number | null; locked: boolean; zoom: number;
  head: RefObject<HTMLDivElement>; pps: MutableRefObject<number>; focus: number;
  onSelect: (id: number) => void; onSeek: (t: number) => void; onAdd: (t: number) => void; onChange: (clips: Clip[], save: boolean) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  const dur = Math.max(0.001, v.duration || 0);
  const inner = Math.max(1, width * zoom);
  const scale = inner / dur;
  pps.current = scale;
  useEffect(() => { // zoomed: keep the selected clip (or the playhead) in view
    if (wrap.current) wrap.current.scrollLeft = focus * scale - width / 2;
  }, [zoom, width]);

  const STRIP_H = 48;
  const ratio = (v.width || 16) / (v.height || 9);
  const count = Math.min(64, Math.max(4, Math.ceil(inner / (STRIP_H * ratio)))); // frames enough to fill it, up to 64
  const strip = `/api/clips/strip?path=${q(v.path || "")}&count=${count}&height=64`;
  const cell = inner / count;
  const fw = Math.max(STRIP_H * ratio, cell), fh = fw / ratio; // each frame covers its cell, its middle shown
  const step = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800].find((s) => s * scale >= 80) || 3600;

  const drag = useRef<{ kind: "move" | "l" | "r" | "seek"; id: number; x0: number; start0: number; frames0: number; moved: boolean } | null>(null);
  const toT = (x: number) => Math.min(dur, Math.max(0, (x - track.current!.getBoundingClientRect().left) / scale));
  const down = (e: React.PointerEvent, kind: "move" | "l" | "r" | "seek", c?: Clip) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    if (!c) {
      drag.current = { kind: "seek", id: 0, x0: e.clientX, start0: 0, frames0: 0, moved: false };
      onSeek(toT(e.clientX));
      return;
    }
    onSelect(c.id);
    drag.current = locked ? null : { kind, id: c.id, x0: e.clientX, start0: c.start, frames0: c.frames, moved: false };
  };
  const move = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === "seek") { onSeek(toT(e.clientX)); return; }
    const dx = e.clientX - d.x0;
    if (!d.moved && Math.abs(dx) < 3) return;
    d.moved = true;
    const c = v.clips.find((x) => x.id === d.id);
    if (!c) return;
    let next = c;
    if (d.kind === "move") {
      let s = d.start0 + dx / scale;
      const tol = 8 / scale; // a shot cut pulls a clip's start or end to a frame clear of it
      for (const t of v.cuts) {
        const after = t + edge / fps, before = t - (edge + c.frames) / fps;
        if (Math.abs(s - after) < tol) { s = after; break; }
        if (Math.abs(s - before) < tol) { s = before; break; }
      }
      next = { ...c, start: place(v, s, c.frames, fps, edge) };
      onSeek(next.start);
    } else if (d.kind === "r") {
      const fit = fitting(v, c.start, lengths, fps, edge);
      if (!fit.length) return;
      next = { ...c, frames: nearest(fit, (d.start0 + d.frames0 / fps + dx / scale - c.start) * fps) };
      onSeek(c.start + (next.frames - 1) / fps);
    } else {
      const end = d.start0 + d.frames0 / fps;
      const fit = lengths.filter((l) => end - l / fps >= -1e-3);
      if (!fit.length) return;
      const frames = nearest(fit, (end - (d.start0 + dx / scale)) * fps);
      next = { ...c, frames, start: place(v, end - frames / fps, frames, fps, edge) };
      onSeek(next.start);
    }
    onChange(v.clips.map((x) => (x.id === c.id ? next : x)), false);
  };
  const up = () => {
    const d = drag.current;
    drag.current = null;
    if (!d || d.kind === "seek") return;
    if (d.moved) onChange(v.clips, true);
    else onSeek(v.clips.find((x) => x.id === d.id)?.start ?? 0);
  };
  const sel = v.clips.find((c) => c.id === selId);

  return (
    <div className="ce-tl" ref={wrap}>
      <div className="ce-track" ref={track} style={{ width: inner }} onPointerDown={(e) => down(e, "seek")} onPointerMove={move}
        onPointerUp={up} onPointerCancel={up}
        onDoubleClick={(e) => !locked && !(e.target as HTMLElement).closest(".ce-clip") && onAdd(toT(e.clientX))}>
        <div className="ce-strip">
          {Array.from({ length: count }, (_, i) => (
            <span key={i} style={{ width: cell, backgroundImage: `url(${strip})`, backgroundSize: `${count * fw}px ${fh}px`,
              backgroundPosition: `${(cell - fw) / 2 - i * fw}px ${(STRIP_H - fh) / 2}px` }} />
          ))}
        </div>
        {sel && <div className="ce-sel-span" style={{ left: sel.start * scale, width: Math.max(2, (sel.frames / fps) * scale) }} />}
        {v.cuts.map((t) => <div key={t} className="ce-cut" style={{ left: t * scale }} title={`Shot cut at ${fmtT(t)}`} />)}
        <div className="ce-lane">
          {v.clips.map((c) => {
            const st = stateOf(v, c);
            const cross = crossed(v, c, fps);
            return (
              <div key={c.id} className={`ce-clip ${st}${c.id === selId ? " sel" : ""}${cross.length ? " warn" : ""}${locked ? " locked" : ""}`}
                style={{ left: c.start * scale, width: Math.max(6, (c.frames / fps) * scale) }} onPointerDown={(e) => down(e, "move", c)}
                title={`Clip ${c.id}: ${fmtT(c.start)}, ${c.frames} frames${st === "new" ? " · not cut yet" : st === "changed" ? " · changed since it was cut" : ""}${cross.length ? " · crosses a shot cut" : ""}`}>
                {!locked && <i className="ce-h l" onPointerDown={(e) => down(e, "l", c)} />}
                <span>{c.id}</span>
                {!locked && <i className="ce-h r" onPointerDown={(e) => down(e, "r", c)} />}
              </div>
            );
          })}
        </div>
        <div className="ce-ticks">
          {Array.from({ length: Math.floor(dur / step) + 1 }, (_, i) => <span key={i} style={{ left: i * step * scale }}>{fmtClock(i * step)}</span>)}
        </div>
        <div className="ce-head" ref={head} />
      </div>
    </div>
  );
}

/** The selected clip's crop over the player: dragged to move, by its corners to resize (keeping ``aspect``). */
function CropBox({ v, crop, aspect, locked, onChange }: {
  v: PlanVideo; crop: Crop; aspect: number | null; locked: boolean; onChange: (crop: Crop, save: boolean) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ kind: string; x0: number; y0: number; c0: Crop; last: Crop } | null>(null);
  const W = v.width || 0, H = v.height || 0;
  const down = (e: React.PointerEvent, kind: string) => {
    if (locked || e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { kind, x0: e.clientX, y0: e.clientY, c0: crop, last: crop };
  };
  const move = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const s = W / box.current!.parentElement!.clientWidth; // screen px -> the video's
    const dx = (e.clientX - d.x0) * s, dy = (e.clientY - d.y0) * s;
    const [x0, y0, w0, h0] = d.c0;
    let c: Crop;
    if (d.kind === "move") {
      c = [Math.min(Math.max(0, x0 + dx), W - w0), Math.min(Math.max(0, y0 + dy), H - h0), w0, h0];
    } else { // the opposite corner stays put
      const left = d.kind.includes("w"), top = d.kind.includes("n");
      const ax = left ? x0 + w0 : x0, ay = top ? y0 + h0 : y0;
      const maxW = left ? ax : W - ax, maxH = top ? ay : H - ay;
      let w = Math.min(maxW, Math.max(MIN_CROP, w0 + (left ? -dx : dx)));
      let h = Math.min(maxH, Math.max(MIN_CROP, h0 + (top ? -dy : dy)));
      if (aspect) { // the width from the smallest that leaves the height MIN_CROP too, to the largest the frame has room for
        const lo = MIN_CROP * Math.max(1, aspect), hi = Math.min(maxW, maxH * aspect);
        if (hi < lo) return; // no room for it at this corner
        w = Math.min(hi, Math.max(lo, w0 + (left ? -dx : dx)));
        h = w / aspect;
      }
      c = [left ? ax - w : ax, top ? ay - h : ay, w, h];
    }
    const [x, y, w, h] = c.map(ev);
    d.last = [Math.max(0, x), Math.max(0, y), Math.min(w, W - Math.max(0, x)), Math.min(h, H - Math.max(0, y))];
    onChange(d.last, false);
  };
  const up = () => {
    const d = drag.current;
    drag.current = null;
    if (d && !sameCrop(d.last, d.c0)) onChange(d.last, true);
  };
  return (
    <div ref={box} className={"ce-crop" + (locked ? " locked" : "")} style={pct(crop, v)} onPointerDown={(e) => down(e, "move")}
      onPointerMove={move} onPointerUp={up} onPointerCancel={up} title={`Crop: ${crop[2]} × ${crop[3]} px`}>
      {!locked && ["nw", "ne", "sw", "se"].map((k) => <i key={k} className={k} onPointerDown={(e) => down(e, k)} />)}
    </div>
  );
}

function TimeInput({ value, max, disabled, onCommit }: { value: number; max: number; disabled: boolean; onCommit: (t: number) => void }) {
  const [draft, setDraft] = useState(fmtT(value));
  useEffect(() => setDraft(fmtT(value)), [value]);
  const commit = () => {
    const t = parseT(draft);
    if (t == null || t > max + 1e-3) setDraft(fmtT(value));
    else if (Math.abs(t - value) > 5e-4) onCommit(t);
  };
  return <input className="mono" value={draft} disabled={disabled} onChange={(e) => setDraft(e.target.value)} onBlur={commit}
    onKeyDown={(e) => e.key === "Enter" && commit()} aria-label="Start" />;
}

/** The selected clip: its file, where it starts, how long it is, its crop, what's wrong with it, and its first and
 *  last frames as the cut gives them. */
function ClipPanel({ plan, v, clip, locked, aspect, onChange, onRemove, onNudge, onAspect, onCropAll }: {
  plan: Plan; v: PlanVideo; clip: Clip; locked: boolean; aspect: string; onChange: (c: Clip) => void; onRemove: () => void;
  onNudge: (by: number) => void; onAspect: (a: string) => void; onCropAll: () => void;
}) {
  const { fps, edge, lengths } = plan;
  const st = stateOf(v, clip);
  const fit = fitting(v, clip.start, lengths, fps, edge);
  const cross = crossed(v, clip, fps), over = overlapping(v, clip, fps);
  const shown = useSettled({ start: clip.start, frames: clip.frames }, 300); // frames load once a drag settles
  const frame = 1 / (v.fps || 30);
  const others = v.clips.some((c) => c.id !== clip.id && !sameCrop(c.crop, clip.crop));
  return (
    <div className="card card-pad ce-panel">
      <div className="row wrap" style={{ gap: 8 }}>
        <strong>Clip {clip.id}</strong>
        <span className={"pill " + (st === "cut" ? "completed" : "running")}>{st === "cut" ? "Cut" : st === "new" ? "Not cut yet" : "Changed"}</span>
        {v.stem && <span className="faint mono ellipsis">{clipFile(v.stem, clip)}</span>}
        <span className="spacer" />
        <button className="btn sm danger" onClick={onRemove} disabled={locked}><IconTrash /> Remove</button>
      </div>
      <div className="ce-fields">
        <div className="field">
          <span>Start</span>
          <div className="row" style={{ gap: 4 }}>
            <button className="btn sm icon" onClick={() => onNudge(-frame)} disabled={locked} aria-label="A frame earlier"><IconLeft /></button>
            <TimeInput value={clip.start} max={latest(v, clip.frames, fps, edge)} disabled={locked}
              onCommit={(t) => onChange({ ...clip, start: place(v, t, clip.frames, fps, edge) })} />
            <button className="btn sm icon" onClick={() => onNudge(frame)} disabled={locked} aria-label="A frame later"><IconRight /></button>
          </div>
        </div>
        <label className="field">
          <span>Length</span>
          <select value={clip.frames} disabled={locked} onChange={(e) => onChange({ ...clip, frames: Number(e.target.value) })}>
            {lengths.map((l) => <option key={l} value={l} disabled={!fit.includes(l) && l !== clip.frames}>{l} frames · {(l / fps).toFixed(2)} s</option>)}
          </select>
        </label>
        <div className="field">
          <span>Crop</span>
          <div className="row" style={{ gap: 6 }}>
            <select value={aspect} disabled={locked} onChange={(e) => onAspect(e.target.value)}>
              <option value="">Whole frame</option>
              {ASPECTS.map(([k]) => <option key={k} value={k}>{k}</option>)}
            </select>
            {others && <button className="btn sm" onClick={onCropAll} disabled={locked} title="Give every clip of this video this crop">All clips</button>}
          </div>
        </div>
      </div>
      {clip.crop && <div className="help">{clip.crop[2]} × {clip.crop[3]} px of {v.width} × {v.height}: drag it on the video, or its corners to resize it.</div>}
      {cross.length > 0 && <div className="validation error">Crosses a shot cut at {cross.map(fmtT).join(", ")}: a clip showing a cut teaches the model to cut.</div>}
      {over.length > 0 && <div className="validation warning">Overlaps clip {over.map((o) => o.id).join(", ")}: the same frames train twice.</div>}
      {v.path && (
        <div className="ce-exacts">
          {[[0, "First frame"], [shown.frames - 1, "Last frame"]].map(([n, label]) => (
            <figure className="ce-exact" key={label}>
              <div className="ce-exact-img" style={{ aspectRatio: `${v.width} / ${v.height}` }}>
                <img src={frameUrl(v, shown.start, Number(n), fps)} alt={`${label} of clip ${clip.id}`} />
                {clip.crop && <><Shade crop={clip.crop} v={v} /><div className="ce-crop-mark" style={pct(clip.crop, v)} /></>}
              </div>
              <figcaption>{label} · {fmtT(shown.start + Number(n) / fps)}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}

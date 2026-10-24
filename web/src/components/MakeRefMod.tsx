import { useEffect, useRef, useState } from "react";
import { api, q, type RefModCheck, type SandboxRefMod } from "../api";
import { IconPlus, IconX } from "./icons";
import { Info, Modal, Num, PathInput, fmtSize } from "./ui";

// what a RefMod is of (kept in the file, where ComfyUI's RefMod nodes show it), and the settings' explanations,
// shared with Create Job's RefMod
export const REFMOD_CONCEPTS: [string, string][] = [
  ["generic", "Generic"], ["identity", "Identity · a person or character"], ["pose_motion", "Pose & motion"],
  ["clothing", "Clothing"], ["background", "Background"], ["style", "Style"],
];
export const REFMOD_INFO = {
  mode: "Compressed averages each reference down to a small grid, then refines the grid (a few hundred steps on the grid alone) so that, scaled back up, it matches the full encode: few tokens, so many references fit, with less detail. Full keeps each encode as it is, every reference cropped to the first one's shape: the most detail, and the most tokens.",
  tokens: "The reference tokens the RefMod may put into the model, which take VRAM and time at every step like the video's own. Over the cap, near-duplicate frames go first, then the rest are spread evenly. 0: no cap. A 1024 × 1024 image encoded in full is 1024 tokens.",
  merge: "Refines one grid against every reference at once, so it settles on what they have in common, instead of stacking a grid per reference: the fewest tokens, for one subject seen in several shots.",
  concept: "What the RefMod is of. It's kept in the file, where ComfyUI's RefMod nodes show it; it doesn't change how it's made.",
  sheet: "Puts the images side by side in one picture, each whole, with a grey gap between them, and encodes that as one reference. Otherwise full encoding crops every reference to the first one's shape, and a merge averages them. The images share the sheet's size, so raise the size for more detail in each. The model wasn't trained on sheets: compare it with the same images as separate references.",
};

// MiniMax H3 (the family that makes RefMods) takes clips of its frame rule, 17n + 5 frames, as 5n + 2 latent
// frames; fewer than 5 frames is the first frame alone
export const refLatentFrames = (frames: number) => (frames < 5 ? 1 : Math.floor((frames - 5) / 17) * 5 + 2);

/** A RefMod's encoding settings, as Create Job's RefMod and the Sandbox's Make a RefMod both name them. */
export type RefModEncodingValues = { mode: "full" | "compressed"; size: number; frames: number; grid: number; refine: number; merge: boolean; max_tokens: number };

/** How a RefMod's pictures are encoded: Compressed or Full, the size, the frames of each clip (``clips``; its frame
 * rule, ``frameRule``, from the family), the token cap and, compressed, the grid, refinement and (``mergeable``)
 * Merge. The one set of fields for Create Job's RefMod and the Sandbox's Make a RefMod. ``perFrame``: the tokens a
 * square frame takes, said under the grid. ``sheet``: the references are one sheet of images. */
export function RefModEncoding({ v, set, frameRule, clips = true, mergeable = true, sheet = false, perFrame }: {
  v: RefModEncodingValues; set: (patch: Partial<RefModEncodingValues>) => void; frameRule?: string; clips?: boolean;
  mergeable?: boolean; sheet?: boolean; perFrame?: number;
}) {
  const compressed = v.mode === "compressed";
  const latents = refLatentFrames(Number(v.frames) || 1);
  return <>
    <div className="field">
      <span>Encoding<Info text={REFMOD_INFO.mode} /></span>
      <div className="seg" role="group" aria-label="How the references are encoded">
        {([["compressed", "Compressed"], ["full", "Full"]] as const).map(([m, label]) => (
          <button key={m} type="button" className={v.mode === m ? "on" : ""} aria-pressed={v.mode === m} onClick={() => set({ mode: m })}>{label}</button>
        ))}
      </div>
      <span className="help">{compressed
        ? "Each reference as a small grid, refined to match its full encode: few tokens, less detail."
        : sheet ? "The sheet's encode as it is: the most detail, and the most tokens." : "Each reference's encode as it is: the most detail, and the most tokens."}</span>
    </div>
    <div className="grid cols-3">
      <Num label="Size (short edge, px)" v={v.size} on={(n) => set({ size: n })} step={32} min={256}
        help={sheet ? "The sheet's short edge; the images share it." : compressed ? "What each reference is encoded at before it's compressed; scaled down to it, never up."
          : "Scaled down to it, never up, in multiples of 32."} />
      {clips && <Num label="Frames per clip" v={v.frames} on={(n) => set({ frames: n })} min={1}
        help={`A clip's first frames at 24 fps, snapped down to ${frameRule || "a length the model takes"}: ${latents} latent frame${latents === 1 ? "" : "s"}. Images are one.`} />}
      <Num label="Token cap" info={REFMOD_INFO.tokens} v={v.max_tokens} on={(n) => set({ max_tokens: n })} step={512} min={0} />
      {compressed && <>
        <Num label="Grid (cells, long edge)" v={v.grid} on={(n) => set({ grid: n })} step={2} min={2}
          help={`Even; the short edge keeps the first reference's shape.${perFrame ? ` ${v.grid} is ${perFrame} tokens per square frame.` : ""}`} />
        <Num label="Refinement steps" v={v.refine} on={(n) => set({ refine: n })} step={50} min={0}
          help="How long each grid is refined to match its full encode. 0: the plain average." />
        {mergeable && (
          <label className="check" style={{ alignSelf: "end", paddingBottom: 8 }}>
            <input type="checkbox" checked={!!v.merge} onChange={(e) => set({ merge: e.target.checked })} /> Merge into one reference<Info text={REFMOD_INFO.merge} />
          </label>
        )}
      </>}
    </div>
  </>;
}

/** What a RefMod is of (from ``concepts``), the folder a copy goes to, and its description: the rest of the shared
 * RefMod fields. */
export function RefModAbout({ concept, copyTo, description, set, concepts = REFMOD_CONCEPTS }: {
  concept: string; copyTo: string; description: string; concepts?: [string, string][];
  set: (patch: { concept?: string; copy_to?: string; description?: string }) => void;
}) {
  return <>
    <div className="grid cols-2">
      <label className="field">
        <span>Concept<Info text={REFMOD_INFO.concept} /></span>
        <select value={concept} onChange={(e) => set({ concept: e.target.value })}>
          {concepts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Save a copy to <span className="faint">· optional</span></span>
        <input value={copyTo} onChange={(e) => set({ copy_to: e.target.value })} placeholder="/path/to/ComfyUI/models/refmods" />
        <span className="help">A folder the finished file is copied to, such as ComfyUI's models/refmods.</span>
      </label>
    </div>
    <label className="field">
      <span>Description <span className="faint">· optional</span></span>
      <textarea rows={2} value={description} onChange={(e) => set({ description: e.target.value })} placeholder="Who or what it is, and how it was made" />
    </label>
  </>;
}

const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
const VIDEO_EXTS = [".mp4", ".webm", ".mov", ".mkv", ".avi"];
const SHEET_MAX = 9;
const isClip = (p: string) => VIDEO_EXTS.includes(p.slice(p.lastIndexOf(".")).toLowerCase());

/** The dialog's settings, kept by the page so they stay between makes. */
export type RefModDraft = {
  paths: string[]; sheet: boolean; name: string; mode: "full" | "compressed"; size: number; frames: number; grid: number;
  refine: number; merge: boolean; max_tokens: number; concept: string; description: string; copy_to: string;
};
export const NEW_REFMOD: RefModDraft = {
  paths: [], sheet: false, name: "", mode: "compressed", size: 1024, frames: 22, grid: 16, refine: 500, merge: false,
  max_tokens: 5120, concept: "generic", description: "", copy_to: "",
};

/** Make a RefMod in the Sandbox, of images and clips or of a sheet of images, with the VAE the Sandbox has loaded.
 * ``making`` is the Sandbox's RefMod when it's the one this dialog asked for: it shows its progress, then what was
 * made; ``onBack`` returns to the settings. */
export function MakeRefModDialog({ draft, setDraft, profileId, gpu, making, onMake, onBack, onClose, frameRule }: {
  draft: RefModDraft; setDraft: (d: RefModDraft) => void; profileId: string; gpu: number | null;
  making: SandboxRefMod | null; onMake: (id: string) => void; onBack: () => void; onClose: () => void;
  frameRule?: string; // the clip lengths the profile's family takes
}) {
  const [check, setCheck] = useState<RefModCheck | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const asked = useRef(0);
  const set = (patch: Partial<RefModDraft>) => setDraft({ ...draft, ...patch });
  const paths = draft.paths.map((p) => p.trim()).filter(Boolean);
  const clips = paths.filter(isClip).length;
  const sheetOk = !clips && paths.length >= 2 && paths.length <= SHEET_MAX;
  const sheet = draft.sheet && sheetOk;
  const compressed = draft.mode === "compressed";
  const body = {
    profile_id: profileId, gpu, paths, sheet, name: draft.name,
    settings: {
      refmod_mode: draft.mode, refmod_size: draft.size, refmod_frames: draft.frames, refmod_grid: draft.grid,
      refmod_refine: draft.refine, refmod_merge: draft.merge, refmod_max_tokens: draft.max_tokens,
      refmod_concept: draft.concept, refmod_description: draft.description, refmod_copy_to: draft.copy_to,
    },
  };
  const key = JSON.stringify(body);

  // what it will make, checked a moment after the settings stop changing
  useEffect(() => {
    if (!paths.length) {
      setCheck(null);
      setErr(null);
      return;
    }
    const n = ++asked.current;
    const t = setTimeout(() => {
      api.post<RefModCheck>("/api/sandbox/refmod/check", body).then(
        (r) => { if (n === asked.current) { setCheck(r); setErr(null); } },
        (e) => { if (n === asked.current) { setCheck(null); setErr(e.message); } });
    }, 400);
    return () => clearTimeout(t);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const make = async () => {
    setBusy(true);
    try {
      onMake((await api.post<{ id: string }>("/api/sandbox/refmod", body)).id);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };
  const cancelMake = () => api.post("/api/sandbox/refmod/cancel").catch((e) => setErr(e.message));

  if (making && (making.state === "queued" || making.state === "running"))
    return (
      <Modal title="Making a RefMod" onClose={onClose} footer={<>
        <button className="btn" onClick={cancelMake}>Cancel</button>
        <button className="btn primary" onClick={onClose}>Close</button>
      </>}>
        <div className="stack">
          <div className="progress busy"><div /></div>
          <div className="muted">{making.state === "queued" ? "Waiting for the generations queued before it" : making.message || "Starting"}</div>
          {err && <div className="validation error">{err}</div>}
          <p className="help">It keeps going if you close this, and is added to RefMods when it's done.</p>
        </div>
      </Modal>
    );

  if (making) {
    const made = making.made;
    return (
      <Modal title={made ? `Made ${made.name}` : making.state === "cancelled" ? "Cancelled" : "Couldn't make the RefMod"} onClose={onClose} footer={<>
        <button className="btn" onClick={onBack}>{made ? "Make another" : "Back"}</button>
        <button className="btn primary" onClick={onClose}>Close</button>
      </>}>
        {made ? (
          <div className="stack">
            <div className={"rm-preview" + (made.preview.length === 1 ? " one" : "")}>
              {made.preview.map((p) => <img key={p.file} src={`/api/sandbox/refmod/file/${making.id}/${p.file}`} alt={`Frame ${p.frame + 1}`} title={`Latent frame ${p.frame + 1}`} />)}
            </div>
            <div className="muted small">
              {made.mode === "full" ? "Full" : "Compressed"} · {made.tokens.toLocaleString()} tokens · {made.references} {made.sheet ? "images on a sheet" : made.references === 1 ? "reference" : "references"}
              {made.frames > 1 ? ` · ${made.frames} latent frames` : ""} · {fmtSize(made.bytes)}{made.copied_to ? ` · copied to ${made.copied_to}` : ""}
            </div>
            <div className="mono small faint">{made.path}</div>
            <p className="help">It's in the RefMods list, and added to this generation's RefMods if there was room. The preview decodes
              what the file holds{made.mode === "compressed" ? ": a compressed RefMod looks blurred, as its grid is small" : ""}.</p>
          </div>
        ) : <div className={making.state === "cancelled" ? "muted" : "validation error"}>{making.message || "It was cancelled."}</div>}
      </Modal>
    );
  }

  const sheetUrl = sheet ? `/api/sandbox/refmod/sheet?${paths.map((p) => `path=${q(p)}`).join("&")}&size=${draft.size}` : "";
  const problem = err || check?.error;
  return (
    <Modal title="Make a RefMod" onClose={onClose} footer={<>
      <button className="btn" onClick={onClose}>Cancel</button>
      <button className="btn primary" onClick={make} disabled={!paths.length || !!problem || busy}>{busy ? "Starting…" : "Make RefMod"}</button>
    </>}>
      <div className="stack">
        <p className="help" style={{ margin: 0 }}>References encoded once and saved, which Reference-to-Video takes without encoding
          them again (as ComfyUI does, with the ComfyUI-MiniMaxH3Mod nodes). The Sandbox makes it with its VAE, after the
          generations queued before it, so the model stays loaded.</p>

        <div className="field">
          <span>Images and clips</span>
          <div className="stack" style={{ gap: 8 }}>
            {draft.paths.map((p, i) => (
              <div key={i} className="rm-ref">
                {p.trim() ? <img className="rm-thumb" src={`/api/thumb?path=${q(p.trim())}`} alt="" /> : <span className="rm-thumb" />}
                <PathInput value={p} onChange={(v) => set({ paths: draft.paths.map((x, j) => (j === i ? v : x)) })}
                  exts={[...IMAGE_EXTS, ...VIDEO_EXTS].join(",")} placeholder="/path/to/image.png or clip.mp4" />
                <button className="btn ghost icon" aria-label="Remove" title="Remove" onClick={() => set({ paths: draft.paths.filter((_, j) => j !== i) })}><IconX /></button>
              </div>
            ))}
            <div className="row">
              <button className="btn" onClick={() => set({ paths: [...draft.paths, ""] })}><IconPlus /> Add image or clip</button>
            </div>
          </div>
        </div>

        <div className="field">
          <label className="check">
            <input type="checkbox" checked={sheet} disabled={!sheetOk}
              onChange={(e) => set({ sheet: e.target.checked, ...(e.target.checked ? { mode: "full" as const } : {}) })} />
            Put the images on one sheet<Info text={REFMOD_INFO.sheet} />
          </label>
          {!sheetOk && <span className="help">{clips ? "Only images go on a sheet." : paths.length < 2 ? "Takes 2 or more images." : `Up to ${SHEET_MAX} images.`}</span>}
        </div>
        {sheet && <img className="rm-sheet" src={sheetUrl} alt="The sheet" />}

        <RefModEncoding v={draft} set={set} frameRule={frameRule} clips={clips > 0} mergeable={paths.length > 1 && !sheet} sheet={sheet} />
        <label className="field">
          <span>Name <span className="faint">· optional</span></span>
          <div className="input-suffix">
            <input value={draft.name} onChange={(e) => set({ name: e.target.value })} placeholder={check?.name || "refmod"} />
            <span>.safetensors</span>
          </div>
        </label>
        <RefModAbout concept={draft.concept} copyTo={draft.copy_to} description={draft.description} set={set} />

        {check && !problem && (
          <div className="muted small">
            {check.kept.toLocaleString()} tokens{check.kept < check.tokens ? ` (the cap keeps ${check.kept.toLocaleString()} of ${check.tokens.toLocaleString()})` : ""}
            {" · "}{check.sheet ? `one sheet, ${check.sheet.columns} × ${check.sheet.rows}, ${check.refs[0].width} × ${check.refs[0].height}`
              : `${check.refs.length} ${check.refs.length === 1 ? "reference" : "references"}`}
          </div>
        )}
        {sheet && compressed && (
          <div className="validation warning">Compressed shrinks the whole sheet to one grid of {draft.grid} cells along its long edge,
            leaving each image only a few: use Full, or a much larger grid.</div>
        )}
        {check?.warning && !problem && <div className="validation warning">{check.warning}</div>}
        {problem && <div className="validation error">{problem}</div>}
      </div>
    </Modal>
  );
}

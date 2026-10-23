import { useEffect, useState } from "react";
import { api, q, type FamilySources } from "../api";

type FileRef = { path: string; name: string };
export type CheckIssue = {
  id: string; level: "error" | "warning" | "info"; title: string; detail: string; count: number;
  files?: FileRef[]; groups?: FileRef[][];
};
// how a model that trains with reference images pairs the files: ``with`` of ``of`` have one, from source datasets
// or the _refs folder; ``most``: the most one file has
export type CheckPairs = { with: number; of: number; from: "sources" | "refs"; most: number };
// longest_clip: the longest clip's frames at 24 fps (0: no clips); pairs: null when nothing pairs (text-to-image)
export type CheckResult = { files: number; images: number; videos: number; audio?: number; issues: CheckIssue[]; longest_clip?: number; pairs?: CheckPairs | null };

/** The dataset health check for training at ``resolution`` with a model ``family`` (re-run as they change);
 * ``frames`` is a video model's clip length (``auto``: each clip at its own length), ``seconds`` a music model's
 * (0: whole songs). With ``references`` (the model trains with reference images) it checks how the files pair
 * with the images of the ``sources`` datasets, or else the _refs folder. */
export function useDatasetCheck(id: string | null, params: { resolution?: string; family?: string; frames?: number; seconds?: number; auto?: boolean; references?: boolean; sources?: string[] }) {
  const [result, setResult] = useState<CheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  const query = `resolution=${q(params.resolution || "")}&family=${q(params.family || "")}&frames=${params.frames || 0}`
    + `&seconds=${params.seconds ?? -1}${params.auto ? "&auto=true" : ""}`
    + (params.references ? `&references=true&sources=${q((params.sources || []).join(","))}` : "");
  useEffect(() => {
    if (!id) { setResult(null); return; }
    let live = true;
    setLoading(true);
    const t = setTimeout(() => {
      api.get<CheckResult>(`/api/datasets/${id}/check?${query}`).then(
        (r) => { if (live) { setResult(r); setError(null); } },
        (e) => { if (live) { setResult(null); setError(e.message); } },
      ).finally(() => live && setLoading(false));
    }, 500);
    return () => { live = false; clearTimeout(t); };
  }, [id, query, nonce]);
  return { result, error, loading, recheck: () => setNonce((n) => n + 1) };
}

const LEVEL_WORD = { error: "Error", warning: "Warning", info: "Note" };

/** Findings, worst first. ``compact`` (a Create Job card) leaves out notes and lists file names only;
 * otherwise near-duplicates show as thumbnails. */
export function CheckIssues({ result, compact = false, onOpen }: {
  result: CheckResult; compact?: boolean; onOpen?: (path: string) => void;
}) {
  const shown = result.issues.filter((i) => !compact || i.level !== "info");
  if (!shown.length) {
    const notes = result.issues.length;
    return <div className="check-ok small">✓ No problems found{compact && notes ? ` (${notes} note${notes === 1 ? "" : "s"} on the dataset page)` : ""}</div>;
  }
  const names = (files: FileRef[], total: number, limit: number) => (
    <span className="mono">{files.slice(0, limit).map((f) => f.name).join(", ")}{total > limit ? `, and ${total - limit} more` : ""}</span>
  );
  return (
    <div className="check-list">
      {shown.map((i) => (
        <details key={i.id} className={`check-issue ${i.level}`}>
          <summary><span className="check-dot" aria-label={LEVEL_WORD[i.level]} />{i.title}</summary>
          <div className="check-body">
            <div className="muted">{i.detail}</div>
            {i.files && (compact || !onOpen ? <div className="small">{names(i.files, i.count, compact ? 8 : 40)}</div> : (
              <div className="check-files">
                {i.files.map((f) => <button key={f.path} className="check-file mono" onClick={() => onOpen(f.path)}>{f.name}</button>)}
                {i.count > i.files.length && <span className="faint small">and {i.count - i.files.length} more</span>}
              </div>
            ))}
            {i.groups && i.groups.slice(0, compact ? 4 : 20).map((g, k) => (
              compact || i.id === "shared_caption" ? <div key={k} className="small">{names(g, g.length, 6)}</div> : (
                <div key={k} className="check-group">
                  {g.map((f) => (
                    <figure key={f.path} onClick={() => onOpen?.(f.path)}>
                      <img src={`/api/thumb?path=${q(f.path)}&size=160`} alt="" loading="lazy" />
                      <figcaption className="mono ellipsis" title={f.name}>{f.name}</figcaption>
                    </figure>
                  ))}
                </div>
              )
            ))}
          </div>
        </details>
      ))}
    </div>
  );
}

/** "38 of 40 files pair with a source image" (a model that trains with reference images), and a button to see them. */
export function PairSummary({ pairs, info, onShow }: { pairs: CheckPairs; info: FamilySources; onShow?: () => void }) {
  const all = pairs.with === pairs.of;
  return (
    <div className="pair-summary">
      <span className={all ? "ok" : pairs.with ? "part" : "none"}>{all ? "✓ " : ""}{pairs.with} of {pairs.of} file{pairs.of === 1 ? "" : "s"} pair
        with {pairs.most > 1 ? `${info.noun}s` : `a ${info.noun}`}</span>
      {pairs.from === "refs" && <span className="faint"> from the _refs folder</span>}
      {!all && <span className="muted"> · {pairs.of - pairs.with} {info.without}</span>}
      {onShow && <button className="btn ghost sm" onClick={onShow}>Show pairs</button>}
    </div>
  );
}

import { useEffect, useState } from "react";
import { api, q, type FamilySources } from "../api";
import { Modal } from "./ui";

type PairRow = { path: string; name: string; kind: string; caption: string; refs: (string | null)[] };
// from: where the reference images come from (null: nowhere, so everything trains without); columns: the source
// datasets' names; unused: the reference images no file pairs with
type Pairs = { from: "sources" | "refs" | null; columns: string[]; rows: PairRow[]; unused: { name: string; files: { path: string; name: string }[] }[] };
type Show = "all" | "with" | "without" | "unused";

const SHOWN = 120; // rows drawn at first; "Show more" adds as many again

function Thumb({ path, name }: { path: string; name: string }) {
  return (
    <a className="pair-thumb" href={`/api/media?path=${q(path)}`} target="_blank" rel="noreferrer" title={name}>
      <img src={`/api/thumb?path=${q(path)}&size=256`} alt={name} loading="lazy" />
    </a>
  );
}

/** A dataset's files beside the reference images they train with (an edit's source images), as training pairs them:
 * from the ``sources`` datasets, or else its _refs folder. ``video``: its clips are files too. */
export function PairsDialog({ id, name, sources, video, info, onClose }: {
  id: string; name: string; sources: string[]; video: boolean; info: FamilySources; onClose: () => void;
}) {
  const [data, setData] = useState<Pairs | null>(null);
  const [error, setError] = useState("");
  const [show, setShow] = useState<Show>("all");
  const [limit, setLimit] = useState(SHOWN);
  const key = sources.join(",");
  useEffect(() => {
    api.get<Pairs>(`/api/datasets/${id}/pairs?sources=${q(key)}${video ? "&video=true" : ""}`).then(setData, (e) => setError(e.message));
  }, [id, key, video]);
  const has = (r: PairRow) => r.refs.some(Boolean);
  const count: Record<Show, number> = {
    all: data?.rows.length ?? 0, with: data?.rows.filter(has).length ?? 0, without: data?.rows.filter((r) => !has(r)).length ?? 0,
    unused: data?.unused.reduce((n, u) => n + u.files.length, 0) ?? 0,
  };
  const rows = (data?.rows ?? []).filter((r) => show === "all" || (show === "with") === has(r));
  const tabs: [Show, string][] = [["all", "All"], ["with", "Paired"], ["without", "Without"], ["unused", "Unused"]];
  const fromSources = data?.from === "sources";
  // a file's reference boxes: one per source dataset (empty where it has none there), or its _refs files
  const refBoxes = (r: PairRow) => (fromSources ? r.refs : r.refs.length ? r.refs : [null]).map((p, k) => (
    p ? <Thumb key={k} path={p} name={p.split("/").pop() || p} />
      : <div key={k} className="pair-thumb none" title={fromSources ? `None in ${data!.columns[k]}` : undefined}>No {info.noun}</div>
  ));
  return (
    <Modal title={<>Pairs · {name}</>} onClose={onClose} wide footer={<button className="btn" onClick={onClose}>Close</button>}>
      {error ? <div className="error-box small">{error}</div> : !data ? <div className="muted">Pairing…</div> : !data.from ? (
        <div className="muted">Nothing pairs: no {info.label.toLowerCase()} is chosen, and the dataset has no _refs folder. Its files {info.without}.</div>
      ) : (
        <div className="stack" style={{ gap: 14, minHeight: "min(60vh, 560px)" }}>
          <div className="row wrap" style={{ gap: 10 }}>
            <div className="seg" role="group" aria-label="Show">
              {tabs.map(([k, label]) => (
                <button key={k} className={show === k ? "on" : ""} onClick={() => { setShow(k); setLimit(SHOWN); }}>
                  {label} <span className="faint">{count[k]}</span>
                </button>
              ))}
            </div>
            <span className="muted small">{fromSources ? `${info.noun[0].toUpperCase() + info.noun.slice(1)}s from ${data.columns.join(", ")}, by file name`
              : `${info.noun[0].toUpperCase() + info.noun.slice(1)}s from the _refs folder, by file name`}</span>
          </div>
          {show === "unused" ? (
            count.unused === 0 ? <div className="check-ok small">✓ Every {info.noun} pairs with a file</div> : data.unused.map((u) => (
              <div key={u.name} className="stack" style={{ gap: 8 }}>
                <div className="muted small">{u.files.length} in {u.name} pair{u.files.length === 1 ? "s" : ""} with no file; nothing trains with them.</div>
                <div className="pair-unused">
                  {u.files.map((f) => (
                    <figure key={f.path}><Thumb path={f.path} name={f.name} /><figcaption className="mono ellipsis" title={f.name}>{f.name}</figcaption></figure>
                  ))}
                </div>
              </div>
            ))
          ) : rows.length === 0 ? (
            <div className="muted small">{show === "without" ? `✓ Every file has a ${info.noun}.` : `No file has a ${info.noun}.`}</div>
          ) : (
            <div className="pairs">
              {show === "without" && <div className="muted small">Files without a {info.noun} {info.without}.</div>}
              {rows.slice(0, limit).map((r) => (
                <div key={r.path} className={"pair-row" + (has(r) ? "" : " lone")}>
                  <div className="pair-refs">{refBoxes(r)}</div>
                  <span className="pair-arrow" aria-hidden>→</span>
                  <Thumb path={r.path} name={r.name} />
                  <div className="pair-text">
                    <div className="mono small ellipsis faint" title={r.name}>{r.name}</div>
                    <div className={"pair-cap" + (r.caption ? "" : " faint")}>{r.caption || "No caption"}</div>
                  </div>
                </div>
              ))}
              {rows.length > limit && <button className="btn" style={{ alignSelf: "center" }} onClick={() => setLimit((n) => n + SHOWN)}>Show more ({rows.length - limit} left)</button>}
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";

type Chunk = { file: string; start: number; offset: number; text: string; reset: boolean; more: boolean };

const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;
const MAX_CHARS = 2_000_000; // older output is dropped from the view (the download has it all)
const MAX_LINES = 4000;

/** Raw output -> display lines, as a terminal shows them: a carriage return goes back to the start
 *  of the line, so a progress bar's last update replaces the earlier ones. */
function toLines(raw: string): string[] {
  return raw.replace(ANSI, "").split("\n").map((line) => {
    if (!line.includes("\r")) return line;
    const parts = line.split("\r").filter((p) => p.length);
    return parts.length ? parts[parts.length - 1] : "";
  });
}

function lineClass(line: string): string {
  if (/^=====/.test(line)) return "t-sep";
  if (/Traceback|Error\b|error:|Exception|out of memory|CUDA error|^\s*File ".*", line \d+/i.test(line)) return "t-err";
  if (/warn/i.test(line)) return "t-warn";
  if (/^step\s+\d+\/\d+/.test(line)) return "t-step";
  return "";
}

/** A job's terminal output, polled incrementally; follows new lines unless scrolled up. */
export function Terminal(props: { jobId: string; live: boolean; title?: string; alt?: { label: string; text: string } }) {
  const [raw, setRaw] = useState("");
  const [dropped, setDropped] = useState(false);
  const [wrap, setWrap] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [follow, setFollow] = useState(true);
  const [showAlt, setShowAlt] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const pos = useRef({ file: "", offset: -1 });
  const rawRef = useRef("");

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    pos.current = { file: "", offset: -1 };
    rawRef.current = "";
    setRaw("");
    setDropped(false);
    const pull = async () => {
      let again = false;
      try {
        const c = await api.get<Chunk>(`/api/jobs/${props.jobId}/log/chunk?offset=${pos.current.offset}&file=${pos.current.file}`);
        if (!alive) return;
        pos.current = { file: c.file, offset: c.offset };
        if (c.reset || c.text) {
          let next = c.reset ? c.text : rawRef.current + c.text;
          if (c.reset) setDropped(c.start > 0);
          if (next.length > MAX_CHARS) {
            next = next.slice(next.indexOf("\n", next.length - MAX_CHARS / 2) + 1);
            setDropped(true);
          }
          rawRef.current = next;
          setRaw(next);
        }
        again = c.more;
      } catch {
        /* the next poll retries */
      }
      if (alive) timer = setTimeout(pull, again ? 50 : props.live ? 1000 : 5000);
    };
    pull();
    return () => { alive = false; clearTimeout(timer); };
  }, [props.jobId, props.live]);

  const lines = useMemo(() => {
    const all = toLines(raw);
    if (all.length && all[all.length - 1] === "") all.pop();
    return all.length > MAX_LINES ? { shown: all.slice(-MAX_LINES), hidden: true } : { shown: all, hidden: false };
  }, [raw]);

  useLayoutEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines, follow, wrap, expanded, showAlt]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const onScroll = () => {
    const el = box.current;
    if (el) setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
  };
  const jump = () => {
    setFollow(true);
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  };

  return (
    <div className={"terminal-card card card-pad" + (expanded ? " expanded" : "")}>
      <div className="row terminal-bar">
        <h2 style={{ margin: 0 }}>{showAlt && props.alt ? props.alt.label : props.title || "Terminal"}</h2>
        {props.live && !showAlt && <span className="terminal-live" title="Updating live">live</span>}
        <span className="spacer" />
        {props.alt && <button className="btn sm ghost" onClick={() => setShowAlt((v) => !v)}>{showAlt ? "Show output" : `Show ${props.alt.label.toLowerCase()}`}</button>}
        <button className={"btn sm ghost" + (wrap ? " on" : "")} onClick={() => setWrap((v) => !v)} title="Wrap long lines">Wrap</button>
        <a className="btn sm ghost" href={`/api/jobs/${props.jobId}/log.txt`} download title="Download the whole log">Download</a>
        <button className="btn sm ghost" onClick={() => setExpanded((v) => !v)} title={expanded ? "Back to the page (Esc)" : "Fill the window"}>{expanded ? "Close" : "Expand"}</button>
      </div>
      <div className={"terminal" + (wrap ? "" : " nowrap")} ref={box} onScroll={onScroll} role="log" aria-live="off">
        {showAlt && props.alt ? props.alt.text : (
          <>
            {(dropped || lines.hidden) && <div className="t-note">Earlier output isn't shown here. Download the log to see all of it.</div>}
            {lines.shown.length === 0 && <div className="t-note">No output yet.</div>}
            {lines.shown.map((l, i) => <div key={i} className={lineClass(l)}>{l || " "}</div>)}
          </>
        )}
      </div>
      {!follow && !showAlt && <button className="btn sm terminal-follow" onClick={jump}>↓ Follow output</button>}
    </div>
  );
}

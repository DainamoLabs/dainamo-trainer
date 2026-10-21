import { useEffect, useRef, useState } from "react";
import { api, fmtTime, type AuthStatus, type CaptionProvider, type CaptionStyle, type EncoderInfo, type Gpu, type LlmStatus, type NotifyKind, type Settings, type WhisperModel } from "../api";
import { browserAlertsBlocked, browserAlertsOn, setBrowserAlerts } from "../components/alerts";
import { PathInput, usePoll, useToast } from "../components/ui";

const MODES: { id: Settings["queue"]["mode"]; label: string; desc: string }[] = [
  { id: "sequential", label: "Sequential", desc: "One job at a time, in queue order, on the first selected GPU." },
  { id: "parallel_single", label: "Parallel · single GPU", desc: "Several jobs share one GPU. Only for jobs small enough to fit in its memory together." },
  { id: "parallel_multi", label: "Parallel · multi-GPU", desc: "Jobs spread across all selected GPUs, up to N per GPU." },
];

const KINDS: { id: NotifyKind; label: string; placeholder: string; help: string }[] = [
  { id: "ntfy", label: "ntfy", placeholder: "https://ntfy.sh/your-secret-topic",
    help: "Subscribe to the same topic in the ntfy app on your phone. Pick a hard-to-guess topic name: anyone who knows it can read the messages." },
  { id: "discord", label: "Discord", placeholder: "https://discord.com/api/webhooks/…",
    help: "In a channel's settings: Integrations → Webhooks → New Webhook → Copy Webhook URL." },
  { id: "slack", label: "Slack", placeholder: "https://hooks.slack.com/services/…", help: "An incoming webhook URL from a Slack app." },
  { id: "json", label: "JSON", placeholder: "https://example.com/hook",
    help: "POSTs {event, title, message, url, job_id, status} as JSON, for Home Assistant or your own scripts." },
];
const EVENTS: [keyof Settings["notifications"], string][] = [
  ["on_completed", "Job finished"], ["on_failed", "Job failed or interrupted"], ["on_stopped", "Job stopped"],
];
type Delivery = { at?: number; ok?: boolean; error?: string; title?: string };

const PROVIDERS: { id: CaptionProvider; label: string; desc: string }[] = [
  { id: "llama", label: "llama.cpp", desc: "Dainamo runs llama-server with a GGUF vision model you choose. It runs on the CPU when no GPU is free." },
  { id: "encoder", label: "Text encoder", desc: "A model's own text encoder writes the text: Qwen-Image 2.1's, Krea 2's or MOSS Image's. Nothing else to download; it needs a free GPU." },
  { id: "server", label: "Your own server", desc: "Any OpenAI-compatible endpoint you run: llama-server, Ollama, LM Studio, vLLM…" },
];
const SERVER_STATE: Record<LlmStatus["state"], [string, string]> = {
  off: ["", "Off"], starting: ["running", "Loading"], ready: ["completed", "Ready"], stopping: ["stopping", "Stopping"], failed: ["failed", "Failed"],
};

/** The server Dainamo runs (llama-server or a text encoder): how it is, Start (with the settings on the page, saved first) and Stop, and its log. */
function ServerPanel({ st, reload, save }: { st: LlmStatus; reload: () => void; save: () => Promise<boolean> }) {
  const toast = useToast();
  const [showLog, setShowLog] = useState(false);
  const [log] = usePoll<{ lines: string[] }>(showLog ? "/api/llm/log?lines=300" : null, 2000);
  const pre = useRef<HTMLPreElement>(null);
  useEffect(() => { if (pre.current) pre.current.scrollTop = pre.current.scrollHeight; }, [log]);
  const act = async (what: "start" | "stop") => {
    try {
      if (what === "start" && !(await save())) return;
      await api.post(`/api/llm/${what}`);
    } catch (e: any) {
      toast(e.message, true);
    }
    reload();
  };
  const [pill, label] = SERVER_STATE[st.state];
  const running = st.state === "starting" || st.state === "ready";
  const idle = st.idle_seconds != null && st.idle_seconds >= 60 ? ` · idle ${Math.floor(st.idle_seconds / 60)} min` : "";
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row wrap">
        <span className={"pill " + pill}><span className="dot" />{label}</span>
        <span className="small muted" style={{ flex: "1 1 260px" }}>
          {st.state === "starting" ? `${st.message} · ${st.loading_seconds ?? 0} s`
            : st.message || (st.state === "off" ? "Starts when Auto-Caption or Enhance prompt needs it." : "")}
          {st.state === "ready" && st.model && <> · {st.model}</>}
          {st.state === "ready" && st.endpoint && <> · <span className="mono">{st.endpoint}</span></>}
          {st.state === "ready" && (st.users > 0 ? " · in use" : idle)}
        </span>
        {running
          ? <button className="btn sm" onClick={() => act("stop")}>Stop</button>
          : <button className="btn sm" onClick={() => act("start")} disabled={st.state === "stopping"}>Start</button>}
        <button className="btn sm" onClick={() => setShowLog(!showLog)}>{showLog ? "Hide log" : "Log"}</button>
      </div>
      {st.state === "failed" && <div className="error-box">{st.error}{st.hint && <div style={{ marginTop: 6 }}>{st.hint}</div>}</div>}
      {showLog && (
        <pre ref={pre} className="terminal" style={{ height: 260, margin: 0 }}>
          {st.command.length > 0 && <div className="t-sep">{st.command.join(" ")}</div>}
          {log?.lines.join("\n") || "Nothing yet."}
        </pre>
      )}
    </div>
  );
}

/** Pick a caption style to edit; add, duplicate or delete styles, and choose the default for new datasets. */
function CaptionStyles({ value, onChange }: { value: Settings["caption_styles"]; onChange: (v: Settings["caption_styles"]) => void }) {
  const [sel, setSel] = useState(value.default);
  const styles = value.styles;
  const cur = styles.find((x) => x.id === sel) || styles[0];
  const put = (list: CaptionStyle[], def = value.default) => onChange({ default: def, styles: list });
  const edit = (patch: Partial<CaptionStyle>) => put(styles.map((x) => (x.id === cur.id ? { ...x, ...patch } : x)));
  const add = (from?: CaptionStyle) => {
    const st = { id: `style-${Date.now().toString(36)}`, name: from ? `${from.name} (copy)` : "New style", prompt: from?.prompt || "" };
    put([...styles, st]);
    setSel(st.id);
  };
  const remove = () => {
    const rest = styles.filter((x) => x.id !== cur.id);
    put(rest, value.default === cur.id ? rest[0].id : value.default);
    setSel(rest[0].id);
  };
  return (
    <div className="field">
      <span>Caption styles</span>
      <div className="row wrap" style={{ gap: 8 }}>
        <select value={cur.id} onChange={(e) => setSel(e.target.value)} style={{ flex: "1 1 200px", width: "auto" }} aria-label="Caption style to edit">
          {styles.map((x) => <option key={x.id} value={x.id}>{x.name}{x.id === value.default ? " (default)" : ""}</option>)}
        </select>
        <button className="btn sm" type="button" onClick={() => add()}>New</button>
        <button className="btn sm" type="button" onClick={() => add(cur)}>Duplicate</button>
        <button className="btn sm" type="button" onClick={remove} disabled={styles.length < 2}>Delete</button>
      </div>
      <div className="caption-style">
        <div className="grid cols-2">
          <label className="field"><span>Name</span><input value={cur.name} onChange={(e) => edit({ name: e.target.value })} /></label>
          <label className="check" style={{ alignSelf: "end", paddingBottom: 10 }}>
            <input type="radio" checked={value.default === cur.id} onChange={() => put(styles, cur.id)} /> Default for datasets that haven't picked one
          </label>
        </div>
        <label className="field"><span>Prompt</span><textarea rows={4} value={cur.prompt} onChange={(e) => edit({ prompt: e.target.value })} /></label>
      </div>
      <span className="help">You pick a style when you start Auto-Caption, and each dataset remembers the one it used, so datasets for different models can each keep their own. The captioner sees images, or a few frames of a clip; it can't hear sound.</span>
    </div>
  );
}

export default function SettingsPage({ gpus, auth, onSignOut }: { gpus: Gpu[]; auth: AuthStatus; onSignOut: () => void }) {
  const toast = useToast();
  const [s, setS] = useState<Settings | null>(null);
  const [test, setTest] = useState<string | null>(null);
  const [notifyTest, setNotifyTest] = useState<string | null>(null);
  const [delivery, setDelivery] = useState<Delivery>({});
  const [browser, setBrowser] = useState(browserAlertsOn());
  const [whisper, setWhisper] = useState<WhisperModel[]>([]);
  // a Dainamo server started before this page was built has no caption provider settings: it needs a restart
  const stale = !!s && (!s.llama || !s.text_encoder);
  const provider = (!stale && s?.captioner.provider) || "server";
  const [llm, reloadLlm] = usePoll<LlmStatus>(provider !== "server" ? "/api/llm" : null, 2000);
  const encPath = provider === "encoder" ? s?.text_encoder.path.trim() : "";
  const [encoder] = usePoll<EncoderInfo>(encPath ? `/api/llm/encoder?path=${encodeURIComponent(encPath)}` : null, 0);
  const [encoders] = usePoll<EncoderInfo[]>(provider === "encoder" ? "/api/llm/encoders" : null, 0);
  useEffect(() => {
    api.get<Settings>("/api/settings").then(setS);
    api.get<Delivery>("/api/notifications").then(setDelivery, () => {});
    api.get<{ whisper_models: WhisperModel[] }>("/api/system").then((sys) => setWhisper(sys.whisper_models), () => {});
  }, []);
  if (!s) return <div className="page muted">Loading…</div>;

  const save = async (quiet = false) => {
    try {
      setS(await api.put<Settings>("/api/settings", s));
      if (!quiet) toast("Settings saved");
      return true;
    } catch (e: any) {
      toast(e.message, true);
      return false;
    }
  };
  const setQ = (patch: Partial<Settings["queue"]>) => setS({ ...s, queue: { ...s.queue, ...patch } });
  const setC = (patch: Partial<Settings["captioner"]>) => setS({ ...s, captioner: { ...s.captioner, ...patch } });
  const setL = (patch: Partial<Settings["llama"]>) => setS({ ...s, llama: { ...s.llama, ...patch } });
  const setT = (patch: Partial<Settings["text_encoder"]>) => setS({ ...s, text_encoder: { ...s.text_encoder, ...patch } });
  const setE = (patch: Partial<Settings["enhancer"]>) => setS({ ...s, enhancer: { ...s.enhancer, ...patch } });
  const toggleGpu = (i: number) => {
    const has = s.queue.gpus.includes(i);
    const next = has ? s.queue.gpus.filter((g) => g !== i) : [...s.queue.gpus, i].sort((a, b) => a - b);
    if (next.length) setQ({ gpus: next });
  };
  const setN = (patch: Partial<Settings["notifications"]>) => setS({ ...s, notifications: { ...s.notifications, ...patch } });
  const kind = KINDS.find((k) => k.id === s.notifications.kind) || KINDS[0];
  const blocked = browserAlertsBlocked();
  const toggleBrowser = async (on: boolean) => {
    if (on && Notification.permission !== "granted" && (await Notification.requestPermission()) !== "granted") {
      toast("The browser didn't allow notifications", true);
      return;
    }
    setBrowserAlerts(on);
    setBrowser(on);
  };
  const runNotifyTest = async () => {
    setNotifyTest("Sending…");
    try {
      const r = await api.post<{ ok: boolean; error?: string }>("/api/notifications/test", s.notifications);
      setNotifyTest(r.ok ? "✓ Sent. Check that it arrived." : `✗ ${r.error}`);
    } catch (e: any) {
      setNotifyTest(`✗ ${e.message}`);
    }
  };
  const signOut = async () => {
    await api.post("/api/auth/logout");
    onSignOut();
  };
  const runTest = async () => {
    setTest("Testing…");
    const r = await api.post<{ ok: boolean; models?: string[]; error?: string }>("/api/captioner/test", s.captioner);
    setTest(r.ok ? `✓ Connected${r.models?.length ? ` · models: ${r.models.join(", ")}` : ""}` : `✗ ${r.error}`);
  };

  return (
    <div className="page" style={{ maxWidth: 900 }}>
      <div className="page-head">
        <div><h1>Settings</h1><p>Queue behaviour, the Sandbox, notifications, remote access and the caption provider.</p></div>
        <button className="btn primary" onClick={() => save()}>Save settings</button>
      </div>

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <h2>Queue</h2>
        <div className="stack">
          {MODES.map((m) => (
            <label key={m.id} className={"ds-pick" + (s.queue.mode === m.id ? " on" : "")} style={{ gridTemplateColumns: "auto 1fr" }}>
              <input type="radio" name="mode" checked={s.queue.mode === m.id} onChange={() => setQ({ mode: m.id })} style={{ accentColor: "var(--accent)" }} />
              <div><div style={{ fontWeight: 600 }}>{m.label}</div><div className="muted small">{m.desc}</div></div>
            </label>
          ))}
          <div className="grid cols-2">
            <div className="field">
              <span className="muted small" style={{ fontWeight: 500 }}>GPUs to use</span>
              <div className="row wrap">
                {(gpus.length ? gpus : [{ index: 0, name: "GPU 0" } as Gpu]).map((g) => (
                  <label key={g.index} className="check"><input type="checkbox" checked={s.queue.gpus.includes(g.index)} onChange={() => toggleGpu(g.index)} />GPU {g.index} <span className="faint">{g.name.replace("NVIDIA GeForce ", "")}</span></label>
                ))}
              </div>
              {s.queue.mode !== "parallel_multi" && <span className="help">Uses the first selected GPU. Jobs pinned to a GPU run on that GPU.</span>}
            </div>
            <label className="field">
              <span>Max jobs per GPU</span>
              <input type="number" min={1} max={8} value={s.queue.max_per_gpu} disabled={s.queue.mode === "sequential"} onChange={(e) => setQ({ max_per_gpu: Math.max(1, Number(e.target.value)) })} />
            </label>
          </div>
        </div>
      </div>

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <h2>Sandbox</h2>
        <div className="grid cols-2">
          <label className="field">
            <span>Release the GPU after (minutes idle)</span>
            <input type="number" min={0} step={5} value={s.sandbox.idle_minutes}
              onChange={(e) => setS({ ...s, sandbox: { idle_minutes: Math.max(0, Number(e.target.value)) } })} />
            <span className="help">0 keeps the models loaded until you release them.</span>
          </label>
          <div className="help" style={{ alignSelf: "center" }}>
            {gpus.length >= 2
              ? "With two or more GPUs, the Sandbox holds the GPU it runs on and training uses the others until it's released."
              : "With one GPU, the Sandbox runs only while no training job is running or queued; a job that's ready to start makes it unload."}
          </div>
        </div>
      </div>

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <h2>Notifications</h2>
        <div className="stack">
          <div className="field">
            <span>In this browser</span>
            <label className="check">
              <input type="checkbox" checked={browser && !blocked} disabled={!!blocked} onChange={(e) => toggleBrowser(e.target.checked)} />
              Notify me when a job finishes or fails, even when this tab is in the background
            </label>
            <span className="help">{blocked || "Applies to this browser only. The app also shows a message and marks the tab's title when a job ends."}</span>
          </div>
          <div className="notify-hook">
            <div className="grid cols-2">
              <label className="field">
                <span>Webhook service</span>
                <select value={s.notifications.kind} onChange={(e) => setN({ kind: e.target.value as NotifyKind })}>
                  {KINDS.map((k) => <option key={k.id} value={k.id}>{k.label}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Webhook URL <span className="faint">· blank turns it off</span></span>
                <input className="mono" value={s.notifications.url} placeholder={kind.placeholder} onChange={(e) => setN({ url: e.target.value })} />
              </label>
            </div>
            <span className="help">{kind.help}</span>
          </div>
          <div className="grid cols-2">
            <div className="field">
              <span>Send when</span>
              <div className="row wrap">
                {EVENTS.map(([key, label]) => (
                  <label key={key} className="check"><input type="checkbox" checked={!!s.notifications[key]} onChange={(e) => setN({ [key]: e.target.checked })} />{label}</label>
                ))}
              </div>
            </div>
            <label className="field">
              <span>This app's address <span className="faint">· optional</span></span>
              <input className="mono" value={s.notifications.link} placeholder="http://192.168.1.20:7865" onChange={(e) => setN({ link: e.target.value })} />
              <span className="help">Adds a link to the job in each message. Use an address your phone can reach.</span>
            </label>
          </div>
          <div className="row wrap">
            <button className="btn" disabled={!s.notifications.url.trim()} onClick={runNotifyTest}>Send a test</button>
            {notifyTest && <span className="small muted">{notifyTest}</span>}
            {!notifyTest && delivery.at && (
              <span className={"small " + (delivery.ok ? "muted" : "validation error")}>
                Last message ({fmtTime(delivery.at)}): {delivery.ok ? "sent" : `failed: ${delivery.error}`}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <h2>Remote access</h2>
        {auth.required ? (
          <div className="row wrap" style={{ justifyContent: "space-between" }}>
            <div className="help" style={{ maxWidth: 560 }}>
              Sign-in is on, because this server can be reached from other devices. This browser stays signed in for 30 days.
              Start the server with <code>--new-key</code> to replace the access key and sign every device out.
            </div>
            <button className="btn" onClick={signOut}>Sign out</button>
          </div>
        ) : (
          <div className="help">
            The server only answers on this machine. To use it from your phone or another computer, start it with{" "}
            <code>--host 0.0.0.0</code>. It then asks for an access key, which it prints in the terminal along with a link
            that signs a device in. Use <code>--password</code> to choose your own.
          </div>
        )}
      </div>

      <div className="card card-pad">
        <h2>Caption provider</h2>
        <p className="help" style={{ marginTop: -6, marginBottom: 12 }}>Writes captions for datasets (Auto-Caption) and enhances prompts in the Sandbox.</p>
        <div className="stack">
          {stale && (
            <div className="validation warning">The Dainamo server is older than this page, so it can't run llama.cpp or a text encoder yet.
              Restart it (stop it and run the launcher again), then reload this page.</div>
          )}
          <div className="grid cols-3">
            {PROVIDERS.map((m) => (
              <label key={m.id} className={"ds-pick" + (provider === m.id ? " on" : "")} style={{ gridTemplateColumns: "auto 1fr", alignItems: "start", opacity: stale && m.id !== "server" ? 0.5 : 1 }}>
                <input type="radio" name="caption-provider" checked={provider === m.id} disabled={stale && m.id !== "server"} onChange={() => setC({ provider: m.id })} style={{ accentColor: "var(--accent)" }} />
                <div><div style={{ fontWeight: 600 }}>{m.label}</div><div className="muted small">{m.desc}</div></div>
              </label>
            ))}
          </div>
          {provider === "llama" && (
            <>
              <div className="grid cols-2">
                <div className="field"><span>Model (.gguf)</span>
                  <PathInput value={s.llama.model} exts=".gguf" placeholder="vision-model.gguf" onChange={(v) => setL({ model: v })} />
                  <span className="help">A vision model (e.g. Qwen2.5-VL or Gemma 3), so it can see images and frames.</span>
                </div>
                <div className="field"><span>Vision projector (mmproj)</span>
                  <PathInput value={s.llama.mmproj} exts=".gguf,.mmproj" placeholder="mmproj.gguf" onChange={(v) => setL({ mmproj: v })} />
                  <span className="help">The mmproj file made for that model, usually next to it. Without one the model can't see.</span>
                </div>
                <div className="field"><span>llama-server program <span className="faint">· blank: the one on the PATH</span></span>
                  <PathInput value={s.llama.binary} placeholder={llm?.on_path || "llama-server"} onChange={(v) => setL({ binary: v })} />
                  <span className="help">{llm && !llm.on_path && !s.llama.binary
                    ? <span className="validation warning">There's no llama-server on the PATH: choose one from a llama.cpp release or your own build.</span>
                    : "From a llama.cpp release (github.com/ggml-org/llama.cpp/releases) or your own build."}</span>
                </div>
                <label className="field"><span>Runs on</span>
                  <select value={s.llama.gpu} onChange={(e) => setL({ gpu: e.target.value })}>
                    <option value="auto">A free GPU, else the CPU</option>
                    {gpus.map((g) => <option key={g.index} value={String(g.index)}>GPU {g.index} · {g.name.replace("NVIDIA GeForce ", "")}</option>)}
                    <option value="cpu">The CPU</option>
                  </select>
                  <span className="help">A free GPU is one no training job or Sandbox is on. Training and the Sandbox come first: they make it stop when they need its GPU, and it runs on the CPU while its GPU is theirs.</span>
                </label>
                <div className="grid cols-2">
                  <label className="field"><span>Context size</span>
                    <input type="number" min={0} step={1024} value={s.llama.ctx_size} onChange={(e) => setL({ ctx_size: Math.max(0, Number(e.target.value)) })} />
                    <span className="help">0 sizes it for the requests{llm ? ` (${llm.auto_ctx} tokens)` : ""}.</span>
                  </label>
                  <label className="field"><span>Stop after (min idle)</span>
                    <input type="number" min={0} step={5} value={s.llama.idle_minutes} onChange={(e) => setL({ idle_minutes: Math.max(0, Number(e.target.value)) })} />
                    <span className="help">0 keeps it loaded until you stop it.</span>
                  </label>
                </div>
                <div className="grid cols-2">
                  <label className="field"><span>Max tokens</span><input type="number" value={s.captioner.max_tokens} onChange={(e) => setC({ max_tokens: Number(e.target.value) })} /></label>
                  <label className="field"><span>Temperature</span><input type="number" step={0.1} value={s.captioner.temperature} onChange={(e) => setC({ temperature: Number(e.target.value) })} /></label>
                </div>
              </div>
              <label className="field"><span>Extra arguments <span className="faint">· optional</span></span>
                <input className="mono" value={s.llama.extra_args} placeholder="-ngl 20" onChange={(e) => setL({ extra_args: e.target.value })} />
                <span className="help">Passed to llama-server as written; e.g. <code>-ngl 20</code> keeps part of the model in system RAM, for a GPU it doesn't fit on.</span>
              </label>
            </>
          )}
          {provider === "encoder" && (
            <>
              <div className="field"><span>Text encoder (.safetensors)</span>
                <PathInput value={s.text_encoder.path} exts=".safetensors" placeholder="text_encoder.safetensors" onChange={(v) => setT({ path: v })}
                  validation={s.text_encoder.path && encoder ? (encoder.ok
                    ? { level: "ok", message: `${encoder.name}, ${encoder.whose}'s text encoder${encoder.vision ? "" : ". This file has no vision part, so it can't see images"}` }
                    : { level: "error", message: encoder.error || "" }) : undefined} />
                <span className="help">The text encoder file one of your profiles uses. Choosing one shows which model it is, or why it can't write text.</span>
              </div>
              {encoders && encoders.length > 0 && (
                <div className="field"><span>In your profiles</span>
                  <div className="stack" style={{ gap: 6 }}>
                    {encoders.map((e) => (
                      <div key={e.path} className="row" style={{ gap: 8 }}>
                        <button className="btn sm" type="button" disabled={!e.ok || e.path === s.text_encoder.path} onClick={() => setT({ path: e.path })}>
                          {e.path === s.text_encoder.path ? "Chosen" : "Use"}
                        </button>
                        <span className="small" style={{ minWidth: 0 }} title={e.ok ? e.path : e.error}>
                          {e.ok ? `${e.name} · ${e.whose}'s` : <span className="muted">{e.path.split(/[\\/]/).pop()}: can't write text</span>}
                          <span className="faint"> · {e.profiles?.join(", ")}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="grid cols-2">
                <label className="field"><span>Runs on</span>
                  <select value={s.text_encoder.gpu} onChange={(e) => setT({ gpu: e.target.value })}>
                    <option value="auto">A free GPU</option>
                    {gpus.map((g) => <option key={g.index} value={String(g.index)}>GPU {g.index} · {g.name.replace("NVIDIA GeForce ", "")}</option>)}
                  </select>
                  <span className="help">One no training job or Sandbox is on. Training and the Sandbox come first: they make it stop when they need its GPU.</span>
                </label>
                <div className="stack" style={{ gap: 12 }}>
                  <label className="field"><span>Stop after (min idle)</span>
                    <input type="number" min={0} step={5} value={s.text_encoder.idle_minutes} onChange={(e) => setT({ idle_minutes: Math.max(0, Number(e.target.value)) })} />
                    <span className="help">0 keeps it loaded until you stop it.</span>
                  </label>
                  <div className="grid cols-2">
                    <label className="field"><span>Max tokens</span><input type="number" value={s.captioner.max_tokens} onChange={(e) => setC({ max_tokens: Number(e.target.value) })} /></label>
                    <label className="field"><span>Temperature</span><input type="number" step={0.1} value={s.captioner.temperature} onChange={(e) => setC({ temperature: Number(e.target.value) })} /></label>
                  </div>
                </div>
              </div>
            </>
          )}
          {provider !== "server" && llm && <ServerPanel st={llm} reload={reloadLlm} save={() => save(true)} />}
          {provider === "server" && (
            <>
              <div className="grid cols-2">
                <label className="field"><span>Endpoint base URL</span><input className="mono" value={s.captioner.endpoint} onChange={(e) => setC({ endpoint: e.target.value })} /><span className="help">e.g. http://127.0.0.1:8080/v1 for llama-server with a vision model (--mmproj)</span></label>
                <label className="field"><span>Model name <span className="faint">· optional</span></span><input value={s.captioner.model} onChange={(e) => setC({ model: e.target.value })} /></label>
                <label className="field"><span>API key <span className="faint">· optional</span></span><input type="password" value={s.captioner.api_key} onChange={(e) => setC({ api_key: e.target.value })} /></label>
                <div className="grid cols-2">
                  <label className="field"><span>Max tokens</span><input type="number" value={s.captioner.max_tokens} onChange={(e) => setC({ max_tokens: Number(e.target.value) })} /></label>
                  <label className="field"><span>Temperature</span><input type="number" step={0.1} value={s.captioner.temperature} onChange={(e) => setC({ temperature: Number(e.target.value) })} /></label>
                </div>
              </div>
              <div className="row"><button className="btn" onClick={runTest}>Test connection</button>{test && <span className="small muted">{test}</span>}</div>
            </>
          )}
          <CaptionStyles value={s.caption_styles} onChange={(v) => setS({ ...s, caption_styles: v })} />
          <div className="field">
            <span>Video clips</span>
            <div className="grid cols-2">
              <label className="field"><span>Frames per clip</span>
                <input type="number" min={1} max={12} value={s.captioner.video_frames} onChange={(e) => setC({ video_frames: Number(e.target.value) })} />
              </label>
              <label className="field"><span>Send frames as</span>
                <select value={s.captioner.video_mode} onChange={(e) => setC({ video_mode: e.target.value as Settings["captioner"]["video_mode"] })}>
                  <option value="frames">Separate images</option>
                  <option value="sheet">One contact sheet</option>
                </select>
              </label>
              <label className="field"><span>Whisper model <span className="faint">· default</span></span>
                <select value={s.speech.model} onChange={(e) => setS({ ...s, speech: { ...s.speech, model: e.target.value } })}>
                  {(whisper.length ? whisper : [{ id: s.speech.model, size: "" }]).map((m) => (
                    <option key={m.id} value={m.id}>{m.id}{m.size ? ` · ${m.size}` : ""}{m.id === "large-v3-turbo" ? " · recommended" : ""}</option>
                  ))}
                </select>
              </label>
              <label className="field"><span>Whisper runs on</span>
                <select value={s.speech.device} onChange={(e) => setS({ ...s, speech: { ...s.speech, device: e.target.value as Settings["speech"]["device"] } })}>
                  <option value="auto">A free GPU, else the CPU</option>
                  <option value="cpu">The CPU</option>
                </select>
              </label>
            </div>
            <span className="help">
              A clip's frames go to the model together, in order and with their times, so it can describe what happens.
              A contact sheet puts them in one numbered grid, for models that take one image; it's also used
              automatically when a server refuses several. Whisper writes the speech in clips with sound into
              their captions, when you turn it on in Auto-Caption (each dataset remembers its choice).
            </span>
          </div>
          <div className="field">
            <span>Prompt enhancer (Sandbox)</span>
            <div className="grid cols-2">
              <label className="field"><span>Max tokens</span><input type="number" min={16} value={s.enhancer.max_tokens} onChange={(e) => setE({ max_tokens: Number(e.target.value) })} /></label>
              <label className="field"><span>Temperature</span><input type="number" step={0.1} min={0} max={2} value={s.enhancer.temperature} onChange={(e) => setE({ temperature: Number(e.target.value) })} /></label>
            </div>
            <span className="help">The model gets an instruction for the kind of workflow and the user's prompt (plus the input images, if the model can see). The same prompt and seed get the same rewrite where the server supports seeds. A reasoning model may need more tokens.</span>
            <details>
              <summary className="muted small" style={{ cursor: "pointer" }}>Instructions</summary>
              <div className="stack" style={{ marginTop: 10 }}>
                {([["image", "Text-to-image"], ["edit", "Image edit"], ["video", "Video"]] as const).map(([k, label]) => (
                  <label key={k} className="field"><span>{label}</span>
                    <textarea rows={4} value={s.enhancer[k]} onChange={(e) => setE({ [k]: e.target.value })} />
                  </label>
                ))}
                <span className="help">Clear one to go back to its default.</span>
              </div>
            </details>
          </div>
        </div>
      </div>
    </div>
  );
}

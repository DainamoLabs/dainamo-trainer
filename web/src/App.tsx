import { Component, useEffect, useState, type ReactNode } from "react";
import { api, SIGNED_OUT, type AuthStatus, type Family, type Gpu, type Job, type SystemRam, type Workflow } from "./api";
import { useJobAlerts } from "./components/alerts";
import { IconImages, IconLayers, IconPlus, IconQueue, IconSettings, IconSparkle } from "./components/icons";
import { usePoll } from "./components/ui";
import CreateJob from "./pages/CreateJob";
import Datasets from "./pages/Datasets";
import Profiles from "./pages/Profiles";
import Queue from "./pages/Queue";
import Sandbox from "./pages/Sandbox";
import SettingsPage from "./pages/Settings";
import SignIn from "./pages/SignIn";

export type Tab = "create" | "queue" | "sandbox" | "datasets" | "profiles" | "settings";

const TABS: { id: Tab; label: string; icon: React.FC<React.SVGProps<SVGSVGElement>> }[] = [
  { id: "create", label: "Create Job", icon: IconPlus },
  { id: "queue", label: "Queue", icon: IconQueue },
  { id: "sandbox", label: "Sandbox", icon: IconSparkle },
  { id: "datasets", label: "Datasets", icon: IconImages },
  { id: "profiles", label: "Profiles", icon: IconLayers },
];

function readHash(): { tab: Tab; arg: string } {
  const [tab, arg] = window.location.hash.replace(/^#\/?/, "").split("/");
  const ok = ["create", "queue", "sandbox", "datasets", "profiles", "settings"].includes(tab);
  return { tab: (ok ? tab : "create") as Tab, arg: arg ? decodeURIComponent(arg) : "" };
}

export function go(tab: Tab, arg = "") {
  window.location.hash = `/${tab}${arg ? "/" + encodeURIComponent(arg) : ""}`;
}

/** A page that throws while rendering shows what went wrong instead of a blank screen. */
class PageError extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="page stack">
        <div className="error-box">This page hit an error: {this.state.error.message}</div>
        <p className="help">If Dainamo was just updated, restart its server (stop it and run the launcher again): a server started
          before the update doesn't have what the new pages ask for.</p>
        <div><button className="btn" onClick={() => window.location.reload()}>Reload</button></div>
      </div>
    );
  }
}

export default function App() {
  const [route, setRoute] = useState(readHash());
  useEffect(() => {
    const on = () => setRoute(readHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  useEffect(() => {
    api.get<AuthStatus>("/api/auth").then(setAuth, () => setAuth({ required: false, signed_in: true }));
    const out = () => setAuth({ required: true, signed_in: false });
    window.addEventListener(SIGNED_OUT, out);
    return () => window.removeEventListener(SIGNED_OUT, out);
  }, []);
  const ready = !!auth && (!auth.required || auth.signed_in);
  const [system] = usePoll<{ gpus: Gpu[]; ram?: SystemRam; families: Family[]; workflows: Workflow[] }>(ready ? "/api/system" : null, 4000);
  const [jobs] = usePoll<Job[]>(ready ? "/api/jobs" : null, 4000);
  useJobAlerts(jobs, (id) => go("queue", id));
  const active = jobs?.filter((j) => j.status === "running" || j.status === "queued" || j.status === "stopping").length || 0;
  const families = system?.families || [];

  if (!auth) return null;
  if (!ready) return <SignIn onDone={() => setAuth({ required: true, signed_in: true })} />;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <img src="/logo.png" alt="" />
          <div className="brand-name">Dainamo<span>Trainer</span></div>
        </div>
        <nav className="nav">
          {TABS.map((t) => (
            <button key={t.id} className={route.tab === t.id ? "active" : ""} onClick={() => go(t.id)}>
              <t.icon /> <span>{t.label}</span>
              {t.id === "queue" && active > 0 && <span className="count">{active}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          {system?.gpus.map((g) => (
            <div className="gpu-card" key={g.index}>
              <div className="name">GPU {g.index} · {g.name.replace("NVIDIA GeForce ", "")}</div>
              <div className="row" style={{ marginTop: 3 }}>
                <span>{(g.mem_used / 1024).toFixed(1)} / {(g.mem_total / 1024).toFixed(0)} GB</span>
                <span>{g.util}% · {g.temp}°</span>
              </div>
              <div className="bar"><div style={{ width: `${(100 * g.mem_used) / g.mem_total}%` }} /></div>
            </div>
          ))}
          <nav className="nav">
            <button className={route.tab === "settings" ? "active" : ""} onClick={() => go("settings")}>
              <IconSettings /> <span>Settings</span>
            </button>
          </nav>
        </div>
      </aside>
      <main className="main">
        <PageError key={route.tab}>
          {route.tab === "create" && <CreateJob families={families} />}
          {route.tab === "queue" && <Queue jobId={route.arg} />}
          {route.tab === "sandbox" && <Sandbox families={families} workflows={system?.workflows || []} gpus={system?.gpus || []} ram={system?.ram} />}
          {route.tab === "datasets" && <Datasets datasetId={route.arg} families={families} />}
          {route.tab === "profiles" && <Profiles families={families} />}
          {route.tab === "settings" && <SettingsPage gpus={system?.gpus || []} auth={auth} onSignOut={() => setAuth({ required: true, signed_in: false })} />}
        </PageError>
      </main>
    </div>
  );
}

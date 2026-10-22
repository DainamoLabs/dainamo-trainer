import { Component, useEffect, useState, type ReactNode } from "react";
import { api, SIGNED_OUT, type AuthStatus, type Family, type Gpu, type Job, type SystemRam, type Workflow } from "./api";
import { useJobAlerts } from "./components/alerts";
import { TabBar, TopBar } from "./components/TopBar";
import { usePoll } from "./components/ui";
import CreateJob, { EditJob } from "./pages/CreateJob";
import Datasets from "./pages/Datasets";
import Profiles from "./pages/Profiles";
import Queue from "./pages/Queue";
import Sandbox from "./pages/Sandbox";
import SettingsPage from "./pages/Settings";
import SignIn from "./pages/SignIn";

export type Tab = "create" | "edit" | "queue" | "sandbox" | "datasets" | "profiles" | "settings";

// #/<tab>/<arg>/<view>: e.g. #/queue/<job id>/log, a job page open on its log
function readHash(): { tab: Tab; arg: string; view: string } {
  const [tab, arg, view] = window.location.hash.replace(/^#\/?/, "").split("/");
  const ok = ["create", "edit", "queue", "sandbox", "datasets", "profiles", "settings"].includes(tab);
  return { tab: (ok ? tab : "queue") as Tab, arg: arg ? decodeURIComponent(arg) : "", view: view || "" };
}

export function go(tab: Tab, arg = "", view = "") {
  window.location.hash = `/${tab}${arg ? "/" + encodeURIComponent(arg) : ""}${arg && view ? "/" + view : ""}`;
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
  const waiting = jobs?.filter((j) => j.status === "queued").length || 0;
  const families = system?.families || [];

  if (!auth) return null;
  if (!ready) return <SignIn onDone={() => setAuth({ required: true, signed_in: true })} />;

  return (
    <div className="app">
      <TopBar tab={route.tab} arg={route.arg} jobs={jobs} gpus={system?.gpus || []} waiting={waiting} />
      <main className="main">
        <PageError key={route.tab}>
          {route.tab === "create" && <CreateJob families={families} />}
          {route.tab === "edit" && <EditJob key={route.arg} id={route.arg} families={families} />}
          {route.tab === "queue" && <Queue jobId={route.arg} view={route.view} />}
          {route.tab === "sandbox" && <Sandbox families={families} workflows={system?.workflows || []} gpus={system?.gpus || []} ram={system?.ram} />}
          {route.tab === "datasets" && <Datasets datasetId={route.arg} families={families} />}
          {route.tab === "profiles" && <Profiles families={families} />}
          {route.tab === "settings" && <SettingsPage gpus={system?.gpus || []} auth={auth} onSignOut={() => setAuth({ required: true, signed_in: false })} />}
        </PageError>
      </main>
      <TabBar tab={route.tab} waiting={waiting} />
    </div>
  );
}

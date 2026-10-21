import { useEffect, useRef } from "react";
import type { Job } from "../api";
import { useToast } from "./ui";

// Job-end alerts in the open app: a toast, a browser notification (if turned on for this browser),
// and a mark in the tab title while the tab is in the background. Webhooks are sent by the server.

export const BROWSER_KEY = "dainamo.notify.browser";
const ENDED: Record<string, string> = { completed: "finished", failed: "failed", stopped: "stopped", interrupted: "was interrupted" };
const TITLE = "Dainamo Trainer";

export function browserAlertsOn(): boolean {
  try {
    return localStorage.getItem(BROWSER_KEY) === "1";
  } catch {
    return false;
  }
}

export function setBrowserAlerts(on: boolean) {
  try {
    localStorage.setItem(BROWSER_KEY, on ? "1" : "0");
  } catch {
    /* not remembered in private mode */
  }
}

/** Why browser notifications can't be used here, or null. */
export function browserAlertsBlocked(): string | null {
  if (typeof Notification === "undefined") return "This browser doesn't support notifications.";
  if (!window.isSecureContext)
    return "Browsers only allow notifications on https:// or on this machine (localhost). Use a webhook below to get them on other devices.";
  if (Notification.permission === "denied") return "Notifications are blocked for this site in the browser's settings.";
  return null;
}

function notifyBrowser(title: string, job: Job, onOpen: () => void) {
  if (!browserAlertsOn() || browserAlertsBlocked() || Notification.permission !== "granted") return;
  const p = job.progress;
  const body = [p.step ? `Step ${p.step}/${p.total_steps ?? "?"}` : "", p.avg_loss != null ? `avg loss ${p.avg_loss.toFixed(4)}` : ""]
    .filter(Boolean).join(" · ");
  try {
    const n = new Notification(title, { body, icon: "/logo.png", tag: job.id });
    n.onclick = () => { window.focus(); onOpen(); n.close(); };
  } catch {
    /* some mobile browsers only allow notifications from a service worker */
  }
}

/** Watches the job list and alerts when a running job ends. */
export function useJobAlerts(jobs: Job[] | null, open: (jobId: string) => void) {
  const toast = useToast();
  const seen = useRef<Map<string, string> | null>(null);

  useEffect(() => {
    const reset = () => { if (!document.hidden) document.title = TITLE; };
    document.addEventListener("visibilitychange", reset);
    return () => document.removeEventListener("visibilitychange", reset);
  }, []);

  useEffect(() => {
    if (!jobs) return;
    const prev = seen.current;
    seen.current = new Map(jobs.map((j) => [j.id, j.status]));
    if (!prev) return; // the first look only records what's there
    for (const j of jobs) {
      const was = prev.get(j.id);
      if ((was !== "running" && was !== "stopping") || !ENDED[j.status]) continue;
      const bad = j.status === "failed" || j.status === "interrupted";
      const title = `${j.name} ${ENDED[j.status]}`;
      toast(title, bad);
      if (j.status !== "stopped") notifyBrowser(title, j, () => open(j.id));
      if (document.hidden) document.title = `${bad ? "✗" : j.status === "completed" ? "✓" : "■"} ${title} · ${TITLE}`;
    }
  }, [jobs]);
}

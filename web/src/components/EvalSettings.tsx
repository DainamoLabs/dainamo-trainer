import { useEffect, useState } from "react";
import { api, type Dataset, type JobDetail } from "../api";
import { IconPlus, IconX } from "./icons";
import { Modal, Num, usePoll, useToast } from "./ui";

type S = Record<string, any>;

export const DEFAULT_PROBES = [
  "a portrait photo of a woman with short dark hair, natural window light",
  "a mountain lake at sunrise, wide landscape photo",
  "a bowl of fruit on a wooden kitchen table",
  "a watercolor painting of a city street in the rain",
];

export const EVAL_DEFAULTS: S = {
  picker_enabled: false, val_holdout: 2, val_dataset: "", val_timesteps: 8,
  sweep_enabled: true, sweep_strengths: "0.6, 0.8, 1.0, 1.2",
  bleed_enabled: false, bleed_width: 512, bleed_height: 512, bleed_seed: 1234, bleed_max: 25, bleed_prompts: DEFAULT_PROBES,
};
const EVAL_KEYS = Object.keys(EVAL_DEFAULTS);

/**
 * The checkpoint picker and concept bleed settings. ``holdout``: "choose" for a new job (how many images
 * to hold out of training), "fixed" for a finished job that held some out (the same ones are used), "none"
 * for a finished job that trained on all of its images (the picker needs a separate dataset). Sliders
 * (``picker`` false) have no picker: they aren't trained to reproduce their images. Music (``bleed`` false)
 * has no concept bleed, which compares images.
 */
export function EvalSettings({ s, set, datasets, triggers, holdout, heldOut = 0, imageCount = 0, trainedOn = [], picker = true, bleed = true, sweep = true }: {
  s: S; set: (k: string) => (v: any) => void; datasets: Dataset[] | null; triggers: string[];
  holdout: "choose" | "fixed" | "none"; heldOut?: number; imageCount?: number; trainedOn?: string[]; picker?: boolean; bleed?: boolean;
  sweep?: boolean; // the strength sweep (not for a finetune, which has no strength)
}) {
  const probes: string[] = s.bleed_prompts || [];
  const setProbes = (p: string[]) => set("bleed_prompts")(p);
  const probesWithTrigger = probes.filter((p) => triggers.some((t) => t && p.toLowerCase().includes(t.toLowerCase())));
  const separate = !!s.val_dataset || holdout === "none";
  return (
    <div className="stack" style={{ gap: 12 }}>
      {picker && <div className={"toggle-card" + (s.picker_enabled ? " on" : "")}>
        <label className="check toggle-head">
          <input type="checkbox" checked={s.picker_enabled} onChange={(e) => set("picker_enabled")(e.target.checked)} />
          <span><strong>Best checkpoint picker</strong>
            <span className="help">Scores every saved checkpoint on images kept out of training, with the same noise each time, and recommends the one that generalises best.</span></span>
        </label>
        {s.picker_enabled && (
          <div className="stack" style={{ gap: 12, marginTop: 12 }}>
            <div className="grid cols-2">
              <label className="field">
                <span>Validation images</span>
                <select value={separate ? "dataset" : "holdout"} disabled={holdout === "none"}
                  onChange={(e) => set("val_dataset")(e.target.value === "dataset" ? (datasets?.find((d) => !trainedOn.includes(d.id))?.id || datasets?.[0]?.id || "") : "")}>
                  {holdout !== "none" && <option value="holdout">{holdout === "choose" ? "Hold out images from training" : `The ${heldOut} image${heldOut === 1 ? "" : "s"} held out in training`}</option>}
                  <option value="dataset">{holdout === "choose" ? "Use a separate dataset" : "A separate dataset"}</option>
                </select>
              </label>
              {separate ? (
                <label className="field">
                  <span>Validation dataset</span>
                  <select value={s.val_dataset} onChange={(e) => set("val_dataset")(e.target.value)}>
                    {!s.val_dataset && <option value="" disabled>Choose a dataset…</option>}
                    {datasets?.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.count} files){trainedOn.includes(d.id) ? " · trained on" : ""}</option>)}
                  </select>
                </label>
              ) : holdout === "choose" ? (
                <Num label="Images to hold out" v={s.val_holdout} on={set("val_holdout")} min={1}
                  help={imageCount ? `They're excluded from training (${imageCount} images selected).` : undefined} />
              ) : null}
            </div>
            {holdout === "none" && (
              <div className="help">This job trained on all of its images, so the picker needs other pictures of the same subject that it didn't train on. Scored on its own training images, the last checkpoint always looks best.</div>
            )}
            {separate && trainedOn.includes(s.val_dataset) && (
              <div className="validation warning">This dataset was trained on, so the picker would measure how well the LoRA memorised it rather than how well it generalises.</div>
            )}
            <div className="grid cols-2">
              <Num label="Timesteps per image" v={s.val_timesteps} on={set("val_timesteps")} min={1} help="More is smoother but slower." />
            </div>
            {sweep && <>
              <label className="check"><input type="checkbox" checked={s.sweep_enabled} onChange={(e) => set("sweep_enabled")(e.target.checked)} />
                At the end, render the recommended checkpoint at strengths</label>
              {s.sweep_enabled && (
                <input value={s.sweep_strengths} onChange={(e) => set("sweep_strengths")(e.target.value)} placeholder="0.6, 0.8, 1.0, 1.2" style={{ maxWidth: 320 }} />
              )}
            </>}
          </div>
        )}
      </div>}

      {bleed && <div className={"toggle-card" + (s.bleed_enabled ? " on" : "")}>
        <label className="check toggle-head">
          <input type="checkbox" checked={s.bleed_enabled} onChange={(e) => set("bleed_enabled")(e.target.checked)} />
          <span><strong>Concept bleed detector</strong>
            <span className="help">Renders prompts without your trigger word at every checkpoint and checks whether they start to look like your training images (bleed), and how much they change at all versus the base model (drift). Some bleed is expected for style LoRAs.</span></span>
        </label>
        {s.bleed_enabled && (
          <div className="stack" style={{ gap: 10, marginTop: 12 }}>
            <div className="section-title" style={{ margin: 0 }}>Probe prompts ({probes.filter((p) => p.trim()).length})</div>
            {probes.map((p, i) => (
              <div className="row" key={i}>
                <input value={p} onChange={(e) => setProbes(probes.map((x, j) => (j === i ? e.target.value : x)))} placeholder="a prompt that doesn't use your trigger word" />
                <button className="btn ghost icon" title="Remove probe" aria-label="Remove probe" onClick={() => setProbes(probes.filter((_, j) => j !== i))}><IconX /></button>
              </div>
            ))}
            {probes.length < 8 && <button className="btn add-prompt" onClick={() => setProbes([...probes, ""])}><IconPlus /> Add probe</button>}
            {probesWithTrigger.length > 0 && <div className="validation warning">A probe mentions your trigger word, so it will always show bleed.</div>}
            <div className="grid cols-4">
              <Num label="Width" v={s.bleed_width} on={set("bleed_width")} step={16} />
              <Num label="Height" v={s.bleed_height} on={set("bleed_height")} step={16} />
              <Num label="Seed" v={s.bleed_seed} on={set("bleed_seed")} />
              <Num label="Bleed limit (%)" v={s.bleed_max} on={set("bleed_max")} min={0} help="Recommendations skip checkpoints above this." />
            </div>
          </div>
        )}
      </div>}
    </div>
  );
}

/** Rough cost of evaluating one checkpoint, for the help text under the settings. */
export function evalCost(s: S, validationImages: string | number): string {
  const probes = (s.bleed_prompts || []).filter((p: string) => p.trim()).length;
  return [
    s.picker_enabled ? `${validationImages} × ${s.val_timesteps} validation passes` : "",
    s.bleed_enabled ? `${probes} probe image${probes === 1 ? "" : "s"}` : "",
  ].filter(Boolean).join(" plus ");
}

/** Run the checkpoint picker and/or concept bleed on a finished job's saved checkpoints. */
export function EvaluateDialog({ job, onClose, onQueued }: { job: JobDetail; onClose: () => void; onQueued: () => void }) {
  const toast = useToast();
  const [datasets] = usePoll<Dataset[]>("/api/datasets", 0);
  const trained: S = job.config?.settings || {};
  const slider = !!trained.slider_enabled;
  const music = !!job.media?.includes("audio");
  const st = job.status_detail || ({} as JobDetail["status_detail"]);
  const prior = st.evaluation;
  const heldOut = trained.picker_enabled && !trained.val_dataset ? Number(trained.val_holdout) || 0 : 0;
  const trainedOn: string[] = (job.config?.datasets || []).map((d: any) => d.id);
  const checkpoints = st.checkpoints || [];
  const [busy, setBusy] = useState(false);
  // the last evaluation's settings, else the job's own, else the defaults; the picker is what this is for
  // (concept bleed for a slider, which has no picker)
  const [s, setS] = useState<S>(() => {
    const pick = (o: S | undefined) => Object.fromEntries(Object.entries(o || {}).filter(([k]) => EVAL_KEYS.includes(k)));
    const merged: S = { ...EVAL_DEFAULTS, ...pick(trained), ...pick(prior?.settings),
      ...(slider ? { picker_enabled: false, ...(prior ? {} : { bleed_enabled: true }) } : prior ? {} : { picker_enabled: true }),
      ...(music ? { bleed_enabled: false } : {}) };
    if (Array.isArray(merged.sweep_strengths)) merged.sweep_strengths = merged.sweep_strengths.join(", ");
    return merged;
  });
  const set = (k: string) => (v: any) => setS((x) => ({ ...x, [k]: v }));
  // with nothing held out in training, a dataset it didn't train on is the starting choice
  useEffect(() => {
    if (!heldOut && !s.val_dataset && datasets?.length) set("val_dataset")(datasets.find((d) => !trainedOn.includes(d.id))?.id || "");
  }, [datasets]);
  const triggers = (job.config?.datasets || []).map((d: any) => String(d.trigger || "")).filter(Boolean);
  const valCount = s.val_dataset ? datasets?.find((d) => d.id === s.val_dataset)?.count ?? "N" : heldOut;
  const ready = (s.picker_enabled || s.bleed_enabled) && !(s.picker_enabled && !heldOut && !s.val_dataset);

  const start = async () => {
    setBusy(true);
    try {
      await api.post(`/api/jobs/${job.id}/evaluate`, { settings: s });
      toast("Checkpoint evaluation queued");
      onQueued();
      onClose();
    } catch (e: any) {
      toast(e.message, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title="Evaluate checkpoints" wide onClose={onClose} footer={
      <>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy || !ready} onClick={start}>Queue evaluation</button>
      </>
    }>
      <div className="stack" style={{ gap: 14 }}>
        <div className="muted small">
          Scores the base model and the {checkpoints.length} saved checkpoint{checkpoints.length === 1 ? "" : "s"} (step{checkpoints.length === 1 ? "" : "s"} {checkpoints.map((c) => c.step).join(", ")}),
          then recommends one. It runs through the queue like a job: it loads the model once, then costs about {evalCost(s, valCount) || "nothing"} per checkpoint.
          {prior ? " It replaces the last checkpoint evaluation." : (trained.picker_enabled || trained.bleed_enabled) ? " The results from training are kept too." : ""}
        </div>
        <EvalSettings s={s} set={set} datasets={datasets} triggers={triggers} holdout={heldOut ? "fixed" : "none"} heldOut={heldOut} trainedOn={trainedOn} picker={!slider} bleed={!music} />
        {s.picker_enabled && !heldOut && datasets && datasets.every((d) => trainedOn.includes(d.id)) && (
          <div className="validation warning">Every dataset was trained on. <a href="#/datasets">Add a dataset</a> of other pictures of the same subject to use the picker.</div>
        )}
      </div>
    </Modal>
  );
}

import { useMemo, useRef, useState } from "react";

type Point = { step: number; loss: number };

function smooth(points: Point[], alpha: number): Point[] {
  let ema: number | null = null;
  return points.map((p) => {
    ema = ema == null ? p.loss : alpha * ema + (1 - alpha) * p.loss;
    return { step: p.step, loss: ema };
  });
}

export function LossChart({ points, totalSteps }: { points: Point[]; totalSteps: number }) {
  const W = 760, H = 220, L = 46, R = 12, T = 12, B = 26;
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);

  const { raw, sm, xMax, yMin, yMax } = useMemo(() => {
    const finite = points.filter((p) => isFinite(p.loss));
    const sm = smooth(finite, 0.95);
    const vals = finite.map((p) => p.loss).sort((a, b) => a - b);
    // clip outliers so one spike doesn't flatten the curve
    const hi = vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.98))] : 1;
    const lo = vals.length ? vals[0] : 0;
    const pad = (hi - lo) * 0.08 || 0.01;
    return { raw: finite, sm, xMax: Math.max(totalSteps || 1, finite.at(-1)?.step || 1), yMin: Math.max(0, lo - pad), yMax: hi + pad };
  }, [points, totalSteps]);

  const x = (s: number) => L + ((W - L - R) * s) / xMax;
  const y = (v: number) => T + (H - T - B) * (1 - (Math.min(v, yMax) - yMin) / (yMax - yMin || 1));
  const path = (ps: Point[]) => ps.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join("");

  const ticks = Array.from({ length: 4 }, (_, i) => yMin + ((yMax - yMin) * i) / 3);
  const xt = Array.from({ length: 5 }, (_, i) => Math.round((xMax * i) / 4));

  const onMove = (e: React.MouseEvent) => {
    if (!ref.current || !sm.length) return;
    const r = ref.current.getBoundingClientRect();
    const step = ((e.clientX - r.left) / r.width * W - L) / (W - L - R) * xMax;
    let best = 0;
    for (let i = 0; i < sm.length; i++) if (Math.abs(sm[i].step - step) < Math.abs(sm[best].step - step)) best = i;
    setHover(best);
  };

  if (!raw.length) return <div className="empty" style={{ padding: 30 }}>Loss will appear once training starts.</div>;
  const hp = hover != null ? sm[hover] : null;
  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="#222" />
          <text x={L - 8} y={y(t) + 4} fill="#6b6b6b" fontSize="10.5" textAnchor="end">{t.toFixed(3)}</text>
        </g>
      ))}
      {xt.map((s) => (
        <text key={s} x={x(s)} y={H - 8} fill="#6b6b6b" fontSize="10.5" textAnchor="middle">{s}</text>
      ))}
      <path d={path(raw)} fill="none" stroke="rgba(255,140,0,0.22)" strokeWidth="1" />
      <path d={path(sm)} fill="none" stroke="#ff8c00" strokeWidth="2" />
      {hp && (
        <g>
          <line x1={x(hp.step)} x2={x(hp.step)} y1={T} y2={H - B} stroke="#444" strokeDasharray="3 3" />
          <circle cx={x(hp.step)} cy={y(hp.loss)} r="4" fill="#ff8c00" stroke="#000" strokeWidth="2" />
          <text x={Math.min(x(hp.step) + 8, W - 120)} y={T + 12} fill="#ededed" fontSize="11.5">
            step {hp.step} · avg {hp.loss.toFixed(4)} · raw {raw[hover!]?.loss.toFixed(4)}
          </text>
        </g>
      )}
    </svg>
  );
}

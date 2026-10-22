import { type Hardware } from "../api";
import { usePoll } from "./ui";

/** A temperature, in the warning colour from 80 °C and the danger colour from 90 °C. */
export function Temp({ c, why }: { c: number | null; why?: string }) {
  if (c == null) return <span className="faint" title={why}>—</span>;
  return <span className={c >= 90 ? "hw-hot" : c >= 80 ? "hw-warm" : ""} title="Temperature">{Math.round(c)}°C</span>;
}

function Meter({ label, pct, value }: { label: string; pct: number; value: string }) {
  return (
    <div>
      <div className="hw-row"><span className="muted">{label}</span><span>{value}</span></div>
      <div className="bar"><div style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} /></div>
    </div>
  );
}

/** Each GPU's use, memory and temperature, the CPU's use and temperature, and system RAM, every 2 seconds. */
export function HardwareCards() {
  const [hw] = usePoll<Hardware>("/api/hardware", 2000);
  if (!hw) return null;
  const { cpu, ram } = hw;
  return (
    <div className="hw-grid">
      {hw.gpus.map((g) => (
        <div className="hw-card" key={g.index}>
          <div className="hw-row hw-head"><span className="hw-name" title={g.name}>GPU {g.index} · {g.name.replace(/^NVIDIA (GeForce )?/, "")}</span><Temp c={g.temp} /></div>
          <Meter label="Use" pct={g.util} value={`${g.util}%`} />
          <Meter label="VRAM" pct={(100 * g.mem_used) / g.mem_total} value={`${(g.mem_used / 1024).toFixed(1)} / ${(g.mem_total / 1024).toFixed(0)} GB`} />
        </div>
      ))}
      <div className="hw-card">
        <div className="hw-row hw-head"><span className="hw-name" title={cpu.name}>CPU · {cpu.name}</span>
          <Temp c={cpu.temp} why="Not reported here: Windows doesn't give programs the CPU's temperature, and some Linux machines lack the sensor driver" /></div>
        <Meter label="Use" pct={cpu.percent} value={`${Math.round(cpu.percent)}%`} />
        <div className="hw-row small faint">{cpu.cores} cores · {cpu.threads} threads</div>
      </div>
      <div className="hw-card">
        <div className="hw-row hw-head"><span className="hw-name">RAM</span></div>
        <Meter label="In use" pct={(100 * ram.used_gb) / ram.total_gb} value={`${ram.used_gb.toFixed(1)} / ${ram.total_gb.toFixed(0)} GB`} />
        <div className="hw-row small faint">{ram.available_gb.toFixed(1)} GB free</div>
      </div>
    </div>
  );
}

/** A temperature, in the warning colour from 80 °C and the danger colour from 90 °C. */
export function Temp({ c, why }: { c: number | null; why?: string }) {
  if (c == null) return <span className="faint" title={why}>—</span>;
  return <span className={c >= 90 ? "hw-hot" : c >= 80 ? "hw-warm" : ""} title="Temperature">{Math.round(c)}°C</span>;
}

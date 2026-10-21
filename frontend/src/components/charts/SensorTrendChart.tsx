import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { SensorRow } from "../../lib/chartData";
import { fmtNum } from "../../lib/format";
import { useChartColors } from "../../hooks/useChartColors";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { TooltipCard } from "./ChartTooltip";

/** User-supplied measurements over sample order (or supplied timestamps). */
export function SensorTrendChart({ rows, features, xAxisLabel, yAxisLabel, height = 280 }: {
  rows: SensorRow[]; features: string[]; xAxisLabel: string; yAxisLabel: string; height?: number;
}) {
  const colors = useChartColors();
  const reduced = useReducedMotion();
  const labelByN = new Map(rows.map((r) => [r.n, String(r.xLabel)]));
  const summary = `Trend of ${features.join(", ")} over ${rows.length} samples (${yAxisLabel}).`;
  return (
    <figure aria-label={summary} className="w-full">
      <figcaption className="sr-only">{summary}</figcaption>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
            <CartesianGrid strokeDasharray="3 4" vertical={false} />
            <XAxis dataKey="n" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} tickLine={false} stroke={colors.grid}
              tickFormatter={(n: number) => labelByN.get(n) ?? String(n)}
              label={{ value: xAxisLabel, position: "insideBottom", offset: -12, fill: colors.faint, fontSize: 11 }} />
            <YAxis width={64} tickLine={false} stroke={colors.grid} tickFormatter={(v: number) => fmtNum(v, 3)}
              label={{ value: yAxisLabel, angle: -90, position: "insideLeft", offset: 4, fill: colors.faint, fontSize: 11 }} />
            <Tooltip content={({ active, payload, label }) => active && payload && payload.length ? (
              <TooltipCard title={`Sample ${labelByN.get(Number(label)) ?? label}`}
                rows={payload.map((p) => ({ label: String(p.name), value: fmtNum(p.value as number), color: p.color }))} />
            ) : null} />
            <Legend verticalAlign="top" align="right" height={24} iconSize={10} />
            {features.map((f, i) => (
              <Line key={f} type="linear" dataKey={f} name={f} stroke={colors.series[i % colors.series.length]}
                strokeWidth={1.5} dot={false} connectNulls isAnimationActive={!reduced} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

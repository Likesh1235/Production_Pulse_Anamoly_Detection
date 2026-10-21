import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtNum } from "../../lib/format";
import { useChartColors } from "../../hooks/useChartColors";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { TooltipCard } from "./ChartTooltip";

export interface DetectorBarRow { detector: string; [metric: string]: string | number | null }

/** Grouped bars of report metrics per detector. Null (undefined) metrics are not drawn. */
export function DetectorComparisonChart({ rows, metrics }: { rows: DetectorBarRow[]; metrics: { key: string; label: string }[] }) {
  const colors = useChartColors();
  const reduced = useReducedMotion();
  const palette = colors.series;
  return (
    <figure aria-label={`Detector comparison for ${rows.map((r) => r.detector).join(", ")}`} className="w-full">
      <div style={{ height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }} barCategoryGap="24%">
            <CartesianGrid strokeDasharray="3 4" vertical={false} />
            <XAxis dataKey="detector" tickLine={false} stroke={colors.grid} />
            <YAxis domain={[0, 1]} tickLine={false} stroke={colors.grid} width={40} tickFormatter={(v: number) => v.toFixed(1)} />
            <Tooltip cursor={{ fill: colors.grid, fillOpacity: 0.35 }} content={({ active, payload, label }) => active && payload && payload.length ? (
              <TooltipCard title={String(label)} rows={payload.map((p) => ({ label: String(p.name), value: p.value === null ? "undefined" : fmtNum(p.value as number), color: p.color }))} />
            ) : null} />
            <Legend verticalAlign="top" align="right" height={24} iconSize={10} />
            {metrics.map((m, i) => (
              <Bar key={m.key} dataKey={m.key} name={m.label} fill={palette[i % palette.length]} fillOpacity={0.85}
                radius={[2, 2, 0, 0]} maxBarSize={28} isAnimationActive={!reduced} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

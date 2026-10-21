import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtNum, fmtPct } from "../../lib/format";
import { useChartColors } from "../../hooks/useChartColors";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { TooltipCard } from "./ChartTooltip";

export interface ContributionDatum { feature: string; value: number; detail?: string }

/** Horizontal ranking of features by reconstruction-error contribution. */
export function ContributionBarChart({ data, valueLabel, asPercent, highlightTop = 1 }: {
  data: ContributionDatum[]; valueLabel: string; asPercent?: boolean; highlightTop?: number;
}) {
  const colors = useChartColors();
  const reduced = useReducedMotion();
  const height = Math.max(120, data.length * 26 + 40);
  const top = data.length ? Math.max(...data.map((d) => d.value)) : 0;
  const summary = `${valueLabel} ranking: ${data.slice(0, 5).map((d) => `${d.feature} ${asPercent ? fmtPct(d.value) : fmtNum(d.value)}`).join("; ")}.`;
  return (
    <figure aria-label={summary} className="w-full">
      <figcaption className="sr-only">{summary}</figcaption>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 16, left: 8 }}>
            <CartesianGrid strokeDasharray="3 4" horizontal={false} />
            <XAxis type="number" tickLine={false} stroke={colors.grid} tickFormatter={(v: number) => (asPercent ? fmtPct(v, 0) : fmtNum(v, 3))}
              label={{ value: valueLabel, position: "insideBottom", offset: -8, fill: colors.faint, fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" width={74} tickLine={false} stroke={colors.grid} interval={0} />
            <Tooltip cursor={{ fill: colors.grid, fillOpacity: 0.35 }} content={({ active, payload }) => {
              const d = active && payload && payload.length ? (payload[0].payload as ContributionDatum) : null;
              return d ? <TooltipCard title={d.feature} rows={[{ label: valueLabel, value: asPercent ? fmtPct(d.value) : fmtNum(d.value) }]}
                footer={d.detail ? <span className="text-2xs text-ink-muted">{d.detail}</span> : undefined} /> : null;
            }} />
            <Bar dataKey="value" name={valueLabel} radius={[0, 3, 3, 0]} isAnimationActive={!reduced} maxBarSize={16}>
              {/* Highlight the leading contributor(s) in the data colour; the rest stay neutral. Not a state colour. */}
              {data.map((d, i) => {
                const lead = highlightTop > 0 && (i < highlightTop || d.value === top); // ties with the leader are highlighted too
                return <Cell key={d.feature} fill={lead ? colors.accent : colors.faint} fillOpacity={lead ? 0.95 : 0.6} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

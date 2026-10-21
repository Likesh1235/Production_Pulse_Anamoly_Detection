import { useMemo } from "react";
import {
  Brush, CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import type { TimelinePoint } from "../../lib/chartData";
import { fmtNum, fmtRatio, fmtDateTime } from "../../lib/format";
import { useChartColors } from "../../hooks/useChartColors";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { TooltipCard } from "./ChartTooltip";
import { AnomalyBadge } from "../ui/primitives";

export interface ErrorTimelineChartProps {
  points: TimelinePoint[];
  xAxisLabel: string;
  title: string;
  height?: number;
  logScale?: boolean;
  showBrush?: boolean;
  timestampsAreDates?: boolean;
}

/** Reconstruction error per sample with the calibrated threshold and anomaly markers. */
export function ErrorTimelineChart({ points, xAxisLabel, title, height = 300, logScale = false, showBrush, timestampsAreDates }: ErrorTimelineChartProps) {
  const colors = useChartColors();
  const reduced = useReducedMotion();
  const labelByN = useMemo(() => new Map(points.map((p) => [p.n, p.xLabel])), [points]);
  const canLog = logScale && points.every((p) => p.error > 0 && p.threshold > 0);
  const anomalies = points.filter((p) => p.isAnomaly).length;
  const summary = `${title}: ${points.length} samples, ${anomalies} above the calibrated threshold.`;
  // One threshold for every point (the usual case): shade the region above it as the alarm zone.
  const uniformThreshold = points.length > 0 && points.every((p) => p.threshold === points[0].threshold) ? points[0].threshold : null;
  const yMax = points.reduce((m, p) => Math.max(m, p.error, p.threshold), 0);
  const showDots = points.length <= 60;

  return (
    <figure className="w-full" aria-label={summary}>
      <figcaption className="sr-only">{summary}</figcaption>
      <div style={{ height }} className="w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
            <CartesianGrid strokeDasharray="3 4" vertical={false} />
            <XAxis dataKey="n" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} tickLine={false}
              stroke={colors.grid} tickFormatter={(n: number) => {
                const l = labelByN.get(n) ?? String(n);
                return timestampsAreDates ? fmtDateTime(l).split(",").slice(-1)[0].trim() : l;
              }}
              label={{ value: xAxisLabel, position: "insideBottom", offset: -12, fill: colors.faint, fontSize: 11 }} />
            <YAxis scale={canLog ? "log" : "auto"} domain={canLog ? ["auto", "auto"] : [0, "auto"]} allowDataOverflow={canLog}
              tickFormatter={(v: number) => fmtNum(v, 3)} width={64} tickLine={false} stroke={colors.grid}
              label={{ value: "Reconstruction error", angle: -90, position: "insideLeft", offset: 4, fill: colors.faint, fontSize: 11 }} />
            <Tooltip cursor={{ stroke: colors.faint, strokeDasharray: "3 3" }} content={({ active, payload }) => {
              const p = active && payload && payload.length ? (payload[0].payload as TimelinePoint) : null;
              if (!p) return null;
              return (
                <TooltipCard title={p.sampleLabel} footer={<AnomalyBadge isAnomaly={p.isAnomaly} />} rows={[
                  ...(p.timestamp ? [{ label: "Time", value: timestampsAreDates ? fmtDateTime(p.timestamp) : p.timestamp }] : [{ label: "Position", value: p.xLabel }]),
                  { label: "Reconstruction error", value: fmtNum(p.error), color: colors.accent },
                  { label: "Threshold", value: fmtNum(p.threshold), color: colors.warn },
                  { label: "Error / threshold", value: fmtRatio(p.ratio) },
                  ...(p.predictionId !== null ? [{ label: "Prediction ID", value: `#${p.predictionId}` }] : []),
                ]} />
              );
            }} />
            <Legend verticalAlign="top" align="right" height={24} iconSize={10} />
            {uniformThreshold !== null && !canLog && (
              <ReferenceArea y1={uniformThreshold} y2={yMax * 1.05} fill={colors.danger} fillOpacity={0.06} stroke="none" ifOverflow="extendDomain" />
            )}
            {/* Straight segments: points are separate samples, so no smoothing is implied between them. */}
            <Line type="linear" dataKey="error" name="Reconstruction error" stroke={colors.accent} strokeWidth={1.5}
              dot={showDots ? { r: 2.5, fill: colors.accent, strokeWidth: 0 } : false} activeDot={{ r: 3.5 }} isAnimationActive={!reduced} />
            <Line type="stepAfter" dataKey="threshold" name="Calibrated threshold" stroke={colors.warn} strokeWidth={1.4}
              strokeDasharray="6 4" dot={false} isAnimationActive={false} />
            <Scatter dataKey="anomalyValue" name="Anomaly" fill={colors.danger} stroke={colors.surface} strokeWidth={1}
              isAnimationActive={!reduced} />
            {showBrush && points.length > 40 && (
              <Brush dataKey="n" height={18} stroke={colors.accent} fill={colors.surface} travellerWidth={8}
                tickFormatter={(n: number) => labelByN.get(n) ?? String(n)} />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

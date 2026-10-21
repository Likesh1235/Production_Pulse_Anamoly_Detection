/** Shapes real inputs and API results into chart rows. Nothing here creates values. */

import type { FeatureContribution, HistoryItem, PredictionResult } from "./api/types";
import type { ParsedSample } from "./validation";

export interface TimelinePoint {
  /** x position: 1-based sample order (or record order for history) */
  n: number;
  xLabel: string;
  sampleLabel: string;
  error: number;
  threshold: number;
  isAnomaly: boolean;
  anomalyValue: number | null;
  ratio: number;
  predictionId: number | null;
  timestamp?: string;
}

export interface ScoredRow {
  input: ParsedSample;
  result: PredictionResult;
}

function point(n: number, xLabel: string, sampleLabel: string, r: { reconstruction_error: number; threshold: number; is_anomaly: boolean }, predictionId: number | null, timestamp?: string): TimelinePoint {
  return {
    n,
    xLabel,
    sampleLabel,
    error: r.reconstruction_error,
    threshold: r.threshold,
    isAnomaly: r.is_anomaly,
    anomalyValue: r.is_anomaly ? r.reconstruction_error : null,
    ratio: r.threshold > 0 ? r.reconstruction_error / r.threshold : Number.NaN,
    predictionId,
    timestamp,
  };
}

/** Session rows (user input + API result) in submission order. */
export function scoredTimeline(rows: ScoredRow[], startIndex = 0): TimelinePoint[] {
  return rows.map(({ input, result }, i) => {
    const n = startIndex + i + 1;
    const label = input.sampleId ?? `#${n}`;
    return point(n, input.timestamp ?? String(n), label, result, result.prediction_id, input.timestamp);
  });
}

/** History arrives newest-first; charts read left-to-right in record order. */
export function historyTimeline(items: HistoryItem[]): TimelinePoint[] {
  return [...items]
    .sort((a, b) => a.id - b.id)
    .map((it) => point(it.id, `#${it.id}`, it.sample_id ?? `record #${it.id}`, it, it.id, it.created_utc));
}

export interface ContributionSummary {
  feature: string;
  /** Number of samples where this feature was among the returned top contributors. */
  appearances: number;
  /** Number of samples where it was the single largest contributor. */
  topOneCount: number;
  meanShare: number;
  maxSquaredError: number;
}

/** Aggregate the per-sample TOP-K contributors returned by the API (features outside a
 *  sample's top-k are unknown, not zero, so they simply do not contribute to that sample). */
export function aggregateContributions(lists: FeatureContribution[][]): ContributionSummary[] {
  const acc = new Map<string, { a: number; t: number; share: number; max: number }>();
  for (const top of lists) {
    top.forEach((c, rank) => {
      const e = acc.get(c.feature) ?? { a: 0, t: 0, share: 0, max: 0 };
      e.a += 1;
      if (rank === 0) e.t += 1;
      e.share += c.share;
      e.max = Math.max(e.max, c.squared_error);
      acc.set(c.feature, e);
    });
  }
  return [...acc.entries()]
    .map(([feature, e]) => ({ feature, appearances: e.a, topOneCount: e.t, meanShare: e.share / e.a, maxSquaredError: e.max }))
    .sort((x, y) => y.topOneCount - x.topOneCount || y.appearances - x.appearances || y.meanShare - x.meanShare);
}

export interface SensorRow {
  n: number;
  xLabel: string;
  [feature: string]: number | string;
}

/** Raw measurements from the user's input (optionally z-scored within the window for display only). */
export function sensorRows(inputs: ParsedSample[], features: string[], startIndex = 0, normalise = false): SensorRow[] {
  const stats = new Map<string, { mean: number; sd: number }>();
  if (normalise) {
    for (const f of features) {
      const vals = inputs.map((s) => s.features[f]).filter((v): v is number => Number.isFinite(v));
      const mean = vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
      const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(vals.length, 1)) || 1;
      stats.set(f, { mean, sd });
    }
  }
  return inputs.map((s, i) => {
    const n = startIndex + i + 1;
    const row: SensorRow = { n, xLabel: s.timestamp ?? String(n) };
    for (const f of features) {
      const v = s.features[f];
      if (!Number.isFinite(v)) continue;
      const st = stats.get(f);
      row[f] = st ? (v - st.mean) / st.sd : v;
    }
    return row;
  });
}

export function anomalyRate(flagged: number, total: number): number | null {
  return total > 0 ? flagged / total : null;
}

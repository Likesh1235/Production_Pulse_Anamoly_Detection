/** TEST-ONLY synthetic fixtures shaped like the real API responses (docs/API.md).
 *  They are never used by the application itself and are not experimental results. */

import type {
  HealthResponse, HistoryItem, HistoryResponse, MetricsResponse, ModelStatusResponse, PredictRequest, PredictResponse,
} from "../lib/api/types";
import { FEATURE_COLUMNS } from "../lib/features";

export const THRESHOLD = 0.25;

export const healthReady: HealthResponse = {
  status: "ok", api_version: "0.1.0",
  model: { available: true, status: "ready", reason: null },
  database: { available: true, status: "ready", reason: null },
};

export const healthNoModel: HealthResponse = {
  ...healthReady,
  model: { available: false, status: "not_found", reason: "No trained model was found in the configured model directory." },
};

export const modelStatusReady: ModelStatusResponse = {
  available: true, status: "ready", reason: null, model_dir: "artifacts/models/autoencoder_v1",
  feature_schema: { n_features: 52, feature_columns: [...FEATURE_COLUMNS] },
  model_id: "abcdef123456", loaded_utc: "2026-10-20T10:00:00+00:00",
  metadata: { threshold: THRESHOLD, threshold_source: "calibrated: test fixture", settings: { hidden_layers: [32, 16], bottleneck: 8 } },
};

export const modelStatusMissing: ModelStatusResponse = {
  available: false, status: "not_found", reason: healthNoModel.model.reason, model_dir: "artifacts/models/autoencoder_v1",
  feature_schema: { n_features: 52, feature_columns: [...FEATURE_COLUMNS] },
};

export const metricsUnavailable: MetricsResponse = { available: false, reason: "No evaluation report has been saved yet." };

/** Features with distinct synthetic values (k shifts all values). */
export function featureRecord(k = 0, order: readonly string[] = FEATURE_COLUMNS): Record<string, number> {
  const out: Record<string, number> = {};
  order.forEach((name) => { out[name] = FEATURE_COLUMNS.indexOf(name) + 0.5 + k; });
  return out;
}

export function csvText(rows: number, order: readonly string[] = FEATURE_COLUMNS, extra?: { header: string; values: (i: number) => string }): string {
  const header = [...order, ...(extra ? [extra.header] : [])].join(",");
  const lines = Array.from({ length: rows }, (_, i) => {
    const rec = featureRecord(i, order);
    return [...order.map((c) => String(rec[c])), ...(extra ? [extra.values(i)] : [])].join(",");
  });
  return [header, ...lines].join("\n");
}

/** Deterministic fake scoring for tests: error = 0.1 * (index + 1); anomaly if above THRESHOLD. */
export function fakePredict(body: PredictRequest, startId = 1): PredictResponse {
  const results = body.samples.map((s, i) => {
    const err = 0.1 * (i + 1);
    return {
      index: i, sample_id: s.sample_id ?? null, prediction_id: startId + i, reconstruction_error: err,
      is_anomaly: err > THRESHOLD, threshold: THRESHOLD,
      top_features: [
        { feature: "xmeas_7", squared_error: 1.5, share: 0.5 },
        { feature: "xmv_10", squared_error: 0.9, share: 0.3 },
      ],
    };
  });
  return {
    request_id: "req0000000001", model_id: "abcdef123456", threshold: THRESHOLD, threshold_source: "calibrated: test fixture",
    n_samples: results.length, n_anomalies: results.filter((r) => r.is_anomaly).length, history_stored: true, warnings: [], results,
  };
}

export function historyItem(id: number, isAnomaly = false): HistoryItem {
  return {
    id, created_utc: "2026-10-20T10:00:00.000+00:00", request_id: `req${id}`, sample_index: 0, sample_id: `s-${id}`,
    reconstruction_error: isAnomaly ? 0.9 : 0.1, threshold: THRESHOLD, is_anomaly: isAnomaly, model_id: "abcdef123456",
    top_features: [{ feature: "xmeas_9", squared_error: 0.4, share: 0.6 }],
  };
}

export function historyPage(ids: number[], total: number, next: number | null, limit = 50): HistoryResponse {
  return { items: ids.map((id) => historyItem(id, id % 2 === 0)), total, limit, offset: 0, next_before_id: next };
}

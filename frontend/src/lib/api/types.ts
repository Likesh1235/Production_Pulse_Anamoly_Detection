/** Types mirroring src/production_pulse/api/schemas.py and routes.py exactly. */

export interface ComponentStatus {
  available: boolean;
  status: string;
  reason?: string | null;
}

/** GET /health */
export interface HealthResponse {
  status: string;
  api_version: string;
  model: ComponentStatus;
  database: ComponentStatus;
}

export interface FeatureSchema {
  n_features: number;
  feature_columns: string[];
}

/** Keys come from metadata.json and are present only if the artifact contains them. */
export interface ModelMetadata {
  model_type?: string;
  format_version?: number;
  created_utc?: string;
  score?: string;
  seed?: number;
  settings?: Record<string, unknown>;
  n_fit_rows?: number;
  n_validation_rows?: number;
  n_calibration_rows?: number;
  training_summary?: Record<string, unknown>;
  library_versions?: Record<string, string>;
  model_sha256?: string;
  preprocessing?: { method?: string; fit_source?: string; n_samples_fit?: number };
  threshold?: number;
  threshold_source?: string;
}

/** GET /model/status */
export interface ModelStatusResponse {
  available: boolean;
  status: string;
  reason?: string | null;
  model_dir: string;
  feature_schema: FeatureSchema;
  model_id?: string;
  loaded_utc?: string;
  metadata?: ModelMetadata;
}

export interface FeatureContribution {
  feature: string;
  squared_error: number;
  share: number;
}

export interface PredictSample {
  features: Record<string, number>;
  sample_id?: string;
}

/** POST /predict body (the API also accepts {features, sample_id} for one sample). */
export interface PredictRequest {
  samples: PredictSample[];
}

export interface PredictionResult {
  index: number;
  sample_id: string | null;
  prediction_id: number | null;
  reconstruction_error: number;
  is_anomaly: boolean;
  threshold: number;
  top_features: FeatureContribution[];
}

export interface PredictResponse {
  request_id: string;
  model_id: string | null;
  threshold: number;
  threshold_source: string;
  n_samples: number;
  n_anomalies: number;
  history_stored: boolean;
  warnings: string[];
  results: PredictionResult[];
}

export interface HistoryItem {
  id: number;
  created_utc: string;
  request_id: string;
  sample_index: number;
  sample_id: string | null;
  reconstruction_error: number;
  threshold: number;
  is_anomaly: boolean;
  model_id: string | null;
  top_features: FeatureContribution[];
}

/** GET /history */
export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
  next_before_id: number | null;
}

export interface HistoryQuery {
  limit?: number;
  offset?: number;
  beforeId?: number;
  anomaliesOnly?: boolean;
}

/** reports/evaluation_metrics.json (production_pulse.reporting) */
export interface EvaluationReport {
  report_version: number;
  generated_utc: string;
  source: string;
  model_id?: string;
  metrics: Record<string, unknown>;
}

/** GET /metrics */
export type MetricsResponse =
  | { available: false; reason: string }
  | { available: true; report: EvaluationReport };

/** Uniform error envelope (production_pulse.api.errors) */
export interface ApiErrorEnvelope {
  error: { code: string; message: string; details?: unknown };
}

/** Error codes produced by the backend (docs/API.md §5). */
export type ApiErrorCode =
  | "validation_error"
  | "invalid_features"
  | "value_out_of_range"
  | "batch_too_large"
  | "model_unavailable"
  | "database_unavailable"
  | "unauthorized"
  | "reload_disabled"
  | "not_found"
  | "method_not_allowed"
  | "internal_error"
  | "http_error";

export const MODEL_STATUS_LABELS: Record<string, string> = {
  ready: "Ready",
  not_loaded: "Not loaded",
  not_found: "No trained model",
  load_failed: "Load failed",
  dependency_missing: "Dependency missing",
  incompatible: "Incompatible schema",
  no_preprocessing: "No scaler",
  not_calibrated: "Not calibrated",
};

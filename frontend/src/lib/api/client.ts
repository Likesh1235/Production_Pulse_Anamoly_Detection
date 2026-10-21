/** Central API client: one place for URLs, timeouts, error envelopes and response checks. */

import { ApiRequestError } from "./errors";
import type {
  ApiErrorEnvelope,
  ComponentStatus,
  HealthResponse,
  HistoryQuery,
  HistoryResponse,
  MetricsResponse,
  ModelStatusResponse,
  PredictRequest,
  PredictResponse,
} from "./types";

export interface ApiClient {
  readonly baseUrl: string;
  health(signal?: AbortSignal): Promise<HealthResponse>;
  modelStatus(signal?: AbortSignal): Promise<ModelStatusResponse>;
  predict(body: PredictRequest, signal?: AbortSignal): Promise<PredictResponse>;
  metrics(signal?: AbortSignal): Promise<MetricsResponse>;
  history(query?: HistoryQuery, signal?: AbortSignal): Promise<HistoryResponse>;
}

export interface ApiClientOptions {
  baseUrl: string;
  timeoutMs: number;
  fetchImpl?: typeof fetch;
}

// ------------------------------------------------------------------ runtime shape checks
const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === "string";
const isNullableStr = (v: unknown) => v === null || v === undefined || isStr(v);

function isComponent(v: unknown): v is ComponentStatus {
  return isRecord(v) && typeof v.available === "boolean" && isStr(v.status) && isNullableStr(v.reason);
}

export function isErrorEnvelope(v: unknown): v is ApiErrorEnvelope {
  return isRecord(v) && isRecord(v.error) && isStr(v.error.code) && isStr(v.error.message);
}

export function isHealthResponse(v: unknown): v is HealthResponse {
  return isRecord(v) && isStr(v.status) && isStr(v.api_version) && isComponent(v.model) && isComponent(v.database);
}

export function isModelStatusResponse(v: unknown): v is ModelStatusResponse {
  if (!isRecord(v) || typeof v.available !== "boolean" || !isStr(v.status) || !isRecord(v.feature_schema)) return false;
  const cols = v.feature_schema.feature_columns;
  return Array.isArray(cols) && cols.every(isStr) && (v.metadata === undefined || isRecord(v.metadata));
}

function isContributions(v: unknown): boolean {
  return Array.isArray(v) && v.every((c) => isRecord(c) && isStr(c.feature) && isNum(c.squared_error) && isNum(c.share));
}

export function isPredictResponse(v: unknown): v is PredictResponse {
  return (
    isRecord(v) && isStr(v.request_id) && isNum(v.threshold) && isNum(v.n_samples) && typeof v.history_stored === "boolean" &&
    Array.isArray(v.warnings) && Array.isArray(v.results) &&
    v.results.every((r) => isRecord(r) && isNum(r.index) && isNum(r.reconstruction_error) &&
      typeof r.is_anomaly === "boolean" && isNum(r.threshold) && isContributions(r.top_features))
  );
}

export function isHistoryResponse(v: unknown): v is HistoryResponse {
  return (
    isRecord(v) && Array.isArray(v.items) && isNum(v.total) && isNum(v.limit) && isNum(v.offset) &&
    (v.next_before_id === null || isNum(v.next_before_id)) &&
    v.items.every((it) => isRecord(it) && isNum(it.id) && isStr(it.created_utc) && isNum(it.reconstruction_error) &&
      isNum(it.threshold) && typeof it.is_anomaly === "boolean" && isContributions(it.top_features))
  );
}

export function isMetricsResponse(v: unknown): v is MetricsResponse {
  if (!isRecord(v) || typeof v.available !== "boolean") return false;
  if (v.available === false) return isStr(v.reason);
  const r = v.report;
  return isRecord(r) && isNum(r.report_version) && isStr(r.generated_utc) && isStr(r.source) && isRecord(r.metrics);
}

// ------------------------------------------------------------------ client
export function historyQueryString(q: HistoryQuery = {}): string {
  const p = new URLSearchParams();
  if (q.limit !== undefined) p.set("limit", String(q.limit));
  if (q.offset) p.set("offset", String(q.offset));
  if (q.beforeId !== undefined) p.set("before_id", String(q.beforeId));
  if (q.anomaliesOnly) p.set("anomalies_only", "true");
  const s = p.toString();
  return s ? `?${s}` : "";
}

export function createApiClient(options: ApiClientOptions): ApiClient {
  const baseUrl = options.baseUrl.replace(/\/+$/, "");
  const doFetch: typeof fetch = options.fetchImpl ?? ((input, init) => globalThis.fetch(input, init));

  async function request<T>(
    endpoint: string,
    init: RequestInit,
    guard: (v: unknown) => v is T,
    external?: AbortSignal,
  ): Promise<T> {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, options.timeoutMs);
    const onExternalAbort = () => controller.abort();
    if (external) {
      if (external.aborted) controller.abort();
      else external.addEventListener("abort", onExternalAbort, { once: true });
    }
    const fail = (kind: "network" | "timeout" | "aborted") =>
      new ApiRequestError({
        kind: timedOut ? "timeout" : external?.aborted ? "aborted" : kind,
        endpoint,
        message: timedOut
          ? `No response from ${endpoint} within ${Math.round(options.timeoutMs / 1000)} s`
          : external?.aborted
            ? "Request cancelled"
            : `Could not reach ${baseUrl}`,
      });

    try {
      let response: Response;
      try {
        response = await doFetch(`${baseUrl}${endpoint}`, {
          ...init,
          signal: controller.signal,
          headers: { Accept: "application/json", ...(init.headers ?? {}) },
        });
      } catch {
        throw fail("network");
      }
      let text: string;
      try {
        text = await response.text();
      } catch {
        throw fail("network");
      }
      let body: unknown = undefined;
      let parsed = false;
      if (text) {
        try {
          body = JSON.parse(text);
          parsed = true;
        } catch {
          parsed = false;
        }
      }
      if (!response.ok) {
        if (parsed && isErrorEnvelope(body)) {
          throw new ApiRequestError({
            kind: "http", endpoint, status: response.status,
            code: body.error.code, message: body.error.message, details: body.error.details,
          });
        }
        throw new ApiRequestError({
          kind: "http", endpoint, status: response.status, code: "http_error",
          message: `The API answered HTTP ${response.status} without a standard error body.`,
        });
      }
      if (!parsed || !guard(body)) {
        throw new ApiRequestError({
          kind: "malformed", endpoint, status: response.status,
          message: `Response from ${endpoint} did not match the expected format.`,
        });
      }
      return body;
    } finally {
      clearTimeout(timer);
      external?.removeEventListener("abort", onExternalAbort);
    }
  }

  return {
    baseUrl,
    health: (signal) => request("/health", { method: "GET" }, isHealthResponse, signal),
    modelStatus: (signal) => request("/model/status", { method: "GET" }, isModelStatusResponse, signal),
    metrics: (signal) => request("/metrics", { method: "GET" }, isMetricsResponse, signal),
    history: (query, signal) => request(`/history${historyQueryString(query)}`, { method: "GET" }, isHistoryResponse, signal),
    predict: (body, signal) =>
      request(
        "/predict",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
        isPredictResponse,
        signal,
      ),
  };
}

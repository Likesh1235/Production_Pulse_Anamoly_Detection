/** Runtime configuration from Vite env variables (see frontend/.env.example). */

function intFromEnv(raw: string | undefined, fallback: number, min: number, max: number): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) return fallback;
  return value;
}

function normaliseBaseUrl(raw: string | undefined): string {
  const value = (raw ?? "").trim() || "http://127.0.0.1:8000";
  return value.replace(/\/+$/, "");
}

export const API_BASE_URL = normaliseBaseUrl(import.meta.env.VITE_API_BASE_URL);

/** Must mirror [api].max_batch_size (the API does not expose it). The API still enforces its own limit (HTTP 413). */
export const MAX_BATCH_SIZE = intFromEnv(import.meta.env.VITE_MAX_BATCH_SIZE, 1000, 1, 100_000);
export const REQUEST_TIMEOUT_MS = intFromEnv(import.meta.env.VITE_REQUEST_TIMEOUT_MS, 15_000, 1_000, 300_000);
export const STATUS_POLL_MS = intFromEnv(import.meta.env.VITE_STATUS_POLL_MS, 20_000, 0, 600_000);
export const APP_VERSION: string = typeof __APP_VERSION__ !== "undefined" ? __APP_VERSION__ : "0.0.0";

/** Upper bound on the number of rows held in memory for one replay/workbench session. */
export const MAX_SESSION_ROWS = 100_000;

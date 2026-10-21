/** Typed request failures. Connection problems (network/timeout) are kept separate from
 *  API answers (http) and from responses that do not match the contract (malformed). */

export type ApiFailureKind = "network" | "timeout" | "http" | "malformed" | "aborted";

export interface ApiRequestErrorInit {
  kind: ApiFailureKind;
  message: string;
  endpoint: string;
  status?: number;
  code?: string;
  details?: unknown;
}

export class ApiRequestError extends Error {
  readonly kind: ApiFailureKind;
  readonly endpoint: string;
  readonly status?: number;
  readonly code?: string;
  readonly details?: unknown;

  constructor(init: ApiRequestErrorInit) {
    super(init.message);
    this.name = "ApiRequestError";
    this.kind = init.kind;
    this.endpoint = init.endpoint;
    this.status = init.status;
    this.code = init.code;
    this.details = init.details;
  }

  /** True when the server could not be reached at all (or did not answer in time). */
  get isConnectionProblem(): boolean {
    return this.kind === "network" || this.kind === "timeout";
  }
}

export function isApiRequestError(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError;
}

export interface ErrorDescription {
  title: string;
  message: string;
  hint?: string;
  items: string[];
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Turn the API's `details` into readable lines (invalid_features / validation_error). */
export function detailLines(details: unknown): string[] {
  if (!Array.isArray(details)) return [];
  return details.slice(0, 20).flatMap((d): string[] => {
    if (!isRecord(d)) return [];
    if (typeof d.sample_index === "number") {
      const parts: string[] = [];
      if (Array.isArray(d.missing_features) && d.missing_features.length) parts.push(`missing ${d.missing_features.join(", ")}`);
      if (Array.isArray(d.unexpected_features) && d.unexpected_features.length) parts.push(`unexpected ${d.unexpected_features.join(", ")}`);
      if (Array.isArray(d.features) && d.features.length) {     // value_out_of_range
        parts.push(`${d.features.join(", ")}${typeof d.reason === "string" ? ` (${d.reason})` : ""}`);
      }
      return [`Sample ${d.sample_index + 1}: ${parts.join("; ") || "invalid features"}`];
    }
    if (typeof d.message === "string") {
      const loc = Array.isArray(d.location) ? d.location.map(String).filter((p) => p !== "body").join(" › ") : "";
      return [loc ? `${loc}: ${d.message}` : d.message];
    }
    return [];
  });
}

export function describeError(error: unknown, baseUrl: string): ErrorDescription {
  if (!isApiRequestError(error)) {
    return { title: "Unexpected error", message: error instanceof Error ? error.message : String(error), items: [] };
  }
  switch (error.kind) {
    case "network":
      return {
        title: "API unreachable",
        message: `Could not connect to the Production Pulse API at ${baseUrl}.`,
        hint: "Start the FastAPI backend, check VITE_API_BASE_URL, and make sure this page's origin is listed in [api].cors_origins.",
        items: [],
      };
    case "timeout":
      return {
        title: "Request timed out",
        message: "The API did not respond in time. No result was produced.",
        hint: "The server may be busy or stopped. Try again, or send a smaller batch.",
        items: [],
      };
    case "malformed":
      return {
        title: "Unexpected response",
        message: "The API answered, but the response did not match the expected format, so it was not used.",
        hint: "Check that the frontend and backend versions match.",
        items: [],
      };
    case "aborted":
      return { title: "Request cancelled", message: "The request was cancelled.", items: [] };
    case "http":
    default: {
      const items = detailLines(error.details);
      const hints: Record<string, string> = {
        model_unavailable: "Train and save a model with notebooks/03_autoencoder_training.ipynb, then restart the API.",
        batch_too_large: "Split the input into smaller batches (Live Monitor replays files in batches).",
        database_unavailable: "Check the API log; the history database path must be inside the project's data/ folder.",
        invalid_features: "Each sample must contain exactly the 52 TEP measurement features.",
        validation_error: "Fix the listed fields and submit again.",
        value_out_of_range: "Check the listed samples for unit or typing errors; the values are far outside any physical range.",
        internal_error: "The server logged the details. Try again or check the API console.",
      };
      const titles: Record<string, string> = {
        model_unavailable: "Model unavailable",
        batch_too_large: "Batch too large",
        database_unavailable: "History database unavailable",
        invalid_features: "Feature names rejected",
        validation_error: "Request rejected by the API",
        value_out_of_range: "Values out of range",
        internal_error: "Server error",
        not_found: "Endpoint not found",
      };
      return {
        title: titles[error.code ?? ""] ?? `HTTP ${error.status ?? "error"}`,
        message: error.message,
        hint: hints[error.code ?? ""],
        items,
      };
    }
  }
}

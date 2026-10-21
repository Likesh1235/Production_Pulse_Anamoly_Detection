/** Export returned prediction results (exactly as received from the API) to CSV. */

import type { PredictionResult } from "./api/types";
import type { ParsedSample } from "./validation";

function cell(value: unknown): string {
  if (value === null || value === undefined) return "";
  let s = String(value);
  // Text that a spreadsheet would run as a formula (CSV injection) gets a leading apostrophe.
  // Only strings are affected: sample_id / timestamp come from the user's CSV; numbers stay exact.
  if (typeof value === "string" && /^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export interface ExportRow {
  input?: ParsedSample;
  result: PredictionResult;
  requestId: string;
  modelId: string | null;
}

export function resultsToCsv(rows: ExportRow[]): string {
  const maxTop = rows.reduce((m, r) => Math.max(m, r.result.top_features.length), 0);
  const header = ["index", "sample_id", "source_row", "timestamp", "prediction_id", "request_id", "model_id",
    "reconstruction_error", "threshold", "is_anomaly"];
  for (let k = 1; k <= maxTop; k++) header.push(`top${k}_feature`, `top${k}_squared_error`, `top${k}_share`);
  const lines = [header.join(",")];
  for (const { input, result, requestId, modelId } of rows) {
    const vals: unknown[] = [result.index, result.sample_id ?? input?.sampleId ?? "", input?.sourceRow ?? "",
      input?.timestamp ?? "", result.prediction_id, requestId, modelId, result.reconstruction_error, result.threshold,
      result.is_anomaly ? "true" : "false"];
    for (let k = 0; k < maxTop; k++) {
      const c = result.top_features[k];
      vals.push(c?.feature ?? "", c?.squared_error ?? "", c?.share ?? "");
    }
    lines.push(vals.map(cell).join(","));
  }
  return lines.join("\r\n") + "\r\n";
}

export function downloadText(filename: string, text: string, mime = "text/csv;charset=utf-8"): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

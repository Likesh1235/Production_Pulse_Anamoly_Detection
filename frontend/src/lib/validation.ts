/** Client-side validation that mirrors the API contract (schemas.py / routes.py):
 *  exactly the 52 feature names (any order), finite numeric values, sample_id <= 128 chars,
 *  and the batch limit. The API re-validates everything; this only gives earlier feedback. */

import Papa from "papaparse";
import { AUX_COLUMNS, TEP_METADATA_COLUMNS, TIMESTAMP_COLUMNS } from "./features";
import type { PredictRequest } from "./api/types";

export type IssueSeverity = "error" | "warning";

export interface ValidationIssue {
  severity: IssueSeverity;
  message: string;
  /** CSV: 1-based line number in the file (header = line 1). JSON: 1-based sample number. */
  row?: number;
  feature?: string;
}

export interface ParsedSample {
  features: Record<string, number>;
  sampleId?: string;
  timestamp?: string;
  /** CSV line number or JSON sample number (1-based). */
  sourceRow: number;
}

export interface ParseResult {
  source: "csv" | "json";
  samples: ParsedSample[];
  issues: ValidationIssue[];
  errorCount: number;
  /** Non-feature columns that were accepted but are not sent as features. */
  ignoredColumns: string[];
  totalRows: number;
  hasTimestamps: boolean;
}

export const MAX_REPORTED_ISSUES = 60;
export const MAX_SAMPLE_ID_LENGTH = 128;
const NUMBER_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
const NON_FINITE_RE = /^[+-]?(?:nan|inf|infinity)$/i;

class Issues {
  readonly list: ValidationIssue[] = [];
  errors = 0;
  add(issue: ValidationIssue) {
    if (issue.severity === "error") this.errors += 1;
    if (this.list.length < MAX_REPORTED_ISSUES) this.list.push(issue);
  }
  error(message: string, extra: Partial<ValidationIssue> = {}) {
    this.add({ severity: "error", message, ...extra });
  }
  warn(message: string, extra: Partial<ValidationIssue> = {}) {
    this.add({ severity: "warning", message, ...extra });
  }
  finish(): { issues: ValidationIssue[]; errorCount: number } {
    const hidden = this.errors - this.list.filter((i) => i.severity === "error").length;
    const issues = [...this.list];
    if (hidden > 0) issues.push({ severity: "error", message: `…and ${hidden} more error(s) not shown.` });
    return { issues, errorCount: this.errors };
  }
}

export type NumericCell = { ok: true; value: number } | { ok: false; reason: "empty" | "non-numeric" | "non-finite" };

/** Strict number parsing for CSV text cells (no locale commas, no units, no NaN/Infinity). */
export function parseNumericCell(raw: unknown): NumericCell {
  const s = typeof raw === "string" ? raw.trim() : raw === undefined || raw === null ? "" : String(raw).trim();
  if (s === "") return { ok: false, reason: "empty" };
  if (NON_FINITE_RE.test(s)) return { ok: false, reason: "non-finite" };
  if (!NUMBER_RE.test(s)) return { ok: false, reason: "non-numeric" };
  const value = Number(s);
  if (!Number.isFinite(value)) return { ok: false, reason: "non-finite" }; // e.g. 1e400 overflows
  return { ok: true, value };
}

function list(names: string[], max = 8): string {
  return names.length > max ? `${names.slice(0, max).join(", ")} … (+${names.length - max})` : names.join(", ");
}

function checkNames(names: string[], featureColumns: readonly string[]): { missing: string[]; unexpected: string[] } {
  const want = new Set(featureColumns);
  const have = new Set(names);
  return {
    missing: featureColumns.filter((c) => !have.has(c)),
    unexpected: names.filter((n) => !want.has(n)),
  };
}

// ------------------------------------------------------------------ CSV
export function parseCsv(text: string, featureColumns: readonly string[]): ParseResult {
  const issues = new Issues();
  const empty = (): ParseResult => ({
    source: "csv", samples: [], ...issues.finish(), ignoredColumns: [], totalRows: 0, hasTimestamps: false,
  });
  if (!text.trim()) {
    issues.error("The file is empty.");
    return empty();
  }

  // Read the header row on its own (no reliance on parser callbacks) to detect duplicates exactly.
  const stripBom = (h: string) => h.replace(/^\uFEFF/, "").trim();
  const headerPass = Papa.parse<string[]>(text, { header: false, preview: 1, skipEmptyLines: "greedy" });
  const rawHeaders: string[] = (headerPass.data[0] ?? []).map((h) => stripBom(String(h)));
  const parsed = Papa.parse<Record<string, string>>(text, {
    header: true,
    skipEmptyLines: "greedy",
    dynamicTyping: false,
    transformHeader: (h: string) => stripBom(h),
  });

  const dupes = rawHeaders.filter((h, i) => rawHeaders.indexOf(h) !== i);
  if (dupes.length) issues.error(`Duplicate column name(s): ${list([...new Set(dupes)])}.`);
  if (rawHeaders.some((h) => h === "")) issues.error("The header row contains an empty column name.");

  const allowedExtra = new Set([...TEP_METADATA_COLUMNS, ...AUX_COLUMNS]);
  const featureLike = rawHeaders.filter((h) => !allowedExtra.has(h));
  const { missing, unexpected } = checkNames(featureLike, featureColumns);
  if (missing.length) issues.error(`Missing feature column(s): ${list(missing)}.`, { row: 1 });
  if (unexpected.length) {
    issues.error(
      `Unexpected column(s): ${list(unexpected)}. Only the 52 TEP features plus sample_id, timestamp/time/datetime, faultNumber, simulationRun and sample are allowed.`,
      { row: 1 },
    );
  }
  const ignoredColumns = rawHeaders.filter((h) => allowedExtra.has(h));
  const tsColumn = TIMESTAMP_COLUMNS.find((c) => rawHeaders.includes(c));

  for (const e of parsed.errors.slice(0, 20)) {
    const line = typeof e.row === "number" ? e.row + 2 : undefined;
    issues.error(`CSV format problem: ${e.message}.`, { row: line });
  }

  const rows = parsed.data;
  if (rows.length === 0) issues.error("The file has a header but no data rows.");
  if (missing.length || unexpected.length || dupes.length) {
    return { source: "csv", samples: [], ...issues.finish(), ignoredColumns, totalRows: rows.length, hasTimestamps: false };
  }

  const samples: ParsedSample[] = [];
  rows.forEach((row, i) => {
    const line = i + 2;
    let rowOk = true;
    const features: Record<string, number> = {};
    for (const name of featureColumns) {
      const cell = parseNumericCell(row[name]);
      if (cell.ok) {
        features[name] = cell.value;
      } else {
        rowOk = false;
        const what = cell.reason === "empty" ? "is empty" : cell.reason === "non-finite" ? "is not finite (NaN/Infinity/overflow)" : `is not a number ("${String(row[name]).slice(0, 24)}")`;
        issues.error(`${name} ${what}.`, { row: line, feature: name });
      }
    }
    let sampleId: string | undefined = row.sample_id?.trim() || undefined;
    if (!sampleId && row.simulationRun !== undefined && row.sample !== undefined) {
      const r = parseNumericCell(row.simulationRun);
      const s = parseNumericCell(row.sample);
      if (r.ok && s.ok) sampleId = `run-${r.value}-sample-${s.value}`;
    }
    if (sampleId && sampleId.length > MAX_SAMPLE_ID_LENGTH) {
      rowOk = false;
      issues.error(`sample_id is longer than ${MAX_SAMPLE_ID_LENGTH} characters.`, { row: line, feature: "sample_id" });
    }
    const timestamp = tsColumn ? row[tsColumn]?.trim() || undefined : undefined;
    if (rowOk) samples.push({ features, sampleId, timestamp, sourceRow: line });
  });

  return {
    source: "csv",
    samples,
    ...issues.finish(),
    ignoredColumns,
    totalRows: rows.length,
    hasTimestamps: Boolean(tsColumn) && samples.some((s) => s.timestamp),
  };
}

// ------------------------------------------------------------------ JSON
function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Accepts the API's formats: {"features": {...}} | {"samples": [...]} | [ {"features": {...}}, ... ]. */
export function parseJson(text: string, featureColumns: readonly string[]): ParseResult {
  const issues = new Issues();
  const done = (samples: ParsedSample[], totalRows: number): ParseResult => ({
    source: "json", samples, ...issues.finish(), ignoredColumns: [], totalRows,
    hasTimestamps: false,
  });
  if (!text.trim()) {
    issues.error("Paste JSON containing one sample or a batch of samples.");
    return done([], 0);
  }
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (e) {
    if (/(^|[\s:,[])-?(NaN|Infinity)(?=[\s,\]}]|$)/.test(text)) {
      issues.error("NaN and Infinity are not valid JSON numbers and are rejected by the API. Replace them with finite values.");
    } else {
      issues.error(`Invalid JSON: ${e instanceof Error ? e.message : String(e)}`);
    }
    return done([], 0);
  }

  let items: unknown[];
  if (Array.isArray(data)) {
    items = data;
  } else if (isPlainObject(data)) {
    if ("features" in data && "samples" in data) {
      issues.error('Send either "features" (one sample) or "samples" (a batch), not both.');
      return done([], 0);
    }
    if ("features" in data) {
      const extra = Object.keys(data).filter((k) => k !== "features" && k !== "sample_id");
      if (extra.length) issues.error(`Unexpected top-level key(s): ${list(extra)}.`);
      items = [data];
    } else if ("samples" in data) {
      const extra = Object.keys(data).filter((k) => k !== "samples");
      if (extra.length) issues.error(`Unexpected top-level key(s): ${list(extra)}.`);
      if (!Array.isArray(data.samples)) {
        issues.error('"samples" must be an array.');
        return done([], 0);
      }
      items = data.samples;
    } else {
      issues.error('Expected {"features": {...}}, {"samples": [...]} or an array of samples.');
      return done([], 0);
    }
  } else {
    issues.error('Expected {"features": {...}}, {"samples": [...]} or an array of samples.');
    return done([], 0);
  }
  if (items.length === 0) {
    issues.error("The batch contains no samples.");
    return done([], 0);
  }

  const samples: ParsedSample[] = [];
  items.forEach((item, i) => {
    const n = i + 1;
    if (!isPlainObject(item)) {
      issues.error("Each sample must be an object with a \"features\" object.", { row: n });
      return;
    }
    const extraKeys = Object.keys(item).filter((k) => k !== "features" && k !== "sample_id");
    if (extraKeys.length) issues.error(`Unexpected key(s) in sample: ${list(extraKeys)}.`, { row: n });
    const feats = item.features;
    if (!isPlainObject(feats)) {
      issues.error('"features" must be an object of feature name → number.', { row: n });
      return;
    }
    let ok = extraKeys.length === 0;
    const { missing, unexpected } = checkNames(Object.keys(feats), featureColumns);
    if (missing.length) { ok = false; issues.error(`Missing feature(s): ${list(missing)}.`, { row: n }); }
    if (unexpected.length) { ok = false; issues.error(`Unexpected feature(s): ${list(unexpected)}.`, { row: n }); }
    const features: Record<string, number> = {};
    for (const name of featureColumns) {
      if (!(name in feats)) continue;
      const v = feats[name];
      if (typeof v !== "number") {
        ok = false;
        const kind = v === null ? "null" : Array.isArray(v) ? "a list" : typeof v === "string" ? "text (send numbers, not strings)" : typeof v;
        issues.error(`${name} is ${kind}; a number is required.`, { row: n, feature: name });
      } else if (!Number.isFinite(v)) {
        ok = false;
        issues.error(`${name} is not finite.`, { row: n, feature: name });
      } else {
        features[name] = v;
      }
    }
    let sampleId: string | undefined;
    if (item.sample_id !== undefined && item.sample_id !== null) {
      if (typeof item.sample_id !== "string") { ok = false; issues.error("sample_id must be text.", { row: n }); }
      else if (item.sample_id.length > MAX_SAMPLE_ID_LENGTH) { ok = false; issues.error(`sample_id is longer than ${MAX_SAMPLE_ID_LENGTH} characters.`, { row: n }); }
      else sampleId = item.sample_id;
    }
    if (ok) samples.push({ features, sampleId, sourceRow: n });
  });
  return done(samples, items.length);
}

// ------------------------------------------------------------------ batch + request building
export function batchLimitIssue(count: number, maxBatchSize: number): ValidationIssue | null {
  if (count <= maxBatchSize) return null;
  return {
    severity: "error",
    message: `${count.toLocaleString("en-US")} samples exceed the batch limit of ${maxBatchSize.toLocaleString("en-US")} per request. Use Live Monitor to replay the file in batches, or split it.`,
  };
}

export function isSubmittable(result: ParseResult, maxBatchSize: number): boolean {
  return result.errorCount === 0 && result.samples.length > 0 && result.samples.length === result.totalRows &&
    batchLimitIssue(result.samples.length, maxBatchSize) === null;
}

/** Build the POST /predict body with features in the canonical order (input order does not matter). */
export function buildPredictRequest(samples: ParsedSample[], featureColumns: readonly string[]): PredictRequest {
  return {
    samples: samples.map((s) => {
      const features: Record<string, number> = {};
      for (const name of featureColumns) features[name] = s.features[name];
      return s.sampleId ? { features, sample_id: s.sampleId } : { features };
    }),
  };
}

export function chunk<T>(items: T[], size: number): T[][] {
  if (size < 1) throw new Error("chunk size must be >= 1");
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

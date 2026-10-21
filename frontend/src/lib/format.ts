/** Number/date formatting. null/undefined/NaN render as an em dash, never as 0. */

const DASH = "—";

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Compact scientific-aware formatting for reconstruction errors and thresholds. */
export function fmtNum(value: unknown, sig = 4): string {
  if (!isFiniteNumber(value)) return DASH;
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 1e-3 || abs >= 1e5) return value.toExponential(Math.max(sig - 1, 1));
  return Number(value.toPrecision(sig)).toString();
}

export function fmtInt(value: unknown): string {
  if (!isFiniteNumber(value)) return DASH;
  return Math.round(value).toLocaleString("en-US");
}

export function fmtPct(value: unknown, digits = 1): string {
  if (!isFiniteNumber(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtRatio(value: unknown): string {
  if (!isFiniteNumber(value)) return DASH;
  return `${value.toFixed(value >= 10 ? 1 : 2)}×`;
}

export function fmtDateTime(iso: unknown): string {
  if (typeof iso !== "string" || !iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

export function fmtRelative(iso: unknown, now: Date = new Date()): string {
  if (typeof iso !== "string") return DASH;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return DASH;
  const s = Math.round((now.getTime() - t) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export function shortId(id: unknown, n = 8): string {
  return typeof id === "string" && id ? id.slice(0, n) : DASH;
}

export function humanise(key: string): string {
  return key.replace(/_/g, " ").replace(/\b([a-z])/g, (m) => m.toUpperCase()).replace(/\bFpr\b/, "FPR")
    .replace(/\bPr Auc\b/, "PR-AUC").replace(/\bRoc Auc\b/, "ROC-AUC").replace(/\bF1\b/, "F1");
}

import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import type { ParseResult, ValidationIssue } from "../../lib/validation";
import { batchLimitIssue } from "../../lib/validation";
import { fmtInt } from "../../lib/format";

/** Pre-submission validation summary with row/feature context. */
export function ValidationReport({ result, maxBatchSize, enforceBatchLimit = true }: {
  result: ParseResult; maxBatchSize: number; enforceBatchLimit?: boolean;
}) {
  const limit = enforceBatchLimit ? batchLimitIssue(result.samples.length, maxBatchSize) : null;
  const issues: ValidationIssue[] = limit ? [...result.issues, limit] : result.issues;
  const errors = result.errorCount + (limit ? 1 : 0);
  const ok = errors === 0 && result.samples.length > 0;
  return (
    <div className="space-y-3" aria-live="polite">
      <div className="grid grid-cols-3 gap-2 text-center">
        <Stat label={result.source === "csv" ? "Data rows" : "Samples"} value={fmtInt(result.totalRows)} />
        <Stat label="Valid samples" value={fmtInt(result.samples.length)} tone={ok ? "ok" : undefined} />
        <Stat label="Errors" value={fmtInt(errors)} tone={errors ? "danger" : "ok"} />
      </div>
      {ok ? (
        <p className="flex items-center gap-2 text-xs text-ok"><CheckCircle2 className="h-4 w-4" aria-hidden />
          Input is valid: {fmtInt(result.samples.length)} sample(s) with all 52 features. Feature order does not matter — values are sent in the model's canonical order.
        </p>
      ) : null}
      {result.ignoredColumns.length > 0 && (
        <p className="text-xs text-ink-muted">Not sent as features (used for labels only): {result.ignoredColumns.map((c) => <span key={c} className="chip mr-1">{c}</span>)}</p>
      )}
      {issues.length > 0 && (
        <ul className="max-h-64 space-y-1 overflow-y-auto rounded-control border border-line bg-surface-1/60 p-2" aria-label="Validation issues">
          {issues.map((issue, i) => (
            <li key={i} className="flex items-start gap-2 text-xs">
              {issue.severity === "error"
                ? <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-danger" aria-label="Error" />
                : <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" aria-label="Warning" />}
              <span className="min-w-0 text-ink">
                {issue.row !== undefined && <span className="chip mr-1.5">{result.source === "csv" ? `line ${issue.row}` : `sample ${issue.row}`}</span>}
                {issue.message}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "ok" | "danger" }) {
  return (
    <div className="rounded-control border border-line bg-surface-1/60 px-2 py-2">
      <p className={tone === "danger" ? "num text-base font-semibold text-danger" : tone === "ok" ? "num text-base font-semibold text-ok" : "num text-base font-semibold text-ink"}>{value}</p>
      <p className="text-2xs text-ink-faint">{label}</p>
    </div>
  );
}

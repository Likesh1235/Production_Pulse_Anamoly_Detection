import { FlaskConical, NotebookPen } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { useApiClient } from "../lib/api/ApiClientContext";
import type { EvaluationReport } from "../lib/api/types";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, humanise } from "../lib/format";
import { useResource } from "../hooks/useResource";
import { useStatus } from "../state/StatusContext";
import { Badge, Banner, EmptyState, KeyValueList, MetricTile, Panel, SkeletonRows } from "../components/ui/primitives";
import { ErrorState } from "../components/ui/ErrorState";
import { DetectorComparisonChart } from "../components/charts/DetectorComparisonChart";
import { ConfusionMatrix } from "../components/evaluation/ConfusionMatrix";

type Rec = Record<string, unknown>;
const isRec = (v: unknown): v is Rec => typeof v === "object" && v !== null && !Array.isArray(v);
const records = (v: unknown): Rec[] | null => (Array.isArray(v) && v.length > 0 && v.every(isRec) ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

const METRIC_COLS = [
  { key: "precision", label: "Precision" }, { key: "recall", label: "Recall" }, { key: "f1", label: "F1" },
  { key: "false_positive_rate", label: "FPR" }, { key: "pr_auc", label: "PR-AUC" }, { key: "roc_auc", label: "ROC-AUC" },
];
const KNOWN = new Set(["setup", "training", "threshold", "clean_false_alarm_rate", "detector_comparison", "hypothesis_assessment_autoencoder", "autoencoder_minus_baselines"]);

type Reading = "autoencoder higher" | "autoencoder lower" | "no clear difference";
interface FamilyComparison { family: string; ae: number | null; other: number | null; diff: number | null; ciLow: number | null; ciHigh: number | null; reading: Reading | null }

/**
 * Autoencoder vs one baseline on PR-AUC per family, built only from what the report contains:
 * bootstrap intervals and readings when the report saved them (autoencoder_minus_baselines), otherwise
 * the raw PR-AUC values without any claim of significance.
 */
function compareWith(comparison: Rec[], diffs: Rec[] | null, baseline: string): FamilyComparison[] {
  const families = [...new Set(comparison.map((r) => String(r.family ?? "all")))];
  return families.map((family) => {
    const pr = (det: string) => num(comparison.find((r) => String(r.family ?? "all") === family && r.detector === det)?.pr_auc);
    const ae = pr("autoencoder"), other = pr(baseline);
    const d = diffs?.find((r) => r.family === family && r.metric === "pr_auc" && r.autoencoder_minus === baseline) ?? null;
    const reading = d && typeof d.reading === "string" ? (d.reading as Reading) : null;
    return { family, ae, other, diff: d ? num(d.mean) : ae !== null && other !== null ? ae - other : null,
      ciLow: d ? num(d.ci_low) : null, ciHigh: d ? num(d.ci_high) : null, reading };
  });
}

function BaselineComparison({ comparison, diffs }: { comparison: Rec[]; diffs: Rec[] | null }) {
  const detectorNames = new Set(comparison.map((r) => String(r.detector)));
  const baselines = ["pca", "rule_check_equal_fpr", "rule_check"].filter((b) => detectorNames.has(b));
  const [baseline, setBaseline] = useState(baselines[0] ?? "");
  if (!detectorNames.has("autoencoder") || !baseline) return null;
  const rows = compareWith(comparison, diffs, baseline);
  const withCi = rows.some((r) => r.reading !== null);
  const count = (reading: Reading) => rows.filter((r) => r.reading === reading).length;
  const tone = (r: FamilyComparison) => (r.reading === "autoencoder higher" ? "ok" : r.reading === "autoencoder lower" ? "danger" : "neutral") as "ok" | "danger" | "neutral";
  return (
    <Panel title="Autoencoder compared with a baseline" subtitle="PR-AUC per anomaly family, from the saved report"
      actions={baselines.length > 1 && (
        <label className="flex items-center gap-2 text-xs text-ink-muted">Baseline
          <select className="select w-auto py-1 text-xs" value={baseline} onChange={(e) => setBaseline(e.target.value)}>
            {baselines.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>)}>
      <p className="mb-3 text-[0.8125rem] text-ink">
        {withCi
          ? <>Autoencoder minus {baseline}: higher on <b className="font-semibold">{count("autoencoder higher")}</b>, no clear difference on <b className="font-semibold">{count("no clear difference")}</b>, lower on <b className="font-semibold">{count("autoencoder lower")}</b> of {rows.length} families (95 % run-level bootstrap intervals).</>
          : <>This report has no confidence intervals, so the differences below are point estimates only; small differences may not be meaningful.</>}
      </p>
      <div className="table-shell" role="region" aria-label={`Autoencoder versus ${baseline}`} tabIndex={0}>
        <table className="data-table">
          <thead><tr>
            <th scope="col">Family</th><th scope="col" className="text-right">Autoencoder PR-AUC</th><th scope="col" className="text-right">{baseline} PR-AUC</th>
            <th scope="col" className="text-right">{withCi ? "Mean difference" : "Difference"}</th>{withCi && <th scope="col" className="text-right">95 % interval</th>}{withCi && <th scope="col">Reading</th>}
          </tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.family}>
                <td className="font-mono text-2xs">{r.family}</td>
                <td className="text-right"><Value v={r.ae} /></td>
                <td className="text-right"><Value v={r.other} /></td>
                <td className="num text-right">{r.diff === null ? "—" : `${r.diff > 0 ? "+" : ""}${fmtNum(r.diff, 3)}`}</td>
                {withCi && <td className="num text-right text-ink-muted">{r.ciLow === null || r.ciHigh === null ? "—" : `${fmtNum(r.ciLow, 3)} to ${fmtNum(r.ciHigh, 3)}`}</td>}
                {withCi && <td>{r.reading ? <Badge tone={tone(r)}>{r.reading}</Badge> : "—"}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function Value({ v, pct }: { v: unknown; pct?: boolean }) {
  if (v === null || v === undefined) {
    return <span className="text-ink-faint" title="Not defined for this case (e.g. no predicted or actual positives)">undefined</span>;
  }
  if (typeof v === "number") return <span className="num">{pct ? fmtPct(v) : Number.isInteger(v) ? fmtInt(v) : fmtNum(v)}</span>;
  if (typeof v === "boolean") return <span>{v ? "yes" : "no"}</span>;
  if (typeof v === "string") return <span>{v}</span>;
  return <code className="font-mono text-2xs">{JSON.stringify(v)}</code>;
}

function objectItems(o: Rec): { label: string; value: ReactNode; mono?: boolean }[] {
  return Object.entries(o).map(([k, v]) => ({ label: humanise(k), value: <Value v={v} />, mono: typeof v === "number" }));
}

function Verdict({ v }: { v: unknown }) {
  if (v === "SUPPORTED") return <Badge tone="ok" dot>Supported</Badge>;
  if (v === "NOT SUPPORTED") return <Badge tone="warn" dot>Not supported</Badge>;
  return <Badge>{String(v ?? "—")}</Badge>;
}

function ReportView({ report }: { report: EvaluationReport }) {
  const m = report.metrics;
  const comparison = records(m.detector_comparison);
  const hypothesis = records(m.hypothesis_assessment_autoencoder);
  const families = useMemo(() => [...new Set((comparison ?? []).map((r) => String(r.family ?? "all")))], [comparison]);
  const [family, setFamily] = useState<string>(families[0] ?? "all");
  const famRows = (comparison ?? []).filter((r) => String(r.family ?? "all") === family);
  const detectors = famRows.map((r) => String(r.detector ?? "detector"));
  const [detector, setDetector] = useState<string | null>(null);
  const cmRow = famRows.find((r) => String(r.detector) === (detector ?? detectors[detectors.length - 1])) ?? null;
  const cm = cmRow && [cmRow.tp, cmRow.fp, cmRow.tn, cmRow.fn].every((x) => num(x) !== null)
    ? { tp: cmRow.tp as number, fp: cmRow.fp as number, tn: cmRow.tn as number, fn: cmRow.fn as number } : null;
  const extra = Object.keys(m).filter((k) => !KNOWN.has(k));
  const clean = num(m.clean_false_alarm_rate);
  const diffs = records(m.autoencoder_minus_baselines);

  // Prevalence of anomalous rows for the selected family, only when the report saved the counts.
  const first = famRows[0] ?? null;
  const nRows = first ? num(first.n) : null;
  const nPositive = first ? num(first.n_positive) : null;
  const prevalence = nRows !== null && nPositive !== null && nRows > 0 ? nPositive / nRows : null;

  return (
    <div className="space-y-5">
      <Banner tone="warn" title="Synthetic anomalies — not proof of real-fault detection">
        Every anomaly scored in this report was injected by a documented generator into fault-free Tennessee Eastman data.
        The numbers describe how well each detector finds those injected patterns. They do not show that real process faults would be detected.
      </Banner>
      <Panel eyebrow="GET /metrics" title={`Saved evaluation report (v${report.report_version})`}
        subtitle={`Generated ${fmtDateTime(report.generated_utc)} by ${report.source}`}
        actions={report.model_id ? <span className="chip">model {report.model_id}</span> : undefined}>
        <div className="grid items-start gap-4 md:grid-cols-3">
          {clean !== null && <MetricTile label="Clean false-alarm rate" value={fmtPct(clean, 2)}
            hint="Share of clean held-out evaluation samples flagged by the autoencoder (measured by the notebook)." />}
          {isRec(m.setup) && <div className="md:col-span-2"><p className="eyebrow mb-2">Experiment setup</p><KeyValueList columns={2} items={objectItems(m.setup)} /></div>}
        </div>
      </Panel>

      {comparison && <BaselineComparison comparison={comparison} diffs={diffs} />}

      {comparison && (
        <Panel title="Detector comparison" subtitle="Same held-out examples and labels for every detector (values exactly as saved)"
          actions={families.length > 1 && (
            <label className="flex items-center gap-2 text-xs text-ink-muted">Anomaly family
              <select className="select w-auto py-1 text-xs" value={family} onChange={(e) => { setFamily(e.target.value); setDetector(null); }}>
                {families.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            </label>)}>
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
            <div className="min-w-0 space-y-4">
              <div className="table-shell" role="region" aria-label="Detector metrics" tabIndex={0}>
                <table className="data-table">
                  <caption className="sr-only">Detector metrics for {family}</caption>
                  <thead><tr><th scope="col">Detector</th>{METRIC_COLS.map((c) => <th key={c.key} scope="col" className="text-right">{c.label}</th>)}<th scope="col" className="text-right">TP / FP / FN</th></tr></thead>
                  <tbody>
                    {famRows.map((r) => (
                      <tr key={String(r.detector)} aria-selected={cmRow === r} className="cursor-pointer" onClick={() => setDetector(String(r.detector))}>
                        <td className="font-medium"><button type="button" className="hover:text-accent" onClick={() => setDetector(String(r.detector))}>{String(r.detector)}</button></td>
                        {METRIC_COLS.map((c) => <td key={c.key} className="text-right"><Value v={r[c.key]} /></td>)}
                        <td className="num text-right text-ink-muted">{[r.tp, r.fp, r.fn].map((x) => (num(x) === null ? "—" : fmtInt(x))).join(" / ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="space-y-1 text-xs text-ink-muted" data-testid="operating-point-note">
                {prevalence !== null && (
                  <p>Anomalous rows in this family: <span className="num">{fmtPct(prevalence, 2)}</span> of {fmtInt(nRows)}. A detector that ranks rows at random has a PR-AUC of about <span className="num">{fmtNum(prevalence)}</span>, so read PR-AUC against that value.</p>
                )}
                <p>Precision, recall and F1 depend on each detector's threshold. Compare them only where the FPR column is similar; detectors with a larger FPR will find more anomalies simply by alarming more often. Accuracy is not shown because it is dominated by the normal rows.</p>
              </div>
              <DetectorComparisonChart metrics={METRIC_COLS.slice(0, 3)}
                rows={famRows.map((r) => ({ detector: String(r.detector), precision: num(r.precision), recall: num(r.recall), f1: num(r.f1) }))} />
            </div>
            <div className="panel min-w-0 p-4">
              {cm && cmRow ? <ConfusionMatrix {...cm} title={`${String(cmRow.detector)} · ${family}`} /> : <EmptyState compact title="No confusion counts in this report row" />}
              <p className="mt-3 text-2xs text-ink-faint">Select a detector row to change the matrix.</p>
            </div>
          </div>
        </Panel>
      )}

      {hypothesis && (
        <Panel title="Anomaly-family detectability (autoencoder)" subtitle="Verdicts computed by the evaluation notebook with its explicit pass criteria — shown as saved">
          <div className="table-shell" role="region" aria-label="Family detectability" tabIndex={0}>
            <table className="data-table">
              <caption className="sr-only">Hypothesis assessment per anomaly family</caption>
              <thead><tr>
                <th scope="col">Family</th><th scope="col">Verdict</th><th scope="col" className="text-right">In range</th>
                <th scope="col" className="text-right">Rule-check recall</th><th scope="col" className="text-right">Rule-check FPR</th>
                <th scope="col" className="text-right">Autoencoder recall</th><th scope="col" className="text-right">Marginal AUC excess</th><th scope="col">Reasons</th>
              </tr></thead>
              <tbody>
                {hypothesis.map((r) => (
                  <tr key={String(r.family)}>
                    <td className="font-mono text-2xs">{String(r.family)}</td>
                    <td><Verdict v={r.verdict} /></td>
                    <td className="text-right"><Value v={r.in_range_fraction} pct /></td>
                    <td className="text-right"><Value v={r.single_feature_recall} pct /></td>
                    <td className="text-right"><Value v={r.single_feature_fpr} pct /></td>
                    <td className="text-right"><Value v={r.multivariate_recall} pct /></td>
                    <td className="text-right"><Value v={r.marginal_auc_excess} /></td>
                    <td className="max-w-[280px] text-xs text-ink-muted">{typeof r.reasons === "string" && r.reasons ? r.reasons : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        {isRec(m.threshold) && <Panel title="Threshold calibration"><KeyValueList items={objectItems(m.threshold)} /></Panel>}
        {isRec(m.training) && <Panel title="Training summary"><KeyValueList items={objectItems(m.training)} /></Panel>}
      </div>

      {extra.length > 0 && (
        <Panel title="Additional report fields" subtitle="Present in the report but without a dedicated view">
          <details><summary className="cursor-pointer text-xs text-ink-muted hover:text-accent">Show {extra.length} field(s)</summary>
            <pre className="mt-2 max-h-96 overflow-auto rounded-control border border-line bg-surface-1 p-3 font-mono text-2xs">{JSON.stringify(Object.fromEntries(extra.map((k) => [k, m[k]])), null, 2)}</pre>
          </details>
        </Panel>
      )}
      {!comparison && !hypothesis && clean === null && (
        <Banner tone="neutral" title="This report has no detector comparison or detectability table">Only the fields listed above were saved.</Banner>
      )}
    </div>
  );
}

export function EvaluationPage() {
  const client = useApiClient();
  const { connection } = useStatus();
  const res = useResource((signal) => client.metrics(signal), [client], connection !== "checking");
  // While the API status is still being checked no request has been made, so do not claim "no report".
  if (res.loading || connection === "checking") return <Panel title="Evaluation"><SkeletonRows rows={6} cols={4} /></Panel>;
  if (res.error) return <ErrorState error={res.error} context="GET /metrics" onRetry={res.reload} />;
  if (!res.data || !res.data.available) {
    return (
      <Panel title="Evaluation and model performance">
        <EmptyState icon={<FlaskConical className="h-5 w-5" />} title="No evaluation report yet"
          description={<>
            <p>{res.data?.reason ?? "The API did not return a report."}</p>
            <p className="mt-2">Reports are produced by <span className="font-mono">notebooks/03_autoencoder_training.ipynb</span> (section 6b), which writes the measured detector comparison and family detectability to <span className="font-mono">reports/evaluation_metrics.json</span>. This page never shows placeholder scores.</p>
          </>}
          actions={<span className="inline-flex items-center gap-1.5 text-xs text-ink-faint"><NotebookPen className="h-3.5 w-3.5" aria-hidden />Run the notebook, then refresh</span>} />
      </Panel>
    );
  }
  return <ReportView report={res.data.report} />;
}

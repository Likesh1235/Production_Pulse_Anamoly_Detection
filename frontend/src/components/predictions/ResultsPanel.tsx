import clsx from "clsx";
import { ArrowDownWideNarrow, Download, Info, ListFilter } from "lucide-react";
import { useMemo, useState } from "react";
import { scoredRows, type SessionRun } from "../../state/SessionContext";
import { scoredTimeline } from "../../lib/chartData";
import { resultsToCsv, downloadText } from "../../lib/csvExport";
import { fmtInt, fmtNum, fmtPct, fmtRatio, shortId } from "../../lib/format";
import { AnomalyBadge, Banner, Button, MetricTile, Readouts } from "../ui/primitives";
import { Tabs } from "../ui/Tabs";
import { ErrorTimelineChart } from "../charts/ErrorTimelineChart";
import { ContributionBarChart } from "../charts/ContributionBarChart";
import { ContributionList } from "./ContributionList";
import { useToast } from "../../state/ToastContext";

type Filter = "all" | "anomaly" | "normal";

/** Per-sample results exactly as returned by POST /predict. */
export function ResultsPanel({ run }: { run: SessionRun }) {
  const rows = useMemo(() => scoredRows(run), [run]);
  const [filter, setFilter] = useState<Filter>("all");
  const [sortByError, setSortByError] = useState(false);
  const [selected, setSelected] = useState<number>(0);
  const { notify } = useToast();

  const anomalies = rows.filter((r) => r.result.is_anomaly).length;
  const thresholds = [...new Set(rows.map((r) => r.result.threshold))];
  const view = useMemo(() => {
    const indexed = rows.map((r, i) => ({ ...r, i }));
    const filtered = indexed.filter((r) => filter === "all" || (filter === "anomaly" ? r.result.is_anomaly : !r.result.is_anomaly));
    return sortByError ? [...filtered].sort((a, b) => b.result.reconstruction_error - a.result.reconstruction_error) : filtered;
  }, [rows, filter, sortByError]);
  const current = rows[selected] ?? rows[0];
  const timeline = useMemo(() => scoredTimeline(rows), [rows]);

  const exportCsv = () => {
    const csv = resultsToCsv(rows.map((r) => ({ input: r.input, result: r.result, requestId: r.result.requestId, modelId: r.result.modelId })));
    const name = `production-pulse-predictions-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    downloadText(name, csv);
    notify({ tone: "info", title: "Results exported", message: `${rows.length} returned result(s) written to ${name}` });
  };

  if (rows.length === 0) return null;
  return (
    <div className="space-y-4">
      <Readouts label="Submission summary" className="grid-cols-2 xl:grid-cols-4">
        <MetricTile cell label="Samples scored" value={fmtInt(rows.length)} />
        <MetricTile cell label="Flagged anomalous" value={fmtInt(anomalies)} tone={anomalies ? "danger" : "ok"}
          footnote={`${fmtPct(anomalies / rows.length)} of this submission`} />
        <MetricTile cell label="Threshold" value={thresholds.length === 1 ? fmtNum(thresholds[0]) : `${thresholds.length} values`} tone="neutral"
          footnote={current?.result.thresholdSource ? "calibrated (see Settings)" : undefined} />
        <MetricTile cell label="Stored in history" value={run.historyStored === false ? "No" : "Yes"} tone={run.historyStored === false ? "warn" : "ok"}
          footnote={current ? `request ${shortId(current.result.requestId)}` : undefined} />
      </Readouts>
      <p className="flex items-start gap-1.5 text-xs leading-relaxed text-ink-faint">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
        The reconstruction error is an anomaly score (mean squared error over the 52 scaled features), not a probability.
        A sample is flagged when its score is above the calibrated threshold.
      </p>
      {run.warnings.length > 0 && (
        <Banner tone="warn" title="The API returned warnings">
          <ul className="list-disc pl-4">{run.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        </Banner>
      )}

      {rows.length > 1 && (
        <div className="panel p-3">
          <ErrorTimelineChart points={timeline} title="Reconstruction error for this submission" height={220}
            xAxisLabel={run.hasTimestamps ? "Supplied timestamp" : "Sample order in submission (no timestamps supplied)"} showBrush />
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ListFilter className="h-3.5 w-3.5 text-ink-faint" aria-hidden />
          <Tabs idPrefix="result-filter" label="Filter returned results" value={filter} onChange={setFilter}
            items={[{ id: "all", label: `All (${rows.length})` }, { id: "anomaly", label: `Anomalous (${anomalies})` }, { id: "normal", label: `Normal (${rows.length - anomalies})` }]} />
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant={sortByError ? "primary" : "secondary"} aria-pressed={sortByError}
            icon={<ArrowDownWideNarrow className="h-3.5 w-3.5" aria-hidden />} onClick={() => setSortByError((v) => !v)}>Sort by error</Button>
          <Button size="sm" icon={<Download className="h-3.5 w-3.5" aria-hidden />} onClick={exportCsv}>Export CSV</Button>
        </div>
      </div>

      <div className="table-shell max-h-[420px]" role="region" aria-label="Per-sample results" tabIndex={0}>
        <table className="data-table">
          <caption className="sr-only">Prediction results; select a row to inspect its contributing features</caption>
          <thead>
            <tr>
              <th scope="col">#</th><th scope="col">Sample</th><th scope="col">Decision</th>
              <th scope="col" className="text-right">Reconstruction error</th><th scope="col" className="text-right">Threshold</th>
              <th scope="col" className="text-right">Error / thr.</th><th scope="col">Top contributor</th><th scope="col" className="text-right">Record</th>
            </tr>
          </thead>
          <tbody>
            {view.map(({ input, result, i }) => (
              <tr key={i} aria-selected={i === selected} className={clsx("cursor-pointer", result.is_anomaly && "bg-danger/[0.04]")} onClick={() => setSelected(i)}>
                <td className="num text-ink-faint">
                  <button type="button" className="rounded px-1 hover:text-accent" onClick={() => setSelected(i)} aria-label={`Inspect sample ${i + 1}`}>{i + 1}</button>
                </td>
                <td className="max-w-[180px] truncate font-mono text-2xs">{result.sample_id ?? input.sampleId ?? `line ${input.sourceRow}`}</td>
                <td><AnomalyBadge isAnomaly={result.is_anomaly} /></td>
                <td className={clsx("num text-right", result.is_anomaly ? "text-danger" : "text-ink")}>{fmtNum(result.reconstruction_error)}</td>
                <td className="num text-right text-ink-muted">{fmtNum(result.threshold)}</td>
                <td className="num text-right">{fmtRatio(result.reconstruction_error / result.threshold)}</td>
                <td className="font-mono text-2xs">{result.top_features[0]?.feature ?? "—"} <span className="text-ink-faint">{fmtPct(result.top_features[0]?.share)}</span></td>
                <td className="num text-right text-ink-faint">{result.prediction_id !== null ? `#${result.prediction_id}` : "not stored"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {view.length === 0 && <p className="p-4 text-center text-xs text-ink-muted">No returned results match this filter.</p>}
      </div>

      {current && (
        <div className="panel grid gap-4 p-4 md:grid-cols-2">
          <div className="min-w-0">
            <p className="eyebrow truncate" title={current.result.sample_id ?? current.input.sampleId ?? undefined}>Sample {selected + 1}: <span className="font-mono">{current.result.sample_id ?? current.input.sampleId ?? `line ${current.input.sourceRow}`}</span></p>
            <p className="mt-1 flex flex-wrap items-center gap-2 text-sm">
              <AnomalyBadge isAnomaly={current.result.is_anomaly} />
              <span className="num">error {fmtNum(current.result.reconstruction_error)}</span>
              <span className="text-ink-faint">vs threshold</span><span className="num">{fmtNum(current.result.threshold)}</span>
            </p>
            <ErrorGauge error={current.result.reconstruction_error} threshold={current.result.threshold} />
            <div className="mt-4"><ContributionList items={current.result.top_features} /></div>
            <p className="mt-3 text-2xs leading-relaxed text-ink-faint">
              Shares are each feature's part of this sample's total squared reconstruction error (top {current.result.top_features.length} returned by the API).
              They show where the reconstruction deviates — not a verified root cause.
            </p>
          </div>
          <ContributionBarChart valueLabel="Squared error" data={current.result.top_features.map((c) => ({ feature: c.feature, value: c.squared_error, detail: `${fmtPct(c.share)} of total` }))} />
        </div>
      )}
    </div>
  );
}

/** Error relative to the alarm threshold on one horizontal scale (threshold marked in amber). */
function ErrorGauge({ error, threshold }: { error: number; threshold: number }) {
  const max = Math.max(error, threshold) * 1.25 || 1;
  const pct = (v: number) => `${Math.min(100, (v / max) * 100)}%`;
  const over = error > threshold;
  return (
    <div className="mt-3" aria-hidden>
      <div className="relative h-2 rounded-full bg-surface-3">
        <div className={clsx("absolute inset-y-0 left-0 rounded-full", over ? "bg-danger" : "bg-ok")} style={{ width: pct(error) }} />
        <div className="absolute -inset-y-1 w-0.5 bg-warn" style={{ left: pct(threshold) }} title="Calibrated threshold" />
      </div>
      <div className="mt-1 flex justify-between text-2xs text-ink-faint">
        <span>0</span><span className="text-warn">threshold {fmtNum(threshold)}</span><span>{fmtNum(max)}</span>
      </div>
    </div>
  );
}

import { Activity, ArrowRight, CheckCircle2, CircleDashed, Database, FlaskConical, Gauge, History as HistoryIcon, Radar, Workflow } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useApiClient } from "../lib/api/ApiClientContext";
import { aggregateContributions, anomalyRate, historyTimeline } from "../lib/chartData";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, fmtRelative, shortId } from "../lib/format";
import { MODEL_STATUS_LABELS, type MetricsResponse } from "../lib/api/types";
import { useResource } from "../hooks/useResource";
import { useStatus } from "../state/StatusContext";
import { ErrorTimelineChart } from "../components/charts/ErrorTimelineChart";
import { ContributionBarChart } from "../components/charts/ContributionBarChart";
import { ModelInfoPanel } from "../components/model/ModelInfoPanel";
import { AnomalyBadge, Banner, EmptyState, MetricTile, Panel, Readouts, SkeletonRows } from "../components/ui/primitives";
import { ErrorState } from "../components/ui/ErrorState";

const RECENT_LIMIT = 200; // <= API history_max_page_size (500)

export function OverviewPage() {
  const client = useApiClient();
  const status = useStatus();
  const online = status.connection === "online";
  const dbOk = status.databaseAvailable;
  const [windowSize, setWindowSize] = useState(100);

  const recent = useResource((signal) => client.history({ limit: RECENT_LIMIT }, signal), [client, status.lastChecked?.getTime()], online && dbOk);
  const flagged = useResource((signal) => client.history({ limit: 1, anomaliesOnly: true }, signal), [client, status.lastChecked?.getTime()], online && dbOk);
  const metrics = useResource<MetricsResponse>((signal) => client.metrics(signal), [client], online);

  const items = recent.data?.items ?? [];
  const total = recent.data?.total ?? null;
  const flaggedTotal = flagged.data?.total ?? null;
  const rate = total !== null && flaggedTotal !== null ? anomalyRate(flaggedTotal, total) : null;
  const timeline = useMemo(() => historyTimeline(items).slice(-windowSize), [items, windowSize]);
  const contributors = useMemo(() => aggregateContributions(items.map((i) => i.top_features)).filter((c) => c.topOneCount > 0).slice(0, 8), [items]);
  const report = metrics.data && metrics.data.available ? metrics.data.report : null;

  const steps = [
    { done: online, label: "API reachable", detail: online ? `API v${status.health?.api_version}` : "Start the FastAPI backend", to: "/settings" },
    { done: status.modelReady, label: "Trained model loaded", detail: status.health ? (status.modelReady ? "Ready" : MODEL_STATUS_LABELS[status.health.model.status] ?? status.health.model.status) : "Unknown", to: "/settings" },
    { done: (total ?? 0) > 0, label: "Predictions stored", detail: total === null ? "Unknown" : `${fmtInt(total)} record(s)`, to: "/predictions" },
    { done: Boolean(report), label: "Evaluation report saved", detail: report ? fmtRelative(report.generated_utc) : "Run notebook 03 §6b", to: "/evaluation" },
  ];

  return (
    <div className="space-y-5">
      {status.connection === "offline" && (
        <Banner tone="danger" title="API unreachable">Nothing on this page can be shown until the API answers. See Settings for the configured base URL.</Banner>
      )}

      <Readouts label="System readouts" className="grid-cols-2 md:grid-cols-3 xl:grid-cols-6">
        <MetricTile cell label="API" loading={status.connection === "checking"}
          value={online ? "Online" : "Offline"} tone={online ? "ok" : "danger"} footnote={online ? `GET /health · v${status.health?.api_version}` : status.healthError?.message} />
        <MetricTile cell label="Model" loading={status.connection === "checking"}
          value={status.health ? (status.modelReady ? "Ready" : MODEL_STATUS_LABELS[status.health.model.status] ?? status.health.model.status) : "Unknown"}
          tone={status.modelReady ? "ok" : status.health ? "warn" : "neutral"} footnote={status.modelStatus?.model_id ? `id ${status.modelStatus.model_id}` : undefined} />
        <MetricTile cell label="Stored predictions" loading={recent.loading}
          value={total === null ? "—" : fmtInt(total)} hint="Total rows in the API's prediction-history database (GET /history total)."
          footnote={!dbOk && online ? "history database unavailable" : "all records in API history"} />
        <MetricTile cell label="Flagged anomalous" loading={flagged.loading}
          value={flaggedTotal === null ? "—" : fmtInt(flaggedTotal)} tone={flaggedTotal ? "danger" : "neutral"}
          hint="Rows in API history with is_anomaly = true, and their share of all stored rows." footnote={rate !== null ? `${fmtPct(rate)} of stored records` : undefined} />
        <MetricTile cell label="Calibrated threshold" value={status.threshold !== null ? fmtNum(status.threshold) : "—"}
          hint={status.thresholdSource ?? "Available once a calibrated model is loaded."} footnote={status.thresholdSource ? "from model metadata" : "no calibrated model"} />
        <MetricTile cell label="Last prediction" loading={recent.loading}
          value={items[0] ? fmtRelative(items[0].created_utc) : "—"} footnote={items[0] ? fmtDateTime(items[0].created_utc) : "no stored predictions"} />
      </Readouts>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Panel title="Reconstruction error — stored predictions" eyebrow="GET /history"
          subtitle={items.length ? `Latest ${fmtInt(timeline.length)} of ${fmtInt(total ?? items.length)} stored records, oldest → newest` : "From the API's prediction history"}
          actions={items.length > 0 && (
            <label className="flex items-center gap-2 text-xs text-ink-muted">Window
              <select className="select w-auto py-1 text-xs" value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value))} aria-label="Record window">
                {[50, 100, 200].map((w) => <option key={w} value={w}>{w} records</option>)}
              </select>
            </label>)}>
          {recent.loading ? <SkeletonRows rows={8} cols={1} /> : recent.error ? (
            <ErrorState error={recent.error} context="GET /history" onRetry={recent.reload} />
          ) : !online ? (
            <EmptyState icon={<Activity className="h-5 w-5" />} title="Waiting for the API" description="Stored predictions appear once the API is reachable." />
          ) : !dbOk ? (
            <EmptyState icon={<Database className="h-5 w-5" />} title="Prediction history unavailable" description={status.health?.database.reason ?? undefined} />
          ) : timeline.length ? (
            <ErrorTimelineChart points={timeline} title="Stored predictions" height={320} showBrush
              xAxisLabel="Stored prediction record ID (chronological)" />
          ) : (
            <EmptyState icon={<Workflow className="h-5 w-5" />} title="No predictions stored yet"
              description="Score samples in the Prediction workbench or replay a held-out file in Live Monitor. Every API prediction is stored and appears here."
              actions={<><Link to="/predictions" className="btn-primary">Open workbench</Link><Link to="/monitor" className="btn-secondary">Live Monitor</Link></>} />
          )}
        </Panel>

        <Panel title="Current status" subtitle="Checked against the API on every refresh">
          <p className="mb-3 text-xs leading-relaxed text-ink-muted">
            {status.modelReady && status.threshold !== null
              ? <>The autoencoder scores each sample by its mean squared reconstruction error over the 52 scaled features. Samples above <span className="tabular text-ink">{fmtNum(status.threshold)}</span> are flagged{status.thresholdSource ? <> ({status.thresholdSource})</> : null}. The score is not a probability of a fault.</>
              : "Scores appear once the API reports a loaded, calibrated model."}
          </p>
          <ol className="divide-y divide-line overflow-hidden rounded-control border border-line">
            {steps.map((s) => (
              <li key={s.label}>
                <Link to={s.to} className="flex items-center gap-3 px-3 py-2 transition-colors hover:bg-surface-3/60">
                  {s.done ? <CheckCircle2 className="h-4 w-4 shrink-0 text-ok" aria-label="done" /> : <CircleDashed className="h-4 w-4 shrink-0 text-ink-faint" aria-label="not done" />}
                  <span className="min-w-0 flex-1"><span className="block text-[0.8125rem] text-ink">{s.label}</span><span className="block truncate text-xs text-ink-faint">{s.detail}</span></span>
                  <ArrowRight className="h-3.5 w-3.5 text-ink-faint" aria-hidden />
                </Link>
              </li>
            ))}
          </ol>
        </Panel>
      </div>

      <div className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
        <Panel title="Recent predictions" subtitle="Newest stored records" bodyClassName="p-0"
          actions={<Link to="/history" className="link text-xs">All history</Link>}>
          {items.length ? (
            <ul className="divide-y divide-line">
              {items.slice(0, 8).map((it) => (
                <li key={it.id} className="flex items-center gap-3 px-4 py-2 text-xs">
                  <AnomalyBadge isAnomaly={it.is_anomaly} />
                  <span className="num text-ink-faint">#{it.id}</span>
                  <span className="min-w-0 flex-1 truncate font-mono text-[0.6875rem] text-ink-muted">{it.sample_id ?? `req ${shortId(it.request_id)} · sample ${it.sample_index + 1}`}</span>
                  <span className={it.is_anomaly ? "num text-danger" : "num"}>{fmtNum(it.reconstruction_error)}</span>
                  <span className="hidden w-16 text-right text-ink-faint sm:inline">{fmtRelative(it.created_utc)}</span>
                </li>
              ))}
            </ul>
          ) : <div className="p-4"><EmptyState compact title="No predictions yet" description="Stored predictions will be listed here." /></div>}
        </Panel>
        <Panel title="Top contributing sensors" subtitle={items.length ? `Times ranked #1 across the latest ${fmtInt(items.length)} stored records` : "From stored predictions"}
          actions={<Link to="/sensors" className="link inline-flex items-center gap-1 text-xs"><Radar className="h-3 w-3" aria-hidden />Explore</Link>}>
          {contributors.length ? (
            <ContributionBarChart valueLabel="Times ranked #1" data={contributors.map((c) => ({ feature: c.feature, value: c.topOneCount, detail: `in a top list ${c.appearances}×` }))} />
          ) : <EmptyState compact title="No contributions yet" />}
        </Panel>
        <div className="space-y-5 lg:col-span-2 xl:col-span-1">
          <ModelInfoPanel compact />
          <Panel title="Latest evaluation" eyebrow="GET /metrics" subtitle="Synthetic anomalies only" actions={<Link to="/evaluation" className="link inline-flex items-center gap-1 text-xs"><FlaskConical className="h-3 w-3" aria-hidden />Open</Link>}>
            {metrics.loading || status.connection === "checking" ? <SkeletonRows rows={3} cols={2} /> : metrics.error ? <ErrorState error={metrics.error} context="GET /metrics" onRetry={metrics.reload} />
              : report ? (
                <div className="space-y-1 text-xs text-ink-muted">
                  <p>Report generated <span className="text-ink">{fmtDateTime(report.generated_utc)}</span></p>
                  <p>Source <span className="font-mono text-ink">{report.source}</span></p>
                  {typeof report.metrics.clean_false_alarm_rate === "number" && (
                    <p>Clean false-alarm rate <span className="num text-ink">{fmtPct(report.metrics.clean_false_alarm_rate, 2)}</span></p>
                  )}
                </div>
              ) : <EmptyState compact icon={<FlaskConical className="h-5 w-5" />} title="No evaluation report yet"
                description={metrics.data && !metrics.data.available ? metrics.data.reason : "The API is not reachable."} />}
          </Panel>
        </div>
      </div>
    </div>
  );
}

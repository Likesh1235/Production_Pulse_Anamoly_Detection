import { Activity, AlertTriangle, Pause, Play, Square } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { useApiClient } from "../lib/api/ApiClientContext";
import { isApiRequestError } from "../lib/api/errors";
import { MAX_BATCH_SIZE, MAX_SESSION_ROWS } from "../lib/config";
import { aggregateContributions, scoredTimeline, sensorRows } from "../lib/chartData";
import { fmtInt, fmtNum, fmtRatio } from "../lib/format";
import { buildPredictRequest, chunk, parseCsv } from "../lib/validation";
import { newRunId, scoredRows, useSession, type SessionResult, type SessionRun } from "../state/SessionContext";
import { useStatus } from "../state/StatusContext";
import { useToast } from "../state/ToastContext";
import { ModelReadinessBanner } from "../components/model/ModelReadinessBanner";
import { CsvDropZone, type LoadedFile } from "../components/predictions/CsvDropZone";
import { ValidationReport } from "../components/predictions/ValidationReport";
import { ErrorTimelineChart } from "../components/charts/ErrorTimelineChart";
import { ContributionBarChart } from "../components/charts/ContributionBarChart";
import { SensorTrendChart } from "../components/charts/SensorTrendChart";
import { AnomalyBadge, Badge, Banner, Button, EmptyState, MetricTile, Panel, ProgressBar, Readouts } from "../components/ui/primitives";
import { ErrorState } from "../components/ui/ErrorState";
import { FeaturePicker } from "../components/ui/FeaturePicker";

type Phase = "idle" | "running" | "paused" | "complete" | "stopped" | "failed";
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const WINDOWS = [200, 500, 1000, 2000];

export function LiveMonitorPage() {
  const client = useApiClient();
  const status = useStatus();
  const session = useSession();
  const { notify } = useToast();
  const [file, setFile] = useState<LoadedFile | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [batchSize, setBatchSize] = useState(Math.min(250, MAX_BATCH_SIZE));
  const [paceMs, setPaceMs] = useState(500);
  const [windowSize, setWindowSize] = useState(500);
  const [sensors, setSensors] = useState<string[]>(["xmeas_7", "xmeas_9"]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<unknown>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const pausedRef = useRef(false);
  const stopRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const parsed = useMemo(() => (file ? parseCsv(file.text, status.featureColumns) : null), [file, status.featureColumns]);
  const run = session.runs.find((r) => r.id === runId) ?? null;
  const rows = useMemo(() => (run ? scoredRows(run) : []), [run]);
  const scored = rows.length;
  const total = run?.inputs.length ?? parsed?.samples.length ?? 0;
  const tooLarge = (parsed?.samples.length ?? 0) > MAX_SESSION_ROWS;
  const canStart = Boolean(parsed && parsed.errorCount === 0 && parsed.samples.length > 0 && parsed.samples.length === parsed.totalRows)
    && status.modelReady && !tooLarge && (phase === "idle" || phase === "complete" || phase === "stopped" || phase === "failed");

  const windowRows = rows.slice(-windowSize);
  const windowStart = rows.length - windowRows.length;
  const timeline = useMemo(() => scoredTimeline(windowRows, windowStart), [windowRows, windowStart]);
  const anomalies = rows.filter((r) => r.result.is_anomaly);
  const contributions = useMemo(() => aggregateContributions(windowRows.map((r) => r.result.top_features)).slice(0, 10), [windowRows]);
  const sensorData = useMemo(() => sensorRows(windowRows.map((r) => r.input), sensors, windowStart), [windowRows, sensors, windowStart]);

  const start = async () => {
    if (!parsed || !canStart) return;
    const samples = parsed.samples;
    const id = newRunId();
    const newRun: SessionRun = {
      id, source: "replay", label: `Replay · ${file?.name ?? "file"}`, createdAt: new Date().toISOString(), inputs: samples,
      results: samples.map(() => undefined), hasTimestamps: parsed.hasTimestamps, status: "running", warnings: [], fileName: file?.name,
    };
    session.addRun(newRun);
    setRunId(id);
    setError(null);
    stopRef.current = false;
    pausedRef.current = false;
    setPhase("running");
    const batches = chunk(samples.map((_, i) => i), batchSize);
    let flagged = 0;
    for (let b = 0; b < batches.length; b++) {
      while (pausedRef.current && !stopRef.current) await sleep(150);
      if (stopRef.current) {
        session.updateRun(id, (r) => ({ ...r, status: "stopped" }));
        setPhase("stopped");
        return;
      }
      const idx = batches[b];
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const res = await client.predict(buildPredictRequest(idx.map((i) => samples[i]), status.featureColumns), controller.signal);
        flagged += res.n_anomalies;
        session.updateRun(id, (r) => {
          const results = [...r.results];
          for (const x of res.results) {
            const target = idx[x.index];
            if (target !== undefined) results[target] = { ...x, requestId: res.request_id, modelId: res.model_id, thresholdSource: res.threshold_source } as SessionResult;
          }
          return {
            ...r, results,
            warnings: [...new Set([...r.warnings, ...res.warnings])],
            historyStored: (r.historyStored ?? true) && res.history_stored,
          };
        });
      } catch (e) {
        if (isApiRequestError(e) && e.kind === "aborted") {
          session.updateRun(id, (r) => ({ ...r, status: "stopped" }));
          setPhase("stopped");
          return;
        }
        setError(e);
        session.updateRun(id, (r) => ({ ...r, status: "failed" }));
        setPhase("failed");
        if (isApiRequestError(e) && (e.isConnectionProblem || e.code === "model_unavailable")) void status.refresh();
        return;
      } finally {
        abortRef.current = null;
      }
      if (paceMs > 0 && b < batches.length - 1) await sleep(paceMs);
    }
    session.updateRun(id, (r) => ({ ...r, status: "complete" }));
    setPhase("complete");
    notify({ tone: flagged ? "danger" : "ok", title: "Replay finished", message: `${fmtInt(samples.length)} samples scored by the API in ${batches.length} batch(es); ${fmtInt(flagged)} flagged.` });
  };

  const pause = () => { pausedRef.current = true; setPhase("paused"); session.updateRun(runId ?? "", (r) => ({ ...r, status: "paused" })); };
  const resume = () => { pausedRef.current = false; setPhase("running"); session.updateRun(runId ?? "", (r) => ({ ...r, status: "running" })); };
  const stop = () => { stopRef.current = true; pausedRef.current = false; abortRef.current?.abort(); };

  return (
    <div className="space-y-5">
      <ModelReadinessBanner />
      <Banner tone="info" title="User-initiated replay — not a continuous plant stream">
        The current API scores samples on request. This workspace replays a CSV file that you supply, sending it to POST /predict in batches.
        Every batch is scored by the API and stored in its history. No data is bundled with the dashboard.
      </Banner>

      <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <Panel title="Replay source" subtitle="Held-out CSV with the 52 features">
          <div className="space-y-4">
            <CsvDropZone file={file} disabled={phase === "running" || phase === "paused"}
              onLoaded={(f) => { setFile(f); setFileError(null); setPhase("idle"); setError(null); }} onError={setFileError} />
            {fileError && <p role="alert" className="text-xs text-danger">{fileError}</p>}
            {parsed && <ValidationReport result={parsed} maxBatchSize={MAX_BATCH_SIZE} enforceBatchLimit={false} />}
            {tooLarge && <p className="text-xs text-danger">This file has more than {fmtInt(MAX_SESSION_ROWS)} rows; split it to keep the browser responsive.</p>}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="batch-size" className="field-label">Batch size</label>
                <input id="batch-size" type="number" className="input num" min={1} max={MAX_BATCH_SIZE} value={batchSize}
                  disabled={phase === "running" || phase === "paused"}
                  onChange={(e) => setBatchSize(Math.max(1, Math.min(MAX_BATCH_SIZE, Math.floor(Number(e.target.value) || 1))))} />
                <p className="mt-1 text-2xs text-ink-faint">max {fmtInt(MAX_BATCH_SIZE)} (API limit)</p>
              </div>
              <div>
                <label htmlFor="pace" className="field-label">Pause between batches</label>
                <select id="pace" className="select" value={paceMs} onChange={(e) => setPaceMs(Number(e.target.value))}>
                  <option value={0}>none</option><option value={250}>0.25 s</option><option value={500}>0.5 s</option>
                  <option value={1000}>1 s</option><option value={2000}>2 s</option>
                </select>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {(phase === "running") ? (
                <Button icon={<Pause className="h-3.5 w-3.5" aria-hidden />} onClick={pause}>Pause</Button>
              ) : phase === "paused" ? (
                <Button variant="primary" icon={<Play className="h-3.5 w-3.5" aria-hidden />} onClick={resume}>Resume</Button>
              ) : (
                <Button variant="primary" icon={<Play className="h-3.5 w-3.5" aria-hidden />} onClick={() => void start()} disabled={!canStart}>
                  Start replay
                </Button>
              )}
              <Button variant="danger" icon={<Square className="h-3.5 w-3.5" aria-hidden />} onClick={stop} disabled={phase !== "running" && phase !== "paused"}>Stop</Button>
            </div>
            {!status.modelReady && <p className="text-xs text-ink-faint">Replay is disabled until the API reports a ready model.</p>}
            {run && <ProgressBar value={scored} max={total} label={`${fmtInt(scored)} / ${fmtInt(total)} samples scored`} />}
            {run && <p className="text-2xs text-ink-faint">Status: <Badge tone={phase === "failed" ? "danger" : phase === "complete" ? "ok" : "neutral"}>{phase}</Badge></p>}
          </div>
        </Panel>

        <div className="min-w-0 space-y-5">
          {error !== null && <ErrorState error={error} context="Replay stopped" />}
          <Readouts label="Replay summary" className="grid-cols-2 lg:grid-cols-4">
            <MetricTile cell label="Scored (this replay)" value={fmtInt(scored)} footnote={run ? run.label : "no replay yet"} />
            <MetricTile cell label="Flagged anomalous" value={fmtInt(anomalies.length)} tone={anomalies.length ? "danger" : "neutral"}
              footnote={scored ? `${((anomalies.length / scored) * 100).toFixed(1)}% of scored samples` : undefined} />
            <MetricTile cell label="Threshold" value={rows.length ? fmtNum(rows[rows.length - 1].result.threshold) : "—"} footnote="as returned by the API" />
            <MetricTile cell label="History" value={run ? (run.historyStored === false ? "Not stored" : "Stored") : "—"} tone={run?.historyStored === false ? "warn" : "neutral"} />
          </Readouts>

          <Panel title="Reconstruction error vs calibrated threshold" subtitle={run ? `${run.label}: last ${fmtInt(windowRows.length)} scored samples` : "Start a replay to see API results"}
            actions={
              <label className="flex items-center gap-2 text-xs text-ink-muted">Window
                <select className="select w-auto py-1 text-xs" value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value))} aria-label="Sample window">
                  {WINDOWS.map((w) => <option key={w} value={w}>{w}</option>)}
                </select>
              </label>}>
            {timeline.length ? (
              <ErrorTimelineChart points={timeline} title="Replay reconstruction error" height={300} showBrush
                xAxisLabel={run?.hasTimestamps ? "Supplied timestamp" : "Sample order in the replayed file (no timestamps supplied)"} />
            ) : <EmptyState icon={<Activity className="h-5 w-5" />} title="No replayed samples yet"
                description="Choose a held-out CSV, check validation, and start the replay. Points appear only as the API returns scores." />}
          </Panel>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel title="Anomaly events" subtitle="Most recent samples above the threshold in this replay">
              {anomalies.length ? (
                <ul className="max-h-72 space-y-1.5 overflow-y-auto" aria-label="Anomaly events">
                  {anomalies.slice(-12).reverse().map(({ input, result }) => (
                    <li key={`${result.requestId}-${result.index}`} className="flex items-center gap-3 rounded-control border border-danger/30 bg-danger/[0.06] px-3 py-2 text-xs">
                      <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-danger" aria-hidden />
                      <span className="min-w-0 flex-1 truncate font-mono">{result.sample_id ?? input.sampleId ?? `line ${input.sourceRow}`}</span>
                      <span className="num text-danger">{fmtNum(result.reconstruction_error)}</span>
                      <span className="num text-ink-faint">{fmtRatio(result.reconstruction_error / result.threshold)}</span>
                      <span className="chip">{result.top_features[0]?.feature ?? "—"}</span>
                    </li>
                  ))}
                </ul>
              ) : <EmptyState compact title={scored ? "No anomalies flagged so far" : "No events yet"} description={scored ? "Every scored sample is at or below the threshold." : undefined} />}
            </Panel>
            <Panel title="Strongest contributors" subtitle={`Times ranked #1 among the API's top-${rows[0]?.result.top_features.length ?? 5} features (current window)`}>
              {contributions.length ? (
                <ContributionBarChart valueLabel="Times ranked #1"
                  data={contributions.filter((c) => c.topOneCount > 0).map((c) => ({ feature: c.feature, value: c.topOneCount, detail: `in top list ${c.appearances}×, mean share ${(c.meanShare * 100).toFixed(1)}%` }))} />
              ) : <EmptyState compact title="No contributions yet" />}
            </Panel>
          </div>

          <Panel title="Selected sensors (raw values from your file)" subtitle="Same window as the error chart">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
              {sensorData.length && sensors.length ? (
                <SensorTrendChart rows={sensorData} features={sensors} yAxisLabel="Raw value (as supplied)"
                  xAxisLabel={run?.hasTimestamps ? "Supplied timestamp" : "Sample order (no timestamps supplied)"} />
              ) : <EmptyState compact title="Nothing to plot" description="Select sensors and replay a file." />}
              <FeaturePicker features={status.featureColumns} selected={sensors} onChange={setSensors} max={4} />
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

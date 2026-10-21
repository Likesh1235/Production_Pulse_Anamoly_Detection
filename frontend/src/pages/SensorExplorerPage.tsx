import { Radar } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { aggregateContributions, scoredTimeline, sensorRows } from "../lib/chartData";
import { fmtInt } from "../lib/format";
import { useSession, type SessionResult } from "../state/SessionContext";
import type { ParsedSample } from "../lib/validation";
import { useStatus } from "../state/StatusContext";
import { ErrorTimelineChart } from "../components/charts/ErrorTimelineChart";
import { SensorTrendChart } from "../components/charts/SensorTrendChart";
import { ContributionBarChart } from "../components/charts/ContributionBarChart";
import { Banner, Button, EmptyState, Panel } from "../components/ui/primitives";
import { FeaturePicker } from "../components/ui/FeaturePicker";
import { Tabs } from "../components/ui/Tabs";

type Display = "raw" | "zscore";
type Rank = "top1" | "share";
const WINDOW_SIZES = [100, 250, 500, 1000, 2000];

export function SensorExplorerPage() {
  const { runs, activeRun, setActiveRun } = useSession();
  const { featureColumns } = useStatus();
  const [selected, setSelected] = useState<string[]>(["xmeas_7", "xmeas_13", "xmeas_16"]);
  const [display, setDisplay] = useState<Display>("raw");
  const [rank, setRank] = useState<Rank>("top1");
  const [windowSize, setWindowSize] = useState(500);
  const [start, setStart] = useState(0);

  const run = activeRun;
  const n = run?.inputs.length ?? 0;
  const maxStart = Math.max(0, n - windowSize);
  useEffect(() => { setStart((s) => Math.min(s, maxStart)); }, [maxStart]);

  const inputs = useMemo(() => (run ? run.inputs.slice(start, start + windowSize) : []), [run, start, windowSize]);
  const scoredInWindow = useMemo(() => {
    if (!run) return [];
    const out: { input: ParsedSample; result: SessionResult; i: number }[] = [];
    for (let i = start; i < Math.min(start + windowSize, n); i++) {
      const r = run.results[i];
      if (r) out.push({ input: run.inputs[i], result: r, i });
    }
    return out;
  }, [run, start, windowSize, n]);
  const sensorData = useMemo(() => sensorRows(inputs, selected, start, display === "zscore"), [inputs, selected, start, display]);
  const timeline = useMemo(() => scoredTimeline(scoredInWindow.map(({ input, result }) => ({ input, result })), 0)
    .map((p, k) => ({ ...p, n: scoredInWindow[k].i + 1, xLabel: scoredInWindow[k].input.timestamp ?? String(scoredInWindow[k].i + 1) })), [scoredInWindow]);
  const ranking = useMemo(() => aggregateContributions(scoredInWindow.map((s) => s.result.top_features)), [scoredInWindow]);
  const topK = scoredInWindow[0]?.result.top_features.length ?? 0;

  if (!run) {
    return (
      <Panel title="Sensor explorer">
        <EmptyState icon={<Radar className="h-5 w-5" />} title="No session data to explore"
          description="This page analyses samples you scored in this browser session — it never loads bundled or synthetic data. Score a file in the Prediction workbench or replay one in Live Monitor first."
          actions={<><Link to="/predictions" className="btn-primary">Open workbench</Link><Link to="/monitor" className="btn-secondary">Open Live Monitor</Link></>} />
      </Panel>
    );
  }

  const xLabel = run.hasTimestamps ? "Supplied timestamp" : "Sample order in the source file (no timestamps supplied)";
  return (
    <div className="space-y-5">
      <Panel title="Data source" subtitle="Samples scored in this browser session (not persisted)"
        actions={
          <label className="flex items-center gap-2 text-xs text-ink-muted">Session run
            <select className="select w-auto py-1 text-xs" value={run.id} onChange={(e) => setActiveRun(e.target.value)} aria-label="Session run">
              {runs.map((r) => <option key={r.id} value={r.id}>{r.label} · {fmtInt(r.inputs.length)} rows · {r.source}</option>)}
            </select>
          </label>}>
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
          <div>
            <label htmlFor="window-start" className="field-label">Window start: sample {fmtInt(start + 1)} – {fmtInt(Math.min(start + windowSize, n))} of {fmtInt(n)}</label>
            <input id="window-start" type="range" min={0} max={maxStart} step={1} value={start} disabled={maxStart === 0}
              onChange={(e) => setStart(Number(e.target.value))} className="w-full accent-[rgb(var(--c-accent))]" />
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-xs text-ink-muted">Window size
              <select className="select mt-1 w-28 py-1 text-xs" value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value))}>
                {WINDOW_SIZES.map((w) => <option key={w} value={w}>{w}</option>)}
              </select>
            </label>
            <Button size="sm" onClick={() => setStart(maxStart)} disabled={start === maxStart}>Jump to latest</Button>
          </div>
        </div>
        <p className="mt-3 text-2xs text-ink-faint">{fmtInt(scoredInWindow.length)} of {fmtInt(inputs.length)} samples in this window have API results.</p>
      </Panel>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel title="Measurement trends" subtitle={display === "raw" ? "Raw values exactly as supplied in your input" : "Display-only z-scores within this window — NOT the model's scaling"}
          actions={<Tabs idPrefix="display" label="Value display" value={display} onChange={setDisplay}
            items={[{ id: "raw", label: "Raw values" }, { id: "zscore", label: "Window z-score" }]} />}>
          {selected.length ? (
            <SensorTrendChart rows={sensorData} features={selected} xAxisLabel={xLabel} height={320}
              yAxisLabel={display === "raw" ? "Raw measurement" : "z-score within window"} />
          ) : <EmptyState compact title="Select at least one sensor" />}
          <p className="mt-2 text-2xs text-ink-faint">The model's own scaled (transformed) inputs are computed inside the API and are not returned, so they are not shown here.</p>
        </Panel>
        <Panel title="Sensors">
          <FeaturePicker features={featureColumns} selected={selected} onChange={setSelected} max={6} />
        </Panel>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Panel title="Reconstruction error (same window)" subtitle="API output per sample with the calibrated threshold">
          {timeline.length ? (
            <ErrorTimelineChart points={timeline} title="Reconstruction error in window" xAxisLabel={xLabel} height={280} />
          ) : <EmptyState compact title="No API results in this window" description="These samples have not been scored yet." />}
        </Panel>
        <Panel title="Per-feature contribution ranking" subtitle={topK ? `Built from the top-${topK} contributors the API returns per sample` : undefined}
          actions={<Tabs idPrefix="rank" label="Ranking metric" value={rank} onChange={setRank}
            items={[{ id: "top1", label: "#1 count" }, { id: "share", label: "Mean share" }]} />}>
          {ranking.length ? (
            <>
              <ContributionBarChart asPercent={rank === "share"} valueLabel={rank === "top1" ? "Times ranked #1" : "Mean share of squared error"}
                data={[...ranking].filter((c) => rank !== "top1" || c.topOneCount > 0)
                  .sort((a, b) => (rank === "top1" ? b.topOneCount - a.topOneCount : b.meanShare - a.meanShare)).slice(0, 12)
                  .map((c) => ({ feature: c.feature, value: rank === "top1" ? c.topOneCount : c.meanShare, detail: `in a top list ${c.appearances}×` }))} />
              <Banner tone="neutral" title="How to read this">
                Features outside a sample's top {topK || "k"} are unknown for that sample and are not counted. A high contribution shows where the
                reconstruction deviates; it is not a calibrated causal explanation of a fault.
              </Banner>
            </>
          ) : <EmptyState compact title="No contributions in this window" />}
        </Panel>
      </div>
    </div>
  );
}

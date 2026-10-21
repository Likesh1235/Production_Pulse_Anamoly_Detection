import { Braces, FileSpreadsheet, ListRestart, Send, Square, Workflow } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useApiClient } from "../lib/api/ApiClientContext";
import { isApiRequestError } from "../lib/api/errors";
import { MAX_BATCH_SIZE } from "../lib/config";
import { fmtInt, fmtNum } from "../lib/format";
import { buildPredictRequest, isSubmittable, parseCsv, parseJson, type ParseResult } from "../lib/validation";
import { newRunId, useSession, type SessionResult, type SessionRun } from "../state/SessionContext";
import { useStatus } from "../state/StatusContext";
import { useToast } from "../state/ToastContext";
import { MODEL_STATUS_LABELS } from "../lib/api/types";
import { ModelReadinessBanner } from "../components/model/ModelReadinessBanner";
import { CsvDropZone, type LoadedFile } from "../components/predictions/CsvDropZone";
import { InputPreview } from "../components/predictions/InputPreview";
import { ResultsPanel } from "../components/predictions/ResultsPanel";
import { ValidationReport } from "../components/predictions/ValidationReport";
import { Banner, Button, EmptyState, Panel } from "../components/ui/primitives";
import { ErrorState } from "../components/ui/ErrorState";
import { Tabs } from "../components/ui/Tabs";

type Mode = "csv" | "json";

const JSON_HELP = `{"samples": [
  {"sample_id": "optional-id", "features": {"xmeas_1": <number>, ..., "xmv_11": <number>}},
  ...
]}
or one sample: {"features": {...all 52 features...}}`;

export function PredictionsPage() {
  const client = useApiClient();
  const status = useStatus();
  const session = useSession();
  const { notify } = useToast();
  const [mode, setMode] = useState<Mode>("csv");
  const [file, setFile] = useState<LoadedFile | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [jsonText, setJsonText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const parsed: ParseResult | null = useMemo(() => {
    if (mode === "csv") return file ? parseCsv(file.text, status.featureColumns) : null;
    return jsonText.trim() ? parseJson(jsonText, status.featureColumns) : null;
  }, [mode, file, jsonText, status.featureColumns]);
  const run = session.runs.find((r) => r.id === runId) ?? null;

  const valid = parsed !== null && isSubmittable(parsed, MAX_BATCH_SIZE);
  const blocked: string | null = !parsed ? "Provide a CSV file or JSON first."
    : !valid ? "Fix the validation errors before submitting."
      : status.connection !== "online" ? "The API is not reachable."
        : !status.modelReady ? `The model is not ready (${MODEL_STATUS_LABELS[status.health?.model.status ?? ""] ?? status.health?.model.status ?? "unknown"}).`
          : null;

  const reset = () => {
    setFile(null); setFileError(null); setJsonText(""); setError(null); setRunId(null);
  };
  /** A template with every feature name the model expects; values are left empty for the user to fill in. */
  const insertTemplate = () => {
    const features = Object.fromEntries(status.featureColumns.map((c) => [c, null]));
    setJsonText(JSON.stringify({ samples: [{ sample_id: "my-sample-1", features }] }, null, 2));
  };

  const submit = async () => {
    if (!parsed || blocked) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setSubmitting(true);
    setError(null);
    try {
      const res = await client.predict(buildPredictRequest(parsed.samples, status.featureColumns), controller.signal);
      const results: (SessionResult | undefined)[] = parsed.samples.map(() => undefined);
      for (const r of res.results) {
        if (r.index >= 0 && r.index < results.length) {
          results[r.index] = { ...r, requestId: res.request_id, modelId: res.model_id, thresholdSource: res.threshold_source };
        }
      }
      const newRun: SessionRun = {
        id: newRunId(), source: "workbench", label: mode === "csv" ? file?.name ?? "CSV" : "Pasted JSON",
        createdAt: new Date().toISOString(), inputs: parsed.samples, results, hasTimestamps: parsed.hasTimestamps,
        status: "complete", warnings: res.warnings, historyStored: res.history_stored, fileName: file?.name,
      };
      session.addRun(newRun);
      setRunId(newRun.id);
      if (res.results.length !== parsed.samples.length) {
        setError(new Error(`The API returned ${res.results.length} result(s) for ${parsed.samples.length} sample(s).`));
      }
      notify({
        tone: res.n_anomalies ? "danger" : "ok",
        title: `API scored ${fmtInt(res.n_samples)} sample(s)`,
        message: `${fmtInt(res.n_anomalies)} above the calibrated threshold ${fmtNum(res.threshold)}`,
      });
    } catch (e) {
      if (isApiRequestError(e) && e.kind === "aborted") return;
      setError(e);
      if (isApiRequestError(e) && (e.isConnectionProblem || e.code === "model_unavailable")) void status.refresh();
    } finally {
      setSubmitting(false);
      abortRef.current = null;
    }
  };

  return (
    <div className="space-y-5">
      <ModelReadinessBanner />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel title="Input" eyebrow="POST /predict" subtitle="Load samples, check the validation report, then submit"
          actions={<Tabs idPrefix="input-mode" label="Input method" value={mode} onChange={(m) => { setMode(m); setError(null); }}
            items={[{ id: "csv", label: "CSV upload", icon: <FileSpreadsheet className="h-3.5 w-3.5" aria-hidden /> },
              { id: "json", label: "Paste JSON", icon: <Braces className="h-3.5 w-3.5" aria-hidden /> }]} />}>
          <div className="space-y-4">
            <div role="tabpanel" id={`input-mode-panel-${mode}`} aria-labelledby={`input-mode-tab-${mode}`}>
              {mode === "csv" ? (
                <>
                  <CsvDropZone file={file} onLoaded={(f) => { setFile(f); setFileError(null); setError(null); }} onError={setFileError} />
                  {fileError && <p role="alert" className="mt-2 text-xs text-danger">{fileError}</p>}
                </>
              ) : (
                <div>
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <label htmlFor="json-input" className="field-label mb-0">Samples JSON</label>
                    <button type="button" className="link text-xs" onClick={insertTemplate}
                      title="Inserts all feature names the model expects, with empty values for you to fill in">Insert feature template</button>
                  </div>
                  <textarea id="json-input" value={jsonText} onChange={(e) => setJsonText(e.target.value)} spellCheck={false}
                    rows={12} aria-describedby="json-help" className="input font-mono text-xs leading-relaxed"
                    placeholder='{"samples": [{"features": {"xmeas_1": 0.25, "...": "..."}}]}' />
                  <details id="json-help" className="mt-2 text-xs text-ink-muted">
                    <summary className="cursor-pointer select-none text-ink-muted hover:text-accent">Accepted format</summary>
                    <pre className="mt-2 overflow-x-auto rounded-control border border-line bg-surface-1 p-2 font-mono text-2xs">{JSON_HELP}</pre>
                    <p className="mt-1">Feature names can be in any order. Values must be finite numbers (no strings, null, NaN or Infinity).</p>
                  </details>
                </div>
              )}
            </div>

            {parsed && <ValidationReport result={parsed} maxBatchSize={MAX_BATCH_SIZE} />}
            {parsed && parsed.samples.length > 0 && <InputPreview samples={parsed.samples} featureColumns={status.featureColumns} />}
            {parsed && parsed.samples.length > MAX_BATCH_SIZE && (
              <p className="text-xs text-ink-muted">Large file? <Link className="text-accent hover:underline" to="/monitor">Replay it in batches in Live Monitor</Link>.</p>
            )}

            <div className="divider" />
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="primary" icon={<Send className="h-3.5 w-3.5" aria-hidden />} onClick={() => void submit()}
                disabled={Boolean(blocked)} loading={submitting} aria-describedby="submit-hint">
                {submitting ? "Scoring…" : `Submit ${parsed?.samples.length ? fmtInt(parsed.samples.length) : ""} sample(s)`}
              </Button>
              {submitting ? (
                <Button variant="ghost" icon={<Square className="h-3.5 w-3.5" aria-hidden />} onClick={() => abortRef.current?.abort()}>Cancel</Button>
              ) : (file || jsonText || run || error !== null) && (
                <Button variant="ghost" icon={<ListRestart className="h-3.5 w-3.5" aria-hidden />} onClick={reset}>Reset</Button>
              )}
              <p id="submit-hint" className="text-xs text-ink-faint">{blocked ?? `Batch limit ${fmtInt(MAX_BATCH_SIZE)} samples per request.`}</p>
            </div>
          </div>
        </Panel>

        <Panel title="Prediction results" subtitle="Reconstruction error, threshold, decision and top contributors, exactly as returned by the API">
          <div className="space-y-4">
            {error !== null && <ErrorState error={error} context="POST /predict" onRetry={blocked ? undefined : () => void submit()} />}
            {submitting && <Banner tone="info" title="Waiting for the API…">The request has been sent; results appear only when the API answers.</Banner>}
            {run ? <ResultsPanel run={run} /> : !submitting && (
              <EmptyState icon={<Workflow className="h-5 w-5" />} title="No predictions in this view yet"
                description="Upload a CSV or paste JSON, check the validation report, then submit. Results are shown only after the API returns them." />
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}

import { Moon, Sun } from "lucide-react";
import { useStatus } from "../state/StatusContext";
import { useTheme } from "../state/ThemeContext";
import { API_BASE_URL, APP_VERSION, MAX_BATCH_SIZE, REQUEST_TIMEOUT_MS, STATUS_POLL_MS } from "../lib/config";
import { FEATURE_COLUMNS, featureGroup } from "../lib/features";
import { MODEL_STATUS_LABELS } from "../lib/api/types";
import { fmtDateTime, fmtInt, fmtNum, humanise } from "../lib/format";
import { Badge, Banner, Button, KeyValueList, Panel } from "../components/ui/primitives";
import { ErrorState } from "../components/ui/ErrorState";
import { Tabs } from "../components/ui/Tabs";

function show(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return fmtNum(v);
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Read-only system information. Backend settings live in configs/default.toml and cannot be changed here. */
export function SettingsPage() {
  const s = useStatus();
  const { theme, setTheme } = useTheme();
  const meta = s.modelStatus?.metadata;
  const builtInMatches = s.schemaFromApi && s.featureColumns.length === FEATURE_COLUMNS.length && s.featureColumns.every((c, i) => c === FEATURE_COLUMNS[i]);
  return (
    <div className="space-y-5">
      <Banner tone="neutral" title="Read-only">
        Backend configuration (host, port, batch limit, CORS, model directory) is set in <span className="font-mono">configs/default.toml</span> [api] or PP_API_* environment variables and requires an API restart.
        The dashboard cannot change it. Secrets such as the admin token are never shown or sent by this dashboard.
      </Banner>
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Dashboard" subtitle="Frontend build settings (Vite env)">
          <KeyValueList items={[
            { label: "Frontend version", value: APP_VERSION, mono: true },
            { label: "API base URL (VITE_API_BASE_URL)", value: API_BASE_URL, mono: true },
            { label: "Batch limit used for validation (VITE_MAX_BATCH_SIZE)", value: fmtInt(MAX_BATCH_SIZE), mono: true },
            { label: "Request timeout", value: `${REQUEST_TIMEOUT_MS / 1000} s`, mono: true },
            { label: "Status polling", value: STATUS_POLL_MS > 0 ? `every ${STATUS_POLL_MS / 1000} s` : "off", mono: true },
          ]} />
          <div className="mt-4 flex items-center gap-3">
            <span className="text-xs text-ink-muted">Theme</span>
            <Tabs idPrefix="theme" label="Theme" value={theme} onChange={setTheme}
              items={[{ id: "dark", label: "Dark", icon: <Moon className="h-3.5 w-3.5" aria-hidden /> }, { id: "light", label: "Light", icon: <Sun className="h-3.5 w-3.5" aria-hidden /> }]} />
          </div>
        </Panel>

        <Panel title="API" eyebrow="GET /health" subtitle="Latest health check" actions={<Button size="sm" onClick={() => void s.refresh()} loading={s.refreshing}>Re-check</Button>}>
          {s.health ? (
            <KeyValueList items={[
              { label: "Status", value: <Badge tone="ok" dot>{s.health.status}</Badge> },
              { label: "API version", value: s.health.api_version, mono: true },
              { label: "Model", value: <Badge tone={s.health.model.available ? "ok" : "warn"}>{MODEL_STATUS_LABELS[s.health.model.status] ?? s.health.model.status}</Badge> },
              { label: "History database", value: <Badge tone={s.health.database.available ? "ok" : "warn"}>{s.health.database.status}</Badge> },
              { label: "Last checked", value: s.lastChecked ? s.lastChecked.toLocaleTimeString() : "—" },
            ]} />
          ) : s.connection === "checking" ? <p className="text-xs text-ink-muted">Checking…</p> : <ErrorState error={s.healthError} context="GET /health" onRetry={() => void s.refresh()} />}
          {s.health && !s.health.model.available && s.health.model.reason && <p className="mt-3 text-xs text-ink-muted">{s.health.model.reason}</p>}
        </Panel>

        <Panel title="Model metadata" eyebrow="GET /model/status" subtitle="Only fields present in the saved artifact">
          {s.modelStatus ? (
            <KeyValueList items={[
              { label: "Availability", value: <Badge tone={s.modelStatus.available ? "ok" : "warn"}>{MODEL_STATUS_LABELS[s.modelStatus.status] ?? s.modelStatus.status}</Badge> },
              { label: "Model directory (relative)", value: s.modelStatus.model_dir, mono: true },
              ...(s.modelStatus.model_id ? [{ label: "Model ID", value: s.modelStatus.model_id, mono: true }] : []),
              ...(typeof meta?.threshold === "number" ? [{ label: "Calibrated threshold", value: fmtNum(meta.threshold), mono: true }] : []),
              ...(meta?.threshold_source ? [{ label: "Calibration source", value: meta.threshold_source }] : []),
              ...(meta?.created_utc ? [{ label: "Artifact created", value: fmtDateTime(meta.created_utc) }] : []),
              ...(meta?.model_type ? [{ label: "Model type", value: meta.model_type, mono: true }] : []),
              ...(meta?.score ? [{ label: "Score", value: meta.score }] : []),
              ...(meta?.preprocessing?.method ? [{ label: "Preprocessing", value: `${meta.preprocessing.method}${meta.preprocessing.fit_source ? `, fitted on ${meta.preprocessing.fit_source}` : ""}` }] : []),
            ]} />
          ) : s.connection === "checking" ? <p className="text-xs text-ink-muted">Checking…</p> : <ErrorState error={s.modelStatusError} context="GET /model/status" onRetry={() => void s.refresh()} />}
        </Panel>

        <div className="space-y-5">
          <Panel title="Library versions" subtitle="Recorded in the model artifact when it was saved">
            {meta?.library_versions ? (
              <KeyValueList items={Object.entries(meta.library_versions).map(([k, v]) => ({ label: k, value: String(v), mono: true }))} />
            ) : <p className="text-xs text-ink-muted">Not available: no loaded model artifact reports library versions.</p>}
          </Panel>
          {(meta?.settings || meta?.training_summary) && (
            <Panel title="Training configuration and result" subtitle="Hyper-parameters and training summary saved with the model">
              <div className="grid gap-x-6 gap-y-4 xl:grid-cols-2">
                {meta?.settings && (
                  <div><p className="eyebrow mb-2">Hyper-parameters</p>
                    <KeyValueList items={Object.entries(meta.settings).map(([k, v]) => ({ label: humanise(k), value: show(v), mono: true }))} /></div>
                )}
                {meta?.training_summary && (
                  <div><p className="eyebrow mb-2">Training summary</p>
                    <KeyValueList items={Object.entries(meta.training_summary).map(([k, v]) => ({ label: humanise(k), value: show(v), mono: true }))} /></div>
                )}
              </div>
            </Panel>
          )}
        </div>
      </div>

      <Panel title="Feature schema" subtitle={`${s.featureColumns.length} features in the model's canonical order, ${s.schemaFromApi ? "from GET /model/status" : "built-in copy (API not reached yet)"}. xmv = manipulated variables, xmeas = measurements.`}
        actions={s.schemaFromApi && <Badge tone={builtInMatches ? "ok" : "warn"}>{builtInMatches ? "matches dashboard" : "differs from built-in list"}</Badge>}>
        <ol className="grid grid-cols-3 gap-1.5 sm:grid-cols-6 lg:grid-cols-9 xl:grid-cols-[repeat(13,minmax(0,1fr))]">
          {s.featureColumns.map((f, i) => (
            <li key={f} className="chip justify-between" title={featureGroup(f) === "XMV" ? "Manipulated variable" : "Process measurement"}>
              <span className="text-ink-faint">{i + 1}</span><span className={featureGroup(f) === "XMV" ? "text-accent" : "text-ink"}>{f}</span>
            </li>
          ))}
        </ol>
      </Panel>
    </div>
  );
}

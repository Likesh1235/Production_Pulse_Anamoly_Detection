import { Cpu } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useStatus } from "../../state/StatusContext";
import { MODEL_STATUS_LABELS } from "../../lib/api/types";
import { fmtDateTime, fmtInt, fmtNum } from "../../lib/format";
import { Badge, EmptyState, KeyValueList, Panel, Skeleton } from "../ui/primitives";
import { ErrorState } from "../ui/ErrorState";

/** Model card from GET /model/status. Only fields present in the response are shown. */
export function ModelInfoPanel({ compact }: { compact?: boolean }) {
  const { modelStatus, modelStatusError, connection, refresh } = useStatus();
  const title = "Model and threshold";
  if (connection === "checking" && !modelStatus) {
    return <Panel title={title}><div className="space-y-2"><Skeleton className="h-4 w-2/3" /><Skeleton className="h-4 w-1/2" /><Skeleton className="h-4 w-3/4" /></div></Panel>;
  }
  if (!modelStatus) {
    return <Panel title={title}><ErrorState error={modelStatusError} context="GET /model/status" onRetry={() => void refresh()} /></Panel>;
  }
  const m = modelStatus.metadata ?? {};
  const settings = m.settings as Record<string, unknown> | undefined;
  const ts = m.training_summary as Record<string, unknown> | undefined;
  const items: { label: string; value: ReactNode; mono?: boolean }[] = [];
  if (modelStatus.model_id) items.push({ label: "Model ID", value: modelStatus.model_id, mono: true });
  if (typeof m.threshold === "number") items.push({ label: "Calibrated threshold", value: fmtNum(m.threshold), mono: true });
  if (m.threshold_source) items.push({ label: "Threshold source", value: <span title={m.threshold_source}>{m.threshold_source}</span> });
  if (m.score) items.push({ label: "Score", value: m.score });
  if (settings && Array.isArray(settings.hidden_layers) && typeof settings.bottleneck === "number") {
    const h = settings.hidden_layers as number[];
    items.push({ label: "Architecture", value: [modelStatus.feature_schema.n_features, ...h, settings.bottleneck, ...[...h].reverse(), modelStatus.feature_schema.n_features].join(" → "), mono: true });
  }
  if (!compact) {
    if (ts && typeof ts.best_epoch === "number") items.push({ label: "Best epoch", value: `${ts.best_epoch}${typeof ts.epochs_run === "number" ? ` of ${ts.epochs_run} run` : ""}`, mono: true });
    if (ts && typeof ts.best_val_loss === "number") items.push({ label: "Best validation loss", value: fmtNum(ts.best_val_loss), mono: true });
    if (typeof m.n_fit_rows === "number") items.push({ label: "Training rows", value: fmtInt(m.n_fit_rows), mono: true });
    if (typeof m.n_calibration_rows === "number") items.push({ label: "Calibration rows", value: fmtInt(m.n_calibration_rows), mono: true });
    if (m.created_utc) items.push({ label: "Trained (artifact created)", value: fmtDateTime(m.created_utc) });
  }
  if (modelStatus.loaded_utc) items.push({ label: "Loaded by API", value: fmtDateTime(modelStatus.loaded_utc) });

  return (
    <Panel title={title} eyebrow="GET /model/status"
      actions={<Badge tone={modelStatus.available ? "ok" : "warn"} dot>{MODEL_STATUS_LABELS[modelStatus.status] ?? modelStatus.status}</Badge>}>
      {!modelStatus.available && (
        <p className="mb-3 text-xs leading-relaxed text-ink-muted">{modelStatus.reason ?? "The model is not ready."}</p>
      )}
      {items.length > 0 ? <KeyValueList items={items} /> : (
        <EmptyState compact icon={<Cpu className="h-5 w-5" />} title="No model metadata"
          description="The API has no loaded model artifact to describe yet." />
      )}
      {!compact && <p className="mt-3 text-2xs text-ink-faint">Full metadata: <Link className="link" to="/settings">Settings</Link></p>}
    </Panel>
  );
}

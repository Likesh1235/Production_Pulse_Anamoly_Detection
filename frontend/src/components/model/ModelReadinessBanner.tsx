import { useStatus } from "../../state/StatusContext";
import { MODEL_STATUS_LABELS } from "../../lib/api/types";
import { API_BASE_URL } from "../../lib/config";
import { Banner, Button } from "../ui/primitives";

/** Prominent readiness banner; reflects only the latest /health answer. */
export function ModelReadinessBanner() {
  const { connection, health, refresh, refreshing, threshold } = useStatus();
  if (connection === "checking") {
    return <Banner tone="neutral" title="Checking the API…">Waiting for the first response from {API_BASE_URL}/health.</Banner>;
  }
  if (connection === "offline" || !health) {
    return (
      <Banner tone="danger" title="API unreachable — predictions are not possible"
        actions={<Button size="sm" onClick={() => void refresh()} loading={refreshing}>Retry</Button>}>
        No response from {API_BASE_URL}. Start the FastAPI backend (docs/FRONTEND.md §4) and check CORS settings.
      </Banner>
    );
  }
  if (!health.model.available) {
    return (
      <Banner tone="warn" title={`Model not ready — ${MODEL_STATUS_LABELS[health.model.status] ?? health.model.status}`}
        actions={<Button size="sm" onClick={() => void refresh()} loading={refreshing}>Re-check</Button>}>
        {health.model.reason ?? "The API cannot score samples until a trained, calibrated model is loaded."}
      </Banner>
    );
  }
  // Ready: one quiet line instead of a banner (the status strip already shows the state).
  return (
    <p role="status" className="flex items-center gap-2 text-xs text-ink-muted">
      <span className="h-1.5 w-1.5 rounded-full bg-ok" aria-hidden />
      Model ready{threshold !== null ? `, threshold ${threshold.toPrecision(4)}` : ""}. Samples are scored by the API; nothing is computed in the browser.
    </p>
  );
}

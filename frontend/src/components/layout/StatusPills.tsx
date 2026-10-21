import clsx from "clsx";
import { Loader2 } from "lucide-react";
import { useStatus } from "../../state/StatusContext";
import { MODEL_STATUS_LABELS } from "../../lib/api/types";
import type { Tone } from "../ui/primitives";

interface Pill { key: string; label: string; value: string; tone: Tone | "checking"; title: string }

/** Derives every status from real API responses. Nothing is green until /health has answered. */
export function useStatusPills(): Pill[] {
  const { connection, health, healthError, lastChecked } = useStatus();
  const checked = lastChecked ? `Last checked ${lastChecked.toLocaleTimeString()}` : "Not checked yet";
  const api: Pill = connection === "checking"
    ? { key: "api", label: "API", value: "Checking…", tone: "checking", title: "Waiting for the first /health response" }
    : connection === "online"
      ? { key: "api", label: "API", value: "Connected", tone: "ok", title: `GET /health answered (API v${health?.api_version}). ${checked}` }
      : { key: "api", label: "API", value: healthError?.kind === "timeout" ? "Timed out" : "Unreachable", tone: "danger", title: `${healthError?.message ?? "No response"}. ${checked}` };
  const model: Pill = !health
    ? { key: "model", label: "Model", value: "Unknown", tone: "neutral", title: "Model state is unknown until the API answers" }
    : health.model.available
      ? { key: "model", label: "Model", value: "Ready", tone: "ok", title: "Model loaded, schema validated and threshold calibrated" }
      : { key: "model", label: "Model", value: MODEL_STATUS_LABELS[health.model.status] ?? health.model.status, tone: "warn", title: health.model.reason ?? health.model.status };
  const db: Pill = !health
    ? { key: "db", label: "History", value: "Unknown", tone: "neutral", title: "History database state is unknown until the API answers" }
    : health.database.available
      ? { key: "db", label: "History", value: "Available", tone: "ok", title: "Prediction history database reachable" }
      : { key: "db", label: "History", value: "Unavailable", tone: "warn", title: health.database.reason ?? "History database unavailable" };
  return [api, model, db];
}

const DOT: Record<Tone, string> = {
  ok: "bg-ok", warn: "bg-warn", danger: "bg-danger", info: "bg-info", accent: "bg-accent", neutral: "bg-ink-faint",
};

/** One joined status strip (API | Model | History), like an annunciator panel. */
export function StatusPills() {
  const pills = useStatusPills();
  return (
    <ul aria-label="System status" className="flex w-full items-stretch overflow-hidden rounded-control border border-line bg-surface-2 text-xs sm:w-auto">
      {pills.map((p, i) => (
        <li key={p.key} title={p.title}
          className={clsx("flex flex-1 items-center gap-1.5 px-2.5 py-1.5 sm:flex-none", i > 0 && "border-l border-line",
            p.tone === "danger" && "bg-danger/10", p.tone === "warn" && "bg-warn/10")}>
          {p.tone === "checking"
            ? <Loader2 className="h-3 w-3 animate-spin text-ink-faint" aria-hidden />
            : <span className={clsx("h-1.5 w-1.5 shrink-0 rounded-full", DOT[p.tone])} aria-hidden />}
          <span className="text-ink-faint">{p.label}</span>
          <span className="font-medium text-ink" data-testid={`status-${p.key}`}>{p.value}</span>
        </li>
      ))}
    </ul>
  );
}

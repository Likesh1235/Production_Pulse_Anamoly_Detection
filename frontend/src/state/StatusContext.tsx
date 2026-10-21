import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useApiClient } from "../lib/api/ApiClientContext";
import { ApiRequestError, isApiRequestError } from "../lib/api/errors";
import type { HealthResponse, ModelStatusResponse } from "../lib/api/types";
import { FEATURE_COLUMNS } from "../lib/features";

export type Connection = "checking" | "online" | "offline";

export interface StatusValue {
  /** "online" ONLY after GET /health actually answered; never assumed. */
  connection: Connection;
  health: HealthResponse | null;
  healthError: ApiRequestError | null;
  modelStatus: ModelStatusResponse | null;
  modelStatusError: ApiRequestError | null;
  lastChecked: Date | null;
  refreshing: boolean;
  refresh: () => Promise<void>;
  /** Authoritative feature order from the API when known, else the built-in copy. */
  featureColumns: readonly string[];
  schemaFromApi: boolean;
  modelReady: boolean;
  databaseAvailable: boolean;
  threshold: number | null;
  thresholdSource: string | null;
}

const StatusContext = createContext<StatusValue | null>(null);

function toApiError(e: unknown, endpoint: string): ApiRequestError {
  return isApiRequestError(e) ? e : new ApiRequestError({ kind: "network", endpoint, message: String(e) });
}

export function StatusProvider({ children, pollMs }: { children: ReactNode; pollMs: number }) {
  const client = useApiClient();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<ApiRequestError | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [modelStatusError, setModelStatusError] = useState<ApiRequestError | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const alive = useRef(true);
  const inFlight = useRef<Promise<void> | null>(null);

  const refresh = useCallback(async () => {
    if (inFlight.current) return inFlight.current;
    const run = (async () => {
      setRefreshing(true);
      const [h, m] = await Promise.allSettled([client.health(), client.modelStatus()]);
      if (!alive.current) return;
      if (h.status === "fulfilled") { setHealth(h.value); setHealthError(null); }
      else { setHealth(null); setHealthError(toApiError(h.reason, "/health")); }
      if (m.status === "fulfilled") { setModelStatus(m.value); setModelStatusError(null); }
      else { setModelStatus(null); setModelStatusError(toApiError(m.reason, "/model/status")); }
      setLastChecked(new Date());
      setRefreshing(false);
    })();
    inFlight.current = run;
    try { await run; } finally { inFlight.current = null; }
  }, [client]);

  useEffect(() => {
    alive.current = true;
    void refresh();
    if (pollMs <= 0) return () => { alive.current = false; };
    const id = window.setInterval(() => {
      if (document.visibilityState !== "hidden") void refresh();
    }, pollMs);
    return () => { alive.current = false; window.clearInterval(id); };
  }, [refresh, pollMs]);

  const value = useMemo<StatusValue>(() => {
    const apiCols = modelStatus?.feature_schema.feature_columns;
    const schemaFromApi = Array.isArray(apiCols) && apiCols.length > 0;
    const meta = modelStatus?.metadata;
    const connection: Connection = lastChecked === null ? "checking" : health ? "online" : "offline";
    return {
      connection,
      health,
      healthError,
      modelStatus,
      modelStatusError,
      lastChecked,
      refreshing,
      refresh,
      featureColumns: schemaFromApi && apiCols ? apiCols : FEATURE_COLUMNS,
      schemaFromApi,
      modelReady: Boolean(health?.model.available),
      databaseAvailable: Boolean(health?.database.available),
      threshold: typeof meta?.threshold === "number" ? meta.threshold : null,
      thresholdSource: typeof meta?.threshold_source === "string" ? meta.threshold_source : null,
    };
  }, [health, healthError, modelStatus, modelStatusError, lastChecked, refreshing, refresh]);

  return <StatusContext.Provider value={value}>{children}</StatusContext.Provider>;
}

export function useStatus(): StatusValue {
  const v = useContext(StatusContext);
  if (!v) throw new Error("useStatus must be used inside <StatusProvider>");
  return v;
}

/** In-memory store for predictions made in THIS browser session (workbench runs and
 *  user-initiated replays). It holds the user's own input rows next to the API's
 *  results so charts can show raw measurements and reconstruction errors together.
 *  Nothing is persisted in the browser; the API's /history is the persistent record. */

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import type { PredictionResult } from "../lib/api/types";
import type { ParsedSample } from "../lib/validation";

export interface SessionResult extends PredictionResult {
  requestId: string;
  modelId: string | null;
  thresholdSource: string;
}

export type RunSource = "workbench" | "replay";

export interface SessionRun {
  id: string;
  source: RunSource;
  label: string;
  createdAt: string;
  inputs: ParsedSample[];
  /** results[i] belongs to inputs[i]; undefined until scored */
  results: (SessionResult | undefined)[];
  hasTimestamps: boolean;
  status: "running" | "paused" | "complete" | "stopped" | "failed";
  warnings: string[];
  /** false if the API reported that any batch could not be saved to its history */
  historyStored?: boolean;
  fileName?: string;
}

interface SessionValue {
  runs: SessionRun[];
  activeRun: SessionRun | null;
  setActiveRun: (id: string) => void;
  addRun: (run: SessionRun) => void;
  updateRun: (id: string, update: (run: SessionRun) => SessionRun) => void;
  clearRuns: () => void;
}

const MAX_RUNS = 6;
const SessionContext = createContext<SessionValue | null>(null);

export function newRunId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<SessionRun[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  const addRun = useCallback((run: SessionRun) => {
    setRuns((prev) => [run, ...prev].slice(0, MAX_RUNS));
    setActiveId(run.id);
  }, []);
  const updateRun = useCallback((id: string, update: (run: SessionRun) => SessionRun) => {
    setRuns((prev) => prev.map((r) => (r.id === id ? update(r) : r)));
  }, []);
  const clearRuns = useCallback(() => { setRuns([]); setActiveId(null); }, []);

  const value = useMemo<SessionValue>(() => ({
    runs,
    activeRun: runs.find((r) => r.id === activeId) ?? runs[0] ?? null,
    setActiveRun: setActiveId,
    addRun,
    updateRun,
    clearRuns,
  }), [runs, activeId, addRun, updateRun, clearRuns]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const v = useContext(SessionContext);
  if (!v) throw new Error("useSession must be used inside <SessionProvider>");
  return v;
}

export function scoredRows(run: SessionRun) {
  const out: { input: ParsedSample; result: SessionResult }[] = [];
  run.results.forEach((result, i) => { if (result) out.push({ input: run.inputs[i], result }); });
  return out;
}

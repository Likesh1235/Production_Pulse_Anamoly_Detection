import { useCallback, useEffect, useRef, useState } from "react";
import { ApiRequestError, isApiRequestError } from "../lib/api/errors";

export interface Resource<T> {
  data: T | null;
  error: ApiRequestError | Error | null;
  loading: boolean;
  reload: () => void;
}

/** Load data with cancellation on unmount/dependency change. Errors are surfaced, never replaced by defaults. */
export function useResource<T>(load: (signal: AbortSignal) => Promise<T>, deps: unknown[], enabled = true): Resource<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiRequestError | Error | null>(null);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (!enabled) { setLoading(false); return; }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    loadRef.current(controller.signal)
      .then((d) => { if (!controller.signal.aborted) setData(d); })
      .catch((e: unknown) => {
        if (controller.signal.aborted || (isApiRequestError(e) && e.kind === "aborted")) return;
        setData(null);
        setError(e instanceof Error ? e : new Error(String(e)));
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  // Until the first request for an enabled resource has produced data or an error, report loading.
  // Otherwise the render before the effect starts the request would look like an empty result.
  const pending = enabled && data === null && error === null;
  return { data, error, loading: loading || pending, reload };
}

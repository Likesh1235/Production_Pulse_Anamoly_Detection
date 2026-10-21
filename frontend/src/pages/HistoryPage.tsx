import { ChevronLeft, ChevronRight, History as HistoryIcon, RefreshCw, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { useApiClient } from "../lib/api/ApiClientContext";
import type { HistoryItem } from "../lib/api/types";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, fmtRatio, shortId } from "../lib/format";
import { useResource } from "../hooks/useResource";
import { useStatus } from "../state/StatusContext";
import { AnomalyBadge, Button, EmptyState, Panel, SkeletonRows, Switch } from "../components/ui/primitives";
import { ErrorState } from "../components/ui/ErrorState";

const PAGE_SIZES = [25, 50, 100]; // all <= API history_max_page_size (default 500)

/** Stable cursor paging over GET /history (newest first, ordered by id). */
export function HistoryPage() {
  const client = useApiClient();
  const { connection } = useStatus();
  const [pageSize, setPageSize] = useState(50);
  const [anomaliesOnly, setAnomaliesOnly] = useState(false);
  // cursors[k] = before_id used to load page k (undefined for the newest page)
  const [cursors, setCursors] = useState<(number | undefined)[]>([undefined]);
  const page = cursors.length - 1;
  const beforeId = cursors[page];

  const res = useResource(
    (signal) => client.history({ limit: pageSize, beforeId, anomaliesOnly }, signal),
    [client, pageSize, beforeId, anomaliesOnly],
    connection !== "checking",
  );

  const items: HistoryItem[] = useMemo(() => {
    const seen = new Set<number>();
    return (res.data?.items ?? []).filter((it) => (seen.has(it.id) ? false : (seen.add(it.id), true)));
  }, [res.data]);
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  // Client-side search over the records already loaded on this page (the API has no text search).
  const shown = q ? items.filter((it) => [it.sample_id ?? "", it.request_id, `#${it.id}`, ...it.top_features.map((c) => c.feature)]
    .some((v) => v.toLowerCase().includes(q))) : items;
  const nextCursor = res.data?.next_before_id ?? null;
  const reset = () => setCursors([undefined]);

  return (
    <Panel title="Stored predictions" eyebrow="GET /history"
      subtitle={res.data ? `${fmtInt(res.data.total)} record(s)${anomaliesOnly ? " flagged as anomalous" : ""}, newest first, page ${page + 1}` : "Records persisted by the API"}
      actions={
        <>
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" aria-hidden />
            <input className="input w-48 py-1 pl-8 text-xs" placeholder="Search this page" value={query} onChange={(e) => setQuery(e.target.value)}
              aria-label="Search the records on this page by sample, request, record or feature" />
          </div>
          <Switch checked={anomaliesOnly} onChange={(v) => { setAnomaliesOnly(v); reset(); }} label="Anomalies only" description="server-side filter" />
          <label className="flex items-center gap-2 text-xs text-ink-muted">Rows
            <select className="select w-auto py-1 text-xs" value={pageSize} aria-label="Rows per page" onChange={(e) => { setPageSize(Number(e.target.value)); reset(); }}>
              {PAGE_SIZES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <Button size="sm" icon={<RefreshCw className="h-3.5 w-3.5" aria-hidden />} onClick={() => { reset(); res.reload(); }}>Newest</Button>
        </>
      }
      bodyClassName="p-0">
      {res.loading || connection === "checking" ? <SkeletonRows rows={10} cols={7} /> : res.error ? (
        <div className="p-4"><ErrorState error={res.error} context="GET /history" onRetry={res.reload} /></div>
      ) : items.length === 0 ? (
        <div className="p-4">
          <EmptyState icon={<HistoryIcon className="h-5 w-5" />} title={page > 0 ? "No older records" : anomaliesOnly ? "No anomalous records stored" : "No predictions stored yet"}
            description={page > 0 ? "You have reached the oldest stored record." : "Predictions made through the API (workbench or replay) are stored automatically."} />
        </div>
      ) : shown.length === 0 ? (
        <div className="p-4"><EmptyState compact title="No records on this page match your search" description="The search only looks at the records loaded on this page. Clear it or move to another page." /></div>
      ) : (
        <div className="table-shell rounded-none border-0" role="region" aria-label="Prediction history table" tabIndex={0}>
          <table className="data-table">
            <caption className="sr-only">Prediction history, newest first</caption>
            <thead>
              <tr>
                <th scope="col">Time (local)</th><th scope="col">Record</th><th scope="col">Request / sample</th><th scope="col">Status</th>
                <th scope="col" className="text-right">Reconstruction error</th><th scope="col" className="text-right">Threshold</th>
                <th scope="col" className="text-right">Error / thr.</th><th scope="col">Top contributors</th><th scope="col">Model</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((it) => (
                <tr key={it.id} className={it.is_anomaly ? "bg-danger/[0.04]" : undefined}>
                  <td className="whitespace-nowrap text-xs text-ink-muted" title={it.created_utc}>{fmtDateTime(it.created_utc)}</td>
                  <td className="num">#{it.id}</td>
                  <td className="font-mono text-2xs text-ink-muted">
                    {shortId(it.request_id)} / {it.sample_index + 1}{it.sample_id && <span className="block max-w-[160px] truncate text-ink">{it.sample_id}</span>}
                  </td>
                  <td><AnomalyBadge isAnomaly={it.is_anomaly} /></td>
                  <td className={it.is_anomaly ? "num text-right text-danger" : "num text-right"}>{fmtNum(it.reconstruction_error)}</td>
                  <td className="num text-right text-ink-muted">{fmtNum(it.threshold)}</td>
                  <td className="num text-right">{fmtRatio(it.reconstruction_error / it.threshold)}</td>
                  <td>
                    <div className="flex flex-wrap gap-1">
                      {it.top_features.slice(0, 3).map((c) => <span key={c.feature} className="chip" title={`squared error ${fmtNum(c.squared_error)}`}>{c.feature} {fmtPct(c.share, 0)}</span>)}
                      {it.top_features.length === 0 && <span className="text-2xs text-ink-faint">—</span>}
                    </div>
                  </td>
                  <td className="font-mono text-2xs text-ink-faint">{it.model_id ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <nav aria-label="History pagination" className="flex flex-wrap items-center justify-between gap-2 border-t border-line px-4 py-3">
        <p className="text-2xs text-ink-faint">Cursor paging by record id keeps pages stable while new predictions arrive.</p>
        <div className="flex gap-2">
          <Button size="sm" icon={<ChevronLeft className="h-3.5 w-3.5" aria-hidden />} disabled={page === 0 || res.loading}
            onClick={() => setCursors((c) => c.slice(0, -1))}>Newer</Button>
          <Button size="sm" disabled={nextCursor === null || res.loading} onClick={() => nextCursor !== null && setCursors((c) => [...c, nextCursor])}>
            Older<ChevronRight className="h-3.5 w-3.5" aria-hidden />
          </Button>
        </div>
      </nav>
    </Panel>
  );
}

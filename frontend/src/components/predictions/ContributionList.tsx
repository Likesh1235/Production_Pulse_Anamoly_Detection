import type { FeatureContribution } from "../../lib/api/types";
import { fmtNum, fmtPct } from "../../lib/format";

/** Top contributing features for one sample, exactly as returned by the API. */
export function ContributionList({ items }: { items: FeatureContribution[] }) {
  if (!items.length) return <p className="text-xs text-ink-faint">No contributions returned.</p>;
  const max = Math.max(...items.map((c) => c.share), 1e-12);
  return (
    <ol className="space-y-1.5">
      {items.map((c, i) => (
        <li key={c.feature} className="grid grid-cols-[1.25rem_5.5rem_1fr_auto] items-center gap-2 text-xs">
          <span className="num text-ink-faint">{i + 1}</span>
          <span className="font-mono text-ink">{c.feature}</span>
          <span className="h-1.5 overflow-hidden rounded-full bg-surface-3" aria-hidden>
            <span className={i === 0 ? "block h-full rounded-full bg-accent" : "block h-full rounded-full bg-ink-faint/70"} style={{ width: `${(c.share / max) * 100}%` }} />
          </span>
          <span className="num text-right text-ink-muted" title={`squared error ${fmtNum(c.squared_error)}`}>{fmtPct(c.share)}</span>
        </li>
      ))}
    </ol>
  );
}

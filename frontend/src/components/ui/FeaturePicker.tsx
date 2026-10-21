import clsx from "clsx";
import { Search, X } from "lucide-react";
import { useId, useMemo, useState } from "react";
import { featureGroup, featureGroupLabel, type FeatureGroup } from "../../lib/features";

/** Searchable multi-select over the real feature schema, grouped XMEAS / XMV. */
export function FeaturePicker({ features, selected, onChange, max = 6, label = "Sensors" }: {
  features: readonly string[]; selected: string[]; onChange: (next: string[]) => void; max?: number; label?: string;
}) {
  const [query, setQuery] = useState("");
  const searchId = useId();
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const out: Record<FeatureGroup, string[]> = { XMEAS: [], XMV: [] };
    for (const f of features) if (!q || f.toLowerCase().includes(q)) out[featureGroup(f)].push(f);
    return out;
  }, [features, query]);
  const toggle = (f: string) => {
    if (selected.includes(f)) onChange(selected.filter((x) => x !== f));
    else if (selected.length < max) onChange([...selected, f]);
  };
  return (
    <fieldset className="min-w-0">
      <legend className="field-label">{label} <span className="text-ink-faint">({selected.length}/{max} selected)</span></legend>
      {selected.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {selected.map((f) => (
            <span key={f} className="inline-flex items-center gap-1 rounded-md border border-accent/40 bg-accent/10 py-0.5 pl-2 pr-1 font-mono text-2xs text-accent">
              {f}
              <button type="button" onClick={() => toggle(f)} className="rounded p-0.5 hover:bg-accent/20" aria-label={`Remove ${f}`}>
                <X className="h-3 w-3" aria-hidden />
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" aria-hidden />
        <label htmlFor={searchId} className="sr-only">Search features</label>
        <input id={searchId} className="input pl-8" placeholder="Search e.g. xmeas_7, xmv" value={query} onChange={(e) => setQuery(e.target.value)} />
      </div>
      <div className="mt-2 max-h-56 overflow-y-auto rounded-control border border-line bg-canvas p-2">
        {(Object.keys(groups) as FeatureGroup[]).map((g) => groups[g].length > 0 && (
          <div key={g} className="mb-2 last:mb-0">
            <p className="eyebrow mb-1">{featureGroupLabel(g)}</p>
            <div className="grid gap-1" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(5.75rem, 1fr))" }}>
              {groups[g].map((f) => {
                const on = selected.includes(f);
                const disabled = !on && selected.length >= max;
                return (
                  <label key={f} className={clsx("flex min-w-0 cursor-pointer items-center gap-1.5 rounded px-1.5 py-1 font-mono text-2xs",
                    on ? "bg-accent/10 text-accent" : "text-ink-muted hover:bg-surface-3", disabled && "cursor-not-allowed opacity-40")}>
                    <input type="checkbox" className="h-3 w-3 accent-[rgb(var(--c-accent))]" checked={on} disabled={disabled} onChange={() => toggle(f)} />
                    {f}
                  </label>
                );
              })}
            </div>
          </div>
        ))}
        {groups.XMEAS.length + groups.XMV.length === 0 && <p className="p-2 text-xs text-ink-faint">No feature matches “{query}”.</p>}
      </div>
    </fieldset>
  );
}

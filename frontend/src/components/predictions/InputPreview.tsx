import type { ParsedSample } from "../../lib/validation";
import { fmtNum } from "../../lib/format";

/** Preview of parsed input (first rows, first features) before anything is sent. */
export function InputPreview({ samples, featureColumns, maxRows = 6, maxFeatures = 6 }: {
  samples: ParsedSample[]; featureColumns: readonly string[]; maxRows?: number; maxFeatures?: number;
}) {
  if (samples.length === 0) return null;
  const cols = featureColumns.slice(0, maxFeatures);
  const hasTs = samples.some((s) => s.timestamp);
  return (
    <div>
      <p className="mb-1.5 text-xs text-ink-muted">Preview — first {Math.min(maxRows, samples.length)} of {samples.length} sample(s), {cols.length} of {featureColumns.length} features (canonical order)</p>
      <div className="table-shell max-h-60">
        <table className="data-table">
          <caption className="sr-only">Parsed input preview</caption>
          <thead>
            <tr>
              <th scope="col">#</th><th scope="col">Source</th><th scope="col">Sample ID</th>{hasTs && <th scope="col">Timestamp</th>}
              {cols.map((c) => <th key={c} scope="col" className="text-right">{c}</th>)}
              <th scope="col" className="text-ink-faint">…</th>
            </tr>
          </thead>
          <tbody>
            {samples.slice(0, maxRows).map((s, i) => (
              <tr key={i}>
                <td className="num text-ink-faint">{i + 1}</td>
                <td className="num text-ink-faint">{s.sourceRow}</td>
                <td className="font-mono text-2xs">{s.sampleId ?? <span className="text-ink-faint">—</span>}</td>
                {hasTs && <td className="font-mono text-2xs">{s.timestamp ?? "—"}</td>}
                {cols.map((c) => <td key={c} className="num text-right">{fmtNum(s.features[c], 5)}</td>)}
                <td className="text-2xs text-ink-faint">+{featureColumns.length - cols.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

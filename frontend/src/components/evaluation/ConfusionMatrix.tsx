import { fmtInt, fmtPct } from "../../lib/format";

/** 2x2 confusion matrix from counts in the saved report (rows = actual, columns = predicted). */
export function ConfusionMatrix({ tp, fp, tn, fn, title }: { tp: number; fp: number; tn: number; fn: number; title: string }) {
  const actualNeg = tn + fp;
  const actualPos = fn + tp;
  const cell = (value: number, rowTotal: number, kind: "correct" | "miss" | "false-alarm") => (
    <td className={kind === "correct" ? "bg-accent/10" : kind === "false-alarm" ? "bg-warn/10" : "bg-danger/10"}>
      <span className="num block text-lg font-semibold text-ink">{fmtInt(value)}</span>
      <span className="num block text-2xs text-ink-muted">{rowTotal ? fmtPct(value / rowTotal) : "—"} of row</span>
    </td>
  );
  return (
    <table className="w-full table-fixed border-separate border-spacing-1 text-center text-xs">
      <caption className="mb-2 text-left text-xs text-ink-muted">{title}</caption>
      <thead>
        <tr><th className="w-28" /><th scope="col" className="eyebrow">Predicted normal</th><th scope="col" className="eyebrow">Predicted anomaly</th></tr>
      </thead>
      <tbody>
        <tr><th scope="row" className="eyebrow text-left">Actual normal</th>{cell(tn, actualNeg, "correct")}{cell(fp, actualNeg, "false-alarm")}</tr>
        <tr><th scope="row" className="eyebrow text-left">Actual anomaly</th>{cell(fn, actualPos, "miss")}{cell(tp, actualPos, "correct")}</tr>
      </tbody>
    </table>
  );
}

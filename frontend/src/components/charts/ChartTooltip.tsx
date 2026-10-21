import type { ReactNode } from "react";

export function TooltipCard({ title, rows, footer }: { title: ReactNode; rows: { label: string; value: ReactNode; color?: string }[]; footer?: ReactNode }) {
  return (
    <div className="min-w-[180px] rounded-control border border-line-strong bg-surface-3 px-3 py-2 text-xs" style={{ boxShadow: "var(--shadow-pop)" }}>
      <p className="mb-1.5 font-medium text-ink">{title}</p>
      <dl className="space-y-0.5">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center justify-between gap-4">
            <dt className="flex items-center gap-1.5 text-ink-muted">
              {r.color && <span className="h-2 w-2 rounded-sm" style={{ background: r.color }} aria-hidden />}{r.label}
            </dt>
            <dd className="num text-ink">{r.value}</dd>
          </div>
        ))}
      </dl>
      {footer && <div className="mt-1.5 border-t border-line pt-1.5">{footer}</div>}
    </div>
  );
}

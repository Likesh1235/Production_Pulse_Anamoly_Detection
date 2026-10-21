import clsx from "clsx";

/**
 * Brand mark: a signal trace crossing a dashed alarm limit, i.e. the product's core idea
 * (reconstruction error rising above a calibrated threshold).
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 28 28" fill="none" className={clsx("h-7 w-7 shrink-0", className)} aria-hidden>
      <rect x="0.75" y="0.75" width="26.5" height="26.5" rx="5" className="fill-surface-3 stroke-line-strong" strokeWidth="1.5" />
      <path d="M5 11.5h18" className="stroke-warn" strokeWidth="1.3" strokeDasharray="2 2" />
      <path d="M5 19.5h4l2-3 2.5 4.5 3-13 2.5 9.5 1.5-2H23" className="stroke-accent" strokeWidth="1.8"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Wordmark({ collapsed }: { collapsed?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark />
      {!collapsed && (
        <div className="leading-tight">
          <p className="text-[0.9375rem] font-semibold text-ink">Production Pulse</p>
          <p className="text-xs text-ink-faint">Process anomaly detection</p>
        </div>
      )}
    </div>
  );
}

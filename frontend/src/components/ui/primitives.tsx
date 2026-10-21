import clsx from "clsx";
import { AlertTriangle, CheckCircle2, CircleDashed, Info, Loader2, XCircle } from "lucide-react";
import { forwardRef, useId, useState, type ButtonHTMLAttributes, type ReactNode } from "react";

// ------------------------------------------------------------------ Panel
export function Panel({ title, subtitle, actions, children, className, bodyClassName, eyebrow, id }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
  bodyClassName?: string; eyebrow?: string; id?: string;
}) {
  const headingId = useId();
  return (
    <section className={clsx("panel min-w-0", className)} aria-labelledby={title ? headingId : undefined} id={id}>
      {(title || actions) && (
        <header className="panel-header">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              {title && <h2 id={headingId} className="panel-title">{title}</h2>}
              {eyebrow && <span className="source-tag" title="Data source">{eyebrow}</span>}
            </div>
            {subtitle && <p className="panel-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

// ------------------------------------------------------------------ Button
type Variant = "primary" | "secondary" | "ghost" | "danger";
export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant; size?: "sm" | "md"; icon?: ReactNode; loading?: boolean;
}
// Literal class names: Tailwind only generates classes it can find verbatim in the source, so a template
// string like `btn-${variant}` would silently drop btn-ghost / btn-danger from the build.
const VARIANT_CLASS: Record<Variant, string> = {
  primary: "btn-primary", secondary: "btn-secondary", ghost: "btn-ghost", danger: "btn-danger",
};
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", icon, loading, children, className, disabled, type = "button", ...rest }, ref,
) {
  return (
    <button ref={ref} type={type} disabled={disabled || loading} aria-busy={loading || undefined}
      className={clsx(VARIANT_CLASS[variant], size === "sm" && "btn-sm", className)} {...rest}>
      {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
});

// ------------------------------------------------------------------ Badge / status
export type Tone = "ok" | "warn" | "danger" | "info" | "neutral" | "accent";
const TONE_CLASSES: Record<Tone, string> = {
  ok: "border-ok/40 bg-ok/10 text-ok",
  warn: "border-warn/45 bg-warn/10 text-warn",
  danger: "border-danger/50 bg-danger/10 text-danger",
  info: "border-info/40 bg-info/10 text-info",
  accent: "border-accent/45 bg-accent/10 text-accent",
  neutral: "border-line-strong bg-surface-3 text-ink-muted",
};
const DOT: Record<Tone, string> = {
  ok: "bg-ok", warn: "bg-warn", danger: "bg-danger", info: "bg-info", accent: "bg-accent", neutral: "bg-ink-faint",
};

export function Badge({ tone = "neutral", children, dot, className, title }: {
  tone?: Tone; children: ReactNode; dot?: boolean; className?: string; title?: string;
}) {
  return (
    <span title={title} className={clsx("inline-flex items-center gap-1.5 whitespace-nowrap rounded-[3px] border px-1.5 py-px text-xs font-medium", TONE_CLASSES[tone], className)}>
      {dot && <span className={clsx("h-1.5 w-1.5 rounded-full", DOT[tone])} aria-hidden />}
      {children}
    </span>
  );
}

export function AnomalyBadge({ isAnomaly }: { isAnomaly: boolean }) {
  // Red is reserved for anomalies/errors, green for normal; amber marks the alarm threshold.
  return isAnomaly
    ? <Badge tone="danger" dot>Anomaly</Badge>
    : <Badge tone="ok" dot>Normal</Badge>;
}

export function StatusIcon({ tone, className }: { tone: Tone | "checking"; className?: string }) {
  const cls = clsx("h-4 w-4 shrink-0", className);
  if (tone === "checking") return <Loader2 className={clsx(cls, "animate-spin text-ink-faint")} aria-hidden />;
  if (tone === "ok") return <CheckCircle2 className={clsx(cls, "text-ok")} aria-hidden />;
  if (tone === "warn") return <AlertTriangle className={clsx(cls, "text-warn")} aria-hidden />;
  if (tone === "danger") return <XCircle className={clsx(cls, "text-danger")} aria-hidden />;
  if (tone === "info" || tone === "accent") return <Info className={clsx(cls, "text-info")} aria-hidden />;
  return <CircleDashed className={clsx(cls, "text-ink-faint")} aria-hidden />;
}

// ------------------------------------------------------------------ Metric readout
const VALUE_TONE: Record<Tone, string> = {
  ok: "text-ink", warn: "text-warn", danger: "text-danger", info: "text-ink", accent: "text-ink", neutral: "text-ink",
};

/**
 * One instrument readout: label, value, footnote. State is shown by a small coloured dot (and by colour on
 * the value only for warning/danger), never by tinting the whole tile. Inside <Readouts> it renders as a
 * cell of a joined strip; on its own it is a bordered panel.
 */
export function MetricTile({ label, value, unit, hint, tone = "neutral", icon, footnote, loading, cell }: {
  label: string; value: ReactNode; unit?: string; hint?: string; tone?: Tone; icon?: ReactNode; footnote?: ReactNode; loading?: boolean; cell?: boolean;
}) {
  return (
    <div className={clsx(cell ? "readout-cell" : "panel px-4 py-3")}>
      <div className="flex items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-xs font-medium text-ink-muted">
          {tone !== "neutral" && <span className={clsx("h-1.5 w-1.5 rounded-full", DOT[tone])} aria-hidden />}
          {label}{hint && <InfoTip text={hint} />}
        </p>
        {icon && <span className="text-ink-faint" aria-hidden>{icon}</span>}
      </div>
      {loading ? <div className="skeleton mt-2 h-7 w-24" aria-label="Loading" /> : (
        <p className={clsx("mt-1 flex items-baseline gap-1 text-[1.375rem] font-semibold leading-tight tabular", VALUE_TONE[tone])}>
          {value}{unit && <span className="text-xs font-medium text-ink-muted">{unit}</span>}
        </p>
      )}
      {footnote && <p className="mt-0.5 truncate text-xs text-ink-faint">{footnote}</p>}
    </div>
  );
}

/** A joined strip of readouts separated by hairlines (pass <MetricTile cell /> children). */
export function Readouts({ children, className, label }: { children: ReactNode; className?: string; label?: string }) {
  return <div role="group" aria-label={label} className={clsx("readouts", className)}>{children}</div>;
}

// ------------------------------------------------------------------ Tooltip
export function InfoTip({ text, label = "More information" }: { text: string; label?: string }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex normal-case tracking-normal">
      <button type="button" aria-label={label} aria-describedby={open ? id : undefined}
        className="rounded-full text-ink-faint hover:text-accent focus-visible:text-accent"
        onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
        onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}>
        <Info className="h-3 w-3" aria-hidden />
      </button>
      {open && (
        <span role="tooltip" id={id} className="absolute left-1/2 top-full z-40 mt-1.5 w-64 -translate-x-1/2 rounded-control border border-line-strong bg-surface-3 px-2.5 py-2 text-xs font-normal leading-relaxed text-ink" style={{ boxShadow: "var(--shadow-pop)" }}>
          {text}
        </span>
      )}
    </span>
  );
}

// ------------------------------------------------------------------ Empty / skeleton / banner
export function EmptyState({ icon, title, description, actions, compact }: {
  icon?: ReactNode; title: string; description?: ReactNode; actions?: ReactNode; compact?: boolean;
}) {
  return (
    <div className={clsx("flex flex-col items-center justify-center rounded-panel border border-dashed border-line-strong text-center", compact ? "px-4 py-6" : "grid-bg px-6 py-12")}>
      {icon && <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-control border border-line-strong bg-surface-2 text-ink-muted" aria-hidden>{icon}</div>}
      <p className="text-[0.875rem] font-semibold text-ink">{title}</p>
      {description && <div className="mt-1 max-w-md text-xs leading-relaxed text-ink-muted">{description}</div>}
      {actions && <div className="mt-4 flex flex-wrap justify-center gap-2">{actions}</div>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton", className)} aria-hidden />;
}

export function SkeletonRows({ rows = 6, cols = 6 }: { rows?: number; cols?: number }) {
  return (
    <div role="status" aria-label="Loading" className="space-y-2 p-3">
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="flex gap-3">
          {Array.from({ length: cols }, (_, c) => <Skeleton key={c} className="h-4 flex-1" />)}
        </div>
      ))}
    </div>
  );
}

const BANNER_CLASSES: Record<Tone, string> = {
  ok: "border-line border-l-ok bg-surface-2",
  warn: "border-warn/35 border-l-warn bg-warn/[0.06]",
  danger: "border-danger/40 border-l-danger bg-danger/[0.07]",
  info: "border-line border-l-info bg-surface-2",
  accent: "border-line border-l-accent bg-surface-2",
  neutral: "border-line border-l-line-strong bg-surface-2",
};

export function Banner({ tone, title, children, actions }: { tone: Tone; title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={clsx("flex flex-wrap items-start gap-3 rounded-panel border border-l-[3px] px-4 py-3", BANNER_CLASSES[tone])}>
      <StatusIcon tone={tone} className="mt-0.5" />
      <div className="min-w-0 flex-1">
        <p className="text-[0.8125rem] font-semibold text-ink">{title}</p>
        {children && <div className="mt-0.5 text-xs leading-relaxed text-ink-muted">{children}</div>}
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </div>
  );
}

// ------------------------------------------------------------------ Key/value list
export function KeyValueList({ items, columns = 1 }: { items: { label: string; value: ReactNode; mono?: boolean }[]; columns?: 1 | 2 }) {
  return (
    <dl className={clsx("grid gap-x-6 gap-y-2.5", columns === 2 && "sm:grid-cols-2")}>
      {items.map((it) => (
        <div key={it.label} className="flex min-w-0 items-baseline justify-between gap-3 border-b border-line/70 pb-2">
          <dt className="shrink-0 text-xs text-ink-muted">{it.label}</dt>
          <dd className={clsx("min-w-0 truncate text-right text-[0.8125rem] text-ink", it.mono && "font-mono text-xs")}>{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}

// ------------------------------------------------------------------ Switch
export function Switch({ checked, onChange, label, description, disabled }: {
  checked: boolean; onChange: (v: boolean) => void; label: string; description?: string; disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex items-center gap-2.5">
      <button id={id} type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
        className={clsx("relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors disabled:opacity-40",
          checked ? "border-accent/60 bg-accent/30" : "border-line-strong bg-surface-3")}>
        <span className={clsx("inline-block h-3.5 w-3.5 rounded-full transition-transform", checked ? "translate-x-[18px] bg-accent" : "translate-x-[3px] bg-ink-faint")} />
        <span className="sr-only">{label}</span>
      </button>
      <label htmlFor={id} className="cursor-pointer select-none text-xs text-ink-muted">
        <span className="text-ink">{label}</span>{description && <span className="block text-2xs text-ink-faint">{description}</span>}
      </label>
    </div>
  );
}

// ------------------------------------------------------------------ Progress
export function ProgressBar({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div>
      <div className="mb-1 flex justify-between text-2xs text-ink-muted"><span>{label}</span><span className="num">{pct.toFixed(0)}%</span></div>
      <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={max} aria-valuenow={value}
        className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
        <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

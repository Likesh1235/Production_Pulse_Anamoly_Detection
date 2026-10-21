import { CheckCircle2, AlertTriangle, XCircle, Info, X } from "lucide-react";
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import clsx from "clsx";

export type ToastTone = "ok" | "warn" | "danger" | "info";
interface Toast { id: number; tone: ToastTone; title: string; message?: string }
interface ToastValue { notify: (t: Omit<Toast, "id">) => void }

const ToastContext = createContext<ToastValue | null>(null);
const ICONS = { ok: CheckCircle2, warn: AlertTriangle, danger: XCircle, info: Info };
const TONES = { ok: "border-ok/40 text-ok", warn: "border-warn/40 text-warn", danger: "border-danger/50 text-danger", info: "border-info/40 text-info" };

/** Toasts confirm ACTUAL completed actions only (e.g. an API response was received). */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);
  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const notify = useCallback((t: Omit<Toast, "id">) => {
    const id = next.current++;
    setToasts((prev) => [...prev.slice(-3), { ...t, id }]);
    window.setTimeout(() => dismiss(id), 6000);
  }, [dismiss]);
  const value = useMemo(() => ({ notify }), [notify]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div aria-live="polite" aria-atomic="false" className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[min(92vw,360px)] flex-col gap-2">
        {toasts.map((t) => {
          const Icon = ICONS[t.tone];
          return (
            <div key={t.id} role="status" style={{ boxShadow: "var(--shadow-pop)" }} className={clsx("panel pointer-events-auto flex items-start gap-3 border-l-[3px] px-3 py-2.5", TONES[t.tone])}>
              <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <div className="min-w-0 flex-1">
                <p className="text-[0.8125rem] font-medium text-ink">{t.title}</p>
                {t.message && <p className="mt-0.5 text-xs text-ink-muted">{t.message}</p>}
              </div>
              <button type="button" className="btn-icon -mr-1 -mt-0.5" onClick={() => dismiss(t.id)} aria-label="Dismiss notification">
                <X className="h-3.5 w-3.5" aria-hidden />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastValue {
  const v = useContext(ToastContext);
  if (!v) throw new Error("useToast must be used inside <ToastProvider>");
  return v;
}

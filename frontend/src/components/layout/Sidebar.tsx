import clsx from "clsx";
import { PanelLeftClose, PanelLeftOpen, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { NavLink } from "react-router-dom";
import { ROUTES, ROUTE_GROUPS } from "../../routes";
import { APP_VERSION } from "../../lib/config";
import { fmtNum } from "../../lib/format";
import { MODEL_STATUS_LABELS } from "../../lib/api/types";
import { useStatus } from "../../state/StatusContext";
import { Wordmark } from "./Logo";

function NavItems({ collapsed, onNavigate }: { collapsed: boolean; onNavigate?: () => void }) {
  return (
    <div className="space-y-4">
      {ROUTE_GROUPS.map((group) => (
        <div key={group}>
          {collapsed ? <div className="mx-3 mb-1.5 border-t border-line" aria-hidden /> : (
            <p className="mb-1 px-2.5 text-xs font-medium text-ink-faint">{group}</p>
          )}
          <ul className="space-y-px">
            {ROUTES.filter((r) => r.group === group).map(({ path, label, icon: Icon, description }) => (
              <li key={path}>
                <NavLink to={path} end={path === "/"} onClick={onNavigate} title={collapsed ? label : description}
                  className={({ isActive }) => clsx(
                    "relative flex h-9 items-center gap-3 rounded-control px-2.5 text-[0.8125rem] transition-colors",
                    collapsed && "justify-center",
                    isActive ? "bg-surface-3 font-medium text-ink" : "text-ink-muted hover:bg-surface-3/60 hover:text-ink",
                  )}>
                  {({ isActive }) => (
                    <>
                      <span className={clsx("absolute inset-y-2 left-0 w-0.5 rounded-r bg-accent transition-opacity", isActive ? "opacity-100" : "opacity-0")} aria-hidden />
                      <Icon className={clsx("h-4 w-4 shrink-0", isActive && "text-accent")} aria-hidden />
                      {collapsed ? <span className="sr-only">{label}</span> : <span className="truncate">{label}</span>}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

/** Model state as reported by the API (nothing is shown as ready until /health and /model/status answer). */
function ModelCard() {
  const { connection, modelReady, threshold, modelStatus, health } = useStatus();
  const state = connection !== "online" ? { tone: "bg-ink-faint", text: connection === "checking" ? "Checking API…" : "API unreachable" }
    : modelReady ? { tone: "bg-ok", text: "Model ready" }
      : { tone: "bg-warn", text: health ? `Model: ${MODEL_STATUS_LABELS[health.model.status] ?? health.model.status}` : "Model state unknown" };
  return (
    <div className="mx-2 mb-2 rounded-control border border-line bg-surface-2 px-3 py-2.5 text-xs">
      <p className="flex items-center gap-2 font-medium text-ink"><span className={clsx("h-1.5 w-1.5 rounded-full", state.tone)} aria-hidden />{state.text}</p>
      {modelReady && (
        <dl className="mt-1.5 space-y-0.5 text-ink-faint">
          <div className="flex justify-between gap-2"><dt>Threshold</dt><dd className="tabular text-ink-muted">{threshold !== null ? fmtNum(threshold) : "—"}</dd></div>
          {modelStatus?.model_id && <div className="flex justify-between gap-2"><dt>Model id</dt><dd className="font-mono text-ink-muted">{modelStatus.model_id}</dd></div>}
        </dl>
      )}
    </div>
  );
}

/** Desktop navigation rail (collapsible). */
export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <aside aria-label="Primary" className={clsx(
      "sticky top-0 hidden h-screen shrink-0 flex-col border-r border-line bg-surface-1 transition-[width] duration-200 lg:flex",
      collapsed ? "w-[64px]" : "w-[236px]",
    )}>
      <div className={clsx("flex h-[60px] shrink-0 items-center border-b border-line", collapsed ? "justify-center" : "px-4")}>
        <Wordmark collapsed={collapsed} />
      </div>
      <nav id="primary-nav" aria-label="Main navigation" className="flex-1 overflow-y-auto px-2 py-4">
        <NavItems collapsed={collapsed} />
      </nav>
      {!collapsed && <ModelCard />}
      <div className={clsx("flex items-center border-t border-line px-2 py-2", collapsed ? "justify-center" : "justify-between")}>
        {!collapsed && <span className="pl-2 text-xs text-ink-faint">Dashboard v{APP_VERSION}</span>}
        <button type="button" className="btn-icon" onClick={onToggle} aria-controls="primary-nav" aria-expanded={!collapsed}
          aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}>
          {collapsed ? <PanelLeftOpen className="h-4 w-4" aria-hidden /> : <PanelLeftClose className="h-4 w-4" aria-hidden />}
        </button>
      </div>
    </aside>
  );
}

/** Mobile drawer navigation (modal dialog; Escape closes; focus returns to the trigger). */
export function MobileNav({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); previous?.focus?.(); };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 lg:hidden">
      <div className="absolute inset-0 bg-canvas/75" onClick={onClose} aria-hidden />
      <div role="dialog" aria-modal="true" aria-label="Navigation" className="absolute inset-y-0 left-0 flex w-[272px] flex-col border-r border-line bg-surface-1" style={{ boxShadow: "var(--shadow-pop)" }}>
        <div className="flex h-[60px] items-center justify-between border-b border-line px-4">
          <Wordmark />
          <button ref={closeRef} type="button" className="btn-icon" onClick={onClose} aria-label="Close navigation"><X className="h-4 w-4" aria-hidden /></button>
        </div>
        <nav aria-label="Main navigation" className="flex-1 overflow-y-auto px-2 py-4"><NavItems collapsed={false} onNavigate={onClose} /></nav>
        <ModelCard />
      </div>
    </div>
  );
}

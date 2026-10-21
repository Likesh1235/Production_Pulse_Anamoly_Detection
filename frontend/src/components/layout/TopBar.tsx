import { Menu, Moon, RefreshCw, Sun } from "lucide-react";
import { useLocation } from "react-router-dom";
import clsx from "clsx";
import { routeFor } from "../../routes";
import { useStatus } from "../../state/StatusContext";
import { useTheme } from "../../state/ThemeContext";
import { StatusPills } from "./StatusPills";

export function TopBar({ onOpenMobileNav }: { onOpenMobileNav: () => void }) {
  const { pathname } = useLocation();
  const meta = routeFor(pathname);
  const { refresh, refreshing } = useStatus();
  const { theme, toggle } = useTheme();
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-canvas/95 backdrop-blur">
      <div className="mx-auto flex min-h-[60px] w-full max-w-[1600px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2 sm:px-6">
        <button type="button" className="btn-icon lg:hidden" onClick={onOpenMobileNav} aria-label="Open navigation">
          <Menu className="h-5 w-5" aria-hidden />
        </button>
        <div className="min-w-0 flex-1">
          <nav aria-label="Breadcrumb" className="text-xs text-ink-faint">
            <span>{meta?.group ?? "Production Pulse"}</span>
            <span className="mx-1.5 text-line-strong" aria-hidden>/</span>
            <span className="text-ink-muted">{meta?.label ?? "Not found"}</span>
          </nav>
          <h1 className="truncate text-[1.0625rem] leading-snug text-ink">{meta?.title ?? "Page not found"}</h1>
        </div>
        <div className="order-last w-full sm:order-none sm:w-auto"><StatusPills /></div>
        <div className="flex items-center gap-0.5">
          <button type="button" className="btn-icon" onClick={() => void refresh()} aria-label="Refresh status" disabled={refreshing}
            title="Check the API, model and history again">
            <RefreshCw className={clsx("h-4 w-4", refreshing && "animate-spin")} aria-hidden />
          </button>
          <button type="button" className="btn-icon" onClick={toggle} aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
            {theme === "dark" ? <Sun className="h-4 w-4" aria-hidden /> : <Moon className="h-4 w-4" aria-hidden />}
          </button>
        </div>
      </div>
    </header>
  );
}

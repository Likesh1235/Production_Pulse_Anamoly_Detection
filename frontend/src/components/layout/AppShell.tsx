import { useCallback, useState } from "react";
import { Outlet } from "react-router-dom";
import { MobileNav, Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

const KEY = "production-pulse-nav-collapsed";

function initialCollapsed(): boolean {
  try { return window.localStorage.getItem(KEY) === "1"; } catch { return false; }
}

export function AppShell() {
  const [collapsed, setCollapsed] = useState(initialCollapsed);
  const [mobileOpen, setMobileOpen] = useState(false);
  const toggle = useCallback(() => setCollapsed((c) => {
    try { window.localStorage.setItem(KEY, c ? "0" : "1"); } catch { /* ignore */ }
    return !c;
  }), []);
  const closeMobile = useCallback(() => setMobileOpen(false), []);
  return (
    <div className="flex min-h-screen">
      <a href="#main" className="sr-only z-50 rounded bg-surface-3 px-3 py-2 text-accent focus:not-sr-only focus:fixed focus:left-3 focus:top-3">Skip to content</a>
      <Sidebar collapsed={collapsed} onToggle={toggle} />
      <MobileNav open={mobileOpen} onClose={closeMobile} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onOpenMobileNav={() => setMobileOpen(true)} />
        <main id="main" tabIndex={-1} className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-5 focus:outline-none sm:px-6 lg:py-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

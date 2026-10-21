import { Compass } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { ROUTES } from "../routes";
import { EmptyState, Panel } from "../components/ui/primitives";

export function NotFoundPage() {
  const { pathname } = useLocation();
  return (
    <Panel title="Page not found">
      <EmptyState icon={<Compass className="h-5 w-5" />} title="This page does not exist"
        description={<>There is no page at <span className="font-mono text-ink">{pathname}</span>. Choose one of the sections below.</>}
        actions={<Link to="/" className="btn-primary">Back to overview</Link>} />
      <nav aria-label="All sections" className="mt-4">
        <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {ROUTES.filter((r) => r.path !== "/").map(({ path, label, description, icon: Icon }) => (
            <li key={path}>
              <Link to={path} className="flex items-start gap-3 rounded-control border border-line px-3 py-2.5 transition-colors hover:border-accent/60 hover:bg-surface-3/50">
                <Icon className="mt-0.5 h-4 w-4 shrink-0 text-ink-muted" aria-hidden />
                <span><span className="block text-[0.8125rem] font-medium text-ink">{label}</span><span className="block text-xs text-ink-faint">{description}</span></span>
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </Panel>
  );
}

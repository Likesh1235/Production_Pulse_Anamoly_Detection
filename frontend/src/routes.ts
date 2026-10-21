import { Activity, FlaskConical, History, LayoutDashboard, Radar, Settings2, Workflow, type LucideIcon } from "lucide-react";

export type RouteGroup = "Monitoring" | "Analysis" | "System";
export interface RouteMeta { path: string; label: string; title: string; description: string; icon: LucideIcon; group: RouteGroup }

export const ROUTES: RouteMeta[] = [
  { path: "/", label: "Overview", title: "Operations overview", description: "API, model and stored-prediction summary", icon: LayoutDashboard, group: "Monitoring" },
  { path: "/monitor", label: "Live Monitor", title: "Live monitor", description: "User-initiated batch replay of a CSV file, not a live plant connection", icon: Activity, group: "Monitoring" },
  { path: "/predictions", label: "Predictions", title: "Prediction workbench", description: "Validate samples, score them with the API and inspect the results", icon: Workflow, group: "Monitoring" },
  { path: "/sensors", label: "Sensor Explorer", title: "Sensor explorer", description: "Measurements and reconstruction error side by side", icon: Radar, group: "Analysis" },
  { path: "/history", label: "History", title: "Prediction history", description: "Every prediction the API has stored", icon: History, group: "Analysis" },
  { path: "/evaluation", label: "Evaluation", title: "Evaluation and model performance", description: "Saved evaluation report on synthetic anomalies", icon: FlaskConical, group: "Analysis" },
  { path: "/settings", label: "Settings", title: "System and model settings", description: "Read-only configuration and model metadata", icon: Settings2, group: "System" },
];

export function routeFor(pathname: string): RouteMeta | undefined {
  return ROUTES.find((r) => (r.path === "/" ? pathname === "/" : pathname.startsWith(r.path)));
}

export const ROUTE_GROUPS: RouteGroup[] = ["Monitoring", "Analysis", "System"];

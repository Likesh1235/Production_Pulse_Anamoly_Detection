import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { OverviewPage } from "./pages/OverviewPage";
import { LiveMonitorPage } from "./pages/LiveMonitorPage";
import { PredictionsPage } from "./pages/PredictionsPage";
import { SensorExplorerPage } from "./pages/SensorExplorerPage";
import { HistoryPage } from "./pages/HistoryPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { SettingsPage } from "./pages/SettingsPage";
import { NotFoundPage } from "./pages/NotFoundPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<OverviewPage />} />
        <Route path="monitor" element={<LiveMonitorPage />} />
        <Route path="predictions" element={<PredictionsPage />} />
        <Route path="sensors" element={<SensorExplorerPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

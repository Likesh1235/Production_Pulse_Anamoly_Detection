import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import { App } from "./App";
import { AppProviders } from "./providers";
import { createApiClient } from "./lib/api/client";
import { API_BASE_URL, REQUEST_TIMEOUT_MS, STATUS_POLL_MS } from "./lib/config";

const client = createApiClient({ baseUrl: API_BASE_URL, timeoutMs: REQUEST_TIMEOUT_MS });
const root = document.getElementById("root");
if (!root) throw new Error("#root element missing in index.html");

createRoot(root).render(
  <StrictMode>
    <AppProviders client={client} pollMs={STATUS_POLL_MS}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AppProviders>
  </StrictMode>,
);

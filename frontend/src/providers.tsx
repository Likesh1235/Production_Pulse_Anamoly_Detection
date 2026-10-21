import type { ReactNode } from "react";
import type { ApiClient } from "./lib/api/client";
import { ApiClientProvider } from "./lib/api/ApiClientContext";
import { StatusProvider } from "./state/StatusContext";
import { SessionProvider } from "./state/SessionContext";
import { ThemeProvider } from "./state/ThemeContext";
import { ToastProvider } from "./state/ToastContext";

/** All app-wide providers in one place (also used by tests with a mock client). */
export function AppProviders({ client, pollMs, children }: { client: ApiClient; pollMs: number; children: ReactNode }) {
  return (
    <ThemeProvider>
      <ApiClientProvider client={client}>
        <StatusProvider pollMs={pollMs}>
          <SessionProvider>
            <ToastProvider>{children}</ToastProvider>
          </SessionProvider>
        </StatusProvider>
      </ApiClientProvider>
    </ThemeProvider>
  );
}

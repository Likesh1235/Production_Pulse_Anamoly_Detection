import { createContext, useContext, type ReactNode } from "react";
import type { ApiClient } from "./client";

const ApiClientContext = createContext<ApiClient | null>(null);

/** Provides the API client; tests pass a mock implementing the same interface. */
export function ApiClientProvider({ client, children }: { client: ApiClient; children: ReactNode }) {
  return <ApiClientContext.Provider value={client}>{children}</ApiClientContext.Provider>;
}

export function useApiClient(): ApiClient {
  const client = useContext(ApiClientContext);
  if (!client) throw new Error("useApiClient must be used inside <ApiClientProvider>");
  return client;
}

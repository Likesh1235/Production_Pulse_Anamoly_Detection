import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi, type Mock } from "vitest";
import type { ReactElement } from "react";
import type { ApiClient } from "../lib/api/client";
import { App } from "../App";
import { AppProviders } from "../providers";
import { healthReady, historyPage, metricsUnavailable, modelStatusReady, fakePredict } from "./fixtures";

/** Every endpoint method is a vitest mock; cast to ApiClient when handing it to the app. */
export type MockClient = { baseUrl: string } & Record<Exclude<keyof ApiClient, "baseUrl">, Mock>;

/** A mock API client; no backend or trained model is needed. */
export function mockClient(overrides: Partial<Record<keyof ApiClient, unknown>> = {}): MockClient {
  const client = {
    baseUrl: "http://api.test",
    health: vi.fn().mockResolvedValue(healthReady),
    modelStatus: vi.fn().mockResolvedValue(modelStatusReady),
    metrics: vi.fn().mockResolvedValue(metricsUnavailable),
    history: vi.fn().mockResolvedValue(historyPage([], 0, null)),
    predict: vi.fn().mockImplementation(async (body) => fakePredict(body)),
    ...overrides,
  };
  return client as unknown as MockClient;
}

export function renderWithProviders(ui: ReactElement, { client = mockClient(), route = "/" }: { client?: MockClient; route?: string } = {}) {
  return {
    client,
    ...render(
      <AppProviders client={client as unknown as ApiClient} pollMs={0}>
        <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
      </AppProviders>,
    ),
  };
}

export function renderRoute(route: string, client: MockClient = mockClient()) {
  return renderWithProviders(<App />, { client, route });
}

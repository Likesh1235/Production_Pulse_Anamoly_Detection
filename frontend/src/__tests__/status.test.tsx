import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiRequestError } from "../lib/api/errors";
import { healthNoModel, modelStatusMissing } from "../test/fixtures";
import { mockClient, renderRoute } from "../test/utils";

describe("status header", () => {
  it("shows a neutral checking state before /health answers — never green by default", () => {
    const client = mockClient({ health: vi.fn(() => new Promise(() => {})), modelStatus: vi.fn(() => new Promise(() => {})) });
    renderRoute("/settings", client);
    expect(screen.getByTestId("status-api")).toHaveTextContent("Checking…");
    expect(screen.getByTestId("status-model")).toHaveTextContent("Unknown");
    expect(screen.queryByText("Connected")).not.toBeInTheDocument();
  });

  it("reports API, model and database separately from real responses", async () => {
    renderRoute("/settings");
    expect(await screen.findByText("Connected")).toBeInTheDocument();
    expect(screen.getByTestId("status-model")).toHaveTextContent("Ready");
    expect(screen.getByTestId("status-db")).toHaveTextContent("Available");
  });

  it("shows an unavailable model while the API is up", async () => {
    renderRoute("/", mockClient({ health: vi.fn().mockResolvedValue(healthNoModel), modelStatus: vi.fn().mockResolvedValue(modelStatusMissing) }));
    expect(await screen.findByText("Connected")).toBeInTheDocument();
    expect(screen.getByTestId("status-model")).toHaveTextContent("No trained model");
  });

  it("shows the API as unreachable on network failure", async () => {
    const down = new ApiRequestError({ kind: "network", endpoint: "/health", message: "Could not reach http://api.test" });
    renderRoute("/", mockClient({ health: vi.fn().mockRejectedValue(down), modelStatus: vi.fn().mockRejectedValue(down) }));
    expect(await screen.findByText("Unreachable")).toBeInTheDocument();
    expect(screen.queryByText("Connected")).not.toBeInTheDocument();
    expect(screen.getByTestId("status-model")).toHaveTextContent("Unknown");
    const banner = screen.getAllByRole("alert")[0];
    expect(within(banner).getByText("API unreachable")).toBeInTheDocument();
  });
});

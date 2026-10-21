import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiRequestError } from "../lib/api/errors";
import { healthNoModel, historyPage, modelStatusMissing } from "../test/fixtures";
import { mockClient, renderRoute } from "../test/utils";

describe("overview", () => {
  it("explains what is missing when nothing is stored yet", async () => {
    renderRoute("/", mockClient({ health: vi.fn().mockResolvedValue(healthNoModel), modelStatus: vi.fn().mockResolvedValue(modelStatusMissing) }));
    expect(await screen.findByText("No predictions stored yet")).toBeInTheDocument();
    expect(screen.getAllByText("No trained model").length).toBeGreaterThan(0);
    expect(await screen.findByText("No evaluation report yet")).toBeInTheDocument();   // /metrics resolves independently of /history
    expect(screen.getByText("No evaluation report has been saved yet.")).toBeInTheDocument();
  });

  it("derives counts from GET /history totals only", async () => {
    const history = vi.fn().mockImplementation(async (q: { anomaliesOnly?: boolean }) =>
      q.anomaliesOnly ? historyPage([4], 2, 4, 1) : historyPage([4, 3, 2, 1], 4, null, 200));
    renderRoute("/", mockClient({ history }));
    expect(await screen.findByText("50.0% of stored records")).toBeInTheDocument();
  });

  it("does not invent values when the API is unreachable", async () => {
    const down = new ApiRequestError({ kind: "network", endpoint: "/health", message: "down" });
    const client = mockClient({ health: vi.fn().mockRejectedValue(down), modelStatus: vi.fn().mockRejectedValue(down) });
    renderRoute("/", client);
    expect(await screen.findByText("Offline")).toBeInTheDocument();
    expect(client.history).not.toHaveBeenCalled();
    expect(screen.queryByText(/of stored records/)).not.toBeInTheDocument();
  });
});

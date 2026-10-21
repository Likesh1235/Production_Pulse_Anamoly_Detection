import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiRequestError } from "../lib/api/errors";
import { historyPage } from "../test/fixtures";
import { mockClient, renderRoute } from "../test/utils";

describe("history page", () => {
  it("pages with the API's before_id cursor and back", async () => {
    const history = vi.fn().mockImplementation(async (q: { beforeId?: number }) =>
      q.beforeId === undefined ? historyPage([10, 9, 8], 5, 8, 3) : historyPage([7, 6], 5, null, 3));
    const user = userEvent.setup();
    renderRoute("/history", mockClient({ history }));
    expect(await screen.findByText("#10")).toBeInTheDocument();
    expect(history).toHaveBeenLastCalledWith(expect.objectContaining({ limit: 50, beforeId: undefined, anomaliesOnly: false }), expect.anything());
    await user.click(screen.getByRole("button", { name: /Older/ }));
    expect(await screen.findByText("#7")).toBeInTheDocument();
    expect(history).toHaveBeenLastCalledWith(expect.objectContaining({ beforeId: 8 }), expect.anything());
    expect(screen.queryByText("#10")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Older/ })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /Newer/ }));
    expect(await screen.findByText("#10")).toBeInTheDocument();
  });

  it("does not render duplicate records", async () => {
    const page = historyPage([4, 4, 3], 3, null);
    renderRoute("/history", mockClient({ history: vi.fn().mockResolvedValue(page) }));
    await screen.findByText("#3");
    expect(screen.getAllByText("#4")).toHaveLength(1);
  });

  it("uses the server-side anomalies filter and resets the cursor", async () => {
    const history = vi.fn().mockResolvedValue(historyPage([2], 1, null));
    const user = userEvent.setup();
    renderRoute("/history", mockClient({ history }));
    await screen.findByText("#2");
    await user.click(screen.getByRole("switch", { name: /Anomalies only/ }));
    await waitFor(() => expect(history).toHaveBeenLastCalledWith(expect.objectContaining({ anomaliesOnly: true, beforeId: undefined }), expect.anything()));
  });

  it("shows empty and error states", async () => {
    renderRoute("/history");
    expect(await screen.findByText("No predictions stored yet")).toBeInTheDocument();
  });

  it("shows a recoverable error when the database is unavailable", async () => {
    const err = new ApiRequestError({ kind: "http", status: 503, code: "database_unavailable", endpoint: "/history", message: "The prediction history database is unavailable." });
    renderRoute("/history", mockClient({ history: vi.fn().mockRejectedValue(err) }));
    expect(await screen.findByText(/History database unavailable/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});

import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { csvText } from "../test/fixtures";
import { mockClient, renderRoute } from "../test/utils";

describe("live monitor (user-initiated replay)", () => {
  it("is labelled as a replay and sends the file in batches", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    renderRoute("/monitor", client);
    await screen.findByText("Connected");
    expect(screen.getByText(/User-initiated replay — not a continuous plant stream/)).toBeInTheDocument();
    await user.upload(screen.getByLabelText("CSV file"), new File([csvText(5)], "held_out.csv", { type: "text/csv" }));
    fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "2" } });
    await user.selectOptions(screen.getByLabelText("Pause between batches"), "0");
    await user.click(screen.getByRole("button", { name: /Start replay/ }));
    await waitFor(() => expect(client.predict).toHaveBeenCalledTimes(3));
    expect(client.predict.mock.calls.map((c) => c[0].samples.length)).toEqual([2, 2, 1]);
    expect(await screen.findByText("Replay finished")).toBeInTheDocument();
  });

  it("explains the empty sensor explorer without inventing data", async () => {
    renderRoute("/sensors");
    expect(await screen.findByText("No session data to explore")).toBeInTheDocument();
  });
});

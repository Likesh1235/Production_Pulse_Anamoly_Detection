import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiRequestError } from "../lib/api/errors";
import { FEATURE_COLUMNS } from "../lib/features";
import { csvText, featureRecord, healthNoModel, modelStatusMissing } from "../test/fixtures";
import { mockClient, renderRoute, type MockClient } from "../test/utils";

async function openJson(client: MockClient = mockClient()) {
  const user = userEvent.setup();
  renderRoute("/predictions", client);
  await screen.findByText("Connected");
  await user.click(screen.getByRole("tab", { name: /Paste JSON/ }));
  return { user, client, textarea: screen.getByLabelText("Samples JSON") };
}

const batchJson = JSON.stringify({ samples: [{ features: featureRecord(0) }, { features: featureRecord(1) }, { features: featureRecord(2), sample_id: "c" }] });

describe("prediction workbench", () => {
  it("submits valid JSON and shows the API's results, including anomalies", async () => {
    const { user, client, textarea } = await openJson();
    await user.click(textarea);
    await user.paste(batchJson);
    expect(await screen.findByText(/Input is valid: 3 sample/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Submit 3 sample/ }));
    expect(await screen.findByRole("region", { name: "Per-sample results" })).toBeInTheDocument();
    expect(screen.getAllByRole("region", { name: "Prediction results" })).toHaveLength(1);   // the panel landmark is not duplicated
    expect(client.predict).toHaveBeenCalledTimes(1);
    expect(screen.getAllByText("Anomaly").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Normal").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.3").length).toBeGreaterThan(0);          // numeric value stays visible
  });

  it("sends features in canonical order regardless of input order", async () => {
    const { user, client, textarea } = await openJson();
    await user.click(textarea);
    await user.paste(JSON.stringify({ features: featureRecord(0, [...FEATURE_COLUMNS].reverse()) }));
    await user.click(await screen.findByRole("button", { name: /Submit 1 sample/ }));
    await waitFor(() => expect(client.predict).toHaveBeenCalled());
    const body = client.predict.mock.calls[0][0];
    expect(Object.keys(body.samples[0].features)).toEqual([...FEATURE_COLUMNS]);
    expect(body.samples[0].features).toEqual(featureRecord(0));
  });

  it.each([
    ["{bad", /Invalid JSON/],
    [JSON.stringify({ features: { ...featureRecord(), xmeas_2: "3.1" } }), /xmeas_2 is text/],
    [JSON.stringify({ features: Object.fromEntries(Object.entries(featureRecord()).filter(([k]) => k !== "xmv_11")) }), /Missing feature\(s\): xmv_11/],
  ])("blocks invalid JSON before submission (%#)", async (text, message) => {
    const { user, client, textarea } = await openJson();
    await user.click(textarea);
    await user.paste(text);
    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Submit/ })).toBeDisabled();
    expect(client.predict).not.toHaveBeenCalled();
  });

  it("accepts a valid CSV upload in any column order", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    renderRoute("/predictions", client);
    await screen.findByText("Connected");
    const file = new File([csvText(4, [...FEATURE_COLUMNS].reverse())], "held_out.csv", { type: "text/csv" });
    await user.upload(screen.getByLabelText("CSV file"), file);
    expect(await screen.findByText(/Input is valid: 4 sample/)).toBeInTheDocument();
    expect(screen.getByText("held_out.csv")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Submit 4 sample/ }));
    await waitFor(() => expect(client.predict).toHaveBeenCalledTimes(1));
  });

  it("reports CSV problems with line context", async () => {
    const user = userEvent.setup();
    renderRoute("/predictions");
    await screen.findByText("Connected");
    const lines = csvText(2).split("\n");
    lines[1] = lines[1].replace(/^[^,]+/, "abc");
    await user.upload(screen.getByLabelText("CSV file"), new File([lines.join("\n")], "bad.csv", { type: "text/csv" }));
    expect(await screen.findByText(/xmeas_1 is not a number/)).toBeInTheDocument();
    expect(screen.getByText("line 2")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Submit/ })).toBeDisabled();
  });

  it.each([
    [new ApiRequestError({ kind: "http", status: 422, code: "invalid_features", endpoint: "/predict", message: "Each sample must contain exactly the 52 TEP measurement features (any order).", details: [{ sample_index: 0, unexpected_features: ["flow"] }] }), "Feature names rejected", /Sample 1: unexpected flow/],
    [new ApiRequestError({ kind: "http", status: 413, code: "batch_too_large", endpoint: "/predict", message: "At most 2 samples per request (got 3)." }), "Batch too large", /At most 2 samples per request/],
    [new ApiRequestError({ kind: "http", status: 503, code: "model_unavailable", endpoint: "/predict", message: "No trained model was found.", details: { model_status: "not_found" } }), "Model unavailable", /No trained model was found/],
    [new ApiRequestError({ kind: "network", endpoint: "/predict", message: "Could not reach http://api.test" }), "API unreachable", /Could not connect/],
    [new ApiRequestError({ kind: "timeout", endpoint: "/predict", message: "No response" }), "Request timed out", /did not respond in time/],
  ])("handles failures without fabricating results (%#)", async (error, title, message) => {
    const client = mockClient({ predict: vi.fn().mockRejectedValue(error) });
    const { user, textarea } = await openJson(client);
    await user.click(textarea);
    await user.paste(batchJson);
    await user.click(await screen.findByRole("button", { name: /Submit 3 sample/ }));
    expect(await screen.findByText(new RegExp(title))).toBeInTheDocument();
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Per-sample results" })).not.toBeInTheDocument();
    expect(screen.queryByText("Samples scored")).not.toBeInTheDocument();
  });

  it("disables submission when the model is not ready", async () => {
    const client = mockClient({ health: vi.fn().mockResolvedValue(healthNoModel), modelStatus: vi.fn().mockResolvedValue(modelStatusMissing) });
    const { user, textarea } = await openJson(client);
    await user.click(textarea);
    await user.paste(batchJson);
    expect(await screen.findByText(/The model is not ready/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Submit/ })).toBeDisabled();
    expect(screen.getByText(/Model not ready/)).toBeInTheDocument();
  });
});

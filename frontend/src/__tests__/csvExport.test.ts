import { describe, expect, it } from "vitest";
import { resultsToCsv } from "../lib/csvExport";
import type { PredictionResult } from "../lib/api/types";

function result(sample_id: string | null, error = 0.25): PredictionResult {
  return { index: 0, sample_id, prediction_id: 7, reconstruction_error: error, is_anomaly: false, threshold: 0.885,
    top_features: [{ feature: "xmeas_1", squared_error: 1.5, share: 0.4 }] };
}

describe("resultsToCsv", () => {
  it("neutralises spreadsheet formulas in user-supplied text (CSV injection)", () => {
    for (const evil of ["=HYPERLINK(\"http://x\")", "+1+1", "-2+3", "@SUM(A1)", "\t=1", "\r=1"]) {
      const csv = resultsToCsv([{ result: result(evil), requestId: "r1", modelId: "m1" }]);
      const row = csv.split("\r\n")[1];
      const sampleCell = row.split(",")[1].replace(/^"/, "");
      expect(sampleCell.startsWith("'")).toBe(true);
    }
    const ts = resultsToCsv([{ result: result("ok"), requestId: "r1", modelId: "m1",
      input: { features: {}, sourceRow: 2, timestamp: "=cmd|' /C calc'!A0" } as never }]);
    expect(ts).toContain("'=cmd");
  });

  it("leaves numbers, ordinary text and negative numeric values untouched", () => {
    const csv = resultsToCsv([{ result: result("line-3", 0.123), requestId: "r1", modelId: "m1" }]);
    const row = csv.split("\r\n")[1].split(",");
    expect(row[1]).toBe("line-3");
    expect(row[7]).toBe("0.123");
    expect(row[10]).toBe("xmeas_1");
  });
});

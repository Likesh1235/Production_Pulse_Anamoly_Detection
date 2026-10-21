import { describe, expect, it } from "vitest";
import { detailLines } from "../lib/api/errors";

describe("detailLines", () => {
  it("lists the features of a value_out_of_range error instead of a generic label", () => {
    const lines = detailLines([{ sample_index: 0, features: ["xmeas_1"], reason: "scaled value exceeds float32 range" }]);
    expect(lines).toEqual(["Sample 1: xmeas_1 (scaled value exceeds float32 range)"]);
  });

  it("still formats invalid_features details", () => {
    expect(detailLines([{ sample_index: 1, missing_features: ["xmv_3"] }])).toEqual(["Sample 2: missing xmv_3"]);
  });
});

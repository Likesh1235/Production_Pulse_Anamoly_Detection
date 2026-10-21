import { describe, expect, it } from "vitest";
import { FEATURE_COLUMNS } from "../lib/features";
import { batchLimitIssue, buildPredictRequest, chunk, isSubmittable, parseCsv, parseJson, parseNumericCell } from "../lib/validation";
import { csvText, featureRecord } from "../test/fixtures";

const F = FEATURE_COLUMNS;
const errors = (r: { issues: { severity: string; message: string }[] }) => r.issues.filter((i) => i.severity === "error").map((i) => i.message).join(" | ");

describe("parseNumericCell", () => {
  it.each([["1.5", 1.5], ["-2", -2], ["3e-4", 3e-4], [" .5 ", 0.5], ["+7.", 7]])("accepts %s", (raw, v) => {
    expect(parseNumericCell(raw)).toEqual({ ok: true, value: v });
  });
  it.each([["", "empty"], ["abc", "non-numeric"], ["1,5", "non-numeric"], ["NaN", "non-finite"], ["-Infinity", "non-finite"], ["1e400", "non-finite"]])(
    "rejects %s", (raw, reason) => { expect(parseNumericCell(raw)).toEqual({ ok: false, reason }); });
});

describe("parseCsv", () => {
  it("parses a valid file", () => {
    const r = parseCsv(csvText(3), F);
    expect(r.errorCount).toBe(0);
    expect(r.samples).toHaveLength(3);
    expect(r.samples[0].sourceRow).toBe(2);
    expect(isSubmittable(r, 1000)).toBe(true);
  });

  it("is independent of column order", () => {
    const a = parseCsv(csvText(2), F);
    const b = parseCsv(csvText(2, [...F].reverse()), F);
    expect(buildPredictRequest(b.samples, F)).toEqual(buildPredictRequest(a.samples, F));
    expect(Object.keys(buildPredictRequest(b.samples, F).samples[0].features)).toEqual([...F]);
  });

  it("reports missing and unexpected columns", () => {
    const missing = parseCsv(csvText(1, F.filter((c) => c !== "xmv_3")), F);
    expect(errors(missing)).toMatch(/Missing feature column\(s\): xmv_3/);
    const extra = parseCsv(csvText(1, F, { header: "pressure_kpa", values: () => "1" }), F);
    expect(errors(extra)).toMatch(/Unexpected column\(s\): pressure_kpa/);
    expect(extra.samples).toHaveLength(0);
  });

  it("accepts TEP metadata/helper columns without sending them", () => {
    const text = csvText(2, F, { header: "simulationRun", values: () => "7" }).replace(/^/, "").split("\n")
      .map((line, i) => (i === 0 ? `${line},sample,timestamp` : `${line},${i},2026-10-20T10:0${i}:00Z`)).join("\n");
    const r = parseCsv(text, F);
    expect(r.errorCount).toBe(0);
    expect(r.ignoredColumns).toEqual(["simulationRun", "sample", "timestamp"]);
    expect(r.samples[0].sampleId).toBe("run-7-sample-1");
    expect(r.hasTimestamps).toBe(true);
    expect(Object.keys(buildPredictRequest(r.samples, F).samples[0].features)).not.toContain("simulationRun");
  });

  it("reports non-numeric and non-finite cells with line and feature", () => {
    const lines = csvText(2).split("\n");
    const cells = lines[2].split(",");
    cells[F.indexOf("xmeas_4")] = "high";
    cells[F.indexOf("xmv_1")] = "NaN";
    lines[2] = cells.join(",");
    const r = parseCsv(lines.join("\n"), F);
    expect(r.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({ row: 3, feature: "xmeas_4", severity: "error" }),
      expect.objectContaining({ row: 3, feature: "xmv_1", message: expect.stringMatching(/not finite/) }),
    ]));
    expect(r.samples).toHaveLength(1);
    expect(isSubmittable(r, 1000)).toBe(false);
  });

  it("rejects empty files, header-only files and duplicate headers", () => {
    expect(errors(parseCsv("   ", F))).toMatch(/empty/);
    expect(errors(parseCsv(F.join(","), F))).toMatch(/no data rows/);
    expect(errors(parseCsv(`${F.join(",")},xmeas_1\n${Array(53).fill(1).join(",")}`, F))).toMatch(/Duplicate column/);
  });
});

describe("parseJson", () => {
  it("accepts one sample, a batch and a bare array", () => {
    expect(parseJson(JSON.stringify({ features: featureRecord() }), F).samples).toHaveLength(1);
    const batch = parseJson(JSON.stringify({ samples: [{ features: featureRecord(0) }, { features: featureRecord(1), sample_id: "b" }] }), F);
    expect(batch.errorCount).toBe(0);
    expect(batch.samples[1].sampleId).toBe("b");
    expect(parseJson(JSON.stringify([{ features: featureRecord() }]), F).samples).toHaveLength(1);
  });

  it("is independent of key order", () => {
    const a = parseJson(JSON.stringify({ features: featureRecord(0) }), F);
    const b = parseJson(JSON.stringify({ features: featureRecord(0, [...F].reverse()) }), F);
    expect(buildPredictRequest(b.samples, F)).toEqual(buildPredictRequest(a.samples, F));
  });

  it.each([
    ["{bad", /Invalid JSON/],
    ['{"features": {"xmeas_1": NaN}}', /NaN and Infinity/],
    [JSON.stringify({ features: featureRecord(), samples: [] }), /not both/],
    [JSON.stringify({ samples: [] }), /no samples/],
    [JSON.stringify({ rows: [] }), /Expected/],
    [JSON.stringify({ features: featureRecord(), label: 1 }), /Unexpected top-level/],
  ])("rejects %s", (text, pattern) => {
    expect(errors(parseJson(text, F))).toMatch(pattern);
  });

  it("reports missing/unexpected features and non-numeric values per sample", () => {
    const feats = featureRecord();
    delete feats.xmv_11;
    const bad = { ...featureRecord(), xmeas_2: "3.1", simulationRun: 4 } as Record<string, unknown>;
    const r = parseJson(JSON.stringify({ samples: [{ features: feats }, { features: bad }] }), F);
    const text = errors(r);
    expect(text).toMatch(/Missing feature\(s\): xmv_11/);
    expect(text).toMatch(/Unexpected feature\(s\): simulationRun/);
    expect(text).toMatch(/xmeas_2 is text/);
    expect(r.samples).toHaveLength(0);
  });
});

describe("batch limits", () => {
  it("flags oversized batches and chunks correctly", () => {
    expect(batchLimitIssue(1000, 1000)).toBeNull();
    expect(batchLimitIssue(1001, 1000)?.message).toMatch(/exceed the batch limit of 1,000/);
    const r = parseCsv(csvText(3), F);
    expect(isSubmittable(r, 2)).toBe(false);
    expect(chunk([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });
});

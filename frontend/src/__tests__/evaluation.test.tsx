import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiRequestError } from "../lib/api/errors";
import type { MetricsResponse } from "../lib/api/types";
import { mockClient, renderRoute } from "../test/utils";

/** TEST-ONLY synthetic report with the structure written by notebook 03 §6b. */
const report: MetricsResponse = {
  available: true,
  report: {
    report_version: 1, generated_utc: "2026-10-20T12:00:00+00:00", source: "notebooks/03_autoencoder_training.ipynb", model_id: "abcdef123456",
    metrics: {
      setup: { seed: 42, anomaly_rate: 0.05 },
      clean_false_alarm_rate: 0.0123,
      detector_comparison: [
        { family: "correlation_break", detector: "rule_check", n: 1000, n_positive: 100, precision: 0.1, recall: 0.05, f1: 0.07, false_positive_rate: 0.04, pr_auc: null, roc_auc: 0.5, tp: 5, fp: 45, tn: 900, fn: 95 },
        { family: "correlation_break", detector: "autoencoder", n: 1000, n_positive: 100, precision: 0.6, recall: 0.4, f1: 0.48, false_positive_rate: 0.01, pr_auc: 0.5, roc_auc: 0.8, tp: 40, fp: 27, tn: 918, fn: 60 },
      ],
      hypothesis_assessment_autoencoder: [
        { family: "correlation_break", verdict: "NOT SUPPORTED", in_range_fraction: 1, single_feature_recall: 0.05, single_feature_fpr: 0.04, multivariate_recall: 0.4, marginal_auc_excess: 0.01, reasons: "multivariate detector misses it" },
      ],
    },
  },
};

describe("evaluation page", () => {
  it("shows 'No evaluation report yet' with the API's reason", async () => {
    renderRoute("/evaluation");
    expect(await screen.findByText("No evaluation report yet")).toBeInTheDocument();
    expect(screen.getByText("No evaluation report has been saved yet.")).toBeInTheDocument();
    expect(screen.queryByText(/Precision/)).not.toBeInTheDocument();
  });

  it("does not claim 'no report' while the API status is still being checked", async () => {
    const never = () => new Promise<never>(() => {});
    const client = mockClient({ health: vi.fn().mockImplementation(never), modelStatus: vi.fn().mockImplementation(never) });
    renderRoute("/evaluation", client);
    expect(await screen.findByRole("status", { name: "Loading" })).toBeInTheDocument();
    expect(screen.queryByText("No evaluation report yet")).not.toBeInTheDocument();
    expect(client.metrics).not.toHaveBeenCalled();
  });

  it("renders only values present in a saved report", async () => {
    renderRoute("/evaluation", mockClient({ metrics: vi.fn().mockResolvedValue(report) }));
    expect(await screen.findByText("Detector comparison")).toBeInTheDocument();
    expect(screen.getAllByText("autoencoder").length).toBeGreaterThan(0);
    expect(screen.getAllByText("undefined").length).toBeGreaterThan(0);     // null PR-AUC is not shown as 0
    expect(screen.getByText("Not supported")).toBeInTheDocument();
    expect(screen.getByText("multivariate detector misses it")).toBeInTheDocument();
    expect(screen.getByText("1.23%")).toBeInTheDocument();
  });

  it("states that the anomalies are synthetic and gives the PR-AUC random reference", async () => {
    renderRoute("/evaluation", mockClient({ metrics: vi.fn().mockResolvedValue(report) }));
    expect(await screen.findByText(/Synthetic anomalies — not proof of real-fault detection/)).toBeInTheDocument();
    const note = await screen.findByTestId("operating-point-note");
    expect(note).toHaveTextContent("10.00%");                       // 100 / 1000 anomalous rows
    expect(note).toHaveTextContent(/PR-AUC of about\s*0\.1,/);       // random reference = prevalence
    expect(note).toHaveTextContent(/similar/);                       // operating-point caveat
  });

  it("omits the prevalence sentence when the report has no row counts", async () => {
    const bare: MetricsResponse = JSON.parse(JSON.stringify(report));
    if (bare.available) {
      for (const r of bare.report.metrics.detector_comparison as Record<string, unknown>[]) { delete r.n; delete r.n_positive; }
    }
    renderRoute("/evaluation", mockClient({ metrics: vi.fn().mockResolvedValue(bare) }));
    const note = await screen.findByTestId("operating-point-note");
    expect(note).not.toHaveTextContent(/PR-AUC of about/);
    expect(note).toHaveTextContent(/similar/);
  });

  it("summarises autoencoder vs PCA from the report's bootstrap readings, without claiming more", async () => {
    const withPca: MetricsResponse = JSON.parse(JSON.stringify(report));
    if (withPca.available) {
      const m = withPca.report.metrics;
      (m.detector_comparison as Record<string, unknown>[]).push(
        { family: "correlation_break", detector: "pca", precision: 0.7, recall: 0.5, f1: 0.58, false_positive_rate: 0.01, pr_auc: 0.6, roc_auc: 0.85 });
      m.autoencoder_minus_baselines = [
        { family: "correlation_break", metric: "pr_auc", autoencoder_minus: "pca", mean: -0.1, ci_low: -0.12, ci_high: -0.08, reading: "autoencoder lower" },
      ];
    }
    renderRoute("/evaluation", mockClient({ metrics: vi.fn().mockResolvedValue(withPca) }));
    const region = await screen.findByRole("region", { name: "Autoencoder versus pca" });
    expect(region).toHaveTextContent("autoencoder lower");
    expect(region).toHaveTextContent("-0.1");
    expect(screen.getByText(/lower on/)).toHaveTextContent(/higher on 0, no clear difference on 0, lower on 1 of 1 families/);
  });

  it("marks differences as point estimates when the report has no intervals", async () => {
    renderRoute("/evaluation", mockClient({ metrics: vi.fn().mockResolvedValue(report) }));
    expect(await screen.findByText(/has no confidence intervals/)).toBeInTheDocument();
  });

  it("surfaces connection errors instead of an empty report", async () => {
    const err = new ApiRequestError({ kind: "network", endpoint: "/metrics", message: "down" });
    renderRoute("/evaluation", mockClient({ metrics: vi.fn().mockRejectedValue(err) }));
    expect(await screen.findByText(/API unreachable/)).toBeInTheDocument();
  });
});

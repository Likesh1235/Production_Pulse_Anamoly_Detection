import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiClient, historyQueryString } from "../lib/api/client";
import { ApiRequestError, describeError } from "../lib/api/errors";
import { healthReady, fakePredict, featureRecord, historyPage } from "../test/fixtures";

function response(status: number, body: unknown, raw?: string) {
  return new Response(raw ?? JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function clientWith(fetchImpl: typeof fetch, timeoutMs = 2000) {
  return createApiClient({ baseUrl: "http://api.test/", timeoutMs, fetchImpl });
}

afterEach(() => vi.useRealTimers());

describe("api client", () => {
  it("parses a valid health response and strips the trailing slash", async () => {
    const f = vi.fn().mockResolvedValue(response(200, healthReady));
    const c = clientWith(f as unknown as typeof fetch);
    await expect(c.health()).resolves.toEqual(healthReady);
    expect(f.mock.calls[0][0]).toBe("http://api.test/health");
  });

  it("posts predictions as JSON", async () => {
    const body = { samples: [{ features: featureRecord() }] };
    const f = vi.fn().mockResolvedValue(response(200, fakePredict(body)));
    await clientWith(f as unknown as typeof fetch).predict(body);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("http://api.test/predict");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(body);
  });

  it.each([
    [422, "invalid_features", "Each sample must contain exactly the 52 TEP measurement features (any order)."],
    [413, "batch_too_large", "At most 1000 samples per request (got 1200)."],
    [503, "model_unavailable", "No trained model was found in the configured model directory."],
  ])("maps HTTP %s error envelopes", async (status, code, message) => {
    const f = vi.fn().mockResolvedValue(response(status, { error: { code, message, details: code === "invalid_features" ? [{ sample_index: 0, missing_features: ["xmv_3"] }] : undefined } }));
    const err = await clientWith(f as unknown as typeof fetch).predict({ samples: [] }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiRequestError);
    expect(err).toMatchObject({ kind: "http", status, code, message });
    if (code === "invalid_features") expect(describeError(err, "x").items).toEqual(["Sample 1: missing xmv_3"]);
  });

  it("treats non-envelope error bodies as http_error", async () => {
    const f = vi.fn().mockResolvedValue(response(502, null, "<html>Bad gateway</html>"));
    await expect(clientWith(f as unknown as typeof fetch).health()).rejects.toMatchObject({ kind: "http", status: 502, code: "http_error" });
  });

  it("distinguishes network failures from API errors", async () => {
    const f = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await clientWith(f as unknown as typeof fetch).health().catch((e) => e);
    expect(err).toMatchObject({ kind: "network" });
    expect(err.isConnectionProblem).toBe(true);
    expect(describeError(err, "http://api.test").title).toBe("API unreachable");
  });

  it("times out", async () => {
    vi.useFakeTimers();
    const f = vi.fn().mockImplementation((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    }));
    const pending = clientWith(f as unknown as typeof fetch, 1000).health().catch((e) => e);
    await vi.advanceTimersByTimeAsync(1001);
    const err = await pending;
    expect(err).toMatchObject({ kind: "timeout" });
    expect(describeError(err, "x").title).toBe("Request timed out");
  });

  it("supports caller cancellation", async () => {
    const controller = new AbortController();
    const f = vi.fn().mockImplementation((_url: string, init: RequestInit) => new Promise((_r, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    }));
    const pending = clientWith(f as unknown as typeof fetch).health(controller.signal).catch((e) => e);
    controller.abort();
    expect(await pending).toMatchObject({ kind: "aborted" });
  });

  it.each([
    ["invalid JSON", "{not json"],
    ["wrong shape", JSON.stringify({ status: "ok" })],
    ["wrong field type", JSON.stringify({ ...healthReady, model: { available: "yes", status: "ready" } })],
  ])("rejects malformed responses (%s)", async (_name, raw) => {
    const f = vi.fn().mockResolvedValue(response(200, null, raw));
    await expect(clientWith(f as unknown as typeof fetch).health()).rejects.toMatchObject({ kind: "malformed" });
  });

  it("builds history query strings from the real parameters", async () => {
    expect(historyQueryString({})).toBe("");
    expect(historyQueryString({ limit: 50, beforeId: 17, anomaliesOnly: true })).toBe("?limit=50&before_id=17&anomalies_only=true");
    const f = vi.fn().mockResolvedValue(response(200, historyPage([3, 2], 2, null)));
    await clientWith(f as unknown as typeof fetch).history({ limit: 2 });
    expect(f.mock.calls[0][0]).toBe("http://api.test/history?limit=2");
  });
});

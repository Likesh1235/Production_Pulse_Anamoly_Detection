import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());

// jsdom lacks ResizeObserver (used by Recharts' ResponsiveContainer)
class ResizeObserverStub { observe() {} unobserve() {} disconnect() {} }
if (!("ResizeObserver" in globalThis)) {
  (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverStub;
}

if (typeof window.matchMedia !== "function") {
  window.matchMedia = (query: string) => ({
    matches: false, media: query, onchange: null,
    addEventListener: () => {}, removeEventListener: () => {}, addListener: () => {}, removeListener: () => {},
    dispatchEvent: () => false,
  }) as unknown as MediaQueryList;
}

if (!URL.createObjectURL) URL.createObjectURL = () => "blob:test";
if (!URL.revokeObjectURL) URL.revokeObjectURL = () => {};

// Blob.text() polyfill for older jsdom versions
if (typeof Blob.prototype.text !== "function") {
  Blob.prototype.text = function text(this: Blob) {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(this);
    });
  };
}

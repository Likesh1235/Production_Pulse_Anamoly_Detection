import { useMemo } from "react";
import { useTheme } from "../state/ThemeContext";

/** Resolve theme tokens to concrete colors for SVG charts (falls back to dark-theme values). */
const FALLBACK: Record<string, string> = {
  accent: "91 188 208", teal: "79 181 160", warn: "226 162 59", danger: "230 104 96", ok: "82 181 127",
  info: "127 167 216", "ink-muted": "155 165 177", "ink-faint": "112 122 134", line: "41 49 59", "surface-2": "22 27 33",
};

function read(name: string): string {
  const v = typeof window !== "undefined" ? getComputedStyle(document.documentElement).getPropertyValue(`--c-${name}`).trim() : "";
  return `rgb(${v || FALLBACK[name]})`;
}

export interface ChartColors {
  accent: string; teal: string; warn: string; danger: string; ok: string; info: string;
  muted: string; faint: string; grid: string; surface: string; series: string[];
}

export function useChartColors(): ChartColors {
  const { theme } = useTheme();
  return useMemo(() => {
    const c = {
      accent: read("accent"), teal: read("teal"), warn: read("warn"), danger: read("danger"), ok: read("ok"),
      info: read("info"), muted: read("ink-muted"), faint: read("ink-faint"), grid: read("line"), surface: read("surface-2"),
    };
    // Series palette avoids amber/red/green, which are reserved for alarm limit, anomaly and normal state.
    const series = [c.accent, c.info, "rgb(176 150 210)", c.teal, "rgb(150 160 172)"];
    return { ...c, series };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
}

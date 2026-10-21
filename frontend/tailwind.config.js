/** Design tokens live in src/index.css as CSS variables (RGB triplets) so dark/light themes
 *  switch without rebuilding. Tailwind maps semantic names to those variables. */
const token = (name) => `rgb(var(--c-${name}) / <alpha-value>)`;

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        canvas: token("canvas"),
        surface: { 1: token("surface-1"), 2: token("surface-2"), 3: token("surface-3") },
        line: { DEFAULT: token("line"), strong: token("line-strong") },
        ink: { DEFAULT: token("ink"), muted: token("ink-muted"), faint: token("ink-faint") },
        accent: { DEFAULT: token("accent"), strong: token("accent-strong") },
        teal: token("teal"),
        ok: token("ok"),
        warn: token("warn"),
        danger: token("danger"),
        info: token("info"),
      },
      fontFamily: {
        // IBM Plex is self-hosted through @fontsource (works offline during a demo).
        sans: ["IBM Plex Sans", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "Cascadia Mono", "Consolas", "monospace"],
      },
      fontSize: { "2xs": ["0.6875rem", { lineHeight: "1rem" }] },
      borderRadius: { panel: "var(--radius-panel)", control: "var(--radius-control)" },
      boxShadow: {
        panel: "var(--shadow-panel)",
        glow: "0 0 0 1px rgb(var(--c-accent) / 0.45)",
        "glow-warn": "0 0 0 1px rgb(var(--c-warn) / 0.45)",
      },
      transitionTimingFunction: { snappy: "cubic-bezier(.2,.8,.2,1)" },
    },
  },
  plugins: [],
};

# Production Pulse — Dashboard (React + Tailwind)

*Production Pulse — Industrial Intelligence*: the web dashboard for the FastAPI backend (`docs/API.md`).

> **Status (2026-10-21):** all frontend files are written, but **nothing has been installed, type-checked, built or run yet**, and the component tests have **not** been executed. A trained model and an evaluation report do not exist yet either. Do not treat the dashboard as working until it has been run against the backend (final verification stage).
>
> **Checked by reading the code (2026-10-21), not by running it:** the endpoints, query parameters, response fields, error codes and saved-report keys that the dashboard uses agree with `src/production_pulse/api/` and with what notebook 03 writes. This is a consistency review. It is not a test.

---

## 1. Prerequisites and checks (Windows PowerShell)

The dashboard needs **Node.js ≥ 18.18** (an LTS version, 20 or 22, is recommended) and **npm**. The 2026-10-21 preflight found **Node v22.20.0** and **npm 11.6.1** on your Windows PATH (`logs/verification/00_node_version.out.txt`, `00_npm_version.out.txt`); they have not yet been used for this project. Check them yourself with:

```powershell
node --version
npm --version
```

* "not recognized": install the Node.js LTS from https://nodejs.org (it includes npm), then open a **new** terminal.
* If PowerShell blocks `npm` with an *execution policy* error, use `npm.cmd` instead of `npm` in every command below. No system setting needs to change.
* Nothing is installed globally. Packages go into `frontend\node_modules\` only, which `.gitignore` excludes.

## 2. Install and run (later, at the verification stage)

```powershell
cd "<project folder>\frontend"
npm install            # reads package.json, creates node_modules\ and package-lock.json
npm run dev            # http://127.0.0.1:5173  (the port must match the API's CORS origins)
```

Other scripts:

| command | purpose |
|---|---|
| `npm run typecheck` | TypeScript check (`tsc --noEmit`) |
| `npm test` | Vitest + React Testing Library (all API calls mocked; no backend or model needed) |
| `npm run build` | production build into `frontend\dist\` |
| `npm run preview` | serve the build at http://127.0.0.1:4173 (add this origin to the API's CORS list first) |

**OneDrive tip:** `node_modules` contains tens of thousands of files. Consider pausing OneDrive sync during `npm install`, or excluding the folder from sync.

## 3. Configuration (`frontend\.env.local`)

Copy the example and edit it:

```powershell
Copy-Item .env.example .env.local
```

| variable | default | meaning |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL (no trailing slash) |
| `VITE_MAX_BATCH_SIZE` | `1000` | must equal `[api].max_batch_size`. The API does not publish its limit; it answers **413** if the frontend sends more |
| `VITE_REQUEST_TIMEOUT_MS` | `15000` | per-request timeout |
| `VITE_STATUS_POLL_MS` | `20000` | `/health` + `/model/status` polling interval (`0` turns it off; the top-bar refresh button always works) |

`VITE_*` values are compiled into the browser bundle and visible to every user, so **never put secrets there**. The API's admin token (`PP_API_ADMIN_TOKEN`) is not used by the dashboard at all. Restart `npm run dev` after changing `.env.local`.

## 4. Backend: start it and allow the dashboard's origin

1. **Model:** run `notebooks/03_autoencoder_training.ipynb`.
   * It saves the model to `artifacts/models/autoencoder_v1/`, the API's default `model_dir`.
   * Section 6b writes `reports/evaluation_metrics.json`, which the Evaluation page shows.
2. **CORS:** the API only accepts browser requests from explicit origins (`[api].cors_origins`). The defaults cover the common Vite dev ports: `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:5174`, and `http://127.0.0.1:5174`. For another origin, for example `npm run preview` on port 4173:
   ```powershell
   $env:PP_API_CORS_ORIGINS = "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:4173"
   ```
   `"*"` is rejected by the backend on purpose.
3. **Start the API** in a second terminal, from the project root:
   ```powershell
   cd "<project folder>"
   .\.venv313\Scripts\python.exe -m uvicorn production_pulse.api.main:app --app-dir src --host 127.0.0.1 --port 8000
   ```
4. Open http://127.0.0.1:5173. The top bar reads **API: Connected** only after `GET /health` has actually answered.

## 5. Pages

| page | data source | notes |
|---|---|---|
| **Overview** | `/health`, `/model/status`, `/history` (latest 200 + anomaly total), `/metrics` | Counts come from the history `total` fields. The timeline x-axis is the stored record ID. A checklist shows what is missing |
| **Live Monitor** | your CSV + `POST /predict` | **User-initiated batch replay** with start, pause, resume and stop, a batch size up to the limit, and an optional pause between batches. It is labelled as a replay, not a plant stream. Every batch is stored in the API history |
| **Predictions** | your CSV/JSON + `POST /predict` | Validation with line/sample context, a preview, submit and cancel, results with filter and sort, per-sample contributions, and CSV export of the returned results |
| **Sensor Explorer** | session data (your inputs + API results) | Searchable picker over the 52 features; raw values or a display-only window z-score; the error chart and a contribution ranking for the same window |
| **History** | `GET /history` | `before_id` cursor paging (Newer/Older), a server-side *anomalies only* filter, page sizes 25/50/100, and de-duplication by record id |
| **Evaluation** | `GET /metrics` | Renders only a valid saved report: setup, clean false-alarm rate, detector comparison table and chart, confusion matrix, family detectability verdicts. Otherwise it says "No evaluation report yet" |
| **Settings** | config + `/health` + `/model/status` | Read-only: API URL, versions, model metadata and library versions (only when returned), the feature schema, and the theme |

**Data honesty rules built into the UI:**
* No status turns green until the API has answered.
* There is no fallback or demo data. Failed requests show the real error: unreachable, timeout, 422 with per-field details, 413, 503, or a malformed response.
* `null` metrics from a report are shown as "undefined", never as 0.
* Contribution rankings state that they are built from the **top-5 features returned per sample**. Reconstruction error is described as a deviation signal, **not** a causal explanation.
* The model's scaled inputs are not returned by the API, so only raw values (and an explicitly labelled display-only z-score) are plotted.
* Session data lives in memory only; refreshing the page clears it. The API history is the persistent record.

## 6. Project structure

```
frontend/
├─ index.html, vite.config.ts, tailwind.config.js, postcss.config.js, tsconfig.json, package.json, .env.example
├─ public/favicon.svg
└─ src/
   ├─ main.tsx, App.tsx (routes), providers.tsx, routes.ts, index.css (design tokens)
   ├─ lib/
   │  ├─ api/  client.ts (fetch + timeout + response checks), errors.ts (failure kinds, readable errors), types.ts, ApiClientContext.tsx
   │  ├─ validation.ts (CSV/JSON parsing mirroring the API rules), features.ts, chartData.ts, csvExport.ts, format.ts, config.ts
   ├─ state/   StatusContext (health polling), SessionContext (in-memory runs), ThemeContext, ToastContext
   ├─ hooks/   useResource, useChartColors, useReducedMotion
   ├─ components/
   │  ├─ layout/  AppShell, Sidebar (+ mobile drawer), TopBar, StatusPills, Logo
   │  ├─ ui/      primitives (Panel, Button, Badge, MetricTile, EmptyState, Banner, …), Tabs, FeaturePicker, ErrorState
   │  ├─ charts/  ErrorTimelineChart, SensorTrendChart, ContributionBarChart, DetectorComparisonChart
   │  ├─ predictions/  CsvDropZone, ValidationReport, InputPreview, ResultsPanel, ContributionList
   │  ├─ evaluation/   ConfusionMatrix
   │  └─ model/        ModelInfoPanel, ModelReadinessBanner
   ├─ pages/   Overview, LiveMonitor, Predictions, SensorExplorer, History, Evaluation, Settings, NotFound
   ├─ test/    setup.ts, utils.tsx (mock client), fixtures.ts (TEST-ONLY synthetic API-shaped data)
   └─ __tests__/  validation, client, status, predictions, history, evaluation, overview, monitor, accessibility
```

The design system lives in `src/index.css` (CSS variables for the dark and light themes) and `tailwind.config.js` (semantic colour names: `canvas`, `surface-1..3`, `line`, `ink`, `accent`, `teal`, `ok`, `warn`, `danger`). Icons come from `lucide-react`, charts from `recharts` and CSV parsing from `papaparse`.

Design rules (2026-10-21 redesign): a quiet graphite interface where colour carries state, following high-performance HMI practice. Steel cyan = interaction and data series; green = normal; amber = warning and the calibrated alarm threshold; red = anomaly or error. Typeface: IBM Plex Sans and IBM Plex Mono, self-hosted through `@fontsource` so the dashboard looks the same offline. Shared building blocks are in `src/components/ui/primitives.tsx` (`Panel`, `Button`, `Badge`, `MetricTile`/`Readouts` instrument strip, `Banner`, `EmptyState`, `KeyValueList`). Charts draw straight segments between samples (no smoothing) and shade the region above the threshold.

## 7. CSV and JSON input formats

* **CSV:** a header row with all 52 feature names (`xmeas_1`…`xmeas_41`, `xmv_1`…`xmv_11`) in any order.
  * Optional extra columns: `sample_id`, `timestamp`/`time`/`datetime`, and the TEP metadata columns `faultNumber`, `simulationRun` and `sample`. These are used for labels and are **not** sent.
  * When `sample_id` is absent but `simulationRun` and `sample` are present, the sample id becomes `run-<r>-sample-<s>`.
  * Any other column is an error.
* **JSON:** the API formats: `{"features": {...}}`, `{"samples": [{"features": {...}, "sample_id": "..."}]}`, or an array of sample objects.
* **Values:** finite numbers only. Text such as `"1.5"`, `null`, `NaN`, `Infinity` and overflowing numbers are rejected, exactly as the API does.
* To export held-out rows from the TEP testing file for a replay, write a CSV from a notebook into `data/` or `reports/`. Keep the *holdout* runs for the live viva demo.

## 8. Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| Top bar: **API Unreachable** | backend not running, wrong URL, or CORS | Start Uvicorn (§4). Check `VITE_API_BASE_URL`. In the browser DevTools console a *CORS* error means the dashboard origin is missing from `[api].cors_origins` / `PP_API_CORS_ORIGINS` |
| **Model: No trained model** | no `artifacts/models/autoencoder_v1/metadata.json` | Run notebook 03, then restart the API (or use the admin-protected `POST /model/reload`, see API.md) |
| **Model: Not calibrated / No scaler / Incompatible schema / Load failed** | the artifact is incomplete or changed | Re-run notebook 03 sections 3–6; check the API console log |
| **413 Batch too large** | `VITE_MAX_BATCH_SIZE` is larger than the API's `max_batch_size` | Align the two values, or replay the file in Live Monitor with a smaller batch size |
| **422 Feature names rejected** | missing or extra columns | Use exactly the 52 names; see the validation report |
| **History unavailable** | the API could not open its SQLite file | Check the API log; `database_path` must be inside `data/` |
| Evaluation shows **No evaluation report yet** | notebook 03 §6b has not run | Run it; the page reads `reports/evaluation_metrics.json` through `GET /metrics` |
| Port 5173 in use | another dev server | Stop it. `strictPort` keeps the port fixed so that CORS matches |

## 9. Known limitations

* **Unverified:** not installed, built, type-checked or tested yet. The dependency versions in `package.json` are pinned, but `npm install` has not resolved them on your machine.
* **No live streaming:** the API scores on request only.
* **Very large replays:** files above ~100,000 rows, or about 80 MB, are refused in the browser. Charts show a sliding window (default 500 points).
* **The batch limit is duplicated in config** (`VITE_MAX_BATCH_SIZE`) because the API does not expose it.
* **History has no search:** the API offers only the anomalies filter and cursor paging.
* **Session data is lost on page reload**, by design (no browser storage of process data). Only the theme and the navigation-collapse state are remembered.
* Fonts are bundled with the build (no network needed).

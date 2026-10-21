# Production Pulse — Multivariate Process Anomaly Detection Using an Autoencoder

Course 21CSE732P · Applied AI in Manufacturing (Industry 5.0) · Worklet 13, Tier 2 (Integration)

Production Pulse detects abnormal operating states in multivariate process-sensor data. Each sensor can stay inside its own normal range while the *combination* of sensors is abnormal, and single-variable rule checks cannot see that. An autoencoder is trained on normal data only. A sample whose reconstruction error is above a calibrated threshold is flagged. The project compares the autoencoder with a single-feature rule check and with a PCA baseline, on labelled synthetic anomalies injected into the Tennessee Eastman Process (TEP) dataset.

---

## 1. Project status — read this first

This README describes what exists. Evidence is labelled **Windows** (logged earlier in `logs/verification/`) or **Cloud** (Linux workspace). Cloud results are not Windows results, and nothing here has been tested on a clean machine.

| Component | State | Evidence |
|---|---|---|
| Data pipeline, scaler, run-level splits | Done | Windows pytest + real-data tests; Cloud real-data tests |
| Synthetic anomalies (6 families), rule check, PCA, metrics | Done | Cloud: corrected notebook 02 executed (evaluation-role runs only) |
| Autoencoder (52-32-16-8-16-32-52) | Trained earlier; saved in `artifacts/models/autoencoder_v1` with an integrity `manifest.json` | Notebook 03 executed (Windows); manifest added in Cloud |
| Equal-FPR comparison (rule / PCA / autoencoder) | Done, Cloud | [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md). The autoencoder beats the rule check but **does not beat PCA** |
| FastAPI backend | Done | Cloud pytest incl. offline/API parity; one network pass per prediction |
| React + Tailwind dashboard | Done | Cloud typecheck, Vitest, build |
| Robustness (input noise) and ablation | Done, Cloud (single seed per variant) | [`docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md`](docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md) |
| Windows verification (2026-10-21, 17:36–17:41 IST) | pytest 416 passed / 13 skipped (exit 0); real-data, parity and demo tests 20 passed; API smoke test 15/15 checks (exit 0, auto-selected port); frontend typecheck, 78 tests and build (exit 0); notebooks 01–04 executed (exit 0) | `logs/verification/summary.txt`. Later changes (CSV export, error display, demo scripts) were verified in the cloud only |

All anomalies in the evaluation are **synthetic**; the results do not show detection of real plant faults. The 50 holdout runs were not used for fitting or calibration, but runs 42, 196, 241 and 304 were scored by the earlier notebook 02 selection. The authoritative status is [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

---

## 2. Architecture

```
 TEP .RData files (read-only, project root)
          │  pyreadr (data.py: validation, 100 MB load guard)
          ▼
 run-level splits (splitting.py, workflow.py)         seeded, whole simulation runs, no leakage
   training file → train 0.8 / validation 0.2
   testing file  → calibration 0.2 / evaluation 0.7 / holdout 0.1 (holdout not loaded by default; 4 runs already scored, see below)
          │  scaler fitted on the train split only (preprocessing.py)
          ▼
 anomaly_generation.py ── 6 labelled synthetic families (correlated, individually in range)
          │
          ├─► baselines.py   single-feature rule check · PCA reconstruction error (SPE)
          ├─► autoencoder.py Keras dense AE 52→32→16→8→16→32→52, threshold calibrated on calibration runs
          └─► evaluation.py  precision/recall/F1, confusion matrix, ROC-/PR-AUC, FPR, family verdicts
          │
          ▼  artifacts/models/autoencoder_v1/  (model.keras, metadata.json, history.csv, fingerprints.npz)
              reports/evaluation_metrics.json  (written by notebook 03, section 6b)
          │
 FastAPI (src/production_pulse/api)  ──  SQLite history (data/api/predictions.sqlite3)
   GET /health · GET /model/status · POST /model/reload · POST /predict · GET /metrics · GET /history
          │  JSON over HTTP, CORS limited to explicit origins
          ▼
 React + Vite + Tailwind dashboard (frontend/)  — Overview, Live Monitor (replay), Predictions,
                                                  Sensor Explorer, History, Evaluation, Settings
```

Design rules that the code follows:
* The original datasets are never modified. All outputs go through `safe_output_path`, which refuses protected files.
* Splits are by whole `simulationRun`, never by random rows. The scaler, the PCA and the rule limits are fitted on training data only.
* `workflow.prepare_fault_free_data` does not load the holdout runs unless `include_holdout=True` is passed explicitly. They are reserved for the live demo. **Exception, stated plainly:** the first version of notebook 02 drew its 40 runs from all 500 testing runs, so with seed 42 it *scored* 4 holdout runs (42, 196, 241, 304) and 10 calibration runs. They were never used to fit, train or calibrate anything, but those 4 holdout runs are **not unseen** and are excluded from demo selection (`splitting.HOLDOUT_RUNS_PREVIOUSLY_SCORED`). Notebook 02 now draws only from the evaluation role (`splitting.select_runs`). The old executed copy in `reports/executed/` is kept as historical evidence of the old selection. Also note that notebook 01 loads the whole testing file for aggregate summaries and histograms, so holdout rows appear in those descriptive plots (no model or threshold used them).
* The API never trains a model. It starts without one (`model.status = not_found`) and answers `POST /predict` with 503 until a model exists.
* Nothing in the dashboard is a placeholder. A failed request shows the real error, and a missing report shows "No evaluation report yet".

More detail: [`docs/ARCHITECTURE_PLAN.md`](docs/ARCHITECTURE_PLAN.md), [`docs/API.md`](docs/API.md), [`docs/FRONTEND.md`](docs/FRONTEND.md).

---

## 3. Folder structure

```
Production_Pulse_Anamoly_Detection/
├─ TEP_*.RData (4 files)            original datasets — read-only, never edit, git-ignored
├─ Mark Split 21CSE732P rubrics .pdf, Applied AI Industry Worklets Student Template.xlsx   course originals
├─ README.md                        this file
├─ PROJECT_STATUS.md                implemented vs tested vs trained vs verified, environment record
├─ pyproject.toml                   package metadata, runtime dependencies, pytest settings
├─ requirements-lock.txt            exact versions found in the working .venv313 (snapshot)
├─ configs/default.toml             all settings (split fractions, autoencoder, API)
├─ src/production_pulse/            Python package
│   ├─ paths.py config.py logging_config.py   foundation
│   ├─ data.py preprocessing.py splitting.py workflow.py
│   ├─ anomaly_generation.py baselines.py autoencoder.py evaluation.py plots.py reporting.py
│   └─ api/                         FastAPI app, routes, service, store, schemas, settings, errors
├─ notebooks/                       01 data exploration · 02 synthetic-anomaly validation · 03 autoencoder training
├─ tests/                           pytest suite (371 tests collected)
├─ scripts/                         show_paths.py, verify_checksums.py, verify/ (staged verification helpers)
├─ frontend/                        React + TypeScript + Tailwind dashboard (Vite)
├─ docs/                            API, frontend, architecture, checksums, logbook, rubric checklist,
│                                   report outline, methodology, experiment templates, viva Q&A, demo script
├─ reports/figures/                 figures written by the notebooks
├─ data/ artifacts/ logs/           GENERATED (git-ignored): derived data and SQLite DB / trained model / logs
└─ .venv313/  (.venv/ unused)       Python environments — do not modify
```

---

## 4. Prerequisites

| Tool | Needed for | Version on this project's machine |
|---|---|---|
| Windows 10/11, PowerShell, VS Code | everything | — |
| Python | backend, notebooks, tests | **3.13.14** in `.venv313` (verified). `.venv` (3.14.3) is unused: leave it alone |
| Node.js ≥ 18.18 and npm | dashboard | Node v22.20.0, npm 11.6.1 found on PATH (preflight log, 2026-10-21) |
| Free RAM | notebooks 01–03 use the two small fault-free files only | The two large *faulty* `.RData` files (494 MB and 837 MB) are **never loaded** by the current code |

Python packages are already installed in `.venv313`. Do **not** reinstall or upgrade them. The runtime list is in `pyproject.toml`, and the exact working snapshot is in `requirements-lock.txt`.

**OneDrive tip.** This folder is inside OneDrive. `node_modules` contains tens of thousands of files, and SQLite files can be briefly locked while OneDrive syncs. Pause OneDrive sync during `npm install`, or exclude `frontend\node_modules`.

---

## 5. Windows setup

Open VS Code, choose **File → Open Folder…**, and select the project folder. Open a terminal with **Terminal → New Terminal** (PowerShell). The prompt should already be in the project folder.

```powershell
# 1. Go to the project folder (skip if the terminal already opened there)
#    <project folder> = the folder that contains pyproject.toml, e.g. C:\Users\<you>\Documents\Production_Pulse_Anamoly_Detection
cd "<project folder>"

# 2. Check the project's Python (does not change anything)
.\.venv313\Scripts\python.exe --version                 # expect: Python 3.13.14

# 3. Check that the project folder is found and see the resolved paths (read-only)
.\.venv313\Scripts\python.exe scripts\show_paths.py

# 4. Check the original files are unchanged (read-only; expect "ALL MATCH")
.\.venv313\Scripts\python.exe scripts\verify_checksums.py
```

You do not need to "activate" the environment: calling `.\.venv313\Scripts\python.exe` directly always uses it. If you prefer activation, use `.\.venv313\Scripts\Activate.ps1`. If PowerShell blocks it with an execution-policy error, keep using the direct path. No system setting needs to change.

**Do not re-create `.venv313`** with the commands in `requirements-lock.txt`. Those are only for a *new* machine.

### Run the tests (optional now; this was done on 2026-10-21)

```powershell
.\.venv313\Scripts\python.exe -m pytest -p no:cacheprovider -q
# the two real-data tests are skipped unless you opt in:
$env:PP_RUN_REALDATA = "1"
.\.venv313\Scripts\python.exe -m pytest -p no:cacheprovider -q tests\test_real_data.py
Remove-Item Env:PP_RUN_REALDATA
```

### Notebooks (JupyterLab, port 8889 because 8888 is occupied)

```powershell
.\.venv313\Scripts\python.exe -m jupyter lab --port 8889
```

Choose the kernel **Python 3 (ipykernel)** from `.venv313`. Run `01_data_exploration` → `02_synthetic_anomaly_validation` → `03_autoencoder_training`, each from top to bottom. The notebooks in the project contain no outputs. Notebook 03 trains the model, calibrates the threshold, saves `artifacts/models/autoencoder_v1/` and writes `reports/evaluation_metrics.json`. **Do not change hyper-parameters or thresholds after looking at the evaluation numbers.**

---

## 6. Start the backend (FastAPI)

The API runs without a trained model, but `/predict` then returns 503 `model_unavailable`. Train the model first (notebook 03) for real predictions.

```powershell
cd "<project folder>"
.\.venv313\Scripts\python.exe -m uvicorn production_pulse.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation. Stop the server with `Ctrl+C`. If the browser cannot reach it, check that nothing else is using port 8000.

Alternative that takes host and port from `configs/default.toml`:

```powershell
$env:PYTHONPATH = "src"
.\.venv313\Scripts\python.exe -m production_pulse.api
```

---

## 7. Start the frontend (React + Tailwind)

Status: **written, not yet installed, type-checked, tested, built or run.** The first run will show whether it works.

```powershell
cd "<project folder>\frontend"
Copy-Item .env.example .env.local        # optional; the defaults already point to http://127.0.0.1:8000
npm install                              # creates node_modules\ and package-lock.json
npm run dev                              # http://127.0.0.1:5173
```

If PowerShell blocks `npm` with an execution-policy message, use `npm.cmd` instead of `npm`. The other scripts are `npm run typecheck`, `npm test` (all API calls mocked), `npm run build` and `npm run preview` (port 4173; add that origin to the API's CORS list first). Use a second terminal for the backend. The top bar reads **API: Connected** only after `GET /health` has actually answered.

---

## 8. Configuration

| What | Where | Notes |
|---|---|---|
| Seeds, splits, autoencoder, API, logging | `configs/default.toml` | Values must equal the built-in defaults in `config.py`; a test checks this |
| API overrides | environment variables `PP_API_HOST`, `PP_API_PORT`, `PP_API_MAX_BATCH_SIZE`, `PP_API_MODEL_DIR`, `PP_API_DATABASE_PATH`, `PP_API_METRICS_REPORT`, `PP_API_CORS_ORIGINS`, `PP_API_STORE_INPUTS` | Full table in `docs/API.md` §3 |
| Model-reload secret | `PP_API_ADMIN_TOKEN` (≥ 16 characters, environment only, never in a file) | Without it `POST /model/reload` answers 403. The dashboard never uses it |
| Dashboard | `frontend\.env.local` (copy of `frontend\.env.example`) | `VITE_API_BASE_URL`, `VITE_MAX_BATCH_SIZE` (must equal `max_batch_size`), `VITE_REQUEST_TIMEOUT_MS`, `VITE_STATUS_POLL_MS`. `VITE_*` values end up in the browser, so never put secrets there |

Rules enforced by the backend: paths are relative to the project root, the database must be inside `data/`, `artifacts/`, `reports/` or `logs/`, and `"*"` is rejected as a CORS origin.

---

## 9. API usage

All errors have the shape `{"error": {"code": "...", "message": "...", "details": ...}}`. Examples assume the API runs on port 8000.

```powershell
# Is the API up? Model and database are reported separately.
Invoke-RestMethod http://127.0.0.1:8000/health

# Model card and feature schema
Invoke-RestMethod http://127.0.0.1:8000/model/status

# Latest saved evaluation report (or {"available": false, ...} if notebook 03 has not written one)
Invoke-RestMethod http://127.0.0.1:8000/metrics

# Newest-first prediction history, anomalies only, 20 rows
Invoke-RestMethod "http://127.0.0.1:8000/history?limit=20&anomalies_only=true"
```

`POST /predict` takes **raw** values for all 52 features (any order). It accepts either `{"features": {...}}` or `{"samples": [{"features": {...}, "sample_id": "..."}]}`, with at most `max_batch_size` samples. The response gives `reconstruction_error`, `is_anomaly` (error above the calibrated threshold) and the top contributing features per sample. A complete PowerShell example, the response format, the endpoint tables and the error codes are in [`docs/API.md`](docs/API.md). Use samples from the *evaluation* runs for testing. Keep the *holdout* runs for the live demo ([`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)).

---

## 10. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `python` or `pip` is not recognized | Using the system command, not the project's Python | Use `.\.venv313\Scripts\python.exe` as shown above |
| `ModuleNotFoundError: production_pulse` | Running a script without `src` on the path | Run from the project folder. For uvicorn, keep `--app-dir src`. In PowerShell, set `$env:PYTHONPATH = "src"` |
| Jupyter says port 8888 is busy | Another program uses 8888 | Use `--port 8889` as shown above |
| Notebook uses the wrong kernel | Kernel from another environment | Select the `.venv313` Python 3 kernel; `.venv` (3.14) is not used |
| API: `model_unavailable` (503) / `/health` shows `not_found` | Notebook 03 has not been run | Run notebook 03, then restart the API (or use the admin-protected `POST /model/reload`) |
| Notebook 03 refuses to save, because the model folder already exists | `OVERWRITE = False` in the settings cell protects an existing model | Set `OVERWRITE = True` **only if you intend to replace the saved model**; otherwise keep it. Note that a replaced model means all earlier results no longer match it |
| API: `incompatible`, `no_preprocessing`, `not_calibrated`, `load_failed` | The artifact is incomplete or was changed | Re-run notebook 03 sections 3–6. The status table is in `docs/API.md` §1 |
| API: 422 `invalid_features` | Missing or unexpected feature names | Send exactly the 52 names `xmeas_1`…`xmeas_41`, `xmv_1`…`xmv_11` |
| API: 413 `batch_too_large` | More samples than `max_batch_size` | Send smaller batches, and keep `VITE_MAX_BATCH_SIZE` equal to the API value |
| `history_stored: false` with a warning | SQLite briefly locked (OneDrive) or the database is unavailable | Retry. Predictions are still returned. Check the API log |
| Dashboard shows **API Unreachable** | Backend not running, wrong URL, or CORS | Start uvicorn; check `VITE_API_BASE_URL`. A CORS error in the browser console means the dashboard origin is missing from `[api].cors_origins` / `PP_API_CORS_ORIGINS` |
| Dashboard shows "No evaluation report yet" | `reports/evaluation_metrics.json` does not exist | Run notebook 03 (section 6b) |
| `npm` blocked by an execution-policy message | PowerShell script policy | Use `npm.cmd` instead of `npm` |
| `npm install` is very slow or errors on file locks | OneDrive is syncing `node_modules` | Pause OneDrive sync or exclude the folder, then retry |
| Port 5173 or 8000 already in use | Another server | Stop it. `strictPort` keeps 5173 fixed so that CORS matches |
| Checksum check reports a mismatch | An original file was modified or corrupted | Stop. Do not run anything else. Restore the file from your original copy. Baselines: `docs/DATA_CHECKSUMS.md` |

---

## 11. Academic documents

| Document | Purpose |
|---|---|
| [`docs/RUBRIC_CHECKLIST.md`](docs/RUBRIC_CHECKLIST.md) | Rubric and deliverables checklist with evidence-based status |
| [`docs/RESEARCH_LOGBOOK.md`](docs/RESEARCH_LOGBOOK.md) | Chronological log of experiments, failures and decisions |
| [`docs/RESEARCH_METHODOLOGY.md`](docs/RESEARCH_METHODOLOGY.md) | Research question, hypotheses, data roles, anomaly injection, detectors, evaluation protocol, threats to validity |
| [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) | Chapter outline of the final report, with fields to fill from measured results |
| [`docs/EXPERIMENT_TEMPLATES.md`](docs/EXPERIMENT_TEMPLATES.md) | Blank templates for recording experiments (all empty until a run is done) |
| [`docs/VIVA_QA.md`](docs/VIVA_QA.md) | Viva questions with answers based on the design and the code |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | Step-by-step held-out-run demonstration |
| [`docs/DATA_CHECKSUMS.md`](docs/DATA_CHECKSUMS.md) | SHA-256 baselines for the six original files |

## 12. Windows PowerShell quick commands

Run from the project folder, with the existing `.venv313` (Python 3.13). Items marked * have no recorded Windows run yet (they were run in the cloud workspace).

```powershell
.\.venv313\Scripts\Activate.ps1                      # 1. activate the environment (setup/install: section 5)
python -m pytest -q                                   # 3. tests (add: $env:PP_RUN_REALDATA="1" for real-data tests)
.\scripts\verify\run.ps1 nb02                        # 4. corrected notebook 02 -> reports\executed\ (new file name if one exists)
.\scripts\verify\run.ps1 nb04                        #    equal-FPR comparison (writes .rerun-<time> report files if earlier ones exist)
.\scripts\verify\run.ps1 manifest                    #    check model integrity manifest
.\scripts\verify\run.ps1 api                         #    API smoke test on its OWN free port (PP_API_SMOKE_PORT, default auto)
python -m uvicorn production_pulse.api.main:app --app-dir src --host 127.0.0.1 --port 8000   # 5. backend
cd frontend; npm install; npm run dev                 # 6. frontend (section 7)
python scripts\demo_prediction.py --offline            # 7. real fault-free evaluation-role sample -> /predict *
$env:PP_API_METRICS_REPORT="reports/evaluation_equal_fpr.json"   # 8. show the equal-FPR report in the dashboard, then restart the backend
python scripts\make_demo_samples.py                    # 9. demo request files -> reports\demo\ (see docs\DEMO_SCRIPT.md) *
python scripts\export_holdout_demo.py --list           #    held-out runs for the live demo; then --run <R> --seed <S> *
python scripts\run_experiments.py robustness --tag win01   # 10. robustness experiment (new files only) *
```

Demonstration procedure: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md). What is safe to publish: [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

`run.ps1` exits with the first non-zero exit code of a stage (0 = all passed). The API smoke test never touches a server you already run (e.g. on 8000); it uses an isolated history database `data/api/verification_smoke.sqlite3`; exit 3 means the chosen port is unavailable. `run.ps1` never overwrites an existing executed notebook; it falls back to a `.rerun-<timestamp>` name. The CSV Live Monitor is a user-initiated replay, not a live plant stream.

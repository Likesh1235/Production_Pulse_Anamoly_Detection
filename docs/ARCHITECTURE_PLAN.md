# Proposed Architecture & Implementation Plan — Production Pulse

Status: **Approved; code and packaging written.** Sections 1–9 are the original plan (kept for traceability); §10 records what is written, what has been tested, what has been run on real data, and what changed. Sections 1–9 are the *plan*, not a description of the finished project (for example, the plan's separate `backend/` folder became `src/production_pulse/api/`).

> **Note (2026-10-21):** the status columns in §10 record the state when they were written and are partly out of date (the model is trained, the API and dashboard have been run, Phase 5 experiments exist). The current status is in `PROJECT_STATUS.md` §0 and `docs/RUBRIC_CHECKLIST.md`.

## 1. Data facts this plan is based on (verified 2026-10-20)
| File | Size on disk | R object | Shape | Verified how |
|---|---|---|---|---|
| TEP_FaultFree_Training.RData | 24.7 MB | `fault_free_training` | 250,000 × 55 (500 runs × 500 samples) | fully loaded |
| TEP_FaultFree_Testing.RData | 47.3 MB | `fault_free_testing` | 480,000 × 55 (500 runs × 960 samples) | fully loaded |
| TEP_Faulty_Training.RData | 494 MB | `faulty_training` | ≈5,000,000 rows (inferred) | header + decompressed size only |
| TEP_Faulty_Testing.RData | 837 MB | `faulty_testing` | ≈9,600,000 rows (inferred) | header + decompressed size only |

Columns: `faultNumber, simulationRun, sample` + 41 measured (`xmeas_1..41`) + 11 manipulated (`xmv_1..11`) = **52 features**.

## 2. How to read the data
- Python: `pyreadr.read_r(path)` → dict of pandas DataFrames. Fine for fault-free files (~110–210 MB in RAM).
- Faulty files: convert **once** to Parquet (float32, one file per fault number) under `data/interim/`; afterwards every notebook reads only the slice it needs. Raw `.RData` files stay untouched in the project root.
  - If your Windows PC has ≥16 GB RAM: load with pyreadr once, downcast, write Parquet.
  - If less: use a small streaming reader that walks the R serialization column by column (R stores data.frames column-wise) and writes to disk without holding the whole table. I'll decide once I know your RAM.
- Note: in the commonly used version of this dataset, faults are introduced at sample 21 (training) / 161 (testing). **Not verified on your files yet** — Phase 2 will check this from the data before we rely on it.

## 3. Leakage-free split
| Split | Source | Use |
|---|---|---|
| Train | Fault-free training, runs 1–400 | Fit scaler + autoencoder + PCA + rule limits |
| Validation (normal) | Fault-free training, runs 401–500 | Early stopping, threshold calibration |
| Calibration-injected | Fault-free testing, runs 1–100 + injected anomalies (seed A) | Optional F-score threshold tuning, ablation |
| Test-injected | Fault-free testing, runs 101–450 + injected anomalies (seed B) | Final benchmark (all models, same anomalies) |
| **Locked demo set** | Fault-free testing runs 451–500 + faulty-testing runs not used elsewhere | Never looked at until the live viva demo (rubric U6) |
| Real-fault check | Faulty testing (subset of runs) | Secondary evaluation on genuine TEP faults |
Split is by `simulationRun`, scaler fit on train only, every split written with a manifest (run IDs + hash).

## 4. Synthetic multivariate anomaly injection (core R&D)
Goal (from worklet): perturbations that are **correlated** across variables but keep **every variable inside its own normal range**, so a single-variable rule check cannot see them.
Planned injection families (each with severity levels, window lengths, seeds, logged metadata):
1. **Correlation break** — move strongly correlated variable pairs/groups in opposite directions.
2. **Minor-component shift** — push samples along low-variance principal directions (unusual combinations of normal values).
3. **Slow coordinated drift** — ramped joint drift of a variable group.
4. **Sensor decoupling** — replace one sensor with a plausible but de-synchronised copy of itself.
Constraint check: after injection, every value is clipped/verified to lie inside the rule-baseline limits computed on train; samples breaking this are logged and rejected. Labels: per-sample `is_anomaly`, `anomaly_type`, `severity`, `event_id`.

## 5. Models
- **Rule-based baseline** — per-variable limits (mean ± 3σ, and percentile variant) from train data.
- **PCA baseline** — Hotelling T² and SPE/Q statistics (standard in process monitoring).
- **Autoencoder (main)** — Keras dense AE 52 → 32 → 16 → 8 → 16 → 32 → 52, MSE loss, early stopping on validation.
- Optional variant for ablation: windowed AE (k consecutive samples).
- **Threshold calibration** — 99th / 99.5th percentile of validation reconstruction error, mean + kσ, and F1-optimal on calibration set; compared in an experiment.

## 6. Evaluation
Precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC, false-alarm rate on normal data, MCC, event-level detection rate and detection delay, recall per injection type and severity. Key figure: anomalies the rule check misses vs. the AE catches.

## 7. Robustness & ablation
- Robustness: added Gaussian noise, severity sweep, missing/frozen sensor, real TEP faults.
- Ablation: bottleneck size, depth, no scaling, threshold method, point vs window input, training data size, xmv-only / xmeas-only features.

## 8. Software
```
Production_Pulse_Anamoly_Detection/
├─ *.RData, rubric files            (raw, untouched)
├─ data/interim/  data/processed/   (generated; git-ignored)
├─ notebooks/ 01_eda … 06_robustness_ablation.ipynb
├─ src/production_pulse/  config, data_loading, splitting, preprocessing,
│                          injection, models/{rules,pca,autoencoder}, thresholds, evaluation
├─ artifacts/  model.keras, scaler.joblib, thresholds.json, metrics.json, run manifests
├─ backend/   FastAPI app: GET /health, GET /model/status, POST /predict,
│             GET /metrics, GET /history (SQLite)
├─ frontend/  React + Vite + Tailwind dashboard: overview, live sensor replay,
│             anomaly alerts, prediction table, metrics & confusion matrix
├─ tests/     pytest: loader, split leakage, injection in-range constraint,
│             thresholds, API endpoints; frontend smoke tests
└─ docs/      PROJECT_STATUS, RUBRIC_CHECKLIST, RESEARCH_LOGBOOK, report, viva Q&A
```

## 9. Phases (mapped to rubric checkpoints)
| Phase | Content | Rubric |
|---|---|---|
| 1 | Audit, plan (this) | FP-I |
| 2 | Windows env setup, data conversion, EDA, literature notes | FP-I / FP-II |
| 3 | Splits, preprocessing, injection + labeled test set | FP-II / PBL-I |
| 4 | Rule + PCA baselines, first AE, threshold calibration | PBL-I |
| 5 | Benchmark, robustness, ablation | PBL-II |
| 6 | FastAPI backend + tests | PBL-III |
| 7 | React dashboard + live held-out demo | PBL-III |
| 8 | Report, logbook polish, viva prep, final package | R21 |

## 10. Implementation status and changes (updated 2026-10-21)

Legend: **Written** = code and unit tests exist · **Tests run** = the automated tests were run, result on file (full Windows suite 2026-10-21: 371 collected, 369 passed, 2 skipped, 0 failed; real-data tests 2/2 passed; `logs/verification/`) · **Executed on real data** = a notebook or model was run on the TEP data, with results in the research logbook.

| Component | Module | Written | Tests run (Windows) / executed on real data |
|---|---|---|---|
| Paths, config, logging, output safety | `paths.py`, `config.py`, `logging_config.py` | Yes (Phase 1) | Tests run: passed (2026-10-21 log) |
| Loading/validation, scaling, run splits | `data.py`, `preprocessing.py`, `splitting.py` | Yes (Phase 2) | Tests run: passed, plus 2/2 real-data tests (2026-10-21 log). Notebook 01 not run on Windows |
| Synthetic anomaly families (6) | `anomaly_generation.py` | Yes (Phase 3) | Tests run: passed. Real data: Linux smoke run only (preliminary); Windows run pending |
| Rule check, PCA (SPE) baselines | `baselines.py` | Yes (Phase 3) | Tests run: passed. Real data: Linux smoke run only (preliminary) |
| Metrics, detector comparison, hypothesis test | `evaluation.py` | Yes (Phase 3) | Tests run: passed. Real data: Linux smoke run only (preliminary) |
| Shared data roles (train/validation/calibration/evaluation/holdout) | `workflow.py` | Yes (Phase 4) | Tests run: passed. **Not run on the full real data** |
| Dense autoencoder: training, scoring, calibration, save/load | `autoencoder.py` | Yes (Phase 4) | Tests run: passed (unit tests and tiny synthetic-data models). **No model trained on TEP** |
| Figures (history, score distributions, per-feature error) | `plots.py` | Yes (Phase 4) | Tests run: passed. No real figure produced yet |
| Notebooks 01 / 02 / 03 | `notebooks/` | Yes (no outputs stored) | **Not run on Windows** (01 and 02 smoke-tested in the Linux workspace only) |
| FastAPI backend (health, model status/reload, predict, metrics, history; SQLite) | `api/`, `reporting.py` | Yes (Phase 6) | Tests run: passed (stub detector, tiny synthetic model). **API never started live; no trained model served** |
| React + Tailwind dashboard (Overview, Live Monitor, Predictions, Sensor Explorer, History, Evaluation, Settings) | `frontend/` | Yes (Phase 7) | **Not installed, built, type-checked or run**; component tests written, **not executed** |
| Packaging: README, dependency declaration, lock file, `.gitignore` | `README.md`, `pyproject.toml`, `requirements-lock.txt`, `.gitignore` | Yes (2026-10-21) | Not installed from scratch; the lock file is a snapshot of the working `.venv313` |
| Academic documents: methodology, report outline, experiment templates, viva Q&A, demo script | `docs/` | Yes (2026-10-21) | Templates and outlines only. No experiment is recorded in them. The demo helper code is unexecuted |
| Final report, completed viva pack, final package | — | No | — |

### Changes against sections 3–5
1. **Split (§3):** whole runs are assigned by a **seeded random** split, not contiguous run ranges (see the logbook entry of 2026-10-20). The roles are:
   * training file → **train 0.8 / validation 0.2** (`[split]`);
   * testing file → **calibration 0.2 / evaluation 0.7 / holdout 0.1** (`[evaluation]`).
   Each file is split separately, because run IDs repeat across files.
2. **Calibration data (§3, §5):** thresholds are calibrated on the testing-file **calibration** runs, not on the validation split. The validation split already drives early stopping, so a threshold chosen on it would be biased. The evaluation runs are only for reporting, and the holdout runs are not loaded unless explicitly requested (live demo, rubric U6).
3. **Injection (§4):** six families are implemented — coordinated_shift, coordinated_drift (= "slow coordinated drift"), short_event, correlation_break, decoupling (= "sensor decoupling", with values from another run), and minor_component_shift.
   * The in-range rule became "no *new* range violations" (fairness fix).
   * Success is decided by explicit pass criteria (`evaluation.HypothesisCriteria`), never assumed.
4. **Baselines (§5):** the PCA baseline uses SPE/Q only; Hotelling T² is not implemented. The preliminary Phase 3 run showed that SPE misses shifts along the lowest-variance directions, so a variance-weighted baseline (T²/Mahalanobis) is a candidate for Phase 5.
5. **Autoencoder (§5):**
   * Architecture as planned: 52 → 32 → 16 → 8 → 16 → 32 → 52, ReLU, linear output, MSE loss, Adam (configurable in `[autoencoder]`).
   * Training uses normal training rows only, early stopping on the validation split, and **always** restores the best-validation weights.
   * Score = mean squared reconstruction error per sample.
   * Saved artifacts: `model.keras` plus `metadata.json` (feature order, settings, threshold, embedded scaler, model checksum, library versions), training history, and row fingerprints for the leakage checks.
6. **Not yet decided / not run:** threshold strategies other than "target FPR on calibration-normal data", the windowed autoencoder variant, detection delay, and robustness/ablation runs (Phase 5).

### Phase 6 additions — API (written; automated tests run on Windows 2026-10-21; never started live)
* The architecture follows §8, with a few refinements:
  * `GET /model/status` is extended with an admin-protected `POST /model/reload`;
  * `GET /history` supports `before_id` cursor paging;
  * `GET /metrics` serves `reports/evaluation_metrics.json`, which is written by notebook 03 through `reporting.write_evaluation_report`.
* Live inference calls `detect_raw(..., check_unseen=False)`. The leakage refusal is an *evaluation* safeguard; for a live service, rejecting a sample because it equals a training row would be wrong. Evaluation code keeps the default `check_unseen=True`.
* The SQLite history lives at `data/api/predictions.sqlite3` and goes through `safe_output_path`. Raw inputs are not stored by default.
* Full API documentation: `docs/API.md`.

### Phase 7 additions — dashboard (written; not installed, built, tested or run)
* `frontend/`: Vite + React 18 + TypeScript + Tailwind 3, with React Router, Recharts, PapaParse and lucide-react (declared in `frontend/package.json`, not installed).
* One typed API client (`src/lib/api/client.ts`):
  * every endpoint has a timeout and a runtime check of the response shape;
  * failures are kept apart as network, timeout, HTTP error (using the backend error envelope and codes) and malformed response;
  * there is no fallback data.
* Status comes only from `GET /health` and `GET /model/status`; nothing is shown as healthy before the API answers.
* Client-side validation mirrors the API rules: exactly 52 names in any order, finite numbers only, `sample_id` ≤ 128 characters, and the batch limit.
  * The API does not expose `max_batch_size`, so the frontend uses `VITE_MAX_BATCH_SIZE` (default 1000) and also handles HTTP 413.
* "Live Monitor" is an explicitly labelled **user-initiated replay** of a user-supplied CSV in batches. The current API has no streaming source.
* History uses the backend's `before_id` cursor and its server-side `anomalies_only` filter.
* The Evaluation page renders only fields present in a saved report (`reports/evaluation_metrics.json`, notebook 03 §6b).
* Full documentation: `docs/FRONTEND.md`.

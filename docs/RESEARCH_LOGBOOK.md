# Research Logbook — Production Pulse (Worklet 13)

Format follows the "Research Logbook" sheet of the course template: Date · Activity/Experiment · Observation/Result · Decision/Pivot Taken · Next Steps. Log failures and dead ends too — they are graded.

---

### 2026-10-20 — Phase 1: project & environment audit

**Activity / Experiment**
- Listed project folder; read rubric PDF and worklet workbook.
- Inspected the 4 TEP `.RData` files without loading the two large faulty files into memory (read gzip header + stream-decompressed byte count only).
- Fully loaded the two small fault-free files (in a sandbox copy) to confirm columns.

**Observation / Result**
- Files are gzip-compressed R workspaces (`RDX2` format). Each contains one data.frame: `fault_free_training`, `fault_free_testing`, `faulty_training`, `faulty_testing`.
- Fault-free training: 250,000 rows × 55 cols (500 simulation runs × 500 samples); fault-free testing: 480,000 rows × 55 cols (500 runs × 960 samples). No missing values. faultNumber = 0 throughout.
- Columns: `faultNumber, simulationRun, sample, xmeas_1..xmeas_41, xmv_1..xmv_11` → 52 process variables.
- Faulty files decompress to 2.16 GB and 4.15 GB; by bytes-per-row this implies ≈5.0 M and ≈9.6 M rows (inferred, not yet loaded).
- Linux sandbox on the laptop has 3.8 GB RAM — too small to load `faulty_testing` (≈4.1 GB as float64) in one go.

**Decision / Pivot Taken**
- Treat fault-free data as the primary source (autoencoder is trained on normal data only; anomalies will be synthetically injected per the worklet R&D requirement).
- Use faulty files only as a secondary, real-fault check, after a one-time memory-safe conversion to Parquet (float32, partitioned by fault).
- Split by `simulationRun` (not by random rows) to avoid temporal leakage.

**Next Steps**
- Student to run Windows environment checks (see PROJECT_STATUS.md) and report RAM.
- Approve architecture; then Phase 2 (environment setup + data conversion + EDA).

---

### 2026-10-20 — Phase 1 (foundation) and Phase 2 (data pipeline)

**Activity / Experiment**
- Built the project foundation: portable root detection, config (`configs/default.toml`), logging, write-safety for outputs, checksum script.
- Built `data.py` (read-only `.RData` loading + validation + summaries), `preprocessing.py` (training-only scaling), `splitting.py` (run-level splits), notebook 01.

**Observation / Result**
- Windows (student-reported): Phase 1 and Phase 2 pytest suites passed in `.venv313`; optional real-data tests on both fault-free files passed; all six original-file checksums MATCH.
- Linux workspace run of notebook 01 code: fault-free training 250,000 × 55 (500 runs × 500 samples), testing 480,000 × 55 (500 runs × 960 samples); no NaN/inf/duplicate rows; `faultNumber` is `float64` in the training file but `int32` in the testing file.

**Decision / Pivot Taken — design choice: seeded random run-based split**
- `docs/ARCHITECTURE_PLAN.md` §3 planned contiguous run ranges (runs 1–400 train / 401–500 validation). Implemented instead: a **seeded random assignment of whole runs** (`split_runs`, seed = `random_seed` = 42, fractions 0.8/0.2 in `configs/default.toml`).
- Why: keeps whole runs together (no temporal leakage, as planned) while avoiding any systematic effect of run numbering; reproducible from the seed; the split is serialisable (`RunSplit.to_json`) and overlap is checked automatically.
- Result (Linux workspace): 400 train runs / 100 validation runs; validation features after training-only scaling had mean −0.0029, std 0.9971.
- Added a 100 MB load guard so the two large faulty files cannot be loaded by accident.

**Next Steps**
- Phase 3: synthetic multivariate anomalies, baselines, evaluation.

---

### 2026-10-20 — Phase 3: synthetic anomalies, baselines, evaluation (code complete; TEP results preliminary)

**Activity / Experiment — completed (code + unit tests)**
- `anomaly_generation.py`: six seeded families — coordinated_shift, coordinated_drift, short_event, correlation_break, decoupling, minor_component_shift; features and ground truth returned separately; events never cross a run; every family starts as `UNVERIFIED`.
- `baselines.py`: single-feature rule check (training quantile limits 0.0005/0.9995) and PCA reconstruction error (SPE); fit on training-normal, threshold calibrated on separate calibration-normal data; row fingerprints raise `LeakageError` if fit/calibration/evaluation rows overlap.
- `evaluation.py`: precision, recall, F1, FPR, confusion matrix, PR-AUC, ROC-AUC with explicit undefined cases; `compare_detectors`; `assess_family_detectability` with explicit pass criteria (in-range ≥ 0.99, rule-check recall − FPR ≤ 0.05, multivariate recall ≥ 0.5).
- Notebook `02_synthetic_anomaly_validation.ipynb` (not yet run on Windows).

**Observation / Result — failures and pivots during development (all measured)**
1. *Clipping bias (synthetic data):* clipping every modified value into the training range also "cleaned" clean values already slightly outside it, so the rule check scored recall 0.0 on anomalies, below its own FPR 0.039. **Pivot:** injections may not create *new* range violations; values already outside keep their original extremeness.
2. *Marginal check v1 was misleading:* pooling all anomalous rows diluted random-feature families (a 4σ coordinated shift scored AUC 0.644 and looked "hidden"). **Pivot:** per feature, compare only the rows where that feature was modified, using distance from the normal median.
3. *Marginal check v2 on TEP:* decoupling scored AUC 0.72 although donors are genuine normal values. Cause: strong autocorrelation (lag-1 ≈ 0.993); xmeas_18 was modified in only 3 events, and the *clean* values on the same rows already gave AUC 0.66. **Pivot:** compare against the same rows before injection (excess AUC ≤ 0.05) and require ≥ 3 events per feature.
4. *Synthetic finding:* minor_component_shift needed magnitude ≈ 15 before PCA-SPE recall reached 0.90 (0.053 at magnitude 6).

**Preliminary TEP results — Linux workspace smoke run of notebook 02 (Python 3.10 / pandas 2.3.3; Windows run is the reference)**
- Setup: seed 42; fit on 200,000 rows (train split), PCA threshold calibrated for FPR 0.01 on 50,000 validation rows, evaluation on 38,400 rows (40 runs of the fault-free testing file); anomaly rate 0.05 per family; PCA kept 31 components (0.901 of variance).
- False-alarm rate on clean held-out data: rule check 0.0451, PCA 0.0097.
- In-range condition met for all six families; rule-check recall ≈ its false-alarm rate for all six.
- **All six families NOT SUPPORTED — the multivariate (PCA) detector misses them.** PCA recall: coordinated_shift 0.311, coordinated_drift 0.170, short_event 0.293, correlation_break 0.160, decoupling 0.074, minor_component_shift 0.014.
- Severity sweep: coordinated_shift reaches the criterion only at 2.5σ (PCA recall 0.753); minor_component_shift stays ≤ 0.059 even at 25σ. SPE is dominated by the larger residual directions (smallest minor eigenvalue 1.3e-4 vs largest 8.0), so shifts along the smallest-variance directions add little error.
- Single-feature separability (excess AUC): shift-type families 0.23–0.35 (visible to a single feature); correlation_break 0.009 and minor_component_shift 0.020 (hidden); decoupling 0.063 (borderline).

**Decision / Pivot Taken**
- Keep the failures visible: magnitudes were **not** tuned to make families pass.
- The hypothesis is currently **not supported for any family with the PCA baseline**. Whether the autoencoder changes this is the Phase 4 question.

**Next Steps (planned, pending decision)**
- Student runs the Phase 3 tests and notebook 02 on Windows; record those numbers as the reference.
- Phase 4 options: autoencoder as the multivariate detector; possibly add a variance-weighted baseline (Hotelling T² / Mahalanobis); report severity curves rather than single points.

---

### 2026-10-20 — Phase 4: autoencoder (code implemented; NOT yet trained)

**Activity / Experiment — implemented (code + unit tests; tests deliberately not run yet, as instructed)**
- `autoencoder.py`: configurable dense autoencoder (default 52 → 32 → 16 → 8 → 16 → 32 → 52):
  * trained on normal training rows only, with early stopping on the separate validation split;
  * score = per-sample mean squared reconstruction error, plus a per-feature error breakdown;
  * threshold calibrated on separate calibration runs;
  * save/load with feature-order, schema and checksum validation, an embedded scaler for raw-value inference, and leakage fingerprints.
- `workflow.py`: one function prepares all data roles. Training file → train/validation; testing file → calibration/evaluation/holdout (holdout locked).
- `plots.py`: training history, score distributions, per-feature errors; saving only via `reports/figures`.
- Config: new `[evaluation]` and `[autoencoder]` sections (the Phase 1–3 sections are unchanged).
- Notebook `03_autoencoder_training.ipynb` (not run; contains no outputs).

**Observation / Result**
- No training run has happened yet, so there are **no model results**.
- Pitfall found while writing the code: Keras' `EarlyStopping(restore_best_weights=True)` restores the best weights *only if* early stopping triggers. If training reaches the epoch limit, the last weights are kept, which would not match the reported best epoch.

**Decision / Pivot Taken**
- Added a callback that always restores the best-validation-loss weights; a unit test covers the "no early stop" case.
- **Thresholds are calibrated on testing-file calibration runs, not on the validation split**, because the validation split already selects the model (early stopping). This differs from the Phase 3 notebook, where PCA was calibrated on the validation split. Notebook 03 recalibrates PCA on the calibration runs so that all detectors are compared on equal terms.
- The evaluation runs are for reporting only; hyper-parameter studies (Phase 5) must use train/validation/calibration data. The holdout runs stay unseen for the live demo.
- The Phase 3 pass criteria (`HypothesisCriteria`) and generator defaults are unchanged, so autoencoder verdicts are comparable with the PCA verdicts.

**Next Steps (planned)**
- At the final verification stage: run the full test suite and notebooks 02 and 03 on Windows; record the training summary, threshold, clean false-alarm rate and the per-family verdicts (autoencoder vs PCA).
- Phase 5: benchmark, robustness and ablation (candidates: bottleneck size, depth, denoising, threshold strategy, a variance-weighted baseline).

---

### 2026-10-20 — Phase 6: FastAPI backend (code implemented; tests NOT run; no model served yet)

**Activity / Experiment — implemented (not executed)**
- `src/production_pulse/api/` (app factory, routes, model service, SQLite store, schemas, settings, error handlers):
  * `GET /health` reports the API, the model and the database separately;
  * `GET /model/status` shows only metadata present in `metadata.json`;
  * `POST /model/reload` reloads the model deliberately and needs an admin token from the environment;
  * `POST /predict` takes one sample or a batch, validates strictly, reorders features to the canonical order, and returns top contributing features;
  * `GET /metrics` serves a saved report or says it is unavailable;
  * `GET /history` pages newest-first, with a cursor.
- `reporting.py`: strict-JSON evaluation reports. Notebook 03 gained an optional cell 6b that saves its *measured* comparison as the report `/metrics` serves; the existing cells are unchanged.
- Config `[api]` (local host, explicit CORS origins, batch limit, model and database paths) with validated `PP_API_*` environment overrides; secrets come from the environment only.
- Tests: config, store, reporting, app (stub detector, no TensorFlow) and an integration test (tiny autoencoder on synthetic normal data). **Written, not executed** (deferred by instruction).
- `docs/API.md` with endpoints, error codes, Windows/Uvicorn commands and frontend connection notes.

**Observation / Result**
- No results. There is no trained model yet, so a started API would currently report `model.status = "not_found"`.

**Decision / Pivot Taken**
- Live inference uses `detect_raw(check_unseen=False)`. The training-row refusal is an evaluation guard and would wrongly reject legitimate live samples; evaluation code keeps the guard on.
- The API starts without a model (health stays available; `/predict` returns 503) and never trains on its own.
- Request validation checks feature *names* against the canonical 52-feature schema before checking model availability, so clients get schema errors even when no model is loaded.
- History failures do not block predictions (`history_stored: false` plus a warning), but `/history` reports 503 when the database is unavailable.
- Raw sensor values are not persisted by default (data minimisation).

**Next Steps (planned)**
- Phase 7: React + Tailwind dashboard against this API.
- Final verification: run all tests; run notebooks 02 and 03; start the API and record real `/health`, `/model/status` and `/metrics` output.

---

### 2026-10-20 — Phase 7: React + Tailwind dashboard (code implemented; NOT installed, built or run)

**Activity / Experiment — implemented (not executed)**
- `frontend/`: Vite + React + TypeScript + Tailwind dashboard "Production Pulse — Industrial Intelligence" with seven pages:
  * Overview, Live Monitor (user-initiated batch replay), Prediction workbench (CSV/JSON, validation, preview, results, CSV export);
  * Sensor Explorer, History (cursor paging), Evaluation (saved report only), Settings (read-only).
- A central typed API client with timeouts, response-shape checks and separate failure kinds. Client-side validation mirrors the API contract.
- Dark industrial design system (CSS-variable tokens) with a consistent light theme, keyboard navigation, reduced-motion support and responsive layouts.
- Vitest + React Testing Library tests with a mocked API client and test-only synthetic fixtures. **Written, not executed** (deferred by instruction).
- `docs/FRONTEND.md`: setup, configuration, CORS, troubleshooting.

**Observation / Result**
- No results. Node/npm were present only in Claude's Linux workspace (v22.23.2 / 10.9.8); the Windows versions are unverified. Nothing has been installed, and the dashboard has never connected to the API.

**Decision / Pivot Taken**
- **Batch limit:** the API contract does not expose `max_batch_size`, so the dashboard mirrors it with `VITE_MAX_BATCH_SIZE` and treats HTTP 413 as authoritative. The backend was not changed.
- **"Live Monitor" is a replay:** the API has no streaming source, so it is labelled as a user-initiated replay of a user-supplied file, not real-time plant monitoring.
- **Contribution rankings:** these are built from the API's top-5 contributors per sample and are labelled as such. Reconstruction error is presented as a deviation signal, not a causal explanation.
- **What can be plotted:** only raw input values plus an explicitly labelled display-only z-score. The model's scaled inputs are not returned by the API.
- **Duplicate-header detection:** CSV duplicate headers are detected with a separate header-row pass instead of relying on parser callback behaviour.

**Next Steps (planned)**
- Final verification: pytest, notebooks 01→03, start the API, then `npm install`, `npm run typecheck`, `npm test` and `npm run dev`; record what actually works.

---

### 2026-10-21 - Final verification: two autoencoder unit tests failed on Windows; diagnosed and corrected

**Activity / Experiment**
- The first full Windows `pytest` run reported two failures, both in `tests/test_autoencoder.py` (user-reported; the Windows log was not available to Claude at the time of the fix).
- Both were diagnosed by reading the code, then reproducing the failing fixture in a cloud container with the same library versions as `.venv313` (TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1, pytest 9.1.1; Python 3.13.16 there vs 3.13.14 on Windows). That reproduction is **diagnostic only** and is not a Windows test result.

**Observation / Result**
1. `test_best_weights_restored_even_without_early_stop` - **test-code defect, not an implementation defect.** The last line asserted `trained.n_fit_rows_ ...`, but this test builds its own detector `det` and has no `trained` fixture (the name resolved to the module-level fixture function, giving an `AttributeError`). The line had been left behind when the test was inserted next to `test_training_history_and_best_weights`, where it belongs.
2. `test_off_manifold_rows_score_higher` - **the "score at least doubles" assertion is not justified for this fixture; no implementation defect found.** The container reproduced the reported values exactly (mean score 0.7113 for clean rows, 1.0357 with `xmeas_5` shifted by +4 standard deviations; ratio 1.46, increase 0.3244).
   - The score is the *mean over 52 features* of squared reconstruction errors. A +4 sigma shift of one feature that the network does not reproduce can therefore add about 4.0^2 / 52 = 0.308 to the score; the observed +0.324 matches this, and the shifted sensor's own squared error rose from 1.005 to 16.30 (the shift was not reconstructed, as expected).
   - A ratio of 2 would need a clean-row error below roughly 0.31. The fixture network (hidden 16, bottleneck 4, 8 epochs, 1,000 training rows) is still improving at epoch 8 (validation loss 1.003 -> 0.665; a predictor that always outputs the mean scores 1.023), so its clean error is 0.71. The ratio measured the fixture's training quality, not the detector.
   - Context only (same data, not part of the test): the best linear 4-component PCA reconstruction gives ratio 2.83 (0.152 -> 0.430); two longer-trained autoencoders (60 and 300 epochs, lr 0.005, diagnostic runs) gave ratios 2.02 and 2.45. These runs are not project results.

**Decision / Pivot Taken**
- Item 1: moved the row-count assertion back into `test_training_history_and_best_weights` and gave the best-weights test its own check on `det` (assertions of the best-weights test itself unchanged).
- Item 2: kept the test as an implementation sanity check (no effectiveness claim, data and anomaly-family definitions unchanged) but replaced the ratio by three criteria that follow from the score's arithmetic and do not depend on how well the fixture is trained: shifted sensor's squared error rises by >= 0.5 x 4.0^2; mean score rises by >= 0.5 x 4.0^2 / 52; score rises for >= 90 % of rows. Observed in the container: 15.29, 0.324 and 98.3 % (295 of 300 rows).
- Limitation to state in the report: this test shows the scoring path responds to an off-manifold shift in the expected, analytically predicted way. It says nothing about detection performance on TEP, which is measured only by notebook 03 / the evaluation report.
- Status: **Windows confirmation of the corrected tests is pending** (see PROJECT_STATUS.md).

---

### 2026-10-21 — Packaging and academic deliverables; documentation audit (no experiment run)

**Activity / Experiment**
- Read the Windows verification evidence in `logs/verification/` (written by `scripts/verify/run.ps1`; not re-run in this stage).
- Wrote `README.md`, declared runtime dependencies in `pyproject.toml`, wrote `requirements-lock.txt`, reviewed and rewrote `.gitignore`, and wrote the academic documents: `RESEARCH_METHODOLOGY.md`, `REPORT_OUTLINE.md`, `EXPERIMENT_TEMPLATES.md`, `VIVA_QA.md`, `DEMO_SCRIPT.md`.
- Compared the dashboard's API client (`frontend/src/lib/api/`) and Evaluation page with the backend code (`routes.py`, `schemas.py`, `errors.py`, `reporting.py`) and with the keys written by notebook 03 §6b (`evaluation.compare_detectors`, `assess_family_detectability`). This was done by reading code. Nothing was run, installed, trained or built. The two environments were not modified.

**Observation / Result**
- Test evidence on file (2026-10-21, Windows, `.venv313`): full suite 371 collected → 369 passed, 2 skipped, 0 failed (28.2 s). The 2 skipped tests are the opt-in real-data tests, which passed 2/2 when run with `PP_RUN_REALDATA=1`. This also confirms the two earlier autoencoder test fixes (entry above): they were listed as pending Windows confirmation, and the log shows no failures.
- The same preflight recorded: all six original-file checksums MATCH, all package imports OK, Node v22.20.0, npm 11.6.1, RAM 15.25 GB. TensorFlow uses the CPU only.
- Dashboard and API agree on: endpoints and methods; query parameters (`limit`, `offset`, `before_id`, `anomalies_only`); request and response fields; the error envelope and all error codes; the model-status values; and the report keys (`setup`, `training`, `threshold`, `clean_false_alarm_rate`, `detector_comparison` with `detector` and `family` columns, `hypothesis_assessment_autoencoder`). The backend's CORS settings allow the headers the dashboard sends. **No source inconsistency was found, so no source file was changed.**
- Documentation inconsistencies found and fixed (all in `docs/` and `PROJECT_STATUS.md`):
  1. `docs/API.md` §5 did not list the 405 `method_not_allowed` and generic `http_error` codes that `errors.py` produces and the frontend type `ApiErrorCode` includes.
  2. Several documents said the tests had not been run and the Node/npm and RAM values were unknown. The 2026-10-21 logs now provide those facts. Status tables were rewritten to separate **written / tested / trained or executed / verified end-to-end**.
  3. `docs/ARCHITECTURE_PLAN.md` §1–9 describes the original plan (for example a `backend/` folder and six notebooks); a note now says §10 is the as-built record.
- Dependency finding: nothing in `src/` imports scikit-learn, scipy, seaborn or joblib. `scikit-learn` appears only in an optional cross-check test (`importorskip`). They are therefore declared as an optional `analysis` extra, not as runtime dependencies. `pyarrow` is a runtime dependency only for saving the labelled features as Parquet. `tomli` is needed only on Python < 3.11.
- Process error (mine): while validating `pyproject.toml` I ran a package download from inside the project folder, which left a stray `tomli-2.5.0-py3-none-any.whl` in the project root. It was the only file affected. I asked for delete permission and removed it. No original file, environment or data was touched.

**Decision / Pivot Taken**
- No result is recorded in any new academic document. All results fields are blank templates, and results-dependent viva answers are marked `[YOUR RESULTS]`.
- The lock file is a read-only snapshot taken from the `.dist-info` records of `.venv313`. It describes the environment that passed the tests; it was not produced by installing anything.
- `.gitignore` directory patterns were anchored to the project root (`/data/`, `/artifacts/`, `/logs/`) so that they cannot hide a source folder, and `*.sqlite*`, `*.db`, `*.npz`, `*.tmp`, `/reports/evaluation_metrics.json`, `/reports/*.csv` and `/reports/executed/` were added. `frontend/package-lock.json` is deliberately not ignored. The patterns were tested on a throwaway file tree in a scratch folder, with `git check-ignore`.
- The demo helper code in `DEMO_SCRIPT.md` is marked as **not executed**; it must be dry-run on the evaluation split before use.

**Next Steps**
- Final integration and verification, in this order (each stage records its own log): `scripts\verify\run.ps1 nb01`, `nb02`, `nb03` (trains the model; do not change settings after looking at the evaluation numbers); `api`; `frontend`; manual dashboard run against the API.
- Record every measured result with `docs/EXPERIMENT_TEMPLATES.md`, update the status tables, then implement and run Phase 5 and write the report.

### 2026-10-21 — Correction: notebook 02 run selection ignored the run roles (Priority 1 of the audit)

**Activity / Experiment**
- Read-only audit finding C1, then fixed. Notebook 02 drew its 40 evaluation runs with `rng(SEED).choice(all 500 testing runs, 40)`, ignoring the calibration / evaluation / holdout split used by `workflow.py`.
- Re-derived the split with `splitting.split_runs` (default config, seed 42, run ids 1..500) on a synthetic frame of run ids; no dataset was loaded. The old draw's first ids (32, 41, 42, 45, 46, 62, 89, 95, 112, 139) equal those printed by the executed notebook.

**Observation / Result**
- The old draw contained 26 evaluation, 10 calibration and 4 holdout runs. The holdout runs were 42, 196, 241 and 304. They were scored only; nothing was fitted, trained or calibrated on them.
- Earlier statements in this logbook and in the documents that the holdout stays "unseen" were therefore not fully accurate for those 4 runs.

**Decision / Pivot Taken**
- Added `splitting.select_runs` and `splitting.HOLDOUT_RUNS_PREVIOUSLY_SCORED`. Notebook 02 now draws only from the evaluation role. `tests/test_run_selection.py` pins the rule and re-derives the 4 run ids.
- Excluded the 4 runs from demo selection. Corrected `README.md`, `docs/DEMO_SCRIPT.md`, `docs/RESEARCH_METHODOLOGY.md` and `docs/VIVA_QA.md`.
- `reports/executed/02_synthetic_anomaly_validation.executed.ipynb`, the saved model, threshold and `reports/evaluation_metrics.json` were not touched. Old notebook 02 outputs belong to the old selection.

**Next Steps**
- Re-run notebook 02 once, after the demo-safety decision, into a separate output name so the historical executed copy is kept.

### 2026-10-21 — Priority 2: offline-vs-API parity test; notebook 03 holdout wording corrected

**Activity / Experiment**
- Added `tests/test_api_offline_parity.py`. It sends the same raw 52-feature rows through the OFFLINE path (`AutoencoderDetector.load` -> `scaler.transform_frame` -> `detect`) and through `POST /predict` (in-process FastAPI `TestClient`), using the saved `artifacts/models/autoencoder_v1` copied into a temporary fake project.
- Compared scaled features (captured inside the API call), reconstruction error, threshold and anomaly decision. Tolerances are documented in the test module docstring.
- Ran it in a cloud container (Linux, Python 3.13, TensorFlow 2.21.0, Keras 3.15.1, numpy 2.5.4, pandas 3.0.6 - the versions recorded in `metadata.json`), **not** in the Windows `.venv313`.

**Observation / Result**
- Cloud run: synthetic-deterministic rows 6 passed; with `PP_RUN_REALDATA=1` (5 rows of one evaluation-role testing-file run) 12 passed. Observed scaled-value and API-vs-offline score differences were exactly 0.0; versus an independent numpy reference the relative difference was about 3e-16 (synthetic) and 5e-16 (real).
- A negative control (a 0.1 % input scale error injected into a scratch copy of the API code) made the scaled-feature and score tests fail, so the tests can detect a mismatch.
- The 5 real rows were all below the threshold; the real-data case therefore does not exercise a positive decision. The synthetic rows do (2 below, 2 above).
- Not yet run on Windows.

**Decision / Pivot Taken**
- Notebook 03's source markdown no longer says the holdout is unseen: it discloses that runs 42, 196, 241 and 304 were scored by the earlier notebook 02 selection. Executed notebooks were not touched.

**Next Steps**
- Run the new test on Windows in `.venv313` and record the counts.


## 2026-10-21 — Equal-FPR comparison, integrity manifest, single-pass inference (Cloud)

**Done:** corrected notebook 02 executed to a new file; notebook 04 compares rule check (fixed and equal-FPR), PCA and the saved autoencoder on 350 evaluation runs at target FPR 1 % (thresholds from calibration runs only); manifest + `detect_with_errors` added; Evaluation page now flags synthetic anomalies and shows prevalence/random PR-AUC.
**Result:** see `docs/EVALUATION_RESULTS.md`. The autoencoder beats the equal-FPR rule check on 5 of 6 families but is not better than PCA on any family and is clearly worse on correlation_break and decoupling. minor_component_shift is at chance for all. Absolute recall is low.
**Decision:** no retraining and no magnitude changes; the weak result is reported as is.
**Next:** run the new code on Windows; Phase 5 ablations (train/validation/calibration only).


## 2026-10-21 (later) — Extreme history records, notebook 04 on Windows, robustness and ablation

**Extreme prediction records (history ids 6, 7, 8).** All three: score 4100.632679931901, threshold 0.8851983375918899, flagged, `sample_index` 0, `sample_id` null, one sample per request, three different request IDs, created 2026-10-21 08:40:52, 08:41:19 and 08:41:20 UTC (about 14:10 IST), model 3f2ade8e658c. They are not in `logs/verification/30_api_results.json` (that smoke run wrote ids 1-5 and 9-13 with `eval-run2-s1..s5` samples) and the API log keeps no request IDs.
Evidence: identical scores to the last digit means the same input three times; the top contributor shares (xmeas_11 47.6 %, then xmv_2, xmeas_30, xmeas_29, xmv_1) imply scaled deviations of about 105 to 319 training standard deviations. Using the saved scaler, the largest contributions correspond to raw values of roughly 4 or 156 for xmeas_11 (training mean 80.1) and 0.6 or 27 for xmeas_30 (mean 13.8): the input was far outside the training range. All-zero, all-one, all-minus-one and all-100 inputs were tested and do not reproduce this score.
Not established: the exact payload (inputs are not stored, `store_inputs=false`) or who sent it. A hand-typed or placeholder sample is consistent with the evidence; `reports/normal_sample.json` was created about two minutes later. No record was deleted and no threshold changed. The model behaved correctly: it flagged an input hundreds of standard deviations from normal.

**Notebook 04 on Windows.** `logs/verification/summary.txt` shows two `nb04` starts at about 15:24 IST with no completion line and `23_nb04.err.txt` ending after TensorFlow start-up messages and no error. The cloud run needs about 5 minutes (the bootstrap cell alone about 4 minutes with no progress output), so these look like interrupted runs, not a code failure; there is no record of a completed Windows run. A genuine defect was found while checking: the notebook would have failed at its last cell with a file-exists error once the report files were present (it never overwrites). Fixed: if the report files exist, the run now writes new files with a `.rerun-<timestamp>` suffix. Re-executed in the cloud to `reports/executed/04_equal_fpr_comparison.rerun-cloud.executed.ipynb`.

**Experiments.** See `docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md`. Noise at 0.25 std multiplies the fixed-threshold false-alarm rate by about 4 and 0.5 std by about 30; PCA stays slightly ahead. Seed variation of the base configuration is large; bottleneck 4 and 16 reach about the PCA level on one seed each; denoising 0.2 is worse. Negative/inconclusive results are reported as such.


## 2026-10-21 (evening) — Smoke-test port fix, overflow fix, demo tooling, finalization

| Date | Activity / Experiment | Observation / Result | Decision / Pivot Taken | Next Steps |
|---|---|---|---|---|
| 2026-10-21 | API smoke test kept failing with exit 3 | It was hard-wired to port 8000, where my own dev server was running; it also wrote smoke predictions into the real history DB and exited 0 even when checks failed | Port from `PP_API_SMOKE_PORT` (default: free port chosen by the OS), own server only, isolated DB, every response checked; `run.ps1` passes exit codes through | Windows run 17:36 IST: 15/15 checks, exit 0 |
| 2026-10-21 | Adversarial input probe of `/predict` | `xmeas_1 = 1e300` returned HTTP 200 with a null score and stored a NULL history row (float32 overflow) | Reject with 422 `value_out_of_range`, store nothing; regression test fails on the old code and passes on the new | — |
| 2026-10-21 | Demo example check | My first "fault-like" example (±4 sd on two pressures) was also caught by the rule check, so it did not show the project's point (individually in-range anomalies) | Replaced by a row from the documented generator (coordinated_shift) chosen by a fixed rule; kept the ±4 sd case only as an "obvious deviation" | Context printed: on that run AE flagged 14/69 injected rows, rule check 0 |
| 2026-10-21 | Held-out demo files | Exported holdout run 23 (clean + coordinated_shift seed 42 + labels) with `scripts/export_holdout_demo.py`; deliberately **not scored** | Examiner can pick another holdout run and seed with `--run/--seed`; scoring only after the demo with `--score` | Hold the live demo; fill the record sheet |
| 2026-10-21 | CSV export review | Exported `sample_id`/timestamp came from the user's file and could run as spreadsheet formulas (CSV injection) | Text cells starting with = + - @ get a leading apostrophe; numbers untouched; regression test added | — |

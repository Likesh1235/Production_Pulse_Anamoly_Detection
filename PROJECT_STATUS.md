# PROJECT STATUS — Production Pulse

**Title:** Production Pulse — Multivariate Process Anomaly Detection Using an Autoencoder
**Course:** 21CSE732P · Worklet No. 13 (Tier 2 — Integration)
**Current phase:** implementation, experiments and verification complete; next: live demo, report writing and viva (yours). See §0 for the current snapshot; §1–§5 are kept as the original audit record.
**Last updated:** 2026-10-21 (finalization pass: demo tooling, CSV-export fix, documentation refresh)

## 0. Current snapshot (2026-10-21, finalization pass)

Evidence is labelled **W** (Windows `.venv313`, `logs/verification/summary.txt`) or **C** (cloud Linux workspace). C results are not Windows results.

| Area | State | Evidence |
|---|---|---|
| Data pipeline, scaler, run-level splits | Done | W pytest + real-data tests; notebook 01 exit 0 (W) |
| Synthetic anomalies + rule check + PCA | Done | corrected notebook 02 exit 0 (W, 17:41 IST) |
| Autoencoder + threshold | Trained, saved, integrity manifest | notebook 03 (W); manifest verified (W, C) |
| Equal-FPR comparison (350 runs) | Done | notebook 04 exit 0 (W), matches C within 1.4e-7; `docs/EVALUATION_RESULTS.md` |
| Robustness + ablation | Done, single seed per variant | C only; `docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md` |
| API | Done | W smoke test 15/15 checks, exit 0 (17:36 IST); out-of-range inputs → 422 |
| Dashboard | Done | W typecheck, 78 tests, build (17:40 IST); C headless-browser runs against the real API |
| Demo tooling | Done | `scripts/make_demo_samples.py`, `scripts/export_holdout_demo.py`, `docs/DEMO_SCRIPT.md` (C rehearsal; live demo not yet held) |
| Report / viva | Materials ready | `docs/REPORT_OUTLINE.md` (with what you must supply), `docs/VIVA_QA.md` |

**Latest Windows runs (17:36–17:41 IST):** pytest 416 passed, 13 skipped (opt-in real-data tests), 0 failed; real-data/parity/demo tests 20 passed; API smoke test exit 0; frontend install/typecheck/test/build exit 0; notebook 02 (corrected) exit 0. Changes made after those runs (CSV-export formula protection, error-detail display, demo scripts and their tests, docs) were verified in C only; rerun `.\scripts\verify\run.ps1 pytest` and `.\scripts\verify\run.ps1 frontend` on Windows to cover them.

**Scientific findings (unchanged):** synthetic anomalies only; the autoencoder beats the single-feature rule check at equal false-alarm rate on 5 of 6 families, but PCA matches or beats it on all six; absolute recall is low; all pre-registered hypotheses are NOT SUPPORTED (autoencoder recall < 0.5).

**Holdout:** never used for fitting, calibration or tuning; runs 42, 196, 241 and 304 were scored once by an early notebook and are excluded from the demo. Holdout run 23 has been exported for the demo but not scored.

**Not claimed:** clean-machine installation, real-fault detection, autoencoder superiority over PCA, live streaming, deployment.

## 1. Folder contents (verified)
| File | Size | Type | Role |
|---|---|---|---|
| Applied AI Industry Worklets Student Template.xlsx | 47 KB | Excel workbook | Worklet catalog, selection form, deliverables tracker, logbook template |
| Mark Split 21CSE732P rubrics .pdf | 96 KB | PDF (2 pages) | Mark distribution & sub-criteria |
| TEP_FaultFree_Training.RData | 24.7 MB | gzip R workspace | `fault_free_training`, 250,000 × 55 |
| TEP_FaultFree_Testing.RData | 47.3 MB | gzip R workspace | `fault_free_testing`, 480,000 × 55 |
| TEP_Faulty_Training.RData | 494 MB | gzip R workspace | `faulty_training`, ≈5.0 M rows (inferred) |
| TEP_Faulty_Testing.RData | 837 MB | gzip R workspace | `faulty_testing`, ≈9.6 M rows (inferred) |

Added in Phase 1 (new files only, nothing overwritten): `PROJECT_STATUS.md`, `docs/RUBRIC_CHECKLIST.md`, `docs/RESEARCH_LOGBOOK.md`, `docs/ARCHITECTURE_PLAN.md`.
Added in the packaging stage (2026-10-21, new files only): `README.md`, `requirements-lock.txt`, `docs/RESEARCH_METHODOLOGY.md`, `docs/REPORT_OUTLINE.md`, `docs/EXPERIMENT_TEMPLATES.md`, `docs/VIVA_QA.md`, `docs/DEMO_SCRIPT.md`; modified: `pyproject.toml` (dependencies), `.gitignore`.
Added in Phase 2 Step 2.1: `docs/DATA_CHECKSUMS.md`. Also present: `docs/_access_check.txt` (68-byte write-access test file; deletion is blocked for Claude — safe for you to delete).

### 1a. Integrity check
Original dataset and rubric checksums verified as **matching** on 20 October 2026 — all six files (4 × `.RData`, rubric PDF, Excel template) MATCH their baseline SHA-256 fingerprints. Baseline values: `docs/DATA_CHECKSUMS.md`.

## 2. Environment
### 2a. Linux sandbox on your laptop (used by Claude) — verified
Python 3.10.12, pip 25.3, git 2.34.1, Node v22.23.2, npm 10.9.8, numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9, seaborn 0.13.2. RAM 3.8 GB, 2 CPUs. **Missing:** scikit-learn, tensorflow, fastapi, uvicorn, pyreadr, pyarrow, jupyter, pytest. Read/write access to the project folder: **verified** (these files were written there).

### 2b. Your Windows installation (what you will use in VS Code) — UPDATED 2026-10-20
- Windows Python environment: **Python 3.13.14**
- Intended project environment: **`.venv313`** (in the project folder)
- `.venv` (Python 3.14.3): **unused for now** — preserve it without modification
- Core package imports passed on Windows *(reported by student)*
- JupyterLab 4.6.4 starts successfully on **port 8889**; port 8888 is already occupied *(reported by student)*
- Package versions found in `.venv313` (checked by Claude from installed-package records, not by importing): tensorflow 2.21.0, keras 3.15.1, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, fastapi 0.153.0, uvicorn 0.54.0, pyreadr 0.5.7, pyarrow 26.0.0, jupyterlab 4.6.4, pytest 9.1.1
- Windows preflight 2026-10-21 (`logs/verification/`): Windows 11 Home (build 10.0.26220), RAM 15.25 GB total, 16 logical CPUs, PowerShell 5.1; all imports OK in `.venv313` including numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, httpx 0.28.1, pydantic 2.14.1; TensorFlow runs on CPU only (native Windows)
- Node.js **v22.20.0** and npm **11.6.1** found on PATH (preflight log). Not yet used for this project
- Still unknown: Git and VS Code versions on Windows
- Note: Claude cannot run Windows `.exe` programs (including `.venv313` Python) or reach JupyterLab on localhost; Windows-side runs are done by the student.

## 3. Blockers / risks
1. ~~Windows versions unknown~~ *(Update 2026-10-21: Python 3.13.14 / `.venv313`, JupyterLab, Node v22.20.0, npm 11.6.1 and RAM 15.25 GB are recorded in §2b; only Git and VS Code versions remain unknown and are not needed.)*
2. `faulty_testing` needs ≈4.1 GB RAM as float64 → needs a memory-safe one-time conversion (plan in docs/ARCHITECTURE_PLAN.md §2).
3. Project lives inside **OneDrive**: a Python virtual environment and `node_modules` folder (tens of thousands of files) would be synced constantly. Plan: put the virtual env outside OneDrive or exclude it — to be decided with you in Phase 2.
4. Folder name spelling "Anamoly" — left as-is (renaming could break OneDrive links); just be aware.
5. Course template sheets "Project Selection Form" and "Deliverables Tracker" are still blank.

## 4. Windows checks requested in the Phase 1 audit (completed — see §2b and §0)
Open VS Code → Terminal → New Terminal (PowerShell) and run each line:
```powershell
python --version
py -0p
pip --version
git --version
node --version
npm --version
code --version
jupyter --version
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
```
"Not recognized" for a command just means that tool is missing — that is fine, paste it anyway.

## 5. Original phase plan from the Phase 1 audit (superseded by §0 — kept for the record)
- [x] Phase 1 — Audit, rubric checklist, architecture proposal (this file)
- [ ] Phase 2 — Windows environment setup, data conversion to Parquet, EDA
- [ ] Phase 3 — Splits, preprocessing, anomaly injection + labeled test set
- [ ] Phase 4 — Rule & PCA baselines, autoencoder, threshold calibration
- [ ] Phase 5 — Benchmark, robustness, ablation
- [ ] Phase 6 — FastAPI backend + tests
- [ ] Phase 7 — React/Tailwind dashboard + held-out demo
- [ ] Phase 8 — Report, viva prep, final package

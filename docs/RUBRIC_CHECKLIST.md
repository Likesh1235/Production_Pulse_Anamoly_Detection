# Rubric & Requirements Checklist — Production Pulse

Course: Applied AI for Hyper-Personalized Production Systems / "Applied AI in Manufacturing — Industry 5.0 Project Worklets" (21CSE732P), AY 2026-27 (Odd), Sem-5, Batch-1, Slot-B. Faculty listed in rubric PDF: Dr. Udendharn.

Sources (read-only, never modified):
- `Mark Split 21CSE732P rubrics .pdf` (2 pages) — mark split and sub-criteria
- `Applied AI Industry Worklets Student Template.xlsx` — sheets: Read Me, Worklet Catalog, Project Selection Form, Deliverables Tracker, Research Logbook

Status legend: `[ ]` not started · `[~]` in progress or partly met · `[x]` done **and verified with evidence**. Updated 2026-10-21 from the files and logs in the project (W = Windows log, C = cloud workspace run). Ticks mean the work and its evidence exist, not that the hypothesis was confirmed. Marks are awarded by the faculty, not by this list.

---

## A. Our worklet (from Worklet Catalog, row 16)

| Field | Text in the workbook |
|---|---|
| Worklet No. | **13** |
| Tier | Tier 2 — Integration (multi-sensor deep learning, real-time detection, ensembles) |
| Title | Multivariate Process Anomaly Detection (Autoencoder) |
| Problem Statement | Process parameters that individually stay within limits can still combine into an abnormal operating state undetectable by single-variable rule checks. |
| USP | Autoencoder-based reconstruction error flags multivariate deviations that rule-based systems are structurally blind to. |
| R&D Component | Students must design a synthetic multivariate-anomaly injection scheme (correlated but individually-in-range perturbations, documented methodology) on a public process dataset, since this exact anomaly type is rarely pre-labeled anywhere, then prove the autoencoder catches it while a single-variable rule-check baseline does not. |
| Tech stack | Python, TensorFlow/Keras (autoencoder), public multivariate process dataset |

> Note: the "Project Selection Form" and "Deliverables Tracker" sheets in the template are still blank (no worklet number, team or dates entered). You need to fill these in yourself (or ask me to prepare a filled copy — the original will not be overwritten).

### A1. Worklet 13 official deliverables
- [x] D1. Synthetic multivariate-anomaly injection methodology **and** labeled test set *(6 families in `anomaly_generation.py`; labels kept separate from features; seeded. Labelled evaluation sets: notebooks 02 and 04 (350 evaluation runs, rate 0.05, seed 42). W: notebook 02 corrected run 17:41, exit 0)*
- [x] D2. Autoencoder model with reconstruction-error threshold calibration *(`artifacts/models/autoencoder_v1`, 52-32-16-8-16-32-52, best epoch 149 of 159; threshold 0.885198 from 96,000 calibration rows at target FPR 1 %; integrity manifest. W: notebook 03 exit 0)*
- [x] D3. Comparative validation: autoencoder vs. rule-based baseline **on the same injected anomalies** *(notebook 04 + `docs/EVALUATION_RESULTS.md`: same rows, same 1 % FPR; autoencoder beats the equal-FPR rule check on 5 of 6 families, bootstrap intervals exclude 0. W: notebook 04 exit 0, matches C within 1.4e-7)*
- [~] D4. Research logbook *(`docs/RESEARCH_LOGBOOK.md`, Phase 1 to 2026-10-21 including failures and pivots; copy entries into the template's "Research Logbook" sheet yourself; keep it going until the viva)*

### A2. Key R&D requirement (what the viva will test)
- [x] Anomalies are **correlated** across variables and each variable stays **individually within its normal range**, verified programmatically *(notebook 03 hypothesis table: in-range fraction 1.00 for all six families)*
- [x] Injection methodology documented (types, magnitudes, durations, seeds) *(`anomaly_generation.py`, `docs/RESEARCH_METHODOLOGY.md` §4, saved settings)*
- [~] Evidence that the single-feature rule check **misses** these anomalies while the autoencoder catches them *(rule check: recall minus its own false-alarm rate ≤ 0.01 for every family = blind, criterion met. Autoencoder (notebook 03, 60 runs): recall 0.01–0.31 at about 1 % FPR, so it catches some events but not most, and minor_component_shift not at all; the pre-registered criterion "autoencoder recall ≥ 0.5" is **not met** for any family, so every family is NOT SUPPORTED. Report this honestly)*

## B. Universal Anti-Shortcut Checklist (Deliverables Tracker sheet)
- [x] U1. Dataset constructed/synthesised/augmented *(synthetic in-range multivariate anomalies injected into the public TEP fault-free data)*
- [x] U2. Comparative benchmark with results *(rule check, PCA, autoencoder at equal FPR; ROC-AUC, PR-AUC, precision/recall/F1, bootstrap intervals. Finding: PCA ≥ autoencoder on all six families)*
- [x] U3. Ablation or robustness test with results *(`docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md`: input-noise robustness of the saved model and 5 ablation models. C only; single seed per variant)*
- [~] U4. Logbook shows iteration *(failures and pivots recorded; transfer to the template sheet is yours)*
- [~] U5. Viva can probe "why" *(`docs/VIVA_QA.md` includes results-based answers; practise aloud)*
- [~] U6. Demo on data not previously shown *(procedure and tools ready: `docs/DEMO_SCRIPT.md`, `scripts/export_holdout_demo.py`; holdout run prepared but not scored; the live demo itself is still to be held)*

## C. Mark distribution (100 marks) — what exists as evidence

| Checkpoint | Sub-criterion (marks) | Evidence in the project | State |
|---|---|---|---|
| FP-I (10) | Worklet understanding (4) | `docs/RESEARCH_METHODOLOGY.md` §1–2, this file §A | written |
| | Literature / dataset survey (3) | dataset survey in methodology §3; **no literature survey in the repo** | **you must supply** |
| | Approach and team work-plan (3) | `docs/ARCHITECTURE_PLAN.md` | team plan: **you must supply** |
| FP-II (10) | Dataset plan (4) | `anomaly_generation.py`, methodology §4 | done |
| | Tooling matching the catalog stack (3) | Python 3.13, TensorFlow/Keras, FastAPI, React (W preflight) | done |
| | First logbook entries (3) | logbook from 2026-10-20 | done |
| PBL-I (15) | Dataset finalised and documented (5) | notebooks 01–02, `docs/DATA_CHECKSUMS.md` | done |
| | Working baseline with results (5) | rule check and PCA, notebook 02/04 | done |
| | Logbook depth (5) | logbook pivots (clipping bias, marginal check, PCA finding, holdout disclosure, port fix) | done |
| PBL-II (20) | ≥2 methods benchmarked (7) | notebook 04, `docs/EVALUATION_RESULTS.md` | done |
| | Ablation / robustness (7) | `docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md` | done (C, single seed) |
| | Logbook failures and progress vs plan (6) | logbook, `PROJECT_STATUS.md` | done |
| PBL-III (20) | Final quantitative results (8) | `docs/EVALUATION_RESULTS.md` | done |
| | Live demo on held-out data (7) | `docs/DEMO_SCRIPT.md` | **to be performed** |
| | Completeness vs deliverables (5) | §A1 | D1–D3 done, D4 ongoing |
| R21 (25) | Final report (10) | `docs/REPORT_OUTLINE.md` (outline, sources, placeholders) | **you must write** |
| | Live viva (10) | `docs/VIVA_QA.md` | practise |
| | Logbook trail (5) | logbook | keep updating |

## D. Notes stated in the rubric / workbook
- A generic AI-generated script with no experimental trail behind it will not score well at any checkpoint.
- The Research Logbook is scored at every checkpoint from FP-II onward and "carries more evaluation weight than the polish of the final code" (Read Me sheet).
- Marks are cumulative in spirit — a strong PBL-III cannot substitute for a missing baseline at PBL-I.
- The working demo will be run live on data not shown before.
- Logbook columns required by the template: Date · Activity/Experiment · Observation/Result · Decision/Pivot Taken · Next Steps.

## E. Extra scope from the project brief (beyond the rubric)
- [x] PCA baseline *(implemented; at equal FPR it matches or beats the autoencoder)*
- [~] Additional metrics *(confusion matrix, ROC-AUC, PR-AUC, false-alarm rate, bootstrap intervals done; detection delay not implemented)*
- [x] FastAPI backend: health, model status, predict, metrics, history, protected reload *(W: API smoke test 15/15 checks, exit 0; out-of-range inputs return 422)*
- [x] React + Tailwind dashboard *(W: typecheck, 78 tests, build exit 0; C: run against the real API in a headless browser)*
- [x] Automated tests and saved artifacts *(W: 416 passed, 13 opt-in skips; real-data 20 passed; model manifest verified)*
- [~] Reproducible package *(README, lock files, `run.ps1`; not tested on a clean machine)*

## F. Progress log (updated 2026-10-21)

| Phase | Status | Evidence |
|---|---|---|
| 1 Foundation | Completed | W pytest |
| 2 Data pipeline | Completed | W pytest + real-data tests; notebook 01 exit 0 |
| 3 Synthetic anomalies + baselines | Completed | notebook 02 (corrected) exit 0 on W |
| 4 Autoencoder + calibration | Completed | notebook 03 exit 0 on W; manifest verified |
| 5 Benchmark, robustness, ablation | Completed (C; single seed per ablation variant) | notebook 04 (W exit 0), `reports/experiments/` |
| 6 FastAPI + SQLite | Completed | W API smoke test 15/15 |
| 7 Dashboard | Completed | W typecheck/test/build; C headless-browser run |
| 8 Report, viva, demo | Materials prepared; report writing and live demo are yours | `docs/REPORT_OUTLINE.md`, `docs/VIVA_QA.md`, `docs/DEMO_SCRIPT.md` |

Design choice recorded in the logbook (2026-10-20): **seeded random run-based split** replaces the contiguous run ranges in ARCHITECTURE_PLAN §3.

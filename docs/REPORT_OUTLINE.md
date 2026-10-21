# Final Report Outline — Production Pulse

Course 21CSE732P · Worklet 13 · Rubric R21 (final report: technical accuracy, completeness, structure — 10 marks).

> **How to use this outline.** It is a skeleton. Text in `[FILL: …]` must be replaced by **measured** values or by your own writing. (Status refreshed 2026-10-21; the experiments listed as Ready have been run.) Do not paste a number that did not come from a saved report, an executed notebook or a test log. Each chapter lists its source material and its current readiness: **Ready** = the material exists, **Needs run** = needs an experiment that has not been run, **Needs writing** = only your own text is missing.

Target length: 35–50 pages plus appendices (adjust to the course's guideline).

| # | Chapter | Content | Source material | Readiness |
|---|---|---|---|---|
| 0 | Title page, declaration, acknowledgements, abstract (≤ 250 words), keywords, contents, figure/table lists | Abstract states the problem, method, **measured** headline result and its limits | written last; headline numbers in `docs/EVALUATION_RESULTS.md` | Needs writing (+ your details, see below) |
| 1 | Introduction | 1.1 Industrial motivation (undetected multivariate abnormal states). 1.2 Problem statement and USP (worklet 13). 1.3 Research question and hypotheses H1–H3. 1.4 Objectives. 1.5 Contributions. 1.6 Report structure | `docs/RESEARCH_METHODOLOGY.md` §1–2; rubric sheet "Worklet Catalog" row 13 | Needs writing |
| 2 | Literature and background | 2.1 Process monitoring: single-variable limits, PCA (SPE, Hotelling T²). 2.2 Autoencoders for anomaly detection. 2.3 Threshold selection. 2.4 The Tennessee Eastman Process and its public simulation datasets. 2.5 Synthetic anomaly injection in the literature. 2.6 Gap this project addresses | `[FILL: your literature survey; rubric FP-I asks for it — at least the sources you actually read]` | Needs writing |
| 3 | Dataset and exploratory analysis | 3.1 Files, shapes, columns (3 identifier + 52 process variables). 3.2 Validation checks (missing values, duplicates, dtypes). 3.3 Statistics, autocorrelation, correlation structure. 3.4 Checksum protection | `reports/executed/01_data_exploration.executed.ipynb` (Windows run); `docs/DATA_CHECKSUMS.md`; logbook Phase 1–2 entries | Ready |
| 4 | Methodology | 4.1 Run-level leakage-free split and roles. 4.2 Preprocessing (training-only scaling). 4.3 Synthetic anomaly families, range rule, labels. 4.4 Detectors: rule check, PCA, autoencoder. 4.5 Threshold calibration. 4.6 Evaluation protocol and hypothesis criteria. 4.7 Reproducibility | `docs/RESEARCH_METHODOLOGY.md` §3–6, §9; `docs/ARCHITECTURE_PLAN.md` §10 | Ready |
| 5 | System implementation | 5.1 Architecture diagram. 5.2 Python package modules. 5.3 FastAPI service: endpoints, validation, model lifecycle, SQLite history, error contract. 5.4 React + Tailwind dashboard: pages, API client, validation, data-honesty rules. 5.5 Software testing strategy and test evidence | `README.md` §2; `docs/API.md`; `docs/FRONTEND.md`; `logs/verification/` | Ready (Windows: API smoke 15/15, frontend 78 tests + build; screenshots: take your own) |
| 6 | Experiments and results | 6.1 Synthetic-data validation of the generator (notebook 02). 6.2 Autoencoder training: epochs run, best epoch, best validation loss, early-stopping outcome. 6.3 Threshold calibration and clean false-alarm rate. 6.4 Detector comparison per family: precision, recall, F1, FPR, PR-AUC, ROC-AUC, confusion matrices. 6.5 Hypothesis verdicts per family (H1–H3). 6.6 Per-feature error analysis of example events | notebooks 02, 03, 04 (executed copies in `reports/executed/`); `reports/evaluation_metrics.json` (03, 60 runs); `reports/evaluation_equal_fpr.json` + `docs/EVALUATION_RESULTS.md` (04, 350 runs, equal FPR) | Ready |
| 7 | Robustness and ablation | 7.1 Severity sweeps. 7.2 Architecture ablations (bottleneck, depth, denoising, regularisation). 7.3 Threshold-strategy comparison. 7.4 Noise / missing / frozen sensor tests. 7.5 Real TEP fault check (secondary) | `docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md`; `reports/experiments/*.csv/json` (input noise; bottleneck 4/16, denoising, seeds 43/44). Not done: severity sweeps, missing/frozen sensors, real faults | Ready (single seed; cloud run) |
| 8 | Demonstration on held-out data | 8.1 Protocol (frozen model, locked holdout, announced seed). 8.2 What was shown and the recorded outcome | `docs/DEMO_SCRIPT.md`, completed record sheet after the live demo | **Needs the live demo** |
| 9 | Discussion | 9.1 Which hypotheses were supported, which were not, and why. 9.2 What the autoencoder adds over rule check and PCA (only as far as measured). 9.3 Failures and pivots (clipping bias, marginal-check redesign, Phase 3 PCA finding, threshold-source change, best-weights pitfall, the two test fixes). 9.4 Threats to validity | logbook; `docs/RESEARCH_METHODOLOGY.md` §8; `docs/EVALUATION_RESULTS.md` ("Honest reading") | Needs writing |
| 10 | Conclusion and future work | Answer to the research question in measured terms. Future work: Hotelling T²/Mahalanobis baseline, windowed autoencoder, event-level metrics and detection delay, real-fault evaluation, repeated seeds, streaming source, authentication | — | Needs writing |
| A | Appendix A — Configuration | `configs/default.toml`, library versions (`requirements-lock.txt`), seeds | repo | Ready |
| B | Appendix B — Research logbook | Full `docs/RESEARCH_LOGBOOK.md` (failures included; scored separately, 5 marks) | repo | Ready (grows) |
| C | Appendix C — API reference | `docs/API.md` | repo | Ready |
| D | Appendix D — Test evidence | Windows 2026-10-21 17:36–17:41 IST: pytest 416 passed / 13 skipped; real-data 20 passed; API smoke 15/15; frontend typecheck, 78 tests, build; notebooks 01–04 exit 0 | `logs/verification/summary.txt` | Ready |
| E | Appendix E — Rubric traceability | `docs/RUBRIC_CHECKLIST.md` | repo | Ready |
| F | References | Cited sources only | — | Needs writing |

## Mandatory figures and tables (create from measured outputs)

| ID | Item | Produced by | Status |
|---|---|---|---|
| F1 | System architecture diagram | draw from `README.md` §2 | Needs writing |
| F2 | Run-level split and data roles | roles table in `reports/evaluation_equal_fpr.json` → `reproducibility.roles` | Ready |
| F3 | Example run time series and correlation matrix | inline figures in `reports/executed/01_data_exploration.executed.ipynb` (3 images) | Ready (copy the image) |
| F4 | One example per injected family (clean vs injected) | inline figures in the corrected executed notebook 02 (3 images) | Ready (copy the image) |
| F5 | Autoencoder training history with best epoch | inline figure in `03_autoencoder_training.executed.ipynb` (do not re-run notebook 03; it retrains) | Ready (copy the image) |
| F6 | Reconstruction-error distribution on normal data with the threshold | same executed notebook 03 | Ready (copy the image) |
| F7 | Normal vs injected score distribution (correlation_break example) | same executed notebook 03 | Ready (copy the image) |
| F8 | Per-feature error of an injected event | same executed notebook 03 | Ready (copy the image) |
| F9 | Detector comparison chart; score vs sample index | dashboard Evaluation page (screenshot) or executed notebook 04 | Ready / your screenshot |
| F10 | Dashboard screenshots (each page) | take them yourself while following `docs/DEMO_SCRIPT.md` §3 | Needs your screenshots |
| T1 | Dataset summary | executed notebook 01 | Ready |
| T2 | Hyper-parameters and settings | `configs/default.toml` | Ready |
| T3 | Detector comparison per family (all metrics) | `reports/equal_fpr_comparison.csv`, `docs/EVALUATION_RESULTS.md` | Ready |
| T4 | Hypothesis verdict table with reasons | `reports/evaluation_metrics.json` → `hypothesis_assessment_autoencoder` (all NOT SUPPORTED: autoencoder recall < 0.5) | Ready |
| T5 | Ablation and robustness tables | `docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md` | Ready |
| T6 | Software test summary | `logs/verification/summary.txt` | Ready |

## Rules for writing the results chapters
1. Every number cites its source file or notebook cell, and the date of the run.
2. State the exact setup next to each number: seed, number of evaluation runs, anomaly rate, target false-alarm rate, which environment (Windows `.venv313`).
3. Report negative findings: every family is *not supported* under the pre-registered criteria (autoencoder recall < 0.5), and PCA matches or beats the autoencoder at equal false-alarm rate.
4. Report undefined metrics as *undefined*.
5. Say where each number was produced (Windows `.venv313` or the cloud workspace). The robustness and ablation results are cloud-only.
6. Describe the Live Monitor as a user-triggered CSV replay, not live streaming; the reconstruction error as a score, not a fault probability; contributions as locations of poor reconstruction, not causes.

## Information only you can supply (do not let anyone guess these)
- [FILL: student name(s), register number(s), team/batch name, section] — the template's "Project Selection Form" sheet is blank.
- [FILL: mentor / faculty guide name and department] (the rubric PDF lists the course faculty as Dr. Udendharn; confirm your guide).
- [FILL: project start date, review dates (FP-I, FP-II, PBL-I–III) and any feedback you actually received].
- [FILL: literature survey — only sources you have read, with full references]. No references are stored in this project; none should be invented.
- [FILL: declaration, acknowledgements, college name and logo as required by your department's report format].
- [FILL: dashboard screenshots you take yourself] and the completed demo record sheet (`docs/DEMO_SCRIPT.md` §6).
- [FILL: copy the logbook entries into the template's "Research Logbook" sheet and the deliverables into "Deliverables Tracker"] if your faculty asks for the workbook.


# Experiment-Recording Templates — Production Pulse

> **Everything in this file is blank on purpose.** Copy a template, fill it in **after** the experiment has been run, and keep the filled copy in `docs/experiments/` (create the folder when you need it) or paste it into `docs/RESEARCH_LOGBOOK.md`. Never fill a field with an expected or hoped-for value. If a run fails, record the failure.

Conventions: dates as `YYYY-MM-DD`; metrics exactly as printed or saved (4 decimals); write `undefined` where the code reports an undefined metric; write `—` for "not measured".

---

## T1. Experiment record (one per experiment or run)

```
Experiment ID:        EXP-___            (EXP-001, EXP-002, ...)
Date / time:          ____-__-__ __:__
Phase / rubric item:  ______________      (e.g. Phase 5 / PBL-II ablation)
Question / hypothesis: _____________________________________________
Decided before the run? (criteria written down before looking)   yes / no

Environment:          Windows · .venv313 · Python ____ · TensorFlow ____ · numpy ____ · pandas ____
Git commit or file-hash of the code: ______________  (or "no git; zip date ____")
Command or notebook:  ______________________________________________
Seed(s):              ____
Settings changed from configs/default.toml (only the differences):
  ______________________________________________
Data roles used:      train ☐  validation ☐  calibration ☐  evaluation ☐  holdout ☐ (holdout must stay unchecked)
Evaluation runs used: ____ of ____       Anomaly rate: ____   Families: ____________
Model artifact / model_id: ______________  Threshold: ______  Threshold source: ______________
Output files:         ______________________________________________  (paths under reports/ or artifacts/)
Run time / hardware:  ______ min · RAM ____ GB

RESULTS (copy from the saved output; do not retype from memory)
  Clean false-alarm rate on evaluation data: ______
  (insert tables T3 / T4 / T5 as needed)

Observation (what the numbers show, in plain words):
  ______________________________________________
Unexpected behaviour / failures / warnings:
  ______________________________________________
Decision / pivot (what changes because of this):
  ______________________________________________
Does this change an earlier conclusion?  yes / no — which: ____________
Next step: ______________________________________________
Evidence pointer (log file, executed notebook, figure): ______________
```

## T2. Research-logbook entry (the five columns required by the course template)

| Date | Activity / Experiment | Observation / Result | Decision / Pivot Taken | Next Steps |
|---|---|---|---|---|
| | | | | |

Rules: log failures and dead ends too (they are graded). Link to the experiment record (T1) instead of repeating long tables.

## T3. Per-family detector comparison (one block per run)

Run ID: `EXP-___` · seed `____` · evaluation runs `____` · anomaly rate `____`

| Family | Detector | Precision | Recall | F1 | FPR | PR-AUC | ROC-AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coordinated_shift | rule_check | | | | | | | | | | |
| | pca | | | | | | | | | | |
| | autoencoder | | | | | | | | | | |
| coordinated_drift | rule_check | | | | | | | | | | |
| | pca | | | | | | | | | | |
| | autoencoder | | | | | | | | | | |
| short_event | … | | | | | | | | | | |
| correlation_break | … | | | | | | | | | | |
| decoupling | … | | | | | | | | | | |
| minor_component_shift | … | | | | | | | | | | |

## T4. Hypothesis verdicts per family (autoencoder as the multivariate detector)

Criteria used (default, unchanged unless stated): in-range ≥ 0.99 · rule-check recall − FPR ≤ 0.05 · multivariate recall ≥ 0.5 · marginal AUC excess ≤ 0.05

| Family | In-range fraction | Rule-check recall | Rule-check FPR | Autoencoder recall | Marginal AUC excess | Verdict | Reasons (as printed) |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Same table again with PCA as the multivariate detector (for H3): `[ ]` filled · run ID `EXP-___`

## T5. Ablation table (one per studied variable)

Study: `______` (e.g. bottleneck size) · selection data: validation + calibration only · seed(s): `____`

| Setting | Epochs run | Best epoch | Best validation loss | Threshold | Clean FPR (calibration) | Recall per family (fill columns) | Notes |
|---|---|---|---|---|---|---|---|
| (baseline: default) | | | | | | | |
| | | | | | | | |

Final configuration chosen: `______` · reason (based on validation/calibration data only): `______` · evaluation run for the chosen configuration: `EXP-___` (one run).

## T6. Robustness / severity sweep

| Family | Magnitude | Autoencoder recall | PCA recall | Rule-check recall | In-range fraction |
|---|---|---|---|---|---|
| | | | | | |

Perturbation studies (noise, missing or frozen sensor, real faults): record the perturbation definition, its seed, and the clean false-alarm rate **under the same perturbation** next to the recall.

| Perturbation | Parameter | Clean FPR | Recall (family) | Notes |
|---|---|---|---|---|
| | | | | |

## T7. Threshold-strategy comparison (planned)

| Strategy | Fitted on | Threshold | Clean FPR (evaluation) | Recall (family) | F1 |
|---|---|---|---|---|---|
| target-FPR on calibration (current) | calibration | | | | |
| | | | | | |

## T8. Training record (notebook 03)

| Item | Value |
|---|---|
| Date, environment | |
| Seed | |
| Architecture / settings changes | |
| Epochs run · best epoch · best validation loss | |
| Early stopping triggered? | |
| Training time | |
| Calibration: target FPR · threshold · number of calibration rows | |
| Reload check ("reloaded model reproduces calibration scores") | |
| Model folder and `model_sha256` (first 12 characters = API `model_id`) | |

## T9. Failure and pivot log

| Date | What failed or surprised | Evidence (log, number) | Cause (known / suspected / unknown) | Change made | Effect on earlier results |
|---|---|---|---|---|---|
| | | | | | |

## T10. Software verification record

| Date | Stage | Command | Result (counts as printed) | Log file | Notes |
|---|---|---|---|---|---|
| 2026-10-21 | pytest, full suite | `scripts\verify\run.ps1 pytest` | 371 collected: 369 passed, 2 skipped, 0 failed | `logs/verification/10_pytest_full.*` | recorded from the log |
| 2026-10-21 | pytest, real data | same, `PP_RUN_REALDATA=1` | 2 passed | `logs/verification/11_pytest_realdata.*` | recorded from the log |
| | notebooks 01–03 | | | | |
| | API start and calls | | | | |
| | `npm install` / typecheck / test / build | | | | |
| | dashboard against API (manual) | | | | |

## T11. Demo record sheet — see `docs/DEMO_SCRIPT.md` §6.

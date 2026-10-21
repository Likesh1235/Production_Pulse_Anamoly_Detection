# Research Methodology — Production Pulse

> **Status of this document.** It describes the *design* that the code implements, and the experiments that are *planned*. No result is stated here. Every number belongs in a measured report (`reports/evaluation_metrics.json`, an executed notebook, or `docs/RESEARCH_LOGBOOK.md`). Each item is labelled **Implemented** (code and unit tests exist) or **Planned** (not implemented yet).

## 1. Problem and research question

Process variables that individually stay within their limits can still combine into an abnormal operating state. A single-variable rule check cannot see such a state (worklet 13, USP).

**Research question.** Can an autoencoder trained only on normal operation detect *multivariate* anomalies that are individually in range, where a single-variable rule check cannot, and how does it compare with a classical multivariate baseline (PCA)?

## 2. Hypotheses and how they are decided

Hypotheses are tested per anomaly family with explicit pass criteria fixed in the code (`evaluation.HypothesisCriteria`). They are not adjusted after seeing results. A family only counts as **SUPPORTED** if all of the following hold on held-out evaluation data:

| Criterion | Default | Meaning |
|---|---|---|
| In-range fraction ≥ 0.99 | `min_in_range_fraction` | The injected values stay inside each variable's normal range (the anomaly is "individually normal") |
| Rule-check recall − its false-alarm rate ≤ 0.05 | `max_single_feature_excess` | The single-feature rule check does not detect it |
| Multivariate detector recall ≥ 0.5 | `min_multivariate_recall` | The detector under test (autoencoder, or PCA) detects it |
| Marginal AUC excess ≤ 0.05 (where measurable) | `max_marginal_auc_excess` | No single feature separates injected from clean rows |

| ID | Hypothesis | Decided by |
|---|---|---|
| H1 | The injected families are individually in range and invisible to the single-feature rule check | in-range fraction and rule-check criteria above |
| H2 | The autoencoder detects each family that H1 holds for | autoencoder recall at the calibrated threshold (notebook 03) |
| H3 | The autoencoder is at least as good as the PCA baseline on the same examples | `compare_detectors` on identical labels (precision, recall, F1, AUCs) |

A hypothesis may turn out **not supported**. That outcome is reported as it is (the Phase 3 preliminary run already showed that for PCA; see the logbook).

## 3. Data

* **Source:** the four local Tennessee Eastman Process `.RData` files (three identifier columns `faultNumber`, `simulationRun`, `sample` and 52 process variables: `xmeas_1`…`xmeas_41`, `xmv_1`…`xmv_11`). They are read-only and protected by SHA-256 baselines (`docs/DATA_CHECKSUMS.md`).
* **Verified on these files:** fault-free training 250,000 × 55 (500 runs × 500 samples); fault-free testing 480,000 × 55 (500 runs × 960 samples); no missing values.
* **Faulty files** (494 MB, 837 MB) are not used by the current code. A real-fault check (**Planned**) needs a one-time memory-safe conversion first.
* **Not verified on these files:** the fault-introduction sample index (commonly documented as sample 21 in training and 161 in testing). Check it from the data before relying on it.
* **Citation:** confirm the dataset's original publication from your course material or the source you downloaded it from, and cite that in the report.

### 3.1 Data roles (Implemented — `workflow.py`, `splitting.py`)

Splits assign **whole simulation runs** with a seeded random split (seed 42). Each file is split separately, because run numbers repeat across files.

| Role | Source file | Fraction | Used for | Never used for |
|---|---|---|---|---|
| train | fault-free training | 0.8 | fitting the scaler, autoencoder, PCA, rule limits, anomaly-generator statistics | evaluation |
| validation | fault-free training | 0.2 | early stopping | threshold calibration, evaluation |
| calibration | fault-free testing | 0.2 | thresholds only (target false-alarm rate) | training, final reporting |
| evaluation | fault-free testing | 0.7 | final reporting only; synthetic anomalies are injected here | tuning of any kind |
| holdout | fault-free testing | 0.1 | the live demonstration only. Not loaded by `workflow.py` unless `include_holdout=True` (see the exception below) | everything else |

**Exception — holdout runs already scored.** The first version of notebook 02 drew its 40 runs from all 500 testing runs, ignoring this role split. With seed 42 that selection contained 4 holdout runs (42, 196, 241, 304), 10 calibration runs and 26 evaluation runs. The runs were only scored: nothing was fitted, trained or calibrated on them, and no threshold depends on them. Consequences: (a) the 4 holdout runs are **not unseen** and are excluded from demo selection (`splitting.HOLDOUT_RUNS_PREVIOUSLY_SCORED`); the other 46 were not scored by any detector. Notebook 01 (exploration) loads the whole testing file for aggregate summaries and histograms, so all holdout rows entered those descriptive plots; no model, threshold or detector used them; (b) the old notebook 02 results, including `reports/executed/02_synthetic_anomaly_validation.executed.ipynb`, belong to that old selection and are kept as historical evidence only; (c) notebook 02 now draws runs only from the evaluation role with `splitting.select_runs`, so it must be re-run before its numbers are quoted as results of the corrected selection (not done yet). The evaluation pool was also seen during Phase 3 design (26 of the old 40 runs were evaluation runs), so "evaluation data never informed any decision" is not strictly true either. Notebook 03's reported metrics use the evaluation role only and are not affected by the holdout issue.

**Deviation from the original plan** (architecture plan §3): random run assignment instead of contiguous run ranges, and calibration on the testing file's calibration runs instead of the validation split (the validation split already selects the model through early stopping). Both are recorded in the logbook.

### 3.2 Leakage controls (Implemented)
* Scaler fitted on the train split only (`preprocessing.FeatureScaler`).
* Row fingerprints detect overlap between fit, calibration and evaluation rows; baselines and the autoencoder raise `LeakageError` / refuse evaluation data that was seen in fitting.
* The live API turns that refusal off on purpose (`check_unseen=False`), because rejecting a legitimate live sample for equalling a training row would be wrong.
* Ground-truth labels and metadata are kept in separate frames and never enter the feature matrix.

## 4. Synthetic multivariate anomaly injection (Implemented — `anomaly_generation.py`)

Anomalies are injected into a copy of the **scaled** evaluation data. The generator is fitted on training-normal data only. All injection is seeded and logged per event (family, group, length, magnitude, modified features, number of clipped values).

| Family | Idea | Magnitude means |
|---|---|---|
| `coordinated_shift` | sustained step on k random features with random signs | shift in training standard deviations |
| `coordinated_drift` | the same, ramping linearly over the event | final shift in standard deviations |
| `short_event` | brief half-sine pulse (2–6 samples by default) on k features | peak shift in standard deviations |
| `correlation_break` | for a correlated group, the partners' deviations from their mean are inverted | inversion strength in (0, 1] |
| `decoupling` | partner features replaced by genuine normal values from a different time/run | blend fraction in (0, 1] |
| `minor_component_shift` | shift along the lowest-variance principal directions | shift in standard deviations along that direction |

* **Range rule:** an injection may not create a *new* range violation. Values are clipped into the training quantile range, or no further out than the clean value already was. This was a deliberate fairness fix, because clipping would otherwise "clean" slightly-out-of-range values and handicap the rule check (logbook, Phase 3 pivot 1).
* **Every family starts as `UNVERIFIED`.** Whether it is truly individually-in-range and invisible to single-feature rules is measured, not assumed (§2).
* Magnitudes are generator defaults. They have **not** been tuned to make any family pass.

## 5. Detectors

| Detector | Status | Fitted on | Score | Threshold |
|---|---|---|---|---|
| Single-feature rule check | Implemented | train: per-feature training quantile limits (0.0005 / 0.9995) | flags a row if any feature leaves its limits | built-in limits; its false-alarm rate is *measured*, not set |
| PCA reconstruction error (SPE/Q) | Implemented | train; components keep 0.90 of the variance | squared reconstruction error | calibrated on calibration-normal runs for the target false-alarm rate |
| Dense autoencoder (main) | Trained (notebook 03): best epoch 149 of 159, best validation loss 0.5205, 4,764 parameters; saved in `artifacts/models/autoencoder_v1` with an integrity manifest | train, early stopping on validation; always restores best-validation weights | mean squared reconstruction error over the 52 scaled features | calibrated on calibration-normal runs: the smallest observed score for which at most the target fraction (default 0.01) of calibration-normal samples score above it (`evaluation.threshold_for_fpr`) |
| Hotelling T² / Mahalanobis baseline | **Not implemented** (future work, motivated by the finding that SPE misses low-variance directions) | — | — | — |
| Windowed autoencoder | **Not implemented** (future work) | — | — | — |

Autoencoder default architecture: 52 → 32 → 16 → 8 → 16 → 32 → 52, ReLU, linear output, MSE loss, Adam (learning rate 0.001), batch 256, up to 200 epochs, early stopping patience 10, deterministic mode. All settings are in `[autoencoder]` of `configs/default.toml`.

## 6. Evaluation protocol

1. Fit on `train`, early-stop on `validation` (autoencoder), calibrate every threshold on `calibration` only.
2. Measure the clean false-alarm rate on `evaluation` (no injection). Notebook 03 used a seeded random subset of **60 evaluation runs** (`EVAL_RUNS_LIMIT = 60`); notebook 04 uses **all 350** evaluation runs with every calibrated detector at the same target false-alarm rate. Both use an anomaly rate of **0.05** per family, the same as notebook 02. State the values actually used in every report.
3. For each family, inject into `evaluation` with a fixed seed. Score all detectors **on the same injected rows and labels**.
4. Compute precision, recall, F1, false-positive rate, specificity, accuracy, confusion matrix, ROC-AUC and PR-AUC. Undefined cases (for example no predicted positives) are reported as *undefined*, never as 0.
5. Decide each hypothesis with the fixed criteria in §2.
6. Report on `evaluation` only once per configuration. Do not change hyper-parameters or thresholds after looking.
7. **Done since:** an equal-false-alarm-rate comparison of all detectors on all 350 evaluation runs with run-level bootstrap intervals (notebook 04, `docs/EVALUATION_RESULTS.md`). **Not done:** event-level detection rate, detection delay, MCC, repeated seeds.
8. The holdout runs are reserved for the live demo (`docs/DEMO_SCRIPT.md`), except the 4 already-scored runs, which are excluded from it (see §3.1).

## 7. Robustness and ablation (Phase 5 — **partly done**, cloud runs; results in `docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md`)

Rules: use `train`, `validation` and `calibration` runs for any study that chooses settings. Report on `evaluation` only for the final chosen configuration.

| Study | Variable | Supported by current code? |
|---|---|---|
| Bottleneck size, depth | `bottleneck`, `hidden_layers` | Bottleneck 4 and 16 **run** (single seed each); depth not run |
| Denoising autoencoder | `noise_std` > 0 | **Run** (noise_std 0.2, worse than base) |
| Regularisation | `dropout`, `l2` | Yes (config) |
| Severity sweep | anomaly `magnitude` per family | Yes (generator) |
| Training-data size | number of training runs | Needs a small helper |
| No scaling | scaler `method` | Needs checking |
| Threshold strategy (percentile, mean + kσ, F1-optimal) | — | Not implemented (only target false-alarm rate) |
| Additive Gaussian noise on evaluation data | `scripts/run_experiments.py robustness` | **Run** (0.1 / 0.25 / 0.5 training sd; fixed and recalibrated thresholds) |
| Missing or frozen sensor | — | Not implemented |
| xmv-only or xmeas-only features | — | Not implemented |
| Real TEP faults (secondary check) | faulty files | Not implemented (needs conversion) |

## 8. Threats to validity

* **Synthetic anomalies are not real faults.** Success on injected families does not show success on real plant faults. The planned real-fault check addresses this only partly.
* **Generator choices drive the outcome.** The families, magnitudes and event lengths are design decisions. The report must show severity curves, not a single point.
* **Strong autocorrelation** (lag-1 about 0.993 measured in the Phase 3 smoke run) makes neighbouring samples near-duplicates. Splitting by run avoids leakage between splits, but effective sample size is smaller than the row count.
* **Single seed.** One split and one seed give one estimate. Re-training the base configuration with seeds 43 and 44 showed large seed-to-seed variation (mean PR-AUC 0.166–0.235), so single-seed differences below about 0.07 PR-AUC are not meaningful.
* **Threshold depends on the calibration runs.** The realised false-alarm rate on evaluation data can differ from the target. It is measured and reported.
* **Environment.** Notebooks 01–04 ran on Windows (`.venv313`); notebook 04 results agree with the cloud run within 1.4e-7. The robustness and ablation experiments ran only in the cloud workspace.
* **Reconstruction error is not a causal explanation.** Per-feature contributions show where the model reconstructs badly, not why the process deviates.

## 9. Reproducibility
* Fixed seed (`random_seed = 42`), saved split manifests, all settings in `configs/default.toml`, saved model metadata (feature order, settings, threshold, scaler, checksum, library versions).
* Exact package versions: `requirements-lock.txt`.
* Original data protected by checksums; outputs only through `safe_output_path`.
* Test suite on Windows (2026-10-21 17:39 IST): 416 passed, 13 skipped (opt-in real-data tests), 0 failed; with `PP_RUN_REALDATA=1`: real-data, parity and demo tests 20 passed.

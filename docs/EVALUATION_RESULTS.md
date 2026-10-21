# Evaluation results — equal-FPR comparison (notebook 04)

All numbers below come from `reports/executed/04_equal_fpr_comparison.executed.ipynb`, run in a **cloud (Linux) workspace** with Python 3.13, TensorFlow 2.21.0, numpy 2.5.4, pandas 3.0.6. They have **not** been reproduced on Windows yet (see "Reproduce on Windows").
Files: `reports/evaluation_equal_fpr.json`, `reports/equal_fpr_comparison.csv`, `reports/equal_fpr_bootstrap.csv`, `reports/equal_fpr_differences.csv`.

## What is evaluated

* **Anomalies are SYNTHETIC** (injected into fault-free data, 6 families, anomaly rate 5 %, seed 42). Nothing here shows detection of real plant faults.
* **Data roles** (run-level splits, seed 42): train 400 runs and validation 100 runs from the training file; calibration 100 runs, **evaluation 350 runs (336,000 rows)** and holdout 50 runs from the testing file. The holdout was not loaded by notebook 04. Runs 42, 196, 241 and 304 were scored by the earlier notebook 02 selection; they are excluded here and the holdout must not be described as never-scored.
* **Operating point:** every calibrated detector gets a threshold at target FPR 1 % on the **calibration** runs only. Nothing was tuned on the evaluation or holdout runs. `rule_check_fixed` (threshold 1.0, the original baseline) is shown for reference and has a 4.4 % false-alarm rate, so it is not comparable at the same point.
* The saved model, scaler and threshold were **not retrained**. Notebook 04 asserts that the saved scaler and threshold equal a fresh train-only fit / calibration-only computation (difference 0.0).
* Prevalence is 5 %, so the random PR-AUC reference is about 0.05. Accuracy is not used.

## Clean false-alarm rate on evaluation runs (should be near 1 %)

| Detector | Threshold | False-alarm rate |
|---|---|---|
| rule_check_fixed | 1.0 | 4.41 % |
| rule_check_equal_fpr | 1.120 | 1.08 % |
| pca | 11.875 | 0.97 % |
| autoencoder | 0.885 | 1.04 % |

## Per-family results at ~1 % FPR (evaluation runs; recall = detection rate)

| Family | Detector | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| coordinated_shift | rule (equal FPR) | 0.058 | 0.012 | 0.021 | 0.752 | 0.111 |
| | PCA | 0.627 | 0.308 | 0.414 | 0.898 | 0.468 |
| | autoencoder | 0.617 | 0.313 | 0.416 | 0.889 | 0.463 |
| coordinated_drift | rule | 0.058 | 0.013 | 0.021 | 0.649 | 0.081 |
| | PCA | 0.502 | 0.185 | 0.270 | 0.743 | 0.272 |
| | autoencoder | 0.470 | 0.174 | 0.254 | 0.736 | 0.255 |
| short_event | rule | 0.047 | 0.010 | 0.017 | 0.701 | 0.097 |
| | PCA | 0.612 | 0.291 | 0.394 | 0.820 | 0.408 |
| | autoencoder | 0.589 | 0.282 | 0.381 | 0.813 | 0.392 |
| correlation_break | rule | 0.049 | 0.011 | 0.018 | 0.507 | 0.051 |
| | PCA | 0.481 | 0.170 | 0.251 | 0.718 | 0.253 |
| | autoencoder | 0.318 | 0.092 | 0.142 | 0.650 | 0.150 |
| decoupling | rule | 0.055 | 0.012 | 0.020 | 0.510 | 0.051 |
| | PCA | 0.291 | 0.075 | 0.120 | 0.654 | 0.133 |
| | autoencoder | 0.197 | 0.048 | 0.078 | 0.603 | 0.095 |
| minor_component_shift | rule | 0.065 | 0.014 | 0.023 | 0.508 | 0.052 |
| | PCA | 0.048 | 0.009 | 0.016 | 0.520 | 0.052 |
| | autoencoder | 0.055 | 0.011 | 0.019 | 0.515 | 0.052 |

(Rule ROC/PR-AUC are identical for the fixed and equal-FPR variants because they rank by the same score; only the threshold differs.)

## Paired differences (autoencoder minus baseline), run-level bootstrap (100 resamples, 95 % percentile intervals)

* **vs rule check at equal FPR:** the autoencoder is higher on recall, ROC-AUC and PR-AUC with intervals excluding 0 for coordinated_shift, coordinated_drift, short_event, correlation_break and decoupling. minor_component_shift: no clear difference.
* **vs PCA:** no clear difference on coordinated_shift and coordinated_drift; the autoencoder is **lower** on short_event (small gap), correlation_break and decoupling (clear gaps, e.g. PR-AUC −0.10 and −0.04); minor_component_shift shows no meaningful difference.

## Honest reading

* The autoencoder clearly beats a per-feature rule check at the same false-alarm rate, but **it does not beat the linear PCA baseline** on any family, and is worse on three.
* Absolute detection is low at 1 % FPR (recall about 0.05 to 0.31). `minor_component_shift` is at chance for every detector.
* The hypothesis "the autoencoder is better than PCA" is **not supported** by this experiment.
* Sample-index effects: clean false-alarm rates by sample band (1-250, 251-500, 501-750, 751-960) vary mildly (about 0.8 % to 1.1 % for the autoencoder on calibration runs). The training file only has samples 1-500, so bands above 500 are outside the training range; see `sample_index_effects` in the JSON.

## Limitations

* One seed, one anomaly rate, one architecture (52-32-16-8-16-32-52), no Phase 5 ablations or robustness experiments.
* 350 evaluation runs are fixed by the seeded split and eligible runs exclude the holdout and the four previously scored runs. The evaluation pool was visible during Phase 3 design of the anomaly families, so the design may be biased in ways not measurable here.
* Bootstrap uses 100 resamples (percentile intervals, run-level clusters); more resamples would tighten the intervals.
* The `reproducibility.manifest` field says `absent` because notebook 04 was executed before the manifest was created; the manifest has since been written (`artifacts/models/autoencoder_v1/manifest.json`).

## Reproduce on Windows

```powershell
.\scripts\verify\run.ps1 nb04      # writes reports\executed\04_equal_fpr_comparison.executed.ipynb (never overwrites an existing one)
$env:PP_API_METRICS_REPORT = "reports/evaluation_equal_fpr.json"   # let the dashboard Evaluation page show this report
```

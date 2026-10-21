# Robustness and ablation experiments (cloud run, 2026-10-21)

Script: `scripts/run_experiments.py` (`robustness`, `ablation`). Results: `reports/experiments/robustness_20261021.{csv,json}`, `reports/experiments/ablation_20261021.{csv,json}`; experimental models: `artifacts/experiments/20261021/<variant>/`.
Environment: cloud Linux, Python 3.13, TensorFlow 2.21.0. **Not yet run on Windows.** Anomalies are SYNTHETIC (6 families, rate 5 %, seed 42).

Controls: fit on train, early stopping on validation, thresholds from calibration rows only (target FPR 1 %), results on the 350 evaluation runs (336,000 rows). Holdout never loaded. The saved model `autoencoder_v1` was only read (its SHA-256 and manifest verified unchanged afterwards). **No variant was selected or tuned on these results, and none replaces the saved model.**

## 1. Input-noise robustness (saved model, no retraining)
Gaussian noise (std = 0.1 / 0.25 / 0.5 training standard deviations) was added to the scaled features of both normal and injected evaluation rows. Two threshold policies: the clean calibration threshold kept fixed, and a threshold recalibrated on noise-added *calibration* rows.

Clean-row false-alarm rate, fixed threshold (target 1 %):

| noise std | autoencoder | PCA | rule (equal FPR) |
|---|---|---|---|
| 0 | 1.04 % | 0.97 % | 1.08 % |
| 0.1 | 1.31 % | 1.22 % | 1.18 % |
| 0.25 | 4.03 % | 3.50 % | 1.76 % |
| 0.5 | 33.99 % | 30.42 % | 5.33 % |

Mean over six families of ROC-AUC / PR-AUC (threshold-free): autoencoder 0.701/0.235 → 0.647/0.119 and PCA 0.725/0.264 → 0.670/0.165 from noise 0 to 0.5. With thresholds recalibrated on noisy calibration data, false alarms stay near 1 % and mean recall falls (autoencoder 0.153 → 0.059, PCA 0.173 → 0.099).
Reading: with a fixed threshold, noise of 0.25 std already multiplies false alarms by about 4 and 0.5 std makes the alarm unusable; ranking quality degrades gradually. PCA is slightly better than the autoencoder at every noise level; neither is robust. The rule check is less sensitive to false alarms but detects almost nothing.

## 2. Model ablation (separate experimental models, seed 42 unless stated)
Mean over six families at ~1 % FPR (clean false-alarm rate 0.97–1.04 % for all):

| model | recall | ROC-AUC | PR-AUC |
|---|---|---|---|
| saved autoencoder_v1 (bottleneck 8) | 0.153 | 0.701 | 0.235 |
| same config, seed 43 | 0.130 | 0.692 | 0.209 |
| same config, seed 44 | 0.099 | 0.679 | 0.166 |
| bottleneck 4 | 0.182 | 0.710 | 0.270 |
| bottleneck 16 | 0.181 | 0.712 | 0.270 |
| denoising (noise_std 0.2) | 0.091 | 0.672 | 0.157 |
| PCA baseline | 0.173 | 0.725 | 0.264 |

Reading (honest):
* Seed-to-seed variation of the same configuration is large (PR-AUC 0.166 to 0.235), so single-seed differences below about 0.07 PR-AUC cannot be called real.
* Bottleneck 4 and 16 scored about 0.270 PR-AUC, above all three base-config seeds and about equal to PCA (0.264), but each has a single seed. This is a lead, not a finding; confirming it needs several seeds per variant and a decision rule fixed beforehand.
* Denoising with noise 0.2 was worse than the base configuration.
* No variant was clearly better than PCA; the saved model sits at the upper end of its own seed range, so the headline comparison with PCA in `EVALUATION_RESULTS.md` is, if anything, flattering to the saved model.
* Not run: depth/width ablations, multiple seeds per variant, an alarm policy (e.g. several consecutive alarms), real faults.

## Reproduce on Windows
```powershell
python scripts\run_experiments.py robustness --tag win01     # about 4 min
python scripts\run_experiments.py ablation --tag win01       # about 20-30 min; trains 5 models under artifacts\experiments\win01
```
Outputs use new file names and refuse to overwrite existing ones.

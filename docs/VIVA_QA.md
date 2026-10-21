# Viva Preparation — Questions and Answers

Rubric R21: the viva (10 marks) tests whether you can justify **data and modelling decisions on the spot**, not only describe what was built (anti-shortcut item U5).

> **Read this first.**
> * The answers explain the *design* and cite the file where it is implemented. They are correct for the code as written.
>
> * Questions marked **[YOUR RESULTS]** are answered from saved outputs (updated 2026-10-21): `reports/evaluation_metrics.json` (notebook 03, 60 evaluation runs), `docs/EVALUATION_RESULTS.md` (notebook 04, 350 runs, equal false-alarm rate) and `docs/EXPERIMENTS_ROBUSTNESS_ABLATION.md`. Re-read them before the viva and quote only what is there.
> * Be ready to say what is *not* done: repeated seeds per ablation variant, severity sweeps, event-level metrics and detection delay, real-fault evaluation, authentication on prediction endpoints, and a live plant stream (the Live Monitor replays a file).
> * If you do not know, say what you would measure to find out. That scores better than a guess.

---

## A. Problem and idea

**A1. What problem does the project solve?**
Process variables can each stay inside their individual limits while their *combination* is abnormal. A rule check on single variables cannot see that. The project detects such multivariate abnormal states with an autoencoder trained on normal operation.

**A2. What is the R&D component, and why is it needed?**
Anomalies that are correlated across variables but individually in range are rarely labelled in public data. So the project designs a synthetic injection scheme (six documented families) on real normal TEP data, with labels, and then *measures* whether the rule check misses them and the autoencoder catches them. (`anomaly_generation.py`)

**A3. Why an autoencoder rather than a classifier?**
Real abnormal states are rare, varied and mostly unlabelled. An autoencoder learns only what normal looks like. Anything it reconstructs badly is flagged, with no anomaly examples needed in training. A supervised classifier would only recognise the anomaly types it was trained on.

**A4. Why not just use PCA?**
PCA is the classical multivariate baseline and is in the project (`baselines.py`). The autoencoder is non-linear, so it could in principle model non-linear relations that PCA cannot. Whether it actually does better here is a measured question (H3), not an assumption.

**A5. What are your hypotheses, and what would count as failure?**
H1: the injected families are individually in range and invisible to the single-feature rule check. H2: the autoencoder detects them. H3: the autoencoder is at least as good as PCA. The pass criteria are fixed in `evaluation.HypothesisCriteria` (in-range ≥ 0.99; rule-check recall − FPR ≤ 0.05; multivariate recall ≥ 0.5). A family that misses them is reported as NOT SUPPORTED. See `docs/RESEARCH_METHODOLOGY.md` §2.

## B. Data

**B1. Which data did you use, and what did you verify?**
The four local TEP `.RData` files. Verified on the fault-free files: training 250,000 × 55 (500 runs × 500 samples), testing 480,000 × 55 (500 runs × 960 samples), no missing values. 52 process variables: 41 measured (`xmeas_1..41`) and 11 manipulated (`xmv_1..11`). I did not verify claims about the fault-introduction sample from the data. (logbook, Phase 1)

**B2. Why didn't you use the faulty files?**
The anomalies are synthetic by design (the worklet requires it), and the faulty files are large (494 MB and 837 MB). They need a memory-safe one-time conversion first. A real-fault check is planned as a secondary evaluation, not done. A guard in `data.py` refuses files over 100 MB unless explicitly allowed.

**B3. Why split by simulation run instead of by random rows?**
Samples inside a run are strongly autocorrelated (lag-1 about 0.993 in my smoke run). Random row splits would put near-duplicate neighbours in training and test and inflate performance. Whole runs are assigned to one role only. (`splitting.py`)

**B4. Why a seeded random run split instead of runs 1–400 / 401–500, as in your first plan?**
Contiguous ranges could carry a systematic effect of run numbering. A seeded random assignment keeps runs whole and is reproducible from the seed. I recorded this change in the logbook.

**B5. Explain the five data roles.**
train (fit everything), validation (early stopping only), calibration (threshold only), evaluation (final reporting only), holdout (locked for the live demo). See `docs/RESEARCH_METHODOLOGY.md` §3.1.

**B6. Why is the threshold calibrated on separate calibration runs, not on the validation split?**
The validation split already selects the model through early stopping. A threshold chosen on it would be optimistically biased. The calibration runs are never used for training or model selection. (logbook, Phase 4)

**B7. How do you prevent leakage?**
Run-level splits; the scaler fitted on the train split only; row fingerprints that raise `LeakageError` if fit, calibration and evaluation rows overlap; labels kept in separate frames; the holdout not loaded unless `include_holdout=True`. Honest caveat: an early version of notebook 02 scored 4 holdout runs (42, 196, 241, 304) because its run draw ignored the role split. They were never used for fitting or calibration, are excluded from the demo, and the notebook is now corrected (`docs/RESEARCH_METHODOLOGY.md` §3.1).

**B8. Why scale at all, and why scale on train only?**
Sensors have very different units and ranges. Without scaling, large-valued sensors dominate the squared error. Scaling statistics from the evaluation data would leak information about it into the model input.

## C. Synthetic anomalies

**C1. Describe your injection scheme.**
Six seeded families injected into a copy of scaled evaluation data: `coordinated_shift`, `coordinated_drift`, `short_event`, `correlation_break`, `decoupling`, `minor_component_shift`. Each event logs its family, group, length, magnitude and modified features. The generator learns ranges, correlations and covariance from training-normal data only. (`anomaly_generation.py`)

**C2. How do you know the injected values stay in range?**
It is not assumed. Injections are clipped so they cannot create a *new* range violation, and `assess_family_detectability` measures the in-range fraction per family (criterion ≥ 0.99). Every family starts as UNVERIFIED.

**C3. Why can't you just clip everything into the normal range?**
I found that this created a bias (logbook, Phase 3, failure 1): clipping also "cleaned" clean values that were already slightly outside the training range, so anomalous rows looked more normal than untouched rows. The rule check then scored recall 0.0, below its own false-alarm rate 0.039. The fix: an injection may not create new violations; values already outside keep their original extremeness.

**C4. Did you tune the anomaly magnitudes until the autoencoder succeeded?**
No. Magnitudes are the generator defaults and were deliberately not tuned to make families pass (logbook, Phase 3 decision). Severity sweeps are the right way to show where detection starts.

**C5. What is "decoupling" and why is it hard to detect?**
Partner features are replaced by genuine normal values from another time or run. Each value is perfectly normal, but the relationship between features is broken.

**C6. Why do you say synthetic anomalies are a limitation?**
Success on injected families does not prove success on real plant faults. The families are my design choices. That is why a real-fault secondary check is planned and why the report must show severity curves.

## D. Detectors

**D1. How does the rule check work and what is its false-alarm rate?**
Per-feature limits from training-normal quantiles (0.0005 and 0.9995). A row is flagged if *any* feature leaves its limits. Its false-alarm rate on clean data is measured, not set (0.0451 in my preliminary Linux smoke run, versus 0.0097 for PCA calibrated to 0.01; these preliminary figures are from a subset of runs, not the Windows reference).

**D2. Describe the autoencoder.**
Dense network 52 → 32 → 16 → 8 → 16 → 32 → 52, ReLU, linear output, MSE loss, Adam (learning rate 0.001), batch 256, up to 200 epochs, early stopping with patience 10 on the validation split. Trained on normal data only. The best-validation weights are always restored. (`configs/default.toml`, `autoencoder.py`)

**D3. What is the anomaly score?**
The mean squared reconstruction error over the 52 scaled features. A sample is flagged if the score is above the calibrated threshold.

**D4. How is the threshold chosen?**
`threshold_for_fpr`: the smallest observed calibration-normal score such that at most the target fraction (default 0.01) of calibration-normal samples score above it. The realised false-alarm rate on evaluation data can differ and is measured and reported.

**D5. Why a bottleneck of 8?**
It is a default design choice that forces compression of 52 correlated variables. It is **not** tuned or justified by experiments yet. The bottleneck ablation (Phase 5) would test it. Say that honestly.

**D6. What bug did you find with early stopping?**
Keras' `EarlyStopping(restore_best_weights=True)` restores the best weights only if early stopping triggers. If training hits the epoch limit, the last weights are kept, which would not match the reported best epoch. I added a callback that always restores the best-validation weights, with a unit test for the "no early stop" case.

**D7. A single-sensor anomaly adds little to a mean over 52 features. Is that a problem?**
It is a property of the score. A +4σ shift of one feature that is not reconstructed adds about 4² / 52 ≈ 0.31 to the mean squared error. For subtle or single-feature anomalies this is small compared with normal variation. Per-feature contributions (`reconstruction_errors`) show where the error sits. It is a known trade-off, and a different score (for example a maximum or a weighted sum) would be a design change to test.

**D8. Why did the PCA baseline fail in your preliminary run?**
PCA-SPE measures error in the *residual* directions. Shifts along the lowest-variance directions barely change it (smallest minor eigenvalue 1.3e-4 versus largest 8.0). A variance-weighted statistic (Hotelling T² / Mahalanobis) is a candidate for Phase 5. This was a preliminary Linux smoke run on a subset, not the Windows reference run.

## E. Evaluation

**E1. Why not just report accuracy?**
Anomalies are rare, so a detector that never flags anything has high accuracy. I report precision, recall, F1, false-positive rate, PR-AUC, ROC-AUC and the confusion matrix. PR-AUC is more informative than ROC-AUC under strong class imbalance.

**E2. What does "undefined" mean in your tables?**
For example, precision is undefined if nothing was predicted positive. The code returns an explicit undefined value with a reason. It is never reported as 0.

**E3. Why must all detectors be evaluated on the same examples?**
Otherwise differences could come from the data, not the detectors. `compare_detectors` checks that labels and predictions are aligned by index.

**E4. What is the difference between calibration and evaluation data?**
Calibration data sets the threshold. Evaluation data is used only to report. Looking at evaluation results and then changing the threshold would turn them into tuning data and bias the result.

**E5. [YOUR RESULTS] What is the autoencoder's clean false-alarm rate, and how does it compare with the target 0.01?**
1.05 % on 60 clean evaluation runs (notebook 03; seed 42, rate 0.05) and 1.04 % on all 350 evaluation runs (notebook 04). The target was 1 % on the 96,000 calibration rows, so the threshold transfers well to new runs of the same normal process.

**E6. [YOUR RESULTS] Which families does the autoencoder detect, and which not? Is each hypothesis supported?**
All six families are **NOT SUPPORTED** under the criteria fixed in advance. Two of the three criteria are met: every injected value is inside the normal range (in-range fraction 1.00), and the rule check is blind (its recall is within 0.01 of its own false-alarm rate). The third fails: autoencoder recall at about 1 % false alarms is 0.31 (coordinated_shift), 0.28 (short_event), 0.17 (coordinated_drift), 0.09 (correlation_break), 0.06 (decoupling) and 0.01 (minor_component_shift), below the 0.5 required. So the autoencoder detects some in-range multivariate anomalies that the rule check cannot, but most are missed.

**E7. [YOUR RESULTS] Does the autoencoder beat the rule check and PCA?**
At the same 1 % false-alarm rate on 350 runs: it beats the rule check on 5 of 6 families (e.g. coordinated_shift recall 0.31 vs 0.01, PR-AUC 0.46 vs 0.11), with bootstrap intervals that exclude zero. It does **not** beat PCA: no clear difference on coordinated_shift and coordinated_drift, and PCA is better on short_event, correlation_break (PR-AUC 0.25 vs 0.15) and decoupling. minor_component_shift is at chance for every detector. Random PR-AUC is about 0.05 (prevalence 5 %).

**E8. [YOUR RESULTS] The autoencoder misses most anomalies and does not beat PCA. Why report it?**
That would be a legitimate, reportable finding. Explain with measured evidence (per-feature errors, score distributions, severity sweep), and do not retune on the evaluation data.

**E9. What are the limits of your evaluation?**
Synthetic anomalies only; one split and one injection seed; run-level bootstrap intervals but no repeated training seeds for the main model (re-training with seeds 43/44 moved mean PR-AUC between 0.17 and 0.21, the saved model has 0.23); autocorrelation reduces the effective sample size; no detection delay or event-level metrics; the evaluation pool was visible while the anomaly families were designed.

**E10. [YOUR RESULTS] Why might PCA do as well as the autoencoder here?**
The fault-free TEP data are close to linear-Gaussian around one operating point, so a linear subspace (31 components, 90 % of variance) already captures the normal correlation structure; the small autoencoder (bottleneck 8) adds non-linearity that the data may not need. In the ablation, bottlenecks of 4 and 16 reached about PCA's level (PR-AUC about 0.27 vs 0.26) on one seed each, which is a lead, not a finding.

**E11. [YOUR RESULTS] How robust is the detector to sensor noise?**
With the threshold kept fixed, adding noise of 0.25 training standard deviations raised the false-alarm rate from 1.04 % to 4.0 %, and 0.5 sd to 34 %; PCA behaved similarly (30 %). Re-calibrating on noisy calibration data keeps alarms at 1 % but recall falls. Neither detector is robust; a real deployment would need re-calibration and an alarm policy.

**E12. Is the reconstruction error a probability that a fault is present?**
No. It is the mean squared reconstruction error of the 52 scaled features. The threshold is set so that about 1 % of normal calibration rows exceed it. A row above it is "unusual compared with training data", not "x % likely to be a fault".

**E13. Your examples were chosen by you. Isn't that cherry-picking?**
The demo example is chosen by a fixed rule (first injected row that the autoencoder flags and the rule check does not), and the script prints the context: on that run the autoencoder flagged 14 of 69 injected rows and the rule check 0. The held-out run and the injection seed are chosen by the examiner, and the aggregate results are in `docs/EVALUATION_RESULTS.md`.

## F. Software

**F1. Why FastAPI?**
Typed request validation, automatic interactive documentation (`/docs`), and a simple deployment path. The API is a thin layer: `routes.py` (HTTP), `service.py` (model lifecycle and prediction), `store.py` (SQLite), `schemas.py` (validation).

**F2. How does `/predict` work?**
Validates the JSON strictly (finite numbers only, no coercion of strings, no extra keys). Checks the feature names against the 52 canonical names (any order), then reorders them. The model's own training scaler is applied, the reconstruction error is computed, and the result with the top contributing features is stored and returned.

**F3. What happens if no model exists?**
The API starts anyway. `/health` stays up and reports the model status (`not_found`, `incompatible`, `not_calibrated`, ...). `/predict` returns 503 `model_unavailable`. The API never trains a model.

**F4. Why does live inference turn off the "unseen data" check?**
That check is an *evaluation* safeguard against leakage. A live service must not reject a legitimate sample because it equals a training row.

**F5. How is the API protected?**
Local by default (127.0.0.1), explicit CORS origins (`*` is rejected), model reload only with an admin token from the environment, no stack traces or file paths in errors, raw input values not stored by default. Known limits: no authentication on read and predict endpoints, and no request-body size limit beyond the batch limit. It is meant for local or demo use. (`docs/API.md` §8)

**F6. What does "Live Monitor" really do?**
It is a user-started replay of a CSV in batches against `/predict`. It is labelled as a replay because the API has no streaming source. It is not real-time plant monitoring.

**F7. Are the feature contributions an explanation of the cause?**
No. They show which features the model reconstructs worst (the API returns the top 5 per sample). That is a deviation signal, not a causal explanation.

**F8. How did you test it?**
371 pytest tests collected. On 2026-10-21 on Windows (`.venv313`): 369 passed, 2 skipped (real-data tests, opt-in), 0 failed; the real-data tests then passed 2/2. The suite includes API tests with a stub detector and an integration test with a tiny model on synthetic data. **Not yet run:** the frontend tests, the notebooks, the API with a trained model. I can say exactly what the tests do and do not show.

**F9. What did the two failing tests teach you?**
Both were test-code problems, not implementation defects (logbook, 2026-10-21). One had a stray reference to a fixture that did not exist in that test. The other asserted that the score at least doubles for a +4σ shift on a barely trained fixture, which is not justified by the score's arithmetic. I replaced it with criteria that follow from the arithmetic. The test therefore only shows that the scoring path responds in the expected way. It says nothing about detection performance on TEP.

**F10. How is the work reproducible?**
Seeds, saved split manifests, a single config file, saved model metadata with feature order, threshold, scaler, checksum and library versions, `requirements-lock.txt`, and checksums of the original data.

## G. Process and honesty

**G1. What did not work, and what did you change?**
See the logbook: the clipping bias; the misleading pooled marginal check, redesigned per feature; the decoupling anomaly that looked detectable because of autocorrelation; the preliminary result that PCA misses all six families; the threshold-source change; the early-stopping pitfall; the two test fixes.

**G2. What is complete and what is not?**
Complete and tested: data pipeline, anomaly generation, baselines, trained autoencoder with calibrated threshold, equal-FPR comparison, a first robustness and ablation study, API (smoke-tested on Windows) and dashboard (tests and build on Windows). Not done: repeated seeds per ablation variant, severity sweeps, missing/frozen-sensor tests, real-fault check, detection delay, the live demo and the final report. (`PROJECT_STATUS.md`)

**G3. Why should we believe your results?**
Because each number comes from a saved output with its setup, the splits prevent leakage, criteria were fixed before looking, and negative findings are reported.

**G4. If you had another month, what would you do?**
Repeat the ablation with several seeds per variant and a decision rule fixed in advance; add severity sweeps and missing/frozen-sensor tests; add a Hotelling T² / Mahalanobis baseline and a windowed autoencoder; add event-level metrics and detection delay; evaluate on real TEP faults; repeat with several seeds and report intervals.

**G5. Where would this fail in a real plant?**
Distribution shift (new products, seasons, sensor replacement) would change what "normal" looks like. Noisy or missing sensors would raise false alarms. The model gives no cause. Operators need an alarm policy (for example require several consecutive alarms) that was not evaluated here.

---

## Quick self-test (answer aloud without notes)
1. Which five data roles exist and what is each used for?
2. How is the threshold chosen, and on which data?
3. Why can the rule check be blind to a `correlation_break` event?
4. What does `is_anomaly` mean in the API?
5. Which results are cloud-only, and which were run on Windows?
6. Why are all six hypotheses NOT SUPPORTED, and which criterion failed?
7. Why is the Live Monitor not real-time monitoring?

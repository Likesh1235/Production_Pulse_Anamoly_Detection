# Demonstration Script — Production Pulse

Rubric: PBL-III "Demo run live on previously unseen/held-out data" (7 marks) and anti-shortcut item U6 ("demo is run on data not previously shown, not a rehearsed run").

> **Status (2026-10-21).** The request files and the API/dashboard steps below were rehearsed in the cloud workspace (Linux) against the saved model; the results quoted in §3 come from that rehearsal. The API smoke test, dashboard build and tests have also passed on Windows (`logs/verification/summary.txt`). **No live demo has been held yet**, and the held-out run prepared in §4 has deliberately **not** been scored by anyone.

## 0. Start, check, stop (Windows PowerShell, from the project folder)

```powershell
# Terminal 1 - backend (keep it open)
.\.venv313\Scripts\python.exe -m uvicorn production_pulse.api.main:app --app-dir src --host 127.0.0.1 --port 8000

# Terminal 2 - readiness check: status "ok", model "ready", database "ready"
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 4

# Terminal 2 - dashboard (keep it open), then open http://127.0.0.1:5173
cd frontend; npm run dev
```
Stop: press **Ctrl+C** in each terminal (Uvicorn and Vite shut down cleanly).
If port 8000 is already in use, a server is probably still running from earlier: check `Invoke-RestMethod http://127.0.0.1:8000/health`; if it answers, use it, or stop it with Ctrl+C in its own terminal.

Prepare the request files once (new files only; they refuse to overwrite):
```powershell
.\.venv313\Scripts\python.exe scripts\make_demo_samples.py          # reports\demo\01..06 *.json and demo_samples.csv
.\.venv313\Scripts\python.exe scripts\export_holdout_demo.py        # reports\demo\holdout_run<R>*.csv (not scored)
```

## 1. Principles
1. **Frozen before the demo.** The model, its threshold and all settings are fixed (model id `3f2ade8e658c`, threshold 0.885198, target false-alarm rate 1 % on 96,000 calibration rows). Nothing is retrained or re-tuned afterwards.
2. **Held-out means not shown or tuned on.** The 50 holdout runs (seeded split) were never used to fit, calibrate or tune anything. Runs 42, 196, 241 and 304 were scored once by an early notebook selection and are excluded.
3. **The audience chooses what is not fixed.** Let the examiner pick a holdout run and an injection seed, so the run cannot have been rehearsed (§4).
4. **Report what happens.** A missed anomaly or a false alarm is a result. Do not re-run until it looks good.

## 2. Freeze checklist (the day before; tick only what you actually did)

| ☐ | Step | Command | Record |
|---|---|---|---|
| ☐ | Original files unchanged | `.\.venv313\Scripts\python.exe scripts\verify_checksums.py` → "ALL MATCH" | |
| ☐ | Model files intact | `.\.venv313\Scripts\python.exe scripts\write_artifact_manifest.py --check` → "verified" | |
| ☐ | Tests pass | `.\scripts\verify\run.ps1 pytest` → `RESULT exit=0` | counts: |
| ☐ | API smoke test passes (own free port; does not touch your server) | `.\scripts\verify\run.ps1 api` → `RESULT exit=0` | |
| ☐ | Dashboard builds | `.\scripts\verify\run.ps1 frontend` → `RESULT exit=0` | |
| ☐ | Note model id and threshold | dashboard **Settings**, or `Invoke-RestMethod http://127.0.0.1:8000/model/status` | |
| ☐ | Decide which evaluation report the Evaluation page shows | default `reports\evaluation_metrics.json` (notebook 03, 60 runs). For the equal-FPR comparison, start the backend after `$env:PP_API_METRICS_REPORT="reports/evaluation_equal_fpr.json"` | |
| ☐ | Back up the model folder and reports | copy `artifacts\models\autoencoder_v1` and `reports\` with the date | |

## 3. Live sequence (about 12 minutes)

| # | Show | How | Rehearsal result (cloud) | What to say |
|---|---|---|---|---|
| 1 | Startup and health | §0; dashboard **Overview** | API Online, Model Ready, id `3f2ade8e658c` | "The API loads the saved model once and checks its integrity manifest. Status turns green only after the API answered." |
| 2 | Normal sensor input | **Predictions → Paste JSON** with `reports\demo\01_normal.json` → Submit | error 0.3932 < 0.8852 → normal | "A real fault-free sample from an evaluation run. Its reconstruction error is below the calibrated threshold." |
| 3 | In-range multivariate anomaly | `02_in_range_multivariate.json` | error 1.7030 → anomaly; top contributor `xmv_7` | "Every value is inside the single-feature limits; the rule check did not flag it, the autoencoder did. It is a SYNTHETIC coordinated shift from the documented generator. On that run the autoencoder flagged 14 of 69 injected rows and the rule check 0 (`02_in_range_multivariate.context.json`), so most such rows are still missed." |
| 4 | Obvious deviation | `03_obvious_deviation.json` | error 1.0055 → anomaly; top contributor `xmeas_7` | "A large ±4 sd change; the rule check would catch this too." |
| 5 | Interpreting error and threshold | open the result row; contribution bars | — | "The score is the mean squared reconstruction error of the 52 scaled features. The threshold lets about 1 % of normal calibration rows through. It is a score, not a probability of a fault. Contributions show where reconstruction is poor, not the cause." |
| 6 | Dashboard and CSV | **Predictions → CSV upload** `reports\demo\demo_samples.csv` → Submit; **Export CSV** | 3 scored, 2 flagged | "The CSV is validated in the browser first; the API validates again." |
| 7 | History persistence | **History** (try "Anomalies only") | scored rows stored; invalid requests not stored | "Every scored sample is stored in SQLite without its raw values." |
| 8 | Held-out run | §4: **Live Monitor**, upload the examiner-chosen `holdout_run<R>_<family>_seed<S>.csv`, replay in batches | not rehearsed (deliberately unseen) | "This is a user-triggered replay of a file in batches, not a live plant stream." Afterwards run `--score` (§4) and fill §6. |
| 9 | Evaluation and PCA comparison | **Evaluation** page; `docs\EVALUATION_RESULTS.md` | — | "At the same 1 % false-alarm rate the autoencoder beats the rule check on 5 of 6 families, but PCA is as good or better on all six. Recall is low. All anomalies are synthetic." |
| 10 | Invalid input | Paste `04_missing_feature.json`, then `05_out_of_range.json` | 422 `invalid_features`; 422 `value_out_of_range` | "Bad inputs are rejected with a clear message and nothing is stored." |
| 11 | Limitations | — | — | Synthetic anomalies only; PCA is stronger; single seed; replay, not streaming; reconstruction error is not a fault probability; contributions are not causes; no authentication on prediction endpoints. |

## 4. Held-out data for the live part

1. Show the runs that may be chosen: `.\.venv313\Scripts\python.exe scripts\export_holdout_demo.py --list`. Ask the examiner for one run and an injection seed (any integer).
2. Create the files (clean run, injected copy, separate labels file; nothing is scored):
   `.\.venv313\Scripts\python.exe scripts\export_holdout_demo.py --run <R> --family coordinated_shift --seed <S>`
   A prepared fallback exists: holdout run 23, coordinated_shift, seed 42 (`reports\demo\holdout_run23*.csv`, not scored).
3. Dashboard → **Live Monitor**: upload `holdout_run<R>_<family>_seed<S>.csv`, replay with batch size about 100. The labels file is **not** uploaded.
4. After the replay, for the record sheet: `.\.venv313\Scripts\python.exe scripts\export_holdout_demo.py --run <R> --family <family> --seed <S> --score` prints injected rows flagged (recall) and false alarms on normal rows.

## 5. If something goes wrong

| Situation | Action |
|---|---|
| API shows model `not_found` or `load_failed` | Restore the backed-up `artifacts\models\autoencoder_v1` folder and restart the API |
| Dashboard shows "API unreachable" | Check terminal 1 is running and the dashboard origin is in `[api].cors_origins` (README §10) |
| Port 8000 busy | An earlier server is still running; use it, or stop it with Ctrl+C in its terminal |
| `npm run dev` fails | Use the API directly: `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -InFile reports\demo\01_normal.json` and say the dashboard could not be run |
| A family is missed entirely on the held-out run | Report it. It is a finding. Do not change the threshold |
| Tempted to re-run with a new seed | Do not. If you do, record both runs |

## 6. Record sheet (fill during the demo)

| Item | Value |
|---|---|
| Date, place, examiner | |
| model_id, threshold, threshold source | |
| Holdout run (chosen by) | |
| Injection family and seed (chosen by) | |
| Injected rows / flagged (from `--score`) | |
| False alarms on normal rows | |
| Observations (which events were caught or missed) | |
| Any deviation from this procedure | |

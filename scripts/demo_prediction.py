r"""Score ONE real fault-free sample through the running API (and, optionally, offline) and print both answers.

    python scripts\demo_prediction.py                       # sample 240 of the first EVALUATION-role run
    python scripts\demo_prediction.py --sample 800 --offline
    python scripts\demo_prediction.py --run 17 --sample 5   # only runs of the evaluation role are accepted

Needs the API to be running (see README) unless you only want the offline answer (``--offline-only``).
The sample comes from the fault-free TESTING file, read-only. Only runs of the *evaluation* role are
allowed: calibration runs set the threshold, and holdout runs are reserved for the live demonstration,
so the script refuses them. This is a normal (fault-free) sample, so a correctly working system should
usually NOT flag it (about 1 % of normal rows are flagged by design).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from production_pulse.config import load_config  # noqa: E402
from production_pulse.data import FEATURE_COLUMNS, load_raw_dataset  # noqa: E402
from production_pulse.paths import get_paths  # noqa: E402
from production_pulse.splitting import HOLDOUT_RUNS_PREVIOUSLY_SCORED, select_runs, split_runs  # noqa: E402


def pick_sample(run: int | None, sample: int):
    config = load_config(root=ROOT)
    paths = get_paths(ROOT, config=config)
    raw = load_raw_dataset("fault_free_testing", paths=paths, config=config)
    split = split_runs(raw, config.evaluation.fractions(), seed=config.random_seed, group_columns=config.split.group_columns)
    evaluation = split.runs("evaluation")
    if run is None:
        run = int(select_runs(split, "evaluation", 1, seed=config.random_seed, exclude=HOLDOUT_RUNS_PREVIOUSLY_SCORED)[0])
    elif run not in evaluation:
        role = next((r for r in split.names if run in split.runs(r)), "unknown")
        raise SystemExit(f"Run {run} belongs to the '{role}' role. Only evaluation-role runs are accepted by this script.")
    rows = raw[(raw["simulationRun"] == run) & (raw["sample"] == sample)]
    if len(rows) != 1:
        raise SystemExit(f"Run {run} has no sample {sample} (valid samples: 1..960).")
    return run, rows.iloc[0], paths


def post_predict(url: str, features: dict[str, float], sample_id: str) -> dict:
    body = json.dumps({"samples": [{"features": features, "sample_id": sample_id}]}).encode("utf-8")
    request = urllib.request.Request(f"{url.rstrip('/')}/predict", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API returned HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach the API at {url}: {exc.reason}. Start it first (see README).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=None, help="simulation run of the evaluation role (default: first selected run)")
    ap.add_argument("--sample", type=int, default=240, help="sample index 1..960 (default 240)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--offline", action="store_true", help="also score offline with the saved model and compare")
    ap.add_argument("--offline-only", action="store_true", help="do not call the API")
    args = ap.parse_args(argv)

    run, row, paths = pick_sample(args.run, args.sample)
    features = {c: float(row[c]) for c in FEATURE_COLUMNS}
    sample_id = f"fault_free_testing/run{run}/sample{args.sample}"
    print(f"Sample: {sample_id} (fault-free, evaluation role, 52 raw features)")

    api = None
    if not args.offline_only:
        api = post_predict(args.url, features, sample_id)
        result, top = api["results"][0], api["results"][0]["top_features"]
        print(f"API    : error {result['reconstruction_error']:.6f} | threshold {api['threshold']:.6f} | "
              f"{'ANOMALY' if result['is_anomaly'] else 'normal'} | model {api['model_id']}")
        print("         largest per-feature contributions (where reconstruction deviates; not a verified cause):")
        for t in top[:5]:
            print(f"           {t['feature']:<10} squared error {t['squared_error']:.4f} ({t['share']:.1%} of total)")
    if args.offline or args.offline_only:
        import pandas as pd

        from production_pulse.autoencoder import AutoencoderDetector

        det = AutoencoderDetector.load("artifacts/models/autoencoder_v1", paths=paths)
        frame = pd.DataFrame([features])[list(FEATURE_COLUMNS)]
        d = det.detect(det.scaler.transform_frame(frame), check_unseen=False)
        print(f"Offline: error {d.scores[0]:.6f} | threshold {d.threshold:.6f} | {'ANOMALY' if d.predictions[0] else 'normal'}")
        if api is not None:
            same = abs(d.scores[0] - api["results"][0]["reconstruction_error"]) <= 1e-5 * max(abs(d.scores[0]), 1e-12) \
                and bool(d.predictions[0]) == api["results"][0]["is_anomaly"]
            print("Offline and API agree:", same)
            return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

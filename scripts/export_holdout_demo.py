r"""Export ONE held-out (never fitted, never calibrated) run as CSV files for the live demonstration.

    python scripts\export_holdout_demo.py            # picks a holdout run reproducibly (seed from the config)
    python scripts\export_holdout_demo.py --run 117  # a specific HOLDOUT run (other roles are refused)

Writes (new files only; refuses to overwrite):
* reports\demo\holdout_run<R>.csv                  the 960 real fault-free rows of that run, unchanged
* reports\demo\holdout_run<R>_<family>_seed<S>.csv        the same rows with SYNTHETIC events from the project's
                                                          documented generator (default family coordinated_shift,
                                                          rate 0.05; individually in-range values)
* reports\demo\holdout_run<R>_<family>_seed<S>_labels.csv sample, label (1 = injected row), event_id

    python scripts\export_holdout_demo.py --run 117 --family correlation_break --seed 2718   # examiner's choice

Why: the course rubric asks for a demo "on previously unseen/held-out data". The holdout role exists for this.
Runs 42, 196, 241 and 304 were scored once by an early notebook and are excluded. This script deliberately does
NOT score the exported rows, so the run stays unseen until the live demo. Load the CSV in the dashboard
(Live Monitor or Predictions). The injected event is a constructed example, NOT a recorded plant fault.
The holdout rows are used only for the demonstration, never to fit, calibrate or tune anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from production_pulse import get_paths, load_config  # noqa: E402
from production_pulse.anomaly_generation import FAMILIES  # noqa: E402
from production_pulse.data import FEATURE_COLUMNS, load_raw_dataset  # noqa: E402
from production_pulse.paths import safe_output_path  # noqa: E402
from production_pulse.splitting import HOLDOUT_RUNS_PREVIOUSLY_SCORED, select_runs, split_runs  # noqa: E402

ANOMALY_RATE = 0.05
F = list(FEATURE_COLUMNS)


def choose_run(split, requested: int | None, seed: int) -> int:
    holdout = set(split.runs("holdout"))
    if requested is None:
        return int(select_runs(split, "holdout", 1, seed=seed, exclude=HOLDOUT_RUNS_PREVIOUSLY_SCORED)[0])
    if requested not in holdout:
        role = next((r for r in split.names if requested in split.runs(r)), "unknown")
        raise SystemExit(f"Run {requested} belongs to the '{role}' role; only holdout runs can be exported for the demo.")
    if requested in HOLDOUT_RUNS_PREVIOUSLY_SCORED:
        raise SystemExit(f"Run {requested} was scored by an earlier notebook selection, so it is not unseen. Pick another.")
    return requested


def score_exported(paths, out: Path, run: int, family: str, seed: int) -> int:
    """Offline check for the record sheet: hits, misses and false alarms of the saved model on the exported file."""
    from production_pulse.autoencoder import AutoencoderDetector

    feats = out / f"holdout_run{run}_{family}_seed{seed}.csv"
    labels = out / f"holdout_run{run}_{family}_seed{seed}_labels.csv"
    if not feats.exists() or not labels.exists():
        raise SystemExit(f"Export it first (python scripts\\export_holdout_demo.py --run {run} --family {family} --seed {seed}).")
    df, lab = pd.read_csv(feats), pd.read_csv(labels)
    if not df["sample"].equals(lab["sample"]):
        raise SystemExit("labels file does not match the injected file")
    det = AutoencoderDetector.load("artifacts/models/autoencoder_v1", paths=paths)
    d = det.detect_raw(df[list(det.feature_columns)], check_unseen=False)
    y, p = lab["label"].to_numpy() == 1, d.predictions
    print(f"holdout run {run}: threshold {d.threshold:.4f}; injected rows {int(y.sum())}, flagged {int((p & y).sum())} "
          f"(recall {(p & y).sum() / max(y.sum(), 1):.3f}); normal rows {int((~y).sum())}, false alarms {int((p & ~y).sum())} "
          f"({(p & ~y).sum() / max((~y).sum(), 1):.3%}). SYNTHETIC event, not a real fault.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, help="holdout run id (default: reproducible choice)")
    ap.add_argument("--family", default="coordinated_shift", choices=sorted(FAMILIES), help="anomaly family")
    ap.add_argument("--seed", type=int, default=42, help="injection seed (let the examiner choose one)")
    ap.add_argument("--list", action="store_true", help="print the holdout runs that may be chosen, then stop")
    ap.add_argument("--score", action="store_true",
                    help="AFTER the live demo: score the exported injected file offline and compare with its labels")
    a = ap.parse_args()
    paths = get_paths()
    cfg = load_config(root=paths.root)
    raw = load_raw_dataset("fault_free_testing", paths=paths, config=cfg)
    split = split_runs(raw, cfg.evaluation.fractions(), seed=cfg.random_seed, group_columns=cfg.split.group_columns)
    if a.list:
        print("holdout runs available for the demo:", sorted(set(split.runs("holdout")) - set(HOLDOUT_RUNS_PREVIOUSLY_SCORED)))
        return 0
    run = choose_run(split, a.run, cfg.random_seed)
    rows = raw[raw["simulationRun"] == run].sort_values("sample").reset_index(drop=True)
    if len(rows) != 960:
        raise SystemExit(f"Run {run} has {len(rows)} rows, expected 960.")

    out = paths.reports_dir / "demo"
    tag = f"holdout_run{run}_{a.family}_seed{a.seed}"
    names = [f"holdout_run{run}.csv", f"{tag}.csv", f"{tag}_labels.csv"]
    if a.score:
        return score_exported(paths, out, run, a.family, a.seed)
    if (out / names[1]).exists():
        print(f"{names[1]} already exists; nothing written (choose another --seed or --family).")
        return 0
    targets = [out / n for n in names]
    for t in targets[1:]:
        safe_output_path(t, paths, overwrite=False)

    from production_pulse.anomaly_generation import AnomalyGenerator, AnomalySpec
    from production_pulse.workflow import prepare_fault_free_data

    data = prepare_fault_free_data(paths, cfg)                    # train/validation/calibration/evaluation only
    gen = AnomalyGenerator().fit(data["train"])
    scaled = data.scaler.transform_frame(rows[F])
    r = gen.generate(scaled, AnomalySpec(a.family, ANOMALY_RATE), seed=a.seed, groups=rows["simulationRun"])
    clean = rows[["simulationRun", "sample", *F]].copy()
    injected = clean.copy()
    injected[F] = data.scaler.inverse_transform(r.features[F].to_numpy())
    labels = pd.DataFrame({"sample": rows["sample"].to_numpy(), "label": r.labels.to_numpy().astype(int),
                           "event_id": r.row_info["event_id"].to_numpy()})
    for target, frame in zip(targets, (clean, injected, labels)):
        if target.exists():                                        # never overwrite (the clean file may be shared)
            print("kept existing", target.relative_to(paths.root))
            continue
        frame.to_csv(target, index=False)
        print("wrote", target.relative_to(paths.root))
    print(f"holdout run {run}: 960 rows; {a.family}, seed {a.seed}: {int(labels['label'].sum())} injected rows in "
          f"{len(r.events)} event(s). Labels are in the separate file; nothing was scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""Create the JSON/CSV request files used in the live demonstration (docs/DEMO_SCRIPT.md).

    python scripts\make_demo_samples.py            # writes reports\demo\*.json (refuses to overwrite)
    python scripts\make_demo_samples.py --force    # rewrite them

Source: ONE fault-free EVALUATION-role run (the run scripts\demo_prediction.py uses). Holdout and calibration
runs are never used. Nothing in the datasets or the saved model is changed.

* 01_normal.json             a real fault-free row (sample 240), unchanged
* 02_in_range_multivariate.json
                             a row from a SYNTHETIC coordinated_shift event made by the project's documented
                             generator (seed 42, rate 0.05, default magnitude) on that run. Every value stays inside
                             the single-feature rule check's limits. Selection rule (fixed, not cherry-picked by eye):
                             the first injected row, in sample order, that the autoencoder flags and the rule check
                             does not. The script prints how many injected rows of the run there were and how many
                             each detector flagged, so the example can be put in context. If no row qualifies, no
                             file is written.
* 03_obvious_deviation.json  the normal row with xmeas_7 +4 sd and xmeas_13 -4 sd: a large, obvious deviation that
                             BOTH the rule check and the autoencoder are expected to flag
* 04_missing_feature.json    the normal row without xmv_3                -> HTTP 422 invalid_features
* 05_out_of_range.json       the normal row with xmeas_1 = 1e300         -> HTTP 422 value_out_of_range
* 06_batch.json              01, 02 and 03 in one request
* demo_samples.csv           01, 02 and 03 as CSV for the dashboard's CSV upload

All anomalies here are constructed examples, NOT recorded plant faults.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_prediction import pick_sample  # noqa: E402  (evaluation-role row picker, refuses holdout)

from production_pulse.data import FEATURE_COLUMNS  # noqa: E402
from production_pulse.paths import safe_output_path  # noqa: E402

SAMPLE_INDEX = 240
OBVIOUS = {"xmeas_7": +4.0, "xmeas_13": -4.0}
FAMILY, RATE, SEED = "coordinated_shift", 0.05, 42
F = list(FEATURE_COLUMNS)


def build(scale: dict[str, float], row, in_range_row: dict | None, in_range_id: str | None) -> dict[str, dict]:
    base = {c: float(row[c]) for c in F}
    run, samp = int(row["simulationRun"]), int(row["sample"])
    sid = f"fault_free_testing/run{run}/sample{samp}"
    obvious = dict(base)
    for f, k in OBVIOUS.items():
        obvious[f] = base[f] + k * scale[f]
    files = {
        "01_normal.json": {"sample_id": sid, "features": base},
        "03_obvious_deviation.json": {"sample_id": f"{sid}+synthetic_obvious_deviation", "features": obvious},
        "04_missing_feature.json": {"sample_id": f"{sid}-missing-xmv_3", "features": {k: v for k, v in base.items() if k != "xmv_3"}},
        "05_out_of_range.json": {"sample_id": f"{sid}-xmeas_1=1e300", "features": dict(base, xmeas_1=1e300)},
    }
    batch = [files["01_normal.json"]]
    if in_range_row is not None:
        files["02_in_range_multivariate.json"] = {"sample_id": in_range_id, "features": in_range_row}
        batch.append(files["02_in_range_multivariate.json"])
    batch.append(files["03_obvious_deviation.json"])
    files["06_batch.json"] = {"samples": batch}
    return files


def in_range_example(paths, run: int):
    """Generator-injected coordinated_shift row on `run` that the AE flags and the rule check does not."""
    import numpy as np

    from production_pulse import load_config
    from production_pulse.anomaly_generation import AnomalyGenerator, AnomalySpec
    from production_pulse.autoencoder import AutoencoderDetector
    from production_pulse.baselines import SingleFeatureThresholdDetector
    from production_pulse.workflow import prepare_fault_free_data

    cfg = load_config(root=paths.root)
    data = prepare_fault_free_data(paths, cfg)                     # holdout stays locked
    meta = data.metadata["evaluation"]
    keep = (meta["simulationRun"] == run).to_numpy()
    x, m = data["evaluation"][keep], meta[keep]
    gen = AnomalyGenerator().fit(data["train"])
    r = gen.generate(x, AnomalySpec(FAMILY, RATE), seed=SEED, groups=m["simulationRun"])
    ae = AutoencoderDetector.load("artifacts/models/autoencoder_v1", paths=paths)
    rule = SingleFeatureThresholdDetector().fit(data["train"])       # the original fixed-limit rule check
    a, rc = ae.detect(r.features, check_unseen=False), rule.detect(r.features, check_unseen=False)
    lab = r.labels.to_numpy().astype(bool)
    stats = {"run": run, "family": FAMILY, "seed": SEED, "injected_rows": int(lab.sum()),
             "ae_flagged": int((a.predictions & lab).sum()), "rule_flagged": int((rc.predictions & lab).sum()),
             "ae_threshold": float(a.threshold)}
    idx = np.flatnonzero(lab & a.predictions & ~rc.predictions)
    if idx.size == 0:
        return None, None, stats
    i = int(idx[0])
    raw = data.scaler.inverse_transform(r.features.iloc[[i]][F].to_numpy())[0]
    stats.update({"chosen_sample": int(m["sample"].iloc[i]), "ae_score": float(a.scores[i]),
                  "qualifying_rows": int(idx.size)})
    sid = f"fault_free_testing/run{run}/sample{stats['chosen_sample']}+synthetic_{FAMILY}_seed{SEED}"
    return {c: float(v) for c, v in zip(F, raw)}, sid, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite existing demo files")
    a = ap.parse_args()
    run, row, paths = pick_sample(None, SAMPLE_INDEX)
    out_dir = paths.reports_dir / "demo"
    if out_dir.exists() and any(out_dir.glob("0*.json")) and not a.force:
        print(f"Demo files already exist in {out_dir.relative_to(paths.root)}; nothing written. Use --force to recreate them.")
        return 0
    meta = json.loads((paths.root / "artifacts/models/autoencoder_v1/metadata.json").read_text(encoding="utf-8"))
    scale = dict(zip(meta["scaler"]["feature_columns"], meta["scaler"]["scale"]))
    example, example_id, stats = in_range_example(paths, run)
    print("in-range example context:", json.dumps(stats))
    files = build(scale, row, example, example_id)
    for name, body in files.items():
        target = safe_output_path(out_dir / name, paths, overwrite=a.force)
        target.write_text(json.dumps(body, indent=2), encoding="utf-8")
        print("wrote", target.relative_to(paths.root))
    (safe_output_path(out_dir / "02_in_range_multivariate.context.json", paths, overwrite=a.force)
     .write_text(json.dumps(stats, indent=2), encoding="utf-8"))
    target = safe_output_path(out_dir / "demo_samples.csv", paths, overwrite=a.force)
    with target.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", *F])
        for s in files["06_batch.json"]["samples"]:
            w.writerow([s["sample_id"], *[repr(s["features"][c]) for c in F]])
    print("wrote", target.relative_to(paths.root))
    try:
        import pandas as pd

        from production_pulse.autoencoder import AutoencoderDetector

        det = AutoencoderDetector.load("artifacts/models/autoencoder_v1", paths=paths)
        for s in files["06_batch.json"]["samples"]:
            d = det.detect_raw(pd.DataFrame([s["features"]])[F], check_unseen=False)
            print(f"offline score {s['sample_id']}: {d.scores[0]:.4f} vs threshold {d.threshold:.4f} -> "
                  f"{'ANOMALY' if d.predictions[0] else 'normal'}")
    except Exception as exc:  # noqa: BLE001 - informational only
        print("offline scoring skipped:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

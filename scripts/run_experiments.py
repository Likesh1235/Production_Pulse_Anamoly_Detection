r"""Controlled robustness and ablation experiments (fault-free TEP data, SYNTHETIC anomalies).

    python scripts\run_experiments.py robustness      # saved model, no retraining
    python scripts\run_experiments.py ablation        # trains SEPARATE experimental models
    python scripts\run_experiments.py ablation --only bn4 --tag mytag

Rules this script enforces:
* The saved model in artifacts/models/autoencoder_v1 is only READ. Experimental models are saved under
  artifacts/experiments/<tag>/ and never replace it.
* Roles: models fit on train, early stopping on validation, thresholds from calibration rows ONLY
  (target FPR from the config), results reported on the evaluation role. The holdout is never loaded, and
  nothing is selected or tuned on evaluation results (every variant is reported, none is picked).
* Output files are new (reports/experiments/<kind>_<tag>.csv/.json); existing files are never overwritten.
* Seeds and every setting are written next to the results.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from production_pulse import get_paths, load_config, setup_logging  # noqa: E402
from production_pulse.anomaly_generation import FAMILIES, AnomalyGenerator, AnomalySpec  # noqa: E402
from production_pulse.autoencoder import AutoencoderDetector, verify_manifest  # noqa: E402
from production_pulse.baselines import PCAReconstructionDetector, SingleFeatureThresholdDetector  # noqa: E402
from production_pulse.config import AutoencoderSettings  # noqa: E402
from production_pulse.evaluation import compare_detectors  # noqa: E402
from production_pulse.paths import safe_output_path  # noqa: E402
from production_pulse.splitting import HOLDOUT_RUNS_PREVIOUSLY_SCORED  # noqa: E402
from production_pulse.workflow import prepare_fault_free_data  # noqa: E402

SEED = 42
ANOMALY_RATE = 0.05
NOISE_LEVELS = (0.0, 0.1, 0.25, 0.5)      # std of added Gaussian noise, in TRAINING standard deviations
MODEL_DIR = "artifacts/models/autoencoder_v1"

# name -> (settings overrides, training seed). Seeds 43/44 re-train the SAME configuration as the saved model
# to measure seed-to-seed variation, which is the yardstick for reading the other rows.
ABLATION_VARIANTS = {
    "base_seed43": ({}, 43),
    "base_seed44": ({}, 44),
    "bottleneck4": ({"bottleneck": 4}, SEED),
    "bottleneck16": ({"bottleneck": 16}, SEED),
    "denoising_0.2": ({"noise_std": 0.2}, SEED),
}


def _setup():
    paths = get_paths()
    config = load_config(root=paths.root)
    setup_logging(config=config, paths=paths)
    data = prepare_fault_free_data(paths, config)
    assert "holdout" not in data.features, "holdout must stay locked"
    eval_x, eval_meta = data["evaluation"], data.metadata["evaluation"]
    eval_runs = set(eval_meta["simulationRun"].unique().tolist())
    assert not eval_runs & set(data.test_split.runs("calibration")), "calibration/evaluation overlap"
    assert not eval_runs & set(data.test_split.runs("holdout")), "holdout/evaluation overlap"
    assert not eval_runs & set(HOLDOUT_RUNS_PREVIOUSLY_SCORED)
    return paths, config, data, eval_x, eval_meta


def _injected(data, eval_x, eval_meta):
    gen = AnomalyGenerator().fit(data["train"])
    out = {}
    for fam in FAMILIES:
        r = gen.generate(eval_x, AnomalySpec(fam, ANOMALY_RATE), seed=SEED, groups=eval_meta["simulationRun"])
        out[fam] = (r.features, r.labels.to_numpy().astype(bool))
    return out


def _noisy(df: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    if sigma == 0:
        return df
    rng = np.random.default_rng(seed)
    return df + rng.normal(0.0, sigma, size=df.shape)


def _write(rows, meta, kind, tag, paths):
    table = pd.DataFrame(rows)
    csv = safe_output_path(paths.reports_dir / "experiments" / f"{kind}_{tag}.csv", paths, overwrite=False)
    js = safe_output_path(paths.reports_dir / "experiments" / f"{kind}_{tag}.json", paths, overwrite=False)
    table.to_csv(csv, index=False)
    js.write_text(json.dumps({"settings": meta, "rows": rows}, indent=2, default=float), encoding="utf-8")
    print("saved", csv.relative_to(paths.root), "and", js.relative_to(paths.root))
    return table


def _versions():
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()}
    try:
        import tensorflow as tf

        v["tensorflow"] = tf.__version__
    except ImportError:
        pass
    return v


def _metric_rows(common: dict, labels, dets: dict) -> list[dict]:
    t = compare_detectors(labels, dets)
    rows = []
    for name, r in t.iterrows():
        rows.append({**common, "detector": name, "recall": r["recall"], "precision": r["precision"], "f1": r["f1"],
                     "false_positive_rate": r["false_positive_rate"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"]})
    return rows


def robustness(tag: str) -> None:
    paths, config, data, eval_x, eval_meta = _setup()
    fpr = config.evaluation.target_fpr
    ae = AutoencoderDetector.load(MODEL_DIR, paths=paths)       # read only
    manifest_before = verify_manifest(paths.root / MODEL_DIR)
    train, cal = data["train"], data["calibration"]
    pca = PCAReconstructionDetector(n_components=0.90).fit(train)
    rule = SingleFeatureThresholdDetector().fit(train)
    inj = _injected(data, eval_x, eval_meta)
    rows = []
    t0 = time.time()
    for sigma in NOISE_LEVELS:
        seed = SEED + int(round(sigma * 1000))
        for policy in ("fixed_clean_threshold", "recalibrated_on_noisy_calibration"):
            # Fresh detector copies so recalibration never touches the saved model object used elsewhere.
            dets = {"rule_check_equal_fpr": rule, "pca": pca, "autoencoder": ae}
            if policy == "fixed_clean_threshold":
                for d in dets.values():
                    d.calibrate(cal, target_fpr=fpr)
            else:
                noisy_cal = _noisy(cal, sigma, seed + 7)
                for d in dets.values():
                    d.calibrate(noisy_cal, target_fpr=fpr)
            thresholds = {n: float(d.threshold_) for n, d in dets.items()}
            noisy_clean = _noisy(eval_x, sigma, seed + 1)
            clean = {n: d.detect(noisy_clean) for n, d in dets.items()}
            for n, det in clean.items():
                rows.append({"sigma": sigma, "policy": policy, "family": "(clean rows only)", "detector": n,
                             "false_positive_rate": float(det.predictions.mean()), "threshold": thresholds[n]})
            for fam, (feat, labels) in inj.items():
                nf = _noisy(feat, sigma, seed + 2)
                res = {n: d.detect(nf) for n, d in dets.items()}
                rows += _metric_rows({"sigma": sigma, "policy": policy, "family": fam}, labels, res)
            print(f"sigma={sigma} policy={policy} done ({time.time() - t0:.0f}s)", flush=True)
    # restore the saved threshold object state check (calibrate() above changed the in-memory object only)
    meta = {"experiment": "input-noise robustness (saved model, no retraining)", "seed": SEED, "anomaly_rate": ANOMALY_RATE,
            "target_fpr": fpr, "noise_levels_in_training_std": list(NOISE_LEVELS), "n_eval_runs": int(eval_meta["simulationRun"].nunique()),
            "n_eval_rows": int(len(eval_x)), "roles": "thresholds from calibration rows only; results on evaluation role; holdout locked",
            "anomalies": "SYNTHETIC", "model_dir": MODEL_DIR, "manifest_status": manifest_before,
            "model_files_modified": False, "versions": _versions(),
            "note": "noise is added to scaled features of BOTH normal and injected evaluation rows; noise-free sigma=0 re-calibrates "
                    "thresholds on the same calibration rows, so it reproduces the saved threshold."}
    _write(rows, meta, "robustness", tag, paths)
    assert verify_manifest(paths.root / MODEL_DIR) == manifest_before, "saved model files changed"


def ablation(tag: str, only: list[str] | None) -> None:
    paths, config, data, eval_x, eval_meta = _setup()
    fpr = config.evaluation.target_fpr
    train, val, cal = data["train"], data["validation"], data["calibration"]
    inj = _injected(data, eval_x, eval_meta)
    base = config.autoencoder
    pca = PCAReconstructionDetector(n_components=0.90).fit(train)
    pca.calibrate(cal, target_fpr=fpr)
    saved = AutoencoderDetector.load(MODEL_DIR, paths=paths)
    detectors = {"saved_autoencoder_v1": saved, "pca": pca}
    training = {}
    for name, (over, seed) in ABLATION_VARIANTS.items():
        if only and name not in only:
            continue
        settings = AutoencoderSettings(**{**base.__dict__, **over})
        t0 = time.time()
        det = AutoencoderDetector(settings=settings, seed=seed, scaler=data.scaler, verbose=0)
        det.fit(train, val)
        det.calibrate(cal, target_fpr=fpr)
        out_dir = f"artifacts/experiments/{tag}/{name}"
        det.save(out_dir, paths, overwrite=False)
        detectors[name] = det
        training[name] = {"settings_overrides": over, "seed": seed, "seconds": round(time.time() - t0, 1),
                          **{k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in det.training_summary_.items()}}
        print(f"trained {name} in {time.time() - t0:.0f}s: {training[name]}", flush=True)
    rows = []
    for fam, (feat, labels) in inj.items():
        res = {n: d.detect(feat) for n, d in detectors.items()}
        rows += _metric_rows({"family": fam}, labels, res)
    clean = {n: float(d.detect(eval_x).predictions.mean()) for n, d in detectors.items()}
    for n, v in clean.items():
        rows.append({"family": "(clean rows only)", "detector": n, "false_positive_rate": v})
    meta = {"experiment": "model ablation (separate experimental models; saved model untouched)", "seed": SEED,
            "anomaly_rate": ANOMALY_RATE, "target_fpr": fpr, "base_settings": base.__dict__, "variants": training,
            "n_eval_runs": int(eval_meta["simulationRun"].nunique()), "n_eval_rows": int(len(eval_x)),
            "roles": "fit=train, early stopping=validation, threshold=calibration only; results on evaluation role; holdout locked; "
                     "no variant was selected on these results",
            "anomalies": "SYNTHETIC", "experiment_models_dir": f"artifacts/experiments/{tag}", "versions": _versions()}
    _write(rows, meta, "ablation", tag, paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["robustness", "ablation"])
    ap.add_argument("--tag", default="20261021", help="suffix for the NEW output files / experiment folder")
    ap.add_argument("--only", nargs="*", help="ablation: train only these variants")
    a = ap.parse_args()
    robustness(a.tag) if a.kind == "robustness" else ablation(a.tag, a.only)


if __name__ == "__main__":
    main()

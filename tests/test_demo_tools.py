"""Demo helpers: scripts/make_demo_samples.py and scripts/export_holdout_demo.py (no real data, no model)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from conftest import make_tep_frame
from production_pulse.data import FEATURE_COLUMNS
from production_pulse.splitting import HOLDOUT_RUNS_PREVIOUSLY_SCORED, split_runs

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def frame():
    return make_tep_frame(n_runs=40, n_samples=5, seed=3)


def test_demo_samples_change_only_the_documented_features(frame):
    mod = _load("make_demo_samples")
    row = frame.iloc[0]
    scale = {c: 2.0 for c in FEATURE_COLUMNS}
    in_range = {c: float(row[c]) + 0.1 for c in FEATURE_COLUMNS}
    files = mod.build(scale, row, in_range, "x+synthetic_coordinated_shift_seed42")
    normal, obvious = files["01_normal.json"]["features"], files["03_obvious_deviation.json"]["features"]
    assert {c for c in FEATURE_COLUMNS if obvious[c] != normal[c]} == {"xmeas_7", "xmeas_13"}
    assert obvious["xmeas_7"] == pytest.approx(normal["xmeas_7"] + 8.0)
    assert obvious["xmeas_13"] == pytest.approx(normal["xmeas_13"] - 8.0)
    assert set(files["04_missing_feature.json"]["features"]) == set(FEATURE_COLUMNS) - {"xmv_3"}
    assert files["05_out_of_range.json"]["features"]["xmeas_1"] == 1e300
    assert [s["features"] for s in files["06_batch.json"]["samples"]] == [normal, in_range, obvious]
    assert "synthetic" in files["02_in_range_multivariate.json"]["sample_id"]
    no_example = mod.build(scale, row, None, None)                 # no qualifying row -> no 02 file
    assert "02_in_range_multivariate.json" not in no_example
    assert len(no_example["06_batch.json"]["samples"]) == 2


def test_holdout_export_only_accepts_unseen_holdout_runs(frame):
    mod = _load("export_holdout_demo")
    split = split_runs(frame, {"calibration": 0.2, "evaluation": 0.7, "holdout": 0.1}, seed=42,
                       group_columns=["simulationRun"])
    holdout = sorted(split.runs("holdout"))
    chosen = mod.choose_run(split, None, 42)
    assert chosen in holdout and chosen not in HOLDOUT_RUNS_PREVIOUSLY_SCORED
    assert mod.choose_run(split, None, 42) == chosen                       # reproducible
    assert mod.choose_run(split, holdout[0], 42) == holdout[0]
    for role in ("evaluation", "calibration"):
        with pytest.raises(SystemExit, match=role):
            mod.choose_run(split, sorted(split.runs(role))[0], 42)

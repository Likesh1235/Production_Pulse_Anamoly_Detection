"""OPT-IN (PP_RUN_REALDATA=1): the demo script only accepts evaluation-role runs of the real testing file.

Loads the 47 MB fault-free testing file read-only; never touches the large faulty files.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("PP_RUN_REALDATA") != "1", reason="set PP_RUN_REALDATA=1 to run")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_prediction.py"


@pytest.fixture(scope="module")
def demo():
    pytest.importorskip("pyreadr")
    spec = importlib.util.spec_from_file_location("demo_prediction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_prediction"] = module
    spec.loader.exec_module(module)
    return module


def test_default_pick_is_an_evaluation_run_with_a_normal_row(demo):
    from production_pulse.splitting import HOLDOUT_RUNS_PREVIOUSLY_SCORED

    run, row, _ = demo.pick_sample(None, 240)
    assert run not in HOLDOUT_RUNS_PREVIOUSLY_SCORED
    assert int(row["simulationRun"]) == run and int(row["sample"]) == 240 and int(row["faultNumber"]) == 0


@pytest.mark.parametrize("run, role", [(42, "holdout"), (41, "calibration")])
def test_holdout_and_calibration_runs_are_refused(demo, run, role):
    with pytest.raises(SystemExit, match=role):
        demo.pick_sample(run, 10)


def test_unknown_sample_is_refused(demo):
    run, _, _ = demo.pick_sample(None, 1)
    with pytest.raises(SystemExit, match="no sample"):
        demo.pick_sample(run, 5000)

"""Run-selection rule: evaluation runs come only from the evaluation role, and the
holdout runs once scored by an early notebook-02 draw stay out of demo selections.

Uses synthetic run ids only (1..500, as in the TEP testing file); no data is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from production_pulse.config import default_config
from production_pulse.splitting import (
    HOLDOUT_RUNS_PREVIOUSLY_SCORED,
    SplitError,
    select_runs,
    split_runs,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "02_synthetic_anomaly_validation.ipynb"


@pytest.fixture(scope="module")
def cfg():
    return default_config()


@pytest.fixture(scope="module")
def test_split(cfg):
    frame = pd.DataFrame({"simulationRun": np.repeat(np.arange(1, 501), 2)})
    return split_runs(frame, cfg.evaluation.fractions(), seed=cfg.random_seed, group_columns=cfg.split.group_columns)


def test_selection_comes_only_from_the_evaluation_role(test_split, cfg):
    for seed in range(25):
        runs = set(select_runs(test_split, "evaluation", 40, seed=seed).tolist())
        assert len(runs) == 40
        assert runs <= set(test_split.runs("evaluation"))
        assert not runs & set(test_split.runs("calibration"))
        assert not runs & set(test_split.runs("holdout"))


def test_selection_is_sorted_and_reproducible(test_split):
    a = select_runs(test_split, "evaluation", 40, seed=42)
    assert np.array_equal(a, select_runs(test_split, "evaluation", 40, seed=42))
    assert np.array_equal(a, np.sort(a))
    assert not np.array_equal(a, select_runs(test_split, "evaluation", 40, seed=43))


def test_exclude_removes_runs_and_empty_pool_is_an_error(test_split):
    pool = test_split.runs("evaluation")
    banned = pool[:300]
    assert not set(select_runs(test_split, "evaluation", 40, seed=1, exclude=banned).tolist()) & set(banned)
    with pytest.raises(SplitError, match="only"):
        select_runs(test_split, "evaluation", 40, seed=1, exclude=pool[:330])


@pytest.mark.parametrize("kwargs", [{"role": "nope", "n": 5, "seed": 1}, {"role": "evaluation", "n": 0, "seed": 1},
                                    {"role": "evaluation", "n": True, "seed": 1}, {"role": "evaluation", "n": 5, "seed": -1},
                                    {"role": "evaluation", "n": 351, "seed": 1}])
def test_invalid_requests_raise(test_split, kwargs):
    with pytest.raises(SplitError):
        select_runs(test_split, **kwargs)


def test_previously_scored_holdout_runs_match_the_real_split(test_split, cfg):
    """Re-derive the documented runs: the OLD notebook-02 draw (all 500 runs, seed 42, 40 runs)
    hit exactly these holdout runs, and none of them can ever be an evaluation-role selection."""
    assert cfg.random_seed == 42
    old_draw = set(np.random.default_rng(42).choice(np.arange(1, 501), 40, replace=False).tolist())
    in_holdout = sorted(old_draw & set(test_split.runs("holdout")))
    assert tuple(in_holdout) == HOLDOUT_RUNS_PREVIOUSLY_SCORED
    assert not set(HOLDOUT_RUNS_PREVIOUSLY_SCORED) & set(test_split.runs("evaluation"))
    assert set(HOLDOUT_RUNS_PREVIOUSLY_SCORED) <= set(test_split.runs("holdout"))


def test_demo_pool_excludes_previously_scored_runs(test_split):
    demo = select_runs(test_split, "holdout", 10, seed=7, exclude=HOLDOUT_RUNS_PREVIOUSLY_SCORED)
    assert set(demo.tolist()) <= set(test_split.runs("holdout"))
    assert not set(demo.tolist()) & set(HOLDOUT_RUNS_PREVIOUSLY_SCORED)
    # the 4 runs really are in the holdout, so excluding them shrinks the pool from 50 to 46
    with pytest.raises(SplitError):
        select_runs(test_split, "holdout", 47, seed=7, exclude=HOLDOUT_RUNS_PREVIOUSLY_SCORED)


def test_notebook_02_uses_the_role_aware_selection():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'select_runs(test_split, "evaluation"' in code
    assert "config.evaluation.fractions()" in code
    assert 'test_raw["simulationRun"].unique()' not in code, "old all-runs draw is back"
    assert "include_holdout" not in code

"""Tests for run-level, reproducible, non-overlapping splits."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from production_pulse.config import default_config
from production_pulse.paths import get_paths
from production_pulse.splitting import (
    RunSplit,
    SplitError,
    apply_split,
    check_frames_no_overlap,
    check_no_overlap,
    split_runs,
)

from conftest import make_tep_frame

FR = {"train": 0.8, "validation": 0.2}


@pytest.fixture
def df():
    return make_tep_frame(n_runs=50, n_samples=5)


def test_counts_and_full_coverage(df):
    split = split_runs(df, FR, seed=42)
    assert len(split.runs("train")) == 40 and len(split.runs("validation")) == 10
    assert sorted(split.runs("train") + split.runs("validation")) == list(range(1, 51))


def test_reproducible_for_same_seed(df):
    assert split_runs(df, FR, seed=7) == split_runs(df, FR, seed=7)


def test_different_seed_changes_assignment(df):
    assert split_runs(df, FR, seed=1).runs("validation") != split_runs(df, FR, seed=2).runs("validation")


def test_independent_of_row_order(df):
    shuffled = df.sample(frac=1.0, random_state=0)
    assert split_runs(df, FR, seed=3) == split_runs(shuffled, FR, seed=3)


def test_apply_split_keeps_whole_runs_and_no_overlap(df):
    parts = apply_split(df, split_runs(df, FR, seed=42))
    assert sum(len(p) for p in parts.values()) == len(df)
    check_frames_no_overlap(parts)
    tr, va = set(parts["train"]["simulationRun"]), set(parts["validation"]["simulationRun"])
    assert not tr & va
    assert (parts["train"].groupby("simulationRun").size() == 5).all()  # runs not cut in pieces


def test_three_way_split_never_empty():
    small = make_tep_frame(n_runs=3, n_samples=2)
    split = split_runs(small, {"a": 0.98, "b": 0.01, "c": 0.01}, seed=0)
    assert all(len(split.runs(n)) == 1 for n in "abc")


def test_too_few_runs():
    with pytest.raises(SplitError, match="at least"):
        split_runs(make_tep_frame(n_runs=1, n_samples=3), FR, seed=0)


@pytest.mark.parametrize(
    "fractions",
    [{}, {"train": 0.7, "validation": 0.2}, {"train": 1.2, "validation": -0.2}, {"train": "0.8", "validation": 0.2}, [0.8, 0.2]],
)
def test_invalid_fractions(df, fractions):
    with pytest.raises(SplitError):
        split_runs(df, fractions, seed=0)


@pytest.mark.parametrize("seed", [-1, 1.5, "42", True])
def test_invalid_seed(df, seed):
    with pytest.raises(SplitError):
        split_runs(df, FR, seed=seed)


def test_bad_group_column_values(df):
    with pytest.raises(SplitError, match="missing"):
        split_runs(df.drop(columns="simulationRun"), FR, seed=0)
    df.loc[0, "simulationRun"] = 1.5
    with pytest.raises(SplitError, match="whole numbers"):
        split_runs(df, FR, seed=0)


def test_overlap_detected():
    with pytest.raises(SplitError, match="both"):
        check_no_overlap({"train": [(1,), (2,)], "validation": [(2,), (3,)]})
    with pytest.raises(SplitError):
        RunSplit(group_columns=("simulationRun",), assignments={"a": ((1,),), "b": ((1,),)}, seed=0, fractions={"a": 0.5, "b": 0.5})
    a = make_tep_frame(n_runs=3)
    with pytest.raises(SplitError):
        check_frames_no_overlap({"train": a, "validation": a.iloc[:5]})


def test_split_from_other_data_rejected(df):
    split = split_runs(df, FR, seed=0)
    with pytest.raises(SplitError, match="not in the data"):
        apply_split(df[df["simulationRun"] <= 25], split)
    more = pd.concat([df, make_tep_frame(n_runs=51, n_samples=5).query("simulationRun == 51")])
    with pytest.raises(SplitError, match="not in the split"):
        apply_split(more, split)


def test_multi_column_groups():
    a, b = make_tep_frame(n_runs=4, fault=1), make_tep_frame(n_runs=4, fault=2)
    both = pd.concat([a, b], ignore_index=True)
    split = split_runs(both, {"x": 0.5, "y": 0.5}, seed=5, group_columns=("faultNumber", "simulationRun"))
    assert sum(len(split.keys(n)) for n in split.names) == 8
    parts = apply_split(both, split)
    check_frames_no_overlap(parts, ("faultNumber", "simulationRun"))
    with pytest.raises(SplitError):
        split.runs("x")


def test_json_roundtrip_and_safe_save(fake_root, df):
    split = split_runs(df, FR, seed=11)
    assert RunSplit.from_dict(json.loads(json.dumps(split.to_dict()))) == split
    paths = get_paths(fake_root, config=default_config())
    saved = split.save_json("data/processed/split.json", paths)
    assert RunSplit.load_json(saved) == split


def test_config_split_defaults_work(df):
    cfg = default_config()
    split = split_runs(df, cfg.split.fractions(), seed=cfg.random_seed, group_columns=cfg.split.group_columns)
    assert split.names == ("train", "validation")

"""Tests for the equal-operating-point comparison helpers (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from production_pulse.comparison import (
    cluster_bootstrap,
    paired_difference,
    point_estimates,
    prevalence,
    rate_by_sample_band,
)
from production_pulse.evaluation import MetricError


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n_runs, per_run = 40, 100
    groups = np.repeat(np.arange(n_runs), per_run)
    labels = rng.random(groups.size) < 0.05
    good = rng.normal(size=groups.size) + 3.0 * labels          # separates the classes
    random = rng.normal(size=groups.size)                        # does not
    return groups, labels, good, random


def test_prevalence_and_point_estimates_known_values():
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([9, 1, 8, 0, 0, 0, 0, 0, 0, 0.5])
    assert prevalence(labels) == pytest.approx(0.2)
    m = point_estimates(labels, scores, threshold=0.7)           # flags rows 0, 1 (positives) and 2 (a negative)
    assert m["recall"] == pytest.approx(1.0) and m["fpr"] == pytest.approx(1 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(15 / 16)                # 15 of the 16 positive/negative pairs are ranked correctly
    with pytest.raises(MetricError):
        prevalence(np.array([]))
    with pytest.raises(MetricError):
        point_estimates(labels, scores[:5], 0.5)


def test_undefined_rates_are_nan_not_zero():
    m = point_estimates(np.array([1, 0, 0]), np.array([0.0, 0.0, 0.0]), threshold=1.0, auc=False)   # nothing flagged
    assert m["recall"] == 0.0 and np.isnan(m["precision"]) and np.isnan(m["f1"])


def test_bootstrap_is_deterministic_and_brackets_estimate(data):
    groups, labels, good, random = data
    det = {"good": (good, 2.0), "random": (random, 2.0)}
    s1, r1 = cluster_bootstrap(groups, labels, det, n_boot=60, seed=3)
    s2, _ = cluster_bootstrap(groups, labels, det, n_boot=60, seed=3)
    assert s1.equals(s2)
    for (name, metric), row in s1.iterrows():
        assert row["ci_low"] <= row["estimate"] <= row["ci_high"], (name, metric, dict(row))
        assert 30 <= row["n_boot_valid"] <= 60          # precision/F1 are undefined (NaN) in resamples with no alarms
        if metric in ("recall", "fpr", "roc_auc", "pr_auc"):
            assert row["n_boot_valid"] == 60
    assert s1.loc[("good", "roc_auc"), "estimate"] > s1.loc[("random", "roc_auc"), "estimate"] + 0.2
    assert s1.loc[("random", "pr_auc"), "ci_low"] < 0.2          # random PR-AUC stays near the prevalence
    assert len(r1["good"]) == 60 and set(r1["good"].columns) >= {"recall", "fpr", "roc_auc", "pr_auc"}


def test_bootstrap_resamples_whole_runs():
    groups = np.repeat([0, 1], 50)
    labels = np.r_[np.ones(50, bool), np.zeros(50, bool)]        # run 0 all positive, run 1 all negative
    scores = labels.astype(float)
    summary, reps = cluster_bootstrap(groups, labels, {"d": (scores, 0.5)}, n_boot=100, seed=1, auc=False)
    recalls = reps["d"]["recall"].dropna()
    assert set(np.round(recalls, 6)) <= {1.0}                    # any valid resample keeps positives intact
    assert reps["d"]["recall"].isna().sum() > 0                  # resamples without positives are NaN, not dropped silently


def test_paired_difference(data):
    groups, labels, good, random = data
    _, reps = cluster_bootstrap(groups, labels, {"good": (good, 2.0), "random": (random, 2.0)}, n_boot=50, seed=2, auc=False)
    d = paired_difference(reps, "good", "random", "recall")
    assert d["ci_low"] <= d["mean"] <= d["ci_high"] and d["mean"] > 0.3


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"n_boot": True}, {"ci": 1.0}])
def test_bootstrap_validation(data, kwargs):
    groups, labels, good, _ = data
    with pytest.raises(MetricError):
        cluster_bootstrap(groups, labels, {"d": (good, 1.0)}, **kwargs)


def test_bootstrap_input_errors(data):
    groups, labels, good, _ = data
    with pytest.raises(MetricError):
        cluster_bootstrap(groups, labels, {})
    with pytest.raises(MetricError):
        cluster_bootstrap(groups[:-1], labels, {"d": (good, 1.0)})
    with pytest.raises(MetricError):
        cluster_bootstrap(groups, labels, {"d": (good[:-1], 1.0)})
    with pytest.raises(MetricError):
        cluster_bootstrap(np.zeros_like(groups), labels, {"d": (good, 1.0)})      # a single run


def test_rate_by_sample_band():
    sample = np.tile(np.arange(1, 961), 2)
    flags = (sample > 500)                                       # alarms only after sample 500
    table = rate_by_sample_band(flags, sample, [1, 251, 501, 751, 960])
    assert list(table.index) == ["1-250", "251-500", "501-750", "751-960"]
    assert table["rate"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert table["n_rows"].sum() == sample.size                  # the last band includes sample 960
    for bad in ([1], [5, 5, 9], [9, 5]):
        with pytest.raises(MetricError):
            rate_by_sample_band(flags, sample, bad)
    with pytest.raises(MetricError):
        rate_by_sample_band(flags[:-1], sample, [1, 960])

"""Tests for metrics, edge cases and detector comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from production_pulse.evaluation import (
    MetricError,
    average_precision,
    binary_metrics,
    compare_detectors,
    confusion_matrix,
    precision_recall_curve,
    recall_by_group,
    roc_auc,
    threshold_for_fpr,
)

Y = np.array([1, 1, 0, 0, 0, 1, 0, 0])
P = np.array([1, 0, 1, 0, 0, 1, 0, 0])


def test_confusion_matrix_counts():
    cm = confusion_matrix(Y, P)
    assert (cm.tp, cm.fp, cm.tn, cm.fn) == (2, 1, 4, 1)
    np.testing.assert_array_equal(cm.as_array(), [[4, 1], [1, 2]])
    assert cm.as_frame().loc["actual_anomaly", "predicted_anomaly"] == 2


def test_basic_metrics():
    m = binary_metrics(Y, P)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.false_positive_rate == pytest.approx(1 / 5)
    assert m.accuracy == pytest.approx(6 / 8)
    assert set(m.undefined) == {"pr_auc", "roc_auc"}  # no scores given


def test_no_predicted_positives():
    m = binary_metrics(Y, np.zeros_like(Y))
    assert math.isnan(m.precision) and m.undefined["precision"] == "no predicted positives"
    assert m.recall == 0.0 and m.f1 == 0.0
    assert m.to_dict()["precision"] is None


def test_no_true_positives_in_labels():
    y = np.zeros(6, dtype=int)
    m = binary_metrics(y, np.array([0, 1, 0, 0, 0, 0]), scores=np.arange(6.0))
    assert math.isnan(m.recall) and m.undefined["recall"] == "no actual positives"
    assert m.precision == 0.0 and m.f1 == 0.0
    assert m.undefined["pr_auc"] == "no actual positives" and m.undefined["roc_auc"] == "no actual positives"


def test_all_negative_nothing_predicted():
    m = binary_metrics(np.zeros(4), np.zeros(4))
    assert math.isnan(m.f1) and "f1" in m.undefined
    assert m.false_positive_rate == 0.0


def test_no_actual_negatives():
    m = binary_metrics(np.ones(4), np.array([1, 1, 0, 1]), scores=[0.9, 0.8, 0.1, 0.7])
    assert math.isnan(m.false_positive_rate) and m.undefined["false_positive_rate"] == "no actual negatives"
    assert m.pr_auc == pytest.approx(1.0)                       # defined (all positives)
    assert m.undefined["roc_auc"] == "no actual negatives"


def test_average_precision_known_values():
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx(1.0)
    # all scores tied -> AP equals prevalence
    assert average_precision([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.25)
    with pytest.raises(MetricError, match="no actual positives"):
        average_precision([0, 0], [0.1, 0.2])


def test_roc_auc_known_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(MetricError, match="no actual negatives"):
        roc_auc([1, 1], [0.1, 0.2])


def test_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)                    # includes ties
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    p = s > 0.5
    assert binary_metrics(y, p).f1 == pytest.approx(sk.f1_score(y, p))


def test_pr_curve_monotone_recall():
    precision, recall, thr = precision_recall_curve([1, 0, 1, 1, 0], [0.9, 0.7, 0.7, 0.2, 0.1])
    assert np.all(np.diff(recall) >= 0) and recall[-1] == 1.0
    assert np.all(np.diff(thr) < 0)


@pytest.mark.parametrize(
    "y, p",
    [
        ([], []),
        ([1, 0, 2], [1, 0, 1]),
        ([1, 0], [1, 0, 1]),
        ([[1, 0]], [[1, 0]]),
        (["a", "b"], [1, 0]),
        ([1, np.nan], [1, 0]),
    ],
)
def test_invalid_label_inputs(y, p):
    with pytest.raises(MetricError):
        binary_metrics(y, p)


def test_invalid_scores():
    with pytest.raises(MetricError):
        binary_metrics([1, 0], [1, 0], scores=[0.1, np.inf])
    with pytest.raises(MetricError):
        binary_metrics([1, 0], [1, 0], scores=[0.1])


def test_threshold_for_fpr_respects_target():
    rng = np.random.default_rng(1)
    normal = rng.normal(size=1000)
    for target in (0.001, 0.01, 0.05, 0.2):
        t = threshold_for_fpr(normal, target)
        assert (normal > t).mean() <= target
        assert (normal >= t).mean() > target - 0.002            # not overly conservative
    tied = np.zeros(100)
    assert (tied > threshold_for_fpr(tied, 0.05)).mean() == 0.0
    for bad in (0, 1, -0.1, True, "0.1"):
        with pytest.raises(MetricError):
            threshold_for_fpr(normal, bad)
    with pytest.raises(MetricError):
        threshold_for_fpr([1.0, np.nan], 0.1)


def test_compare_detectors_same_examples():
    y = pd.Series(Y, index=[f"r{i}" for i in range(8)])
    groups = np.array(["a", "b", "normal", "normal", "normal", "a", "normal", "normal"])
    table = compare_detectors(
        y,
        {
            "perfect": {"predictions": Y, "scores": Y.astype(float)},
            "silent": (np.zeros(8, int), None),
        },
        groups=groups,
    )
    assert list(table.index) == ["perfect", "silent"]
    assert table.loc["perfect", "f1"] == 1.0 and table.loc["perfect", "pr_auc"] == 1.0
    assert table.loc["silent", "precision"] is None or pd.isna(table.loc["silent", "precision"])
    assert table.loc["perfect", "recall[a]"] == 1.0 and table.loc["silent", "recall[b]"] == 0.0


def test_compare_detectors_rejects_misaligned_or_wrong_length():
    y = pd.Series(Y, index=range(8))
    with pytest.raises(MetricError, match="aligned"):
        compare_detectors(y, {"x": {"predictions": pd.Series(P, index=range(1, 9))}})
    with pytest.raises(MetricError, match="length"):
        compare_detectors(Y, {"x": {"predictions": P[:-1]}})
    with pytest.raises(MetricError):
        compare_detectors(Y, {})
    with pytest.raises(MetricError):
        compare_detectors(Y, {"x": 42})


def test_recall_by_group():
    r = recall_by_group(Y, P, np.array(["a", "b", "n", "n", "n", "a", "n", "n"]))
    assert r.to_dict() == {"a": 1.0, "b": 0.0}

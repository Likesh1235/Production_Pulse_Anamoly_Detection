"""Tests for the single-feature rule check and PCA baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production_pulse.baselines import (
    BaselineError,
    LeakageError,
    PCAReconstructionDetector,
    SingleFeatureThresholdDetector,
)
from production_pulse.data import FEATURE_COLUMNS

from conftest import make_correlated_frame

F = list(FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def data():
    return (
        make_correlated_frame(20, 50, seed=1)[F],   # train-normal
        make_correlated_frame(10, 50, seed=2)[F],   # calibration-normal
        make_correlated_frame(10, 50, seed=3)[F],   # held-out evaluation
    )


# ------------------------------------------------------------------ single-feature rule check
def test_rule_limits_come_from_training_only(data):
    train, cal, test = data
    det = SingleFeatureThresholdDetector().fit(train)
    lim = det.limits()
    np.testing.assert_allclose(lim["lower"], np.quantile(train.to_numpy(), 0.0005, axis=0))
    np.testing.assert_allclose(lim["upper"], np.quantile(train.to_numpy(), 0.9995, axis=0))
    assert det.threshold_ == 1.0 and det.threshold_source_.startswith("rule")
    before = lim.copy()
    det.detect(test)
    pd.testing.assert_frame_equal(det.limits(), before)        # evaluation never changes limits


def test_rule_flags_exactly_out_of_limit_rows(data):
    train, _, test = data
    det = SingleFeatureThresholdDetector().fit(train)
    ex = det.exceedances(test)
    np.testing.assert_array_equal(det.detect(test).predictions, ex.any(axis=1).to_numpy())
    spiked = test.copy()
    spiked.iloc[3, 7] = det.limits()["upper"].iloc[7] + 1.0
    assert det.detect(spiked).predictions[3]
    assert det.exceedances(spiked).iloc[3, 7]


def test_rule_is_blind_to_in_range_joint_changes(data):
    train, _, test = data
    det = SingleFeatureThresholdDetector().fit(train)
    lim = det.limits()
    mixed = pd.DataFrame(np.tile(((lim["lower"] + lim["upper"]) / 2).to_numpy(), (5, 1)), columns=F)
    mixed.iloc[:, ::2] = lim["upper"].to_numpy()[::2] - 1e-9     # every other feature at its edge, still inside
    assert not det.detect(mixed).predictions.any()


# ------------------------------------------------------------------ PCA
def test_pca_fit_is_deterministic_and_reconstructs_low_rank(data):
    train, _, test = data
    a = PCAReconstructionDetector(n_components=5).fit(train)
    b = PCAReconstructionDetector(n_components=5).fit(train)
    np.testing.assert_array_equal(a.components_, b.components_)
    assert a.n_components_ == 5 and a.explained_variance_ratio_.sum() > 0.9
    np.testing.assert_allclose(a.components_ @ a.components_.T, np.eye(5), atol=1e-10)
    assert a.score(test).mean() < 0.1 * np.sum(test.var())       # 5 factors explain most variance


def test_pca_variance_fraction_and_subsampling(data):
    train, _, _ = data
    frac = PCAReconstructionDetector(n_components=0.95).fit(train)
    assert 1 <= frac.n_components_ < len(F) and frac.explained_variance_ratio_.sum() >= 0.95 - 1e-9
    s1 = PCAReconstructionDetector(n_components=5, max_fit_rows=300, seed=4).fit(train)
    s2 = PCAReconstructionDetector(n_components=5, max_fit_rows=300, seed=4).fit(train)
    np.testing.assert_array_equal(s1.components_, s2.components_)  # seeded subsample is reproducible


def test_pca_needs_calibration_and_hits_target_fpr(data):
    train, cal, test = data
    det = PCAReconstructionDetector(n_components=5).fit(train)
    with pytest.raises(BaselineError, match="calibrate"):
        det.detect(test)
    det.calibrate(cal, target_fpr=0.02)
    assert det.detect(cal, check_unseen=False).predictions.mean() <= 0.02
    assert det.threshold_source_.startswith("calibrated")
    off_model = test.copy()
    off_model.iloc[:, 0] += 3.0                                  # breaks the factor structure
    assert det.detect(off_model).predictions.mean() > 0.9


def test_pca_reconstruction_errors_sum_to_score(data):
    train, _, test = data
    det = PCAReconstructionDetector(n_components=5).fit(train)
    np.testing.assert_allclose(det.reconstruction_errors(test).sum(axis=1).to_numpy(), det.score(test))


@pytest.mark.parametrize("kwargs", [{"n_components": 0}, {"n_components": 52}, {"n_components": 1.0},
                                    {"n_components": True}, {"n_components": "5"}, {"max_fit_rows": 1},
                                    {"seed": -3}])
def test_pca_invalid_settings(kwargs):
    with pytest.raises(BaselineError):
        PCAReconstructionDetector(**kwargs)


# ------------------------------------------------------------------ leakage prevention
@pytest.mark.parametrize("cls", [SingleFeatureThresholdDetector, lambda: PCAReconstructionDetector(n_components=5)])
def test_leakage_guards(data, cls):
    train, cal, test = data
    det = cls().fit(train)
    with pytest.raises(LeakageError, match="also used for fitting"):
        det.calibrate(pd.concat([cal, train.iloc[:10]]), target_fpr=0.05)
    det.calibrate(cal, target_fpr=0.05)
    with pytest.raises(LeakageError, match="fitting"):
        det.detect(pd.concat([test, train.iloc[:3]]))
    with pytest.raises(LeakageError, match="calibration"):
        det.detect(pd.concat([test, cal.iloc[:3]]))
    det.detect(test)                                              # clean held-out data is accepted


def test_usage_errors(data):
    train, cal, test = data
    det = SingleFeatureThresholdDetector()
    with pytest.raises(BaselineError, match="fit"):
        det.score(test)
    det.fit(train)
    with pytest.raises(BaselineError, match="already fitted"):
        det.fit(cal)
    with pytest.raises(ValueError, match="DataFrame"):
        det.score(test.to_numpy())
    with pytest.raises(ValueError, match="Metadata"):
        SingleFeatureThresholdDetector(feature_columns=("xmeas_1", "faultNumber"))
    with pytest.raises(BaselineError):
        SingleFeatureThresholdDetector(lower_quantile=0.6)
    with pytest.raises(BaselineError):
        det.set_threshold(float("nan"))
    det.set_threshold(2.5, source="manual test")
    assert det.threshold_info()["threshold"] == 2.5


def test_refit_resets_calibration(data):
    train, cal, _ = data
    det = PCAReconstructionDetector(n_components=5).fit(train)
    det.calibrate(cal, target_fpr=0.05)
    det.fit(cal, refit=True)
    assert det.threshold_ is None and det.n_calibration_rows_ == 0

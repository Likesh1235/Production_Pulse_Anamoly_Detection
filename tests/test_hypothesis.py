"""Measurable experiment for the worklet hypothesis (on SYNTHETIC correlated data).

Hypothesis: an anomaly family can be multivariately abnormal while every
modified feature stays inside its empirical normal range, so a single-feature
rule check misses it but a multivariate detector catches it.

These tests check that the assessment machinery *measures* this and reports
failures. They are NOT evidence about TEP - that comes from notebook 02.
"""

from __future__ import annotations

import pytest

from production_pulse.anomaly_generation import FAMILIES, AnomalyGenerator, AnomalySpec
from production_pulse.baselines import PCAReconstructionDetector, SingleFeatureThresholdDetector
from production_pulse.data import FEATURE_COLUMNS
from production_pulse.evaluation import HypothesisCriteria, MetricError, assess_family_detectability

from conftest import make_correlated_frame

F = list(FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def setup():
    train, cal, test = (make_correlated_frame(40, 50, seed=s) for s in (1, 2, 3))
    single = SingleFeatureThresholdDetector().fit(train[F])
    pca = PCAReconstructionDetector(n_components=5).fit(train[F])
    pca.calibrate(cal[F], target_fpr=0.01)
    return train, test, single, pca


def _assess(setup, spec, policy="clip", min_events=1):
    train, test, single, pca = setup
    gen = AnomalyGenerator(range_policy=policy).fit(train[F])
    r = gen.generate(test[F], spec, seed=7, groups=test["simulationRun"])
    # Synthetic rows are independent (no autocorrelation), so one event per feature is enough
    # for the marginal check here; the TEP notebook keeps the default minimum of 3 events.
    table = assess_family_detectability(
        r.row_info, single.detect(r.features), pca.detect(r.features), features=r.features, events=r.events,
        clean_features=test[F], min_events_for_marginal=min_events,
    )
    return table.loc[spec.family]


def test_relationship_break_is_supported_on_correlated_data(setup):
    row = _assess(setup, AnomalySpec("correlation_break", 0.1))
    assert row["in_range_fraction"] == 1.0
    assert row["single_feature_recall"] - row["single_feature_fpr"] <= 0.05
    assert row["multivariate_recall"] >= 0.5
    assert row["verdict"] == "SUPPORTED"


def test_decoupling_is_supported_and_marginally_hidden(setup):
    # enough events per feature (>= 3) for a stable marginal estimate
    row = _assess(setup, AnomalySpec("decoupling", 0.2), min_events=3)
    assert row["verdict"] == "SUPPORTED"
    assert bool(row["marginally_hidden"]) is True                       # values are genuine normal values
    assert row["marginal_auc_excess"] <= 0.05


def test_unconstrained_large_shift_fails_and_says_why(setup):
    row = _assess(setup, AnomalySpec("coordinated_shift", 0.1, magnitude=6.0), policy="none")
    assert row["verdict"] == "NOT SUPPORTED"
    assert "left the normal range" in row["reasons"]
    assert "single-feature rule check detects it" in row["reasons"]


def test_too_weak_anomaly_fails_on_multivariate_recall(setup):
    row = _assess(setup, AnomalySpec("minor_component_shift", 0.1, magnitude=2.0))
    assert row["verdict"] == "NOT SUPPORTED"
    assert row["reasons"] == "multivariate detector misses it"


def test_in_range_but_marginally_visible_is_flagged(setup):
    """In range for the rule check, yet a single feature still separates it."""
    row = _assess(setup, AnomalySpec("coordinated_shift", 0.1, magnitude=2.5))
    assert row["in_range_fraction"] == 1.0
    assert row["single_feature_recall"] - row["single_feature_fpr"] <= 0.05
    assert bool(row["marginally_hidden"]) is False                       # stricter check exposes it
    assert row["marginal_auc_excess"] > 0.2


def test_marginal_check_controls_for_chance_with_clean_reference(setup):
    """Without injection effect the excess must be ~0 even if the raw AUC is not 0.5."""
    train, test, single, pca = setup
    gen = AnomalyGenerator().fit(train[F])
    r = gen.generate(test[F], AnomalySpec("decoupling", 0.1), seed=7, groups=test["simulationRun"])
    t = assess_family_detectability(r.row_info, single.detect(r.features), pca.detect(r.features),
                                    features=test[F], events=r.events, clean_features=test[F], min_events_for_marginal=1)
    assert abs(t.loc["decoupling", "marginal_auc_excess"]) < 1e-12   # "injected" == clean -> zero excess
    t2 = assess_family_detectability(r.row_info, single.detect(r.features), pca.detect(r.features),
                                     features=r.features, events=r.events, min_events_for_marginal=10_000)
    assert t2["max_marginal_auc"].isna().all() and t2["marginally_hidden"].isna().all()  # too few events -> not computed


def test_every_family_is_measured_not_assumed(setup):
    train, test, single, pca = setup
    gen = AnomalyGenerator().fit(train[F])
    specs = [AnomalySpec(f, 0.03) for f in sorted(FAMILIES)]
    r = gen.generate(test[F], specs, seed=5, groups=test["simulationRun"])
    assert set(r.family_status.values()) == {r.family_status[next(iter(r.family_status))]}  # all UNVERIFIED
    table = assess_family_detectability(r.row_info, single.detect(r.features), pca.detect(r.features))
    assert set(table.index) == set(FAMILIES)
    assert set(table["verdict"]) <= {"SUPPORTED", "NOT SUPPORTED"}
    assert table["marginally_hidden"].isna().all()                 # not computed without features


def test_criteria_are_explicit_and_change_verdicts(setup):
    train, test, single, pca = setup
    gen = AnomalyGenerator().fit(train[F])
    r = gen.generate(test[F], AnomalySpec("correlation_break", 0.1), seed=7, groups=test["simulationRun"])
    strict = HypothesisCriteria(min_multivariate_recall=0.999)
    t = assess_family_detectability(r.row_info, single.detect(r.features), pca.detect(r.features), criteria=strict)
    assert t.loc["correlation_break", "verdict"] == "NOT SUPPORTED"


def test_assessment_input_validation(setup):
    train, test, single, pca = setup
    gen = AnomalyGenerator().fit(train[F])
    r = gen.generate(test[F], AnomalySpec("short_event", 0.05), seed=1)
    det = single.detect(r.features)
    with pytest.raises(MetricError):
        assess_family_detectability(r.row_info.drop(columns="family"), det, det)
    with pytest.raises(MetricError):
        assess_family_detectability(r.row_info.iloc[:-1], det, det)
    with pytest.raises(MetricError, match="No anomalous"):
        assess_family_detectability(r.row_info.assign(is_anomaly=0), det, det)
    with pytest.raises(MetricError, match="both features and events"):
        assess_family_detectability(r.row_info, det, det, features=r.features)
    with pytest.raises(MetricError, match="needs features and events"):
        assess_family_detectability(r.row_info, det, det, clean_features=r.features)
    with pytest.raises(MetricError, match="same rows"):
        assess_family_detectability(r.row_info, det, det, features=r.features, events=r.events,
                                    clean_features=r.features.iloc[::-1])

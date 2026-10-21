"""Tests for seeded synthetic multivariate anomaly generation (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production_pulse.anomaly_generation import (
    FAMILIES,
    UNVERIFIED,
    AnomalyGenerationError,
    AnomalyGenerator,
    AnomalySpec,
)
from production_pulse.config import default_config
from production_pulse.data import FEATURE_COLUMNS, METADATA_COLUMNS
from production_pulse.paths import UnsafeOutputPathError, get_paths

from conftest import make_correlated_frame

F = list(FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def frames():
    return make_correlated_frame(30, 40, seed=1), make_correlated_frame(10, 60, seed=2)


@pytest.fixture(scope="module")
def gen(frames):
    return AnomalyGenerator().fit(frames[0][F])


def _run(gen, ev, spec, seed=3, **kw):
    return gen.generate(ev[F], spec, seed=seed, groups=ev["simulationRun"], **kw)


# ------------------------------------------------------------------ every family
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_family_labels_and_structure(gen, frames, family):
    ev = frames[1]
    r = _run(gen, ev, AnomalySpec(family, 0.08))
    info = r.row_info
    assert list(r.features.columns) == F                      # feature order preserved
    assert r.features.index.equals(ev.index)                  # row identity preserved
    assert not set(METADATA_COLUMNS) & set(r.features.columns)
    assert set(info["is_anomaly"].unique()) <= {0, 1}
    n_anom = int(info["is_anomaly"].sum())
    assert 0.08 * len(ev) <= n_anom < 0.08 * len(ev) + FAMILIES[family].default_event_length[1]
    assert (info.loc[info.is_anomaly == 0, "event_id"] == -1).all()
    assert (info.loc[info.is_anomaly == 0, "family"] == "normal").all()
    assert (info.loc[info.is_anomaly == 1, "family"] == family).all()
    assert r.events["length"].sum() == n_anom
    assert r.family_status == {family: UNVERIFIED}            # never auto-labelled successful
    # only anomalous rows changed
    changed = ~np.isclose(r.features.to_numpy(), ev[F].to_numpy()).all(axis=1)
    assert not changed[info["is_anomaly"].to_numpy() == 0].any()
    assert changed[info["is_anomaly"].to_numpy() == 1].mean() > 0.9


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_events_stay_inside_one_run_and_are_contiguous(gen, frames, family):
    ev = frames[1]
    r = _run(gen, ev, AnomalySpec(family, 0.08))
    for e in r.events.itertuples():
        rows = ev.iloc[e.start_row : e.end_row + 1]
        assert rows["simulationRun"].nunique() == 1 and rows["simulationRun"].iloc[0] == e.group
        assert (r.row_info["event_id"].iloc[e.start_row : e.end_row + 1] == e.event_id).all()
        lo, hi = (FAMILIES[family].default_event_length)
        assert lo <= e.length <= hi


def test_reproducible_and_seed_dependent(gen, frames):
    ev = frames[1]
    spec = [AnomalySpec("correlation_break", 0.05), AnomalySpec("short_event", 0.03)]
    a, b = _run(gen, ev, spec, seed=11), _run(gen, ev, spec, seed=11)
    pd.testing.assert_frame_equal(a.features, b.features)
    pd.testing.assert_frame_equal(a.row_info, b.row_info)
    pd.testing.assert_frame_equal(a.events, b.events)
    c = _run(gen, ev, spec, seed=12)
    assert not a.row_info["is_anomaly"].equals(c.row_info["is_anomaly"])


def test_input_not_modified(gen, frames):
    ev = frames[1]
    before = ev.copy()
    _run(gen, ev, AnomalySpec("coordinated_shift", 0.1, magnitude=3.0))
    pd.testing.assert_frame_equal(ev, before)


def test_metadata_columns_ignored_as_features(gen, frames):
    ev = frames[1]
    r = gen.generate(ev, AnomalySpec("decoupling", 0.05), seed=1, groups=ev["simulationRun"])  # full frame incl. metadata
    assert list(r.features.columns) == F


def test_multiple_families_do_not_overlap(gen, frames):
    ev = frames[1]
    specs = [AnomalySpec(f, 0.04) for f in sorted(FAMILIES)]
    r = _run(gen, ev, specs, min_gap=2)
    assert set(r.events["family"]) == set(FAMILIES)
    ev_sorted = r.events.sort_values("start_row")
    same_group = ev_sorted["group"].to_numpy()[1:] == ev_sorted["group"].to_numpy()[:-1]
    gaps = ev_sorted["start_row"].to_numpy()[1:] - ev_sorted["end_row"].to_numpy()[:-1]
    assert (gaps[same_group] > 2).all()
    assert r.events["event_id"].is_unique


def test_clip_policy_creates_no_new_range_violations(gen, frames):
    ev = frames[1]
    r = _run(gen, ev, AnomalySpec("coordinated_shift", 0.1, magnitude=6.0))
    lo, hi = gen.ranges()["lower"].to_numpy(), gen.ranges()["upper"].to_numpy()
    new, old = r.features.to_numpy(), ev[F].to_numpy()
    assert (new >= np.minimum(lo, old) - 1e-12).all() and (new <= np.maximum(hi, old) + 1e-12).all()
    assert r.row_info["modified_in_range"].all()
    assert r.events["n_new_range_violations"].sum() > 0     # big shift really needed clipping


def test_none_policy_reports_violations(frames):
    tr, ev = frames
    g = AnomalyGenerator(range_policy="none").fit(tr[F])
    r = _run(g, ev, AnomalySpec("coordinated_shift", 0.1, magnitude=6.0))
    anomalous = r.row_info["is_anomaly"] == 1
    assert (~r.row_info.loc[anomalous, "modified_in_range"]).mean() > 0.5
    assert r.row_info.loc[~anomalous, "modified_in_range"].all()


def test_decoupling_uses_real_values_from_other_runs(gen, frames):
    ev = frames[1]
    r = _run(gen, ev, AnomalySpec("decoupling", 0.05))
    for e in r.events.itertuples():
        assert e.donor_group != "" and float(e.donor_group) != float(e.group)
    # every injected value of a replaced feature exists somewhere in the clean data
    e = r.events.iloc[0]
    feat = e["features"].split(";")[0]
    injected = r.features[feat].iloc[e["start_row"] : e["end_row"] + 1].to_numpy()
    assert np.isin(injected, ev[feat].to_numpy()).all()


def test_correlation_break_keeps_anchor_untouched(gen, frames):
    ev = frames[1]
    r = _run(gen, ev, AnomalySpec("correlation_break", 0.05))
    e = r.events.iloc[0]
    sl = slice(e["start_row"], e["end_row"] + 1)
    assert e["anchor_feature"] not in e["features"].split(";")
    np.testing.assert_array_equal(r.features[e["anchor_feature"]].iloc[sl], ev[e["anchor_feature"]].iloc[sl])


def test_correlated_pairs_and_ranges(gen):
    pairs = gen.correlated_pairs(top=5)
    assert len(pairs) == 5 and (pairs["r"].abs() >= gen.correlation_threshold).all()
    rng = gen.ranges()
    assert list(rng.index) == F and (rng["lower"] < rng["upper"]).all()


def test_save_ground_truth_only_in_output_folders(fake_root, gen, frames):
    ev = frames[1]
    r = _run(gen, ev, AnomalySpec("short_event", 0.05))
    paths = get_paths(fake_root, config=default_config())
    out = r.save("data/processed/injection_test", paths)
    saved = pd.read_csv(out["row_info"], index_col=0)
    assert saved["is_anomaly"].sum() == r.labels.sum()
    with pytest.raises(FileExistsError):
        r.save("data/processed/injection_test", paths)
    with pytest.raises(UnsafeOutputPathError):
        r.save("docs", paths)


# ------------------------------------------------------------------ invalid input
@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": "teleport", "anomaly_rate": 0.1},
        {"family": "coordinated_shift", "anomaly_rate": 0.0},
        {"family": "coordinated_shift", "anomaly_rate": 0.6},
        {"family": "coordinated_shift", "anomaly_rate": True},
        {"family": "coordinated_shift", "anomaly_rate": 0.1, "magnitude": -1.0},
        {"family": "coordinated_shift", "anomaly_rate": 0.1, "magnitude": float("inf")},
        {"family": "correlation_break", "anomaly_rate": 0.1, "magnitude": 1.5},
        {"family": "coordinated_shift", "anomaly_rate": 0.1, "n_features": 1},
        {"family": "minor_component_shift", "anomaly_rate": 0.1, "n_features": 3},
        {"family": "short_event", "anomaly_rate": 0.1, "event_length": (0, 3)},
        {"family": "short_event", "anomaly_rate": 0.1, "event_length": (5, 3)},
        {"family": "short_event", "anomaly_rate": 0.1, "event_length": (1.5, 3)},
    ],
)
def test_invalid_specs(kwargs):
    with pytest.raises(AnomalyGenerationError):
        AnomalySpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"range_quantiles": (0.6, 0.9)},
        {"correlation_threshold": 1.0},
        {"n_minor_components": 0},
        {"range_policy": "reject"},
        {"feature_columns": ("xmeas_1", "simulationRun")},
    ],
)
def test_invalid_generator_settings(kwargs):
    with pytest.raises((AnomalyGenerationError, ValueError)):
        AnomalyGenerator(**kwargs)


def test_invalid_generate_calls(gen, frames):
    tr, ev = frames
    spec = AnomalySpec("short_event", 0.05)
    with pytest.raises(AnomalyGenerationError, match="fit"):
        AnomalyGenerator().generate(ev[F], spec, seed=0)
    with pytest.raises(AnomalyGenerationError, match="fit the generator"):
        gen.generate(tr[F], spec, seed=0)                       # reference rows reused
    with pytest.raises(AnomalyGenerationError, match="seed"):
        gen.generate(ev[F], spec, seed=-1)
    with pytest.raises(AnomalyGenerationError, match="groups"):
        gen.generate(ev[F], spec, seed=0, groups=ev["simulationRun"].iloc[:-1])
    with pytest.raises(AnomalyGenerationError, match="unique"):
        gen.generate(pd.concat([ev[F].iloc[:5], ev[F].iloc[:5]]).reset_index(drop=True).set_axis([0] * 10), spec, seed=0)
    with pytest.raises(AnomalyGenerationError, match="Total anomaly_rate"):
        gen.generate(ev[F], [AnomalySpec("short_event", 0.3), AnomalySpec("decoupling", 0.3)], seed=0)
    with pytest.raises(AnomalyGenerationError, match="longer than every group"):
        gen.generate(ev[F], AnomalySpec("coordinated_shift", 0.05, event_length=(100, 120)), seed=0,
                     groups=ev["simulationRun"])
    # 10 runs x 60 rows; with 25-row events and a 20-row gap only one event fits per run
    # (25 + 20 + 25 > 60), so at most 250 of the 300 requested rows can be placed.
    with pytest.raises(AnomalyGenerationError, match="could only place"):
        gen.generate(ev[F], AnomalySpec("coordinated_shift", 0.5, event_length=(25, 25)), seed=0,
                     groups=ev["simulationRun"], min_gap=20)
    bad = ev[F].copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        gen.generate(bad, spec, seed=0)
    with pytest.raises(AnomalyGenerationError):
        gen.generate(ev[F], [], seed=0)


def test_reference_overlap_can_be_explicitly_allowed(gen, frames):
    tr = frames[0]
    r = gen.generate(tr[F], AnomalySpec("short_event", 0.02), seed=0, allow_reference_overlap=True)
    assert r.labels.sum() > 0


def test_no_correlated_features_is_a_clear_error(tep_frame):
    g = AnomalyGenerator(correlation_threshold=0.95).fit(tep_frame[F])   # independent features
    other = tep_frame[F] + 0.001
    with pytest.raises(AnomalyGenerationError, match="correlation_threshold"):
        g.generate(other, AnomalySpec("correlation_break", 0.05), seed=0, groups=tep_frame["simulationRun"])

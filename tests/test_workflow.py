"""Tests for the shared leakage-free data preparation (fake project, synthetic RData)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from production_pulse.config import default_config
from production_pulse.data import FEATURE_COLUMNS, METADATA_COLUMNS
from production_pulse.paths import get_paths
from production_pulse.workflow import prepare_fault_free_data

from conftest import make_correlated_frame


@pytest.fixture
def project(fake_root):
    pyreadr = pytest.importorskip("pyreadr")
    for name, obj, runs, samples, seed in (
        ("TEP_FaultFree_Training.RData", "fault_free_training", 10, 30, 1),
        ("TEP_FaultFree_Testing.RData", "fault_free_testing", 20, 40, 2),
    ):
        target = fake_root / name
        target.unlink()
        pyreadr.write_rdata(str(target), make_correlated_frame(runs, samples, seed=seed), df_name=obj)
    return get_paths(fake_root, config=default_config())


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_roles_are_run_disjoint_and_scaled_on_train_only(project):
    before = {p.name: _digest(p) for p in project.protected_files if p.suffix == ".RData"}
    data = prepare_fault_free_data(project, default_config())
    assert set(data.features) == {"train", "validation", "calibration", "evaluation"}
    tr, va = set(data.metadata["train"]["simulationRun"]), set(data.metadata["validation"]["simulationRun"])
    ca, ev = set(data.metadata["calibration"]["simulationRun"]), set(data.metadata["evaluation"]["simulationRun"])
    assert not tr & va and not ca & ev
    assert len(tr) + len(va) == 10 and len(ca) == 4 and len(ev) == 14     # 0.2 / 0.7 / 0.1 of 20 runs
    np.testing.assert_allclose(data["train"].mean().to_numpy(), 0, atol=1e-9)
    assert data.scaler.n_samples_fit_ == len(data["train"])
    for role, frame in data.features.items():
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        assert list(data.metadata[role].columns) == list(METADATA_COLUMNS)
        assert frame.index.equals(data.metadata[role].index)
    assert before == {p.name: _digest(p) for p in project.protected_files if p.suffix == ".RData"}  # read-only


def test_holdout_is_locked_by_default(project):
    data = prepare_fault_free_data(project, default_config())
    with pytest.raises(KeyError, match="locked"):
        data["holdout"]
    summary = data.summary()
    assert not summary.loc["holdout", "loaded"] and summary.loc["holdout", "n_runs"] == 2
    unlocked = prepare_fault_free_data(project, default_config(), include_holdout=True)
    assert len(unlocked["holdout"]) == 2 * 40


def test_reproducible_and_seeded(project):
    a = prepare_fault_free_data(project, default_config())
    b = prepare_fault_free_data(project, default_config())
    c = prepare_fault_free_data(project, default_config(), seed=7)
    assert a.train_split == b.train_split and a.test_split == b.test_split
    assert a.test_split != c.test_split
    assert a.to_dict()["scaler"] == b.to_dict()["scaler"]

"""Tests for leakage-safe feature scaling (synthetic data only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from production_pulse.config import default_config
from production_pulse.data import FEATURE_COLUMNS
from production_pulse.paths import UnsafeOutputPathError, get_paths
from production_pulse.preprocessing import FeatureScaler, NotFittedError, PreprocessingError

from conftest import make_tep_frame


@pytest.fixture
def train():
    return make_tep_frame(n_runs=8, seed=1)


@pytest.fixture
def test_df():
    df = make_tep_frame(n_runs=4, seed=2)
    df[list(FEATURE_COLUMNS)] += 5.0  # deliberately shifted distribution
    return df


def test_standard_scaling_uses_training_statistics_only(train, test_df):
    scaler = FeatureScaler().fit(train)
    center, scale = scaler.center_.copy(), scaler.scale_.copy()
    out = scaler.transform(test_df)
    np.testing.assert_allclose(center, train[list(FEATURE_COLUMNS)].mean().to_numpy())
    np.testing.assert_allclose(scale, train[list(FEATURE_COLUMNS)].std(ddof=0).to_numpy())
    np.testing.assert_array_equal(scaler.center_, center)  # transform did not change params
    expected = (test_df[list(FEATURE_COLUMNS)].to_numpy() - center) / scale
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    assert abs(float(out.mean())) > 1.0  # shifted test data is NOT re-centred -> no leakage


def test_train_output_is_standardised(train):
    out = FeatureScaler().fit_transform(train)
    assert out.dtype == np.float32 and out.shape == (len(train), 52)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-4)


def test_accidental_refit_blocked(train, test_df):
    scaler = FeatureScaler().fit(train)
    with pytest.raises(PreprocessingError, match="already fitted"):
        scaler.fit(test_df)
    scaler.fit(test_df, refit=True)  # explicit opt-in works


def test_transform_before_fit():
    with pytest.raises(NotFittedError):
        FeatureScaler().transform(make_tep_frame())


def test_feature_order_preserved_regardless_of_input_order(train):
    scaler = FeatureScaler().fit(train)
    scrambled = train[list(reversed(train.columns))]
    np.testing.assert_array_equal(scaler.transform(train), scaler.transform(scrambled))
    frame = scaler.transform_frame(scrambled)
    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert frame.index.equals(scrambled.index)


def test_metadata_never_enters_features(train):
    out = FeatureScaler().fit_transform(train)
    assert out.shape[1] == 52  # 3 metadata columns dropped automatically
    with pytest.raises(PreprocessingError, match="Metadata"):
        FeatureScaler(feature_columns=("xmeas_1", "simulationRun"))
    with pytest.raises(PreprocessingError, match="DataFrame"):
        FeatureScaler().fit(train.to_numpy())  # unnamed arrays refused


def test_missing_or_bad_features(train):
    scaler = FeatureScaler().fit(train)
    with pytest.raises(PreprocessingError, match="missing"):
        scaler.transform(train.drop(columns="xmv_3"))
    bad = train.copy()
    bad.loc[0, "xmeas_1"] = np.nan
    with pytest.raises(PreprocessingError, match="NaN"):
        scaler.transform(bad)
    with pytest.raises(PreprocessingError, match="no rows"):
        scaler.transform(train.iloc[0:0])


def test_constant_feature_handled(train):
    train["xmv_9"] = 3.0
    scaler = FeatureScaler().fit(train)
    assert scaler.constant_features_ == ("xmv_9",)
    out = scaler.transform(train)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, FEATURE_COLUMNS.index("xmv_9")], 0.0)


@pytest.mark.parametrize("method", ["standard", "minmax", "none"])
def test_inverse_roundtrip(train, method):
    scaler = FeatureScaler(method=method).fit(train)
    back = scaler.inverse_transform(scaler.transform(train, dtype=np.float64))
    np.testing.assert_allclose(back, train[list(FEATURE_COLUMNS)].to_numpy(), rtol=1e-10)


def test_minmax_range(train):
    out = FeatureScaler(method="minmax").fit_transform(train)
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6


def test_bad_constructor_args():
    with pytest.raises(PreprocessingError):
        FeatureScaler(method="robust")
    with pytest.raises(PreprocessingError):
        FeatureScaler(feature_columns=())
    with pytest.raises(PreprocessingError):
        FeatureScaler(feature_columns=("xmeas_1", "xmeas_1"))


def test_json_roundtrip_is_exact(train, test_df):
    scaler = FeatureScaler().fit(train, source="unit-test train")
    meta = scaler.to_dict()
    restored = FeatureScaler.from_dict(json.loads(json.dumps(meta)))
    np.testing.assert_array_equal(restored.transform(test_df), scaler.transform(test_df))
    assert meta["feature_columns"] == list(FEATURE_COLUMNS)
    assert meta["fit_source"] == "unit-test train"


def test_corrupted_metadata_rejected(train):
    meta = FeatureScaler().fit(train).to_dict()
    for broken in (
        {**meta, "scale": meta["scale"][:-1]},
        {**meta, "scale": [0.0] * 52},
        {**meta, "format_version": 99},
        {k: v for k, v in meta.items() if k != "center"},
    ):
        with pytest.raises(PreprocessingError):
            FeatureScaler.from_dict(broken)


def test_save_only_inside_output_folders(fake_root, train):
    paths = get_paths(fake_root, config=default_config())
    scaler = FeatureScaler().fit(train)
    saved = scaler.save_json("artifacts/scaler.json", paths)
    assert FeatureScaler.load_json(saved).to_dict() == scaler.to_dict()
    with pytest.raises(FileExistsError):
        scaler.save_json("artifacts/scaler.json", paths)
    with pytest.raises(UnsafeOutputPathError):
        scaler.save_json("TEP_FaultFree_Training.RData", paths, overwrite=True)

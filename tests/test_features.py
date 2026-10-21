"""Tests for the shared feature-matrix and fingerprint helpers."""

from __future__ import annotations

import numpy as np
import pytest

from production_pulse.data import FEATURE_COLUMNS
from production_pulse.features import (
    FeatureInputError,
    feature_matrix,
    overlap_count,
    row_fingerprints,
    validate_feature_columns,
)


def test_matrix_is_ordered_copy(tep_frame):
    scrambled = tep_frame[list(reversed(tep_frame.columns))]
    x = feature_matrix(scrambled)
    np.testing.assert_array_equal(x, tep_frame[list(FEATURE_COLUMNS)].to_numpy())
    x[0, 0] = -1e9
    assert scrambled["xmeas_1"].iloc[0] != -1e9  # caller data untouched


@pytest.mark.parametrize("cols", ["xmeas_1", (), ("xmeas_1", "xmeas_1"), ("xmeas_1", "sample")])
def test_bad_feature_columns(cols):
    with pytest.raises(FeatureInputError):
        validate_feature_columns(cols)


def test_bad_inputs(tep_frame):
    with pytest.raises(FeatureInputError, match="DataFrame"):
        feature_matrix(tep_frame.to_numpy())
    with pytest.raises(FeatureInputError, match="missing"):
        feature_matrix(tep_frame.drop(columns="xmv_1"))
    with pytest.raises(FeatureInputError, match="no rows"):
        feature_matrix(tep_frame.iloc[0:0])
    bad = tep_frame.copy()
    bad.loc[0, "xmeas_2"] = np.inf
    with pytest.raises(FeatureInputError, match="NaN or infinite"):
        feature_matrix(bad)
    bad = tep_frame.copy()
    bad["xmeas_2"] = bad["xmeas_2"].astype(str)
    with pytest.raises(FeatureInputError, match="non-numeric"):
        feature_matrix(bad)


def test_fingerprints_detect_shared_rows(tep_frame):
    x = feature_matrix(tep_frame)
    a, b = row_fingerprints(x[:100]), row_fingerprints(x[90:])
    assert overlap_count(a, b) == 10
    assert overlap_count(row_fingerprints(x[:50]), row_fingerprints(x[50:])) == 0
    with pytest.raises(FeatureInputError):
        row_fingerprints(x[0])

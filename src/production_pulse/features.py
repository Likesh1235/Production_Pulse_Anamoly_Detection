"""Shared helpers for turning DataFrames into model-feature matrices safely.

Used by the Phase 3 modules (anomaly generation, baselines, evaluation).
Features are always selected **by name** in a fixed order, metadata columns
are refused as features, and the returned matrix is always a fresh copy so
callers can never modify the caller's DataFrame through it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .data import FEATURE_COLUMNS, METADATA_COLUMNS


class FeatureInputError(ValueError):
    """Invalid feature input (type, columns, dtype, size or non-finite values)."""


def validate_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """Return feature names as a tuple after checking they are usable."""
    if isinstance(feature_columns, str):
        raise FeatureInputError("feature_columns must be a sequence of names, not a single string")
    cols = tuple(str(c) for c in feature_columns)
    if not cols:
        raise FeatureInputError("feature_columns must not be empty")
    if len(set(cols)) != len(cols):
        raise FeatureInputError("feature_columns contains duplicates")
    leaked = [c for c in cols if c in METADATA_COLUMNS]
    if leaked:
        raise FeatureInputError(f"Metadata column(s) cannot be model features: {leaked}")
    return cols


def feature_matrix(df: Any, feature_columns: Sequence[str] = FEATURE_COLUMNS, *, what: str = "Input") -> np.ndarray:
    """Copy the named feature columns (in the given order) into a float64 array.

    Raises :class:`FeatureInputError` for non-DataFrames, duplicate column
    labels, missing/non-numeric feature columns, empty input or NaN/inf.
    """
    cols = validate_feature_columns(feature_columns)
    if not isinstance(df, pd.DataFrame):
        raise FeatureInputError(
            f"{what} must be a pandas DataFrame with named columns (arrays are refused so that "
            "metadata columns cannot slip in unnoticed)"
        )
    if df.columns.duplicated().any():
        raise FeatureInputError(f"{what} has duplicate column names")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FeatureInputError(f"{what} is missing {len(missing)} feature column(s): {missing[:10]}")
    bad = [c for c in cols if is_bool_dtype(df[c]) or not is_numeric_dtype(df[c])]
    if bad:
        raise FeatureInputError(f"{what} has non-numeric feature column(s): {bad[:10]}")
    if len(df) == 0:
        raise FeatureInputError(f"{what} has no rows")
    values = np.array(df.loc[:, list(cols)].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise FeatureInputError(f"{what} contains NaN or infinite feature values")
    return values


def row_fingerprints(values: np.ndarray) -> np.ndarray:
    """Sorted unique 64-bit hashes of the rows of a feature matrix.

    Used to detect when the same rows are reused for fitting, calibration and
    evaluation (data leakage). Identical rows give identical hashes.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise FeatureInputError(f"Expected a 2-D matrix, got shape {arr.shape}")
    hashes = pd.util.hash_pandas_object(pd.DataFrame(arr), index=False).to_numpy(dtype=np.uint64)
    return np.unique(hashes)


def overlap_count(fingerprints_a: np.ndarray, fingerprints_b: np.ndarray) -> int:
    """Number of row fingerprints shared by two fingerprint arrays."""
    return int(np.intersect1d(fingerprints_a, fingerprints_b, assume_unique=True).size)

"""Feature scaling that is fitted on TRAINING data only.

:class:`FeatureScaler` stores its parameters as plain lists so they can be
saved to JSON (no pickle). It selects features **by name** in a fixed order,
so metadata columns (faultNumber, simulationRun, sample) can never leak into
the model input, and it refuses to be re-fitted by accident.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from . import __version__
from .data import FEATURE_COLUMNS, METADATA_COLUMNS
from .logging_config import get_logger
from .paths import ProjectPaths, safe_output_path

log = get_logger(__name__)

PathLike = Union[str, os.PathLike]
SCALER_FORMAT_VERSION = 1
SCALING_METHODS = ("standard", "minmax", "none")


class PreprocessingError(ValueError):
    """Invalid preprocessing input or state."""


class NotFittedError(PreprocessingError):
    """The scaler was used before ``fit``."""


class FeatureScaler:
    """Per-feature scaling with leakage-safe fit/transform separation.

    method: ``"standard"`` (z-score), ``"minmax"`` (to [0, 1]) or ``"none"``
    (identity; useful for ablation). Features with zero spread in the training
    data get scale 1.0 and are listed in ``constant_features``.
    """

    def __init__(self, feature_columns: Sequence[str] = FEATURE_COLUMNS, method: str = "standard") -> None:
        cols = tuple(str(c) for c in feature_columns)
        if not cols:
            raise PreprocessingError("feature_columns must not be empty")
        if len(set(cols)) != len(cols):
            raise PreprocessingError("feature_columns contains duplicates")
        leaked = [c for c in cols if c in METADATA_COLUMNS]
        if leaked:
            raise PreprocessingError(f"Metadata column(s) cannot be model features: {leaked}")
        if method not in SCALING_METHODS:
            raise PreprocessingError(f"method must be one of {SCALING_METHODS}, got {method!r}")
        self.feature_columns: tuple[str, ...] = cols
        self.method = method
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.constant_features_: tuple[str, ...] = ()
        self.n_samples_fit_: int = 0
        self.fit_source_: str = ""

    # ------------------------------------------------------------------ helpers
    @property
    def is_fitted(self) -> bool:
        return self.center_ is not None

    def _extract(self, df: Any) -> np.ndarray:
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError(
                "Input must be a pandas DataFrame with named columns (arrays are refused so that "
                "metadata columns cannot slip in unnoticed)"
            )
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise PreprocessingError(f"Input is missing {len(missing)} feature column(s): {missing[:10]}")
        bad = [c for c in self.feature_columns if is_bool_dtype(df[c]) or not is_numeric_dtype(df[c])]
        if bad:
            raise PreprocessingError(f"Non-numeric feature column(s): {bad[:10]}")
        values = df.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64)
        if values.shape[0] == 0:
            raise PreprocessingError("Input has no rows")
        if not np.isfinite(values).all():
            raise PreprocessingError("Input contains NaN or infinite feature values")
        return values

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("FeatureScaler is not fitted; call fit() on the TRAINING data first")

    # ------------------------------------------------------------------ API
    def fit(self, train_df: pd.DataFrame, *, source: str = "", refit: bool = False) -> "FeatureScaler":
        """Learn parameters from training data only.

        Raises if already fitted unless ``refit=True`` (prevents accidentally
        re-fitting on validation/test data).
        """
        if self.is_fitted and not refit:
            raise PreprocessingError(
                "Scaler is already fitted. Use transform() for validation/test data, "
                "or pass refit=True if you really mean to re-fit on new TRAINING data."
            )
        x = self._extract(train_df)
        if self.method == "standard":
            center = x.mean(axis=0)
            spread = x.std(axis=0)
        elif self.method == "minmax":
            center = x.min(axis=0)
            spread = x.max(axis=0) - center
        else:  # "none"
            center = np.zeros(x.shape[1])
            spread = np.ones(x.shape[1])
        constant = spread <= 1e-12
        spread = np.where(constant, 1.0, spread)
        self.center_, self.scale_ = center, spread
        self.constant_features_ = tuple(c for c, k in zip(self.feature_columns, constant) if k)
        self.n_samples_fit_ = int(x.shape[0])
        self.fit_source_ = str(source)
        if self.constant_features_:
            log.warning("Constant feature(s) in training data (scale set to 1): %s", self.constant_features_)
        log.info("Fitted %s scaler on %d rows (%s)", self.method, self.n_samples_fit_, source or "unnamed")
        return self

    def transform(self, df: pd.DataFrame, *, dtype: Any = np.float32) -> np.ndarray:
        """Scale features with the fitted (training) parameters. Never re-fits."""
        self._require_fitted()
        x = self._extract(df)
        return ((x - self.center_) / self.scale_).astype(dtype, copy=False)

    def fit_transform(self, train_df: pd.DataFrame, *, source: str = "", dtype: Any = np.float32) -> np.ndarray:
        return self.fit(train_df, source=source).transform(train_df, dtype=dtype)

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Like :meth:`transform` but returns a DataFrame (same index, feature names)."""
        return pd.DataFrame(self.transform(df, dtype=np.float64), index=df.index, columns=list(self.feature_columns))

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_columns):
            raise PreprocessingError(f"Expected shape (n, {len(self.feature_columns)}), got {arr.shape}")
        return arr * self.scale_ + self.center_

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "format_version": SCALER_FORMAT_VERSION,
            "package_version": __version__,
            "method": self.method,
            "feature_columns": list(self.feature_columns),
            "center": [float(v) for v in self.center_],
            "scale": [float(v) for v in self.scale_],
            "constant_features": list(self.constant_features_),
            "n_samples_fit": self.n_samples_fit_,
            "fit_source": self.fit_source_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureScaler":
        try:
            if data["format_version"] != SCALER_FORMAT_VERSION:
                raise PreprocessingError(f"Unsupported scaler format_version {data['format_version']}")
            scaler = cls(data["feature_columns"], method=data["method"])
            center = np.asarray(data["center"], dtype=np.float64)
            scale = np.asarray(data["scale"], dtype=np.float64)
        except KeyError as exc:
            raise PreprocessingError(f"Scaler metadata is missing key {exc}") from exc
        n = len(scaler.feature_columns)
        if center.shape != (n,) or scale.shape != (n,):
            raise PreprocessingError("Scaler metadata has wrong parameter lengths")
        if not (np.isfinite(center).all() and np.isfinite(scale).all() and (scale > 0).all()):
            raise PreprocessingError("Scaler metadata contains invalid parameter values")
        scaler.center_, scaler.scale_ = center, scale
        scaler.constant_features_ = tuple(data.get("constant_features", ()))
        scaler.n_samples_fit_ = int(data.get("n_samples_fit", 0))
        scaler.fit_source_ = str(data.get("fit_source", ""))
        return scaler

    def save_json(self, path: PathLike, paths: Optional[ProjectPaths] = None, *, overwrite: bool = False) -> Path:
        """Save parameters as JSON inside an output folder (e.g. artifacts/)."""
        target = safe_output_path(path, paths, overwrite=overwrite)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: PathLike) -> "FeatureScaler":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = f"fitted on {self.n_samples_fit_} rows" if self.is_fitted else "not fitted"
        return f"FeatureScaler(method={self.method!r}, n_features={len(self.feature_columns)}, {state})"

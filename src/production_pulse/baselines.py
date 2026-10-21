"""Transparent baseline detectors, fitted on TRAINING-normal data only.

Workflow (each step uses a different, run-disjoint data set):

1. ``fit(train_normal)``                    - learn limits / PCA model
2. ``calibrate(calibration_normal, target_fpr)`` - choose the score threshold
3. ``detect(evaluation_data)``              - final, held-out evaluation

Every detector remembers row fingerprints of its fit and calibration data
and raises :class:`LeakageError` if calibration reuses fit rows, or if
evaluation data contains rows it has already seen. Inputs must be
DataFrames with named feature columns (metadata can never become a feature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS
from .evaluation import threshold_for_fpr
from .features import feature_matrix, overlap_count, row_fingerprints, validate_feature_columns
from .logging_config import get_logger

log = get_logger(__name__)


class BaselineError(ValueError):
    """Invalid baseline configuration or usage order."""


class LeakageError(BaselineError):
    """The same rows were used for more than one of fit / calibration / evaluation."""


@dataclass(frozen=True)
class Detection:
    """Output of a detector on one data set."""

    name: str
    scores: np.ndarray
    predictions: np.ndarray
    threshold: float
    threshold_source: str

    @property
    def n_flagged(self) -> int:
        return int(self.predictions.sum())


class _BaseDetector:
    name = "detector"

    def __init__(self, feature_columns: Sequence[str] = FEATURE_COLUMNS) -> None:
        self.feature_columns = validate_feature_columns(feature_columns)
        self.threshold_: Optional[float] = None
        self.threshold_source_: str = ""
        self.n_fit_rows_: int = 0
        self.n_calibration_rows_: int = 0
        self._fit_fp: Optional[np.ndarray] = None
        self._cal_fp: Optional[np.ndarray] = None

    # -- subclasses implement these
    def _fit(self, x: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _score(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _default_threshold(self) -> Optional[tuple[float, str]]:
        return None

    # -- shared logic
    @property
    def is_fitted(self) -> bool:
        return self._fit_fp is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise BaselineError(f"{self.name}: call fit() on training-normal data first")

    def _matrix(self, df: Any, what: str) -> np.ndarray:
        return feature_matrix(df, self.feature_columns, what=what)

    def fit(self, train_normal: pd.DataFrame, *, refit: bool = False) -> "_BaseDetector":
        """Fit on training-normal data. Resets any previous calibration."""
        if self.is_fitted and not refit:
            raise BaselineError(f"{self.name} is already fitted; pass refit=True to re-fit on new TRAINING data")
        x = self._matrix(train_normal, "train_normal")
        self._fit(x)
        self._fit_fp = row_fingerprints(x)
        self._cal_fp = None
        self.n_fit_rows_ = int(x.shape[0])
        self.n_calibration_rows_ = 0
        default = self._default_threshold()
        self.threshold_, self.threshold_source_ = (default if default else (None, ""))
        log.info("%s fitted on %d training-normal rows", self.name, self.n_fit_rows_)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row (higher = more anomalous)."""
        self._require_fitted()
        return self._score(self._matrix(df, "input"))

    def calibrate(self, calibration_normal: pd.DataFrame, *, target_fpr: float) -> float:
        """Set the threshold so at most ``target_fpr`` of calibration-normal rows are flagged.

        The calibration rows must not overlap the fit rows.
        """
        self._require_fitted()
        x = self._matrix(calibration_normal, "calibration_normal")
        fp = row_fingerprints(x)
        shared = overlap_count(fp, self._fit_fp)
        if shared:
            raise LeakageError(f"{self.name}: {shared} calibration row(s) were also used for fitting")
        self.threshold_ = threshold_for_fpr(self._score(x), target_fpr)
        self.threshold_source_ = f"calibrated: target FPR {target_fpr} on {x.shape[0]} calibration-normal rows"
        self._cal_fp = fp
        self.n_calibration_rows_ = int(x.shape[0])
        log.info("%s threshold %.6g (%s)", self.name, self.threshold_, self.threshold_source_)
        return self.threshold_

    def set_threshold(self, value: float, *, source: str = "manual") -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise BaselineError("threshold must be a finite number")
        self.threshold_, self.threshold_source_ = float(value), source

    def assert_unseen(self, df: pd.DataFrame) -> None:
        """Raise :class:`LeakageError` if ``df`` shares rows with fit or calibration data."""
        self._require_fitted()
        fp = row_fingerprints(self._matrix(df, "evaluation data"))
        for label, ref in (("fitting", self._fit_fp), ("calibration", self._cal_fp)):
            if ref is not None:
                shared = overlap_count(fp, ref)
                if shared:
                    raise LeakageError(f"{self.name}: {shared} evaluation row(s) were used for {label}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.detect(df, check_unseen=False).predictions

    def detect(self, df: pd.DataFrame, *, check_unseen: bool = True) -> Detection:
        """Score and threshold ``df``. By default refuses rows seen during fit/calibration."""
        self._require_fitted()
        if self.threshold_ is None:
            raise BaselineError(f"{self.name}: no threshold yet; call calibrate() on calibration-normal data")
        if check_unseen:
            self.assert_unseen(df)
        scores = self.score(df)
        return Detection(self.name, scores, scores > self.threshold_, float(self.threshold_), self.threshold_source_)

    def threshold_info(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "threshold": self.threshold_,
            "threshold_source": self.threshold_source_,
            "n_fit_rows": self.n_fit_rows_,
            "n_calibration_rows": self.n_calibration_rows_,
        }


class SingleFeatureThresholdDetector(_BaseDetector):
    """Rule check: flag a row if ANY feature lies outside its own training limits.

    Limits per feature are training-normal quantiles ``[lower_quantile, upper_quantile]``.
    Score = max over features of ``|x - centre| / half_width`` (``> 1`` means outside a
    limit), so the default rule threshold is exactly 1.0. ``calibrate`` may instead
    pick the threshold for a target false-positive rate.
    """

    name = "single_feature_threshold"

    def __init__(
        self,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        *,
        lower_quantile: float = 0.0005,
        upper_quantile: float = 0.9995,
    ) -> None:
        super().__init__(feature_columns)
        if not (0.0 <= lower_quantile < 0.5 < upper_quantile <= 1.0):
            raise BaselineError("need 0 <= lower_quantile < 0.5 < upper_quantile <= 1")
        self.lower_quantile, self.upper_quantile = float(lower_quantile), float(upper_quantile)
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def _fit(self, x: np.ndarray) -> None:
        self.lower_ = np.quantile(x, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(x, self.upper_quantile, axis=0)
        self._centre = (self.lower_ + self.upper_) / 2.0
        self._half = np.maximum((self.upper_ - self.lower_) / 2.0, 1e-12)

    def _default_threshold(self) -> tuple[float, str]:
        return 1.0, f"rule: outside training quantile limits [{self.lower_quantile}, {self.upper_quantile}]"

    def _score(self, x: np.ndarray) -> np.ndarray:
        return np.max(np.abs(x - self._centre) / self._half, axis=1)

    def limits(self) -> pd.DataFrame:
        self._require_fitted()
        return pd.DataFrame({"lower": self.lower_, "upper": self.upper_}, index=list(self.feature_columns))

    def exceedances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Boolean table: which feature broke its limit on which row (for transparency)."""
        self._require_fitted()
        x = self._matrix(df, "input")
        out = (x < self.lower_) | (x > self.upper_)
        return pd.DataFrame(out, index=df.index, columns=list(self.feature_columns))


class PCAReconstructionDetector(_BaseDetector):
    """PCA fitted on training-normal data; score = squared reconstruction error (SPE / Q).

    ``n_components``: an int (number of components, must be < n_features) or a
    float in (0, 1) = fraction of training variance to keep. Fitting is
    deterministic (SVD with a fixed sign convention); optional subsampling via
    ``max_fit_rows`` is seeded by ``seed``.
    """

    name = "pca_reconstruction"

    def __init__(
        self,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        *,
        n_components: Union[int, float] = 0.90,
        max_fit_rows: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(feature_columns)
        p = len(self.feature_columns)
        if isinstance(n_components, bool):
            raise BaselineError("n_components must be an int or a float fraction")
        if isinstance(n_components, (int, np.integer)):
            if not 1 <= n_components < p:
                raise BaselineError(f"integer n_components must be in [1, {p - 1}]")
        elif isinstance(n_components, float):
            if not 0.0 < n_components < 1.0:
                raise BaselineError("fractional n_components must be in (0, 1)")
        else:
            raise BaselineError("n_components must be an int or a float fraction")
        if max_fit_rows is not None and (isinstance(max_fit_rows, bool) or not isinstance(max_fit_rows, int) or max_fit_rows < 2):
            raise BaselineError("max_fit_rows must be None or an int >= 2")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise BaselineError("seed must be a non-negative integer")
        self.n_components = n_components
        self.max_fit_rows = max_fit_rows
        self.seed = int(seed)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_: int = 0

    def _fit(self, x: np.ndarray) -> None:
        if self.max_fit_rows is not None and x.shape[0] > self.max_fit_rows:
            idx = np.sort(np.random.default_rng(self.seed).choice(x.shape[0], self.max_fit_rows, replace=False))
            x = x[idx]
        if x.shape[0] < 2:
            raise BaselineError("PCA needs at least 2 training rows")
        self.mean_ = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        var = s**2
        ratio = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
        if isinstance(self.n_components, float):
            k = int(np.searchsorted(np.cumsum(ratio), self.n_components) + 1)
            k = min(k, len(self.feature_columns) - 1)
        else:
            k = int(self.n_components)
        if k > vt.shape[0]:
            raise BaselineError(f"Requested {k} components but data supports only {vt.shape[0]}")
        comps = vt[:k].copy()
        signs = np.sign(comps[np.arange(k), np.argmax(np.abs(comps), axis=1)])
        comps *= np.where(signs == 0, 1.0, signs)[:, None]  # deterministic sign convention
        self.components_ = comps
        self.explained_variance_ratio_ = ratio[:k]
        self.n_components_ = k

    def _score(self, x: np.ndarray) -> np.ndarray:
        centred = x - self.mean_
        recon = centred @ self.components_.T @ self.components_
        return np.sum((centred - recon) ** 2, axis=1)

    def reconstruction_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature squared error (which features drive the score)."""
        self._require_fitted()
        centred = self._matrix(df, "input") - self.mean_
        err = (centred - centred @ self.components_.T @ self.components_) ** 2
        return pd.DataFrame(err, index=df.index, columns=list(self.feature_columns))

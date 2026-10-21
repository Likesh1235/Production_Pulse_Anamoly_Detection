"""Model lifecycle and prediction logic (no HTTP here).

Lifecycle
---------
* :meth:`ModelService.load` builds a NEW immutable :class:`ModelState` and swaps it in
  atomically. Requests take a snapshot of the state at their start, so a reload can
  never change the model in the middle of a request. Reloads happen only at startup
  or through the admin-protected ``POST /model/reload`` endpoint - never silently.
* A model is ``ready`` only if it loaded, its feature schema equals the canonical
  52-feature order, a fitted scaler with the same feature order is attached, and a
  finite calibrated threshold exists. Otherwise predictions are refused (HTTP 503).
* Detailed load errors (which may contain file paths) go to the server log only;
  clients see a fixed, path-free reason per status.

Detector interface used (as implemented in ``production_pulse.autoencoder``):
``feature_columns``, ``scaler`` (FeatureScaler), ``threshold_``, ``threshold_source_``,
``scaler.transform_frame(df)`` and ``detect_with_errors(scaled_df, check_unseen=False)`` (one network pass).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..data import FEATURE_COLUMNS
from ..logging_config import get_logger
from ..paths import ProjectPaths

log = get_logger("api.service")

STATUS_REASONS = {
    "ready": None,
    "not_loaded": "Model loading is disabled for this API instance.",
    "not_found": "No trained model was found in the configured model directory. "
                 "Train and save one with notebooks/03_autoencoder_training.ipynb.",
    "load_failed": "The model artifact could not be loaded or failed validation. See the server log for details.",
    "dependency_missing": "TensorFlow/Keras is not available in this Python environment.",
    "incompatible": "The model's feature schema or preprocessing does not match the API's 52-feature schema.",
    "no_preprocessing": "The model artifact has no fitted scaler, so raw sensor values cannot be scored.",
    "not_calibrated": "The model has no calibrated threshold; calibrate and save it before serving predictions.",
}
# metadata.json keys that may be shown to clients (only if present in the file)
PUBLIC_METADATA_KEYS = (
    "model_type", "format_version", "created_utc", "score", "seed", "settings", "n_fit_rows",
    "n_validation_rows", "n_calibration_rows", "training_summary", "library_versions", "model_sha256",
)


class InputOutOfRangeError(ValueError):
    """Finite inputs that are too large for the float32 network, or that make the score non-finite."""

    def __init__(self, details: list[dict[str, Any]]) -> None:
        super().__init__("input values out of the model's numeric range")
        self.details = details


_FLOAT32_MAX = float(np.finfo(np.float32).max)


class ModelUnavailableError(RuntimeError):
    def __init__(self, state: "ModelState") -> None:
        super().__init__(state.reason or state.status)
        self.state = state


@dataclass(frozen=True)
class ModelState:
    status: str
    model_dir: str                               # relative, as configured (no absolute paths exposed)
    detector: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_utc: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.detector is not None

    @property
    def reason(self) -> Optional[str]:
        return STATUS_REASONS.get(self.status)

    @property
    def model_id(self) -> Optional[str]:
        sha = self.metadata.get("model_sha256")
        return sha[:12] if isinstance(sha, str) and sha else None


@dataclass(frozen=True)
class SampleOutcome:
    reconstruction_error: float
    is_anomaly: bool
    top_features: list[dict[str, Any]]


@dataclass(frozen=True)
class BatchOutcome:
    threshold: float
    threshold_source: str
    model_id: Optional[str]
    samples: list[SampleOutcome]


def _default_loader(directory: Path, *, paths: ProjectPaths, expected_feature_columns):
    from ..autoencoder import AutoencoderDetector  # lazy: imports TensorFlow only when needed

    return AutoencoderDetector.load(directory, paths=paths, expected_feature_columns=expected_feature_columns)


def _public_metadata(model_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {k: raw[k] for k in PUBLIC_METADATA_KEYS if raw.get(k) is not None}
    scaler = raw.get("scaler")
    if isinstance(scaler, dict):  # describe preprocessing without dumping its parameters
        out["preprocessing"] = {k: scaler[k] for k in ("method", "fit_source", "n_samples_fit") if k in scaler}
    return out


class ModelService:
    """Holds the current model state and performs predictions."""

    def __init__(
        self,
        model_dir: str,
        paths: ProjectPaths,
        *,
        loader: Optional[Callable[..., Any]] = None,
        top_features: int = 5,
    ) -> None:
        self.model_dir = model_dir
        self.paths = paths
        self.loader = loader or _default_loader
        self.top_features = int(top_features)
        self._state = ModelState("not_loaded", model_dir)
        self._swap_lock = threading.Lock()      # serialises reloads
        self._infer_lock = threading.Lock()     # serialises Keras inference calls

    @property
    def state(self) -> ModelState:
        return self._state  # reading a reference is atomic: callers get a consistent snapshot

    # ------------------------------------------------------------------ lifecycle
    def _build_state(self) -> ModelState:
        model_path = self.paths.root / self.model_dir
        if not (model_path / "metadata.json").is_file():
            log.warning("No model artifact at configured model_dir=%s", self.model_dir)
            return ModelState("not_found", self.model_dir)
        try:
            detector = self.loader(model_path, paths=self.paths, expected_feature_columns=FEATURE_COLUMNS)
        except ImportError as exc:
            log.error("Model dependencies unavailable: %s", exc)
            return ModelState("dependency_missing", self.model_dir)
        except Exception as exc:  # validation errors, corrupt files, ...
            log.error("Model loading failed: %s: %s", type(exc).__name__, exc)
            return ModelState("load_failed", self.model_dir)

        metadata = _public_metadata(model_path)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if tuple(getattr(detector, "feature_columns", ())) != tuple(FEATURE_COLUMNS):
            return ModelState("incompatible", self.model_dir, None, metadata, now)
        scaler = getattr(detector, "scaler", None)
        if scaler is None or not getattr(scaler, "is_fitted", False):
            return ModelState("no_preprocessing", self.model_dir, None, metadata, now)
        if tuple(scaler.feature_columns) != tuple(FEATURE_COLUMNS):
            return ModelState("incompatible", self.model_dir, None, metadata, now)
        threshold = getattr(detector, "threshold_", None)
        if threshold is None or isinstance(threshold, bool) or not math.isfinite(float(threshold)):
            return ModelState("not_calibrated", self.model_dir, None, metadata, now)
        metadata["threshold"] = float(threshold)
        metadata["threshold_source"] = str(getattr(detector, "threshold_source_", ""))
        return ModelState("ready", self.model_dir, detector, metadata, now)

    def load(self) -> ModelState:
        """Load (or reload) the model; the new state replaces the old one atomically."""
        with self._swap_lock:
            new_state = self._build_state()
            self._state = new_state
        log.info("Model status: %s (model_dir=%s)", new_state.status, self.model_dir)
        return new_state

    # ------------------------------------------------------------------ inference
    def predict(self, frame: pd.DataFrame) -> BatchOutcome:
        """Score raw samples whose columns are exactly the canonical feature order."""
        state = self.state  # snapshot for the whole request
        if not state.ready:
            raise ModelUnavailableError(state)
        det = state.detector
        names = list(FEATURE_COLUMNS)
        with self._infer_lock:
            scaled = det.scaler.transform_frame(frame)               # the training scaler stored in the artifact
            z = scaled.to_numpy(dtype=np.float64)
            too_big = ~np.isfinite(z) | (np.abs(z) > _FLOAT32_MAX)   # would overflow the float32 network input
            if too_big.any():
                raise InputOutOfRangeError(_out_of_range_details(too_big, names, "scaled value exceeds float32 range"))
            with np.errstate(over="ignore", invalid="ignore"):
                detection, error_table = det.detect_with_errors(scaled, check_unseen=False)   # ONE network pass
            errors = error_table.to_numpy(dtype=np.float64)
        bad = ~np.isfinite(errors) | ~np.isfinite(np.asarray(detection.scores, dtype=np.float64))[:, None]
        if bad.any():   # finite inputs, but the network or the squared error overflowed
            raise InputOutOfRangeError(_out_of_range_details(bad, names, "reconstruction error is not finite"))
        k = min(self.top_features, len(names))
        samples = []
        for i in range(len(frame)):
            row = errors[i]
            total = float(row.sum())
            top = np.argsort(-row, kind="mergesort")[:k]
            samples.append(
                SampleOutcome(
                    reconstruction_error=float(detection.scores[i]),
                    is_anomaly=bool(detection.predictions[i]),
                    top_features=[
                        {
                            "feature": names[j],
                            "squared_error": float(row[j]),
                            "share": float(row[j] / total) if total > 0 else 0.0,
                        }
                        for j in top
                    ],
                )
            )
        return BatchOutcome(float(detection.threshold), str(detection.threshold_source), state.model_id, samples)


def _out_of_range_details(mask: np.ndarray, names: list[str], reason: str) -> list[dict[str, Any]]:
    out = []
    for i in np.flatnonzero(mask.any(axis=1))[:20]:
        out.append({"sample_index": int(i), "features": [names[j] for j in np.flatnonzero(mask[i])][:10], "reason": reason})
    return out

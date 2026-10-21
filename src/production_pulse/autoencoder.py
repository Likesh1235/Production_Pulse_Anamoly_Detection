"""Dense autoencoder anomaly detector for the 52 TEP measurement features.

The detector follows the same interface as the Phase 3 baselines
(:mod:`production_pulse.baselines`), so it works directly with
``compare_detectors`` and ``assess_family_detectability``:

1. ``fit(train_normal, validation_normal)`` - trains on NORMAL training rows only;
   the separate validation rows drive early stopping (``restore_best_weights``).
2. ``calibrate(calibration_normal, target_fpr=...)`` - threshold on a third,
   disjoint data set (rows used for fitting/validation are refused).
3. ``detect(evaluation_data)`` - final scoring; rows already seen are refused.

Score = mean squared reconstruction error over the 52 features (per sample).
Inputs are SCALED feature frames (see :class:`FeatureScaler`); with a scaler
attached, ``detect_raw`` accepts raw sensor values.

TensorFlow/Keras are imported lazily, so the rest of the package (and the
tests that do not need TensorFlow) works without them.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import __version__
from .baselines import BaselineError, Detection, LeakageError, _BaseDetector
from .config import AutoencoderSettings
from .data import FEATURE_COLUMNS
from .features import overlap_count, row_fingerprints
from .logging_config import get_logger
from .paths import ProjectPaths, get_paths, safe_output_path
from .preprocessing import FeatureScaler

log = get_logger(__name__)

PathLike = Union[str, os.PathLike]
MODEL_FORMAT_VERSION = 1
MODEL_TYPE = "dense_autoencoder"
MODEL_FILE, METADATA_FILE, HISTORY_FILE, FINGERPRINT_FILE = "model.keras", "metadata.json", "history.csv", "fingerprints.npz"
MANIFEST_FILE = "manifest.json"
MANIFEST_FORMAT_VERSION = 1
_MANIFEST_COVERS = (MODEL_FILE, METADATA_FILE, HISTORY_FILE, FINGERPRINT_FILE)


class AutoencoderError(BaselineError):
    """Invalid autoencoder configuration, usage, training failure or artifact."""


def _import_keras():
    try:
        import tensorflow as tf
        import keras
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "TensorFlow/Keras are required for the autoencoder. They are installed in .venv313; "
            "run with .venv313\\Scripts\\python.exe."
        ) from exc
    return tf, keras


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow; optionally request deterministic TF ops.

    Note: enabling op determinism is process-wide and cannot be switched off again.
    """
    tf, keras = _import_keras()
    keras.utils.set_random_seed(int(seed))
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover - backend specific
            log.warning("Could not enable deterministic TensorFlow ops: %s", exc)


def build_autoencoder(n_features: int, settings: AutoencoderSettings, *, seed: int = 0):
    """Build and compile the Keras model (encoder mirrors decoder)."""
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 2:
        raise AutoencoderError("n_features must be an integer >= 2")
    if settings.bottleneck >= n_features:
        raise AutoencoderError(f"bottleneck ({settings.bottleneck}) must be smaller than n_features ({n_features})")
    if settings.hidden_layers and settings.hidden_layers[0] > n_features * 4:
        raise AutoencoderError("first hidden layer is implausibly wide (> 4 x n_features)")
    _, keras = _import_keras()
    reg = keras.regularizers.L2(settings.l2) if settings.l2 > 0 else None

    def dense(units: int, activation: str, name: str, k: int):
        return keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + k),
            kernel_regularizer=reg,
            name=name,
        )

    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    if settings.noise_std > 0:
        x = keras.layers.GaussianNoise(settings.noise_std, seed=seed + 1000, name="input_noise")(x)
    layer_no = 0  # each Dense layer gets its own initializer seed (seed + layer number)
    for i, width in enumerate(settings.hidden_layers, start=1):
        layer_no += 1
        x = dense(width, settings.activation, f"encoder_{i}", layer_no)(x)
        if settings.dropout > 0:
            x = keras.layers.Dropout(settings.dropout, seed=seed + 2000 + i, name=f"encoder_dropout_{i}")(x)
    layer_no += 1
    x = dense(settings.bottleneck, settings.activation, "bottleneck", layer_no)(x)
    for i, width in enumerate(reversed(settings.hidden_layers), start=1):
        layer_no += 1
        x = dense(width, settings.activation, f"decoder_{i}", layer_no)(x)
        if settings.dropout > 0:
            x = keras.layers.Dropout(settings.dropout, seed=seed + 3000 + i, name=f"decoder_dropout_{i}")(x)
    layer_no += 1
    outputs = dense(n_features, settings.output_activation, "reconstruction", layer_no)(x)
    model = keras.Model(inputs, outputs, name="production_pulse_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=settings.learning_rate), loss="mse")
    return model


def _best_weights_callback(keras):
    """Always restore the weights of the epoch with the lowest validation loss.

    Keras' ``EarlyStopping(restore_best_weights=True)`` only restores them when
    early stopping actually triggers; if training reaches ``epochs`` the LAST
    weights are kept. This callback guarantees the model matches ``best_epoch``.
    """

    class BestValidationWeights(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = np.inf
            self.best_epoch = 0
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get("val_loss")
            if value is not None and np.isfinite(value) and value < self.best:
                self.best, self.best_epoch = float(value), epoch + 1
                self.best_weights = self.model.get_weights()

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    return BestValidationWeights()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _library_versions() -> dict[str, str]:
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "production_pulse": __version__}
    try:
        tf, keras = _import_keras()
        versions.update({"tensorflow": tf.__version__, "keras": keras.__version__})
    except ImportError:  # pragma: no cover
        pass
    return versions


def write_manifest(directory: PathLike, paths: Optional[ProjectPaths] = None, *, overwrite: bool = False) -> Path:
    """Write ``manifest.json``: SHA-256 of every artifact file (model, metadata, history, fingerprints).

    The manifest detects accidental damage to the saved artifact, for example a partly synced
    OneDrive file or a hand-edited threshold or scaler in ``metadata.json``. It is NOT a
    security feature: anyone who can edit the files can also rewrite the manifest.
    """
    d = Path(directory)
    if not d.is_absolute():
        d = (paths or get_paths()).root / d
    missing = [name for name in (MODEL_FILE, METADATA_FILE) if not (d / name).is_file()]
    if missing:
        raise AutoencoderError(f"Cannot write a manifest: {missing} missing in {d}")
    target = safe_output_path(d / MANIFEST_FILE, paths, overwrite=overwrite)
    files = {name: _sha256(d / name) for name in _MANIFEST_COVERS if (d / name).is_file()}
    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "algorithm": "sha256",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Detects accidental damage or edits to the saved artifact; not a security signature.",
        "files": files,
    }
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def verify_manifest(directory: PathLike) -> str:
    """Check the artifact against ``manifest.json``.

    Returns ``"verified"`` or ``"absent"`` (older artifact saved before manifests existed).
    Raises :class:`AutoencoderError` on an unreadable manifest, a missing file or a hash mismatch.
    """
    d = Path(directory)
    path = d / MANIFEST_FILE
    if not path.is_file():
        return "absent"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        files = dict(manifest["files"])
        if manifest.get("format_version") != MANIFEST_FORMAT_VERSION or manifest.get("algorithm") != "sha256":
            raise ValueError("unsupported manifest format")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AutoencoderError(f"{MANIFEST_FILE} is unreadable or unsupported: {exc}") from exc
    for name, expected in files.items():
        if name not in _MANIFEST_COVERS:
            raise AutoencoderError(f"{MANIFEST_FILE} lists an unexpected file {name!r}")
        if not (d / name).is_file():
            raise AutoencoderError(f"Artifact file {name} listed in the manifest is missing")
        if _sha256(d / name) != expected:
            raise AutoencoderError(f"Artifact file {name} does not match {MANIFEST_FILE} (file changed or damaged)")
    if METADATA_FILE not in files or MODEL_FILE not in files:
        raise AutoencoderError(f"{MANIFEST_FILE} must cover {MODEL_FILE} and {METADATA_FILE}")
    return "verified"


class AutoencoderDetector(_BaseDetector):
    """Reconstruction-error detector backed by a dense Keras autoencoder."""

    name = "autoencoder"

    def __init__(
        self,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        *,
        settings: Optional[AutoencoderSettings] = None,
        seed: int = 42,
        scaler: Optional[FeatureScaler] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(feature_columns)
        if settings is not None and not isinstance(settings, AutoencoderSettings):
            raise AutoencoderError("settings must be an AutoencoderSettings instance")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise AutoencoderError("seed must be a non-negative integer")
        self.settings = settings or AutoencoderSettings()
        self.seed = int(seed)
        self.verbose = int(verbose)
        self.model_ = None
        self.history_: Optional[pd.DataFrame] = None
        self.training_summary_: dict[str, Any] = {}
        self.n_validation_rows_ = 0
        self.scaler: Optional[FeatureScaler] = None
        if scaler is not None:
            self.attach_scaler(scaler)

    # ------------------------------------------------------------------ scaler
    def attach_scaler(self, scaler: FeatureScaler) -> None:
        """Attach the TRAINING-fitted scaler so raw sensor values can be scored."""
        if not isinstance(scaler, FeatureScaler) or not scaler.is_fitted:
            raise AutoencoderError("scaler must be a fitted FeatureScaler")
        if tuple(scaler.feature_columns) != tuple(self.feature_columns):
            raise AutoencoderError("scaler feature columns differ from the detector's feature columns (order matters)")
        self.scaler = scaler

    # ------------------------------------------------------------------ training
    def _fit(self, x: np.ndarray) -> None:  # pragma: no cover - fit() is overridden
        raise AutoencoderError("use fit(train_normal, validation_normal)")

    def fit(  # type: ignore[override]
        self,
        train_normal: pd.DataFrame,
        validation_normal: Optional[pd.DataFrame] = None,
        *,
        refit: bool = False,
    ) -> "AutoencoderDetector":
        """Train on normal TRAINING rows; early-stop on the separate VALIDATION rows."""
        if validation_normal is None:
            raise AutoencoderError("A separate validation_normal DataFrame is required for early stopping")
        if self.is_fitted and not refit:
            raise AutoencoderError("Autoencoder is already fitted; pass refit=True to retrain on new TRAINING data")
        x_tr = self._matrix(train_normal, "train_normal")
        x_val = self._matrix(validation_normal, "validation_normal")
        if x_tr.shape[0] < 2 or x_val.shape[0] < 1:
            raise AutoencoderError("Need at least 2 training rows and 1 validation row")
        fp_tr, fp_val = row_fingerprints(x_tr), row_fingerprints(x_val)
        shared = overlap_count(fp_tr, fp_val)
        if shared:
            raise LeakageError(f"{shared} validation row(s) are also in the training data")

        _, keras = _import_keras()
        set_global_seed(self.seed, deterministic=self.settings.deterministic)
        model = build_autoencoder(x_tr.shape[1], self.settings, seed=self.seed)
        best_weights = _best_weights_callback(keras)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.settings.patience,
                min_delta=self.settings.min_delta,
                mode="min",
                restore_best_weights=False,  # restoring is done by best_weights (always, see above)
            ),
            keras.callbacks.TerminateOnNaN(),
            best_weights,
        ]
        xt, xv = x_tr.astype(np.float32), x_val.astype(np.float32)
        hist = model.fit(
            xt, xt,
            validation_data=(xv, xv),
            epochs=self.settings.epochs,
            batch_size=self.settings.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        history = pd.DataFrame(hist.history)
        history.index = pd.RangeIndex(1, len(history) + 1, name="epoch")
        if history.empty or not np.isfinite(history[["loss", "val_loss"]].to_numpy()).all():
            raise AutoencoderError("Training produced non-finite losses (diverged); lower learning_rate or check inputs")

        best_epoch = int(history["val_loss"].idxmin())
        if best_weights.best_epoch != best_epoch:  # pragma: no cover - defensive consistency check
            raise AutoencoderError("Internal error: restored weights do not match the best validation epoch")
        self.model_ = model
        self.history_ = history
        self.training_summary_ = {
            "epochs_run": int(len(history)),
            "max_epochs": self.settings.epochs,
            "stopped_early": bool(len(history) < self.settings.epochs),
            "best_epoch": best_epoch,
            "best_val_loss": float(history["val_loss"].min()),
            "final_train_loss": float(history["loss"].iloc[-1]),
            "n_parameters": int(model.count_params()),
            "seed": self.seed,
        }
        # bookkeeping shared with the baselines (leakage guards, threshold reset)
        self._fit_fp = np.union1d(fp_tr, fp_val)
        self._cal_fp = None
        self.n_fit_rows_ = int(x_tr.shape[0])
        self.n_validation_rows_ = int(x_val.shape[0])
        self.n_calibration_rows_ = 0
        self.threshold_, self.threshold_source_ = None, ""
        log.info("Autoencoder trained: %s", self.training_summary_)
        return self

    # ------------------------------------------------------------------ scoring
    def _require_model(self) -> None:
        self._require_fitted()
        if self.model_ is None:
            raise AutoencoderError("No Keras model is loaded")

    def _reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._require_model()
        out = self.model_.predict(x.astype(np.float32), batch_size=max(self.settings.batch_size, 1024), verbose=0)
        return np.asarray(out, dtype=np.float64)

    def _score(self, x: np.ndarray) -> np.ndarray:
        return np.mean((x - self._reconstruct(x)) ** 2, axis=1)

    def reconstruct(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._matrix(df, "input")
        return pd.DataFrame(self._reconstruct(x), index=df.index, columns=list(self.feature_columns))

    def reconstruction_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature squared error; its row mean equals :meth:`score`."""
        x = self._matrix(df, "input")
        return pd.DataFrame((x - self._reconstruct(x)) ** 2, index=df.index, columns=list(self.feature_columns))

    def _scale_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise AutoencoderError("No scaler attached; use attach_scaler() or pass scaled features to detect()")
        return self.scaler.transform_frame(df_raw)

    def score_raw(self, df_raw: pd.DataFrame) -> np.ndarray:
        return self.score(self._scale_raw(df_raw))

    def detect_raw(self, df_raw: pd.DataFrame, *, check_unseen: bool = True) -> Detection:
        """Score RAW (unscaled) sensor values using the attached training scaler."""
        return self.detect(self._scale_raw(df_raw), check_unseen=check_unseen)

    def detect_with_errors(self, df: pd.DataFrame, *, check_unseen: bool = True) -> tuple[Detection, pd.DataFrame]:
        """Like :meth:`detect` on SCALED data, but runs the network ONCE and also returns the
        per-feature squared errors. ``scores`` are the row means of those errors, which is exactly
        what :meth:`score` computes, so the decision is identical to ``detect``.
        """
        self._require_model()
        if self.threshold_ is None:
            raise BaselineError(f"{self.name}: no threshold yet; call calibrate() on calibration-normal data")
        if check_unseen:
            self.assert_unseen(df)
        x = self._matrix(df, "input")
        squared = (x - self._reconstruct(x)) ** 2
        scores = np.mean(squared, axis=1)
        detection = Detection(self.name, scores, scores > self.threshold_, float(self.threshold_), self.threshold_source_)
        return detection, pd.DataFrame(squared, index=df.index, columns=list(self.feature_columns))

    # ------------------------------------------------------------------ persistence
    def save(self, directory: PathLike, paths: Optional[ProjectPaths] = None, *, overwrite: bool = False) -> dict[str, Path]:
        """Save model, metadata (incl. scaler + threshold), history and leakage fingerprints.

        ``directory`` must be inside a project output folder, e.g. ``artifacts/models/ae_v1``.
        Metadata is written last, so an interrupted save is detected on load.
        """
        self._require_model()
        directory = Path(directory)
        targets = {
            "model": safe_output_path(directory / MODEL_FILE, paths, overwrite=overwrite),
            "history": safe_output_path(directory / HISTORY_FILE, paths, overwrite=overwrite),
            "fingerprints": safe_output_path(directory / FINGERPRINT_FILE, paths, overwrite=overwrite),
            "metadata": safe_output_path(directory / METADATA_FILE, paths, overwrite=overwrite),
        }
        self.model_.save(str(targets["model"]))
        (self.history_ if self.history_ is not None else pd.DataFrame()).to_csv(targets["history"])
        empty = np.array([], dtype=np.uint64)
        np.savez(
            targets["fingerprints"],
            fit=self._fit_fp if self._fit_fp is not None else empty,
            calibration=self._cal_fp if self._cal_fp is not None else empty,
        )
        if self.scaler is None:
            log.warning("Saving without a scaler: raw-value inference will not be possible after loading")
        metadata = {
            "format_version": MODEL_FORMAT_VERSION,
            "model_type": MODEL_TYPE,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "feature_columns": list(self.feature_columns),
            "n_features": len(self.feature_columns),
            "score": "mean squared reconstruction error over features",
            "settings": self.settings.to_dict(),
            "seed": self.seed,
            "threshold": self.threshold_,
            "threshold_source": self.threshold_source_,
            "n_fit_rows": self.n_fit_rows_,
            "n_validation_rows": self.n_validation_rows_,
            "n_calibration_rows": self.n_calibration_rows_,
            "training_summary": self.training_summary_,
            "scaler": self.scaler.to_dict() if self.scaler is not None else None,
            "model_sha256": _sha256(targets["model"]),
            "library_versions": _library_versions(),
        }
        targets["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        targets["manifest"] = write_manifest(targets["model"].parent, paths, overwrite=overwrite)
        log.info("Saved autoencoder to %s", targets["model"].parent)
        return targets

    @classmethod
    def load(
        cls,
        directory: PathLike,
        *,
        paths: Optional[ProjectPaths] = None,
        expected_feature_columns: Optional[Sequence[str]] = FEATURE_COLUMNS,
        require_manifest: bool = False,
    ) -> "AutoencoderDetector":
        """Load a saved detector and validate its schema, feature order and file integrity.

        If ``manifest.json`` exists, every artifact file (including the threshold and scaler in
        ``metadata.json``) must match it. Artifacts saved before manifests existed still load, with a
        warning, unless ``require_manifest=True``. Create one with ``scripts/write_artifact_manifest.py``.
        """
        d = Path(directory)
        if not d.is_absolute():
            d = (paths or get_paths()).root / d
        meta_path = d / METADATA_FILE
        if not meta_path.is_file():
            raise AutoencoderError(f"No {METADATA_FILE} in {d} (missing or interrupted save)")
        integrity = verify_manifest(d)
        if integrity == "absent":
            if require_manifest:
                raise AutoencoderError(f"No {MANIFEST_FILE} in {d} and a manifest is required")
            log.warning("No %s in %s: metadata (threshold, scaler) is not integrity-checked", MANIFEST_FILE, d)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AutoencoderError(f"{meta_path} is not valid JSON: {exc}") from exc
        if meta.get("format_version") != MODEL_FORMAT_VERSION or meta.get("model_type") != MODEL_TYPE:
            raise AutoencoderError(
                f"Unsupported artifact (format_version={meta.get('format_version')}, model_type={meta.get('model_type')})"
            )
        cols = tuple(meta.get("feature_columns") or ())
        if expected_feature_columns is not None and cols != tuple(expected_feature_columns):
            missing = [c for c in expected_feature_columns if c not in cols]
            raise AutoencoderError(
                "Saved model feature schema does not match the expected feature order"
                + (f"; missing {missing[:5]}" if missing else " (same names, different order)")
            )
        model_path = d / MODEL_FILE
        if not model_path.is_file():
            raise AutoencoderError(f"Model file missing: {model_path}")
        if _sha256(model_path) != meta.get("model_sha256"):
            raise AutoencoderError("Model file checksum does not match metadata (file changed or mixed artifacts)")

        det = cls(cols, settings=AutoencoderSettings.from_dict(meta["settings"]), seed=int(meta.get("seed", 0)))
        _, keras = _import_keras()
        model = keras.models.load_model(str(model_path), compile=False)
        if tuple(model.input_shape)[-1] != len(cols) or tuple(model.output_shape)[-1] != len(cols):
            raise AutoencoderError(
                f"Model shape {model.input_shape}->{model.output_shape} does not match {len(cols)} features"
            )
        det.model_ = model
        if meta.get("scaler") is not None:
            det.attach_scaler(FeatureScaler.from_dict(meta["scaler"]))
        fp_path = d / FINGERPRINT_FILE
        if fp_path.is_file():
            with np.load(fp_path) as fps:
                det._fit_fp = fps["fit"].astype(np.uint64)
                det._cal_fp = fps["calibration"].astype(np.uint64) if fps["calibration"].size else None
        else:
            log.warning("No fingerprints found: leakage checks against training data are unavailable")
            det._fit_fp = np.array([], dtype=np.uint64)
        hist_path = d / HISTORY_FILE
        if hist_path.is_file():
            det.history_ = pd.read_csv(hist_path, index_col=0)
        det.threshold_ = meta.get("threshold")
        det.threshold_source_ = meta.get("threshold_source", "")
        det.n_fit_rows_ = int(meta.get("n_fit_rows", 0))
        det.n_validation_rows_ = int(meta.get("n_validation_rows", 0))
        det.n_calibration_rows_ = int(meta.get("n_calibration_rows", 0))
        det.training_summary_ = dict(meta.get("training_summary") or {})
        log.info("Loaded autoencoder from %s (threshold=%s)", d, det.threshold_)
        return det

    def summary(self) -> dict[str, Any]:
        """Human-readable state for notebooks/APIs."""
        return {
            "detector": self.name,
            "fitted": self.is_fitted,
            "settings": self.settings.to_dict(),
            "training": self.training_summary_,
            "n_validation_rows": self.n_validation_rows_,
            "has_scaler": self.scaler is not None,
            **self.threshold_info(),
        }

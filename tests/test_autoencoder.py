"""Autoencoder detector tests (need TensorFlow; skipped automatically without it).

Small synthetic correlated data and a tiny network keep these fast. They test
the implementation (training protocol, leakage guards, persistence, schema
checks) - not the model's effectiveness on TEP.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from production_pulse.anomaly_generation import AnomalyGenerator, AnomalySpec  # noqa: E402
from production_pulse.autoencoder import (  # noqa: E402
    AutoencoderDetector,
    AutoencoderError,
    build_autoencoder,
    verify_manifest,
    write_manifest,
)
from production_pulse.baselines import BaselineError, LeakageError  # noqa: E402
from production_pulse.config import AutoencoderSettings, default_config  # noqa: E402
from production_pulse.data import FEATURE_COLUMNS  # noqa: E402
from production_pulse.evaluation import assess_family_detectability  # noqa: E402
from production_pulse.paths import UnsafeOutputPathError, get_paths  # noqa: E402
from production_pulse.preprocessing import FeatureScaler  # noqa: E402

from conftest import make_correlated_frame  # noqa: E402

F = list(FEATURE_COLUMNS)
TINY = AutoencoderSettings(hidden_layers=(16,), bottleneck=4, epochs=8, patience=3, batch_size=64)


@pytest.fixture(scope="module")
def frames():
    raw = {name: make_correlated_frame(n, 50, seed=s) for name, n, s in
           (("train", 20, 1), ("validation", 5, 2), ("calibration", 6, 3), ("test", 6, 4))}
    scaler = FeatureScaler().fit(raw["train"])
    scaled = {k: scaler.transform_frame(v) for k, v in raw.items()}
    return raw, scaled, scaler


@pytest.fixture(scope="module")
def trained(frames):
    _, x, scaler = frames
    det = AutoencoderDetector(settings=TINY, seed=7, scaler=scaler).fit(x["train"], x["validation"])
    det.calibrate(x["calibration"], target_fpr=0.05)
    return det


# ------------------------------------------------------------------ architecture
def test_architecture_mirrors_settings():
    model = build_autoencoder(52, AutoencoderSettings(hidden_layers=(32, 16), bottleneck=8), seed=0)
    units = {layer.name: layer.units for layer in model.layers if hasattr(layer, "units")}
    assert units == {"encoder_1": 32, "encoder_2": 16, "bottleneck": 8, "decoder_1": 16, "decoder_2": 32, "reconstruction": 52}
    assert tuple(model.input_shape) == (None, 52) and tuple(model.output_shape) == (None, 52)


def test_optional_layers_are_added():
    s = AutoencoderSettings(hidden_layers=(16,), bottleneck=4, dropout=0.1, noise_std=0.05, l2=1e-4)
    names = [layer.name for layer in build_autoencoder(52, s).layers]
    assert "input_noise" in names and "encoder_dropout_1" in names and "decoder_dropout_1" in names


@pytest.mark.parametrize("n, s", [(1, TINY), (4, TINY), (True, TINY)])
def test_build_rejects_bad_shapes(n, s):
    with pytest.raises(AutoencoderError):
        build_autoencoder(n, s)


# ------------------------------------------------------------------ training protocol
def test_fit_requires_separate_validation(frames):
    _, x, _ = frames
    det = AutoencoderDetector(settings=TINY)
    with pytest.raises(AutoencoderError, match="validation"):
        det.fit(x["train"])
    with pytest.raises(LeakageError, match="validation row"):
        det.fit(x["train"], pd.concat([x["validation"], x["train"].iloc[:5]]))


def test_training_history_and_best_weights(trained, frames):
    _, x, _ = frames
    h, s = trained.history_, trained.training_summary_
    assert list(h.columns[:2]) == ["loss", "val_loss"] and h.index[0] == 1
    assert 1 <= s["epochs_run"] <= TINY.epochs
    assert s["best_val_loss"] == pytest.approx(h["val_loss"].min())
    restored = trained.model_.evaluate(x["validation"].to_numpy(np.float32), x["validation"].to_numpy(np.float32), verbose=0)
    assert restored == pytest.approx(s["best_val_loss"], rel=1e-3)     # best weights restored
    assert trained.n_fit_rows_ == len(x["train"]) and trained.n_validation_rows_ == len(x["validation"])


def test_best_weights_restored_even_without_early_stop(frames):
    """Keras only restores best weights when EarlyStopping triggers - ours always does."""
    _, x, _ = frames
    s = AutoencoderSettings(hidden_layers=(16,), bottleneck=4, epochs=6, patience=100, batch_size=64, learning_rate=0.05)
    det = AutoencoderDetector(settings=s, seed=2).fit(x["train"], x["validation"])
    assert not det.training_summary_["stopped_early"]
    xv = x["validation"].to_numpy(np.float32)
    assert det.model_.evaluate(xv, xv, verbose=0) == pytest.approx(det.history_["val_loss"].min(), rel=1e-3)
    assert det.n_fit_rows_ == len(x["train"]) and det.n_validation_rows_ == len(x["validation"])


def test_early_stopping_triggers(frames):
    _, x, _ = frames
    s = AutoencoderSettings(hidden_layers=(16,), bottleneck=4, epochs=50, patience=0, min_delta=1e9, batch_size=64)
    det = AutoencoderDetector(settings=s, seed=1).fit(x["train"], x["validation"])
    assert det.training_summary_["stopped_early"] and det.training_summary_["epochs_run"] == 2


def test_refit_is_explicit(trained, frames):
    _, x, _ = frames
    with pytest.raises(AutoencoderError, match="already fitted"):
        trained.fit(x["train"], x["validation"])


# ------------------------------------------------------------------ scoring
def test_scores_and_per_feature_errors(trained, frames):
    _, x, _ = frames
    scores = trained.score(x["test"])
    errors = trained.reconstruction_errors(x["test"])
    assert scores.shape == (len(x["test"]),) and (scores >= 0).all()
    np.testing.assert_allclose(errors.mean(axis=1).to_numpy(), scores, rtol=1e-10)
    assert list(errors.columns) == F and errors.index.equals(x["test"].index)
    assert list(trained.reconstruct(x["test"]).columns) == F


def test_reproducible_with_seed(frames):
    _, x, _ = frames
    a = AutoencoderDetector(settings=TINY, seed=3).fit(x["train"], x["validation"]).score(x["test"])
    b = AutoencoderDetector(settings=TINY, seed=3).fit(x["train"], x["validation"]).score(x["test"])
    c = AutoencoderDetector(settings=TINY, seed=4).fit(x["train"], x["validation"]).score(x["test"])
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)
    assert not np.allclose(a, c)


def test_feature_order_and_schema_validation(trained, frames):
    _, x, _ = frames
    scrambled = x["test"][list(reversed(F))]
    np.testing.assert_array_equal(trained.score(scrambled), trained.score(x["test"]))
    with pytest.raises(ValueError, match="missing"):
        trained.score(x["test"].drop(columns="xmv_4"))
    with pytest.raises(ValueError, match="DataFrame"):
        trained.score(x["test"].to_numpy())
    bad = x["test"].copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        trained.score(bad)
    with pytest.raises(ValueError, match="Metadata"):
        AutoencoderDetector(feature_columns=("xmeas_1", "sample"))


def test_off_manifold_rows_score_higher(trained, frames):
    """Implementation sanity check only (not an effectiveness claim).

    The fixture network is deliberately tiny and barely trained (8 epochs, validation
    loss still falling), so much of the variance is not reconstructed even for normal
    rows. A ratio such as "the score doubles" would therefore measure how well the
    fixture happens to be trained, not whether the detector works: the score is the
    MEAN over all features of the squared errors, so one feature shifted by ``shift``
    standard deviations adds at most about shift**2 / n_features to it when the network
    does not reproduce the shift. The checks below follow from that arithmetic and do
    not depend on the training quality of the fixture.
    """
    _, x, _ = frames
    shift, n_features = 4.0, len(F)
    shifted = x["test"].copy()
    shifted["xmeas_5"] += shift

    err_base = trained.reconstruction_errors(x["test"])["xmeas_5"]
    err_shift = trained.reconstruction_errors(shifted)["xmeas_5"]
    score_base, score_shift = trained.score(x["test"]), trained.score(shifted)

    # 1. the shifted sensor is mostly NOT reproduced: its squared error rises by at least half of shift**2
    assert err_shift.mean() - err_base.mean() >= 0.5 * shift**2
    # 2. the row score rises by at least half of the analytic increase shift**2 / n_features
    assert score_shift.mean() - score_base.mean() >= 0.5 * shift**2 / n_features
    # 3. the effect is not an artefact of a few rows: the score rises for at least 90 % of the rows
    assert (score_shift > score_base).mean() >= 0.9


# ------------------------------------------------------------------ calibration & leakage
def test_calibration_and_leakage_guards(frames):
    _, x, _ = frames
    det = AutoencoderDetector(settings=TINY, seed=7).fit(x["train"], x["validation"])
    with pytest.raises(BaselineError, match="calibrate"):
        det.detect(x["test"])
    with pytest.raises(LeakageError):
        det.calibrate(x["train"].iloc[:100], target_fpr=0.05)        # fit rows
    with pytest.raises(LeakageError):
        det.calibrate(x["validation"], target_fpr=0.05)              # early-stopping rows
    det.calibrate(x["calibration"], target_fpr=0.05)
    assert det.detect(x["calibration"], check_unseen=False).predictions.mean() <= 0.05
    with pytest.raises(LeakageError, match="calibration"):
        det.detect(x["calibration"])
    with pytest.raises(LeakageError, match="fitting"):
        det.detect(pd.concat([x["test"], x["validation"].iloc[:2]]))
    assert det.detect(x["test"]).predictions.dtype == bool


# ------------------------------------------------------------------ raw-value inference
def test_detect_raw_matches_scaled(trained, frames):
    raw, x, _ = frames
    a, b = trained.detect_raw(raw["test"]), trained.detect(x["test"])
    np.testing.assert_allclose(a.scores, b.scores, rtol=1e-6)
    np.testing.assert_array_equal(a.predictions, b.predictions)


def test_scaler_checks(frames):
    raw, x, scaler = frames
    det = AutoencoderDetector(settings=TINY)
    with pytest.raises(AutoencoderError, match="fitted"):
        det.attach_scaler(FeatureScaler())
    with pytest.raises(AutoencoderError, match="feature columns"):
        det.attach_scaler(FeatureScaler(feature_columns=F[:10]).fit(raw["train"]))
    trained_no_scaler = AutoencoderDetector(settings=TINY, seed=7).fit(x["train"], x["validation"])
    with pytest.raises(AutoencoderError, match="No scaler"):
        trained_no_scaler.score_raw(raw["test"])


# ------------------------------------------------------------------ persistence
def test_save_load_roundtrip(fake_root, trained, frames):
    _, x, _ = frames
    paths = get_paths(fake_root, config=default_config())
    files = trained.save("artifacts/models/test_ae", paths)
    assert {p.name for p in files.values()} == {"model.keras", "metadata.json", "history.csv", "fingerprints.npz", "manifest.json"}
    loaded = AutoencoderDetector.load("artifacts/models/test_ae", paths=paths)
    np.testing.assert_allclose(loaded.score(x["test"]), trained.score(x["test"]), rtol=1e-6)
    assert loaded.threshold_ == trained.threshold_ and loaded.threshold_source_ == trained.threshold_source_
    assert loaded.scaler.to_dict() == trained.scaler.to_dict()
    assert loaded.settings == trained.settings and loaded.training_summary_ == trained.training_summary_
    with pytest.raises(LeakageError):
        loaded.detect(x["train"].iloc[:10])                          # fingerprints survive the round trip
    np.testing.assert_array_equal(loaded.detect(x["test"]).predictions, trained.detect(x["test"]).predictions)


def test_save_guards(fake_root, trained, frames):
    _, x, _ = frames
    paths = get_paths(fake_root, config=default_config())
    trained.save("artifacts/models/guarded", paths)
    with pytest.raises(FileExistsError):
        trained.save("artifacts/models/guarded", paths)
    trained.save("artifacts/models/guarded", paths, overwrite=True)
    with pytest.raises(UnsafeOutputPathError):
        trained.save("docs/model", paths)
    with pytest.raises(BaselineError):
        AutoencoderDetector(settings=TINY).save("artifacts/models/unfitted", paths)


def test_load_validation(fake_root, trained):
    paths = get_paths(fake_root, config=default_config())
    files = trained.save("artifacts/models/to_break", paths)
    meta_path, model_path = files["metadata"], files["model"]
    files["manifest"].unlink()   # this test exercises the older schema/checksum checks; manifest tests are below
    with pytest.raises(AutoencoderError, match="No metadata"):
        AutoencoderDetector.load("artifacts/models/does_not_exist", paths=paths)
    with pytest.raises(AutoencoderError, match="feature schema"):
        AutoencoderDetector.load("artifacts/models/to_break", paths=paths, expected_feature_columns=list(reversed(F)))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "format_version": 99}), encoding="utf-8")
    with pytest.raises(AutoencoderError, match="Unsupported"):
        AutoencoderDetector.load("artifacts/models/to_break", paths=paths)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with open(model_path, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(AutoencoderError, match="checksum"):
        AutoencoderDetector.load("artifacts/models/to_break", paths=paths)


def test_constructor_validation():
    with pytest.raises(AutoencoderError):
        AutoencoderDetector(settings={"bottleneck": 4})               # type: ignore[arg-type]
    with pytest.raises(AutoencoderError):
        AutoencoderDetector(seed=-1)


# ------------------------------------------------------------------ Phase 3 integration
def test_plugs_into_phase3_assessment(trained, frames):
    _, x, _ = frames
    gen = AnomalyGenerator().fit(x["train"])
    r = gen.generate(x["test"], AnomalySpec("correlation_break", 0.1), seed=1)
    table = assess_family_detectability(r.row_info, trained.detect(r.features), trained.detect(r.features))
    assert table.loc["correlation_break", "verdict"] in {"SUPPORTED", "NOT SUPPORTED"}


# ------------------------------------------------------------------ artifact manifest (integrity)
def test_manifest_written_and_verified_on_load(fake_root, trained):
    paths = get_paths(fake_root, config=default_config())
    files = trained.save("artifacts/models/with_manifest", paths)
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"model.keras", "metadata.json", "history.csv", "fingerprints.npz"}
    assert verify_manifest(files["model"].parent) == "verified"
    AutoencoderDetector.load("artifacts/models/with_manifest", paths=paths, require_manifest=True)


@pytest.mark.parametrize("what", ["threshold", "scaler"])
def test_edited_metadata_is_rejected_by_manifest(fake_root, trained, what):
    paths = get_paths(fake_root, config=default_config())
    files = trained.save(f"artifacts/models/edited_{what}", paths)
    meta = json.loads(files["metadata"].read_text(encoding="utf-8"))
    if what == "threshold":
        meta["threshold"] = meta["threshold"] * 100
    else:
        meta["scaler"]["scale"][0] = meta["scaler"]["scale"][0] * 2
    files["metadata"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with pytest.raises(AutoencoderError, match="metadata.json does not match"):
        AutoencoderDetector.load(f"artifacts/models/edited_{what}", paths=paths)


def test_damaged_or_missing_artifact_files_are_rejected(fake_root, trained):
    paths = get_paths(fake_root, config=default_config())
    files = trained.save("artifacts/models/damaged", paths)
    with open(files["history"], "ab") as fh:
        fh.write(b"0,0\n")
    with pytest.raises(AutoencoderError, match="history.csv does not match"):
        AutoencoderDetector.load("artifacts/models/damaged", paths=paths)
    files["history"].unlink()
    with pytest.raises(AutoencoderError, match="missing"):
        AutoencoderDetector.load("artifacts/models/damaged", paths=paths)


def test_bad_manifest_is_rejected(fake_root, trained):
    paths = get_paths(fake_root, config=default_config())
    files = trained.save("artifacts/models/bad_manifest", paths)
    files["manifest"].write_text("{not json", encoding="utf-8")
    with pytest.raises(AutoencoderError, match="unreadable"):
        AutoencoderDetector.load("artifacts/models/bad_manifest", paths=paths)
    files["manifest"].write_text(json.dumps({"format_version": 1, "algorithm": "sha256", "files": {"x.txt": "00"}}), encoding="utf-8")
    with pytest.raises(AutoencoderError, match="unexpected file"):
        AutoencoderDetector.load("artifacts/models/bad_manifest", paths=paths)


def test_legacy_artifact_without_manifest_loads_unless_required(fake_root, trained, frames):
    _, x, _ = frames
    paths = get_paths(fake_root, config=default_config())
    files = trained.save("artifacts/models/legacy", paths)
    files["manifest"].unlink()
    assert verify_manifest(files["model"].parent) == "absent"
    loaded = AutoencoderDetector.load("artifacts/models/legacy", paths=paths)
    np.testing.assert_allclose(loaded.score(x["test"]), trained.score(x["test"]), rtol=1e-6)
    with pytest.raises(AutoencoderError, match="manifest is required"):
        AutoencoderDetector.load("artifacts/models/legacy", paths=paths, require_manifest=True)
    created = write_manifest("artifacts/models/legacy", paths)          # what scripts/write_artifact_manifest.py does
    assert created.is_file() and verify_manifest(files["model"].parent) == "verified"
    with pytest.raises(FileExistsError):
        write_manifest("artifacts/models/legacy", paths)               # never overwrites silently


# ------------------------------------------------------------------ single-pass detection
def test_detect_with_errors_equals_detect_and_errors(trained, frames):
    _, x, _ = frames
    test = x["test"]
    detection, errors = trained.detect_with_errors(test)
    reference = trained.detect(test)
    np.testing.assert_array_equal(detection.scores, reference.scores)           # same arithmetic, so bit-identical
    np.testing.assert_array_equal(detection.predictions, reference.predictions)
    assert detection.threshold == reference.threshold
    np.testing.assert_array_equal(errors.to_numpy(), trained.reconstruction_errors(test).to_numpy())
    np.testing.assert_allclose(errors.mean(axis=1).to_numpy(), detection.scores)
    assert list(errors.columns) == F and errors.index.equals(test.index)


def test_detect_with_errors_guards(trained, frames):
    _, x, _ = frames
    with pytest.raises(LeakageError):
        trained.detect_with_errors(x["train"].iloc[:10])                       # unseen check on by default
    trained.detect_with_errors(x["train"].iloc[:10], check_unseen=False)       # what the live API does
    uncalibrated = AutoencoderDetector(settings=TINY, seed=7, scaler=frames[2]).fit(x["train"], x["validation"])
    with pytest.raises(BaselineError, match="threshold"):
        uncalibrated.detect_with_errors(x["test"])

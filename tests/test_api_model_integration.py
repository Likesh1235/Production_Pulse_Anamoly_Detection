"""End-to-end API test with a REAL tiny autoencoder (needs TensorFlow + FastAPI).

The model is trained on synthetic NORMAL data only, in a temporary fake
project; no real TEP data and no previously trained production model are used.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from production_pulse.api import create_app  # noqa: E402
from production_pulse.autoencoder import AutoencoderDetector  # noqa: E402
from production_pulse.config import AutoencoderSettings, default_config  # noqa: E402
from production_pulse.data import FEATURE_COLUMNS  # noqa: E402
from production_pulse.paths import get_paths  # noqa: E402
from production_pulse.preprocessing import FeatureScaler  # noqa: E402

from conftest import make_correlated_frame  # noqa: E402

F = list(FEATURE_COLUMNS)
MODEL_DIR = "artifacts/models/autoencoder_v1"
TINY = AutoencoderSettings(hidden_layers=(16,), bottleneck=4, epochs=5, patience=2, batch_size=64)


def _fake_project(root):
    (root / "src" / "production_pulse").mkdir(parents=True)
    (root / "src" / "production_pulse" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    return get_paths(root, config=default_config())


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train once per module; tests copy the saved artifact into their own fake project."""
    paths = _fake_project(tmp_path_factory.mktemp("train_project"))
    raw = {k: make_correlated_frame(n, 50, seed=s) for k, n, s in (("train", 20, 1), ("val", 5, 2), ("cal", 6, 3), ("test", 4, 4))}
    scaler = FeatureScaler().fit(raw["train"])
    det = AutoencoderDetector(settings=TINY, seed=5, scaler=scaler)
    det.fit(scaler.transform_frame(raw["train"]), scaler.transform_frame(raw["val"]))
    det.calibrate(scaler.transform_frame(raw["cal"]), target_fpr=0.05)
    det.save(MODEL_DIR, paths)
    return paths.root / MODEL_DIR, det, raw


@pytest.fixture
def project(fake_root, trained):
    shutil.copytree(trained[0], fake_root / MODEL_DIR)
    return get_paths(fake_root, config=default_config())


def _client(paths):
    return TestClient(create_app(config=default_config(), paths=paths, environ={}), raise_server_exceptions=False)


def test_real_model_predictions_match_detector(project, trained):
    _, det, raw = trained
    rows = raw["test"].iloc[:5]
    payload = {"samples": [{"features": {c: float(r[c]) for c in reversed(F)}} for _, r in rows.iterrows()]}
    with _client(project) as client:
        assert client.get("/health").json()["model"]["status"] == "ready"
        body = client.post("/predict", json=payload).json()
    expected = det.detect_raw(rows, check_unseen=False)
    np.testing.assert_allclose([r["reconstruction_error"] for r in body["results"]], expected.scores, rtol=1e-5)
    assert [r["is_anomaly"] for r in body["results"]] == expected.predictions.tolist()
    assert body["threshold"] == pytest.approx(det.threshold_)
    errors = det.reconstruction_errors(det.scaler.transform_frame(rows))
    assert body["results"][0]["top_features"][0]["feature"] == errors.iloc[0].idxmax()


def test_real_model_status_uses_saved_metadata(project):
    meta_file = json.loads((project.root / MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    with _client(project) as client:
        s = client.get("/model/status").json()
    assert s["available"] and s["model_id"] == meta_file["model_sha256"][:12]
    assert s["metadata"]["library_versions"]["tensorflow"] == meta_file["library_versions"]["tensorflow"]
    assert s["metadata"]["training_summary"] == meta_file["training_summary"]
    assert s["metadata"]["settings"]["bottleneck"] == 4


def test_tampered_model_is_not_served(project):
    with open(project.root / MODEL_DIR / "model.keras", "ab") as fh:
        fh.write(b"tampered")
    with _client(project) as client:
        assert client.get("/health").json()["model"]["status"] == "load_failed"
        assert client.post("/predict", json={"features": {c: 0.0 for c in F}}).status_code == 503

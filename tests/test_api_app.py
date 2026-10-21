"""FastAPI app tests with an isolated app factory and a STUB detector (no TensorFlow).

The stub implements exactly the detector interface the API uses
(feature_columns, scaler, threshold_, threshold_source_, detect_raw,
reconstruction_errors, detect_with_errors), so HTTP behaviour, validation, persistence, CORS and
error handling are tested without a trained model. Real-model integration is
in test_api_model_integration.py.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from production_pulse.api import create_app  # noqa: E402
from production_pulse.autoencoder import AutoencoderError  # noqa: E402
from production_pulse.baselines import Detection  # noqa: E402
from production_pulse.config import default_config  # noqa: E402
from production_pulse.data import FEATURE_COLUMNS  # noqa: E402
from production_pulse.paths import get_paths  # noqa: E402
from production_pulse.preprocessing import FeatureScaler  # noqa: E402
from production_pulse.reporting import write_evaluation_report  # noqa: E402

from conftest import make_tep_frame  # noqa: E402

F = list(FEATURE_COLUMNS)
MODEL_DIR = "artifacts/models/autoencoder_v1"
TOKEN = "t" * 24
TRAIN = make_tep_frame(n_runs=10, n_samples=50, seed=1)
SCALER = FeatureScaler().fit(TRAIN)


class StubDetector:
    """score = mean of squared SCALED values; per-feature error = squared scaled value."""

    def __init__(self, *, threshold=3.0, feature_columns=FEATURE_COLUMNS, scaler=SCALER, fail=False):
        self.feature_columns = tuple(feature_columns)
        self.scaler = scaler
        self.threshold_ = threshold
        self.threshold_source_ = "calibrated: stub"
        self.fail = fail
        self.check_unseen_calls = []

    def reconstruction_errors(self, scaled):
        return pd.DataFrame(scaled[F].to_numpy() ** 2, index=scaled.index, columns=F)

    def detect_with_errors(self, scaled, *, check_unseen=True):
        self.check_unseen_calls.append(check_unseen)
        if self.fail:
            raise RuntimeError("secret internal failure at C:/very/private/path")
        errors = self.reconstruction_errors(scaled)
        scores = errors.mean(axis=1).to_numpy()
        return Detection("stub", scores, scores > self.threshold_, float(self.threshold_), self.threshold_source_), errors

    def detect_raw(self, df, *, check_unseen=True):
        self.check_unseen_calls.append(check_unseen)
        if self.fail:
            raise RuntimeError("secret internal failure at C:/very/private/path")
        scores = self.reconstruction_errors(self.scaler.transform_frame(df)).mean(axis=1).to_numpy()
        return Detection("stub", scores, scores > self.threshold_, float(self.threshold_), self.threshold_source_)


def _write_metadata(paths, **extra):
    d = paths.root / MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    meta = {"format_version": 1, "model_type": "dense_autoencoder", "created_utc": "2026-10-20T00:00:00+00:00",
            "model_sha256": "ab" * 32, "settings": {"bottleneck": 8}, "scaler": SCALER.to_dict(), **extra}
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def paths(fake_root):
    return get_paths(fake_root, config=default_config())


def make_client(paths, *, detector=None, loader=None, environ=None, api=None, with_model=True, load=True):
    if with_model:
        _write_metadata(paths)
    if loader is None:
        det = detector or StubDetector()

        def loader(directory, *, paths, expected_feature_columns):
            return det
    cfg = default_config()
    if api:
        cfg = dataclasses.replace(cfg, api=dataclasses.replace(cfg.api, **api))
    app = create_app(config=cfg, paths=paths, environ=environ or {}, model_loader=loader, load_model_on_startup=load)
    return TestClient(app, raise_server_exceptions=False)


def sample(i=0, **changes):
    row = TRAIN.iloc[i]
    feats = {c: float(row[c]) for c in F}
    feats.update(changes)
    return {"features": feats}


def anomalous(i=0):
    return sample(i, xmeas_1=float(TRAIN.iloc[i]["xmeas_1"]) + 30.0)


# ------------------------------------------------------------------ startup / health / status
def test_startup_without_model(paths):
    with make_client(paths, with_model=False) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["model"] == {"available": False, "status": "not_found", "reason": h["model"]["reason"]}
        assert h["database"]["available"] is True
        s = client.get("/model/status").json()
        assert s["available"] is False and s["status"] == "not_found" and "metadata" not in s and "model_id" not in s
        r = client.post("/predict", json=sample())
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
        assert r.json()["error"]["details"] == {"model_status": "not_found"}


def test_loading_disabled(paths):
    with make_client(paths, load=False) as client:
        assert client.get("/health").json()["model"]["status"] == "not_loaded"


def test_startup_with_model_and_status(paths):
    with make_client(paths) as client:
        h = client.get("/health").json()
        assert h["model"]["available"] is True and h["model"]["status"] == "ready"
        s = client.get("/model/status").json()
        assert s["feature_schema"]["n_features"] == 52 and s["feature_schema"]["feature_columns"] == F
        assert s["model_id"] == "ab" * 6 and s["model_dir"] == MODEL_DIR
        meta = s["metadata"]
        assert meta["threshold"] == 3.0 and meta["threshold_source"] == "calibrated: stub"
        assert meta["preprocessing"]["method"] == "standard"
        assert "library_versions" not in meta and "training_summary" not in meta   # absent -> omitted, not invented
        assert "center" not in json.dumps(meta)                                       # scaler parameters not dumped


@pytest.mark.parametrize(
    "detector, status",
    [
        (StubDetector(threshold=None), "not_calibrated"),
        (StubDetector(threshold=float("nan")), "not_calibrated"),
        (StubDetector(scaler=None), "no_preprocessing"),
        (StubDetector(feature_columns=list(reversed(F))), "incompatible"),
        (StubDetector(scaler=FeatureScaler(feature_columns=F[:10]).fit(TRAIN)), "incompatible"),
    ],
)
def test_models_that_are_not_ready(paths, detector, status):
    with make_client(paths, detector=detector) as client:
        assert client.get("/health").json()["model"] == {
            "available": False, "status": status, "reason": client.get("/health").json()["model"]["reason"]}
        assert client.post("/predict", json=sample()).status_code == 503


@pytest.mark.parametrize("exc, status", [(AutoencoderError("checksum mismatch in C:/Users/secret/model.keras"), "load_failed"),
                                         (ImportError("No module named tensorflow"), "dependency_missing")])
def test_model_loading_errors_are_reported_without_paths(paths, exc, status):
    def loader(directory, *, paths, expected_feature_columns):
        raise exc

    with make_client(paths, loader=loader) as client:
        body = client.get("/model/status").text + client.get("/health").text
        assert f'"status":"{status}"' in body.replace(" ", "")
        assert "C:/Users" not in body and str(paths.root) not in body


# ------------------------------------------------------------------ predictions
def test_single_prediction(paths):
    det = StubDetector()
    with make_client(paths, detector=det) as client:
        r = client.post("/predict", json={**sample(), "sample_id": "s-1"})
        assert r.status_code == 200
        body = r.json()
        assert body["n_samples"] == 1 and body["history_stored"] is True and body["model_id"] == "ab" * 6
        res = body["results"][0]
        assert det.check_unseen_calls and not any(det.check_unseen_calls)     # live inference: no leakage refusal
        expected = det.detect_raw(pd.DataFrame([sample()["features"]])[F], check_unseen=False).scores[0]
        assert res["reconstruction_error"] == pytest.approx(expected)
        assert res["is_anomaly"] == (expected > 3.0) and res["threshold"] == 3.0 and res["sample_id"] == "s-1"
        tops = res["top_features"]
        assert len(tops) == default_config().api.top_features
        assert [t["squared_error"] for t in tops] == sorted((t["squared_error"] for t in tops), reverse=True)
        assert 0 < sum(t["share"] for t in tops) <= 1 + 1e-9


def test_batch_prediction_preserves_order_and_flags_anomaly(paths):
    with make_client(paths) as client:
        body = client.post("/predict", json={"samples": [sample(0), anomalous(1), {**sample(2), "sample_id": "c"}]}).json()
        assert [r["index"] for r in body["results"]] == [0, 1, 2] and body["results"][2]["sample_id"] == "c"
        assert body["results"][1]["is_anomaly"] is True
        assert body["results"][1]["top_features"][0]["feature"] == "xmeas_1"
        assert body["n_anomalies"] == sum(r["is_anomaly"] for r in body["results"])


def test_feature_order_does_not_matter(paths):
    feats = sample(3)["features"]
    shuffled = dict(reversed(list(feats.items())))
    with make_client(paths) as client:
        a = client.post("/predict", json={"features": feats}).json()["results"][0]["reconstruction_error"]
        b = client.post("/predict", json={"features": shuffled}).json()["results"][0]["reconstruction_error"]
    assert a == b


def test_missing_and_extra_features(paths):
    feats = sample()["features"]
    missing = {k: v for k, v in feats.items() if k != "xmv_3"}
    extra = {**feats, "simulationRun": 1.0}
    with make_client(paths) as client:
        r = client.post("/predict", json={"samples": [{"features": missing}, {"features": extra}, {"features": feats}]})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "invalid_features"
        assert err["details"] == [{"sample_index": 0, "missing_features": ["xmv_3"]},
                                  {"sample_index": 1, "unexpected_features": ["simulationRun"]}]


@pytest.mark.parametrize("value", ["1.5", True, None, [1.0], {"v": 1.0}])
def test_non_numeric_values_rejected(paths, value):
    with make_client(paths) as client:
        r = client.post("/predict", json=sample(xmeas_4=value))
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
        assert "non-numeric" in json.dumps(r.json())


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1" + "0" * 400])
def test_non_finite_values_rejected(paths, literal):
    feats = sample()["features"]
    text = json.dumps({"features": {**feats, "xmeas_9": "__X__"}}).replace('"__X__"', literal)
    with make_client(paths) as client:
        r = client.post("/predict", content=text, headers={"Content-Type": "application/json"})
        assert r.status_code == 422 and "non-finite" in json.dumps(r.json())


class NanStubDetector(StubDetector):
    """Simulates a network whose output overflows for one row (finite input, non-finite error)."""

    def reconstruction_errors(self, scaled):
        out = super().reconstruction_errors(scaled)
        out.iloc[-1, 3] = np.inf
        return out


@pytest.mark.parametrize("value, detector, reason", [
    (1e300, None, "float32"),                      # scaled value beyond float32: rejected before inference
    (None, NanStubDetector(), "not finite"),       # finite input, non-finite reconstruction error
])
def test_finite_but_overflowing_values_are_rejected_and_not_stored(paths, value, detector, reason):
    """Regression: {"xmeas_7": 1e300} used to return HTTP 200 with a null score and a NULL history row."""
    second = sample(1, xmeas_7=value) if value is not None else sample(1)
    with make_client(paths, detector=detector) as client:
        r = client.post("/predict", json={"samples": [sample(0), second]})
        assert r.status_code == 422, r.text
        err = r.json()["error"]
        assert err["code"] == "value_out_of_range"
        assert err["details"][0]["sample_index"] == 1 and reason in err["details"][0]["reason"]
        if value is not None:
            assert err["details"][0]["features"] == ["xmeas_7"]
        assert client.get("/history").json()["total"] == 0          # nothing scored or stored
    with make_client(paths) as client:                               # a normal request still works
        ok = client.post("/predict", json=sample(0))
        assert ok.status_code == 200 and ok.json()["results"][0]["reconstruction_error"] >= 0


@pytest.mark.parametrize(
    "payload",
    [{}, {"samples": []}, {"features": {}}, {"samples": [sample()], "features": sample()["features"]},
     {"samples": [sample()], "extra": 1}, {"samples": [{"features": sample()["features"], "label": 1}]},
     {"features": {f"f{i}": 1.0 for i in range(201)}}],
)
def test_malformed_requests(paths, payload):
    with make_client(paths) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422 and r.json()["error"]["code"] in {"validation_error", "invalid_features"}


def test_batch_size_limit(paths):
    with make_client(paths, api={"max_batch_size": 2}) as client:
        assert client.post("/predict", json={"samples": [sample(0), sample(1)]}).status_code == 200
        r = client.post("/predict", json={"samples": [sample(0), sample(1), sample(2)]})
        assert r.status_code == 413 and r.json()["error"]["code"] == "batch_too_large"


# ------------------------------------------------------------------ history
def test_history_persistence_and_pagination(paths):
    with make_client(paths, api={"history_default_page_size": 2, "history_max_page_size": 3}) as client:
        assert client.get("/history").json() == {"items": [], "total": 0, "limit": 2, "offset": 0, "next_before_id": None}
        ids = []
        for i in range(3):
            ids += [r["prediction_id"] for r in client.post("/predict", json={"samples": [sample(i), anomalous(i)]}).json()["results"]]
        page = client.get("/history").json()
        assert page["total"] == 6 and [it["id"] for it in page["items"]] == ids[::-1][:2]
        nxt = client.get("/history", params={"before_id": page["next_before_id"]}).json()
        assert [it["id"] for it in nxt["items"]] == ids[::-1][2:4]
        off = client.get("/history", params={"limit": 3, "offset": 3}).json()
        assert [it["id"] for it in off["items"]] == ids[::-1][3:6]
        anom = client.get("/history", params={"anomalies_only": True, "limit": 3}).json()
        assert anom["total"] == 3 and all(it["is_anomaly"] for it in anom["items"])
        assert anom["items"][0]["top_features"][0]["feature"] == "xmeas_1"
        for bad in ({"limit": 0}, {"limit": 4}, {"offset": -1}, {"before_id": 0}, {"limit": "ten"}):
            r = client.get("/history", params=bad)
            assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_raw_inputs_not_stored_by_default(paths):
    with make_client(paths) as client:
        client.post("/predict", json=sample())
    db = paths.root / default_config().api.database_path
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*), MAX(inputs) FROM predictions").fetchone() == (1, None)


def test_database_errors(paths, monkeypatch):
    with make_client(paths) as client:
        def boom(*_, **__):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(client.app.state.services.store, "insert_many", boom)
        r = client.post("/predict", json=sample())
        assert r.status_code == 200 and r.json()["history_stored"] is False and r.json()["warnings"]
        assert r.json()["results"][0]["prediction_id"] is None
        monkeypatch.setattr(client.app.state.services.store, "list_predictions", boom)
        assert client.get("/history").json()["error"]["code"] == "database_unavailable"


def test_unsafe_database_path_disables_history_only(paths):
    with make_client(paths, api={"database_path": "docs/history.sqlite3"}) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["database"]["available"] is False and h["model"]["available"] is True
        assert client.get("/history").status_code == 503
        r = client.post("/predict", json=sample())
        assert r.status_code == 200 and r.json()["history_stored"] is False
    assert not (paths.root / "docs" / "history.sqlite3").exists()


# ------------------------------------------------------------------ metrics
def test_metrics_unavailable_then_available_then_invalid(paths):
    with make_client(paths) as client:
        assert client.get("/metrics").json() == {"available": False, "reason": "No evaluation report has been saved yet."}
        write_evaluation_report({"autoencoder": {"f1": 0.5}}, paths, source="unit test")
        body = client.get("/metrics").json()
        assert body["available"] is True and body["report"]["metrics"] == {"autoencoder": {"f1": 0.5}}
        (paths.root / default_config().api.metrics_report).write_text("{broken", encoding="utf-8")
        assert client.get("/metrics").json() == {"available": False, "reason": "The saved evaluation report is invalid."}


# ------------------------------------------------------------------ CORS
def test_cors_allows_only_configured_origins(paths):
    with make_client(paths) as client:
        ok = client.options("/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
        bad = client.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in bad.headers
        simple = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert simple.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert client.get("/health", headers={"Origin": "http://evil.example"}).headers.get("access-control-allow-origin") is None


def test_cors_can_be_disabled(paths):
    with make_client(paths, api={"cors_origins": ()}) as client:
        assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "http://localhost:5173"}).headers


# ------------------------------------------------------------------ errors, reload, safety
def test_internal_errors_do_not_leak(paths):
    with make_client(paths, detector=StubDetector(fail=True)) as client:
        r = client.post("/predict", json=sample())
        assert r.status_code == 500 and r.json() == {"error": {"code": "internal_error", "message": "Internal server error."}}
        assert "private" not in r.text and "Traceback" not in r.text
        nf = client.get("/no/such/route")
        assert nf.status_code == 404 and nf.json()["error"]["code"] == "not_found"


def test_reload_is_deliberate_and_protected(paths):
    with make_client(paths, with_model=False, environ={"PP_API_ADMIN_TOKEN": TOKEN}) as client:
        assert client.get("/health").json()["model"]["status"] == "not_found"
        _write_metadata(paths)                                                    # model appears later...
        assert client.get("/health").json()["model"]["status"] == "not_found"     # ...but is NOT picked up silently
        assert client.post("/model/reload").status_code == 401
        assert client.post("/model/reload", headers={"X-Admin-Token": "wrong" * 5}).status_code == 401
        r = client.post("/model/reload", headers={"X-Admin-Token": TOKEN})
        assert r.status_code == 200 and r.json()["status"] == "ready"
        assert client.post("/predict", json=sample()).status_code == 200
    with make_client(paths) as client:
        assert client.post("/model/reload", headers={"X-Admin-Token": TOKEN}).json()["error"]["code"] == "reload_disabled"


def test_no_writes_to_original_files(paths):
    originals = [p for p in paths.protected_files if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    with make_client(paths) as client:
        client.post("/predict", json={"samples": [sample(0), anomalous(1)]})
        client.get("/history")
        client.get("/metrics")
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    db = paths.root / default_config().api.database_path
    assert db.is_file() and paths.data_dir in db.parents
    new_files = {p for p in paths.root.rglob("*") if p.is_file()} - set(originals)
    allowed = (paths.data_dir, paths.artifacts_dir, paths.root / "src", paths.root / "docs")
    assert all(any(a in p.parents for a in allowed) or p.name == "pyproject.toml" for p in new_files)


def test_dependency_override_hook(paths):
    """Routes obtain services through get_services, so tests/tools can override it."""
    from production_pulse.api.routes import get_services

    with make_client(paths) as client:
        services = client.app.state.services
        client.app.dependency_overrides[get_services] = lambda: services
        assert client.get("/health").status_code == 200
        client.app.dependency_overrides.clear()


def test_scores_are_numbers_not_strings(paths):
    with make_client(paths) as client:
        res = client.post("/predict", json=sample()).json()["results"][0]
    assert isinstance(res["reconstruction_error"], float) and np.isfinite(res["reconstruction_error"])

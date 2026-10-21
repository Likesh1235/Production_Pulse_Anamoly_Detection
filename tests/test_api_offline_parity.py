"""Offline-vs-API parity: the SAVED production model must give the same answer both ways.

For the same valid 52-feature raw input, this compares

* OFFLINE path (what notebook 03 does after a reload):
  ``AutoencoderDetector.load`` -> ``scaler.transform_frame`` -> ``detect``
* API path: ``POST /predict`` served in-process by FastAPI's ``TestClient`` (no external
  server, no network), which runs ``scaler.transform_frame`` + ``detect_with_errors`` inside
  ``ModelService.predict`` (one network pass per request).

Compared per sample: the scaled feature values (captured inside the API call), the
reconstruction error, the threshold and the final anomaly decision.

Safety: the saved model folder is only READ. It is copied into a temporary fake project
for both paths, and a test checks that the original files are byte-identical afterwards.
Nothing is trained, no threshold or metric is regenerated, and the API history database
lives in the temporary folder.

Inputs (each is labelled):
* ``synthetic-deterministic`` - 4 hand-constructed rows, built from the saved scaler's own
  centre/scale (z = 0, small seeded noise, z = +6 everywhere, z = +/-4 alternating). They are
  NOT real plant data and carry no ground-truth label; they only exercise the code on both
  sides of the threshold. Always runs when TensorFlow, FastAPI and the saved model exist.
* ``real-testing-file`` - 5 rows of ONE evaluation-role run of the fault-free TESTING file
  (read-only). Opt-in: set ``PP_RUN_REALDATA=1``. Holdout and calibration runs are never chosen.

Numerical tolerances (documented, deliberately tight):
* scaled features: rtol = atol = 1e-12. Both sides run the same float64 arithmetic on
  identical inputs, so equality is expected to be exact; 1e-12 only guards against
  platform-specific last-bit effects.
* reconstruction error: rtol = 1e-5, atol = 1e-8. Keras computes in float32 (machine epsilon
  1.2e-7); six layers of accumulated rounding, and different batch sizes (batch vs one row
  per request) can change low-order bits. 1e-5 relative is about 80x float32 epsilon.
* threshold: exact equality (a float64 read from metadata.json; JSON round-trips floats exactly).
* decision: exact, checked only for rows farther than 1e-4 (relative) from the threshold, so a
  legitimate last-bit difference cannot flip a borderline row and make the test flaky.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from production_pulse.api import create_app  # noqa: E402
from production_pulse.autoencoder import AutoencoderDetector  # noqa: E402
from production_pulse.config import default_config  # noqa: E402
from production_pulse.data import FEATURE_COLUMNS  # noqa: E402
from production_pulse.paths import get_paths  # noqa: E402

F = list(FEATURE_COLUMNS)
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = "artifacts/models/autoencoder_v1"
SAVED_MODEL = REPO_ROOT / MODEL_DIR

SCALED_RTOL = SCALED_ATOL = 1e-12
SCORE_RTOL, SCORE_ATOL = 1e-5, 1e-8
DECISION_MARGIN = 1e-4          # relative distance from the threshold below which a decision is not compared
SYNTHETIC_SEED = 20261021

pytestmark = pytest.mark.skipif(
    not (SAVED_MODEL / "metadata.json").is_file(),
    reason=f"no saved model at {MODEL_DIR} (run notebook 03 first)",
)


# ------------------------------------------------------------------ inputs
def _synthetic_rows(meta_scaler: dict) -> pd.DataFrame:
    """Deterministic raw rows built from the saved scaler (see module docstring)."""
    center, scale = np.asarray(meta_scaler["center"]), np.asarray(meta_scaler["scale"])
    z = np.vstack(
        [
            np.zeros(52),
            0.2 * np.random.default_rng(SYNTHETIC_SEED).standard_normal(52),
            np.full(52, 6.0),
            np.tile([4.0, -4.0], 26),
        ]
    )
    return pd.DataFrame(center + z * scale, columns=F)


def _real_testing_rows() -> pd.DataFrame:
    """5 rows of one EVALUATION-role run of the fault-free testing file (read-only)."""
    pytest.importorskip("pyreadr")
    from production_pulse.config import load_config
    from production_pulse.data import load_raw_dataset
    from production_pulse.splitting import select_runs, split_runs

    cfg = load_config(root=REPO_ROOT)
    paths = get_paths(REPO_ROOT, config=cfg)
    source = paths.raw_file("fault_free_testing")
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    raw = load_raw_dataset("fault_free_testing", paths=paths, config=cfg)
    split = split_runs(raw, cfg.evaluation.fractions(), seed=cfg.random_seed, group_columns=cfg.split.group_columns)
    run = int(select_runs(split, "evaluation", 1, seed=cfg.random_seed)[0])
    assert run in split.runs("evaluation") and run not in split.runs("holdout") and run not in split.runs("calibration")
    one = raw[raw["simulationRun"] == run].sort_values("sample")
    assert len(one) == 960 and set(one["faultNumber"]) == {0}
    rows = one.iloc[[0, 240, 480, 720, 959]][F].reset_index(drop=True)
    del raw
    assert (source.stat().st_size, source.stat().st_mtime_ns) == before   # original file untouched
    return rows


# ------------------------------------------------------------------ the two paths
def _digests(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


@dataclass
class Run:
    label: str
    raw: pd.DataFrame
    meta: dict
    # offline
    det: AutoencoderDetector
    off_scaled: pd.DataFrame
    off_detection: object
    off_errors: pd.DataFrame
    # independent reference built from metadata.json only
    ref_scaled: np.ndarray
    ref_scores: np.ndarray
    # API
    body: dict
    single_bodies: list
    status: dict
    captured: dict = field(default_factory=dict)
    digests_before: dict = field(default_factory=dict)
    digests_after: dict = field(default_factory=dict)
    digests_copy: dict = field(default_factory=dict)


def _run_both_paths(workdir: Path, label: str, raw: pd.DataFrame) -> Run:
    digests_before = _digests(SAVED_MODEL)
    root = workdir / "fake_project"
    (root / "src" / "production_pulse").mkdir(parents=True)
    (root / "src" / "production_pulse" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    paths = get_paths(root, config=default_config())
    shutil.copytree(SAVED_MODEL, root / MODEL_DIR)
    meta = json.loads((root / MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

    # ---- OFFLINE path (notebook 03 style)
    det = AutoencoderDetector.load(root / MODEL_DIR, paths=paths)
    off_scaled = det.scaler.transform_frame(raw)
    off_detection = det.detect(off_scaled)
    off_errors = det.reconstruction_errors(off_scaled)

    # ---- independent reference: plain numpy from the saved centre/scale, then the model
    center, scale = np.asarray(meta["scaler"]["center"]), np.asarray(meta["scaler"]["scale"])
    ref_scaled = (raw[F].to_numpy(dtype=np.float64) - center) / scale
    ref_frame = pd.DataFrame(ref_scaled, columns=F)
    ref_scores = np.mean((ref_scaled - det.reconstruct(ref_frame).to_numpy()) ** 2, axis=1)

    # ---- API path, in-process
    def payload(frame: pd.DataFrame) -> dict:
        # reversed key order on purpose: the API must put features back in canonical order
        return {"samples": [{"features": {c: float(r[c]) for c in reversed(F)}} for _, r in frame.iterrows()]}

    captured: dict = {"scaled": [], "passes": 0}
    app = create_app(config=default_config(), paths=paths, environ={})
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").json()["model"]["status"] == "ready"
        api_det = app.state.services.model.state.detector
        orig_with_errors, orig_reconstruct = api_det.detect_with_errors, api_det._reconstruct

        def spy_with_errors(df, *a, **k):
            captured["scaled"].append(df.copy())       # what the API feeds to the network
            return orig_with_errors(df, *a, **k)

        def spy_reconstruct(x, *a, **k):
            captured["passes"] += 1                    # every network pass, whoever asks for it
            return orig_reconstruct(x, *a, **k)

        api_det.detect_with_errors, api_det._reconstruct = spy_with_errors, spy_reconstruct
        r = client.post("/predict", json=payload(raw))
        assert r.status_code == 200, r.text
        body = r.json()
        captured["passes_for_batch"] = captured["passes"]
        single = []
        for i in range(len(raw)):
            r1 = client.post("/predict", json=payload(raw.iloc[[i]]))
            assert r1.status_code == 200, r1.text
            single.append(r1.json())
        status = client.get("/model/status").json()
    return Run(label, raw, meta, det, off_scaled, off_detection, off_errors, ref_scaled, ref_scores,
               body, single, status, captured, digests_before, _digests(SAVED_MODEL), _digests(root / MODEL_DIR))


@pytest.fixture(
    scope="module",
    params=[
        "synthetic-deterministic",
        pytest.param(
            "real-testing-file",
            marks=pytest.mark.skipif(os.environ.get("PP_RUN_REALDATA") != "1", reason="set PP_RUN_REALDATA=1 to run"),
        ),
    ],
)
def run(request, tmp_path_factory) -> Run:
    if request.param == "real-testing-file":
        raw = _real_testing_rows()
    else:
        raw = _synthetic_rows(json.loads((SAVED_MODEL / "metadata.json").read_text(encoding="utf-8"))["scaler"])
    return _run_both_paths(tmp_path_factory.mktemp("parity"), request.param, raw)


# ------------------------------------------------------------------ tests
def test_scaled_features_are_identical(run: Run):
    assert len(run.captured["scaled"]) >= 1
    api_scaled = run.captured["scaled"][0]            # what the API fed to the network
    assert list(api_scaled.columns) == F == list(run.off_scaled.columns)
    kw = dict(rtol=SCALED_RTOL, atol=SCALED_ATOL)
    np.testing.assert_allclose(api_scaled.to_numpy(), run.off_scaled.to_numpy(), **kw)
    np.testing.assert_allclose(run.off_scaled.to_numpy(), run.ref_scaled, **kw)                          # (raw-centre)/scale from metadata.json


def test_reconstruction_error_matches(run: Run):
    kw = dict(rtol=SCORE_RTOL, atol=SCORE_ATOL)
    api = np.array([x["reconstruction_error"] for x in run.body["results"]])
    offline = np.asarray(run.off_detection.scores)
    assert len(api) == len(offline) == len(run.raw)
    np.testing.assert_allclose(api, offline, **kw)
    np.testing.assert_allclose(api, run.ref_scores, **kw)
    # one request per sample (batch size 1) must agree with the batched answer
    singles = np.array([b["results"][0]["reconstruction_error"] for b in run.single_bodies])
    np.testing.assert_allclose(singles, offline, **kw)


def test_threshold_is_the_saved_one_on_both_paths(run: Run):
    saved = float(run.meta["threshold"])
    assert run.det.threshold_ == saved
    assert run.off_detection.threshold == saved
    assert run.body["threshold"] == saved
    assert all(b["threshold"] == saved for b in run.single_bodies)
    assert run.status["available"] and run.status["metadata"]["threshold"] == saved
    assert run.body["threshold_source"] == run.off_detection.threshold_source == run.meta["threshold_source"]


def test_anomaly_decision_matches(run: Run):
    thr = float(run.meta["threshold"])
    api_scores = np.array([x["reconstruction_error"] for x in run.body["results"]])
    api_flags = np.array([x["is_anomaly"] for x in run.body["results"]])
    off_flags = np.asarray(run.off_detection.predictions)
    clear = np.abs(api_scores - thr) > DECISION_MARGIN * thr
    assert clear.any(), "every row is borderline; choose different inputs"
    assert np.array_equal(api_flags[clear], off_flags[clear])
    assert np.array_equal(api_flags[clear], api_scores[clear] > thr)             # strict '>' rule
    single_flags = np.array([b["results"][0]["is_anomaly"] for b in run.single_bodies])
    assert np.array_equal(single_flags[clear], off_flags[clear])
    if run.label == "synthetic-deterministic":
        # the synthetic set must exercise BOTH outcomes, otherwise it proves little
        assert api_flags[clear].any() and (~api_flags[clear]).any(), (api_scores.tolist(), thr)


def test_per_feature_contributions_match(run: Run):
    for i, res in enumerate(run.body["results"]):
        errs = run.off_errors.iloc[i]
        top = errs.sort_values(ascending=False, kind="mergesort").iloc[: len(res["top_features"])]
        assert [t["feature"] for t in res["top_features"]] == list(top.index)
        np.testing.assert_allclose([t["squared_error"] for t in res["top_features"]], top.to_numpy(),
                                   rtol=SCORE_RTOL, atol=SCORE_ATOL)


def test_saved_model_folder_was_only_read(run: Run):
    assert run.digests_before == run.digests_after == run.digests_copy
    assert "model.keras" in run.digests_after and "metadata.json" in run.digests_after


def test_one_network_pass_per_request(run: Run):
    """predict() must not run the model twice (once for the score, once for the explanations)."""
    assert run.captured["passes_for_batch"] == 1
    assert run.captured["passes"] == 1 + len(run.raw)      # the batch request + one request per sample
    assert len(run.captured["scaled"]) == 1 + len(run.raw)

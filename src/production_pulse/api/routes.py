"""HTTP endpoints. Business logic lives in :mod:`service`; persistence in :mod:`store`."""

from __future__ import annotations

import hmac
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Header, Query, Request

from .. import __version__
from ..data import FEATURE_COLUMNS
from ..logging_config import get_logger
from ..reporting import ReportError, load_evaluation_report
from .errors import ApiError
from .schemas import HealthResponse, HistoryResponse, PredictRequest, PredictResponse
from .service import InputOutOfRangeError, ModelUnavailableError
from .store import PredictionRecord

log = get_logger("api.routes")
router = APIRouter()
MAX_OFFSET = 10_000_000


def get_services(request: Request):
    """Dependency: the per-app service container (overridable in tests)."""
    return request.app.state.services


def _model_component(state) -> dict[str, Any]:
    return {"available": state.ready, "status": state.status, "reason": state.reason}


# ------------------------------------------------------------------ health & status
@router.get("/health", response_model=HealthResponse, tags=["status"])
def health(services=Depends(get_services)) -> dict[str, Any]:
    """API liveness (always 'ok' while serving) plus separate model and database availability."""
    return {
        "status": "ok",
        "api_version": __version__,
        "model": _model_component(services.model.state),
        "database": services.store_component(),
    }


@router.get("/model/status", tags=["status"])
def model_status(services=Depends(get_services)) -> dict[str, Any]:
    """Model availability, schema and saved metadata. Keys absent from the artifact are omitted."""
    state = services.model.state
    body: dict[str, Any] = {
        **_model_component(state),
        "model_dir": state.model_dir,
        "feature_schema": {"n_features": len(FEATURE_COLUMNS), "feature_columns": list(FEATURE_COLUMNS)},
    }
    if state.model_id:
        body["model_id"] = state.model_id
    if state.loaded_utc:
        body["loaded_utc"] = state.loaded_utc
    if state.metadata:
        body["metadata"] = dict(state.metadata)
    return body


@router.post("/model/reload", tags=["status"])
def reload_model(x_admin_token: Optional[str] = Header(None), services=Depends(get_services)) -> dict[str, Any]:
    """Deliberately reload the model from ``model_dir`` (requires PP_API_ADMIN_TOKEN)."""
    if services.admin_token is None:
        raise ApiError(403, "reload_disabled", "Model reload is disabled; set PP_API_ADMIN_TOKEN to enable it.")
    if x_admin_token is None or not hmac.compare_digest(x_admin_token.encode(), services.admin_token.encode()):
        raise ApiError(401, "unauthorized", "Missing or invalid X-Admin-Token header.")
    state = services.model.load()
    return _model_component(state)


# ------------------------------------------------------------------ prediction
def _frame_from_samples(samples) -> pd.DataFrame:
    expected = set(FEATURE_COLUMNS)
    problems = []
    for i, sample in enumerate(samples):
        names = set(sample.features)
        missing, extra = sorted(expected - names), sorted(names - expected)
        if missing or extra:
            entry: dict[str, Any] = {"sample_index": i}
            if missing:
                entry["missing_features"] = missing
            if extra:
                entry["unexpected_features"] = extra
            problems.append(entry)
    if problems:
        raise ApiError(
            422,
            "invalid_features",
            "Each sample must contain exactly the 52 TEP measurement features (any order).",
            problems[:20],
        )
    # reorder every sample to the model's canonical feature order
    return pd.DataFrame([[s.features[c] for c in FEATURE_COLUMNS] for s in samples], columns=list(FEATURE_COLUMNS))


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(body: PredictRequest, services=Depends(get_services)) -> dict[str, Any]:
    """Score one or more samples of the 52 raw measurement features."""
    n = len(body.samples)
    if n > services.api.max_batch_size:
        raise ApiError(413, "batch_too_large", f"At most {services.api.max_batch_size} samples per request (got {n}).")
    frame = _frame_from_samples(body.samples)
    try:
        outcome = services.model.predict(frame)
    except ModelUnavailableError as exc:
        raise ApiError(503, "model_unavailable", exc.state.reason or "Model is not available.",
                       {"model_status": exc.state.status}) from None
    except InputOutOfRangeError as exc:
        raise ApiError(422, "value_out_of_range",
                       "Sensor values are numerically too large for the model (far outside any physical range); "
                       "nothing was scored or stored.", exc.details) from None

    request_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    warnings: list[str] = []
    ids: list[Optional[int]] = [None] * n
    stored = False
    if services.store is not None:
        records = [
            PredictionRecord(
                created_utc=created,
                request_id=request_id,
                sample_index=i,
                sample_id=sample.sample_id,
                score=res.reconstruction_error,
                threshold=outcome.threshold,
                is_anomaly=res.is_anomaly,
                model_id=outcome.model_id,
                top_features=res.top_features,
                inputs=dict(sample.features),
            )
            for i, (sample, res) in enumerate(zip(body.samples, outcome.samples))
        ]
        try:
            ids = list(services.store.insert_many(records))
            stored = True
        except (sqlite3.Error, OSError) as exc:
            log.error("Could not store prediction history: %s: %s", type(exc).__name__, exc)
            warnings.append("Prediction history could not be saved; results were not stored.")
    else:
        warnings.append("Prediction history database is unavailable; results were not stored.")

    return {
        "request_id": request_id,
        "model_id": outcome.model_id,
        "threshold": outcome.threshold,
        "threshold_source": outcome.threshold_source,
        "n_samples": n,
        "n_anomalies": sum(r.is_anomaly for r in outcome.samples),
        "history_stored": stored,
        "warnings": warnings,
        "results": [
            {
                "index": i,
                "sample_id": sample.sample_id,
                "prediction_id": ids[i],
                "reconstruction_error": res.reconstruction_error,
                "is_anomaly": res.is_anomaly,
                "threshold": outcome.threshold,
                "top_features": res.top_features,
            }
            for i, (sample, res) in enumerate(zip(body.samples, outcome.samples))
        ],
    }


# ------------------------------------------------------------------ metrics & history
@router.get("/metrics", tags=["evaluation"])
def metrics(services=Depends(get_services)) -> dict[str, Any]:
    """Latest saved evaluation report, or an explicit 'unavailable' status (never invented metrics)."""
    try:
        report = load_evaluation_report(services.paths.root / services.api.metrics_report)
    except FileNotFoundError:
        return {"available": False, "reason": "No evaluation report has been saved yet."}
    except (ReportError, OSError) as exc:
        log.warning("Evaluation report unusable: %s", exc)
        return {"available": False, "reason": "The saved evaluation report is invalid."}
    return {"available": True, "report": report}


@router.get("/history", response_model=HistoryResponse, tags=["prediction"])
def history(
    limit: Optional[int] = Query(None, ge=1, description="Page size (default and maximum from config)"),
    offset: int = Query(0, ge=0, le=MAX_OFFSET),
    before_id: Optional[int] = Query(None, ge=1, description="Only rows with id < before_id (stable paging)"),
    anomalies_only: bool = Query(False),
    services=Depends(get_services),
) -> dict[str, Any]:
    """Stored predictions, newest first (ordered by id)."""
    cfg = services.api
    size = cfg.history_default_page_size if limit is None else limit
    if size > cfg.history_max_page_size:
        raise ApiError(422, "validation_error", f"limit must be <= {cfg.history_max_page_size}",
                       [{"location": ["query", "limit"], "message": f"limit must be <= {cfg.history_max_page_size}",
                         "type": "less_than_equal"}])
    if services.store is None:
        raise ApiError(503, "database_unavailable", "The prediction history database is unavailable.")
    try:
        items, total = services.store.list_predictions(limit=size, offset=offset, before_id=before_id,
                                                       anomalies_only=anomalies_only)
    except sqlite3.Error as exc:
        log.error("History query failed: %s", exc)
        raise ApiError(503, "database_unavailable", "The prediction history database is unavailable.") from None
    next_before = items[-1]["id"] if len(items) == size else None
    return {"items": items, "total": total, "limit": size, "offset": offset, "next_before_id": next_before}

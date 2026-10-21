"""Pydantic request/response models.

Value validation is strict on purpose: JSON strings such as "1.5", booleans,
nulls and nested objects are rejected (no silent coercion), and non-finite
numbers (NaN, Infinity - which Python's JSON parser accepts - or integers too
large for a float) are rejected. Feature NAMES are checked against the
model's canonical schema in :mod:`production_pulse.api.service`, so the error
can list exactly which names are missing or unexpected.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_FEATURE_KEYS = 200          # hard cap on keys per sample (the schema has 52)
MAX_FEATURE_NAME_LENGTH = 64


class Sample(BaseModel):
    """One observation: the 52 measurement features by name (any order)."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(..., description="Feature name -> numeric value (52 TEP features, any order)")
    sample_id: Optional[str] = Field(None, max_length=128, description="Optional client identifier, echoed back")

    @field_validator("features")
    @classmethod
    def _numeric_finite(cls, value: dict[str, Any]) -> dict[str, float]:
        if not value:
            raise ValueError("features must not be empty")
        if len(value) > MAX_FEATURE_KEYS:
            raise ValueError(f"too many feature keys ({len(value)} > {MAX_FEATURE_KEYS})")
        long_names = [k for k in value if not k or len(k) > MAX_FEATURE_NAME_LENGTH]
        if long_names:
            raise ValueError(f"feature names must be 1-{MAX_FEATURE_NAME_LENGTH} characters")
        non_numeric, non_finite, clean = [], [], {}
        for name, raw in value.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                non_numeric.append(name)
                continue
            try:
                number = float(raw)
            except OverflowError:
                non_finite.append(name)
                continue
            if not math.isfinite(number):
                non_finite.append(name)
                continue
            clean[name] = number
        problems = []
        if non_numeric:
            problems.append(f"non-numeric value(s) for: {sorted(non_numeric)[:20]}")
        if non_finite:
            problems.append(f"non-finite value(s) for: {sorted(non_finite)[:20]}")
        if problems:
            raise ValueError("; ".join(problems))
        return clean


class PredictRequest(BaseModel):
    """Either ``{"features": {...}}`` (one sample) or ``{"samples": [{"features": {...}}, ...]}``."""

    model_config = ConfigDict(extra="forbid")

    samples: list[Sample] = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _single_to_batch(cls, data: Any) -> Any:
        if isinstance(data, dict) and "features" in data:
            if "samples" in data:
                raise ValueError("send either 'features' (one sample) or 'samples' (a batch), not both")
            single = {"features": data["features"]}
            if "sample_id" in data:
                single["sample_id"] = data["sample_id"]
            rest = {k: v for k, v in data.items() if k not in ("features", "sample_id")}
            return {**rest, "samples": [single]}
        return data


class FeatureContribution(BaseModel):
    feature: str
    squared_error: float
    share: float = Field(..., description="Fraction of this sample's total squared error")


class PredictionResult(BaseModel):
    index: int = Field(..., description="Position of the sample in the request")
    sample_id: Optional[str] = None
    prediction_id: Optional[int] = Field(None, description="History row id (null if not stored)")
    reconstruction_error: float
    is_anomaly: bool
    threshold: float
    top_features: list[FeatureContribution]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow the field name 'model_id'

    request_id: str
    model_id: Optional[str] = None
    threshold: float
    threshold_source: str
    n_samples: int
    n_anomalies: int
    history_stored: bool
    warnings: list[str] = Field(default_factory=list)
    results: list[PredictionResult]


class ComponentStatus(BaseModel):
    available: bool
    status: str
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' whenever the API process is serving requests")
    api_version: str
    model: ComponentStatus
    database: ComponentStatus


class HistoryItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow the field name 'model_id'

    id: int
    created_utc: str
    request_id: str
    sample_index: int
    sample_id: Optional[str] = None
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    model_id: Optional[str] = None
    top_features: list[FeatureContribution]


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int = Field(..., description="Number of rows matching the filter")
    limit: int
    offset: int
    next_before_id: Optional[int] = Field(None, description="Pass as before_id to fetch the next (older) page")

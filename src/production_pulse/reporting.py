"""Saved evaluation reports (written by notebooks, served by ``GET /metrics``).

A report is a JSON object::

    {
      "report_version": 1,
      "generated_utc": "2026-10-20T12:00:00+00:00",
      "source": "notebooks/03_autoencoder_training.ipynb",
      "model_id": "1a2b3c4d5e6f",          # optional
      "metrics": { ... measured values only ... }
    }

Reports are written only through :func:`safe_output_path` (output folders),
NaN/inf become ``null`` (strict JSON), and loading validates the structure.
Nothing here computes or invents metrics.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from .paths import ProjectPaths, get_paths, safe_output_path

PathLike = Union[str, os.PathLike]
REPORT_VERSION = 1
MAX_REPORT_BYTES = 5 * 1024 * 1024


class ReportError(ValueError):
    """A report could not be written or is invalid."""


def to_jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Convert numpy/pandas scalars and containers to strict-JSON values (NaN/inf -> None)."""
    if _depth > 50:
        raise ReportError("report is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(v, _depth=_depth + 1) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, _depth=_depth + 1) for v in value]
    raise ReportError(
        f"Unsupported value of type {type(value).__name__} in report; convert DataFrames with "
        ".to_dict(orient='records') first"
    )


def write_evaluation_report(
    metrics: Mapping[str, Any],
    paths: Optional[ProjectPaths] = None,
    *,
    source: str,
    model_id: Optional[str] = None,
    target: Optional[PathLike] = None,
    overwrite: bool = True,
) -> Path:
    """Write measured metrics as the latest evaluation report (default: config ``api.metrics_report``)."""
    if not isinstance(metrics, Mapping) or not metrics:
        raise ReportError("metrics must be a non-empty mapping of measured values")
    if not isinstance(source, str) or not source.strip():
        raise ReportError("source must name where the metrics were computed (e.g. a notebook path)")
    paths = paths or get_paths()
    if target is None:
        from .config import load_config

        target = load_config(root=paths.root).api.metrics_report
    report = {
        "report_version": REPORT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "metrics": to_jsonable(metrics),
    }
    if model_id is not None:
        report["model_id"] = str(model_id)
    text = json.dumps(report, indent=2, allow_nan=False)
    path = safe_output_path(target, paths, overwrite=overwrite)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atomic replace: readers never see a half-written report
    return path


def load_evaluation_report(path: PathLike) -> dict[str, Any]:
    """Load and validate a report. Raises FileNotFoundError or ReportError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("no evaluation report found")
    if p.stat().st_size > MAX_REPORT_BYTES:
        raise ReportError("evaluation report is too large")
    try:
        data = json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReportError("evaluation report is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ReportError("evaluation report must be a JSON object")
    if data.get("report_version") != REPORT_VERSION:
        raise ReportError("unsupported evaluation report version")
    if not isinstance(data.get("generated_utc"), str) or not isinstance(data.get("source"), str):
        raise ReportError("evaluation report is missing 'generated_utc' or 'source'")
    if not isinstance(data.get("metrics"), dict) or not data["metrics"]:
        raise ReportError("evaluation report has no 'metrics' object")
    return data


def _reject_constant(name: str) -> Any:
    raise ReportError(f"evaluation report contains non-standard JSON value {name}")

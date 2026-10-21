"""SQLite prediction history.

* The database file must pass :func:`production_pulse.paths.safe_output_path`
  (inside data/, artifacts/, reports/ or logs/; never an original file).
* Every statement is parameterised; one short-lived connection per operation.
* Raw sensor values are NOT stored unless ``store_inputs=True``.
* Rows are append-only; ``id`` gives a stable newest-first ordering.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..logging_config import get_logger
from ..paths import ProjectPaths, safe_output_path

log = get_logger("api.store")

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc     TEXT    NOT NULL,
    request_id      TEXT    NOT NULL,
    sample_index    INTEGER NOT NULL CHECK (sample_index >= 0),
    sample_id       TEXT,
    score           REAL    NOT NULL,
    threshold       REAL    NOT NULL,
    is_anomaly      INTEGER NOT NULL CHECK (is_anomaly IN (0, 1)),
    model_id        TEXT,
    top_features    TEXT    NOT NULL,
    inputs          TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions (created_utc);
CREATE INDEX IF NOT EXISTS idx_predictions_anomaly ON predictions (is_anomaly, id);
CREATE INDEX IF NOT EXISTS idx_predictions_request ON predictions (request_id);
"""


class StoreError(RuntimeError):
    """The history database is unavailable or invalid."""


@dataclass(frozen=True)
class PredictionRecord:
    created_utc: str
    request_id: str
    sample_index: int
    sample_id: Optional[str]
    score: float
    threshold: float
    is_anomaly: bool
    model_id: Optional[str]
    top_features: list[dict[str, Any]]
    inputs: Optional[dict[str, float]] = None


class PredictionStore:
    """Append-only prediction history in one SQLite file."""

    def __init__(self, relative_path: str, paths: ProjectPaths, *, store_inputs: bool = False, timeout: float = 5.0) -> None:
        # safe_output_path raises UnsafeOutputPathError for protected/outside paths and creates the folder
        self.path: Path = safe_output_path(relative_path, paths, overwrite=True)
        self.store_inputs = bool(store_inputs)
        self.timeout = float(timeout)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as conn:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, SCHEMA_VERSION):
                    raise StoreError(f"Unsupported history database schema version {version}")
                with conn:
                    conn.executescript(_SCHEMA)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")  # constant, not user input
        except sqlite3.Error as exc:
            raise StoreError(f"Could not initialise the history database: {exc}") from exc

    def ping(self) -> bool:
        try:
            with closing(self._connect()) as conn:
                conn.execute("SELECT 1 FROM predictions LIMIT 1").fetchall()
            return True
        except sqlite3.Error:
            return False

    def insert_many(self, records: Iterable[PredictionRecord]) -> list[int]:
        """Insert records in ONE transaction; returns their row ids in order."""
        ids: list[int] = []
        with closing(self._connect()) as conn:
            with conn:  # commits on success, rolls back on error
                for r in records:
                    cur = conn.execute(
                        "INSERT INTO predictions (created_utc, request_id, sample_index, sample_id, score, threshold,"
                        " is_anomaly, model_id, top_features, inputs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            r.created_utc,
                            r.request_id,
                            int(r.sample_index),
                            r.sample_id,
                            float(r.score),
                            float(r.threshold),
                            1 if r.is_anomaly else 0,
                            r.model_id,
                            json.dumps(r.top_features),
                            json.dumps(r.inputs) if (self.store_inputs and r.inputs is not None) else None,
                        ),
                    )
                    ids.append(int(cur.lastrowid))
        return ids

    def list_predictions(
        self,
        *,
        limit: int,
        offset: int = 0,
        before_id: Optional[int] = None,
        anomalies_only: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Newest first. Returns ``(items, total_matching_filter)``."""
        where, params = [], []
        if anomalies_only:
            where.append("is_anomaly = 1")
        count_where = list(where)
        count_params = list(params)
        if before_id is not None:
            where.append("id < ?")
            params.append(int(before_id))
        w = (" WHERE " + " AND ".join(where)) if where else ""
        cw = (" WHERE " + " AND ".join(count_where)) if count_where else ""
        with closing(self._connect()) as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM predictions{cw}", count_params).fetchone()[0])
            rows = conn.execute(
                "SELECT id, created_utc, request_id, sample_index, sample_id, score, threshold, is_anomaly,"
                f" model_id, top_features FROM predictions{w} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, int(limit), int(offset)],
            ).fetchall()
        items = [
            {
                "id": row["id"],
                "created_utc": row["created_utc"],
                "request_id": row["request_id"],
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "reconstruction_error": row["score"],
                "threshold": row["threshold"],
                "is_anomaly": bool(row["is_anomaly"]),
                "model_id": row["model_id"],
                "top_features": json.loads(row["top_features"]),
            }
            for row in rows
        ]
        return items, total

"""SQLite prediction history (no FastAPI needed)."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from production_pulse.api.store import SCHEMA_VERSION, PredictionRecord, PredictionStore, StoreError
from production_pulse.config import default_config
from production_pulse.paths import UnsafeOutputPathError, get_paths


@pytest.fixture
def paths(fake_root):
    return get_paths(fake_root, config=default_config())


def _rec(i, anomaly=False, sample_id=None, inputs=None):
    return PredictionRecord("2026-10-20T00:00:00.000+00:00", f"req{i // 3}", i % 3, sample_id, 0.1 * i, 1.0,
                            anomaly, "abc123", [{"feature": "xmeas_1", "squared_error": 0.5, "share": 1.0}], inputs)


def test_creates_db_inside_data_folder(paths):
    store = PredictionStore("data/api/history.sqlite3", paths)
    assert store.path.is_file() and paths.data_dir in store.path.parents
    with closing(sqlite3.connect(store.path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_predictions_created", "idx_predictions_anomaly", "idx_predictions_request"} <= indexes
    assert store.ping()
    PredictionStore("data/api/history.sqlite3", paths)            # re-opening an existing DB is fine


@pytest.mark.parametrize("bad", ["docs/history.sqlite3", "history.sqlite3", "src/history.db",
                                 "TEP_FaultFree_Training.RData", "../outside.sqlite3"])
def test_unsafe_locations_refused(paths, bad):
    with pytest.raises(UnsafeOutputPathError):
        PredictionStore(bad, paths)


def test_insert_order_pagination_and_filter(paths):
    store = PredictionStore("data/api/h.sqlite3", paths)
    ids = store.insert_many([_rec(i, anomaly=(i % 2 == 0)) for i in range(7)])
    assert ids == sorted(ids) and len(ids) == 7
    items, total = store.list_predictions(limit=3)
    assert total == 7 and [it["id"] for it in items] == ids[::-1][:3]            # newest first
    page2, _ = store.list_predictions(limit=3, offset=3)
    assert [it["id"] for it in page2] == ids[::-1][3:6]
    cursor, _ = store.list_predictions(limit=3, before_id=items[-1]["id"])
    assert [it["id"] for it in cursor] == ids[::-1][3:6]                         # stable cursor paging
    anom, total_anom = store.list_predictions(limit=10, anomalies_only=True)
    assert total_anom == 4 and all(it["is_anomaly"] for it in anom)
    assert items[0]["top_features"][0]["feature"] == "xmeas_1"


def test_parameterised_sql_stores_text_literally(paths):
    store = PredictionStore("data/api/h.sqlite3", paths)
    nasty = "x'); DROP TABLE predictions; --"
    store.insert_many([_rec(0, sample_id=nasty)])
    items, total = store.list_predictions(limit=5)
    assert total == 1 and items[0]["sample_id"] == nasty


def test_inputs_not_stored_by_default(paths):
    off = PredictionStore("data/api/off.sqlite3", paths)
    on = PredictionStore("data/api/on.sqlite3", paths, store_inputs=True)
    for store in (off, on):
        store.insert_many([_rec(0, inputs={"xmeas_1": 1.0})])
    with closing(sqlite3.connect(off.path)) as c1, closing(sqlite3.connect(on.path)) as c2:
        assert c1.execute("SELECT inputs FROM predictions").fetchone()[0] is None
        assert c2.execute("SELECT inputs FROM predictions").fetchone()[0] == '{"xmeas_1": 1.0}'


def test_failed_batch_is_rolled_back(paths):
    store = PredictionStore("data/api/h.sqlite3", paths)
    bad = PredictionRecord("t", "r", -1, None, 0.1, 1.0, False, None, [])       # violates CHECK
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_many([_rec(0), bad])
    assert store.list_predictions(limit=5)[1] == 0                                # nothing half-written


def test_unknown_schema_version_rejected(paths):
    store = PredictionStore("data/api/h.sqlite3", paths)
    with closing(sqlite3.connect(store.path)) as conn:
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
    with pytest.raises(StoreError, match="schema version"):
        PredictionStore("data/api/h.sqlite3", paths)

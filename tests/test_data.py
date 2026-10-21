"""Tests for production_pulse.data using small synthetic frames only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from production_pulse.config import default_config
from production_pulse.data import (
    EXPECTED_OBJECT_NAMES,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    REQUIRED_COLUMNS,
    DataFileError,
    DataFileNotFoundError,
    DataValidationError,
    LargeFileGuardError,
    RDataObjectError,
    load_raw_dataset,
    load_rdata,
    split_metadata_features,
    summarize_tep_frame,
    validate_tep_frame,
)
from production_pulse.paths import get_paths


# ------------------------------------------------------------------ schema constants
def test_column_constants():
    assert len(FEATURE_COLUMNS) == 52
    assert FEATURE_COLUMNS[0] == "xmeas_1" and FEATURE_COLUMNS[40] == "xmeas_41"
    assert FEATURE_COLUMNS[41] == "xmv_1" and FEATURE_COLUMNS[-1] == "xmv_11"
    assert not set(METADATA_COLUMNS) & set(FEATURE_COLUMNS)
    assert REQUIRED_COLUMNS == METADATA_COLUMNS + FEATURE_COLUMNS
    assert tuple(default_config().data.id_columns) == METADATA_COLUMNS


# ------------------------------------------------------------------ validation
def test_valid_frame_passes(tep_frame):
    assert validate_tep_frame(tep_frame) == []


def test_extra_columns_reported(tep_frame):
    tep_frame["note"] = 1.0
    assert validate_tep_frame(tep_frame) == ["note"]
    with pytest.raises(DataValidationError, match="extra"):
        validate_tep_frame(tep_frame, allow_extra_columns=False)


@pytest.mark.parametrize("bad", [None, [1, 2, 3], np.zeros((3, 55)), pd.DataFrame()])
def test_invalid_input_type_or_empty(bad):
    with pytest.raises(DataValidationError):
        validate_tep_frame(bad)


@pytest.mark.parametrize("col", ["xmeas_7", "xmv_11", "simulationRun"])
def test_missing_column(tep_frame, col):
    with pytest.raises(DataValidationError, match="Missing 1 required"):
        validate_tep_frame(tep_frame.drop(columns=col))


def test_duplicate_column_names(tep_frame):
    dup = pd.concat([tep_frame, tep_frame[["xmeas_1"]]], axis=1)
    with pytest.raises(DataValidationError, match="Duplicate column"):
        validate_tep_frame(dup)


def test_non_numeric_measurement(tep_frame):
    tep_frame["xmeas_3"] = tep_frame["xmeas_3"].astype(str)
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_tep_frame(tep_frame)


def test_boolean_measurement_rejected(tep_frame):
    tep_frame["xmv_2"] = tep_frame["xmv_2"] > 0
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_tep_frame(tep_frame)


def test_nan_rejected(tep_frame):
    tep_frame.loc[5, "xmeas_10"] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        validate_tep_frame(tep_frame)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_inf_rejected(tep_frame, value):
    tep_frame.loc[3, "xmv_5"] = value
    with pytest.raises(DataValidationError, match="Infinite"):
        validate_tep_frame(tep_frame)


def test_fractional_metadata_rejected(tep_frame):
    tep_frame.loc[0, "simulationRun"] = 1.5
    with pytest.raises(DataValidationError, match="whole numbers"):
        validate_tep_frame(tep_frame)


# ------------------------------------------------------------------ summary
def test_summary_values(tep_frame):
    s = summarize_tep_frame(tep_frame)
    assert (s.n_rows, s.n_columns, s.n_feature_columns) == (200, 55, 52)
    assert s.missing_values_total == 0 and s.infinite_values_total == 0
    assert s.duplicate_rows == 0 and s.duplicate_metadata_keys == 0
    assert s.fault_numbers == [0.0]
    assert s.n_runs == 10 and s.runs_per_fault == {"0.0": 10}
    assert s.samples_per_run == {"min": 20, "max": 20, "distinct_counts": [20]}
    assert s.metadata_ranges["sample"] == {"min": 1, "max": 20, "n_unique": 20}
    json.dumps(s.to_dict())  # must be JSON-serialisable
    assert "Distinct runs" in s.format()


def test_summary_reports_dirty_data_without_raising(tep_frame):
    dirty = pd.concat([tep_frame, tep_frame.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "xmeas_2"] = np.nan
    dirty.loc[2, "xmeas_4"] = np.inf
    s = summarize_tep_frame(dirty)
    assert s.duplicate_rows == 1
    assert s.duplicate_metadata_keys == 1
    assert s.missing_values_total == 1 and s.missing_by_column == {"xmeas_2": 1}
    assert s.infinite_values_total == 1
    assert s.samples_per_run["max"] == 21


def test_summary_does_not_modify_input(tep_frame):
    before = tep_frame.copy()
    summarize_tep_frame(tep_frame)
    pd.testing.assert_frame_equal(tep_frame, before)


# ------------------------------------------------------------------ metadata / features
def test_split_metadata_features_order_and_isolation(tep_frame):
    shuffled = tep_frame[list(reversed(tep_frame.columns))]  # scramble column order
    meta, x = split_metadata_features(shuffled)
    assert tuple(meta.columns) == METADATA_COLUMNS
    assert tuple(x.columns) == FEATURE_COLUMNS
    assert not set(METADATA_COLUMNS) & set(x.columns)
    x.iloc[0, 0] = -999.0  # result is a copy
    assert shuffled["xmeas_1"].iloc[0] != -999.0


# ------------------------------------------------------------------ file loading
def test_missing_file(tmp_path):
    with pytest.raises(DataFileNotFoundError):
        load_rdata(tmp_path / "nope.RData")
    with pytest.raises(FileNotFoundError):  # also a standard FileNotFoundError
        load_rdata(tmp_path / "nope.RData")


def test_wrong_suffix_and_folder(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DataFileError):
        load_rdata(f)
    with pytest.raises(DataFileError):
        load_rdata(tmp_path)


def test_invalid_path_type():
    with pytest.raises(TypeError):
        load_rdata(123)  # type: ignore[arg-type]


def test_size_guard_blocks_before_reading(tmp_path):
    f = tmp_path / "big.RData"
    f.write_bytes(b"\0" * (2 * 1_048_576))  # 2 MB of junk: guard must fire before parsing
    with pytest.raises(LargeFileGuardError):
        load_rdata(f, max_file_size_mb=1)


def test_corrupt_rdata(tmp_path):
    pytest.importorskip("pyreadr")
    f = tmp_path / "corrupt.RData"
    f.write_bytes(b"this is not R data")
    with pytest.raises(DataFileError):
        load_rdata(f)


def test_roundtrip_and_object_names(tmp_path, tep_frame):
    pyreadr = pytest.importorskip("pyreadr")
    f = tmp_path / "mini.RData"
    pyreadr.write_rdata(str(f), tep_frame, df_name="fault_free_training")
    df = load_rdata(f, "fault_free_training")
    assert df.shape == tep_frame.shape
    assert validate_tep_frame(df) == []
    assert load_rdata(f).shape == tep_frame.shape  # single object: name optional
    with pytest.raises(RDataObjectError, match="expected R object"):
        load_rdata(f, "faulty_training")


def test_load_raw_dataset_from_fake_project(fake_root, tep_frame):
    pyreadr = pytest.importorskip("pyreadr")
    target = fake_root / "TEP_FaultFree_Training.RData"  # dummy file created by the fixture
    target.unlink()
    pyreadr.write_rdata(str(target), tep_frame, df_name="fault_free_training")
    paths = get_paths(fake_root, config=default_config())
    df = load_raw_dataset("fault_free_training", paths=paths, config=default_config())
    assert df.shape == (200, 55)


def test_load_raw_dataset_guards_large_faulty_files(fake_root):
    """Uses the fake project; the guard fires on size before any reading."""
    big = fake_root / "TEP_Faulty_Testing.RData"
    big.write_bytes(b"\0" * (2 * 1_048_576))
    cfg = default_config().__class__(data=default_config().data.__class__(max_file_size_mb=1))
    paths = get_paths(fake_root, config=cfg)
    with pytest.raises(LargeFileGuardError):
        load_raw_dataset("faulty_testing", paths=paths, config=cfg)


def test_unknown_dataset_key():
    with pytest.raises(KeyError):
        load_raw_dataset("faulty_validation")


def test_expected_object_names_cover_all_raw_files():
    assert set(EXPECTED_OBJECT_NAMES) == {"fault_free_training", "fault_free_testing", "faulty_training", "faulty_testing"}


def test_real_faulty_files_exceed_default_guard():
    """Read-only size check (stat only): the default guard really protects the big files."""
    paths = get_paths()
    limit = default_config().data.max_file_size_mb * 1_048_576
    for key in ("faulty_training", "faulty_testing"):
        assert paths.raw_file(key).stat().st_size > limit
    for key in ("fault_free_training", "fault_free_testing"):
        assert paths.raw_file(key).stat().st_size < limit

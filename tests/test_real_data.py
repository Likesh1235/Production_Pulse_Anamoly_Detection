"""OPT-IN integration test on the student's real fault-free files (read-only).

Skipped unless the environment variable PP_RUN_REALDATA=1 is set. Loads only
the two small fault-free files (~25 MB and ~47 MB); the large faulty files
are never touched. Verifies the shapes recorded in the Phase 1 audit and that
loading does not change the files (size + modification time).
"""

from __future__ import annotations

import os

import pytest

from production_pulse.data import FEATURE_COLUMNS, load_raw_dataset, summarize_tep_frame

pytestmark = pytest.mark.skipif(os.environ.get("PP_RUN_REALDATA") != "1", reason="set PP_RUN_REALDATA=1 to run")

EXPECTED = {
    "fault_free_training": {"rows": 250_000, "runs": 500, "samples": 500},
    "fault_free_testing": {"rows": 480_000, "runs": 500, "samples": 960},
}


@pytest.mark.parametrize("key", list(EXPECTED))
def test_real_fault_free_file(key):
    pytest.importorskip("pyreadr")
    from production_pulse.paths import get_paths

    path = get_paths().raw_file(key)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    df = load_raw_dataset(key)  # validates schema, NaN, inf
    s = summarize_tep_frame(df)
    assert s.n_rows == EXPECTED[key]["rows"] and s.n_columns == 55
    assert s.n_runs == EXPECTED[key]["runs"]
    assert s.samples_per_run["min"] == s.samples_per_run["max"] == EXPECTED[key]["samples"]
    assert s.missing_values_total == 0 and s.infinite_values_total == 0
    assert s.fault_numbers == [0]
    assert list(df.columns[3:]) == list(FEATURE_COLUMNS)
    assert (path.stat().st_size, path.stat().st_mtime_ns) == before  # file untouched

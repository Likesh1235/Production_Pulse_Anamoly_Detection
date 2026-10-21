"""Saved evaluation reports (served by GET /metrics)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from production_pulse.config import default_config
from production_pulse.paths import UnsafeOutputPathError, get_paths
from production_pulse.reporting import ReportError, load_evaluation_report, to_jsonable, write_evaluation_report


@pytest.fixture
def paths(fake_root):
    return get_paths(fake_root, config=default_config())


def test_roundtrip_and_strict_json(paths):
    path = write_evaluation_report({"f1": np.float64(0.5), "undefined": float("nan"), "n": np.int64(3),
                                    "per_family": [{"family": "x", "recall": np.float32(0.25)}]},
                                   paths, source="unit test", model_id="abc")
    assert path == paths.root / default_config().api.metrics_report
    report = load_evaluation_report(path)
    assert report["metrics"]["f1"] == 0.5 and report["metrics"]["undefined"] is None
    assert report["model_id"] == "abc" and report["report_version"] == 1
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_invalid_inputs_and_paths(paths):
    with pytest.raises(ReportError):
        write_evaluation_report({}, paths, source="x")
    with pytest.raises(ReportError):
        write_evaluation_report({"a": 1}, paths, source="")
    with pytest.raises(ReportError, match="DataFrame"):
        to_jsonable({"obj": object()})
    with pytest.raises(UnsafeOutputPathError):
        write_evaluation_report({"a": 1}, paths, source="x", target="docs/metrics.json")


@pytest.mark.parametrize(
    "content",
    ["not json", "[1, 2]", json.dumps({"report_version": 2, "generated_utc": "t", "source": "s", "metrics": {"a": 1}}),
     json.dumps({"report_version": 1, "generated_utc": "t", "source": "s", "metrics": {}}),
     '{"report_version": 1, "generated_utc": "t", "source": "s", "metrics": {"a": NaN}}'],
)
def test_invalid_reports_rejected(tmp_path, content):
    p = tmp_path / "r.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ReportError):
        load_evaluation_report(p)


def test_missing_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evaluation_report(tmp_path / "none.json")

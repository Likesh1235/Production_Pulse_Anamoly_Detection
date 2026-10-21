"""Read-only loading, validation and summary of the TEP ``.RData`` datasets.

Facts used here were verified on the student's files in the Phase 1 audit:
each file holds ONE data.frame named like its key (e.g. ``fault_free_training``)
with columns ``faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11``.

Nothing in this module writes to disk. The meaning of ``faultNumber`` values
and the time at which faults start are NOT assumed; use :func:`summarize_tep_frame`
to inspect what the data actually contains.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .config import RAW_DATA_FILES, ProjectConfig, load_config
from .logging_config import get_logger
from .paths import ProjectPaths, get_paths

log = get_logger(__name__)

PathLike = Union[str, os.PathLike]

METADATA_COLUMNS: tuple[str, ...] = ("faultNumber", "simulationRun", "sample")
XMEAS_COLUMNS: tuple[str, ...] = tuple(f"xmeas_{i}" for i in range(1, 42))
XMV_COLUMNS: tuple[str, ...] = tuple(f"xmv_{i}" for i in range(1, 12))
FEATURE_COLUMNS: tuple[str, ...] = XMEAS_COLUMNS + XMV_COLUMNS  # 52, fixed order
REQUIRED_COLUMNS: tuple[str, ...] = METADATA_COLUMNS + FEATURE_COLUMNS

# R object name inside each file (verified in the Phase 1 audit).
EXPECTED_OBJECT_NAMES: dict[str, str] = {key: key for key in RAW_DATA_FILES}
LARGE_DATASET_KEYS: tuple[str, ...] = ("faulty_training", "faulty_testing")
_RDATA_SUFFIXES = {".rdata", ".rda"}


class DataError(ValueError):
    """Base class for data problems."""


class DataFileNotFoundError(DataError, FileNotFoundError):
    """The requested data file does not exist."""


class DataFileError(DataError):
    """The file exists but is not a readable R data file."""


class RDataObjectError(DataError):
    """The R file does not contain the expected object."""


class DataValidationError(DataError):
    """The DataFrame does not meet the TEP schema/quality rules."""


class LargeFileGuardError(DataError):
    """Refused to load a file above the configured size limit."""


# --------------------------------------------------------------------------- loading
def _import_pyreadr():
    try:
        import pyreadr  # noqa: WPS433 (lazy import keeps package import light)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "pyreadr is required to read .RData files. It is listed in the .venv313 "
            "environment; make sure you are using .venv313\\Scripts\\python.exe."
        ) from exc
    return pyreadr


def load_rdata(
    path: PathLike,
    object_name: Optional[str] = None,
    *,
    max_file_size_mb: Optional[float] = None,
) -> pd.DataFrame:
    """Read one data.frame from an ``.RData``/``.rda`` file (read-only).

    ``object_name``: the R object to return. If omitted the file must contain
    exactly one object. ``max_file_size_mb``: refuse larger files (None = no limit).
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
    p = Path(path)
    if not p.exists():
        raise DataFileNotFoundError(f"Data file not found: {p}")
    if not p.is_file():
        raise DataFileError(f"Not a file: {p}")
    if p.suffix.lower() not in _RDATA_SUFFIXES:
        raise DataFileError(f"Expected an .RData/.rda file, got {p.name!r}")

    size_mb = p.stat().st_size / 1_048_576
    if max_file_size_mb is not None and size_mb > max_file_size_mb:
        raise LargeFileGuardError(
            f"{p.name} is {size_mb:,.1f} MB, above the {max_file_size_mb:,} MB limit. "
            "Large files need an explicit opt-in (allow_large=True) and enough RAM."
        )

    pyreadr = _import_pyreadr()
    log.info("Reading %s (%.1f MB on disk, read-only)", p.name, size_mb)
    try:
        result = pyreadr.read_r(str(p))  # opens the file for reading only
    except Exception as exc:  # pyreadr raises its own error types
        raise DataFileError(f"Could not read {p.name} as R data: {exc}") from exc

    names = list(result.keys())
    if not names:
        raise RDataObjectError(f"{p.name} contains no R objects")
    if object_name is None:
        if len(names) != 1:
            raise RDataObjectError(f"{p.name} contains {len(names)} objects {names}; pass object_name")
        object_name = names[0]
    elif object_name not in result:
        raise RDataObjectError(f"{p.name}: expected R object {object_name!r} but found {names}")

    df = result[object_name]
    if not isinstance(df, pd.DataFrame):
        raise RDataObjectError(f"{p.name}: object {object_name!r} is {type(df).__name__}, not a data frame")
    log.info("Loaded object %r: %d rows x %d columns", object_name, df.shape[0], df.shape[1])
    return df


def load_raw_dataset(
    key: str,
    *,
    paths: Optional[ProjectPaths] = None,
    config: Optional[ProjectConfig] = None,
    validate: bool = True,
    allow_large: bool = False,
) -> pd.DataFrame:
    """Load one of the four original datasets by key, e.g. ``"fault_free_training"``.

    The two faulty files are refused unless ``allow_large=True`` (size guard).
    """
    if key not in RAW_DATA_FILES:
        raise KeyError(f"Unknown dataset key {key!r}; choose from {sorted(RAW_DATA_FILES)}")
    paths = paths if paths is not None else get_paths(config=config)
    config = config if config is not None else load_config(root=paths.root)
    limit = None if allow_large else config.data.max_file_size_mb
    df = load_rdata(paths.raw_file(key), EXPECTED_OBJECT_NAMES[key], max_file_size_mb=limit)
    if validate:
        validate_tep_frame(df)
    return df


# --------------------------------------------------------------------------- validation
def _require_dataframe(df: Any) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise DataValidationError("DataFrame is empty")
    dup = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
    if dup:
        raise DataValidationError(f"Duplicate column names: {dup}")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise DataValidationError(f"Missing {len(missing)} required column(s): {preview}")


def validate_tep_frame(df: Any, *, allow_extra_columns: bool = True) -> list[str]:
    """Check schema and data quality. Returns the list of extra columns (if any).

    Raises :class:`DataValidationError` for: non-DataFrame/empty input, duplicate
    or missing columns, non-numeric columns, NaN, +/-inf, or non-integer metadata.
    """
    _require_dataframe(df)
    _require_columns(df, REQUIRED_COLUMNS)

    non_numeric = [c for c in REQUIRED_COLUMNS if is_bool_dtype(df[c]) or not is_numeric_dtype(df[c])]
    if non_numeric:
        detail = {c: str(df[c].dtype) for c in non_numeric[:10]}
        raise DataValidationError(f"Non-numeric required column(s): {detail}")

    nan_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        raise DataValidationError(f"Missing values (NaN) found: {nan_counts.astype(int).to_dict()}")

    inf_counts = {c: int(np.isinf(df[c].to_numpy(dtype=np.float64)).sum()) for c in REQUIRED_COLUMNS}
    inf_counts = {c: n for c, n in inf_counts.items() if n}
    if inf_counts:
        raise DataValidationError(f"Infinite values found: {inf_counts}")

    for c in METADATA_COLUMNS:
        values = df[c].to_numpy(dtype=np.float64)
        if not np.array_equal(values, np.round(values)):
            raise DataValidationError(f"Metadata column {c!r} must contain whole numbers")

    extra = [str(c) for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra and not allow_extra_columns:
        raise DataValidationError(f"Unexpected extra column(s): {extra}")
    if extra:
        log.warning("Extra columns present (ignored as features): %s", extra)
    return extra


# --------------------------------------------------------------------------- summary
def _py(value: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON-safe summaries."""
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class DataSummary:
    n_rows: int
    n_columns: int
    n_feature_columns: int
    dtype_counts: dict[str, int]
    column_dtypes: dict[str, str]
    missing_values_total: int
    missing_by_column: dict[str, int]
    infinite_values_total: int
    duplicate_rows: int
    duplicate_metadata_keys: int
    metadata_ranges: dict[str, dict[str, Any]]
    fault_numbers: list[Any]
    n_runs: int
    runs_per_fault: dict[str, int]
    samples_per_run: dict[str, Any]
    extra_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        lines = [
            f"Shape                  : {self.n_rows:,} rows x {self.n_columns} columns "
            f"({self.n_feature_columns} measurement features)",
            f"Dtypes                 : {self.dtype_counts}",
            f"Missing values (total) : {self.missing_values_total:,}",
            f"Infinite values        : {self.infinite_values_total:,}",
            f"Duplicate rows         : {self.duplicate_rows:,}",
            f"Duplicate (fault, run, sample) keys: {self.duplicate_metadata_keys:,}",
            f"faultNumber values     : {self.fault_numbers}",
            f"Distinct runs          : {self.n_runs:,}",
            f"Runs per faultNumber   : {self.runs_per_fault}",
            f"Samples per run        : {self.samples_per_run}",
        ]
        for col, r in self.metadata_ranges.items():
            lines.append(f"{col:<23}: min={r['min']}, max={r['max']}, unique={r['n_unique']:,}")
        if self.missing_by_column:
            lines.append(f"Missing by column      : {self.missing_by_column}")
        if self.extra_columns:
            lines.append(f"Extra columns          : {self.extra_columns}")
        return "\n".join(lines)


def summarize_tep_frame(df: Any) -> DataSummary:
    """Describe a TEP frame without modifying it (works even if values are dirty)."""
    _require_dataframe(df)
    _require_columns(df, REQUIRED_COLUMNS)

    dtype_counts: dict[str, int] = {}
    for dt in df.dtypes.astype(str):
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    missing = df.isna().sum()

    numeric_features = [c for c in FEATURE_COLUMNS if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
    n_inf = int(sum(np.isinf(df[c].to_numpy(dtype=np.float64, na_value=np.nan)).sum() for c in numeric_features))

    meta = df[list(METADATA_COLUMNS)]
    ranges = {
        c: {"min": _py(meta[c].min()), "max": _py(meta[c].max()), "n_unique": int(meta[c].nunique())}
        for c in METADATA_COLUMNS
    }
    runs = meta.groupby(["faultNumber", "simulationRun"], dropna=False).size()
    runs_per_fault = meta.groupby("faultNumber", dropna=False)["simulationRun"].nunique()
    counts = runs.to_numpy()
    distinct_counts = sorted(int(x) for x in np.unique(counts))

    return DataSummary(
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        n_feature_columns=len(FEATURE_COLUMNS),
        dtype_counts=dtype_counts,
        column_dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        missing_values_total=int(missing.sum()),
        missing_by_column={str(c): int(n) for c, n in missing.items() if n},
        infinite_values_total=n_inf,
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_metadata_keys=int(meta.duplicated().sum()),
        metadata_ranges=ranges,
        fault_numbers=sorted(_py(v) for v in meta["faultNumber"].dropna().unique()),
        n_runs=int(len(runs)),
        runs_per_fault={str(_py(k)): int(v) for k, v in runs_per_fault.items()},
        samples_per_run={
            "min": int(counts.min()),
            "max": int(counts.max()),
            "distinct_counts": distinct_counts[:20],
        },
        extra_columns=[str(c) for c in df.columns if c not in REQUIRED_COLUMNS],
    )


# --------------------------------------------------------------------------- features
def split_metadata_features(df: pd.DataFrame, *, validate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(metadata, features)`` copies; features are the 52 columns in fixed order."""
    if validate:
        validate_tep_frame(df)
    else:
        _require_dataframe(df)
        _require_columns(df, REQUIRED_COLUMNS)
    return df.loc[:, list(METADATA_COLUMNS)].copy(), df.loc[:, list(FEATURE_COLUMNS)].copy()

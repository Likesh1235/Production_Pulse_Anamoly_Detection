"""Leakage-safe, reproducible splits by simulation run.

All rows of one run (by default the ``simulationRun`` column) go to exactly
one split, so consecutive, autocorrelated samples never appear in both
training and evaluation data.

Important: run IDs are only meaningful **within one file**. The fault-free
training and testing files both number their runs 1..500, but they are
different simulations, so split each file separately. This module makes no
assumption about what ``faultNumber`` means or when a fault starts.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .paths import ProjectPaths, safe_output_path

log = get_logger(__name__)

PathLike = Union[str, os.PathLike]
GroupKey = tuple[int, ...]
SPLIT_FORMAT_VERSION = 1


class SplitError(ValueError):
    """Invalid split request or overlapping/inconsistent splits."""


def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(fractions, Mapping) or not fractions:
        raise SplitError("fractions must be a non-empty mapping such as {'train': 0.8, 'validation': 0.2}")
    clean: dict[str, float] = {}
    for name, value in fractions.items():
        if not isinstance(name, str) or not name:
            raise SplitError("split names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise SplitError(f"fraction for {name!r} must be in (0, 1], got {value!r}")
        clean[name] = float(value)
    if not math.isclose(sum(clean.values()), 1.0, abs_tol=1e-9):
        raise SplitError(f"fractions must sum to 1, got {sum(clean.values())}")
    return clean


def _allocate(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation; every split gets at least one group."""
    k = len(fractions)
    if n < k:
        raise SplitError(f"Need at least {k} runs to make {k} splits, got {n}")
    raw = {name: f * n for name, f in fractions.items()}
    counts = {name: int(math.floor(v)) for name, v in raw.items()}
    leftover = n - sum(counts.values())
    for name in sorted(raw, key=lambda s: (-(raw[s] - counts[s]), list(raw).index(s)))[:leftover]:
        counts[name] += 1
    for name in counts:  # guarantee non-empty splits
        while counts[name] == 0:
            donor = max(counts, key=counts.get)
            counts[donor] -= 1
            counts[name] += 1
    return counts


@dataclass(frozen=True)
class RunSplit:
    """Which run(s) belong to which split. Keys are tuples of group-column values."""

    group_columns: tuple[str, ...]
    assignments: Mapping[str, tuple[GroupKey, ...]]
    seed: int
    fractions: Mapping[str, float]

    def __post_init__(self) -> None:
        check_no_overlap(self.assignments)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.assignments)

    def keys(self, name: str) -> tuple[GroupKey, ...]:
        return self.assignments[name]

    def runs(self, name: str) -> list[int]:
        """Run IDs of one split (only for single-column grouping)."""
        if len(self.group_columns) != 1:
            raise SplitError("runs() needs a single group column; use keys() instead")
        return [k[0] for k in self.assignments[name]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": SPLIT_FORMAT_VERSION,
            "group_columns": list(self.group_columns),
            "seed": self.seed,
            "fractions": dict(self.fractions),
            "assignments": {n: [list(k) for k in ks] for n, ks in self.assignments.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunSplit":
        if data.get("format_version") != SPLIT_FORMAT_VERSION:
            raise SplitError(f"Unsupported split format_version {data.get('format_version')}")
        return cls(
            group_columns=tuple(data["group_columns"]),
            assignments={n: tuple(tuple(int(v) for v in k) for k in ks) for n, ks in data["assignments"].items()},
            seed=int(data["seed"]),
            fractions={k: float(v) for k, v in data["fractions"].items()},
        )

    def save_json(self, path: PathLike, paths: Optional[ProjectPaths] = None, *, overwrite: bool = False) -> Path:
        target = safe_output_path(path, paths, overwrite=overwrite)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: PathLike) -> "RunSplit":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def check_no_overlap(assignments: Mapping[str, Sequence[GroupKey]]) -> None:
    """Raise :class:`SplitError` if any run appears in more than one split (or twice in one)."""
    seen: dict[GroupKey, str] = {}
    for name, keys in assignments.items():
        if len(set(keys)) != len(keys):
            raise SplitError(f"Split {name!r} lists a run more than once")
        for key in keys:
            if key in seen:
                raise SplitError(f"Run {key} is in both {seen[key]!r} and {name!r}")
            seen[key] = name


def _group_keys(df: pd.DataFrame, group_columns: Sequence[str]) -> pd.MultiIndex:
    if not isinstance(df, pd.DataFrame):
        raise SplitError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in group_columns if c not in df.columns]
    if missing:
        raise SplitError(f"Group column(s) missing: {missing}")
    sub = df.loc[:, list(group_columns)]
    if sub.isna().any().any():
        raise SplitError("Group columns contain missing values")
    values = sub.to_numpy(dtype=np.float64)
    if not np.array_equal(values, np.round(values)):
        raise SplitError("Group columns must contain whole numbers (run IDs)")
    return pd.MultiIndex.from_arrays([sub[c].astype(np.int64).to_numpy() for c in group_columns], names=list(group_columns))


def split_runs(
    df: pd.DataFrame,
    fractions: Mapping[str, float],
    *,
    seed: int,
    group_columns: Sequence[str] = ("simulationRun",),
) -> RunSplit:
    """Randomly assign whole runs to splits, reproducibly for a given ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise SplitError(f"seed must be a non-negative integer, got {seed!r}")
    clean = _validate_fractions(fractions)
    groups = tuple(group_columns)
    unique = sorted(set(_group_keys(df, groups).tolist()))
    unique = [tuple(int(v) for v in (k if isinstance(k, tuple) else (k,))) for k in unique]
    counts = _allocate(len(unique), clean)

    order = np.random.default_rng(int(seed)).permutation(len(unique))
    shuffled = [unique[i] for i in order]
    assignments: dict[str, tuple[GroupKey, ...]] = {}
    start = 0
    for name in clean:
        chunk = shuffled[start : start + counts[name]]
        assignments[name] = tuple(sorted(chunk))
        start += counts[name]

    split = RunSplit(group_columns=groups, assignments=assignments, seed=int(seed), fractions=clean)
    log.info("Run split (seed=%d): %s", seed, {n: len(k) for n, k in assignments.items()})
    return split


def apply_split(df: pd.DataFrame, split: RunSplit) -> dict[str, pd.DataFrame]:
    """Return ``{split_name: rows}``. Every run in ``df`` must be assigned, and every
    assigned run must exist in ``df`` (so a split from another file is rejected)."""
    keys = _group_keys(df, split.group_columns)
    present = set(keys.unique().tolist())
    present = {k if isinstance(k, tuple) else (k,) for k in present}
    assigned = {k for ks in split.assignments.values() for k in ks}
    unassigned = present - assigned
    if unassigned:
        raise SplitError(f"{len(unassigned)} run(s) in the data are not in the split, e.g. {sorted(unassigned)[:5]}")
    absent = assigned - present
    if absent:
        raise SplitError(f"{len(absent)} run(s) in the split are not in the data, e.g. {sorted(absent)[:5]}")
    out: dict[str, pd.DataFrame] = {}
    for name, ks in split.assignments.items():
        if len(split.group_columns) == 1:
            mask = keys.get_level_values(0).isin([k[0] for k in ks])
        else:
            mask = keys.isin(list(ks))
        out[name] = df.loc[np.asarray(mask)].copy()
    return out


def check_frames_no_overlap(frames: Mapping[str, pd.DataFrame], group_columns: Sequence[str] = ("simulationRun",)) -> None:
    """Verify that already-split DataFrames share no run (from the SAME source file)."""
    assignments = {}
    for name, frame in frames.items():
        idx = _group_keys(frame, group_columns).unique().tolist()
        assignments[name] = [k if isinstance(k, tuple) else (k,) for k in idx]
    check_no_overlap(assignments)


# Holdout runs that an EARLIER version of notebook 02 scored (its run draw ignored the
# role split). Valid for the default configuration only: seed 42, testing-file roles
# calibration 0.2 / evaluation 0.7 / holdout 0.1, run ids 1..500. tests/test_run_selection.py
# re-derives this tuple from split_runs, so it cannot silently drift from the implementation.
# These runs were never used for fitting or threshold calibration, but they are no longer
# unseen: keep them out of any live-demo selection.
HOLDOUT_RUNS_PREVIOUSLY_SCORED: tuple[int, ...] = (42, 196, 241, 304)


def select_runs(
    split: RunSplit,
    role: str,
    n: int,
    *,
    seed: int,
    exclude: Sequence[int] = (),
) -> np.ndarray:
    """Randomly pick ``n`` run ids from ONE role of ``split`` (sorted, reproducible).

    Runs of every other role can never be returned, and ``exclude`` removes further run ids
    from the pool first. Only run ids are handled here; no data is loaded.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise SplitError(f"seed must be a non-negative integer, got {seed!r}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise SplitError(f"n must be a positive integer, got {n!r}")
    if role not in split.names:
        raise SplitError(f"Unknown role {role!r}; the split has {list(split.names)}")
    excluded = {int(e) for e in exclude}
    pool = sorted(r for r in split.runs(role) if r not in excluded)
    if n > len(pool):
        raise SplitError(f"Asked for {n} runs but role {role!r} has only {len(pool)} after exclusions")
    chosen = np.random.default_rng(int(seed)).choice(np.asarray(pool, dtype=np.int64), size=int(n), replace=False)
    return np.sort(chosen)

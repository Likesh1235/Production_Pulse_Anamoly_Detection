"""Helpers for the equal-operating-point detector comparison (notebook 04).

Pure NumPy/pandas. Nothing here trains or tunes anything.

* :func:`cluster_bootstrap` resamples whole simulation RUNS (not rows), because rows of one run are
  strongly autocorrelated; the interval therefore reflects run-to-run variation, not row counts.
* :func:`rate_by_sample_band` shows whether alarms depend on the sample index (training runs have
  500 samples, testing runs 960).
* :func:`prevalence` and the "random PR-AUC" reference: a detector that ranks rows at random has an
  expected PR-AUC equal to the anomaly prevalence, so PR-AUC must be read against it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluation import MetricError, as_binary, average_precision, roc_auc

METRICS = ("recall", "fpr", "precision", "f1", "roc_auc", "pr_auc")
RATE_METRICS = ("recall", "fpr", "precision", "f1")


def prevalence(labels: Any) -> float:
    """Fraction of rows that are anomalous (also the expected PR-AUC of a random detector)."""
    t = as_binary(labels, "labels")
    if t.size == 0:
        raise MetricError("labels are empty")
    return float(t.mean())


def _rates(label: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = float(np.count_nonzero(pred & label))
    fp = float(np.count_nonzero(pred & ~label))
    n_pos, n_neg = float(label.sum()), float((~label).sum())
    recall = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (tp + fp) and n_pos and (precision + recall) > 0 else float("nan")
    return {"recall": recall, "fpr": fpr, "precision": precision, "f1": f1}


def point_estimates(labels: Any, scores: Any, threshold: float, *, auc: bool = True) -> dict[str, float]:
    """Metrics on the full data at ``scores > threshold`` (the operating point)."""
    t = as_binary(labels, "labels")
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != t.shape:
        raise MetricError("labels and scores must have the same length")
    out = _rates(t, s > float(threshold))
    if auc:
        out["roc_auc"], out["pr_auc"] = roc_auc(t, s), average_precision(t, s)
    return out


def cluster_bootstrap(
    groups: Any,
    labels: Any,
    detectors: Mapping[str, tuple[Any, float]],
    *,
    n_boot: int = 200,
    seed: int = 0,
    auc: bool = True,
    ci: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Bootstrap over whole runs. Every detector sees the SAME resamples, so differences are paired.

    ``detectors`` maps a name to ``(scores, threshold)``. Returns ``(summary, replicates)``:
    ``summary`` is indexed by (detector, metric) with ``estimate`` (full data), ``ci_low``, ``ci_high``
    and ``n_boot_valid``; ``replicates[name]`` has one row per resample. Percentile intervals.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise MetricError("n_boot must be a positive integer")
    if not 0.0 < ci < 1.0:
        raise MetricError("ci must be between 0 and 1")
    if not detectors:
        raise MetricError("detectors must not be empty")
    g = np.asarray(groups)
    t = as_binary(labels, "labels")
    if g.shape != t.shape:
        raise MetricError("groups and labels must have the same length")
    order = np.argsort(g, kind="stable")
    sorted_g = g[order]
    boundaries = np.flatnonzero(np.diff(sorted_g)) + 1
    rows_of_group = np.split(order, boundaries)
    n_groups = len(rows_of_group)
    if n_groups < 2:
        raise MetricError("need at least 2 runs to bootstrap")
    prepared = {}
    for name, (scores, thr) in detectors.items():
        s = np.asarray(scores, dtype=np.float64)
        if s.shape != t.shape:
            raise MetricError(f"scores of {name!r} do not match the labels")
        prepared[name] = (s, float(thr))
    metrics = METRICS if auc else RATE_METRICS
    estimates = {name: point_estimates(t, s, thr, auc=auc) for name, (s, thr) in prepared.items()}
    rng = np.random.default_rng(int(seed))
    reps: dict[str, list[dict[str, float]]] = {name: [] for name in prepared}
    for _ in range(int(n_boot)):
        picked = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([rows_of_group[i] for i in picked])
        lab = t[idx]
        if lab.all() or not lab.any():
            for name in prepared:
                reps[name].append({m: float("nan") for m in metrics})
            continue
        for name, (s, thr) in prepared.items():
            sub = s[idx]
            row = _rates(lab, sub > thr)
            if auc:
                row["roc_auc"], row["pr_auc"] = roc_auc(lab, sub), average_precision(lab, sub)
            reps[name].append(row)
    replicates = {name: pd.DataFrame(rows, columns=list(metrics)) for name, rows in reps.items()}
    lo, hi = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    records = []
    for name, frame in replicates.items():
        for m in metrics:
            vals = frame[m].to_numpy()
            vals = vals[np.isfinite(vals)]
            records.append({
                "detector": name, "metric": m, "estimate": estimates[name][m],
                "ci_low": float(np.percentile(vals, lo)) if vals.size else float("nan"),
                "ci_high": float(np.percentile(vals, hi)) if vals.size else float("nan"),
                "n_boot_valid": int(vals.size),
            })
    return pd.DataFrame(records).set_index(["detector", "metric"]), replicates


def paired_difference(replicates: Mapping[str, pd.DataFrame], a: str, b: str, metric: str, *, ci: float = 0.95) -> dict[str, float]:
    """Bootstrap interval of ``metric(a) - metric(b)`` using the shared resamples."""
    diff = (replicates[a][metric] - replicates[b][metric]).to_numpy()
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    lo, hi = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    return {"mean": float(diff.mean()), "ci_low": float(np.percentile(diff, lo)),
            "ci_high": float(np.percentile(diff, hi)), "n": int(diff.size)}


def rate_by_sample_band(flags: Any, sample_index: Any, edges: Sequence[int]) -> pd.DataFrame:
    """Share of rows flagged inside sample-index bands ``[edges[i], edges[i+1])`` (last band closed).

    Use on CLEAN rows to see whether the false-alarm rate depends on the sample index.
    """
    f = np.asarray(flags, dtype=bool)
    s = np.asarray(sample_index)
    if f.shape != s.shape:
        raise MetricError("flags and sample_index must have the same length")
    e = list(edges)
    if len(e) < 2 or any(b <= a for a, b in zip(e, e[1:])):
        raise MetricError("edges must be strictly increasing with at least two values")
    rows = []
    for i, (lo, hi) in enumerate(zip(e[:-1], e[1:])):
        last = i == len(e) - 2
        mask = (s >= lo) & ((s <= hi) if last else (s < hi))
        n = int(mask.sum())
        rows.append({"band": f"{lo}-{hi if last else hi - 1}", "n_rows": n,
                     "rate": float(f[mask].mean()) if n else float("nan")})
    return pd.DataFrame(rows).set_index("band")

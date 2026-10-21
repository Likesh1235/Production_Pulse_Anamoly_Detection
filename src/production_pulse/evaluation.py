"""Binary anomaly-detection metrics, detector comparison and hypothesis checks.

Conventions (label 1 = anomaly, 0 = normal):

* precision is **undefined** when nothing is predicted positive,
* recall is **undefined** when there are no actual positives,
* false-positive rate is **undefined** when there are no actual negatives,
* F1 = 2TP / (2TP + FP + FN); undefined only when TP = FP = FN = 0,
* PR-AUC (average precision) is undefined without scores or without positives,
* ROC-AUC is undefined without scores or unless both classes are present.

Undefined values are ``nan`` and the reason is recorded in ``undefined``;
``to_dict()`` turns them into ``None`` so results stay valid JSON.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .features import feature_matrix


class MetricError(ValueError):
    """Invalid metric input, or a metric requested where it is undefined."""


# ------------------------------------------------------------------ input checks
def _to_array(values: Any, name: str) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.to_numpy()
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise MetricError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise MetricError(f"{name} is empty")
    return arr


def as_binary(values: Any, name: str = "labels") -> np.ndarray:
    """Validate 0/1 or boolean labels and return a bool array."""
    arr = _to_array(values, name)
    if arr.dtype == bool:
        return arr.copy()
    if not np.issubdtype(arr.dtype, np.number):
        raise MetricError(f"{name} must be boolean or 0/1 numbers, got dtype {arr.dtype}")
    if not np.isfinite(arr.astype(np.float64)).all():
        raise MetricError(f"{name} contains NaN or infinite values")
    if not np.isin(arr, (0, 1)).all():
        raise MetricError(f"{name} must contain only 0 and 1")
    return arr.astype(bool)


def as_scores(values: Any, n: int, name: str = "scores") -> np.ndarray:
    arr = _to_array(values, name).astype(np.float64)
    if arr.size != n:
        raise MetricError(f"{name} has {arr.size} values but labels have {n}")
    if not np.isfinite(arr).all():
        raise MetricError(f"{name} contains NaN or infinite values")
    return arr


def _check_same_length(a: np.ndarray, b: np.ndarray, names: str) -> None:
    if a.size != b.size:
        raise MetricError(f"{names}: length mismatch ({a.size} vs {b.size})")


# ------------------------------------------------------------------ confusion matrix
@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def as_array(self) -> np.ndarray:
        """``[[TN, FP], [FN, TP]]`` (rows = actual 0/1, columns = predicted 0/1)."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=np.int64)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.as_array(),
            index=["actual_normal", "actual_anomaly"],
            columns=["predicted_normal", "predicted_anomaly"],
        )


def confusion_matrix(y_true: Any, y_pred: Any) -> ConfusionMatrix:
    t, p = as_binary(y_true, "y_true"), as_binary(y_pred, "y_pred")
    _check_same_length(t, p, "y_true/y_pred")
    return ConfusionMatrix(
        tp=int(np.sum(t & p)), fp=int(np.sum(~t & p)), tn=int(np.sum(~t & ~p)), fn=int(np.sum(t & ~p))
    )


# ------------------------------------------------------------------ ranking metrics
def _undefined_ranking_reason(t: np.ndarray, *, need_negatives: bool) -> Optional[str]:
    if not t.any():
        return "no actual positives"
    if need_negatives and t.all():
        return "no actual negatives"
    return None


def precision_recall_curve(y_true: Any, scores: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision/recall at each distinct score threshold (descending). Ties handled as groups."""
    t = as_binary(y_true, "y_true")
    s = as_scores(scores, t.size)
    reason = _undefined_ranking_reason(t, need_negatives=False)
    if reason:
        raise MetricError(f"Precision-recall curve undefined: {reason}")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, t_sorted = s[order], t[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), s_sorted.size - 1]
    tps = np.cumsum(t_sorted)[last_of_group]
    fps = (last_of_group + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / t.sum()
    return precision, recall, s_sorted[last_of_group]


def average_precision(y_true: Any, scores: Any) -> float:
    """PR-AUC as step-wise average precision: sum over thresholds of (R_k - R_{k-1}) * P_k."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y_true: Any, scores: Any) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    t = as_binary(y_true, "y_true")
    s = as_scores(scores, t.size)
    reason = _undefined_ranking_reason(t, need_negatives=True)
    if reason:
        raise MetricError(f"ROC-AUC undefined: {reason}")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ------------------------------------------------------------------ summary metrics
def _div(num: float, den: float) -> float:
    return num / den if den else math.nan


@dataclass
class BinaryMetrics:
    n: int
    n_positive: int
    n_negative: int
    n_predicted_positive: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    specificity: float
    accuracy: float
    pr_auc: float
    roc_auc: float
    undefined: dict[str, str] = field(default_factory=dict)

    @property
    def confusion(self) -> ConfusionMatrix:
        return ConfusionMatrix(self.tp, self.fp, self.tn, self.fn)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()}


def binary_metrics(y_true: Any, y_pred: Any, scores: Any = None) -> BinaryMetrics:
    """All threshold metrics from predictions, plus PR/ROC-AUC if ``scores`` are given."""
    cm = confusion_matrix(y_true, y_pred)
    t = as_binary(y_true, "y_true")
    undefined: dict[str, str] = {}

    precision = _div(cm.tp, cm.tp + cm.fp)
    if math.isnan(precision):
        undefined["precision"] = "no predicted positives"
    recall = _div(cm.tp, cm.tp + cm.fn)
    if math.isnan(recall):
        undefined["recall"] = "no actual positives"
    f1 = _div(2 * cm.tp, 2 * cm.tp + cm.fp + cm.fn)
    if math.isnan(f1):
        undefined["f1"] = "no actual and no predicted positives"
    fpr = _div(cm.fp, cm.fp + cm.tn)
    specificity = _div(cm.tn, cm.fp + cm.tn)
    if math.isnan(fpr):
        undefined["false_positive_rate"] = undefined["specificity"] = "no actual negatives"

    if scores is None:
        pr, roc = math.nan, math.nan
        undefined["pr_auc"] = undefined["roc_auc"] = "no scores given"
    else:
        s = as_scores(scores, t.size)
        reason_pr = _undefined_ranking_reason(t, need_negatives=False)
        pr = math.nan if reason_pr else average_precision(t, s)
        if reason_pr:
            undefined["pr_auc"] = reason_pr
        reason_roc = _undefined_ranking_reason(t, need_negatives=True)
        roc = math.nan if reason_roc else roc_auc(t, s)
        if reason_roc:
            undefined["roc_auc"] = reason_roc

    return BinaryMetrics(
        n=cm.total,
        n_positive=cm.tp + cm.fn,
        n_negative=cm.tn + cm.fp,
        n_predicted_positive=cm.tp + cm.fp,
        tp=cm.tp,
        fp=cm.fp,
        tn=cm.tn,
        fn=cm.fn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        specificity=specificity,
        accuracy=(cm.tp + cm.tn) / cm.total,
        pr_auc=pr,
        roc_auc=roc,
        undefined=undefined,
    )


# ------------------------------------------------------------------ thresholds
def threshold_for_fpr(normal_scores: Any, target_fpr: float) -> float:
    """Smallest observed score ``t`` such that the fraction of ``normal_scores > t`` is <= target_fpr.

    Use scores from **calibration-normal** data only (never the final test set).
    Predictions are then ``score > t``.
    """
    if isinstance(target_fpr, bool) or not isinstance(target_fpr, (int, float)) or not 0 < target_fpr < 1:
        raise MetricError(f"target_fpr must be strictly between 0 and 1, got {target_fpr!r}")
    s = _to_array(normal_scores, "normal_scores").astype(np.float64)
    if not np.isfinite(s).all():
        raise MetricError("normal_scores contains NaN or infinite values")
    s = np.sort(s)
    n = s.size
    allowed = int(math.floor(target_fpr * n))  # max number of normal points above threshold
    return float(s[n - 1 - allowed])


# ------------------------------------------------------------------ detector comparison
def _unpack_detection(name: str, det: Any) -> tuple[Any, Any]:
    if hasattr(det, "predictions"):
        return det.predictions, getattr(det, "scores", None)
    if isinstance(det, Mapping):
        if "predictions" not in det:
            raise MetricError(f"Detection {name!r} has no 'predictions'")
        return det["predictions"], det.get("scores")
    if isinstance(det, (tuple, list)) and len(det) == 2:
        return det[0], det[1]
    raise MetricError(f"Detection {name!r} must have .predictions/.scores, be a dict or a (pred, scores) pair")


def recall_by_group(y_true: Any, y_pred: Any, groups: Any) -> pd.Series:
    """Recall within each group of actual positives (e.g. per anomaly family)."""
    t, p = as_binary(y_true, "y_true"), as_binary(y_pred, "y_pred")
    g = _to_array(groups, "groups")
    _check_same_length(t, p, "y_true/y_pred")
    _check_same_length(t, g, "y_true/groups")
    out = {}
    for name in pd.unique(g[t]):
        mask = t & (g == name)
        out[str(name)] = float(p[mask].mean())
    return pd.Series(out, name="recall", dtype=float).sort_index()


def compare_detectors(
    y_true: Any,
    detections: Mapping[str, Any],
    *,
    groups: Any = None,
) -> pd.DataFrame:
    """One row of metrics per detector, all on the SAME examples and labels.

    If ``y_true`` and a detector's predictions are pandas objects, their
    indexes must be identical (guards against comparing misaligned rows).
    ``groups`` (e.g. anomaly family per row) adds ``recall[<group>]`` columns.
    """
    if not detections:
        raise MetricError("detections must not be empty")
    t = as_binary(y_true, "y_true")
    rows = {}
    for name, det in detections.items():
        pred, scores = _unpack_detection(name, det)
        if isinstance(y_true, pd.Series) and isinstance(pred, pd.Series) and not pred.index.equals(y_true.index):
            raise MetricError(f"Detection {name!r} is not aligned with y_true (different index)")
        p = as_binary(pred, f"{name} predictions")
        _check_same_length(t, p, f"y_true/{name}")
        metrics = binary_metrics(t, p, scores).to_dict()
        metrics.pop("undefined")
        if groups is not None:
            for g_name, value in recall_by_group(t, p, groups).items():
                metrics[f"recall[{g_name}]"] = value
        rows[name] = metrics
    return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------------ hypothesis assessment
@dataclass(frozen=True)
class HypothesisCriteria:
    """Explicit pass criteria for 'individually in range, jointly abnormal'.

    A family is SUPPORTED only if ALL hold on held-out data:
    * at least ``min_in_range_fraction`` of its anomalous rows have every
      modified value inside the empirical normal range,
    * the single-feature rule check is no better than its own false-alarm
      rate: recall - FPR <= ``max_single_feature_excess``,
    * the multivariate detector reaches recall >= ``min_multivariate_recall``.
    ``max_marginal_auc`` is reported separately as a stricter check: for each
    feature, the rows where THAT feature was modified are compared with normal
    rows using the distance from the normal median (two-sided ROC-AUC). A value
    near 0.5 means no single feature, looked at on its own, separates them.

    Because process data are strongly autocorrelated, rows of one event are not
    independent and this AUC can be far from 0.5 by chance when few events touch
    a feature. Passing the clean (pre-injection) data as ``clean_features``
    computes the same AUC on the SAME rows before injection; the hidden-ness
    test then uses the excess ``injected AUC - clean AUC`` (<= ``max_marginal_auc_excess``).
    """

    min_in_range_fraction: float = 0.99
    max_single_feature_excess: float = 0.05
    min_multivariate_recall: float = 0.5
    max_marginal_auc: float = 0.65          # used when no clean reference is given
    max_marginal_auc_excess: float = 0.05   # used with clean_features (injected AUC - clean AUC)


def _symmetric_auc(t: np.ndarray, s: np.ndarray) -> float:
    a = roc_auc(t, s)
    return max(a, 1.0 - a)


def assess_family_detectability(
    row_info: pd.DataFrame,
    single_feature: Any,
    multivariate: Any,
    *,
    features: Optional[pd.DataFrame] = None,
    events: Optional[pd.DataFrame] = None,
    clean_features: Optional[pd.DataFrame] = None,
    feature_columns: Optional[Sequence[str]] = None,
    criteria: HypothesisCriteria = HypothesisCriteria(),
    min_rows_for_marginal: int = 10,
    min_events_for_marginal: int = 3,
) -> pd.DataFrame:
    """Measure, per anomaly family, whether the in-range hypothesis holds.

    ``row_info``: per-row ground truth with columns ``is_anomaly``, ``family``
    and ``modified_in_range`` (as produced by the anomaly generator).
    ``single_feature`` / ``multivariate``: detections on the same rows.
    ``features`` + ``events`` (both from the generator): enable the marginal check;
    ``clean_features`` (the data before injection, same rows): enables the excess version.
    Nothing is assumed: every verdict is computed from these inputs.
    """
    needed = {"is_anomaly", "family", "modified_in_range"}
    if not isinstance(row_info, pd.DataFrame) or not needed <= set(row_info.columns):
        raise MetricError(f"row_info must be a DataFrame with columns {sorted(needed)}")
    t = as_binary(row_info["is_anomaly"], "is_anomaly")
    fam = row_info["family"].astype(str).to_numpy()
    in_range = as_binary(row_info["modified_in_range"], "modified_in_range")
    sp, _ = _unpack_detection("single_feature", single_feature)
    mp, _ = _unpack_detection("multivariate", multivariate)
    sp, mp = as_binary(sp, "single_feature predictions"), as_binary(mp, "multivariate predictions")
    _check_same_length(t, sp, "row_info/single_feature")
    _check_same_length(t, mp, "row_info/multivariate")
    normal = ~t
    if not normal.any():
        raise MetricError("Need normal rows to measure false-positive rates")
    if not t.any():
        raise MetricError("No anomalous rows to assess")

    x = None
    if (features is None) != (events is None):
        raise MetricError("The marginal check needs both features and events (or neither)")
    if features is not None:
        if len(features) != t.size:
            raise MetricError("features must have the same rows as row_info")
        needed_ev = {"family", "start_row", "end_row", "modified_features"}
        if not isinstance(events, pd.DataFrame) or not needed_ev <= set(events.columns):
            raise MetricError(f"events must be a DataFrame with columns {sorted(needed_ev)}")
        cols = list(feature_columns) if feature_columns is not None else [str(c) for c in features.columns]
        x = feature_matrix(features, cols, what="features")
        col_pos = {c: j for j, c in enumerate(cols)}
        centre = np.median(x[normal], axis=0)
        deviation = np.abs(x - centre)
        deviation_clean = None
        if clean_features is not None:
            if len(clean_features) != t.size or not clean_features.index.equals(features.index):
                raise MetricError("clean_features must have exactly the same rows/index as features")
            deviation_clean = np.abs(feature_matrix(clean_features, cols, what="clean_features") - centre)
    elif clean_features is not None:
        raise MetricError("clean_features needs features and events as well")

    single_fpr = float(sp[normal].mean())
    multi_fpr = float(mp[normal].mean())
    rows = []
    for family in sorted(pd.unique(fam[t])):
        mask = t & (fam == family)
        in_range_fraction = float(in_range[mask].mean())
        single_recall = float(sp[mask].mean())
        multi_recall = float(mp[mask].mean())
        reasons = []
        if in_range_fraction < criteria.min_in_range_fraction:
            reasons.append("modified values left the normal range")
        if single_recall - single_fpr > criteria.max_single_feature_excess:
            reasons.append("single-feature rule check detects it")
        if multi_recall < criteria.min_multivariate_recall:
            reasons.append("multivariate detector misses it")
        max_auc, max_excess, worst = math.nan, math.nan, ""
        if x is not None:
            rows_by_feature: dict[int, list[np.ndarray]] = {}
            for e in events[events["family"].astype(str) == family].itertuples():
                span = np.arange(int(e.start_row), int(e.end_row) + 1)
                for name in str(e.modified_features).split(";"):
                    if name in col_pos:
                        rows_by_feature.setdefault(col_pos[name], []).append(span)
            best_key = -math.inf
            for j, spans in rows_by_feature.items():
                mod_rows = np.unique(np.concatenate(spans))
                if mod_rows.size < min_rows_for_marginal or len(spans) < min_events_for_marginal:
                    continue
                sub_t = np.r_[np.ones(mod_rows.size, bool), np.zeros(int(normal.sum()), bool)]
                auc = _symmetric_auc(sub_t, np.r_[deviation[mod_rows, j], deviation[normal, j]])
                excess = math.nan
                if deviation_clean is not None:
                    excess = auc - _symmetric_auc(sub_t, np.r_[deviation_clean[mod_rows, j], deviation[normal, j]])
                key = excess if deviation_clean is not None else auc
                if key > best_key:
                    best_key, worst, max_auc, max_excess = key, cols[j], auc, excess
        rows.append(
            {
                "family": family,
                "n_anomalous_rows": int(mask.sum()),
                "in_range_fraction": in_range_fraction,
                "single_feature_recall": single_recall,
                "single_feature_fpr": single_fpr,
                "multivariate_recall": multi_recall,
                "multivariate_fpr": multi_fpr,
                "max_marginal_auc": max_auc,
                "marginal_auc_excess": max_excess,
                "most_separating_feature": worst,
                "marginally_hidden": (
                    None if math.isnan(max_auc)
                    else bool(max_excess <= criteria.max_marginal_auc_excess) if not math.isnan(max_excess)
                    else bool(max_auc <= criteria.max_marginal_auc)
                ),
                "verdict": "SUPPORTED" if not reasons else "NOT SUPPORTED",
                "reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows).set_index("family")

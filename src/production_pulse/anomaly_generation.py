"""Seeded synthetic multivariate anomalies for clean, SCALED fault-free TEP data.

Why: the worklet needs anomalies that are abnormal as a *combination* of
variables while each variable stays inside its own normal range - a case
that real datasets rarely label. Whether a family achieves this is a
**hypothesis**: every generated family starts as ``UNVERIFIED`` and must be
measured with :func:`production_pulse.evaluation.assess_family_detectability`.

Workflow
--------
1. ``gen = AnomalyGenerator().fit(train_normal_scaled)`` - learns empirical
   per-feature ranges, correlations and covariance from TRAINING-normal data.
2. ``result = gen.generate(eval_scaled, [AnomalySpec("correlation_break", 0.05)],
   seed=42, groups=eval_meta["simulationRun"])`` - injects events into a copy
   of held-out data.
3. ``result.features`` (model input, 52 columns, original order & index) and
   ``result.row_info`` / ``result.labels`` / ``result.events`` (ground truth)
   are kept separate; metadata never enters the feature frame.

Anomaly families (``magnitude`` meaning in brackets)
-----------------------------------------------------
* ``coordinated_shift``  - sustained step on k random features with random
  signs [shift in training standard deviations of each feature].
* ``coordinated_drift``  - like the shift but ramping linearly from ~0 to the
  full magnitude over the event (slow joint drift).
* ``short_event``        - brief (default 2-6 samples) half-sine pulse on k
  features [peak shift in standard deviations].
* ``correlation_break``  - for a strongly correlated group, the deviations of
  the partner features from their mean are inverted, reversing the
  relationship with the untouched anchor feature [inversion strength
  alpha in (0, 1]; 1 = full mirror, 0.5 = pulled to the mean].
* ``decoupling``         - partner features of a correlated group are replaced
  by values from a *different* time/run of the same clean data, so each value
  is a genuine normal value but the relationship is broken [blend fraction
  beta in (0, 1]; 1 = full replacement].
* ``minor_component_shift`` - shift along a random combination of the
  lowest-variance principal directions of training data [shift in standard
  deviations *along that direction*; tiny per feature, large jointly].

Range constraint: the empirical normal range of each feature is the training
quantile interval ``[q_low, q_high]``. With ``range_policy="clip"`` (default)
an injection may not create a NEW range violation: each modified value is
clipped into ``[min(q_low, original), max(q_high, original)]``, i.e. into the
normal range, or no further out than the clean value already was (clean data
can lie slightly outside the training quantiles; clipping those values would
make anomalous rows look *more* normal than untouched rows and unfairly
handicap the rule-check baseline). The number of values that had to be
clipped is recorded per event. ``range_policy="none"`` disables clipping (for
ablation); ``modified_in_range`` is then False on rows with new violations.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS
from .features import feature_matrix, overlap_count, row_fingerprints, validate_feature_columns
from .logging_config import get_logger
from .paths import ProjectPaths, safe_output_path

log = get_logger(__name__)

PathLike = Union[str, os.PathLike]
UNVERIFIED = "UNVERIFIED - run evaluation.assess_family_detectability on held-out data"
RANGE_POLICIES = ("clip", "none")
MAX_TOTAL_ANOMALY_RATE = 0.5


class AnomalyGenerationError(ValueError):
    """Invalid generator configuration, input, or an impossible injection request."""


@dataclass(frozen=True)
class FamilyInfo:
    name: str
    description: str
    magnitude_meaning: str
    default_magnitude: float
    magnitude_bounds: tuple[float, float]  # (exclusive low, inclusive high)
    default_n_features: Optional[int]  # None -> all features (not user-selectable)
    min_n_features: int
    default_event_length: tuple[int, int]
    profile: str  # step | ramp | pulse


FAMILIES: dict[str, FamilyInfo] = {
    f.name: f
    for f in (
        FamilyInfo("coordinated_shift", "Sustained step on k random features with random signs",
                   "shift in training std of each feature", 1.5, (0.0, 10.0), 6, 2, (20, 60), "step"),
        FamilyInfo("coordinated_drift", "Linear ramp to the full shift on k random features",
                   "final shift in training std of each feature", 2.0, (0.0, 10.0), 6, 2, (40, 100), "ramp"),
        FamilyInfo("short_event", "Brief half-sine pulse on k random features",
                   "peak shift in training std of each feature", 2.0, (0.0, 10.0), 6, 2, (2, 6), "pulse"),
        FamilyInfo("correlation_break", "Invert partner deviations of a correlated group",
                   "inversion strength alpha (1 = mirror)", 1.0, (0.0, 1.0), 4, 2, (20, 60), "step"),
        FamilyInfo("decoupling", "Replace partners of a correlated group with values from another time",
                   "blend fraction beta (1 = full replacement)", 1.0, (0.0, 1.0), 4, 2, (20, 60), "step"),
        FamilyInfo("minor_component_shift", "Shift along low-variance principal directions",
                   "std along the chosen direction", 6.0, (0.0, 50.0), None, 1, (20, 60), "step"),
    )
}


def _profile(kind: str, length: int) -> np.ndarray:
    if kind == "step":
        return np.ones(length)
    if kind == "ramp":
        return np.arange(1, length + 1, dtype=np.float64) / length
    if kind == "pulse":
        return np.sin(np.pi * (np.arange(length) + 0.5) / length)
    raise AnomalyGenerationError(f"Unknown profile {kind!r}")  # pragma: no cover


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


@dataclass(frozen=True)
class AnomalySpec:
    """What to inject: one family at an approximate fraction of the rows."""

    family: str
    anomaly_rate: float
    magnitude: Optional[float] = None
    n_features: Optional[int] = None
    event_length: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise AnomalyGenerationError(f"Unknown family {self.family!r}; choose from {sorted(FAMILIES)}")
        info = FAMILIES[self.family]
        rate = self.anomaly_rate
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= MAX_TOTAL_ANOMALY_RATE:
            raise AnomalyGenerationError(f"anomaly_rate must be in (0, {MAX_TOTAL_ANOMALY_RATE}], got {rate!r}")
        if self.magnitude is not None:
            m = self.magnitude
            lo, hi = info.magnitude_bounds
            if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or not lo < m <= hi:
                raise AnomalyGenerationError(
                    f"{self.family}: magnitude ({info.magnitude_meaning}) must be in ({lo}, {hi}], got {m!r}"
                )
        if self.n_features is not None:
            if info.default_n_features is None:
                raise AnomalyGenerationError(f"{self.family} always uses all features; leave n_features=None")
            if not _is_int(self.n_features) or self.n_features < info.min_n_features:
                raise AnomalyGenerationError(f"{self.family}: n_features must be an int >= {info.min_n_features}")
        if self.event_length is not None:
            el = self.event_length
            if (not isinstance(el, (tuple, list)) or len(el) != 2 or not all(_is_int(v) for v in el)
                    or not 1 <= el[0] <= el[1]):
                raise AnomalyGenerationError("event_length must be (min, max) integers with 1 <= min <= max")
            object.__setattr__(self, "event_length", (int(el[0]), int(el[1])))

    def resolved(self) -> "AnomalySpec":
        """Copy with every unset field filled from the family defaults."""
        info = FAMILIES[self.family]
        return replace(
            self,
            magnitude=float(self.magnitude if self.magnitude is not None else info.default_magnitude),
            n_features=self.n_features if self.n_features is not None else info.default_n_features,
            event_length=self.event_length or info.default_event_length,
        )


@dataclass
class InjectionResult:
    """Injected features and ground truth, kept strictly separate."""

    features: pd.DataFrame  # model input only (feature columns, original order and index)
    row_info: pd.DataFrame  # per-row labels: is_anomaly, event_id, family, magnitude, ...
    events: pd.DataFrame  # one row per injected event
    settings: dict[str, Any]
    family_status: dict[str, str] = field(default_factory=dict)

    @property
    def labels(self) -> pd.Series:
        return self.row_info["is_anomaly"]

    def save(
        self,
        directory: PathLike,
        paths: Optional[ProjectPaths] = None,
        *,
        overwrite: bool = False,
        include_features: bool = False,
    ) -> dict[str, Path]:
        """Write ground truth (CSV/JSON) - and optionally features (Parquet) - to an output folder."""
        directory = Path(directory)
        out = {
            "row_info": safe_output_path(directory / "row_info.csv", paths, overwrite=overwrite),
            "events": safe_output_path(directory / "events.csv", paths, overwrite=overwrite),
            "settings": safe_output_path(directory / "settings.json", paths, overwrite=overwrite),
        }
        if include_features:
            out["features"] = safe_output_path(directory / "features.parquet", paths, overwrite=overwrite)
        self.row_info.to_csv(out["row_info"], index=True)
        self.events.to_csv(out["events"], index=False)
        meta = {"settings": self.settings, "family_status": self.family_status}
        out["settings"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if include_features:
            self.features.to_parquet(out["features"])  # needs pyarrow
        return out


class AnomalyGenerator:
    """Fit on training-normal scaled features, then inject anomalies into held-out data."""

    def __init__(
        self,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        *,
        range_quantiles: tuple[float, float] = (0.001, 0.999),
        correlation_threshold: float = 0.6,
        n_minor_components: int = 5,
        range_policy: str = "clip",
    ) -> None:
        self.feature_columns = validate_feature_columns(feature_columns)
        q_lo, q_hi = range_quantiles if isinstance(range_quantiles, (tuple, list)) and len(range_quantiles) == 2 else (None, None)
        if q_lo is None or not (0.0 <= q_lo < 0.5 < q_hi <= 1.0):
            raise AnomalyGenerationError("range_quantiles must be (low, high) with 0 <= low < 0.5 < high <= 1")
        if not isinstance(correlation_threshold, (int, float)) or not 0.0 < correlation_threshold < 1.0:
            raise AnomalyGenerationError("correlation_threshold must be in (0, 1)")
        if not _is_int(n_minor_components) or not 1 <= n_minor_components < len(self.feature_columns):
            raise AnomalyGenerationError("n_minor_components must be an int in [1, n_features - 1]")
        if range_policy not in RANGE_POLICIES:
            raise AnomalyGenerationError(f"range_policy must be one of {RANGE_POLICIES}")
        self.range_quantiles = (float(q_lo), float(q_hi))
        self.correlation_threshold = float(correlation_threshold)
        self.n_minor_components = int(n_minor_components)
        self.range_policy = range_policy
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, reference: pd.DataFrame) -> "AnomalyGenerator":
        """Learn ranges, correlations and covariance from TRAINING-normal scaled data."""
        x = feature_matrix(reference, self.feature_columns, what="reference")
        if x.shape[0] < 10:
            raise AnomalyGenerationError("reference needs at least 10 rows")
        self.lower_ = np.quantile(x, self.range_quantiles[0], axis=0)
        self.upper_ = np.quantile(x, self.range_quantiles[1], axis=0)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.varying_ = np.flatnonzero(self.std_ > 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(x, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        self.corr_ = corr
        self.cov_ = np.cov(x, rowvar=False)
        evals, evecs = np.linalg.eigh(self.cov_)  # ascending
        usable = np.flatnonzero(evals > max(evals.max(), 1e-300) * 1e-8)
        if usable.size < self.n_minor_components:
            raise AnomalyGenerationError("Not enough non-degenerate principal directions in reference data")
        pick = usable[: self.n_minor_components]
        self.minor_eigenvalues_ = evals[pick]
        self.minor_vectors_ = evecs[:, pick]
        self.anchor_candidates_ = np.flatnonzero((np.abs(corr) >= self.correlation_threshold).any(axis=1))
        self._ref_fp = row_fingerprints(x)
        self.n_reference_rows_ = int(x.shape[0])
        self._fitted = True
        log.info(
            "AnomalyGenerator fitted on %d rows; %d features have a partner with |r| >= %.2f",
            x.shape[0], self.anchor_candidates_.size, self.correlation_threshold,
        )
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise AnomalyGenerationError("Call fit() on training-normal scaled data first")

    def ranges(self) -> pd.DataFrame:
        """Empirical normal range per feature used for the in-range constraint."""
        self._require_fitted()
        return pd.DataFrame({"lower": self.lower_, "upper": self.upper_}, index=list(self.feature_columns))

    def correlated_pairs(self, top: int = 20) -> pd.DataFrame:
        """Strongest training correlations (|r| >= threshold) - the basis of the relationship families."""
        self._require_fitted()
        i, j = np.triu_indices_from(self.corr_, k=1)
        r = self.corr_[i, j]
        keep = np.abs(r) >= self.correlation_threshold
        order = np.argsort(-np.abs(r[keep]))[:top]
        cols = np.asarray(self.feature_columns)
        return pd.DataFrame({"feature_a": cols[i[keep]][order], "feature_b": cols[j[keep]][order], "r": r[keep][order]})

    # ------------------------------------------------------------------ helpers
    def _correlated_group(self, rng: np.random.Generator, k: int) -> tuple[int, np.ndarray]:
        if self.anchor_candidates_.size == 0:
            raise AnomalyGenerationError(
                f"No feature pair has |r| >= {self.correlation_threshold}; lower correlation_threshold"
            )
        anchor = int(rng.choice(self.anchor_candidates_))
        strength = np.abs(self.corr_[anchor])
        partners = [int(j) for j in np.argsort(-strength) if strength[j] >= self.correlation_threshold][: k - 1]
        return anchor, np.asarray(partners, dtype=int)

    def _donor_rows(self, rng, group_pos: dict, group_code: int, length: int, event_pos: np.ndarray, n: int) -> tuple[np.ndarray, Any]:
        others = [c for c, pos in group_pos.items() if c != group_code and pos.size >= length]
        if others:
            c = others[int(rng.integers(len(others)))]
            pos = group_pos[c]
            s = int(rng.integers(0, pos.size - length + 1))
            return pos[s : s + length], c
        candidates = np.setdiff1d(np.arange(n), event_pos)
        if candidates.size < length:
            raise AnomalyGenerationError("Not enough rows to draw decoupling donors")
        return np.sort(rng.choice(candidates, length, replace=False)), group_code

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        data: pd.DataFrame,
        specs: Union[AnomalySpec, Sequence[AnomalySpec]],
        *,
        seed: int,
        groups: Any = None,
        min_gap: int = 1,
        allow_reference_overlap: bool = False,
    ) -> InjectionResult:
        """Inject events into a COPY of ``data`` (clean, scaled, rows ordered in time within each group).

        ``groups``: run ID per row (e.g. ``metadata["simulationRun"]``); events never
        cross a group boundary. ``min_gap``: normal rows kept between events.
        Raises if ``data`` shares rows with the fit reference unless explicitly allowed.
        """
        self._require_fitted()
        if isinstance(specs, AnomalySpec):
            specs = [specs]
        if not specs or not all(isinstance(s, AnomalySpec) for s in specs):
            raise AnomalyGenerationError("specs must be an AnomalySpec or a non-empty list of them")
        if not _is_int(seed) or seed < 0:
            raise AnomalyGenerationError(f"seed must be a non-negative integer, got {seed!r}")
        if not _is_int(min_gap) or min_gap < 0:
            raise AnomalyGenerationError("min_gap must be a non-negative integer")
        specs = [s.resolved() for s in specs]
        total_rate = sum(s.anomaly_rate for s in specs)
        if total_rate > MAX_TOTAL_ANOMALY_RATE:
            raise AnomalyGenerationError(f"Total anomaly_rate {total_rate:.3f} exceeds {MAX_TOTAL_ANOMALY_RATE}")

        x0 = feature_matrix(data, self.feature_columns, what="data")  # fresh copy, input untouched
        n, p = x0.shape
        if not data.index.is_unique:
            raise AnomalyGenerationError("data index must be unique (row labels identify injected rows)")
        if not allow_reference_overlap:
            shared = overlap_count(row_fingerprints(x0), self._ref_fp)
            if shared:
                raise AnomalyGenerationError(
                    f"{shared} row(s) of data were used to fit the generator; inject into held-out data "
                    "or pass allow_reference_overlap=True"
                )
        if groups is None:
            g_values = np.zeros(n, dtype=np.int64)
        else:
            g_values = np.asarray(groups.to_numpy() if isinstance(groups, (pd.Series, pd.Index)) else groups)
            if g_values.ndim != 1 or g_values.size != n:
                raise AnomalyGenerationError(f"groups must have one value per row ({n}), got shape {g_values.shape}")
            if pd.isna(g_values).any():
                raise AnomalyGenerationError("groups contains missing values")
        codes, uniques = pd.factorize(g_values, sort=True)
        group_pos = {c: np.flatnonzero(codes == c) for c in range(len(uniques))}
        for s in specs:
            if s.n_features is not None and s.n_features > self.varying_.size:
                raise AnomalyGenerationError(f"{s.family}: n_features {s.n_features} > {self.varying_.size} usable features")
            if s.event_length[0] > max(pos.size for pos in group_pos.values()):
                raise AnomalyGenerationError(f"{s.family}: event_length longer than every group")

        rng = np.random.default_rng(int(seed))
        x = x0.copy()
        blocked = np.zeros(n, dtype=bool)
        is_anom = np.zeros(n, dtype=np.int8)
        event_id = np.full(n, -1, dtype=np.int64)
        family = np.full(n, "normal", dtype=object)
        magnitude = np.zeros(n)
        in_range = np.ones(n, dtype=bool)
        n_mod = np.zeros(n, dtype=np.int64)
        events: list[dict[str, Any]] = []
        cols = np.asarray(self.feature_columns)

        for spec in specs:
            info = FAMILIES[spec.family]
            target = int(round(spec.anomaly_rate * n))
            if target < 1:
                raise AnomalyGenerationError(f"{spec.family}: anomaly_rate {spec.anomaly_rate} gives 0 rows for n={n}")
            placed, tries, max_tries = 0, 0, 2000 + 100 * target
            lo, hi = spec.event_length
            while placed < target:
                tries += 1
                if tries > max_tries:
                    raise AnomalyGenerationError(
                        f"{spec.family}: could only place {placed}/{target} anomalous rows; "
                        "reduce anomaly_rate, event_length or min_gap"
                    )
                length = int(rng.integers(lo, hi + 1))
                eligible = [c for c, pos in group_pos.items() if pos.size >= length]
                if not eligible:
                    continue
                sizes = np.array([group_pos[c].size for c in eligible], dtype=np.float64)
                c = eligible[int(rng.choice(len(eligible), p=sizes / sizes.sum()))]
                pos_all = group_pos[c]
                start = int(rng.integers(0, pos_all.size - length + 1))
                pos = pos_all[start : start + length]
                if blocked[pos].any():
                    continue

                block = x[pos].copy()
                original = block.copy()
                anchor, donor_group = None, None
                if spec.family in ("coordinated_shift", "coordinated_drift", "short_event"):
                    idx = np.sort(rng.choice(self.varying_, spec.n_features, replace=False))
                    signs = rng.choice(np.array([-1.0, 1.0]), idx.size)
                    shift = spec.magnitude * self.std_[idx] * signs
                    block[:, idx] += _profile(info.profile, length)[:, None] * shift[None, :]
                elif spec.family == "correlation_break":
                    anchor, idx = self._correlated_group(rng, spec.n_features)
                    idx = np.sort(idx)
                    block[:, idx] -= 2.0 * spec.magnitude * (block[:, idx] - self.mean_[idx])
                elif spec.family == "decoupling":
                    anchor, idx = self._correlated_group(rng, spec.n_features)
                    idx = np.sort(idx)
                    donor_pos, donor_code = self._donor_rows(rng, group_pos, c, length, pos, n)
                    donor_group = uniques[donor_code]
                    block[:, idx] = (1 - spec.magnitude) * block[:, idx] + spec.magnitude * x0[np.ix_(donor_pos, idx)]
                else:  # minor_component_shift
                    w = rng.normal(size=self.n_minor_components)
                    direction = self.minor_vectors_ @ (w / np.linalg.norm(w))
                    direction /= np.linalg.norm(direction)
                    sd_along = math.sqrt(float(direction @ self.cov_ @ direction))
                    delta = spec.magnitude * sd_along * direction
                    block += delta[None, :]
                    idx = np.arange(p)

                sub = block[:, idx]
                lo_eff = np.minimum(self.lower_[idx], original[:, idx])
                hi_eff = np.maximum(self.upper_[idx], original[:, idx])
                outside = (sub < lo_eff) | (sub > hi_eff)  # violations CREATED by the injection
                n_outside = int(outside.sum())
                if self.range_policy == "clip":
                    block[:, idx] = np.clip(sub, lo_eff, hi_eff)
                    row_ok = np.ones(length, dtype=bool)
                else:
                    row_ok = ~outside.any(axis=1)
                x[pos] = block

                eid = len(events)
                is_anom[pos], event_id[pos], family[pos] = 1, eid, spec.family
                magnitude[pos], in_range[pos], n_mod[pos] = spec.magnitude, row_ok, idx.size
                blocked[pos_all[max(0, start - min_gap) : start + length + min_gap]] = True
                if spec.family == "minor_component_shift":
                    top = np.argsort(-np.abs(delta))[:6]
                    feat_desc = ";".join(cols[top])
                else:
                    feat_desc = ";".join(cols[idx])
                events.append(
                    {
                        "event_id": eid,
                        "family": spec.family,
                        "group": _py(uniques[c]),
                        "start_row": int(pos[0]),
                        "end_row": int(pos[-1]),
                        "start_index": _py(data.index[pos[0]]),
                        "end_index": _py(data.index[pos[-1]]),
                        "length": length,
                        "magnitude": spec.magnitude,
                        "n_modified_features": int(idx.size),
                        "features": feat_desc,  # display: modified features (top 6 for minor shift)
                        "modified_features": ";".join(cols[idx]),  # full list of modified features
                        "anchor_feature": "" if anchor is None else str(cols[anchor]),
                        "donor_group": "" if donor_group is None else str(_py(donor_group)),
                        "n_new_range_violations": n_outside,
                        "fraction_new_range_violations": n_outside / sub.size,
                        "range_policy": self.range_policy,
                    }
                )
                placed += length

        features = pd.DataFrame(x, index=data.index.copy(), columns=list(self.feature_columns))
        row_info = pd.DataFrame(
            {
                "is_anomaly": is_anom,
                "event_id": event_id,
                "family": pd.Series(family, dtype=object).to_numpy(),
                "magnitude": magnitude,
                "modified_in_range": in_range,
                "n_modified_features": n_mod,
            },
            index=data.index.copy(),
        )
        settings = {
            "seed": int(seed),
            "specs": [asdict(s) for s in specs],
            "range_quantiles": list(self.range_quantiles),
            "range_policy": self.range_policy,
            "correlation_threshold": self.correlation_threshold,
            "n_minor_components": self.n_minor_components,
            "min_gap": int(min_gap),
            "n_rows": int(n),
            "n_anomalous_rows": int(is_anom.sum()),
            "n_reference_rows": self.n_reference_rows_,
            "feature_columns": list(self.feature_columns),
            "grouped": groups is not None,
        }
        status = {s.family: UNVERIFIED for s in specs}
        log.info("Injected %d events (%d rows) with seed %d", len(events), int(is_anom.sum()), seed)
        return InjectionResult(features, row_info, pd.DataFrame(events), settings, status)


def _py(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value

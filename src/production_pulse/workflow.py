"""One place that prepares the leakage-free data sets used by every model.

Roles (all split by whole simulation runs, all seeded by ``config.random_seed``):

=============  =========================  ==========================================
role           source file                used for
=============  =========================  ==========================================
train          fault-free TRAINING file   fitting scaler, models, generator, limits
validation     fault-free TRAINING file   early stopping / model selection only
calibration    fault-free TESTING file    choosing detection thresholds only
evaluation     fault-free TESTING file    final reporting only (never for tuning)
holdout        fault-free TESTING file    LOCKED: live demo on unseen data
=============  =========================  ==========================================

Run IDs are only unique within one file, so each file is split separately.
The scaler is fitted on the ``train`` rows only. Holdout features are NOT
returned unless ``include_holdout=True`` is passed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .config import ProjectConfig, load_config
from .data import load_raw_dataset, split_metadata_features
from .logging_config import get_logger
from .paths import ProjectPaths, get_paths
from .preprocessing import FeatureScaler
from .splitting import RunSplit, apply_split, split_runs

log = get_logger(__name__)


@dataclass
class PreparedData:
    """Scaled feature frames (model input) and metadata frames per role."""

    scaler: FeatureScaler
    train_split: RunSplit       # training file: train / validation
    test_split: RunSplit        # testing file: calibration / evaluation / holdout
    features: dict[str, pd.DataFrame]
    metadata: dict[str, pd.DataFrame]

    def __getitem__(self, role: str) -> pd.DataFrame:
        if role not in self.features:
            hint = " (holdout is locked; pass include_holdout=True)" if role == "holdout" else ""
            raise KeyError(f"No prepared features for role {role!r}{hint}; available: {sorted(self.features)}")
        return self.features[role]

    def summary(self) -> pd.DataFrame:
        rows = []
        for role, split, source in (
            ("train", self.train_split, "fault_free_training"),
            ("validation", self.train_split, "fault_free_training"),
            ("calibration", self.test_split, "fault_free_testing"),
            ("evaluation", self.test_split, "fault_free_testing"),
            ("holdout", self.test_split, "fault_free_testing"),
        ):
            rows.append(
                {
                    "role": role,
                    "source": source,
                    "n_runs": len(split.keys(role)),
                    "n_rows": len(self.features[role]) if role in self.features else None,
                    "loaded": role in self.features,
                }
            )
        return pd.DataFrame(rows).set_index("role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_split": self.train_split.to_dict(),
            "test_split": self.test_split.to_dict(),
            "scaler": self.scaler.to_dict(),
        }


def prepare_fault_free_data(
    paths: Optional[ProjectPaths] = None,
    config: Optional[ProjectConfig] = None,
    *,
    include_holdout: bool = False,
    seed: Optional[int] = None,
) -> PreparedData:
    """Load both fault-free files (read-only), split by run and scale with TRAIN statistics."""
    paths = paths if paths is not None else get_paths(config=config)
    config = config if config is not None else load_config(root=paths.root)
    seed = config.random_seed if seed is None else seed

    train_raw = load_raw_dataset("fault_free_training", paths=paths, config=config)
    train_split = split_runs(train_raw, config.split.fractions(), seed=seed, group_columns=config.split.group_columns)
    train_parts = apply_split(train_raw, train_split)
    del train_raw

    test_raw = load_raw_dataset("fault_free_testing", paths=paths, config=config)
    test_split = split_runs(test_raw, config.evaluation.fractions(), seed=seed, group_columns=config.split.group_columns)
    test_parts = apply_split(test_raw, test_split)
    del test_raw
    if not include_holdout:
        test_parts.pop("holdout")

    scaler = FeatureScaler().fit(train_parts["train"], source="fault_free_training / train split")
    features: dict[str, pd.DataFrame] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for role, frame in {**train_parts, **test_parts}.items():
        meta, _ = split_metadata_features(frame)
        metadata[role] = meta
        features[role] = scaler.transform_frame(frame)
    log.info(
        "Prepared data: %s",
        {r: len(f) for r, f in features.items()} | {"holdout": "locked" if not include_holdout else len(features["holdout"])},
    )
    return PreparedData(scaler, train_split, test_split, features, metadata)

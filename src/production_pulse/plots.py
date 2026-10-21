"""Reusable figures: training history, score distributions, per-feature errors.

matplotlib is imported lazily. Functions return the Figure and never write to
disk; use :func:`save_figure` to store a figure in ``reports/figures`` through
the project's safe output paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .paths import ProjectPaths, get_paths, safe_output_path

PathLike = Union[str, os.PathLike]


class PlotError(ValueError):
    """Invalid plotting input."""


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for plotting (installed in .venv313)") from exc
    return plt


def _new_axes(ax, figsize):
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def plot_training_history(history: pd.DataFrame, *, best_epoch: Optional[int] = None, log_scale: bool = True, ax=None):
    """Training vs validation loss per epoch (marks the restored best epoch)."""
    if not isinstance(history, pd.DataFrame) or not {"loss", "val_loss"} <= set(history.columns) or history.empty:
        raise PlotError("history must be a non-empty DataFrame with 'loss' and 'val_loss' columns")
    fig, ax = _new_axes(ax, (7, 4))
    epochs = np.arange(1, len(history) + 1)
    ax.plot(epochs, history["loss"].to_numpy(), label="training loss")
    ax.plot(epochs, history["val_loss"].to_numpy(), label="validation loss")
    best = best_epoch if best_epoch is not None else int(np.argmin(history["val_loss"].to_numpy()) + 1)
    ax.axvline(best, color="k", ls="--", lw=0.8, label=f"best epoch ({best})")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Autoencoder training history")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_score_distributions(
    scores: Mapping[str, Any],
    *,
    threshold: Optional[float] = None,
    log_scale: bool = True,
    bins: int = 80,
    title: str = "Reconstruction error",
    ax=None,
):
    """Overlaid histograms of scores for named groups (e.g. normal vs anomalous)."""
    if not isinstance(scores, Mapping) or not scores:
        raise PlotError("scores must be a non-empty mapping {label: values}")
    clean = {}
    for label, values in scores.items():
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0 or not np.isfinite(arr).all():
            raise PlotError(f"scores for {label!r} must be non-empty and finite")
        if log_scale and (arr < 0).any():
            raise PlotError("log scale needs non-negative scores")
        clean[label] = np.log10(arr + 1e-12) if log_scale else arr
    fig, ax = _new_axes(ax, (7, 4))
    lo = min(v.min() for v in clean.values())
    hi = max(v.max() for v in clean.values())
    edges = np.linspace(lo, hi if hi > lo else lo + 1.0, bins + 1)
    for label, arr in clean.items():
        ax.hist(arr, bins=edges, density=True, alpha=0.55, label=f"{label} (n={arr.size:,})")
    if threshold is not None:
        ax.axvline(np.log10(threshold + 1e-12) if log_scale else threshold, color="k", ls="--", lw=1, label="threshold")
    ax.set_xlabel("log10(score)" if log_scale else "score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_errors(errors: Union[pd.DataFrame, pd.Series], *, top: int = 15, title: str = "Mean reconstruction error per feature", ax=None):
    """Bar chart of the features contributing most to reconstruction error."""
    if isinstance(errors, pd.DataFrame):
        series = errors.mean(axis=0)
    elif isinstance(errors, pd.Series):
        series = errors
    else:
        raise PlotError("errors must be a DataFrame (rows x features) or a Series (per feature)")
    if series.empty or not np.isfinite(series.to_numpy(dtype=np.float64)).all():
        raise PlotError("errors must be non-empty and finite")
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise PlotError("top must be a positive integer")
    series = series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = _new_axes(ax, (6, 0.3 * len(series) + 1.5))
    ax.barh(series.index.astype(str), series.to_numpy())
    ax.set_xlabel("squared error")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figure(fig, filename: str, paths: Optional[ProjectPaths] = None, *, overwrite: bool = False, dpi: int = 150) -> Path:
    """Save into ``reports/figures`` (only plain file names, e.g. ``'ae_history.png'``)."""
    if not isinstance(filename, str) or Path(filename).name != filename or not filename:
        raise PlotError("filename must be a plain file name such as 'ae_history.png'")
    if Path(filename).suffix.lower() not in {".png", ".svg", ".pdf", ".jpg"}:
        raise PlotError("figure file must end in .png, .svg, .pdf or .jpg")
    paths = paths or get_paths()
    target = safe_output_path(paths.figures_dir / filename, paths, overwrite=overwrite)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return target

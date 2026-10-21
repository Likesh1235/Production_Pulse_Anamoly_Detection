"""Tests for plotting helpers (headless backend; files only in the fake project)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from production_pulse.config import default_config  # noqa: E402
from production_pulse.paths import get_paths  # noqa: E402
from production_pulse.plots import (  # noqa: E402
    PlotError,
    plot_feature_errors,
    plot_score_distributions,
    plot_training_history,
    save_figure,
)

HIST = pd.DataFrame({"loss": [1.0, 0.5, 0.4, 0.45], "val_loss": [1.1, 0.6, 0.5, 0.55]})


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_history_plot():
    fig = plot_training_history(HIST)
    ax = fig.axes[0]
    assert len(ax.lines) == 3 and ax.get_yscale() == "log"
    with pytest.raises(PlotError):
        plot_training_history(HIST.drop(columns="val_loss"))
    with pytest.raises(PlotError):
        plot_training_history(pd.DataFrame(columns=["loss", "val_loss"]))


def test_score_distribution_plot():
    rng = np.random.default_rng(0)
    fig = plot_score_distributions({"normal": rng.gamma(2, size=500), "anomalous": rng.gamma(5, size=50)}, threshold=3.0)
    assert len(fig.axes[0].patches) > 0
    for bad in ({}, {"x": []}, {"x": [1.0, np.nan]}, {"x": [-1.0, 2.0]}):
        with pytest.raises(PlotError):
            plot_score_distributions(bad)


def test_feature_error_plot():
    errors = pd.DataFrame(np.abs(np.random.default_rng(1).normal(size=(20, 52))), columns=[f"f{i}" for i in range(52)])
    fig = plot_feature_errors(errors, top=5)
    assert len(fig.axes[0].patches) == 5
    with pytest.raises(PlotError):
        plot_feature_errors([1, 2, 3])
    with pytest.raises(PlotError):
        plot_feature_errors(errors, top=0)


def test_save_figure_only_into_figures_folder(fake_root):
    paths = get_paths(fake_root, config=default_config())
    target = save_figure(plot_training_history(HIST), "history.png", paths)
    assert target.is_file() and target.parent == paths.figures_dir
    with pytest.raises(FileExistsError):
        save_figure(plot_training_history(HIST), "history.png", paths)
    for bad in ("../escape.png", "sub/dir.png", "figure.txt", ""):
        with pytest.raises(PlotError):
            save_figure(plot_training_history(HIST), bad, paths)

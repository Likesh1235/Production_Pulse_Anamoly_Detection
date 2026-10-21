"""Shared pytest fixtures. Tests never write inside the real project folder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Fallback so tests work even if pytest's `pythonpath` setting is not applied.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from production_pulse.config import COURSE_FILES, RAW_DATA_FILES  # noqa: E402
from production_pulse.paths import ROOT_ENV_VAR  # noqa: E402

FAKE_RAW_CONTENT = b"ORIGINAL-DATA-DO-NOT-TOUCH"


@pytest.fixture(autouse=True)
def _clear_root_env(monkeypatch):
    """Make sure a PRODUCTION_PULSE_ROOT set on the machine cannot affect tests."""
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A throw-away project root in a temp folder, with dummy 'original' files."""
    root = tmp_path / "fake_project"
    (root / "src" / "production_pulse").mkdir(parents=True)
    (root / "src" / "production_pulse" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "DATA_CHECKSUMS.md").write_text("baseline", encoding="utf-8")
    for name in (*RAW_DATA_FILES.values(), *COURSE_FILES):
        (root / name).write_bytes(FAKE_RAW_CONTENT)
    return root


# ---------------------------------------------------------------- Phase 2 helpers
def make_tep_frame(n_runs: int = 10, n_samples: int = 20, fault: int = 0, seed: int = 0):
    """Small SYNTHETIC frame with the real TEP column layout (not real data)."""
    import numpy as np
    import pandas as pd

    from production_pulse.data import FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    n = n_runs * n_samples
    frame = {
        "faultNumber": np.full(n, float(fault)),
        "simulationRun": np.repeat(np.arange(1, n_runs + 1, dtype=float), n_samples),
        "sample": np.tile(np.arange(1, n_samples + 1, dtype=np.int32), n_runs),
    }
    for j, col in enumerate(FEATURE_COLUMNS):
        frame[col] = rng.normal(loc=10.0 * (j + 1), scale=1.0 + j % 5, size=n)
    return pd.DataFrame(frame)


@pytest.fixture
def tep_frame():
    return make_tep_frame()


# ---------------------------------------------------------------- Phase 3 helpers
def make_correlated_frame(n_runs: int = 20, n_samples: int = 50, seed: int = 0, n_factors: int = 5, noise: float = 0.15):
    """SYNTHETIC frame with strong cross-feature correlations (latent-factor model).

    Same column layout as TEP; features = factors @ loadings + small noise, then
    roughly standardised. Not real data - used to test anomaly families/detectors.
    """
    import numpy as np
    import pandas as pd

    from production_pulse.data import FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    loadings = np.random.default_rng(1234).normal(size=(n_factors, len(FEATURE_COLUMNS)))  # fixed structure
    n = n_runs * n_samples
    feats = rng.normal(size=(n, n_factors)) @ loadings + noise * rng.normal(size=(n, len(FEATURE_COLUMNS)))
    feats = feats / np.sqrt((loadings**2).sum(axis=0) + noise**2)
    frame = pd.DataFrame(feats, columns=list(FEATURE_COLUMNS))
    frame.insert(0, "sample", np.tile(np.arange(1, n_samples + 1), n_runs))
    frame.insert(0, "simulationRun", np.repeat(np.arange(1, n_runs + 1), n_samples).astype(float))
    frame.insert(0, "faultNumber", 0.0)
    return frame

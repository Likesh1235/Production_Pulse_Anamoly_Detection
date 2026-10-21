"""Light checks for scripts/run_experiments.py (no training, no real data)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from production_pulse.config import AutoencoderSettings, load_config

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"


@pytest.fixture(scope="module")
def exp():
    spec = importlib.util.spec_from_file_location("run_experiments", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_noise_is_deterministic_and_zero_is_identity(exp):
    df = pd.DataFrame(np.arange(20.0).reshape(5, 4), columns=list("abcd"))
    assert exp._noisy(df, 0.0, 1) is df
    a, b = exp._noisy(df, 0.5, 7), exp._noisy(df, 0.5, 7)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.to_numpy(), df.to_numpy())
    assert list(a.index) == list(df.index) and list(a.columns) == list(df.columns)


def test_every_ablation_variant_is_a_valid_distinct_configuration(exp):
    base = load_config().autoencoder
    seen = set()
    for name, (over, seed) in exp.ABLATION_VARIANTS.items():
        s = AutoencoderSettings(**{**base.__dict__, **over})   # raises ConfigError if invalid
        key = (tuple(sorted(over.items())), seed)
        assert key not in seen, name
        seen.add(key)
        assert (over or seed != exp.SEED), f"{name} would duplicate the saved model"


def test_saved_model_path_is_never_an_experiment_output(exp):
    assert exp.MODEL_DIR == "artifacts/models/autoencoder_v1"
    src = _SCRIPT.read_text(encoding="utf-8")
    assert 'f"artifacts/experiments/{tag}/{name}"' in src and "overwrite=False" in src

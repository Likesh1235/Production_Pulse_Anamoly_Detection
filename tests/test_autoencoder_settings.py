"""Autoencoder and evaluation settings (no TensorFlow needed)."""

from __future__ import annotations

import pytest

from production_pulse.config import (
    AutoencoderSettings,
    ConfigError,
    EvaluationConfig,
    config_from_dict,
    default_config,
)


def test_defaults_match_architecture_plan():
    s = default_config().autoencoder
    assert s.hidden_layers == (32, 16) and s.bottleneck == 8          # 52-32-16-8-16-32-52
    assert s.activation == "relu" and s.output_activation == "linear"
    assert s.patience == 10 and s.epochs == 200 and s.deterministic is True


def test_evaluation_defaults():
    e = default_config().evaluation
    assert e.fractions() == {"calibration": 0.2, "evaluation": 0.7, "holdout": 0.1}
    assert e.target_fpr == 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_layers": (16, 32)},           # encoder must narrow
        {"hidden_layers": (32, 0)},
        {"hidden_layers": (32.0, 16)},
        {"bottleneck": 0},
        {"bottleneck": 64},                    # wider than hidden layers
        {"activation": "swishy"},
        {"output_activation": "softmax"},
        {"dropout": 0.95},
        {"l2": -1e-3},
        {"noise_std": -0.1},
        {"learning_rate": 0.0},
        {"batch_size": 0},
        {"epochs": 0},
        {"patience": -1},
        {"min_delta": -1.0},
    ],
)
def test_invalid_autoencoder_settings(kwargs):
    with pytest.raises(ConfigError):
        AutoencoderSettings(**kwargs)


def test_shallow_autoencoder_allowed():
    assert AutoencoderSettings(hidden_layers=(), bottleneck=10).hidden_layers == ()


def test_dict_roundtrip_and_type_checks():
    s = AutoencoderSettings(hidden_layers=(24, 12), bottleneck=6, dropout=0.1)
    assert AutoencoderSettings.from_dict(s.to_dict()) == s
    with pytest.raises(ConfigError, match="Unknown"):
        AutoencoderSettings.from_dict({**s.to_dict(), "layers": [1]})
    with pytest.raises(ConfigError):
        AutoencoderSettings.from_dict({**s.to_dict(), "epochs": "200"})


def test_toml_sections_parse():
    c = config_from_dict({"autoencoder": {"hidden_layers": [40, 20], "bottleneck": 10, "learning_rate": 0.0005},
                          "evaluation": {"calibration_fraction": 0.25, "evaluation_fraction": 0.65,
                                         "holdout_fraction": 0.1, "target_fpr": 0.005}})
    assert c.autoencoder.hidden_layers == (40, 20) and c.autoencoder.learning_rate == 0.0005
    assert c.evaluation.target_fpr == 0.005


@pytest.mark.parametrize(
    "kwargs",
    [
        {"calibration_fraction": 0.3},                      # no longer sums to 1
        {"holdout_fraction": 0.0, "evaluation_fraction": 0.8},
        {"target_fpr": 0.0},
        {"target_fpr": 0.6},
    ],
)
def test_invalid_evaluation_config(kwargs):
    with pytest.raises(ConfigError):
        EvaluationConfig(**kwargs)

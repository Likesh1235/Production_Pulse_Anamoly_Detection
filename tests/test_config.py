"""Configuration default and validation tests."""

from __future__ import annotations

import dataclasses

import pytest

from production_pulse import config as cfg_mod
from production_pulse.config import (
    ConfigError,
    PathsConfig,
    ProjectConfig,
    config_from_dict,
    default_config,
    load_config,
)
from production_pulse.paths import find_project_root

needs_toml = pytest.mark.skipif(cfg_mod._toml is None, reason="TOML reader not available (Python < 3.11 without tomli)")


def test_defaults():
    c = default_config()
    assert c.random_seed == 42
    assert c.paths.raw_data_dir == "."
    assert c.paths.data_dir == "data"
    assert c.paths.models_dir == "artifacts/models"
    assert c.paths.figures_dir == "reports/figures"
    assert c.data.id_columns == ("faultNumber", "simulationRun", "sample")
    assert c.data.n_process_variables == 52
    assert c.logging.level == "INFO"


def test_config_is_immutable():
    c = default_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.random_seed = 1  # type: ignore[misc]


@needs_toml
def test_shipped_default_toml_matches_builtin_defaults():
    assert load_config(root=find_project_root()) == ProjectConfig()


@needs_toml
def test_toml_override(tmp_path):
    f = tmp_path / "custom.toml"
    f.write_text('random_seed = 7\n[logging]\nlevel = "DEBUG"\n', encoding="utf-8")
    c = load_config(f)
    assert c.random_seed == 7
    assert c.logging.level == "DEBUG"
    assert c.paths == PathsConfig()  # untouched sections keep defaults


def test_missing_explicit_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_missing_default_file_falls_back_to_defaults(fake_root):
    assert load_config(root=fake_root) == ProjectConfig()


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown"):
        config_from_dict({"paths": {"dat_dir": "x"}})


@pytest.mark.parametrize(
    "data",
    [
        {"random_seed": "42"},
        {"random_seed": True},
        {"logging": {"console": "yes"}},
        {"data": {"id_columns": "faultNumber"}},
        {"paths": "data"},
    ],
)
def test_wrong_types_rejected(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


@pytest.mark.parametrize("bad", ["../outside", "data/../../x", "/abs/path", "C:/Users/x", r"C:\data", ".", ".venv313/out", ""])
def test_unsafe_output_dirs_rejected(bad):
    with pytest.raises(ConfigError):
        PathsConfig(artifacts_dir=bad)


def test_invalid_log_settings_rejected():
    with pytest.raises(ConfigError):
        config_from_dict({"logging": {"level": "LOUD"}})
    with pytest.raises(ConfigError):
        config_from_dict({"logging": {"log_file_name": "../escape.log"}})


def test_negative_seed_rejected():
    with pytest.raises(ConfigError):
        config_from_dict({"random_seed": -1})


# ---------------------------------------------------------------- Phase 2 additions
def test_phase2_defaults():
    c = default_config()
    assert c.data.max_file_size_mb == 100
    assert c.split.fractions() == {"train": 0.8, "validation": 0.2}
    assert c.split.group_columns == ("simulationRun",)


@pytest.mark.parametrize(
    "split",
    [
        {"train_fraction": 0.7},                       # no longer sums to 1
        {"train_fraction": 1.0, "validation_fraction": 0.0},
        {"train_fraction": "0.8"},                     # wrong type
        {"group_columns": []},
    ],
)
def test_invalid_split_config(split):
    with pytest.raises(ConfigError):
        config_from_dict({"split": split})


def test_split_fraction_accepts_numbers():
    c = config_from_dict({"split": {"train_fraction": 0.75, "validation_fraction": 0.25}})
    assert c.split.train_fraction == 0.75


def test_invalid_size_guard():
    with pytest.raises(ConfigError):
        config_from_dict({"data": {"max_file_size_mb": 0}})

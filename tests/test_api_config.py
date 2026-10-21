"""[api] configuration and environment overrides (no FastAPI needed)."""

from __future__ import annotations

import pytest

from production_pulse.api.settings import apply_env_overrides, read_admin_token
from production_pulse.config import ApiConfig, ConfigError, config_from_dict, default_config


def test_defaults_are_local_and_restricted():
    api = default_config().api
    assert api.host == "127.0.0.1" and api.port == 8000
    assert api.model_dir == "artifacts/models/autoencoder_v1"
    assert api.database_path.startswith("data/")
    assert api.cors_origins == (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    )
    assert "*" not in api.cors_origins and all(o.startswith("http://") for o in api.cors_origins)
    assert api.store_inputs is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": 0}, {"port": 70000}, {"max_batch_size": 0}, {"host": "local host"}, {"host": ""},
        {"model_dir": "/abs/model"}, {"model_dir": "C:/models"}, {"model_dir": "../outside"}, {"model_dir": "."},
        {"database_path": "data/api/history.csv"}, {"database_path": "../x.sqlite3"},
        {"metrics_report": "reports/metrics.txt"},
        {"cors_origins": ("*",)}, {"cors_origins": ("localhost:5173",)}, {"cors_origins": ("http://localhost:5173/",)},
        {"cors_origins": ("ftp://localhost",)}, {"top_features": 0}, {"top_features": 53},
        {"history_default_page_size": 600},
    ],
)
def test_invalid_api_config(kwargs):
    with pytest.raises(ConfigError):
        ApiConfig(**kwargs)


def test_toml_style_section():
    c = config_from_dict({"api": {"port": 8100, "cors_origins": ["http://localhost:3000"], "store_inputs": True}})
    assert c.api.port == 8100 and c.api.cors_origins == ("http://localhost:3000",) and c.api.store_inputs


def test_env_overrides_are_applied_and_validated():
    base = default_config().api
    env = {
        "PP_API_PORT": "8001", "PP_API_HOST": "0.0.0.0", "PP_API_MAX_BATCH_SIZE": "10",
        "PP_API_MODEL_DIR": "artifacts/models/v2", "PP_API_DATABASE_PATH": "data/api/demo.db",
        "PP_API_CORS_ORIGINS": "http://localhost:3000, http://127.0.0.1:3000", "PP_API_STORE_INPUTS": "yes",
    }
    api = apply_env_overrides(base, env)
    assert (api.port, api.host, api.max_batch_size) == (8001, "0.0.0.0", 10)
    assert api.cors_origins == ("http://localhost:3000", "http://127.0.0.1:3000") and api.store_inputs
    assert apply_env_overrides(base, {}) is base
    assert apply_env_overrides(base, {"PP_API_CORS_ORIGINS": ""}).cors_origins == ()


@pytest.mark.parametrize(
    "env",
    [{"PP_API_PORT": "eighty"}, {"PP_API_PORT": "99999"}, {"PP_API_CORS_ORIGINS": "*"},
     {"PP_API_MODEL_DIR": "../../secret"}, {"PP_API_STORE_INPUTS": "maybe"}],
)
def test_invalid_env_overrides_fail_fast(env):
    with pytest.raises(ConfigError):
        apply_env_overrides(default_config().api, env)


def test_admin_token_comes_from_environment_only():
    assert read_admin_token({}) is None and read_admin_token({"PP_API_ADMIN_TOKEN": ""}) is None
    assert read_admin_token({"PP_API_ADMIN_TOKEN": "x" * 20}) == "x" * 20
    with pytest.raises(ConfigError):
        read_admin_token({"PP_API_ADMIN_TOKEN": "short"})
    assert "admin" not in " ".join(ApiConfig().to_dict())       # no secret field in the config

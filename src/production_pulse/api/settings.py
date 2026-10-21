"""API configuration: ``[api]`` from configs/default.toml plus environment overrides.

Environment variables (all optional):

==========================  ===========================  ==========================
variable                    overrides                    example
==========================  ===========================  ==========================
PP_API_HOST                 api.host                     127.0.0.1
PP_API_PORT                 api.port                     8001
PP_API_MAX_BATCH_SIZE       api.max_batch_size           500
PP_API_MODEL_DIR            api.model_dir                artifacts/models/autoencoder_v1
PP_API_DATABASE_PATH        api.database_path            data/api/demo.sqlite3
PP_API_METRICS_REPORT       api.metrics_report           reports/evaluation_metrics.json
PP_API_CORS_ORIGINS         api.cors_origins (comma)     http://localhost:5173,http://localhost:3000
PP_API_STORE_INPUTS         api.store_inputs             false
PP_API_ADMIN_TOKEN          (secret, env only)           enables POST /model/reload
==========================  ===========================  ==========================

Overrides are validated exactly like the TOML values (invalid values fail fast).
This module does not import FastAPI.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional

from ..config import ApiConfig, ConfigError

ADMIN_TOKEN_ENV = "PP_API_ADMIN_TOKEN"
MIN_ADMIN_TOKEN_LENGTH = 16

_ENV_FIELDS: dict[str, tuple[str, type]] = {
    "PP_API_HOST": ("host", str),
    "PP_API_PORT": ("port", int),
    "PP_API_MAX_BATCH_SIZE": ("max_batch_size", int),
    "PP_API_MODEL_DIR": ("model_dir", str),
    "PP_API_DATABASE_PATH": ("database_path", str),
    "PP_API_METRICS_REPORT": ("metrics_report", str),
    "PP_API_CORS_ORIGINS": ("cors_origins", tuple),
    "PP_API_STORE_INPUTS": ("store_inputs", bool),
}
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


def apply_env_overrides(api: ApiConfig, environ: Mapping[str, str]) -> ApiConfig:
    """Return ``api`` with any PP_API_* environment overrides applied and validated."""
    changes = {}
    for var, (field_name, kind) in _ENV_FIELDS.items():
        raw = environ.get(var)
        if raw is None:
            continue
        raw = raw.strip()
        if kind is int:
            try:
                changes[field_name] = int(raw)
            except ValueError:
                raise ConfigError(f"{var} must be an integer, got {raw!r}") from None
        elif kind is bool:
            if raw.lower() not in _TRUE | _FALSE:
                raise ConfigError(f"{var} must be true/false, got {raw!r}")
            changes[field_name] = raw.lower() in _TRUE
        elif kind is tuple:
            changes[field_name] = tuple(o.strip() for o in raw.split(",") if o.strip())
        else:
            changes[field_name] = raw
    return replace(api, **changes) if changes else api  # replace() re-runs validation


def read_admin_token(environ: Mapping[str, str]) -> Optional[str]:
    """Admin token for deliberate model reloads; None disables the reload endpoint."""
    token = environ.get(ADMIN_TOKEN_ENV)
    if token is None or token == "":
        return None
    if len(token) < MIN_ADMIN_TOKEN_LENGTH:
        raise ConfigError(f"{ADMIN_TOKEN_ENV} must be at least {MIN_ADMIN_TOKEN_LENGTH} characters")
    return token

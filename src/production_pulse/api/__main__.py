"""``python -m production_pulse.api`` - start Uvicorn with host/port from the config (+ env overrides)."""

from __future__ import annotations

import os

import uvicorn

from ..config import load_config
from ..logging_config import setup_logging
from ..paths import get_paths
from .settings import apply_env_overrides


def main() -> None:
    paths = get_paths()
    config = load_config(root=paths.root)
    api_cfg = apply_env_overrides(config.api, os.environ)
    setup_logging(config=config, paths=paths)
    uvicorn.run("production_pulse.api.main:app", host=api_cfg.host, port=api_cfg.port, reload=False)


if __name__ == "__main__":
    main()

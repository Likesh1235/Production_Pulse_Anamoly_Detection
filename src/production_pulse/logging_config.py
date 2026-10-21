"""Logging setup: console + UTF-8 log file inside the logs/ output folder.

``setup_logging`` is idempotent (safe to call again in a notebook) and only
manages handlers it created itself. Call ``shutdown_logging`` to release the
log file (important on Windows, where open files cannot be removed).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import ProjectConfig, load_config
from .paths import ProjectPaths, get_paths, safe_output_path

LOGGER_NAME = "production_pulse"
_TAG = "_production_pulse_handler"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _own_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, _TAG, False)]


def shutdown_logging() -> None:
    """Remove and close handlers added by :func:`setup_logging`."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in _own_handlers(logger):
        logger.removeHandler(handler)
        handler.close()


def setup_logging(
    level: Optional[str] = None,
    *,
    config: Optional[ProjectConfig] = None,
    paths: Optional[ProjectPaths] = None,
    console: Optional[bool] = None,
    log_to_file: bool = True,
) -> logging.Logger:
    """Configure the ``production_pulse`` logger and return it."""
    if paths is None:
        paths = get_paths(config=config)
    if config is None:
        config = load_config(root=paths.root)
    lvl = (level or config.logging.level).upper()
    use_console = config.logging.console if console is None else console

    shutdown_logging()  # idempotent: drop our previous handlers
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(lvl)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if use_console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        setattr(stream, _TAG, True)
        logger.addHandler(stream)

    if log_to_file:
        log_path: Path = safe_output_path(paths.logs_dir / config.logging.log_file_name, paths, overwrite=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _TAG, True)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Child logger, e.g. ``get_logger(__name__)`` -> production_pulse.<module>."""
    if name == LOGGER_NAME or name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")

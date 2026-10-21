"""Production Pulse — multivariate process anomaly detection (TEP + autoencoder).

Importing this package is lightweight: it does not import TensorFlow or load data.
"""

__version__ = "0.1.0"

from .config import ConfigError, ProjectConfig, default_config, load_config
from .logging_config import get_logger, setup_logging, shutdown_logging
from .paths import (
    ProjectPaths,
    ProjectRootNotFoundError,
    UnsafeOutputPathError,
    ensure_output_dirs,
    find_project_root,
    get_paths,
    is_protected,
    safe_output_path,
)

__all__ = [
    "__version__",
    "ConfigError",
    "ProjectConfig",
    "default_config",
    "load_config",
    "get_logger",
    "setup_logging",
    "shutdown_logging",
    "ProjectPaths",
    "ProjectRootNotFoundError",
    "UnsafeOutputPathError",
    "ensure_output_dirs",
    "find_project_root",
    "get_paths",
    "is_protected",
    "safe_output_path",
]

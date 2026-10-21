"""Project configuration: typed defaults plus an optional TOML override file.

Defaults live in the dataclasses below. ``configs/default.toml`` mirrors them
and may be edited; unknown keys or wrong types raise :class:`ConfigError`
instead of being silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Optional, Union

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:
        _toml = None

PathLike = Union[str, Path]

DEFAULT_CONFIG_RELATIVE = "configs/default.toml"

# Original files supplied by the student. These are NEVER written by the code.
RAW_DATA_FILES: dict[str, str] = {
    "fault_free_training": "TEP_FaultFree_Training.RData",
    "fault_free_testing": "TEP_FaultFree_Testing.RData",
    "faulty_training": "TEP_Faulty_Training.RData",
    "faulty_testing": "TEP_Faulty_Testing.RData",
}
COURSE_FILES: tuple[str, ...] = (
    "Mark Split 21CSE732P rubrics .pdf",
    "Applied AI Industry Worklets Student Template.xlsx",
)
# Paths relative to the project root that output helpers must refuse to write.
PROTECTED_RELATIVE_PATHS: tuple[str, ...] = (
    *RAW_DATA_FILES.values(),
    *COURSE_FILES,
    "docs/DATA_CHECKSUMS.md",
    "docs/_access_check.txt",
)

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when a configuration value or file is invalid."""


def _check_relative_dir(name: str, value: str, *, allow_root: bool) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"paths.{name} must be a non-empty string")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise ConfigError(f"paths.{name} must be relative to the project root, got {value!r}")
    parts = PureWindowsPath(value).parts  # splits on both / and \
    if ".." in parts:
        raise ConfigError(f"paths.{name} must not contain '..', got {value!r}")
    if not allow_root and all(p == "." for p in parts):
        raise ConfigError(f"paths.{name} must be a sub-folder, not the project root")
    if any(p.lower().startswith(".venv") for p in parts):
        raise ConfigError(f"paths.{name} must not point inside a virtual environment")


@dataclass(frozen=True)
class PathsConfig:
    raw_data_dir: str = "."
    data_dir: str = "data"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    logs_dir: str = "logs"

    def __post_init__(self) -> None:
        for f in fields(self):
            _check_relative_dir(f.name, getattr(self, f.name), allow_root=(f.name == "raw_data_dir"))

    def output_dir_names(self) -> tuple[str, ...]:
        """Names of the fields that point at generated-output folders."""
        return tuple(f.name for f in fields(self) if f.name != "raw_data_dir")


@dataclass(frozen=True)
class DataConfig:
    id_columns: tuple[str, ...] = ("faultNumber", "simulationRun", "sample")
    n_process_variables: int = 52
    # Safety guard: refuse to load .RData files larger than this (MB on disk)
    # unless the caller explicitly opts in. Keeps the two faulty files
    # (494 MB / 837 MB) from being loaded by accident.
    max_file_size_mb: int = 100

    def __post_init__(self) -> None:
        if not self.id_columns or not all(isinstance(c, str) and c for c in self.id_columns):
            raise ConfigError("data.id_columns must be a non-empty list of strings")
        if self.n_process_variables <= 0:
            raise ConfigError("data.n_process_variables must be positive")
        if self.max_file_size_mb <= 0:
            raise ConfigError("data.max_file_size_mb must be positive")


@dataclass(frozen=True)
class SplitConfig:
    """Run-level split of the fault-free *training* file into train/validation."""

    train_fraction: float = 0.8
    validation_fraction: float = 0.2
    group_columns: tuple[str, ...] = ("simulationRun",)

    def __post_init__(self) -> None:
        for name in ("train_fraction", "validation_fraction"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"split.{name} must be between 0 and 1 (exclusive), got {value}")
        if abs(self.train_fraction + self.validation_fraction - 1.0) > 1e-9:
            raise ConfigError("split.train_fraction + split.validation_fraction must equal 1")
        if not self.group_columns or not all(isinstance(c, str) and c for c in self.group_columns):
            raise ConfigError("split.group_columns must be a non-empty list of column names")

    def fractions(self) -> dict[str, float]:
        return {"train": self.train_fraction, "validation": self.validation_fraction}


@dataclass(frozen=True)
class EvaluationConfig:
    """Run-level split of the fault-free *testing* file and the calibration target.

    calibration: threshold calibration only; evaluation: final reporting only
    (never used for tuning); holdout: locked for the live demo on unseen data.
    """

    calibration_fraction: float = 0.2
    evaluation_fraction: float = 0.7
    holdout_fraction: float = 0.1
    target_fpr: float = 0.01

    def __post_init__(self) -> None:
        for name in ("calibration_fraction", "evaluation_fraction", "holdout_fraction"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"evaluation.{name} must be between 0 and 1 (exclusive), got {value}")
        total = self.calibration_fraction + self.evaluation_fraction + self.holdout_fraction
        if abs(total - 1.0) > 1e-9:
            raise ConfigError("evaluation calibration/evaluation/holdout fractions must sum to 1")
        if not 0.0 < self.target_fpr < 0.5:
            raise ConfigError(f"evaluation.target_fpr must be in (0, 0.5), got {self.target_fpr}")

    def fractions(self) -> dict[str, float]:
        return {
            "calibration": self.calibration_fraction,
            "evaluation": self.evaluation_fraction,
            "holdout": self.holdout_fraction,
        }


_ACTIVATIONS = ("relu", "tanh", "elu", "selu", "gelu", "sigmoid", "linear")


@dataclass(frozen=True)
class AutoencoderSettings:
    """Dense autoencoder: 52 -> hidden_layers -> bottleneck -> reversed(hidden_layers) -> 52."""

    hidden_layers: tuple[int, ...] = (32, 16)
    bottleneck: int = 8
    activation: str = "relu"
    output_activation: str = "linear"
    dropout: float = 0.0
    l2: float = 0.0
    noise_std: float = 0.0          # >0 -> denoising autoencoder (Gaussian input noise, training only)
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 200
    patience: int = 10              # early stopping on validation loss
    min_delta: float = 0.0
    deterministic: bool = True      # request deterministic TensorFlow ops

    def __post_init__(self) -> None:
        h = self.hidden_layers
        if not isinstance(h, tuple) or not all(isinstance(w, int) and not isinstance(w, bool) and w > 0 for w in h):
            raise ConfigError("autoencoder.hidden_layers must be a list of positive integers")
        b = self.bottleneck
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ConfigError("autoencoder.bottleneck must be a positive integer")
        if h and min(h) < b:
            raise ConfigError("autoencoder.bottleneck must not be wider than any hidden layer")
        if list(h) != sorted(h, reverse=True):
            raise ConfigError("autoencoder.hidden_layers must be non-increasing (encoder narrows)")
        for name in ("activation", "output_activation"):
            if getattr(self, name) not in _ACTIVATIONS:
                raise ConfigError(f"autoencoder.{name} must be one of {_ACTIVATIONS}")
        if not 0.0 <= self.dropout < 0.9:
            raise ConfigError("autoencoder.dropout must be in [0, 0.9)")
        if self.l2 < 0 or self.noise_std < 0 or self.min_delta < 0:
            raise ConfigError("autoencoder.l2, noise_std and min_delta must be >= 0")
        if not 0.0 < self.learning_rate < 1.0:
            raise ConfigError("autoencoder.learning_rate must be in (0, 1)")
        for name in ("batch_size", "epochs"):
            if getattr(self, name) < 1:
                raise ConfigError(f"autoencoder.{name} must be >= 1")
        if self.patience < 0:
            raise ConfigError("autoencoder.patience must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hidden_layers"] = list(self.hidden_layers)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AutoencoderSettings":
        """Validated construction from a plain dict (e.g. saved model metadata)."""
        return _from_mapping(cls, data, "autoencoder")


_ORIGIN_RE = re.compile(r"^https?://[A-Za-z0-9.\-]+(:[0-9]{1,5})?$")


def _check_relative_file(section: str, name: str, value: str, suffixes: tuple[str, ...]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{name} must be a non-empty string")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise ConfigError(f"{section}.{name} must be relative to the project root, got {value!r}")
    if ".." in PureWindowsPath(value).parts:
        raise ConfigError(f"{section}.{name} must not contain '..'")
    if suffixes and PureWindowsPath(value).suffix.lower() not in suffixes:
        raise ConfigError(f"{section}.{name} must end with one of {suffixes}")


@dataclass(frozen=True)
class ApiConfig:
    """FastAPI service settings. Secrets (e.g. the admin token) are NEVER stored here;
    they come from environment variables only (see production_pulse.api.settings)."""

    host: str = "127.0.0.1"
    port: int = 8000
    max_batch_size: int = 1000
    model_dir: str = "artifacts/models/autoencoder_v1"
    database_path: str = "data/api/predictions.sqlite3"
    metrics_report: str = "reports/evaluation_metrics.json"
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    )
    top_features: int = 5
    history_default_page_size: int = 50
    history_max_page_size: int = 500
    store_inputs: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip() or any(ch.isspace() for ch in self.host):
            raise ConfigError("api.host must be a host name or IP address without spaces")
        if not 1 <= self.port <= 65535:
            raise ConfigError("api.port must be between 1 and 65535")
        if not 1 <= self.max_batch_size <= 100_000:
            raise ConfigError("api.max_batch_size must be between 1 and 100000")
        _check_relative_file("api", "model_dir", self.model_dir, ())
        if all(part == "." for part in PureWindowsPath(self.model_dir).parts):
            raise ConfigError("api.model_dir must be a sub-folder, not the project root")
        _check_relative_file("api", "database_path", self.database_path, (".sqlite3", ".sqlite", ".db"))
        _check_relative_file("api", "metrics_report", self.metrics_report, (".json",))
        for origin in self.cors_origins:
            if not isinstance(origin, str) or origin == "*" or not _ORIGIN_RE.match(origin):
                raise ConfigError(
                    f"api.cors_origins entries must be explicit origins like 'http://localhost:5173' "
                    f"(no '*', no path, no trailing slash); got {origin!r}"
                )
        if not 1 <= self.top_features <= 52:
            raise ConfigError("api.top_features must be between 1 and 52")
        if not 1 <= self.history_default_page_size <= self.history_max_page_size <= 10_000:
            raise ConfigError("need 1 <= api.history_default_page_size <= api.history_max_page_size <= 10000")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cors_origins"] = list(self.cors_origins)
        return out


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_file_name: str = "production_pulse.log"
    console: bool = True

    def __post_init__(self) -> None:
        if self.level.upper() not in _VALID_LOG_LEVELS:
            raise ConfigError(f"logging.level must be one of {_VALID_LOG_LEVELS}, got {self.level!r}")
        name = self.log_file_name
        if not name or Path(name).name != name or PureWindowsPath(name).name != name:
            raise ConfigError("logging.log_file_name must be a plain file name (no folders)")


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str = "Production Pulse"
    random_seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    autoencoder: AutoencoderSettings = field(default_factory=AutoencoderSettings)
    api: ApiConfig = field(default_factory=ApiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ConfigError("random_seed must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> ProjectConfig:
    """Return the built-in defaults (no file access)."""
    return ProjectConfig()


def _type_error(where: str, expected: str, value: Any) -> ConfigError:
    return ConfigError(f"{where} must be {expected}, got {type(value).__name__} {value!r}")


def _from_mapping(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise _type_error(where or "config", "a table/section", data)
    defaults = cls()
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown config key(s) in [{where or 'top level'}]: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        key = f"{where}.{name}" if where else name
        current = getattr(defaults, name)
        if is_dataclass(current):
            kwargs[name] = _from_mapping(type(current), value, key)
        elif isinstance(current, tuple):
            if not isinstance(value, (list, tuple)):
                raise _type_error(key, "a list", value)
            kwargs[name] = tuple(value)
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise _type_error(key, "true/false", value)
            kwargs[name] = value
        elif isinstance(current, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise _type_error(key, "an integer", value)
            kwargs[name] = value
        elif isinstance(current, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _type_error(key, "a number", value)
            kwargs[name] = float(value)
        elif isinstance(current, str):
            if not isinstance(value, str):
                raise _type_error(key, "a string", value)
            kwargs[name] = value
        else:  # pragma: no cover - defensive
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any]) -> ProjectConfig:
    """Build a validated :class:`ProjectConfig` from a nested dict."""
    return _from_mapping(ProjectConfig, data, "")


def load_config(path: Optional[PathLike] = None, *, root: Optional[PathLike] = None) -> ProjectConfig:
    """Load configuration.

    * ``path`` given: that TOML file must exist.
    * otherwise ``<project root>/configs/default.toml`` is used if present,
      else the built-in defaults.
    """
    if path is None:
        if root is None:
            from .paths import find_project_root  # local import avoids a cycle

            root = find_project_root()
        cfg_path = Path(root) / DEFAULT_CONFIG_RELATIVE
        if not cfg_path.is_file():
            return default_config()
    else:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if _toml is None:
        raise ConfigError(
            "Reading TOML needs Python 3.11+ (built-in tomllib) or the 'tomli' package. "
            f"Cannot read {cfg_path}."
        )
    with open(cfg_path, "rb") as fh:
        try:
            data = _toml.load(fh)
        except Exception as exc:  # tomllib.TOMLDecodeError
            raise ConfigError(f"Invalid TOML in {cfg_path}: {exc}") from exc
    return config_from_dict(data)

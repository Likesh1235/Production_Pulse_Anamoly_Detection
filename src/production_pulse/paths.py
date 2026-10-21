"""Project-root discovery and safe output-path handling.

Root resolution order (no username or current directory is assumed):
1. an explicit ``start`` folder passed by the caller (searched upwards),
2. the ``PRODUCTION_PULSE_ROOT`` environment variable,
3. the location of this installed source file (searched upwards).

A folder counts as the project root when it contains ``pyproject.toml`` and
``src/production_pulse/__init__.py``.

Write safety: generated files may only be created inside the configured
output folders (data/, artifacts/, reports/, logs/ by default). Original
datasets, course files, the checksum baseline and any ``.RData`` file are
always refused.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .config import PROTECTED_RELATIVE_PATHS, RAW_DATA_FILES, ProjectConfig, load_config

PathLike = Union[str, os.PathLike]

ROOT_ENV_VAR = "PRODUCTION_PULSE_ROOT"
_PACKAGE_DIR = Path(__file__).resolve().parent
_FORBIDDEN_OUTPUT_SUFFIXES = {".rdata", ".rda", ".rds"}


class ProjectRootNotFoundError(RuntimeError):
    """Raised when the project root cannot be located."""


class UnsafeOutputPathError(ValueError):
    """Raised when a write would touch original data or leave the output area."""


def _is_project_root(candidate: Path) -> bool:
    return (candidate / "pyproject.toml").is_file() and (
        candidate / "src" / "production_pulse" / "__init__.py"
    ).is_file()


def _search_upwards(start: Path) -> Optional[Path]:
    for candidate in (start, *start.parents):
        if _is_project_root(candidate):
            return candidate
    return None


def find_project_root(start: Optional[PathLike] = None) -> Path:
    """Return the absolute project root (see module docstring for the order)."""
    if start is not None:
        begin = Path(start).expanduser().resolve()
        found = _search_upwards(begin)
        if found is None:
            raise ProjectRootNotFoundError(f"No Production Pulse project root at or above {begin}")
        return found

    env_value = os.environ.get(ROOT_ENV_VAR)
    if env_value:
        env_root = Path(env_value).expanduser().resolve()
        if not _is_project_root(env_root):
            raise ProjectRootNotFoundError(
                f"{ROOT_ENV_VAR}={env_value!r} is not a Production Pulse project root "
                "(needs pyproject.toml and src/production_pulse/)."
            )
        return env_root

    found = _search_upwards(_PACKAGE_DIR)
    if found is None:
        raise ProjectRootNotFoundError(
            f"Could not locate the project root above {_PACKAGE_DIR}. "
            f"Set {ROOT_ENV_VAR} to the project folder."
        )
    return found


def _norm(path: Path) -> str:
    """Absolute, symlink-resolved, case-normalised (on Windows) string form."""
    return os.path.normcase(str(Path(path).resolve()))


def _is_within(child: Path, parent: Path) -> bool:
    c, p = _norm(child), _norm(parent)
    try:
        return os.path.commonpath([c, p]) == p
    except ValueError:  # e.g. different Windows drives
        return False


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data_dir: Path
    data_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    logs_dir: Path
    configs_dir: Path
    notebooks_dir: Path

    @property
    def output_dirs(self) -> tuple[Path, ...]:
        """Every generated-output folder (created by :func:`ensure_output_dirs`)."""
        return (
            self.data_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.logs_dir,
        )

    @property
    def protected_files(self) -> tuple[Path, ...]:
        """Original files that must never be written (raw data resolved via raw_data_dir)."""
        raw = tuple(self.raw_data_dir / name for name in RAW_DATA_FILES.values())
        others = tuple(
            self.root / rel for rel in PROTECTED_RELATIVE_PATHS if rel not in RAW_DATA_FILES.values()
        )
        return raw + others

    def raw_file(self, key: str) -> Path:
        """Path of an original dataset by key (e.g. 'fault_free_training'). Does not open it."""
        try:
            return self.raw_data_dir / RAW_DATA_FILES[key]
        except KeyError:
            raise KeyError(f"Unknown raw data key {key!r}; choose from {sorted(RAW_DATA_FILES)}") from None


def get_paths(root: Optional[PathLike] = None, config: Optional[ProjectConfig] = None) -> ProjectPaths:
    """Build and validate all project paths. Creates nothing on disk."""
    root_path = find_project_root(root) if root is not None else find_project_root()
    cfg = config if config is not None else load_config(root=root_path)
    pc = cfg.paths

    def at(rel: str) -> Path:
        return (root_path / rel).resolve()

    paths = ProjectPaths(
        root=root_path,
        raw_data_dir=at(pc.raw_data_dir),
        data_dir=at(pc.data_dir),
        interim_dir=at(pc.interim_dir),
        processed_dir=at(pc.processed_dir),
        artifacts_dir=at(pc.artifacts_dir),
        models_dir=at(pc.models_dir),
        reports_dir=at(pc.reports_dir),
        figures_dir=at(pc.figures_dir),
        logs_dir=at(pc.logs_dir),
        configs_dir=at("configs"),
        notebooks_dir=at("notebooks"),
    )
    _validate_layout(paths)
    return paths


def _validate_layout(paths: ProjectPaths) -> None:
    for out in paths.output_dirs:
        if not _is_within(out, paths.root) or _norm(out) == _norm(paths.root):
            raise UnsafeOutputPathError(f"Output folder {out} must be a sub-folder of {paths.root}")
        if _norm(out) == _norm(paths.raw_data_dir):
            raise UnsafeOutputPathError(f"Output folder {out} must not be the raw-data folder")
        for protected in paths.protected_files:
            if _is_within(protected, out):
                raise UnsafeOutputPathError(
                    f"Output folder {out} would contain protected file {protected.name}"
                )


def is_protected(target: PathLike, paths: Optional[ProjectPaths] = None) -> bool:
    """True if ``target`` is an original file (or any .RData-type file)."""
    paths = paths or get_paths()
    p = Path(target)
    p = p if p.is_absolute() else paths.root / p
    if p.suffix.lower() in _FORBIDDEN_OUTPUT_SUFFIXES:
        return True
    n = _norm(p)
    return any(n == _norm(f) for f in paths.protected_files)


def ensure_output_dirs(paths: Optional[ProjectPaths] = None) -> tuple[Path, ...]:
    """Create the generated-output folders if missing (never touches originals)."""
    paths = paths or get_paths()
    _validate_layout(paths)
    for d in paths.output_dirs:
        d.mkdir(parents=True, exist_ok=True)
    return paths.output_dirs


def safe_output_path(
    target: PathLike,
    paths: Optional[ProjectPaths] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate a path for writing a generated file and create its parent folder.

    ``target`` may be absolute or relative to the project root, e.g.
    ``"artifacts/models/autoencoder.keras"``.

    Raises :class:`UnsafeOutputPathError` if the path is an original/protected
    file, has an R data suffix, or lies outside the output folders, and
    :class:`FileExistsError` if it exists and ``overwrite`` is False.
    """
    paths = paths or get_paths()
    p = Path(target)
    p = (p if p.is_absolute() else paths.root / p).resolve()

    if p.suffix.lower() in _FORBIDDEN_OUTPUT_SUFFIXES:
        raise UnsafeOutputPathError(f"Refusing to write an R data file: {p}")
    if is_protected(p, paths):
        raise UnsafeOutputPathError(f"Refusing to write protected original file: {p}")
    if not any(_is_within(p, out) for out in paths.output_dirs):
        allowed = ", ".join(sorted({d.relative_to(paths.root).as_posix() for d in paths.output_dirs}))
        raise UnsafeOutputPathError(f"{p} is outside the output folders ({allowed})")
    if p.is_dir():
        raise UnsafeOutputPathError(f"{p} is a folder, expected a file path")
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists; pass overwrite=True to replace it")

    p.parent.mkdir(parents=True, exist_ok=True)
    return p

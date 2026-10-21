"""Output-folder safety and logging tests (all writes go to a temp fake project)."""

from __future__ import annotations

import logging

import pytest

from production_pulse.config import RAW_DATA_FILES, default_config, config_from_dict
from production_pulse.logging_config import get_logger, setup_logging, shutdown_logging
from production_pulse.paths import (
    UnsafeOutputPathError,
    ensure_output_dirs,
    get_paths,
    safe_output_path,
)

from conftest import FAKE_RAW_CONTENT


@pytest.fixture
def paths(fake_root):
    return get_paths(fake_root, config=default_config())


def test_ensure_output_dirs_creates_and_is_idempotent(paths):
    created = ensure_output_dirs(paths)
    assert all(d.is_dir() for d in created)
    assert ensure_output_dirs(paths) == created  # second call is harmless


def test_allowed_output_path_creates_parent(paths):
    target = safe_output_path("artifacts/models/autoencoder.keras", paths)
    assert target.parent.is_dir()
    assert not target.exists()  # only the folder is made, no file
    assert paths.models_dir in target.parents


@pytest.mark.parametrize("name", list(RAW_DATA_FILES.values()))
def test_original_datasets_refused(paths, name):
    with pytest.raises(UnsafeOutputPathError):
        safe_output_path(name, paths, overwrite=True)
    assert (paths.root / name).read_bytes() == FAKE_RAW_CONTENT


@pytest.mark.parametrize(
    "target",
    [
        "docs/DATA_CHECKSUMS.md",
        "Mark Split 21CSE732P rubrics .pdf",
        "data/processed/copy.RData",       # R data files refused even in output dirs
        "data/../TEP_FaultFree_Training.RData",  # traversal back to an original
        "data/../../escape.txt",           # traversal out of the project
        "notes.txt",                       # project root is not an output folder
        "src/production_pulse/hack.py",    # source code is not an output folder
    ],
)
def test_unsafe_targets_refused(paths, target):
    with pytest.raises(UnsafeOutputPathError):
        safe_output_path(target, paths, overwrite=True)


def test_case_variant_of_original_refused_on_any_os(paths):
    with pytest.raises(UnsafeOutputPathError):
        safe_output_path("tep_faultfree_training.rdata", paths, overwrite=True)


def test_existing_file_needs_explicit_overwrite(paths):
    target = safe_output_path("reports/summary.txt", paths)
    target.write_text("v1", encoding="utf-8")
    with pytest.raises(FileExistsError):
        safe_output_path("reports/summary.txt", paths)
    assert safe_output_path("reports/summary.txt", paths, overwrite=True) == target


def test_output_dir_containing_protected_file_rejected(fake_root):
    cfg = config_from_dict({"paths": {"reports_dir": "docs"}})
    with pytest.raises(UnsafeOutputPathError):
        get_paths(fake_root, config=cfg)


def test_logging_writes_to_logs_dir_without_duplicates(paths):
    try:
        logger = setup_logging("DEBUG", config=default_config(), paths=paths, console=False)
        setup_logging("DEBUG", config=default_config(), paths=paths, console=False)  # call twice
        get_logger("tests").info("hello from tests")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        for h in logger.handlers:
            h.flush()
        log_file = paths.logs_dir / "production_pulse.log"
        text = log_file.read_text(encoding="utf-8")
        assert text.count("hello from tests") == 1
    finally:
        shutdown_logging()
    assert not logging.getLogger("production_pulse").handlers

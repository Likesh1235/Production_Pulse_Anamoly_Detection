"""Project-root and path-layout tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from production_pulse.config import RAW_DATA_FILES, default_config
from production_pulse.paths import (
    ROOT_ENV_VAR,
    ProjectRootNotFoundError,
    find_project_root,
    get_paths,
    is_protected,
)

REAL_ROOT = Path(__file__).resolve().parents[1]


def test_root_found_from_package_location():
    root = find_project_root()
    assert root == REAL_ROOT.resolve()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "production_pulse" / "paths.py").is_file()


def test_root_independent_of_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # launch from an unrelated folder
    assert find_project_root() == REAL_ROOT.resolve()
    monkeypatch.chdir(REAL_ROOT / "tests")  # launch from a sub-folder
    assert find_project_root() == REAL_ROOT.resolve()


def test_root_found_from_nested_start(fake_root):
    nested = fake_root / "src" / "production_pulse"
    assert find_project_root(nested) == fake_root.resolve()


def test_env_var_override(fake_root, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(fake_root))
    assert find_project_root() == fake_root.resolve()


def test_env_var_pointing_to_wrong_folder_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    with pytest.raises(ProjectRootNotFoundError):
        find_project_root()


def test_no_root_outside_project(tmp_path):
    lonely = tmp_path / "not_a_project"
    lonely.mkdir()
    with pytest.raises(ProjectRootNotFoundError):
        find_project_root(lonely)


def test_layout_is_under_root_and_separate_from_raw(fake_root):
    paths = get_paths(fake_root, config=default_config())
    assert paths.root == fake_root.resolve()
    assert paths.raw_data_dir == paths.root
    for out in paths.output_dirs:
        assert paths.root in out.parents
        for protected in paths.protected_files:
            assert out not in protected.parents


def test_get_paths_creates_nothing(fake_root):
    before = sorted(p.name for p in fake_root.iterdir())
    get_paths(fake_root, config=default_config())
    assert sorted(p.name for p in fake_root.iterdir()) == before


def test_raw_file_lookup(fake_root):
    paths = get_paths(fake_root, config=default_config())
    assert paths.raw_file("fault_free_training").name == "TEP_FaultFree_Training.RData"
    with pytest.raises(KeyError):
        paths.raw_file("does_not_exist")


def test_real_project_original_files_present_and_protected():
    """Read-only: only checks that the originals exist (no opening/reading)."""
    paths = get_paths()
    for name in RAW_DATA_FILES.values():
        p = paths.raw_data_dir / name
        assert p.is_file(), f"missing original file {p}"
        assert is_protected(p, paths)

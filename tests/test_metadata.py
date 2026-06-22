"""Tests for run-metadata helpers: git_commit + hardware_info (RULES.md E3).

These capture provenance for every run and must never crash the run, so the assertions are smoke
plus structural (keys present, types correct).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import microscope.config as config_module
from microscope.config import git_commit, hardware_info, project_root


def test_git_commit_returns_non_empty_string() -> None:
    # Arrange / Act
    commit = git_commit()

    # Assert: either a real short hash or the documented 'unknown' sentinel — never empty.
    assert isinstance(commit, str)
    assert commit != ""


def test_git_commit_long_form_returns_non_empty_string() -> None:
    # Arrange / Act
    commit = git_commit(short=False)

    # Assert
    assert isinstance(commit, str)
    assert commit != ""


def test_git_commit_returns_unknown_outside_a_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: point project_root at a non-git tmp dir so `git rev-parse` returns non-zero.
    monkeypatch.setattr(config_module, "project_root", lambda *a, **k: tmp_path)

    # Act
    commit = git_commit()

    # Assert: documented graceful fallback — never raises, never empty.
    assert commit == "unknown"


def test_git_commit_returns_unknown_on_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: simulate git missing / an OS-level failure when invoking the subprocess.
    def _raise_oserror(*args: object, **kwargs: object) -> None:
        raise OSError("git not found")

    monkeypatch.setattr(config_module.subprocess, "run", _raise_oserror)

    # Act
    commit = git_commit()

    # Assert: the except (OSError, SubprocessError) branch returns the sentinel, never raises.
    assert commit == "unknown"


def test_project_root_falls_back_to_cwd_when_no_marker_found(tmp_path: Path) -> None:
    # Arrange: tmp_path (and its parents) contain no pyproject.toml + docs/ marker pair.
    # Act
    root = project_root(start=tmp_path)

    # Assert: documented fallback to the current working directory.
    assert root == Path.cwd()


def test_hardware_info_contains_required_keys() -> None:
    # Arrange / Act
    info = hardware_info()

    # Assert
    assert isinstance(info, dict)
    for key in ("platform", "python", "cpu_count", "gpu"):
        assert key in info


def test_hardware_info_values_are_strings() -> None:
    # Arrange / Act
    info = hardware_info()

    # Assert
    assert all(isinstance(v, str) for v in info.values())


def test_hardware_info_reports_running_python_version() -> None:
    # Arrange
    import platform

    # Act
    info = hardware_info()

    # Assert
    assert info["python"] == platform.python_version()

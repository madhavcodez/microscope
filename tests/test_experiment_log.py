"""Tests for RunRecord.as_row_cells + append_experiment_row (RULES.md E3/R5).

Every reported claim must map to a row in docs/EXPERIMENTS.md. These verify the row shape, markdown
escaping, and that appending creates a header when absent and accumulates rows.
"""

from __future__ import annotations

from pathlib import Path

from microscope.config import (
    EXPERIMENT_COLUMNS,
    RunRecord,
    append_experiment_row,
)


def _make_record(**overrides: str) -> RunRecord:
    base = dict(
        run_id="r1",
        date="2026-06-21",
        config_hash="abc123def456",
        git_commit="deadbee",
        model="EleutherAI/pythia-70m",
    )
    base.update(overrides)
    return RunRecord(**base)


def test_as_row_cells_returns_one_cell_per_column() -> None:
    # Arrange
    record = _make_record()

    # Act
    cells = record.as_row_cells()

    # Assert
    assert len(cells) == len(EXPERIMENT_COLUMNS)


def test_as_row_cells_all_cells_are_strings() -> None:
    # Arrange
    record = _make_record(seed="0", tokens="1000000")

    # Act
    cells = record.as_row_cells()

    # Assert
    assert all(isinstance(c, str) for c in cells)


def test_as_row_cells_escapes_pipe_characters() -> None:
    # Arrange: a pipe in a value would break the markdown table if not escaped.
    record = _make_record(notes="a|b|c")

    # Act
    cells = record.as_row_cells()

    # Assert
    assert "a\\|b\\|c" in cells
    assert "a|b|c" not in cells


def test_as_row_cells_replaces_newlines_with_spaces() -> None:
    # Arrange: a newline would split one row across multiple markdown lines.
    record = _make_record(key_results="line1\nline2")

    # Act
    cells = record.as_row_cells()

    # Assert
    assert "line1 line2" in cells
    assert all("\n" not in c for c in cells)


def test_append_experiment_row_creates_header_when_file_absent(tmp_path: Path) -> None:
    # Arrange
    target = tmp_path / "EXPERIMENTS.md"
    record = _make_record()

    # Act
    append_experiment_row(record, path=target)

    # Assert
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# EXPERIMENTS")
    # Header row lists every column name.
    for column in EXPERIMENT_COLUMNS:
        assert column in text


def test_append_experiment_row_appends_data_row(tmp_path: Path) -> None:
    # Arrange
    target = tmp_path / "EXPERIMENTS.md"
    record = _make_record(run_id="run-xyz")

    # Act
    append_experiment_row(record, path=target)

    # Assert
    text = target.read_text(encoding="utf-8")
    assert "run-xyz" in text
    assert text.rstrip().endswith("|")


def test_append_experiment_row_accumulates_multiple_rows(tmp_path: Path) -> None:
    # Arrange
    target = tmp_path / "EXPERIMENTS.md"

    # Act: two appends -> two data rows under the same header.
    append_experiment_row(_make_record(run_id="first"), path=target)
    append_experiment_row(_make_record(run_id="second"), path=target)

    # Assert
    text = target.read_text(encoding="utf-8")
    assert "first" in text
    assert "second" in text
    data_rows = [
        line for line in text.splitlines() if line.startswith("| ") and "run_id" not in line
    ]
    assert len(data_rows) == 2


def test_append_experiment_row_creates_parent_directory(tmp_path: Path) -> None:
    # Arrange: target nested in a not-yet-existing subdir.
    target = tmp_path / "docs" / "EXPERIMENTS.md"
    record = _make_record()

    # Act
    append_experiment_row(record, path=target)

    # Assert
    assert target.exists()

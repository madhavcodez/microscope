"""Tests for the RULES.md R1 reproduction gate (reproduction before novelty).

Two layers are exercised:

* ``microscope.config.reproduction_logged`` — the parser. It answers "has the pipeline reproduced a
  known Gemma Scope result yet?" by scanning the EXPERIMENTS.md markdown table for a data row whose
  ``label`` cell contains ``reproduced``. It must fail CLOSED (return False) on a missing file, an
  absent header, or a table with no data rows, so the gate never opens without positive evidence.
* ``microscope.cli train`` — the consumer. When the gate is shut it must exit 3 (the R1 gate code)
  WITHOUT loading/seeding the config; when the gate is open it must get past R1 and hit the GPU/E4
  gate (exit 2). Exit 3 (R1) and exit 2 (GPU) must stay distinct so the two failures are
  distinguishable by callers and CI.

The CLI reads ``microscope.config.EXPERIMENTS_PATH`` at call time, so pointing that constant at a
``tmp_path`` table drives the *real* parser end-to-end through the CLI (stronger than only patching
the function). A couple of tests also patch ``microscope.cli.reproduction_logged`` directly to pin
the CLI's branch logic independently of the parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from microscope._pending import GpuImplementationPending, GpuStackUnavailable
from microscope.cli import app
from microscope.config import (
    ReproductionGateError,
    reproduction_logged,
)

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_EXPERIMENTS = REPO_ROOT / "docs" / "EXPERIMENTS.md"

# Canonical header + separator matching docs/EXPERIMENTS.md (label column present). Kept minimal:
# the parser keys off the "| run_id" header prefix and the column whose header contains "label",
# not off the full 18-column width, so a trimmed table is a faithful contract fixture.
_HEADER = "| run_id | date | label (repro/novel/inconclusive) | notes |"
_SEP = "|--------|------|-----------------------------------|-------|"


def _write_table(
    path: Path, *data_rows: str, header: str = _HEADER, sep: str | None = _SEP
) -> Path:
    """Write a minimal EXPERIMENTS.md-style markdown table to ``path`` and return it."""
    lines = ["# EXPERIMENTS", "", header]
    if sep is not None:
        lines.append(sep)
    lines.extend(data_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- parser: reproduction_logged() ---------------------------------------------------------------


def test_reproduction_logged_default_true_against_real_experiments() -> None:
    # Arrange: the committed docs/EXPERIMENTS.md has 'reproduced' rows (repro-001/002/003).
    assert REAL_EXPERIMENTS.exists(), "real EXPERIMENTS.md must exist for this assertion"

    # Act: no path arg -> uses the module default EXPERIMENTS_PATH.
    result = reproduction_logged()

    # Assert
    assert result is True


def test_reproduction_logged_true_when_reproduced_row_present(tmp_path: Path) -> None:
    # Arrange
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| repro-001 | 2026-06-21 | reproduced | gemma scope recon in ballpark |",
    )

    # Act / Assert
    assert reproduction_logged(table) is True


def test_reproduction_logged_false_when_only_novel_and_inconclusive(tmp_path: Path) -> None:
    # Arrange: a populated table where NO row is 'reproduced' must keep the gate shut.
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| n1 | 2026-06-21 | novel | a new finding |",
        "| i1 | 2026-06-21 | inconclusive | scorer-limited |",
    )

    # Act / Assert
    assert reproduction_logged(table) is False


def test_reproduction_logged_false_when_header_only_no_data_rows(tmp_path: Path) -> None:
    # Arrange: header + separator but zero data rows -> nothing reproduced yet.
    table = _write_table(tmp_path / "EXPERIMENTS.md")

    # Act / Assert
    assert reproduction_logged(table) is False


def test_reproduction_logged_false_when_file_missing(tmp_path: Path) -> None:
    # Arrange: path that does not exist must fail CLOSED, not raise.
    missing = tmp_path / "does_not_exist" / "EXPERIMENTS.md"

    # Act / Assert
    assert reproduction_logged(missing) is False


def test_reproduction_logged_false_when_header_absent(tmp_path: Path) -> None:
    # Arrange: a markdown file with a 'reproduced' word but no '| run_id' header row. Without the
    # header the parser cannot locate the label column, so it must fail closed.
    f = tmp_path / "EXPERIMENTS.md"
    f.write_text("# EXPERIMENTS\n\nProse mentioning reproduced, no table.\n", "utf-8")

    # Act / Assert
    assert reproduction_logged(f) is False


def test_reproduction_logged_false_when_label_column_absent(tmp_path: Path) -> None:
    # Arrange: a valid '| run_id' header + data row, but no 'label' column at all. With no label
    # column the gate cannot be satisfied, even though a cell says 'reproduced'.
    header = "| run_id | date | notes |"
    sep = "|--------|------|-------|"
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| r1 | 2026-06-21 | reproduced something |",
        header=header,
        sep=sep,
    )

    # Act / Assert
    assert reproduction_logged(table) is False


# --- parser robustness (within the documented contract: rows are pipe-prefixed) ------------------


def test_reproduction_logged_true_for_mixed_case_reproduced(tmp_path: Path) -> None:
    # Arrange: matching is case-insensitive ('Reproduced' / 'REPRODUCED' both count).
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| r1 | 2026-06-21 | Reproduced | mixed case |",
    )

    # Act / Assert
    assert reproduction_logged(table) is True


def test_reproduction_logged_true_with_interior_and_trailing_whitespace(tmp_path: Path) -> None:
    # Arrange: cells are stripped, so padding around 'reproduced' must not defeat the match.
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "|  r1  |  2026-06-21  |    reproduced    |  padded cells   |",
    )

    # Act / Assert
    assert reproduction_logged(table) is True


def test_reproduction_logged_true_with_colon_aligned_separator(tmp_path: Path) -> None:
    # Arrange: a colon-aligned separator (|:---:|) must still be recognised as a separator and
    # skipped, not mistaken for a data row.
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| r1 | 2026-06-21 | reproduced | colon-aligned sep |",
        sep="|:------:|:----:|:---------------------------------:|:-----:|",
    )

    # Act / Assert
    assert reproduction_logged(table) is True


def test_reproduction_logged_true_when_reproduced_is_substring_of_cell(tmp_path: Path) -> None:
    # Arrange: documented contract is substring containment, so 'reproduced (partial)' counts.
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| r1 | 2026-06-21 | reproduced (partial) | longer cell |",
    )

    # Act / Assert
    assert reproduction_logged(table) is True


def test_reproduction_logged_finds_reproduced_among_many_rows(tmp_path: Path) -> None:
    # Arrange: a single 'reproduced' row anywhere in the table opens the gate (any() semantics).
    table = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| n1 | 2026-06-21 | novel | a |",
        "| i1 | 2026-06-21 | inconclusive | b |",
        "| r1 | 2026-06-21 | reproduced | c |",
        "| n2 | 2026-06-21 | novel | d |",
    )

    # Act / Assert
    assert reproduction_logged(table) is True


def test_reproduction_logged_skips_non_pipe_lines_between_data_rows(tmp_path: Path) -> None:
    # Arrange: a hand-edited log can have blank lines / prose interleaved among data rows. The
    # parser must skip non-pipe-prefixed lines (not crash or stop) and still find a later
    # 'reproduced' row.
    f = tmp_path / "EXPERIMENTS.md"
    f.write_text(
        "# EXPERIMENTS\n\n"
        f"{_HEADER}\n{_SEP}\n"
        "| n1 | 2026-06-21 | novel | first |\n"
        "\n"  # blank line between data rows
        "a stray prose note that is not a table row\n"
        "| r1 | 2026-06-21 | reproduced | after the gap |\n",
        encoding="utf-8",
    )

    # Act / Assert
    assert reproduction_logged(f) is True


def test_reproduction_logged_false_when_table_has_no_leading_pipes(tmp_path: Path) -> None:
    # Arrange: pin the documented boundary. Data/header rows are spec-defined to start with '|';
    # a table written WITHOUT leading pipes is out of contract and must fail closed (the parser
    # only treats '| run_id...' as the header and only pipe-prefixed lines as data rows).
    f = tmp_path / "EXPERIMENTS.md"
    f.write_text(
        "# EXPERIMENTS\n\n"
        "run_id | date | label (repro/novel/inconclusive) | notes\n"
        "------ | ---- | -------------------------------- | -----\n"
        "r1 | 2026-06-21 | reproduced | no leading pipe\n",
        encoding="utf-8",
    )

    # Act / Assert
    assert reproduction_logged(f) is False


# --- exception type contract ---------------------------------------------------------------------


def test_reproduction_gate_error_is_runtimeerror_subclass() -> None:
    # Arrange/Act/Assert: R1 exception is a RuntimeError (callers catching RuntimeError work).
    assert issubclass(ReproductionGateError, RuntimeError)


def test_reproduction_gate_error_distinct_from_gpu_gates() -> None:
    # Arrange / Act / Assert: R1 (research-validity) must NOT be confused with the GPU/E4 gates
    # (environment availability). They are separate hierarchies on purpose.
    assert not issubclass(ReproductionGateError, GpuImplementationPending)
    assert not issubclass(ReproductionGateError, GpuStackUnavailable)
    assert not issubclass(GpuImplementationPending, ReproductionGateError)
    assert not issubclass(GpuStackUnavailable, ReproductionGateError)


# --- CLI: train gate behaviour -------------------------------------------------------------------


def test_train_blocked_with_exit_3_when_gate_shut_via_real_parser(
    tmp_path: Path, smoke_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: point the module's EXPERIMENTS_PATH at a no-repro table so the CLI's
    # reproduction_logged() runs the REAL parser end-to-end and returns False.
    norepro = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| n1 | 2026-06-21 | novel | nothing reproduced yet |",
    )
    monkeypatch.setattr("microscope.config.EXPERIMENTS_PATH", norepro)

    # Act
    result = runner.invoke(app, ["train", "--config", str(smoke_config_path)])

    # Assert: exit 3 == R1 gate, message names R1, and _prepare was NOT reached (no config identity
    # line printed) -> no config load/seed before the gate.
    assert result.exit_code == 3
    assert "R1" in result.output
    assert "BLOCKED" in result.output
    assert "hash=" not in result.output  # _prepare not reached
    assert "GATED" not in result.output  # never got to the GPU stage


def test_train_blocked_exit_3_via_function_monkeypatch(
    smoke_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: pin the CLI branch directly — force the gate predicate False regardless of the table.
    monkeypatch.setattr("microscope.cli.reproduction_logged", lambda: False)

    # Act
    result = runner.invoke(app, ["train", "--config", str(smoke_config_path)])

    # Assert
    assert result.exit_code == 3
    assert "hash=" not in result.output  # config never loaded/seeded


def test_train_passes_r1_then_hits_gpu_gate_exit_2_via_real_parser(
    tmp_path: Path, smoke_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a table WITH a 'reproduced' row opens R1 (real parser), so train proceeds to
    # _prepare and then the GPU/E4 stub gate.
    repro = _write_table(
        tmp_path / "EXPERIMENTS.md",
        "| repro-001 | 2026-06-21 | reproduced | gemma scope recon |",
    )
    monkeypatch.setattr("microscope.config.EXPERIMENTS_PATH", repro)

    # Act
    result = runner.invoke(app, ["train", "--config", str(smoke_config_path)])

    # Assert: past R1 (exit 2 not 3), GPU gate surfaced, prelude printed config identity.
    assert result.exit_code == 2
    assert result.exit_code != 3  # R1 did not fire
    assert "GATED" in result.output
    assert "hash=" in result.output  # _prepare WAS reached


def test_train_passes_r1_exit_2_via_function_monkeypatch(
    smoke_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: force the gate open at the CLI boundary; must reach the GPU gate (exit 2).
    monkeypatch.setattr("microscope.cli.reproduction_logged", lambda: True)

    # Act
    result = runner.invoke(app, ["train", "--config", str(smoke_config_path)])

    # Assert
    assert result.exit_code == 2
    assert "GATED" in result.output


def test_train_r1_and_gpu_exit_codes_are_distinct(
    tmp_path: Path, smoke_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: run the same command under both gate states and capture both exit codes.
    norepro = _write_table(
        tmp_path / "norepro.md", "| n1 | 2026-06-21 | novel | x |"
    )
    repro = _write_table(
        tmp_path / "repro.md", "| r1 | 2026-06-21 | reproduced | y |"
    )

    # Act: shut gate.
    monkeypatch.setattr("microscope.config.EXPERIMENTS_PATH", norepro)
    shut = runner.invoke(app, ["train", "--config", str(smoke_config_path)])
    # Act: open gate.
    monkeypatch.setattr("microscope.config.EXPERIMENTS_PATH", repro)
    open_ = runner.invoke(app, ["train", "--config", str(smoke_config_path)])

    # Assert: R1 gate (3) and GPU gate (2) are different codes -> distinguishable failures.
    assert shut.exit_code == 3
    assert open_.exit_code == 2
    assert shut.exit_code != open_.exit_code


def test_train_gate_shut_blocks_before_missing_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: R1 is checked BEFORE _prepare/load_config, so even a non-existent config path must
    # still surface the R1 gate (exit 3), not a FileNotFoundError. Pins the ordering.
    norepro = _write_table(
        tmp_path / "EXPERIMENTS.md", "| n1 | 2026-06-21 | novel | x |"
    )
    monkeypatch.setattr("microscope.config.EXPERIMENTS_PATH", norepro)
    missing_cfg = tmp_path / "nope.yaml"

    # Act
    result = runner.invoke(app, ["train", "--config", str(missing_cfg)])

    # Assert: R1 fired first (exit 3), config never loaded.
    assert result.exit_code == 3
    assert "R1" in result.output

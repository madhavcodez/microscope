"""Tests for microscope.cli via typer.testing.CliRunner.

'info' is fully CPU-functional and must exit 0. The GPU-bound stages load+hash+seed the config, then
hit the E4 'pending' gate, which the CLI surfaces as a non-zero exit (code 2) with a GATED message.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from microscope.cli import app

runner = CliRunner()


def test_info_exits_cleanly() -> None:
    # Arrange / Act
    result = runner.invoke(app, ["info"])

    # Assert
    assert result.exit_code == 0


def test_info_reports_version_and_metadata() -> None:
    # Arrange / Act
    result = runner.invoke(app, ["info"])

    # Assert: the info table surfaces the metadata layer fields.
    assert "MicroScope" in result.output
    assert "git commit" in result.output
    assert "python" in result.output


def test_no_args_shows_help_without_crashing() -> None:
    # Arrange / Act: app configured with no_args_is_help=True.
    result = runner.invoke(app, [])

    # Assert: help exit is conventional (0 or 2 depending on Typer/Click), never an error traceback.
    assert result.exit_code in (0, 2)
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_train_with_valid_config_surfaces_gpu_gate(smoke_config_path: Path) -> None:
    # Arrange / Act: a valid config gets past load/hash/seed, then the stub raises -> gated exit.
    result = runner.invoke(app, ["train", "--config", str(smoke_config_path)])

    # Assert: gated stage exits non-zero (code 2) and surfaces the gate, not a raw traceback.
    assert result.exit_code != 0
    assert result.exit_code == 2
    assert "GATED" in result.output


def test_train_prelude_logs_config_hash_and_seed(smoke_config_path: Path) -> None:
    # Arrange / Act
    result = runner.invoke(app, ["train", "--config", str(smoke_config_path)])

    # Assert: the shared prelude prints the config identity before gating.
    assert "hash=" in result.output
    assert "seed=" in result.output


def test_train_with_missing_config_exits_non_zero(tmp_path: Path) -> None:
    # Arrange: a path that does not exist -> load_config raises FileNotFoundError (not gated).
    missing = tmp_path / "nope.yaml"

    # Act
    result = runner.invoke(app, ["train", "--config", str(missing)])

    # Assert
    assert result.exit_code != 0


def test_control_with_invalid_kind_is_rejected(smoke_config_path: Path) -> None:
    # Arrange / Act: 'kind' must be 'randomized' or 'steering'.
    result = runner.invoke(app, ["control", "--config", str(smoke_config_path), "--kind", "bogus"])

    # Assert
    assert result.exit_code != 0


def test_control_steering_surfaces_gpu_gate(smoke_config_path: Path) -> None:
    # Arrange / Act: steering control dispatches to the GPU stub -> gated exit.
    result = runner.invoke(
        app, ["control", "--config", str(smoke_config_path), "--kind", "steering"]
    )

    # Assert
    assert result.exit_code == 2
    assert "GATED" in result.output


def test_reproduce_surfaces_gpu_gate(smoke_config_path: Path) -> None:
    # Arrange / Act: Phase-1 reproduce command dispatches to a GPU stub.
    result = runner.invoke(app, ["reproduce", "--config", str(smoke_config_path)])

    # Assert
    assert result.exit_code == 2
    assert "GATED" in result.output


def test_eval_surfaces_gpu_gate(smoke_config_path: Path) -> None:
    # Arrange / Act: SAEBench eval command dispatches to a GPU stub.
    result = runner.invoke(app, ["eval", "--config", str(smoke_config_path)])

    # Assert
    assert result.exit_code == 2
    assert "GATED" in result.output


def test_circuit_surfaces_gpu_gate(smoke_config_path: Path) -> None:
    # Arrange / Act: circuit-discovery command dispatches to a GPU stub.
    result = runner.invoke(app, ["circuit", "--config", str(smoke_config_path)])

    # Assert
    assert result.exit_code == 2
    assert "GATED" in result.output


def test_autointerp_surfaces_gpu_gate(smoke_config_path: Path) -> None:
    # Arrange / Act: autointerp command (within feature cap) dispatches to a GPU stub.
    result = runner.invoke(
        app,
        [
            "autointerp",
            "--config",
            str(smoke_config_path),
            "--scorer-model",
            "EleutherAI/pythia-70m",
        ],
    )

    # Assert
    assert result.exit_code == 2
    assert "GATED" in result.output


def test_control_randomized_surfaces_gpu_gate(smoke_config_path: Path) -> None:
    # Arrange / Act: randomized-model control dispatches to a GPU stub.
    result = runner.invoke(
        app, ["control", "--config", str(smoke_config_path), "--kind", "randomized"]
    )

    # Assert
    assert result.exit_code == 2
    assert "GATED" in result.output

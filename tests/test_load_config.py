"""Tests for microscope.config.load_config + RunConfig (RULES.md E2/E5).

Loads a real committed config, fails fast on a missing path, and rejects non-mapping YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from microscope.config import RunConfig, load_config


def test_load_config_loads_real_committed_config(smoke_config_path: Path) -> None:
    # Arrange / Act
    cfg = load_config(smoke_config_path)

    # Assert
    assert isinstance(cfg, RunConfig)
    assert cfg.name == "pythia70m_smoke"
    assert cfg.model == "EleutherAI/pythia-70m-deduped"
    assert cfg.seed == 0


def test_load_config_preserves_stage_specific_extra_fields(smoke_config_path: Path) -> None:
    # Arrange / Act: extra='allow' means stage-specific keys survive for hashing/logging.
    cfg = load_config(smoke_config_path)

    # Assert
    dumped = cfg.model_dump()
    assert dumped["width"] == 16384
    assert dumped["trainer"] == "topk"


def test_load_config_accepts_string_path(smoke_config_path: Path) -> None:
    # Arrange / Act
    cfg = load_config(str(smoke_config_path))

    # Assert
    assert cfg.name == "pythia70m_smoke"


def test_load_config_raises_file_not_found_on_missing_path(tmp_path: Path) -> None:
    # Arrange
    missing = tmp_path / "does_not_exist.yaml"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_load_config_raises_value_error_on_non_mapping_yaml(tmp_path: Path) -> None:
    # Arrange: a YAML list is valid YAML but not a config mapping.
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n- c\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError):
        load_config(bad)


def test_load_config_raises_value_error_on_empty_yaml(tmp_path: Path) -> None:
    # Arrange: an empty file parses to None, which is not a mapping.
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError):
        load_config(empty)


def test_load_config_raises_value_error_on_scalar_yaml(tmp_path: Path) -> None:
    # Arrange: a bare scalar is not a mapping.
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just-a-string\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError):
        load_config(scalar)

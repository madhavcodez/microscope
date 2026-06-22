"""Tests for microscope.config.config_hash (RULES.md E2).

A config's identity is its content hash: stable across calls, insensitive to key order,
sensitive to value changes, and a fixed-length lowercase hex string.
"""

from __future__ import annotations

import re

from microscope.config import RunConfig, config_hash


def test_config_hash_is_stable_across_calls() -> None:
    # Arrange
    cfg = {"name": "x", "model": "EleutherAI/pythia-70m", "seed": 0}

    # Act
    first = config_hash(cfg)
    second = config_hash(cfg)

    # Assert
    assert first == second


def test_config_hash_insensitive_to_key_order() -> None:
    # Arrange
    a = {"name": "x", "model": "m", "seed": 0}
    b = {"seed": 0, "model": "m", "name": "x"}

    # Act / Assert
    assert config_hash(a) == config_hash(b)


def test_config_hash_sensitive_to_value_changes() -> None:
    # Arrange
    a = {"name": "x", "model": "m", "seed": 0}
    b = {"name": "x", "model": "m", "seed": 1}

    # Act / Assert
    assert config_hash(a) != config_hash(b)


def test_config_hash_is_12_char_hex_by_default() -> None:
    # Arrange
    cfg = {"name": "x", "model": "m", "seed": 0}

    # Act
    h = config_hash(cfg)

    # Assert
    assert len(h) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", h)


def test_config_hash_respects_length_argument() -> None:
    # Arrange
    cfg = {"name": "x", "model": "m"}

    # Act
    h = config_hash(cfg, length=8)

    # Assert
    assert len(h) == 8
    assert re.fullmatch(r"[0-9a-f]{8}", h)


def test_config_hash_accepts_pydantic_model_and_matches_equivalent_dict() -> None:
    # Arrange: a RunConfig dumped to JSON must hash identically to the same mapping.
    model = RunConfig(name="x", model="m", seed=3)

    # Act
    from_model = config_hash(model)
    from_dict = config_hash(model.model_dump(mode="json"))

    # Assert
    assert from_model == from_dict

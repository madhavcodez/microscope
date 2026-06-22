"""Determinism tests for microscope.config.set_seed (RULES.md E1).

Same seed must reproduce both the numpy and python-random sequences; different seeds must diverge.
"""

from __future__ import annotations

import random

import numpy as np

from microscope.config import set_seed


def test_same_seed_reproduces_numpy_sequence() -> None:
    # Arrange
    set_seed(1234)
    first = np.random.rand(8).tolist()

    # Act
    set_seed(1234)
    second = np.random.rand(8).tolist()

    # Assert
    assert first == second


def test_same_seed_reproduces_python_random_sequence() -> None:
    # Arrange
    set_seed(42)
    first = [random.random() for _ in range(8)]

    # Act
    set_seed(42)
    second = [random.random() for _ in range(8)]

    # Assert
    assert first == second


def test_different_seeds_produce_different_numpy_sequences() -> None:
    # Arrange
    set_seed(1)
    seq_one = np.random.rand(16).tolist()

    # Act
    set_seed(2)
    seq_two = np.random.rand(16).tolist()

    # Assert
    assert seq_one != seq_two


def test_different_seeds_produce_different_python_random_sequences() -> None:
    # Arrange
    set_seed(7)
    seq_one = [random.random() for _ in range(16)]

    # Act
    set_seed(8)
    seq_two = [random.random() for _ in range(16)]

    # Assert
    assert seq_one != seq_two


def test_set_seed_sets_pythonhashseed_env_var() -> None:
    # Arrange / Act
    set_seed(99)

    # Assert
    import os

    assert os.environ["PYTHONHASHSEED"] == "99"


def test_set_seed_non_deterministic_flag_is_accepted() -> None:
    # Arrange / Act: deterministic=False must still seed RNGs without error.
    set_seed(5, deterministic=False)
    first = np.random.rand(4).tolist()
    set_seed(5, deterministic=False)
    second = np.random.rand(4).tolist()

    # Assert
    assert first == second

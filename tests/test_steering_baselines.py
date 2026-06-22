"""Tests for microscope.steering.baselines.difference_of_means (RULES.md R2 simple baseline).

Library-independent linear algebra: verify correctness on a known input, unit-norm normalization,
and that every documented invalid input raises ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from microscope.steering.baselines import difference_of_means


def test_difference_of_means_correct_on_known_input_unnormalized() -> None:
    # Arrange: pos mean = [3, 3], neg mean = [1, 1] -> direction = [2, 2].
    positive = np.array([[2.0, 2.0], [4.0, 4.0]])
    negative = np.array([[0.0, 0.0], [2.0, 2.0]])

    # Act
    direction = difference_of_means(positive, negative, normalize=False)

    # Assert
    np.testing.assert_allclose(direction, np.array([2.0, 2.0]))


def test_difference_of_means_normalized_is_unit_norm() -> None:
    # Arrange
    positive = np.array([[2.0, 2.0], [4.0, 4.0]])
    negative = np.array([[0.0, 0.0], [2.0, 2.0]])

    # Act
    direction = difference_of_means(positive, negative, normalize=True)

    # Assert
    assert np.isclose(np.linalg.norm(direction), 1.0)
    np.testing.assert_allclose(direction, np.array([1.0, 1.0]) / np.sqrt(2.0))


def test_difference_of_means_default_normalizes() -> None:
    # Arrange
    positive = np.array([[10.0, 0.0, 0.0]])
    negative = np.array([[0.0, 0.0, 0.0]])

    # Act: default normalize=True.
    direction = difference_of_means(positive, negative)

    # Assert
    np.testing.assert_allclose(direction, np.array([1.0, 0.0, 0.0]))


def test_difference_of_means_returns_one_dimensional_direction() -> None:
    # Arrange
    positive = np.zeros((3, 5))
    positive[:, 0] = 1.0
    negative = np.zeros((2, 5))

    # Act
    direction = difference_of_means(positive, negative)

    # Assert
    assert direction.ndim == 1
    assert direction.shape == (5,)


def test_difference_of_means_raises_on_wrong_ndim() -> None:
    # Arrange: 1-D inputs are not the required (n, d) shape.
    positive = np.array([1.0, 2.0, 3.0])
    negative = np.array([0.0, 1.0, 2.0])

    # Act / Assert
    with pytest.raises(ValueError):
        difference_of_means(positive, negative)


def test_difference_of_means_raises_on_empty_positive() -> None:
    # Arrange
    positive = np.empty((0, 4))
    negative = np.ones((2, 4))

    # Act / Assert
    with pytest.raises(ValueError):
        difference_of_means(positive, negative)


def test_difference_of_means_raises_on_empty_negative() -> None:
    # Arrange
    positive = np.ones((2, 4))
    negative = np.empty((0, 4))

    # Act / Assert
    with pytest.raises(ValueError):
        difference_of_means(positive, negative)


def test_difference_of_means_raises_on_feature_dim_mismatch() -> None:
    # Arrange: d=3 vs d=4.
    positive = np.ones((2, 3))
    negative = np.ones((2, 4))

    # Act / Assert
    with pytest.raises(ValueError):
        difference_of_means(positive, negative)


def test_difference_of_means_raises_on_zero_vector_when_normalizing() -> None:
    # Arrange: identical means -> zero direction -> cannot normalize.
    positive = np.array([[1.0, 1.0], [1.0, 1.0]])
    negative = np.array([[1.0, 1.0], [1.0, 1.0]])

    # Act / Assert
    with pytest.raises(ValueError):
        difference_of_means(positive, negative, normalize=True)


def test_difference_of_means_zero_vector_allowed_when_not_normalizing() -> None:
    # Arrange: identical means with normalize=False should return the zero vector, not raise.
    positive = np.array([[1.0, 1.0], [1.0, 1.0]])
    negative = np.array([[1.0, 1.0], [1.0, 1.0]])

    # Act
    direction = difference_of_means(positive, negative, normalize=False)

    # Assert
    np.testing.assert_allclose(direction, np.array([0.0, 0.0]))


def test_difference_of_means_accepts_list_input() -> None:
    # Arrange: np.asarray inside the function should coerce list-of-lists.
    positive = [[2.0, 2.0]]
    negative = [[0.0, 0.0]]

    # Act
    direction = difference_of_means(positive, negative, normalize=False)

    # Assert
    np.testing.assert_allclose(direction, np.array([2.0, 2.0]))

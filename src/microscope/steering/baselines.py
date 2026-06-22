"""Steering directions: the simple difference-of-means baseline + the SAE-feature steering contract.

The difference-of-means direction is library-independent linear algebra, so it is implemented for
real here and is unit-testable on CPU. Applying a steering vector to a live model (and measuring its
effect) needs the model on the GPU host and is a contract (E4).

RULES.md R2: any steering claim must compare the SAE-feature intervention against this simple
baseline. R3: report the effect over the evaluated set, honestly, even if the simple baseline wins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._pending import pending
from ..config import RunConfig


def difference_of_means(
    positive: NDArray[np.floating], negative: NDArray[np.floating], *, normalize: bool = True
) -> NDArray[np.floating]:
    """Compute a concept direction as mean(positive activations) - mean(negative activations).

    This is the standard "simple vector" steering baseline: the direction separating activations on
    concept-present inputs from concept-absent inputs.

    Args:
        positive: Array of shape ``(n_pos, d)`` — activations where the concept is present.
        negative: Array of shape ``(n_neg, d)`` — activations where the concept is absent.
        normalize: If True, return a unit-norm direction (the default; magnitude is applied at
            steering time as a separate, tunable coefficient).

    Returns:
        A 1-D direction of shape ``(d,)``.

    Raises:
        ValueError: if inputs are not 2-D, are empty, or have mismatched feature dimensions.
    """
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError(
            f"Expected 2-D arrays; got positive.ndim={pos.ndim}, negative.ndim={neg.ndim}."
        )
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise ValueError("Both positive and negative activation sets must be non-empty.")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(
            f"Feature-dim mismatch: positive d={pos.shape[1]} vs negative d={neg.shape[1]}."
        )

    direction = pos.mean(axis=0) - neg.mean(axis=0)
    if normalize:
        norm = float(np.linalg.norm(direction))
        # Exact == 0.0 is intentional (not a < epsilon check): both inputs are class means, so the
        # only way the difference is truly zero is exact cancellation — a genuinely undefined
        # direction, not floating-point noise we should silently normalize.
        if norm == 0.0:
            raise ValueError("Difference-of-means is the zero vector; cannot normalize.")
        direction = direction / norm
    return direction


def steer_with_sae_feature(
    config: RunConfig, sae: Any, feature_idx: int, coefficient: float
) -> Any:
    """Apply an SAE-feature steering intervention to the live model and measure its effect.

    CONTRACT — GPU host (E4). Compared head-to-head against :func:`difference_of_means` (R2).
    """
    raise pending("steer_with_sae_feature", "nnsight + dictionary_learning", "Phase 4")

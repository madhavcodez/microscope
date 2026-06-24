"""Tests for microscope.eval.reconstruction (RULES.md R5: the Phase-1 grading metrics).

These three functions are the load-bearing math of the Phase-1 reproduction gate (ADR-0003):
variance-explained and mean-L0 are exactly the numbers compared against the published Gemma Scope
ballpark. They are pure (no model/SAE loading), so they are verified here on tiny, hand-checkable
tensors plus a ``FakeSAE`` stand-in (no transformer_lens / sae_lens, those are GPU-only).
"""

from __future__ import annotations

import math

import pytest
import torch

from microscope.eval.reconstruction import (
    mean_l0,
    reconstruction_metrics,
    variance_explained,
)


class FakeSAE:
    """A library-free SAE stand-in exposing ``.encode`` / ``.decode`` (+ ``.dtype``).

    ``encode`` returns a fixed, known sparse code; ``decode`` returns a fixed reconstruction. This
    lets :func:`reconstruction_metrics` be exercised with no real SAE and no model loading, so the
    test asserts pure metric wiring (which numbers come from where) rather than any GPU behaviour.
    """

    def __init__(self, feats: torch.Tensor, recon: torch.Tensor) -> None:
        self._feats = feats
        self._recon = recon
        self.dtype = torch.float32

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._feats

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return self._recon


# variance_explained


def test_variance_explained_perfect_reconstruction_is_one() -> None:
    # Arrange: recon == x exactly -> SSE == 0 -> VE == 1.0.
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Act
    ve = variance_explained(x, x.clone())

    # Assert
    assert ve == 1.0


def test_variance_explained_mean_predictor_is_zero() -> None:
    # Arrange: recon == per-feature mean broadcast -> SSE == total variance -> VE == 0.0.
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    recon = x.mean(dim=0).expand_as(x)

    # Act
    ve = variance_explained(x, recon)

    # Assert
    assert ve == 0.0


def test_variance_explained_hand_computed_intermediate_value() -> None:
    # Arrange: a fully hand-checkable 2x1 case.
    #   x = [[0], [2]], recon = [[0.5], [1.5]]
    #   x.mean(0) = [1] -> total_var = (0-1)^2 + (2-1)^2 = 2
    #   sse = (0-0.5)^2 + (2-1.5)^2 = 0.25 + 0.25 = 0.5
    #   VE = 1 - 0.5/2 = 0.75
    x = torch.tensor([[0.0], [2.0]])
    recon = torch.tensor([[0.5], [1.5]])

    # Act
    ve = variance_explained(x, recon)

    # Assert
    assert math.isclose(ve, 0.75, rel_tol=0.0, abs_tol=1e-12)


def test_variance_explained_worse_than_mean_is_negative() -> None:
    # Arrange: recon far from x (SSE > total variance) -> VE < 0 (the ADR-0003 negative trap).
    #   x = [[0], [2]] -> total_var = 2 ; recon = [[5], [-3]] -> sse = 25 + 25 = 50
    #   VE = 1 - 50/2 = -24.0
    x = torch.tensor([[0.0], [2.0]])
    recon = torch.tensor([[5.0], [-3.0]])

    # Act
    ve = variance_explained(x, recon)

    # Assert
    assert ve < 0.0
    assert math.isclose(ve, -24.0, rel_tol=0.0, abs_tol=1e-9)


def test_variance_explained_returns_python_float() -> None:
    # Arrange
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Act
    ve = variance_explained(x, x.clone())

    # Assert: a plain float, not a 0-d tensor.
    assert isinstance(ve, float)


def test_variance_explained_raises_on_shape_mismatch() -> None:
    # Arrange: differing shapes are a programming error, not a 0-VE result.
    x = torch.zeros((3, 2))
    recon = torch.zeros((2, 2))

    # Act / Assert
    with pytest.raises(ValueError, match="shape mismatch"):
        variance_explained(x, recon)


def test_variance_explained_raises_on_zero_variance() -> None:
    # Arrange: constant x has no variance to explain -> ratio undefined.
    x = torch.ones((3, 2))
    recon = torch.ones((3, 2))

    # Act / Assert
    with pytest.raises(ValueError, match="variance"):
        variance_explained(x, recon)


# mean_l0


def test_mean_l0_counts_strictly_positive_per_row() -> None:
    # Arrange: nonzeros-per-row = {2, 1, 2} -> mean = 5/3.
    feats = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
        ]
    )

    # Act
    l0 = mean_l0(feats)

    # Assert
    assert math.isclose(l0, 5.0 / 3.0, rel_tol=0.0, abs_tol=1e-6)


def test_mean_l0_ignores_zeros_and_negatives() -> None:
    # Arrange: only strictly-positive entries count. Row activity = {1, 1} -> mean = 1.0.
    #   row0: [-5, 0, 7] -> one positive (7)
    #   row1: [0, 4, -2] -> one positive (4)
    feats = torch.tensor(
        [
            [-5.0, 0.0, 7.0],
            [0.0, 4.0, -2.0],
        ]
    )

    # Act
    l0 = mean_l0(feats)

    # Assert
    assert math.isclose(l0, 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_mean_l0_all_active_equals_width() -> None:
    # Arrange: every entry positive -> L0 per row == d_sae == 4.
    feats = torch.ones((5, 4))

    # Act
    l0 = mean_l0(feats)

    # Assert
    assert math.isclose(l0, 4.0, rel_tol=0.0, abs_tol=1e-6)


def test_mean_l0_all_zero_is_zero() -> None:
    # Arrange: no active features anywhere -> L0 == 0.0.
    feats = torch.zeros((3, 6))

    # Act
    l0 = mean_l0(feats)

    # Assert
    assert l0 == 0.0


def test_mean_l0_returns_python_float() -> None:
    # Arrange
    feats = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    # Act
    l0 = mean_l0(feats)

    # Assert
    assert isinstance(l0, float)


# reconstruction_metrics


def test_reconstruction_metrics_identity_recon_and_known_code() -> None:
    # Arrange: decode returns x exactly -> variance_explained == 1.0.
    #   encode returns a known sparse code with nonzeros-per-row {2, 1} -> mean_l0 == 1.5.
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    feats = torch.tensor(
        [
            [1.0, 0.0, 2.0],  # 2 active
            [0.0, 3.0, 0.0],  # 1 active
        ]
    )
    sae = FakeSAE(feats=feats, recon=activations.clone())

    # Act
    metrics = reconstruction_metrics(activations, sae)

    # Assert: VE from perfect recon, L0 from the known code.
    assert metrics["variance_explained"] == 1.0
    assert math.isclose(metrics["mean_l0"], 1.5, rel_tol=0.0, abs_tol=1e-6)


def test_reconstruction_metrics_returns_all_four_keys() -> None:
    # Arrange
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    feats = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    sae = FakeSAE(feats=feats, recon=activations.clone())

    # Act
    metrics = reconstruction_metrics(activations, sae)

    # Assert
    assert set(metrics.keys()) == {"variance_explained", "mean_l0", "n_tokens", "d_sae"}


def test_reconstruction_metrics_n_tokens_from_activations_rows() -> None:
    # Arrange: n_tokens must come from activations.shape[0], not from the code.
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sae = FakeSAE(feats=feats, recon=activations.clone())

    # Act
    metrics = reconstruction_metrics(activations, sae)

    # Assert
    assert metrics["n_tokens"] == 3
    assert isinstance(metrics["n_tokens"], int)


def test_reconstruction_metrics_d_sae_from_feature_width() -> None:
    # Arrange: d_sae must come from feats.shape[-1] (here 5), independent of activation width (2).
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    feats = torch.tensor([[1.0, 0.0, 0.0, 0.0, 2.0], [0.0, 3.0, 0.0, 0.0, 0.0]])
    sae = FakeSAE(feats=feats, recon=activations.clone())

    # Act
    metrics = reconstruction_metrics(activations, sae)

    # Assert
    assert metrics["d_sae"] == 5
    assert isinstance(metrics["d_sae"], int)


def test_reconstruction_metrics_does_no_loading() -> None:
    # Arrange: a FakeSAE whose only capability is encode/decode proves the function never tries to
    # load a model or SAE (RULES.md separation: loading lives in reproduce.gemma_scope, not here).
    activations = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    feats = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    recon = activations.clone()
    sae = FakeSAE(feats=feats, recon=recon)

    # Act: completes purely from the stand-in object, no network/library access.
    metrics = reconstruction_metrics(activations, sae)

    # Assert
    assert metrics["variance_explained"] == 1.0
    assert math.isclose(metrics["mean_l0"], 1.5, rel_tol=0.0, abs_tol=1e-6)

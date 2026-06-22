"""Tests for microscope._pending (RULES.md E4 GPU-pending sentinel).

GPU-bound stages that are NOT yet wired to their library API raise GpuImplementationPending (a
NotImplementedError subclass) whose message names the stage; those stubs are only asserted to raise,
never implemented or exercised. Stages that ARE implemented but run only on the Modal [gpu] image
(e.g. train_coder, ADR-0004) instead raise GpuStackUnavailable when the GPU stack is absent — see
test_train_coder.py for that unit's full coverage; the relevant case is mirrored here.
"""

from __future__ import annotations

import pytest

from microscope._pending import GpuImplementationPending, GpuStackUnavailable, pending
from microscope.config import RunConfig


def test_pending_returns_gpu_implementation_pending_instance() -> None:
    # Arrange / Act
    err = pending("train_coder", "dictionary_learning", "Phase 2")

    # Assert
    assert isinstance(err, GpuImplementationPending)


def test_gpu_implementation_pending_subclasses_not_implemented_error() -> None:
    # Assert: must read as a NotImplementedError to callers/handlers.
    assert issubclass(GpuImplementationPending, NotImplementedError)
    assert isinstance(pending("s", "lib", "p"), NotImplementedError)


def test_pending_message_names_stage_lib_and_phase() -> None:
    # Arrange / Act
    err = pending("steer_with_sae_feature", "nnsight", "Phase 4")
    message = str(err)

    # Assert
    assert "steer_with_sae_feature" in message
    assert "nnsight" in message
    assert "Phase 4" in message


def test_pending_message_marks_deliberate_stub_not_bug() -> None:
    # Assert: the message must self-document as intentional (not a failure to fix).
    message = str(pending("x", "lib", "p"))
    assert "not a bug" in message.lower()


def test_train_coder_is_implemented_not_pending() -> None:
    # Arrange: train_coder is no longer a GpuImplementationPending stub — it is the real Phase-2
    # sparsify wrapper (ADR-0004) that validates the config, then gates on the GPU stack. On a valid
    # config (layer + width + k present) the absent sparsify surfaces GpuStackUnavailable, NOT
    # GpuImplementationPending — same stub->implemented transition reproduce/harvest already made.
    from microscope.saes.train import train_coder

    cfg = RunConfig(name="x", model="EleutherAI/pythia-70m", layer=3, width=4096, k=32)

    # Act / Assert
    with pytest.raises(GpuStackUnavailable) as exc_info:
        train_coder(cfg, "sae")
    assert not isinstance(exc_info.value, GpuImplementationPending)


def test_steer_with_sae_feature_stub_raises_pending() -> None:
    # Arrange
    from microscope.steering.baselines import steer_with_sae_feature

    cfg = RunConfig(name="x", model="EleutherAI/pythia-70m")

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="steer_with_sae_feature"):
        steer_with_sae_feature(cfg, None, 0, 1.0)

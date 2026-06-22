"""Tests for microscope._pending (RULES.md E4 GPU-pending sentinel).

GPU-bound stages raise GpuImplementationPending (a NotImplementedError subclass) whose message names
the stage. The stub stage functions are only asserted to raise — never implemented or exercised.
"""

from __future__ import annotations

import pytest

from microscope._pending import GpuImplementationPending, pending
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


def test_train_coder_stub_raises_pending() -> None:
    # Arrange
    from microscope.saes.train import train_coder

    cfg = RunConfig(name="x", model="EleutherAI/pythia-70m")

    # Act / Assert: GPU stub raises and names its stage; never actually implemented.
    with pytest.raises(GpuImplementationPending, match="train_coder"):
        train_coder(cfg, "sae")


def test_steer_with_sae_feature_stub_raises_pending() -> None:
    # Arrange
    from microscope.steering.baselines import steer_with_sae_feature

    cfg = RunConfig(name="x", model="EleutherAI/pythia-70m")

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="steer_with_sae_feature"):
        steer_with_sae_feature(cfg, None, 0, 1.0)

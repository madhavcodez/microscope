"""GPU-bound stage stubs must raise GpuImplementationPending (RULES.md E4).

Per the task brief and RULES.md E4, these GPU/library-bound stages are intentional documented stubs;
they are asserted to raise (and to name their stage) and are NOT otherwise implemented or exercised.

One genuine CPU-verifiable branch is also covered here: run_autointerp enforces the
<=500-feature cap (RULES.md C3) with a ValueError *before* reaching the GPU gate.
"""

from __future__ import annotations

import pytest

from microscope._pending import GpuImplementationPending
from microscope.config import RunConfig


@pytest.fixture()
def cfg() -> RunConfig:
    return RunConfig(name="x", model="EleutherAI/pythia-70m")


def test_harvest_activations_stub_raises_pending(cfg: RunConfig) -> None:
    # Arrange
    from microscope.activations import harvest_activations

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="harvest_activations"):
        harvest_activations(cfg)


def test_load_pretrained_sae_stub_raises_pending(cfg: RunConfig) -> None:
    # Arrange
    from microscope.reproduce.gemma_scope import load_pretrained_sae

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="load_pretrained_sae"):
        load_pretrained_sae(cfg)


def test_reproduce_stub_raises_pending(cfg: RunConfig) -> None:
    # Arrange
    from microscope.reproduce.gemma_scope import reproduce

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="reproduce"):
        reproduce(cfg)


def test_run_saebench_stub_raises_pending(cfg: RunConfig) -> None:
    # Arrange
    from microscope.eval.saebench import run_saebench

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="run_saebench"):
        run_saebench(cfg, sae=None)


def test_randomized_model_control_stub_raises_pending(cfg: RunConfig) -> None:
    # Arrange
    from microscope.eval.controls import randomized_model_control

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="randomized_model_control"):
        randomized_model_control(cfg, n_features=10, scorer_model="local")


def test_discover_circuit_stub_raises_pending(cfg: RunConfig) -> None:
    # Arrange
    from microscope.circuits.discover import discover_circuit

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="discover_circuit"):
        discover_circuit(cfg, sae=None)


def test_run_autointerp_stub_raises_pending_under_cap(cfg: RunConfig) -> None:
    # Arrange: n_features within the cap reaches the GPU gate (not the cap check).
    from microscope.autointerp.run import run_autointerp

    # Act / Assert
    with pytest.raises(GpuImplementationPending, match="run_autointerp"):
        run_autointerp(cfg, sae=None, n_features=100, scorer_model="local")


def test_run_autointerp_raises_value_error_over_feature_cap(cfg: RunConfig) -> None:
    # Arrange: this branch is CPU-verifiable real logic (RULES.md C3 cap), not a GPU stub.
    from microscope.autointerp.run import MAX_FEATURES_PER_RUN, run_autointerp

    # Act / Assert: exceeding the cap raises ValueError BEFORE the GPU gate.
    with pytest.raises(ValueError, match="cap"):
        run_autointerp(cfg, sae=None, n_features=MAX_FEATURES_PER_RUN + 1, scorer_model="local")


def test_randomized_model_control_raises_value_error_over_feature_cap(cfg: RunConfig) -> None:
    # Arrange: the randomized control runs the FULL auto-interp pipeline, so it is bound by the same
    # <=500 cap as the real run (RULES.md C3) — it must not be an unbounded-spend back door.
    from microscope.autointerp.run import MAX_FEATURES_PER_RUN
    from microscope.eval.controls import randomized_model_control

    # Act / Assert: exceeding the cap raises ValueError BEFORE the GPU gate.
    with pytest.raises(ValueError, match="cap"):
        randomized_model_control(cfg, n_features=MAX_FEATURES_PER_RUN + 1, scorer_model="local")

"""GPU-bound stage gates (RULES.md E4): pending stubs vs. implemented-but-GPU-only stages.

Two kinds of GPU gate live here:

* **Pending stubs** (autointerp / saebench / controls / circuit / steer / the old
  ``harvest_activations`` training-buffer path): not yet wired to their library APIs, so they raise
  :class:`GpuImplementationPending` and name their stage. Never implemented or exercised on CPU.
* **Implemented, GPU-only stages** (``harvest_resid_activations`` / ``load_pretrained_sae`` /
  ``reproduce`` — the Phase-1 reproduction unit): these contain the verified recipe (ADR-0003) and
  raise :class:`GpuStackUnavailable` (a ``RuntimeError`` subclass) naming the Modal ``[gpu]`` image
  when ``transformer_lens`` / ``sae_lens`` are absent (i.e. on this CPU box). They must NOT raise
  GpuImplementationPending — that would mean the implementation regressed back to a stub. The CLI
  catches GpuStackUnavailable to render the gate as exit code 2 (see tests/test_cli.py).

One genuine CPU-verifiable branch is also covered here: run_autointerp enforces the
<=500-feature cap (RULES.md C3) with a ValueError *before* reaching the GPU gate.
"""

from __future__ import annotations

import pytest

from microscope._pending import GpuImplementationPending, GpuStackUnavailable
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


def test_load_pretrained_sae_raises_gpu_stack_unavailable(cfg: RunConfig) -> None:
    # Arrange: load_pretrained_sae is now IMPLEMENTED (not a GpuImplementationPending stub). On this
    # CPU box sae_lens is absent, so it raises GpuStackUnavailable naming the Modal [gpu] image. The
    # lazy import keeps the module importable on CPU.
    from microscope.reproduce.gemma_scope import load_pretrained_sae

    # Act / Assert
    with pytest.raises(GpuStackUnavailable, match=r"sae_lens"):
        load_pretrained_sae(cfg)


def test_load_pretrained_sae_error_is_runtime_not_pending(cfg: RunConfig) -> None:
    # Arrange: pin the behaviour change — GpuStackUnavailable IS a RuntimeError (callers catching
    # RuntimeError still work) but is NOT GpuImplementationPending (it is implemented, not a stub).
    from microscope.reproduce.gemma_scope import load_pretrained_sae

    # Act
    with pytest.raises(GpuStackUnavailable) as exc_info:
        load_pretrained_sae(cfg)

    # Assert
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, GpuImplementationPending)
    assert "[gpu]" in str(exc_info.value).lower()


def test_reproduce_raises_gpu_stack_unavailable(cfg: RunConfig) -> None:
    # Arrange: reproduce() now does real work (load SAE -> harvest -> metrics). Its first step
    # (load_pretrained_sae) hits the missing sae_lens import on CPU, so reproduce raises
    # GpuStackUnavailable naming the Modal [gpu] image -- NOT GpuImplementationPending.
    from microscope.reproduce.gemma_scope import reproduce

    # Act / Assert
    with pytest.raises(GpuStackUnavailable, match=r"\[gpu\]"):
        reproduce(cfg)


def test_reproduce_error_is_runtime_not_pending(cfg: RunConfig) -> None:
    # Arrange: confirm reproduce() is genuinely implemented, not a renamed stub.
    from microscope.reproduce.gemma_scope import reproduce

    # Act
    with pytest.raises(GpuStackUnavailable) as exc_info:
        reproduce(cfg)

    # Assert
    assert isinstance(exc_info.value, RuntimeError)
    assert not isinstance(exc_info.value, GpuImplementationPending)


def test_harvest_resid_activations_raises_gpu_stack_unavailable(cfg: RunConfig) -> None:
    # Arrange: harvest_resid_activations is the verified (implemented) recipe. transformer_lens is
    # absent on CPU, so it raises GpuStackUnavailable naming the Modal [gpu] image (NOT pending).
    # The import gate fires before the layer-is-None check, so cfg's missing layer never matters.
    from microscope.activations import harvest_resid_activations

    # Act / Assert
    with pytest.raises(GpuStackUnavailable, match=r"transformer_lens"):
        harvest_resid_activations(cfg)


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

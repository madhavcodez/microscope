"""Shared fixtures + path helpers for the MicroScope CPU-verifiable test suite.

Only the CPU-verifiable layer is exercised here (config/determinism/metadata, the
difference-of-means steering baseline, the GPU-pending sentinel, and the CLI). GPU-bound stage
functions are intentional documented stubs (RULES.md E4) and are only asserted to raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Repo root resolved relative to this test file: tests/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"


@pytest.fixture()
def configs_dir() -> Path:
    """Absolute path to the committed run-config directory."""
    return CONFIGS_DIR


@pytest.fixture()
def smoke_config_path() -> Path:
    """A real, committed config that loads cleanly (Pythia-70M smoke test).

    This is the Phase-1 config: it has width but intentionally NO ``k`` (Phase 1 used a sparsity
    target, not a TopK ``k``). It is the right fixture for the config/CLI-prelude/R1-gate tests that
    only need *a config that loads*; for exercising the Phase-2 ``train`` command past validation,
    use :func:`train_config_path` (which has the ``k`` the sparsify wrapper requires).
    """
    return CONFIGS_DIR / "pythia70m_smoke.yaml"


@pytest.fixture()
def train_config_path() -> Path:
    """The Phase-2 training config (ADR-0004): has width + k for the sparsify wrapper.

    ``microscope.saes.train.coder_config_dict`` requires an explicit ``width`` AND ``k`` (a fair
    SAE-vs-transcoder comparison needs both stated). The Phase-1 ``pythia70m_smoke.yaml`` lacks
    ``k``, so the ``train`` command now fails validation on it *before* the GPU gate. Tests that
    want ``train`` to pass validation and reach the GPU/E4 gate (exit 2) must use this config.
    """
    return CONFIGS_DIR / "train_pythia70m_smoke.yaml"

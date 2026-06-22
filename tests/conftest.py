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
    """A real, committed config that loads cleanly (Pythia-70M smoke test)."""
    return CONFIGS_DIR / "pythia70m_smoke.yaml"

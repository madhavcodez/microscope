"""Train an SAE or a skip-transcoder on >=1 layer (wraps dictionary_learning).

CONTRACT — Phase 2. Implemented on the GPU host after verifying the dictionary_learning API
(ActivationBuffer, the trainSAE-style entry point, and TopK/JumpReLU trainers) — RULES.md E4.
Smoke-test on Pythia-70M first (cheap, fits the 6 GB local GPU) before spending on Gemma-2-2B (C4).
Every training run is logged with full metadata (E3).
"""

from __future__ import annotations

from typing import Any, Literal

from ..config import RunConfig
from .._pending import pending

CoderKind = Literal["sae", "transcoder"]


def train_coder(config: RunConfig, kind: CoderKind) -> dict[str, Any]:
    """Train a sparse coder of the given ``kind`` per ``config`` and return run metrics.

    Args:
        config: Run config (model, layer, width, sparsity target, dataset, tokens, seed, ...).
        kind: 'sae' or 'transcoder' (skip-transcoder).

    Returns:
        A metrics dict (final reconstruction error, L0, steps, wall-clock) for logging to
        docs/EXPERIMENTS.md (E3).
    """
    raise pending("train_coder", "dictionary_learning", "Phase 2")

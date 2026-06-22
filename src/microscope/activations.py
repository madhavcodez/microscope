"""Activation harvesting from model hookpoints (nnsight).

CONTRACT — implemented on the GPU host after verifying the nnsight API (RULES.md E4).
Prefers the in-memory / small-token-budget path; large on-disk activation caches can reach ~100 GB
and must be avoided or cleaned up (RULES.md C3).
"""

from __future__ import annotations

from typing import Any

from .config import RunConfig
from ._pending import pending


def harvest_activations(config: RunConfig, *, max_tokens: int | None = None) -> Any:
    """Collect activations at ``config.hookpoint`` for ``config.model`` over ``config.dataset``.

    Args:
        config: Run configuration specifying model, hookpoint, dataset, and token budget.
        max_tokens: Optional override of the token budget (defaults to ``config.n_tokens``).

    Returns:
        An activation buffer / tensor compatible with the dictionary_learning trainer (exact type
        fixed once the nnsight + dictionary_learning APIs are verified on the host).
    """
    raise pending("harvest_activations", "nnsight", "Phase 1/2")

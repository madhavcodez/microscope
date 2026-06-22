"""Discover + validate one editable causal feature circuit on a concrete behavior.

CONTRACT — Phase 5. Implemented on the GPU host after verifying the sparse-feature-circuits API
(RULES.md E4). Default behavior: bias-in-bios profession classification. Produce ONE clean,
validated circuit (not several partial ones) and save the graph artifact. Changing the circuit
target is a research-design Human-Decision Gate.
"""

from __future__ import annotations

from typing import Any

from .._pending import pending
from ..config import RunConfig

DEFAULT_TASK = "bias_in_bios_profession_classification"


def discover_circuit(config: RunConfig, sae: Any, *, task: str = DEFAULT_TASK) -> dict[str, Any]:
    """Discover and validate a sparse feature circuit for ``task``; return + persist the graph.

    Returns a dict with the circuit graph (nodes = features/error terms, edges = causal effects),
    validation metrics, and the path to the saved artifact.
    """
    raise pending("discover_circuit", "sparse-feature-circuits", "Phase 5")

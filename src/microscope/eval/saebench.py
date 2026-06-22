"""Run a subset of SAEBench and return a scorecard (wraps sae-bench).

CONTRACT — Phase 1/3. Implemented on the GPU host after verifying the sae-bench install + usage
(RULES.md E4). SAEBench is documented to run on a 24 GB card for Gemma-2-2B. Returns aggregate
metrics for logging (E3) and the SAE-vs-transcoder head-to-head (R3 — aggregates only).
"""

from __future__ import annotations

from typing import Any

from .._pending import pending
from ..config import RunConfig


def run_saebench(config: RunConfig, sae: Any) -> dict[str, Any]:
    """Evaluate ``sae`` with a subset of SAEBench and return the scorecard metrics."""
    raise pending("run_saebench", "sae-bench", "Phase 1/3")

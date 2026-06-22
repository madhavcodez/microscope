"""Load pretrained Gemma Scope SAEs for Gemma-2-2B and reproduce a known auto-interp + SAEBench result.

CONTRACT — Phase 1, the HARD GATE (RULES.md R1). Implemented on the GPU host after verifying the
sae_lens / HF Gemma Scope loading path (E4). Produced numbers must land in the documented ballpark
and be logged to docs/EXPERIMENTS.md, labelled 'reproduced' (R4/R5), before Phase 2 may begin.
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig
from .._pending import pending


def load_pretrained_sae(config: RunConfig) -> Any:
    """Load a pretrained Gemma Scope SAE for the configured layer/hookpoint.

    Returns the loaded SAE object (type fixed once the sae_lens / HF loading path is verified).
    """
    raise pending("load_pretrained_sae", "sae_lens / HF Gemma Scope", "Phase 1")


def reproduce(config: RunConfig) -> dict[str, Any]:
    """Run auto-interp + SAEBench on the pretrained SAE and return the metrics to be logged.

    The caller logs the returned metrics to docs/EXPERIMENTS.md with label='reproduced' and compares
    against the published ballpark; a large miss is a bug to fix before proceeding (R1 gate).
    """
    raise pending("reproduce", "sae_lens + delphi + sae-bench", "Phase 1")

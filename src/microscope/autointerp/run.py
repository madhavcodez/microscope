"""Generate + score feature explanations with delphi, using a LOCAL scorer model (no paid API).

CONTRACT — Phase 1/3. Implemented on the GPU host after verifying delphi's current local-model usage
(RULES.md E4). Produces detection / fuzzing / intruder-detection scores. Capped at <= 500 features
per run (RULES.md C3); raising the cap or switching to a paid API is a Human-Decision Gate.
"""

from __future__ import annotations

from typing import Any

from .._pending import pending
from ..config import RunConfig

MAX_FEATURES_PER_RUN = 500  # RULES.md C3 — hard cap unless the human raises it (Gate).


def run_autointerp(
    config: RunConfig,
    sae: Any,
    *,
    n_features: int,
    scorer_model: str,
) -> dict[str, Any]:
    """Explain and score ``n_features`` features of ``sae`` using a local ``scorer_model``.

    Args:
        config: Run config (model, hookpoint, dataset, seed, ...).
        sae: A loaded/trained SAE or transcoder.
        n_features: Number of features to interpret. Must be <= MAX_FEATURES_PER_RUN (C3).
        scorer_model: HF id of the LOCAL scorer model delphi will run on the GPU (no paid API).

    Returns:
        Aggregate scores (detection / fuzzing / intruder) over the evaluated feature set (R3 — no
        cherry-picking; report aggregates), plus per-feature detail for the demo.

    Raises:
        ValueError: if ``n_features`` exceeds the cap (the cap is a Gate, not a code change).
    """
    if n_features > MAX_FEATURES_PER_RUN:
        raise ValueError(
            f"n_features={n_features} exceeds the auto-interp cap of {MAX_FEATURES_PER_RUN} "
            f"(RULES.md C3). Raising the cap is a Human-Decision Gate."
        )
    raise pending("run_autointerp", "delphi", "Phase 1/3")

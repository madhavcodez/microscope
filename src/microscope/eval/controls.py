"""Phase 4, the mandatory adversarial controls (RULES.md R2). The differentiator of MicroScope.

Two controls, each attached to a class of claim:
  1. randomized-model control, run the full auto-interp pipeline on a randomized-weight copy of the
     model and report the GAP between real and randomized interpretability scores. Tests whether
     high scores are partly an artifact of token statistics. (any interpretability claim.)
  2. steering baseline, see microscope.steering: compare an SAE-feature steering intervention
     against a simple difference-of-means direction for the same concept. (any steering claim.)

CONTRACT for the randomized-model control below, implemented on the GPU host after E4 verification.
"""

from __future__ import annotations

from typing import Any

from .._pending import pending
from ..autointerp.run import MAX_FEATURES_PER_RUN
from ..config import RunConfig


def randomized_model_control(
    config: RunConfig, *, n_features: int, scorer_model: str
) -> dict[str, Any]:
    """Run auto-interp on a randomized-weight copy of the model; return scores for the gap analysis.

    The caller compares these against the real-model auto-interp aggregates and reports the gap
    (RULES.md R2). A small gap is itself a meaningful (possibly 'inconclusive') finding to report.

    Raises:
        ValueError: if ``n_features`` exceeds the auto-interp cap. The randomized control runs the
            *full* auto-interp pipeline, so it is bound by the same ≤500 cap as the real run
            (RULES.md C3); without this, the control would be an unbounded-spend back door.
    """
    if n_features > MAX_FEATURES_PER_RUN:
        raise ValueError(
            f"n_features={n_features} exceeds the auto-interp cap of {MAX_FEATURES_PER_RUN} "
            f"(RULES.md C3). Raising the cap is a Human-Decision Gate."
        )
    raise pending("randomized_model_control", "nnsight + delphi", "Phase 4")

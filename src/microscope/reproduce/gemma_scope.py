"""Load pretrained Gemma Scope SAEs for Gemma-2-2B and reproduce their reconstruction quality.

Phase 1, the HARD GATE (RULES.md R1): reproduce a KNOWN property of a pretrained Gemma Scope SAE
before any custom training. The verified recipe (ADR-0003, proven on the Modal GPU host) loads the
canonical SAE via ``sae_lens``, harvests TransformerLens ``resid_post`` activations with the BOS
token excluded, and measures variance-explained + mean L0, landing at 0.797 / 83 for
``layer_12/width_16k``. The caller logs the returned metrics to docs/EXPERIMENTS.md, labelled
'reproduced' (R4/R5).

``sae_lens`` and ``transformer_lens`` are imported lazily (inside functions) so this module still
imports on the CPU base box where the interp stack is absent, they live only on the Modal ``[gpu]``
image (ADR-0003).
"""

from __future__ import annotations

from typing import Any

from ..activations import harvest_resid_activations
from ..config import RunConfig
from ..eval.reconstruction import reconstruction_metrics


def load_pretrained_sae(config: RunConfig) -> Any:
    """Load the canonical Gemma Scope SAE for ``config``'s layer/width.

    Verified loader (ADR-0003): ``sae_lens.SAE.from_pretrained("gemma-scope-2b-pt-res-canonical",
    "layer_<L>/width_<W>/canonical", device, dtype="bfloat16")``. ``from_pretrained`` may return the
    SAE object directly or a tuple whose ``[0]`` element is the SAE, both are handled.

    The layer comes from ``config.layer``; the width from a ``config.width`` field (configs use
    ``extra='allow'``, so the YAML can carry ``width``), defaulting to ``"16k"``, the width proven
    in the reproduction.

    Args:
        config: Run configuration. Uses ``layer`` (required) and an optional ``width`` extra field.

    Returns:
        The loaded SAE object (exposes ``.encode`` / ``.decode``), on CUDA if available else CPU.

    Raises:
        RuntimeError: if ``sae_lens`` is not importable, this runs on the Modal ``[gpu]`` image.
        ValueError: if ``config.layer`` is ``None``.
    """
    try:
        from sae_lens import SAE
    except ImportError as exc:  # sae_lens lives only on the Modal [gpu] image (ADR-0003).
        from .._pending import GpuStackUnavailable

        raise GpuStackUnavailable(
            "load_pretrained_sae requires 'sae_lens', which is not installed on this machine. This "
            "stage runs on the Modal [gpu] image (see infra/modal_app.py and docs/adr/0003), it "
            "cannot run on the CPU base box."
        ) from exc

    import torch

    if config.layer is None:
        raise ValueError("config.layer is required to select the Gemma Scope SAE release.")

    width = getattr(config, "width", None) or "16k"
    sae_id = f"layer_{config.layer}/width_{width}/canonical"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = SAE.from_pretrained(
        "gemma-scope-2b-pt-res-canonical", sae_id, device=device, dtype="bfloat16"
    )
    return loaded[0] if isinstance(loaded, tuple) else loaded


def reproduce(config: RunConfig) -> dict[str, Any]:
    """Reproduce the pretrained Gemma Scope SAE's reconstruction quality and return the metrics.

    Wires the verified recipe (ADR-0003): load the canonical SAE (:func:`load_pretrained_sae`),
    harvest BOS-excluded TransformerLens ``resid_post`` activations
    (:func:`microscope.activations.harvest_resid_activations`), then compute reconstruction metrics
    (:func:`microscope.eval.reconstruction.reconstruction_metrics`). The SAE is fed activations at
    its own dtype (bfloat16); the metric functions cast to float internally.

    The caller logs the returned metrics to docs/EXPERIMENTS.md with ``label='reproduced'`` and
    compares against the published ballpark (≈0.797 var-explained / ≈83 L0 at layer 12 width 16k);
    a large miss is a bug to fix before Phase 2 may begin (R1 gate).

    Args:
        config: Run configuration (model, layer, optional width/dataset/n_tokens/seed).

    Returns:
        The dict from :func:`reconstruction_metrics`: ``variance_explained``, ``mean_l0``,
        ``n_tokens``, ``d_sae``.

    Raises:
        RuntimeError: if the interp stack (``sae_lens`` / ``transformer_lens``) is not installed -
            this runs on the Modal ``[gpu]`` image (ADR-0003).
    """
    sae = load_pretrained_sae(config)
    activations = harvest_resid_activations(config)

    # Feed the SAE activations at its own (bfloat16) precision, mirroring the proven Modal recipe.
    sae_dtype = getattr(sae, "dtype", None)
    encode_input = activations.to(sae_dtype) if sae_dtype is not None else activations

    return reconstruction_metrics(encode_input, sae)

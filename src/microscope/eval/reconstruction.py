"""SAE reconstruction quality metrics: variance-explained and mean L0 (RULES.md E5).

These are the metrics the Phase-1 reproduction is graded on (ADR-0003): they were proven on the
Modal GPU host to give variance_explained 0.797 / mean L0 83 for the canonical Gemma Scope SAE
(``layer_12/width_16k``). The functions here are **pure**: given an activation tensor (and, for
:func:`reconstruction_metrics`, an SAE object exposing ``.encode``/``.decode``) they compute and
return numbers. They load no model and no SAE, that is the caller's job (see
:mod:`microscope.reproduce.gemma_scope`). This keeps them unit-testable with small tensors.

``torch`` is imported lazily inside each function so this module still imports on a CPU-only base
box where torch may be present but the heavy interp stack is not (RULES.md: ``pip install -e .``
works on CPU). Type hints use string forward references / ``Any`` to avoid importing torch at top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    import torch


def variance_explained(x: torch.Tensor, recon: torch.Tensor) -> float:
    """Fraction of activation variance the reconstruction captures (1.0 = perfect; higher better).

    Computed over **all elements** of the tensors as

        1 - sum((x - recon)^2) / sum((x - x.mean(0))^2)

    i.e. one minus the fraction of variance unexplained (FVU). The denominator is the total
    variance of ``x`` about its per-feature mean (``x.mean(0)`` is the mean over the token/batch
    dimension, broadcast back over rows). This matches the proven Modal recipe (ADR-0003). Sums are
    accumulated in float64 to avoid catastrophic cancellation when x and recon are close.

    Args:
        x: Original activations, shape ``[N, d_model]`` (N tokens, d_model features).
        recon: SAE reconstruction of ``x``, same shape.

    Returns:
        The variance-explained as a Python float. Can be negative if the reconstruction is worse
        than predicting the mean (the -4.5 trap in ADR-0003 from the wrong activation source).

    Raises:
        ValueError: if ``x`` and ``recon`` have different shapes, or the total variance is zero
            (a constant ``x`` has no variance to explain, so the ratio is undefined).
    """
    import torch

    if x.shape != recon.shape:
        raise ValueError(f"shape mismatch: x {tuple(x.shape)} vs recon {tuple(recon.shape)}")

    x64 = x.to(torch.float64)
    recon64 = recon.to(torch.float64)

    sse = ((x64 - recon64) ** 2).sum()
    total_var = ((x64 - x64.mean(dim=0)) ** 2).sum()
    if total_var.item() == 0.0:
        raise ValueError("total variance of x is zero; variance_explained is undefined")

    return (1.0 - (sse / total_var)).item()


def mean_l0(feats: torch.Tensor) -> float:
    """Mean number of active (strictly positive) SAE features per token.

    L0 is the sparsity metric for an SAE: it counts how many latent features fire on each input and
    averages over tokens. Computed as ``(feats > 0).float().sum(dim=-1).mean()``, the per-token
    active-count, averaged over all tokens.

    Args:
        feats: SAE feature activations, shape ``[N, d_sae]`` (N tokens, d_sae latents).

    Returns:
        The mean per-token L0 as a Python float.
    """
    import torch  # noqa: F401  (lazy guard so this module imports on a torch-less base box)

    return (feats > 0).float().sum(dim=-1).mean().item()


def reconstruction_metrics(activations: Any, sae: Any) -> dict[str, Any]:
    """Encode ``activations`` with ``sae``, decode, and return reconstruction quality metrics.

    This is the metric half of the Phase-1 reproduction. It is pure given its inputs: it calls
    ``sae.encode`` / ``sae.decode`` but performs **no model or SAE loading** (RULES.md separation -
    loading lives in :mod:`microscope.reproduce.gemma_scope`). The SAE may use a lower-precision
    dtype (e.g. bfloat16); the encode call is fed activations as-is and the metrics cast to float
    internally, mirroring the proven Modal recipe (ADR-0003). ``d_sae`` is read from feats' width.

    Args:
        activations: Original activations ``x`` of shape ``[N, d_model]`` (a torch tensor).
        sae: An SAE object exposing ``encode(x) -> feats`` and ``decode(feats) -> recon``.

    Returns:
        A dict with:
          - ``variance_explained`` (float): see :func:`variance_explained`.
          - ``mean_l0`` (float): see :func:`mean_l0`.
          - ``n_tokens`` (int): number of token rows in ``activations``.
          - ``d_sae`` (int): SAE latent dimension (active-feature width), from ``feats.shape[-1]``.
    """
    feats = sae.encode(activations)
    recon = sae.decode(feats)

    return {
        "variance_explained": variance_explained(activations, recon),
        "mean_l0": mean_l0(feats),
        "n_tokens": int(activations.shape[0]),
        "d_sae": int(feats.shape[-1]),
    }

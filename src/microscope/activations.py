"""Activation harvesting from model hookpoints.

Two paths live here:

* :func:`harvest_resid_activations`, the **verified** Gemma Scope recipe (ADR-0003): collect the
  TransformerLens ``blocks.<L>.hook_resid_post`` residual stream, excluding the BOS token, returned
  as a single stacked ``[N, d_model]`` tensor ready for :func:`microscope.eval.reconstruction`.
  This is the activation source the Gemma Scope SAEs were trained against, feeding raw HuggingFace
  ``output_hidden_states`` instead gave variance-explained ≈ -4.5 (ADR-0003).
* :func:`harvest_activations`, the dictionary_learning training buffer path (Phase 2), still a
  documented E4 stub until its nnsight + dictionary_learning APIs are verified on the host.

Both prefer the in-memory / small-token-budget path; large on-disk activation caches can reach
~100 GB and must be avoided or cleaned up (RULES.md C3). ``transformer_lens`` is imported lazily so
this module still imports on the CPU base box where TransformerLens is not installed (it lives only
on the Modal ``[gpu]`` image, ADR-0003).
"""

from __future__ import annotations

from typing import Any

from ._pending import GpuStackUnavailable, pending
from .config import RunConfig


def harvest_resid_activations(config: RunConfig, *, max_tokens: int | None = None) -> Any:
    """Harvest TransformerLens ``resid_post`` activations for ``config``, excluding the BOS token.

    Implements the verified Gemma Scope activation recipe (ADR-0003): build a TransformerLens
    :class:`HookedTransformer` for ``config.model``, run each document through it with a cache
    filter on ``blocks.<layer>.hook_resid_post``, drop position 0 (the BOS token, whose residual
    norm is an outlier and is excluded from all stats), and concatenate the per-document
    activations into one ``[N, d_model]`` tensor. The result is the ``x`` consumed by
    :func:`microscope.eval.reconstruction.reconstruction_metrics`.

    The layer is taken from ``config.layer``; the dataset from ``config.dataset`` (defaulting to the
    Gemma Scope reproduction corpus ``NeelNanda/pile-10k`` when unset). The token budget defaults to
    ``config.n_tokens`` and may be overridden by ``max_tokens``; documents are truncated to a fixed
    sequence length and accumulated until the budget is reached. Runs under ``torch.no_grad()`` and
    moves activations to CPU as they are gathered to keep GPU memory flat (RULES.md C3).

    Args:
        config: Run configuration. Uses ``model`` (required), ``layer`` (required for the
            hookpoint), ``dataset`` (optional), and ``n_tokens`` (optional token budget).
        max_tokens: Optional override of the token budget (defaults to ``config.n_tokens``; ``None``
            on both means "use all documents in the loaded split").

    Returns:
        A CPU ``torch.Tensor`` of shape ``[N, d_model]`` (``float32``), the BOS-excluded residual
        stream activations, ready for the reconstruction metrics.

    Raises:
        RuntimeError: if ``transformer_lens`` is not importable, this path runs only on the Modal
            ``[gpu]`` image (ADR-0003), never on the CPU base box.
        ValueError: if ``config.layer`` is ``None`` (the hookpoint needs a concrete layer index).
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:  # transformer_lens lives only on the Modal [gpu] image (ADR-0003).
        raise GpuStackUnavailable(
            "harvest_resid_activations requires 'transformer_lens', which is not installed on this "
            "machine. This stage runs on the Modal [gpu] image (see infra/modal_app.py and "
            "docs/adr/0003), it cannot run on the CPU base box."
        ) from exc

    import torch
    from datasets import load_dataset

    if config.layer is None:
        raise ValueError(
            "config.layer is required to build the resid_post hookpoint "
            "('blocks.<layer>.hook_resid_post')."
        )

    layer = config.layer
    hook = f"blocks.{layer}.hook_resid_post"
    dataset_id = config.dataset or "NeelNanda/pile-10k"
    token_budget = max_tokens if max_tokens is not None else config.n_tokens
    seq_len = 128  # fixed truncation length, matching the proven reproduction recipe (ADR-0003).

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(config.model, dtype="bfloat16").to(device)

    split = "train" if token_budget is None else f"train[:{max(token_budget // seq_len + 1, 1)}]"
    ds = load_dataset(dataset_id, split=split)
    texts = [t for t in ds["text"] if t and t.strip()]

    chunks: list[Any] = []
    n_tokens = 0
    with torch.no_grad():
        for text in texts:
            toks = model.to_tokens(text)[:, :seq_len]  # to_tokens prepends BOS at position 0.
            _, cache = model.run_with_cache(
                toks, names_filter=hook, stop_at_layer=layer + 1, return_type=None
            )
            x = cache[hook][0, 1:].float().cpu()  # drop BOS (position 0); [seq-1, d_model] on CPU.
            if x.shape[0] == 0:
                continue
            chunks.append(x)
            n_tokens += x.shape[0]
            if token_budget is not None and n_tokens >= token_budget:
                break

    if not chunks:
        raise ValueError(
            f"no activations harvested from dataset {dataset_id!r}; all documents were empty."
        )

    acts = torch.cat(chunks, dim=0)
    return acts[:token_budget] if token_budget is not None else acts


def harvest_activations(config: RunConfig, *, max_tokens: int | None = None) -> Any:
    """Collect activations at ``config.hookpoint`` for ``config.model`` over ``config.dataset``.

    The dictionary_learning **training-buffer** path (Phase 2). For the Phase-1 reproduction's
    residual-stream activations use :func:`harvest_resid_activations` instead.

    Args:
        config: Run configuration specifying model, hookpoint, dataset, and token budget.
        max_tokens: Optional override of the token budget (defaults to ``config.n_tokens``).

    Returns:
        An activation buffer / tensor compatible with the dictionary_learning trainer (exact type
        fixed once the nnsight + dictionary_learning APIs are verified on the host).
    """
    raise pending("harvest_activations", "nnsight", "Phase 1/2")

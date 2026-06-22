"""Train an SAE or a skip-transcoder on >=1 layer (wraps EleutherAI ``sparsify``).

CONTRACT — Phase 2 (ADR-0004). Both the SAE and the skip-transcoder are trained with the SAME
``sparsify`` ``SaeConfig`` / ``Trainer``; they differ ONLY by two flags — ``transcode`` and
``skip_connection`` — held against a shared ``num_latents`` (width) and ``k`` (L0). That shared
recipe is the whole point: it gives Phase 3 a methodologically fair SAE-vs-skip-transcoder
head-to-head instead of a confounded one.

This module splits cleanly into two layers so the distinction is testable WITHOUT a GPU:

* :func:`coder_config_dict` — a **pure** function (no ``sparsify`` / ``torch`` import) that maps a
  :class:`~microscope.config.RunConfig` + a ``kind`` to the flat dict of intended sparsify settings.
  This is the TESTABLE CORE; it encodes the SAE-vs-skip-transcoder invariant and validates inputs.
* :func:`train_coder` — the **GPU-only** entry point. It imports ``sparsify`` lazily (raising
  :class:`~microscope._pending.GpuStackUnavailable` naming the Modal ``[gpu]`` image when absent,
  mirroring :func:`microscope.activations.harvest_resid_activations`), builds the real
  ``SaeConfig`` / ``TrainConfig`` from :func:`coder_config_dict`, loads the HF model + a dataset,
  constructs the ``Trainer``, launches training, saves the dictionary, and returns a metrics dict.

Per RULES.md E4 the sparsify API was verified on the Modal image (sparsify 1.3.0, ADR-0004 §"E4
verification results"); this wrapper is written against that verified surface. NO training is run
here — it runs on Modal (Phase 2, unit 2). Smoke-test on Pythia-70M before spending on Gemma-2-2B
(RULES.md C4); every run is logged with full metadata (E3).
"""

from __future__ import annotations

from typing import Any, Literal

from .._pending import GpuStackUnavailable
from ..config import RunConfig

CoderKind = Literal["sae", "transcoder"]

# Defaults for the sparsify settings that are not first-class RunConfig fields. RunConfig uses
# extra='allow' (config.py), so width / k / batch_size / lr / save_dir / run_name may be supplied
# per-YAML; these constants are the fallbacks when a YAML omits the optional ones. width and k have
# NO default on purpose — a fair SAE-vs-transcoder comparison hinges on an explicit, shared width
# and k, so we force the config to state them rather than silently guessing (ADR-0004).
DEFAULT_ACTIVATION = "topk"  # TopK => k is the exact L0 (ADR-0004); 'groupmax' is the other option.
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_SAVE_DIR = "outputs/coders"

# sparsify activation functions (ADR-0004 / SaeConfig.activation). Validated so a typo in a YAML
# fails fast (RULES.md input validation) instead of surfacing deep inside sparsify on the GPU host.
_VALID_ACTIVATIONS = ("topk", "groupmax")


def _coerce_positive_int(value: Any, *, field: str) -> int:
    """Coerce ``value`` to a strictly-positive int or raise a clear ValueError naming ``field``.

    YAML may yield an int already, or a string/float if the author wrote ``"64"`` or ``64.0``;
    accept those but reject anything non-numeric, non-integral, or <= 0 (fail fast — E2/validation).
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.") from exc
    if as_float <= 0 or as_float != int(as_float):
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    return int(as_float)


def coder_config_dict(config: RunConfig, kind: CoderKind) -> dict[str, Any]:
    """Map a run config + coder ``kind`` to the flat dict of intended ``sparsify`` settings (PURE).

    This function imports nothing heavy — it is the CPU-testable core that encodes the central
    Phase-2 invariant (ADR-0004): an **SAE** is ``transcode=False, skip_connection=False`` and a
    **skip-transcoder** is ``transcode=True, skip_connection=True``, while ``num_latents`` (width)
    and ``k`` (TopK L0) are taken from the SAME config so the two coders are directly comparable.

    Required fields (``width`` and ``k``) MUST be present on the config (as typed fields or, more
    usually, as ``extra='allow'`` YAML keys); they have no default because a fair head-to-head
    depends on an explicit shared width/sparsity. Optional fields fall back to module constants.

    Args:
        config: The run config. Reads typed fields ``model``, ``layer``, ``hookpoint``, ``dataset``,
            ``n_tokens``, ``seed``, ``name`` plus these ``extra`` keys when present: ``width``
            (=> ``num_latents``, required), ``k`` (=> TopK L0, required), ``activation``,
            ``batch_size``, ``lr``, ``save_dir``, ``run_name``.
        kind: ``'sae'`` or ``'transcoder'`` (skip-transcoder). Drives ``transcode`` /
            ``skip_connection``.

    Returns:
        A flat ``dict`` with keys: ``kind``, ``model``, ``activation``, ``num_latents``, ``k``,
        ``transcode``, ``skip_connection``, ``layers``, ``hookpoints`` (``None`` if unset),
        ``dataset`` (``None`` if unset), ``batch_size``, ``lr``, ``n_tokens`` (``None`` if unset),
        ``seed``, ``run_name``, ``save_dir``. :func:`train_coder` splits this into ``SaeConfig`` vs
        ``TrainConfig`` fields; keeping it flat keeps the unit test trivial.

    Raises:
        ValueError: if ``kind`` is not in ``{'sae', 'transcoder'}``; if ``width`` or ``k`` is
            missing/non-positive; if ``activation`` is not a recognised sparsify activation; or if
            ``config.layer`` is ``None`` (sparsify needs a concrete layer for the hookpoint).
    """
    if kind not in ("sae", "transcoder"):
        raise ValueError(f"kind must be 'sae' or 'transcoder', got {kind!r}.")

    extras: dict[str, Any] = dict(config.model_extra or {})

    if "width" not in extras or extras["width"] is None:
        raise ValueError(
            "config is missing 'width' (the dictionary size => sparsify num_latents). "
            "An SAE and a transcoder MUST share an explicit width for a fair comparison (ADR-0004)."
        )
    if "k" not in extras or extras["k"] is None:
        raise ValueError(
            "config is missing 'k' (the TopK sparsity => exact L0). "
            "An SAE and a transcoder MUST share an explicit k for a fair comparison (ADR-0004)."
        )

    num_latents = _coerce_positive_int(extras["width"], field="width")
    k = _coerce_positive_int(extras["k"], field="k")

    activation = str(extras.get("activation", DEFAULT_ACTIVATION)).lower()
    if activation not in _VALID_ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {_VALID_ACTIVATIONS}, got {activation!r} "
            "(sparsify SaeConfig.activation, ADR-0004)."
        )

    if config.layer is None:
        raise ValueError(
            "config.layer is required: sparsify needs a concrete layer index to place the "
            "hookpoint (TrainConfig.layers / hookpoints, ADR-0004)."
        )

    batch_size = _coerce_positive_int(
        extras.get("batch_size", DEFAULT_BATCH_SIZE), field="batch_size"
    )

    lr_raw = extras.get("lr", DEFAULT_LR)
    try:
        lr = float(lr_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lr must be a number, got {lr_raw!r}.") from exc
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr!r}.")

    # THE KEY INVARIANT (ADR-0004): SAE => both flags False; skip-transcoder => both flags True.
    is_transcoder = kind == "transcoder"

    run_name = str(extras.get("run_name", f"{config.name}-{kind}"))
    save_dir = str(extras.get("save_dir", DEFAULT_SAVE_DIR))

    return {
        "kind": kind,
        "model": config.model,
        "activation": activation,
        "num_latents": num_latents,
        "k": k,
        "transcode": is_transcoder,
        "skip_connection": is_transcoder,
        "layers": [config.layer],
        "hookpoints": [config.hookpoint] if config.hookpoint else None,
        "dataset": config.dataset,
        "batch_size": batch_size,
        "lr": lr,
        "n_tokens": config.n_tokens,
        "seed": config.seed,
        "run_name": run_name,
        "save_dir": save_dir,
    }


def train_coder(config: RunConfig, kind: CoderKind) -> dict[str, Any]:
    """Train a sparse coder of the given ``kind`` per ``config`` and return run metrics (GPU-only).

    Builds the sparsify ``SaeConfig`` / ``TrainConfig`` from :func:`coder_config_dict`, loads the HF
    model + a streaming dataset, constructs ``Trainer(cfg, dataset, model)`` (the verified entry
    point, ADR-0004), launches training, saves the trained dictionary with ``save_to_disk``, and
    returns a metrics dict for logging to ``docs/EXPERIMENTS.md`` (E3). All heavy imports
    (``sparsify``, ``transformers``, ``datasets``, ``torch``) are lazy so this module still imports
    on the CPU base box.

    Args:
        config: Run config (model, layer, width, k, dataset, n_tokens, seed, ...). See
            :func:`coder_config_dict` for the fields consumed.
        kind: ``'sae'`` or ``'transcoder'`` (skip-transcoder).

    Returns:
        A metrics dict: ``{kind, width, k, transcode, skip_connection, model, layer, save_path,
        run_name, ...}`` plus reconstruction stats (FVU / variance-explained) IF sparsify exposes
        them post-training; otherwise those are left out here and computed in unit 2/3.

    Raises:
        GpuStackUnavailable: if ``sparsify`` is not importable — this stage runs only on the Modal
            ``[gpu]`` image (ADR-0003/0004), never on the CPU base box.
        ValueError: propagated from :func:`coder_config_dict` for invalid/missing config fields.
    """
    # Validate + derive the flat settings on CPU FIRST, so a bad config fails fast before we pay to
    # import the GPU stack or spin up the model (RULES.md input validation; same ordering as the
    # pure core's contract).
    settings = coder_config_dict(config, kind)

    try:
        import sparsify
    except ImportError as exc:  # sparsify lives only on the Modal [gpu] image (ADR-0004).
        raise GpuStackUnavailable(
            "train_coder requires 'sparsify' (EleutherAI), which is not installed on this machine. "
            "This stage runs on the Modal [gpu] image (see infra/modal_app.py and docs/adr/0004) — "
            "it cannot run on the CPU base box."
        ) from exc

    import torch
    from datasets import load_dataset
    from sparsify.data import chunk_and_tokenize
    from transformers import AutoModel, AutoTokenizer

    # --- Build the sparsify configs from the verified flat settings (ADR-0004) ---
    # SaeConfig carries the coder-shape fields; the SAE/skip-transcoder distinction is exactly these
    # two flags + the shared width/k (the fair-comparison invariant).
    sae_config = sparsify.SaeConfig(
        activation=settings["activation"],
        num_latents=settings["num_latents"],
        k=settings["k"],
        transcode=settings["transcode"],
        skip_connection=settings["skip_connection"],
    )

    # TrainConfig wraps the SaeConfig + the training-loop knobs. hookpoints override layers when an
    # explicit hookpoint string is given; otherwise sparsify derives the hookpoint from layers=[N].
    train_kwargs: dict[str, Any] = {
        "sae": sae_config,
        "batch_size": settings["batch_size"],
        "lr": settings["lr"],
        "layers": settings["layers"],
        "init_seeds": [settings["seed"]],
        "run_name": settings["run_name"],
        "save_dir": settings["save_dir"],
    }
    if settings["hookpoints"] is not None:
        train_kwargs["hookpoints"] = settings["hookpoints"]
    train_config = sparsify.TrainConfig(**train_kwargs)

    # --- Load the model + tokenizer (lazy, GPU-only) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(settings["model"], torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(settings["model"])

    # sparsify computes activations on-the-fly, but Trainer needs a TOKENIZED arrow Dataset
    # (verified ADR-0004 unit-2): chunk_and_tokenize -> Trainer(cfg, dataset, model) -> .fit().
    # Non-streaming so it is indexable; sliced to the n_tokens budget to control cost (C3/C4).
    ctx_len = int(getattr(config, "ctx_len", 1024))
    raw = load_dataset(settings["dataset"] or "NeelNanda/pile-10k", split="train")
    tokenized = chunk_and_tokenize(raw, tokenizer, max_seq_len=ctx_len, text_key="text")
    if settings["n_tokens"]:
        n_examples = max(1, int(settings["n_tokens"]) // ctx_len)
        tokenized = tokenized.select(range(min(n_examples, len(tokenized))))

    trainer = sparsify.Trainer(train_config, tokenized, model)

    # --- Launch training + save the dictionary, then return metrics (the unit-2 work) ---
    # The configs, model, dataset, and Trainer above are the real, verified GPU path (ADR-0004). The
    # ONE remaining unverified API is the train-launch method name: ADR-0004 left '.fit()' vs
    # '.train()' UNRESOLVED and says it is confirmed on the Modal [gpu] image in unit 2 — so it is
    # NOT guessed on this CPU box. _launch_train_and_save isolates that single open call.
    return _launch_train_and_save(trainer, sparsify, settings, layer=config.layer)


def _launch_train_and_save(
    trainer: Any, sparsify: Any, settings: dict[str, Any], *, layer: int | None
) -> dict[str, Any]:
    """Launch the sparsify training run, save the trained dictionary, and return run metrics.

    Split out from :func:`train_coder` so the one remaining E4-unverified call — the train-launch
    method name (``.fit()`` vs ``.train()``, ADR-0004) — is isolated and clearly flagged.
    ``trainer`` and ``sparsify`` are the live GPU objects; this body only ever executes on the Modal
    ``[gpu]`` image (its caller raises :class:`GpuStackUnavailable` before reaching here on CPU).

    Args:
        trainer: A constructed ``sparsify.Trainer``.
        sparsify: The imported ``sparsify`` module (for ``Sae.load_from_disk`` on reload).
        settings: The flat settings from :func:`coder_config_dict`.
        layer: The trained layer index (for the metrics row).

    Returns:
        The metrics dict (kind, width, k, flags, model, layer, save_path, run_name, ...).
    """
    save_path = f"{settings['save_dir']}/{settings['run_name']}"

    # Verified (ADR-0004 unit-2 E4 on sparsify 1.3.0): Trainer.fit() launches training; the Trainer
    # writes the trained dictionary under save_dir/run_name (TrainConfig.save_dir + run_name +
    # save_every/save_best). `sparsify` is unused here (launch needs only the trainer) but kept
    # in the signature for forward-compat (a future load_from_disk reload). Per-coder reconstruction
    # (FVU/variance-explained) is computed at eval time (unit 3), not fabricated here.
    _ = sparsify
    trainer.fit()
    return {
        "kind": settings["kind"],
        "model": settings["model"],
        "layer": layer,
        "width": settings["num_latents"],
        "k": settings["k"],
        "transcode": settings["transcode"],
        "skip_connection": settings["skip_connection"],
        "run_name": settings["run_name"],
        "save_path": save_path,
    }

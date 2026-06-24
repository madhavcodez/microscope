# ADR 0003: Modal execution platform + the Gemma Scope activation recipe (E4)
- Status: accepted
- Date: 2026-06-21

## Context
Gate #1 (ADR-0002) chose "rented cloud, 24 GB". The user already had **Modal** credentials configured
(`~/.modal.toml`, profile `madhavcbusiness`, a prior `cortexia-train` app) plus an `hf-token` Modal
secret. Modal is serverless + per-second billed, which is materially better for the $30 cap than a pod:
**no idle burn**, GPU cost accrues only while a function runs.

While verifying the reproduction recipe (E4) we hit a real correctness trap worth recording.

## Decision
- **Execution platform: Modal.** GPU = **L4 (24 GB, ~$0.80/hr)**, cheapest 24 GB option, fits
  Gemma-2-2B comfortably (~22 GB usable confirmed). Code lives in `infra/modal_app.py`: a built image
  (torch 2.7.1 / transformers 5.12.1 / nnsight 0.7 / sae-lens 6.44.3 + the source interp libs), an HF
  secret, and a persistent `microscope-hf-cache` Volume so the ~5 GB model downloads once.
- **Verified library facts (E4):** Gemma Scope SAEs load via `sae_lens.SAE.from_pretrained(
  "gemma-scope-2b-pt-res-canonical", "layer_<L>/width_<W>/canonical")`. Auto-interp = `delphi`
  (scorers `DetectionScorer`/`FuzzingScorer`/`IntruderScorer`; local scorer = `delphi.clients.Offline`
  via vllm, no paid API). SAEBench pip package ships `absorption` + `sparse_probing` evals.
  `feature-circuits` is NOT pip-installable (no setup.py) → Phase 5 clones it.
- **Activation recipe (the trap):** Gemma Scope SAEs must be fed the **TransformerLens**
  `blocks.<L>.hook_resid_post` residual stream, NOT raw HuggingFace `output_hidden_states`. Using HF
  hidden states gave variance-explained ≈ **-4.5** (recon worse than the mean) across every candidate
  index; switching to `HookedTransformer.run_with_cache` gave **0.797** var-explained / **L0 83** at
  layer 12 width 16k, squarely in the documented Gemma Scope ballpark. The BOS token is excluded from
  the stats (its residual norm is an outlier).

## Alternatives considered
- **RunPod/Vast pod** (ADR-0002 default), rejected in favor of Modal: a pod bills while idle, which is
  the main way a $30 budget evaporates; Modal's per-second model removes that risk and needs no SSH.
- **Raw HF `output_hidden_states` for activations**, rejected: empirically wrong for Gemma Scope (see
  above). TransformerLens is the canonical source the SAEs were trained against.

## Consequences
- (+) Per-second billing + the HF cache volume keep the budget tiny: the full E4 verification + the
  reproduction run cost ≈ $0.25 total.
- (+) The verified recipe (TL resid_post, BOS excluded, sae_lens loader) is the foundation for the rest
  of Phase 1 (auto-interp + SAEBench) and Phase 3.
- (-) transformers 5.x is very new; some interp libs warn (vllm prints a v4-deprecation notice). Watch
  for friction when wiring delphi's Offline scorer; pin if needed and record here.
- The proven Modal reproduction logic will be refactored into `src/microscope/` (reproduce/activations/
  eval/autointerp), replacing the E4 stubs, with the CLI pointed at Modal entrypoints.

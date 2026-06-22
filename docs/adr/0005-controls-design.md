# ADR 0005: Phase-4 controls design (the differentiator)
- Status: accepted (design); pre-registered choices below are fixed — changing them is a Gate
- Date: 2026-06-22

## Context
Phase 4 is the project's differentiator (RULES.md R2): adversarial controls that test whether the
interpretability is *real*, not an artifact. Two controls, each attached to a class of claim. Phase 1–3
showed the auto-interp scorer is weak (near-chance), so the controls are designed to NOT be hostage to
the scorer — the primary signal is scorer-independent.

## Decision

### Control A — randomized-model control (MULTI-AXIS)
Tests whether an SAE trained on the *real* model captures structure beyond token statistics, by comparing
it to an SAE trained on a **randomized-weight** model (the Heap et al. "random transformer" control).

- **Primary axis (scorer-INDEPENDENT):** train an SAE with sparsify (the SAME Phase-2 recipe: layer 12,
  width 16384, k=64, ~10M tokens, bf16) on the activations of a **fully randomized-weight** Gemma-2-2B,
  then compare **SAEBench sparse_probing** accuracy: real-model SAE (train-g2-sae) vs random-model SAE.
  A positive gap (real > random, CI excluding zero) = the real SAE encodes model-learned structure, not
  just token co-occurrence. This is the load-bearing control number because it does not use the LLM
  scorer.
  - **Randomization choice:** **ALL weights** re-initialized from the model's own init distribution
    (the standard "random transformer"), with a **logged seed** (random_seed=0). Rationale: all-weights
    is the established control (Heap et al.); per-layer or studied-layer-only randomization tests a
    narrower question and is non-standard — recorded here as the deliberate, justified choice. The
    embedding/tokenizer are kept (so token statistics are preserved — the control isolates *learned*
    structure from *token* structure).
- **Secondary axis (scorer-dependent, may be INCONCLUSIVE):** auto-interp detection/fuzz gap, real-model
  SAE vs random-model SAE, using the **same** local Qwen2.5-3B scorer, the **same** token set, and the
  **same** feature-sample size (100, seed 0) — so the gap is apples-to-apples. Reported with a bootstrap
  CI; accepted as possibly inconclusive (the scorer is weak, but the *gap* partly cancels that weakness).
- **Apples-to-apples:** identical scorer / tokens / sample / SAEBench config across the real and random
  runs; only the model weights differ.

### Control B — steering vs simple baseline
Tests whether an SAE *feature* steers a target concept better than a plain difference-of-means direction
(the AxBench question). Uses the already-implemented `microscope.steering.difference_of_means`.

- **Pre-registered concept:** a single bias-in-bios profession contrast — **"nurse" vs "professor"**
  (both are bias_in_bios classes already used by repro-003's SAEBench probing, so labeled data exists).
  Steer generations toward "nurse".
- **Two directions compared:** (1) the SAE feature most associated with the concept (selected by probing
  the SAE features on the labeled contrast — the top-1 feature), steered via its decoder direction; (2)
  the difference-of-means direction between "nurse" and "professor" residual activations
  (`difference_of_means`). Both added to resid at layer 12 with a swept coefficient.
- **Metric (pre-registered):** **steering success rate** = fraction of generations a concept classifier
  (the same linear probe trained on the contrast) labels as the target concept, subject to a
  **fluency constraint**: mean generation perplexity (under Gemma-2-2B itself) must stay below
  **1.5× the unsteered baseline perplexity** (steered text that is gibberish does not count as success).
  Report success-rate-at-matched-fluency for both directions, head-to-head, with a bootstrap CI on the
  difference over generated samples.
- **Honest expectation (stated up front):** the simple difference-of-means baseline may match or beat the
  SAE feature (the AxBench finding). That is a valid, valuable result — reported plainly, not as a
  failure.

## Alternatives considered
- **Auto-interp gap as the PRIMARY randomized control** — rejected as primary: the scorer is near-chance
  (repro-004), so the auto-interp gap may be inconclusive. It is kept as the *secondary* axis; the
  scorer-independent SAEBench probing gap is primary.
- **Per-layer / studied-layer-only randomization** — rejected as default (non-standard); all-weights is
  the established control. Noted as a possible follow-up.
- **A bespoke steering success metric without a fluency constraint** — rejected: steering can trivially
  "induce" a concept by degenerating the text; the perplexity bound makes success meaningful.

## Consequences
- (+) The primary randomized-model signal is scorer-independent → can be conclusive even though
  auto-interp is not.
- (−) The randomized-model control requires a SECOND sparsify training run (on the random model) — a
  cost-gated GPU run (~$0.33, ~25 min, like Phase 2); if it threatens the $30 cap, STOP and present
  options as a Gate.
- (−) Both controls need the SAEBench adapter for the custom sparsify SAE (sparsify SparseCoder →
  sae_lens-compatible interface) — built + verified before the control runs (Phase-3 dependency).
- Commits us to: the pre-registered concept, metrics, randomization scheme, and sample size above;
  changing any of them later is a Human-Decision Gate (D1/Gate-4).

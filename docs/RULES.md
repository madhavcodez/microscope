# RULES, MicroScope operating rules (canonical copy)

These rules govern all work in this repo. Read them first.

## Research integrity (highest priority)
- **R1. Reproduction before novelty.** Do not train your own SAE/transcoder until the pipeline has
  reproduced a known Gemma Scope result (Phase 1 gate passed and logged).
- **R2. Controls are mandatory, not optional.** Any interpretability claim must be accompanied by the
  randomized-model control. Any steering claim must be accompanied by the simple-baseline comparison.
  If a control is missing, the work is incomplete.
- **R3. No cherry-picking.** Report aggregate scores over the full evaluated feature set (or a
  pre-registered random sample), never hand-picked features. The demo may show one nice feature, but the
  report uses aggregates.
- **R4. Label every claim** as reproduced, novel, or inconclusive. Inconclusive is an acceptable,
  valuable outcome.
- **R5. Claims trace to logged numbers.** No statement in the report without a corresponding row in
  docs/EXPERIMENTS.md.

## Reproducibility / engineering
- **E1. Determinism:** set and log seeds for every run (Python, NumPy, PyTorch, CUDA). Same config +
  seed must reproduce results.
- **E2. Config-driven:** every run is parameterized by a versioned YAML in experiments/configs/; log
  the config hash with results. No magic numbers in code.
- **E3. Log run metadata:** model, layer/hookpoint, SAE/transcoder type + width + sparsity, dataset,
  token count, seed, git commit, hardware, wall-clock, and cost estimate.
- **E4. Verify before you code.** Before using any library API (dictionary_learning, delphi, sae-bench,
  nnsight, sae_lens, HF), confirm the actual current API by reading the installed package source /
  --help / docstrings. Do not write against remembered APIs, these libraries change. If an API differs
  from this spec, follow the library and record the difference in an ADR.
- **E5. Type hints + docstrings** on public functions. Pydantic for configs. Keep modules single-purpose.

## Cost guardrails (hard)
- **C1. Total GPU spend target ≤ $80.** Auto-interp uses a local scorer by default, no paid API.
- **C2. Before any GPU run expected to exceed $15 or 2 hours, stop and ask the human (Gate).**
- **C3. Cap auto-interp at ≤ 500 features per run** unless the human raises it. Caching activations can
  consume ~100 GB disk, prefer the in-memory / no-cache path or a small token budget, and clean up
  caches.
- **C4. Use the smallest model that proves the point.** Pythia-70M for smoke tests; Gemma-2-2B for real
  results. Do not scale to larger models without a Gate.

## Decision discipline
- **D1.** When something is ambiguous, do not silently pick a product/scope direction. Write an ADR
  proposing the options, choose the conservative/reversible one, proceed, and flag it for the human if
  it affects scope or research validity (Gate).
- **D2. Git:** work on feature branches, commit frequently with clear messages, open the work for review.
  Never run destructive commands (git reset --hard, git clean -fd, force-push,
  rm -rf outside the repo's own caches). Commit before risky operations.

## Human-Decision Gates (stop and ask the human only here)
1. A GPU run expected to exceed $15 or 2 hours (C2).
2. Raising the auto-interp feature cap or switching from local scorer to a paid API (C3).
3. Scaling beyond Gemma-2-2B (C4).
4. Any research-design choice that affects validity and isn't already specified (e.g., changing the
   circuit target, changing what counts as the steering baseline, changing eval metrics).
5. Any irreversible or destructive action.
6. Genuine ambiguity in scope after you've written an ADR with options (D1).

At a gate: summarize the decision, give 2-3 options with tradeoffs, give your recommendation, and wait.

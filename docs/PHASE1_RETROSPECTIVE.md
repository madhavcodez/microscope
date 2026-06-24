# Phase 1 Retrospective, decisions, findings, open questions

_Written at the Phase-1 stopping point (2026-06-22). A deliberate pause before Phase 2. This captures
what was decided, what was learned empirically, and the questions worth thinking about before picking
up. Companion to [REPORT.md](REPORT.md) (the finding) and [EXPERIMENTS.md](EXPERIMENTS.md) (the log)._

## 1. Where Phase 1 landed

The Phase-1 hard gate (R1: reproduce a known Gemma Scope result before any custom training) is met on
the **objective** axes and demonstrated on the **auto-interp** axis:

- **Reconstruction, REPRODUCED.** Canonical Gemma Scope SAE (`gemma-scope-2b-pt-res`, layer 12,
  width 16k): variance-explained **0.797**, mean L0 **83** (`repro-001`); consistent across layers
  5/12/19 (`repro-002`). Squarely in the documented Gemma Scope range (~0.79-0.81).
- **SAEBench sparse-probing, REPRODUCED.** SAE probe top-1 **0.767** vs residual-stream baseline
  **0.688** (`repro-003`), the SAE beats the baseline by ~8 points, matching the SAEBench paper's
  qualitative finding (residual ~0.65, SAEs above).
- **Auto-interp, pipeline reproduced; absolute scores scorer-dependent.** delphi runs end-to-end on
  Modal with a *local* scorer (no API). See REPORT.md for the number and its honest label.

Total spend through Phase 1: **~$2 of the $30 cap** (Modal, per-second billed). Reproduction-first is
satisfied; Phase 2 (custom training) is unblocked but intentionally not started.

## 2. Architectural decisions (and why)

| Decision | Rationale | ADR |
|---|---|---|
| **File-as-memory workflow** (durable state lives in the repo, not in memory) | Each work session starts fresh; durable state must be files. Forces honest logging. | RULES.md |
| **Gemma-2-2B primary, Pythia-70M smoke** | The interp tooling (Gemma Scope, SAEBench, transcoder paper) is validated here, reproduction needs known-good references. | 0001 |
| **Modal serverless GPU** (not a rented pod) | Per-second billing = no idle burn, the #1 way a $30 budget dies. User already had Modal creds + credits + an `hf-token` secret. | 0003 |
| **E4-first: honest stubs until the live API is verified** | These libs (transformers 5.x, sae-lens 6.x, delphi, nnsight 0.7) are far newer than the spec assumed; writing against remembered APIs would have been fiction. Every wrapper was written only after introspecting the installed package on Modal. | RULES.md E4 |
| **TransformerLens residual stream as the activation source** | The single most load-bearing empirical decision, see §3. | 0003 |
| **Local scorer for auto-interp** (delphi `Offline`/vLLM) | Keeps auto-interp at $0 API spend (C1). | 0003 |
| **`GpuStackUnavailable(RuntimeError)`** for implemented-but-needs-GPU stages | Lets the CPU box import + unit-test the package while the CLI renders the GPU gate cleanly (exit 2), distinct from "not implemented yet". | code |

## 3. Things found (the empirical lessons)

These are the non-obvious things that only surfaced by running on real hardware, the reason E4 and
"if it's far off, it's a bug, fix it before logging" exist.

1. **HF `output_hidden_states` ≠ TransformerLens `hook_resid_post` for Gemma-2.** Feeding the SAE raw
   HF hidden states gave variance-explained **-4.5** (worse than predicting the mean) at every
   candidate layer index. Switching to `HookedTransformer.run_with_cache` → **0.797**. The Gemma Scope
   SAEs were trained on the TL residual stream; the BOS token is an outlier and must be excluded.
   _This would have been an easy silent "reproduction failure" to mislabel._
2. **Library versions are years ahead of the spec.** Resolved on Modal: torch 2.7.1, transformers
   **5.12.1**, nnsight 0.7, sae-lens 6.44.3, delphi 0.1.3. The spec's described APIs were mostly stale;
   the real APIs were found by introspection, not assumption.
3. **A chain of real infra bugs, each distinct, each fixed empirically** (this is what "build on real
   hardware" costs): (a) `torchvision`'s video API crashes the HF `datasets` torch formatter →
   uninstall it; (b) **flashinfer JIT-compiles CUDA kernels at runtime but the image has the CUDA
   *runtime*, not the *toolkit* (no `nvcc`/`CUDA_HOME`)** → vLLM `EngineCore` init failed; remove
   flashinfer so vLLM uses prebuilt FlashAttention + native sampler; (c) `VLLM_USE_V1=0` is a no-op
   (V0 is gone in this vLLM); (d) delphi's `verbose=True` calls a plotly/kaleido viz step that crashes
   when kaleido is absent → `verbose=False`.
4. **delphi has native Gemma Scope support**, no sae_lens needed for it. `sparse_model` must literally
   contain `"gemma"` and be `owner/<repo>` with the repo ending `-res`/`-mlp`; `hookpoints` is the
   params subdir `layer_L/width_W/average_l0_<int>` (no `canonical`, pick a concrete L0 that exists).
5. **SAEBench's pip package is enough for `sparse_probing` + `absorption`** standalone (datasets
   auto-download); it accepts native sae_lens `(release, sae_id)` tuples, no custom SAE wrapper.
6. **Auto-interp scorer size matters a lot.** A 1.5B local scorer fails delphi's structured
   detection/fuzzing output format (`Parsing selections failed` en masse) → near-useless scores. A
   3B+ scorer is needed even to get signal; frontier-scale scorers are what the papers used.

## 4. High-level open questions (think about before Phase 2+)

1. **Auto-interp validity.** With a local 3B scorer, are detection/fuzzing/intruder scores meaningful,
   or only the *method* is reproduced? What scorer size is the honest minimum? This directly affects
   whether the controls in Phase 4 (randomized-model gap) are interpretable. _Likely answer: report
   the local-scorer number as a baseline, label paper-comparison "inconclusive," and make the
   randomized-model **gap** (same scorer, real vs random weights) the real signal, the gap cancels
   scorer weakness._
2. **Transcoder support (Phase 2 risk).** `dictionary_learning` 0.1.0's top level exposed SAEs +
   trainers (TopK/JumpReLU/Standard/BatchTopK) but **no obvious skip-transcoder class**. Verify whether
   transcoders live in a submodule, a newer version, or a separate repo before committing to the
   spec's "SAE *and* skip-transcoder" deliverable. This is the biggest unknown for Phase 2.
3. **Circuits (Phase 5).** `feature-circuits` is not pip-installable (research repo, no `setup.py`) →
   must clone + run its scripts, or use `dictionary_learning`'s attribution utilities. Decide the
   integration approach; the bias-in-bios target is set (changing it is a Gate).
4. **Controls design (Phase 4, the differentiator).** Randomized-model control: randomize *all* weights
   vs per-layer vs only the studied layer? Use the *same* scorer + token set so the gap is apples-to-
   apples. Steering baseline: which concept(s), and what's the fair difference-of-means construction?
5. **R3 sample size.** Auto-interp ran on 20 latents (a smoke). For a reported aggregate, pre-register
   a larger random feature sample (e.g. 100-200, ≤500 cap) so the number isn't cherry-picked or noisy.
6. **SAEBench depth.** `repro-003` used a single dataset (smoke). The paper number is the 8-dataset ×
   k∈{1,2,5} mean, decide whether to run the full suite for a paper-grade comparison (more GPU time).
7. **Budget pacing.** ~$28 remains. Phase 2 training (SAE + transcoder) is the most expensive step;
   estimate cost on Pythia-70M first, then a single solid Gemma-2-2B config, before any sweep.

## 5. Cost ledger (Phase 1)

Modal L4 @ ~$0.80/hr, per-second. Approx: GPU smoke + probes ~$0.05; reconstruction (3 runs incl. the
HF-hidden-states diagnosis) ~$0.2; multi-layer ~$0.08; SAEBench ~$0.10; auto-interp (5 attempts, each
loading model + caching before the failure point) ~$1.2-1.5. **Total ≈ $2 of $30.** The repeated
auto-interp attempts were the main cost, each had to load the model + cache activations before
hitting the next distinct infra bug. The HF-cache Volume kept model re-downloads free after the first.

## 6. Resume pointer

Read docs/RULES.md → docs/PROGRESS.md → ADRs 0001-0003 → this file → REPORT.md. The proven
Modal recipe is in `infra/modal_app.py`; the package mirror (lazy-imported, CPU-importable) is in
`src/microscope/`. Phase 2 starts with the transcoder-support question (§4.2) on Pythia-70M.

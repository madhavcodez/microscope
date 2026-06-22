<!--
REPORT.md — the finding. Every number here traces to a row in docs/EXPERIMENTS.md (R5).
Phase 1 only. Phases 2-6 (custom training, controls, circuit) are NOT done — see "What's next".
-->

# MicroScope — Report (Phase 1)

## The question

Before building anything novel, can MicroScope **reproduce known Gemma Scope SAE results** on Gemma-2-2B
— reconstruction fidelity, a SAEBench metric, and the auto-interp pipeline — **honestly and cheaply**
(≤ $30)? Reproduction-first (RULES.md R1) is the gate: no custom training until this passes. This report
covers Phase 1; it labels every claim **reproduced / novel / inconclusive** and ties each to a logged
run in [EXPERIMENTS.md](EXPERIMENTS.md).

## Setup

Gemma-2-2B, canonical Gemma Scope SAEs (`gemma-scope-2b-pt-res`), run on Modal (L4 24 GB, per-second
billed). Activations are the TransformerLens `blocks.<L>.hook_resid_post` residual stream with the BOS
token excluded — the source the SAEs were trained on (see [ADR-0003](adr/0003-modal-execution-and-activation-recipe.md);
using raw HF hidden states instead gave variance-explained −4.5). Auto-interp uses a **local** scorer
(delphi `Offline`/vLLM) — no paid API.

## Results

### 1. Reconstruction fidelity — **REPRODUCED**  (`repro-001`, `repro-002`)

The canonical layer-12 / width-16k SAE reconstructs the residual stream with **variance-explained 0.797
and mean L0 83**, inside the documented Gemma Scope range (~0.79–0.81). The result is stable across
depth — layers 5 / 12 / 19 give VE 0.802 / 0.796 / 0.794 — and layer 12 reproduces its own value across
runs (0.797 vs 0.796), confirming determinism.

### 2. SAEBench sparse probing — **REPRODUCED**  (`repro-003`)

On the bias-in-bios probing task, the **SAE probe top-1 accuracy is 0.767 vs a residual-stream baseline
of 0.688** — the SAE beats the raw-residual baseline by ~8 points, matching the SAEBench paper's
qualitative finding (residual baseline ~0.65, SAEs clearly above). (Single-dataset smoke; the paper's
headline is the 8-dataset × k∈{1,2,5} mean — scaling to that is a follow-up, not a different result.)

### 3. Automated interpretability (delphi, local scorer) — **METHOD REPRODUCED; scores INCONCLUSIVE for paper comparison**  (`repro-004`)

The delphi auto-interp pipeline runs end-to-end on the pretrained SAE with a local scorer (no API):
cache latents → explain → score (detection + fuzzing). On 18 latents (~1,355 detection / ~1,370 fuzz
examples) with a **Qwen2.5-3B-Instruct** local scorer, the aggregate accuracies are **detection 0.544,
fuzzing 0.529** — barely above the 0.5 chance line. **The method is reproduced** (valid scores over
1,300+ examples); the **absolute scores are inconclusive** for comparison to the literature, and they
depend heavily on the scorer: a 1.5B scorer could not follow delphi's structured output format at all
(mass parse failures), and even 3B sits near chance, versus the frontier-scale scorers the published
detection numbers (~0.8–0.9) used. The takeaway is honest and useful: the bottleneck is scorer capacity,
not the pipeline. The research-meaningful number in Phase 4 will therefore be the **randomized-model
gap** (the *same* scorer on real vs randomized weights), which cancels scorer weakness and isolates
whether the interpretability is real.

## Honest scope

| Claim | Label | Evidence |
|---|---|---|
| Gemma Scope SAE reconstruction fidelity (VE/L0) | **reproduced** | repro-001, repro-002 |
| SAE sparse-probing beats residual baseline | **reproduced** | repro-003 |
| Auto-interp *pipeline* works on the pretrained SAE (local scorer) | **reproduced (method)** | repro-004 |
| Auto-interp *absolute* detection/fuzz scores match the literature | **inconclusive** | scorer-size dependent (repro-004) |
| Anything about custom-trained SAEs / transcoders, controls, or circuits | **not done** | Phases 2–6 |

Nothing novel is claimed yet — Phase 1 is reproduction only, by design (R1).

## What's next (Phases 2–6 — NOT started)

Custom SAE + skip-transcoder training (Phase 2), auto-interp + SAEBench head-to-head on the trained
coders (Phase 3), the **controls** that are the project's differentiator — randomized-model baseline +
steering-vs-difference-of-means (Phase 4), one feature circuit (Phase 5), and the full write-up
(Phase 6). Open questions and risks are in [PHASE1_RETROSPECTIVE.md](PHASE1_RETROSPECTIVE.md) (notably:
verifying skip-transcoder support in dictionary_learning, and the auto-interp scorer-size question).

## Reproducibility & cost

Every number above maps to a row in [EXPERIMENTS.md](EXPERIMENTS.md) with its git commit, config,
hardware, and seed. Total Phase-1 GPU spend ≈ **$2 of $30** (Modal, per-second). The verified recipe
lives in `infra/modal_app.py`; the CPU-importable package mirror is in `src/microscope/`.

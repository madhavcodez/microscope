# ADR 0001: Model and tooling choices
- Status: accepted
- Date: 2026-06-21

## Context
MicroScope must train SAEs and transcoders on a small open model, auto-interpret and score their
features, evaluate with SAEBench, discover one feature circuit, and run adversarial controls — and it
must do so reproducibly, on a tight GPU budget (≤ $80), with honest evaluation as the defining property.
The single most load-bearing constraint is **reproduction before novelty (R1)**: the toolkit must
reproduce a known-good interpretability result before training anything custom. That makes the choice of
model and libraries a research-validity decision, not a convenience one — we must build on references
that already have published numbers we can check against.

## Decision
- **Primary model: Gemma-2-2B.** It is the model the modern interpretability tooling is validated on:
  Gemma Scope ships pretrained SAEs for it, SAEBench reports numbers on it, and the transcoder /
  skip-transcoder literature uses it. Building here means reproducing against known-good references.
- **Smoke-test model: Pythia-70M.** Cheap enough to confirm a training loop converges on CPU / a 6 GB
  GPU before spending on Gemma-2-2B. `dictionary_learning` baselines are published on Pythia-70M, so it
  doubles as a second reproduction reference for the training path.
- **SAE / transcoder training: `dictionary_learning`** (Saprmarks/EleutherAI ecosystem). Provides
  ActivationBuffer + trainer entry points and the skip-transcoder variants; sparse-feature-circuits is
  built on the same ecosystem, so the circuit work composes.
- **Auto-interp: `delphi`** (formerly sae-auto-interp) with a **LOCAL scorer model** on the GPU — no
  paid API by default (C1). Produces detection / fuzzing / intruder-detection scores.
- **Evaluation: `sae-bench` (SAEBench)** — documented to run on a 24 GB card for Gemma-2-2B; gives the
  standard scorecard for both SAE and transcoder.
- **Circuits: `sparse-feature-circuits`** — discovers + validates an editable causal feature circuit;
  same dictionary_learning ecosystem.
- **Activations: `nnsight`** for hookpoint access; **`sae_lens` or the official HF Gemma Scope release**
  to load pretrained SAEs for the Phase-1 reproduction.

## Alternatives considered
- **Gemma-3 / Gemma Scope 2 as primary** — rejected as primary. Noted in the spec as a "stretch": less
  validated tooling, fewer published reference numbers to reproduce against, which directly undercuts R1.
  May revisit only after the Gemma-2-2B pipeline is green (would be a Gate, per C4).
- **A larger model (7B+) for "nicer" features** — rejected. Violates C4 (smallest model that proves the
  point) and the $80 budget; would require a Gate to even consider.
- **Hand-rolled SAE/auto-interp/eval code** — rejected. Re-implementing means no known-good reference to
  reproduce against (breaks R1) and far more surface area for subtle metric bugs. Wrapping the
  battle-tested libraries is the research-correct choice; E4 forces us to verify their real APIs.
- **Paid API scorer (OpenRouter etc.) for auto-interp** — rejected as default. Local scorer keeps us
  inside the budget (C1) and removes a recurring cost. Switching to an API is an explicit Gate (C3).

## Consequences
- (+) Every Phase-1 number can be checked against a published reference; novelty (custom SAE/transcoder,
  controls, circuit) is built on a verified foundation.
- (+) One coherent ecosystem (dictionary_learning + delphi + sae-bench + sparse-feature-circuits) means
  the training, interp, eval, and circuit stages share data structures.
- (−) Gemma-2-2B needs ~24 GB VRAM for the real runs → a rented GPU host is required (see ADR-0002);
  the local 6 GB GPU only covers Pythia-70M smoke tests. This is a hard external dependency and the
  first Human-Decision Gate.
- (−) These libraries change; the spec's described APIs may be stale. E4 is mandatory — every wrapper
  must be written against the installed package's real API, with any divergence recorded in a new ADR.
- Commits us to: reproduction-first sequencing (Phase 1 is a hard gate), local-scorer auto-interp, and
  treating any model/scope escalation as a Gate.

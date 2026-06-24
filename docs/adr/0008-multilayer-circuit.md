# ADR 0008: Multi-layer feature-set circuit target + method (pre-registered)
- Status: accepted (design); choices below fixed BEFORE the run (R3). Changing the circuit target/metric later is Gate 4.
- Date: 2026-06-23

## Context
ADR-0006 built and validated a **single-layer** (L12) SAE-feature circuit for the bias-in-bios
profession behavior, using our **custom sparsify SAE**. PROGRESS.md lists a deferred follow-up:
"Multi-layer (cross-component) circuit." This ADR pre-registers the conservative extension of that
single-layer circuit to **several layers**.

A multi-layer circuit needs SAEs at more than one layer. We have only ONE custom SAE (L12), so this
unit uses the **pretrained Gemma Scope SAEs** (`gemma-scope-2b-pt-res-canonical`) at layers
**5, 12, 19**, the three layers already reproduced in Phase 1 (`repro-002`, VE ~0.79-0.80 at all
three). These are residual-stream SAEs trained on the **TransformerLens** `blocks.<L>.hook_resid_post`
activation with **BOS excluded**, exactly the Phase-1 `reproduce_recon` recipe, which we reuse verbatim.
Using raw HF residual activations gave VE -4.5 for Gemma Scope (ADR-0003), so the TransformerLens recipe
is mandatory here; this is why this unit does NOT reuse `circuit_eval`'s sparsify/`coder.encode`/HF path.

Per E4 ("follow the library, or record the difference") and C4 ("smallest thing that proves the point"),
we again implement the **core circuit method directly** rather than wiring in the sparse-feature-circuits
library, and document the deviation here (D1). The single-layer ADR-0006 method is mirrored layer-by-layer.

## Decision

### Target behavior (Gate-4 choice, documented, SAME as ADR-0006 for continuity)
The **bias-in-bios profession distinction** (the two most-frequent classes, 21 vs 19), read as a
**classification** signal, identical behavior and data to Phase-4 Control-A and the Phase-5 single-layer
circuit. Reusing it keeps Phases 4-5 mutually reinforcing and lets the multi-layer result be compared
directly to the L12 single-layer circuit (`circuit-g2-sae`).

### What the circuit is (and is NOT, stated plainly)
A **cross-layer feature-SET circuit**: the small set of Gemma Scope SAE latents, drawn from THREE layers
(L5, L12, L19), that together carry the profession distinction. Nodes = SAE features at their layers; the
"edge" is feature → behavior (the L-wise readout), as in ADR-0006, extended to a node set that spans
depth. **This is NOT a full feature→feature cross-layer causal edge graph** (the heavier
attribution-patching / sparse-feature-circuits construction). We do not compute feature→feature edges or
intervene causally across layers. We scope this honestly as a **cross-layer feature-set circuit + a
depth build-up curve**, NOT a causal edge graph (R4). The heavier edge-graph version is named as the
remaining follow-up.

### Method (pre-registered)
1. **Per-layer feature activations.** For each layer L in {5, 12, 19}: load the Gemma Scope SAE
   `layer_<L>/width_16k/canonical`; run the labeled texts through `HookedTransformer` Gemma-2-2B with
   `run_with_cache(names_filter="blocks.<L>.hook_resid_post", stop_at_layer=L+1)`; take the resid at that
   hook, **drop the BOS position** (`[0, 1:]`), `sae.encode` it, and **mean-pool over tokens** per example
   → one dense feature vector per example per layer (the Phase-1 recipe, exactly).
2. **Attribution (probe-INDEPENDENT, to avoid circularity), per layer.** Rank each feature within its
   layer by `|mean_act(class1) - mean_act(class0)|` over the labeled set (model-intrinsic, no validation
   probe). Take **top-`K_per_layer`** features per layer.
3. **Multi-layer circuit = the UNION of the per-layer top-K** features (a small cross-layer node set).
   `K_per_layer ∈ {3, 5, 10}` → circuit sizes 9, 15, 30 nodes (C3: well under any cap).
4. **Faithfulness validation (the mandatory control, R2/R3).** Build the per-example feature matrix by
   **concatenating the circuit's features across layers**, train a FRESH logistic probe on ONLY those
   columns (sufficiency accuracy), and compare to:
   - a **same-size RANDOM cross-layer feature set**, the same count drawn at random across the three
     layers' full dictionaries (`numpy.random.default_rng(0)`), the mandatory control; and
   - the **full-feature ceiling**, a probe on all features of all three layers concatenated.
   Report accuracy + a **bootstrap 95% CI on the (circuit - random) gap**; the circuit is faithful iff it
   recovers most of the ceiling AND the gap CI **excludes 0**.
5. **Cross-layer build-up curve.** Using the chosen `K_per_layer`, report probe accuracy for the
   cumulative circuits **L5-only**, **L5+L12**, **L5+L12+L19** (circuit features only). This shows whether
   the concept **accumulates across depth**. Reported as a curve (no causal claim attached).

Seeds set + logged (E1; `numpy.random.default_rng(0)`, sklearn `random_state=0`, single fixed
train/test split shared across all probes for a paired comparison). Config-style params logged (E3).

### Honest expectations (R4)
- If a small cross-layer top-K set recovers ~full accuracy and beats the random cross-layer control →
  a **sparse multi-layer (cross-layer feature-set) circuit** (novel, with a control).
- If it barely beats random → the concept is **distributed across features/layers**, not a sparse
  cross-layer circuit (also a valid, labeled outcome).
- The build-up curve is **descriptive** (which layers carry the signal), NOT a causal-edge claim.
- **Same token-influence caveat as Control A / ADR-0006:** Control A showed much of the probing signal is
  **token-level** (a random-*model* SAE still probed 0.86), so this circuit partly reflects token
  features (profession words), not purely abstract semantics. Reported, not hidden.

## Alternatives considered
- **Full feature→feature cross-layer edge graph (attribution patching / sparse-feature-circuits).**
  Rejected for this budget/unit: needs the dictionary_learning + nnsight + task-harness integration whose
  friction we already documented in ADR-0006/0007, plus causal patching across three layers. Named as the
  remaining follow-up; this unit is scoped as a feature-set circuit + build-up, stated honestly.
- **Reuse the custom L12 SAE only.** Rejected: a single-layer SAE cannot make a *multi-layer* circuit; the
  pretrained Gemma Scope SAEs are the lowest-friction way to get faithful SAEs at L5/L12/L19 (already
  reproduced in Phase 1).
- **Ranking by the validation probe's own coefficients.** Rejected as circular (same reason as ADR-0006);
  class-mean-difference attribution is probe-independent.

## Consequences
- (+) A controlled, non-circular, multi-layer feature-set circuit + a depth build-up curve, with no new
  library dependency, directly comparable to the single-layer `circuit-g2-sae`.
- (-) Cross-layer feature-SET + probe-readout scope (documented), NOT a causal feature→feature edge graph,
  and NOT a generation-logit behavior (professions are multi-token; same rationale as ADR-0006).
- One cheap GPU run: TransformerLens Gemma-2-2B + 3 Gemma Scope SAEs + feature extraction on ~600 examples
  across 3 layers (~$1-2, ~15-25 min on L4). Commits us to the target + method above (Gate 4 to change).
  Budget guard for this unit: STOP and ask if it would exceed ~$5 or take >3 GPU iterations.

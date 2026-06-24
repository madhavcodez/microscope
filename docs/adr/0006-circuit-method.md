# ADR 0006: Phase-5 feature-circuit target + method (pre-registered)
- Status: accepted (design); choices below fixed before the run (R3). Changing the circuit target later is Gate 4.
- Date: 2026-06-22

## Context
Phase 5 = "find ONE feature circuit." The toolkit spec suggested the sparse-feature-circuits library
(Marks et al.), but that library is built around dictionary_learning SAEs + nnsight + its own task
harness, wiring our **sparsify** SAE + Gemma-2-2B + Modal into it is the same class of high-friction
integration that made the SAEBench-on-custom-coder adapter a deferred item. Per E4 ("follow the library,
or record the difference") and C4 ("smallest thing that proves the point"), we implement the **core
circuit method directly** and document the deviation here (D1).

## Decision

### Target behavior (Gate-4 choice, documented)
The **bias-in-bios profession distinction** (the two most frequent classes, 21 vs 19), the SAME behavior
and data as the Phase-4 Control-A probing gap. Reusing it is conservative: labeled data + the real SAE
already exist, and it keeps Phases 4-5 mutually reinforcing. The behavior is a **classification** signal
read at layer 12, not a single-token LM output, so the circuit is defined w.r.t. that readout (scope note
below), not w.r.t. a generation logit.

### What the circuit is
A **single-layer (L12) SAE-feature circuit**: the small set of SAE latents that carry the profession
distinction. Nodes = SAE features; the "edge" is feature → behavior (the readout). It is NOT a multi-layer
cross-component circuit (feature→feature edges across layers), we have one SAE at one layer, and a
multi-layer circuit is out of budget. Stated plainly, not overclaimed.

### Method (pre-registered)
1. **Attribution (probe-INDEPENDENT, to avoid circularity):** rank each feature by the absolute
   difference of its mean activation between the two classes, `|mean_act(c1) - mean_act(c0)|`, over the
   labeled set. This is a model-intrinsic importance, computed without the validation probe.
2. **Circuit = top-K** features by that attribution, for K ∈ {5, 10, 20, 50}.
3. **Causal faithfulness validation:** train a FRESH logistic probe on ONLY the top-K features
   (sufficiency accuracy) and compare to a FRESH probe on **K RANDOM features** (the mandatory control,
   R2/R3), plus the full-16384 probe as the ceiling. The circuit is "real/faithful" iff top-K both
   recovers most of the full accuracy AND beats random-K with a bootstrap 95% CI on the gap excluding
   zero. Seeds logged (E1); random-K uses `numpy.random.default_rng(0)`.

### Honest expectations
- If a small top-K recovers ~full accuracy and beats random-K → a **sparse, faithful** profession circuit
  (novel, with a control). If top-K barely beats random-K → the concept is **distributed**, not a sparse
  circuit (also a valid, labeled outcome, R4).
- Because Control A showed much of the probing signal is **token-level** (preserved embeddings), the
  circuit partly reflects token features; we report that caveat rather than claim a purely semantic circuit.

## Alternatives considered
- **sparse-feature-circuits library**, rejected for this budget (integration friction; needs
  dictionary_learning + nnsight + task harness). Possible follow-up.
- **Ranking by the validation probe's own coefficients**, rejected as circular (importance and the
  validation metric would share the probe). Class-mean-difference attribution is independent.
- **A generation-logit behavior**, rejected: professions are multi-token and the dataset is
  classification; a clean single-token model output isn't available here.

## Consequences
- (+) A controlled, non-circular, causally-validated circuit with no new library dependency.
- (-) Single-layer + probe-readout scope (documented), not a multi-layer model-output circuit.
- One cheap GPU run (~$0.10, like probing_eval). Commits us to the target + method above (Gate 4 to change).

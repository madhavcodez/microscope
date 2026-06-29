# ADR 0009: Multi-concept, leak-free attribution + permutation/paired statistics (pre-registered)
- Status: accepted (design); choices below fixed BEFORE the run (R3). Changing the circuit construction (train-only attribution), the concept set, or the metrics later is Gate 4.
- Date: 2026-06-28

## Context
Every conclusive *novel* claim so far, the single-layer SAE-feature circuit (ADR-0006, `circuit-g2-sae`),
the multi-layer feature-set circuit (ADR-0008), the Phase-4 randomized-model probing control, and steering,
rests on **ONE concept**: the bias-in-bios profession distinction (the two most-frequent classes, 21 vs 19).
With n=1 concept, generalization is **unproven**: we cannot tell a real sparse-circuit effect from a quirk
of that single contrast.

There is also a **test-leak** in the original construction. `circuit_eval` / `multilayer_circuit_eval`
compute the feature attribution (`|mean_act(class1) - mean_act(class0)|`) over the **FULL labeled set,
including the test split, BEFORE the train/test split**, then select top-K, then probe. The leak is
**asymmetric**: only the circuit's top-K *selection* sees test-label-informed attribution; the random-K
control draws blind. So the leak can inflate not just absolute faithfulness but the **circuit-vs-random
gap** that the whole claim turns on. This is exactly the validity-affecting kind of design issue that R2/R3
exist to catch, so it gets fixed and pre-registered, not patched silently.

Per D1 (conservative/reversible) this is a **solidification follow-up**: all three changes land in NEW,
backward-compatible functions so the original functions and their logged artifacts are **untouched** and
the old-vs-new delta stays checkable.

## Decision

Three changes, all in new functions, `circuit_multi_eval`, `probing_multi_eval`,
`saebench_sparse_probing_paper`, leaving `circuit_eval` / `multilayer_circuit_eval` / the Phase-4 control
and their artifacts as-is.

### (a) Multi-concept replication (R3, no cherry-picking)
A helper `_profession_contrasts` derives a **deterministic** set of distinct binary profession contrasts
from the top-K most-frequent professions. The **first pair reproduces the original {21, 19} contrast**
(continuity with ADR-0006/0008); the remaining pairs add diversity across other frequent professions. The
circuit and the real/random control are run across **all** derived contrasts, and an **across-concept
aggregate** is reported. **Pre-registered: every derived contrast is reported**, none dropped, regardless of
outcome (R3). The derivation rule and the resulting contrast list are logged (E2/E3) so the set is fixed
before results.

### (b) Held-out (leak-free) attribution
Attribution is computed on the **TRAIN split only**; top-K is selected from that train-only attribution;
the probe is then fit and scored on the **held-out test split**. This removes the asymmetric leak (selection
no longer sees test labels). The **original full-set attribution is computed alongside**, purely to
**quantify** how much the leak inflated the old numbers, so the survival (or not) of the ADR-0006/0008
headline is directly checkable, not asserted.

### (c) Statistics
- **Permutation null for random-K.** Replace the single random-K draw with an **R=100** random-K
  permutation null. Report the random-K **mean**, **95% band**, and a **permutation p = fraction of random
  subsets with accuracy ≥ circuit accuracy**.
- **Genuinely paired gap bootstrap.** Use **one shared resample index** for both arms (circuit and random)
  per bootstrap iteration, fixing the prior bootstrap that drew **independent** indices per arm while being
  labeled "paired". **10,000** bootstrap iterations; report the 95% CI on the (circuit − random) gap.
- **Multiple-comparison control.** Apply **Holm-Bonferroni** across the per-concept tests so the
  family-wise error from running multiple concepts is controlled.
- **SAEBench headline.** `saebench_sparse_probing_paper` widens SAEBench `sparse_probing` to the
  **multi-dataset × k ∈ {1, 2, 5}** paper headline, vs the single-dataset / k=1 **smoke** in `repro-003`.

Seeds set + logged (E1; `numpy.random.default_rng(0)`, sklearn `random_state=0`; one fixed train/test split
reused across arms within a concept for a paired comparison). Config-style params + the contrast list logged
(E2/E3).

### Pre-registered primary endpoint + success vs honest-negative (R4, BOTH acceptable)
Primary K = **K=10, single-layer** (the ADR-0006 primary).
- **SUCCESS** = at K=10 the **leak-free** circuit-vs-random gap CI still **excludes 0** AND permutation
  **p < 0.05** for the **majority of concepts after Holm**, AND **real > random** model control holds across
  concepts. Absolute faithfulness may drop **modestly** from the leaked numbers, that is expected and fine.
- **HONEST-NEGATIVE** (equally reported, R4) = faithfulness drops **sharply** (⇒ the 94-97% headline was
  substantially **leak-driven**; the headline is restated, not buried), **or** a concept's gap CI now
  **includes 0** (⇒ the effect is **concept-dependent**, not general).

No fabricated numbers. Every result row goes to **docs/EXPERIMENTS.md** and every claim traces to one (R5).

## Alternatives considered
- **Amend `circuit_eval` / `multilayer_circuit_eval` in place.** Rejected (D1): mutating the original
  functions would overwrite the leaked baseline and make the old-vs-new inflation delta uncheckable. New
  functions keep both numbers.
- **Keep n=1 concept and only fix the leak.** Rejected: leak-free-but-single-concept still leaves
  generalization unproven (R3-spirit); the cheap win is doing both together on the existing data.
- **Random subsampling of professions for the contrast set.** Rejected as non-deterministic / cherry-pick-
  adjacent; the top-K-frequency derivation is fixed and logged so the concept set is pre-registered (R3).
- **Independent-index "paired" bootstrap + single random draw (status quo).** Rejected: the prior bootstrap
  was paired in name only, and one random-K draw gives no null distribution, hence the shared-index
  bootstrap + R=100 permutation null.

## Consequences
- (+) Generalization tested across concepts, the asymmetric test-leak removed, and a real permutation null
  + genuinely paired bootstrap + Holm correction, with the leaked baseline preserved for an explicit
  inflation check. The SAEBench headline matches the paper rather than a smoke.
- (+) Conservative and reversible: original functions and artifacts untouched; new functions only (D1).
- (-) More compute than a single contrast (multiple concepts × R=100 null × 10,000 bootstrap), still cheap,
  reuses existing extracted features / labeled data; same data path as `circuit_eval` / `probing_eval`.
- **Gate-4 note.** This changes **how the circuit is constructed** (attribution is now **train-only**) and
  adds concepts + metrics, a **validity-affecting** design choice. It is therefore recorded here as an ADR
  and **flagged at Gate 4**. It is conservative and reversible (original functions and artifacts untouched);
  if a leak-free, multi-concept result diverges from the leaked single-concept headline, the **honest-
  negative** restatement above is the committed outcome (R4).

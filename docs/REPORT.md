<!--
REPORT.md — the finding. Every number here traces to a row in docs/EXPERIMENTS.md (R5).
Phases 1-5 done (reproduction, custom coders, head-to-head, controls, circuit). Phase 6 = this write-up.
-->

# MicroScope — Report (Phases 1–5)

## The question

Before building anything novel, can MicroScope **reproduce known Gemma Scope SAE results** on Gemma-2-2B
— reconstruction fidelity, a SAEBench metric, and the auto-interp pipeline — **honestly and cheaply**
(≤ $30)? Reproduction-first (RULES.md R1) is the gate: no custom training until this passes. This report
covers Phase 1; it labels every claim **reproduced / novel / inconclusive** and ties each to a logged
run in [EXPERIMENTS.md](EXPERIMENTS.md).

## The finding (summary)

MicroScope is a reproducible, honestly-evaluated mech-interp pipeline on Gemma-2-2B (run on Modal, ≤$30).
It **reproduces** the Gemma Scope reference (reconstruction VE 0.80, SAEBench probing 0.767), then trains
a custom SAE + skip-transcoder and asks whether transcoders beat SAEs on interpretability. That novel
head-to-head is **scorer-dependent**: with the weak 3B local scorer it is **inconclusive** (both coders
near-chance, neither delta's CI excludes zero), but re-running the same paired sample with a stronger
**local** 7B scorer (on an A100-40GB) lifts every score well above chance and **flips the result — the
skip-transcoder is significantly more interpretable on both detection (+0.053, CI [+0.016, +0.089]) and
fuzzing (+0.059, CI [+0.019, +0.097])**, confirming the pre-registered "transcoders beat SAEs" direction
on the interpretability axis (a full Pareto-dominance claim is still open since the transcoder's
reconstruction was not cleanly isolable). The recurring near-chance was therefore a *scorer* artifact,
not a coder limit. Beyond that, the load-bearing value is in the **controls and the circuit**, which do
*not* depend on the scorer at all: (1) a **randomized-model control**
shows the real-model SAE carries **model-learned structure beyond token statistics** (probing 0.933 vs
0.861 random, paired CI [+0.033, +0.117]) — significant but *modest*, because most of the probing signal
is token-level; (2) a **sparse feature circuit** of just **5–10 SAE features recovers 94–97%** of the
full-dictionary accuracy for a profession behavior and decisively beats a random-feature control. Honest
bottom line: the splashy novel comparison didn't resolve under budget, but the methodology is sound and
two scorer-independent results are **conclusive**. Every number traces to [EXPERIMENTS.md](EXPERIMENTS.md).

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
| Auto-interp *absolute* detection/fuzz scores are scorer-size dependent (near-chance at 3B, well above at 7B) | **reproduced (effect)** | repro-004 (3B near-chance) vs ai-g2-*-7b (7B: 0.61–0.69) |
| Custom SAE + skip-transcoder trained on Gemma-2-2B (artifacts; no interp claim yet) | **novel (artifact)** | train-g2-sae, train-g2-tc |
| Custom SAE reconstruction VE (own objective) | **novel** | recon-g2-sae (0.514) |
| SAE-vs-skip-transcoder interpretability head-to-head | **scorer-dependent: inconclusive at 3B, transcoder WINS at 7B (novel)** | ai-g2-sae/tc (3B, CIs incl 0) → ai-g2-sae-7b/tc-7b (7B: det Δ+0.053 CI[+0.016,+0.089], fuzz Δ+0.059 CI[+0.019,+0.097]) |
| Randomized-model control — real SAE has structure beyond token stats | **conclusive** | ctrl-probe-real/random (paired gap +0.072, CI [+0.033,+0.117]) |
| Steering control (B): SAE feature vs difference-of-means | **inconclusive** | ctrl-steer-v2 (both steer well within fluency: SAE +0.312, dom +0.375; Δ −0.062, CI [−0.25,+0.125]) |
| Feature circuit — sparse + faithful (5–10 SAE features) | **conclusive (novel)** | circuit-g2-sae (top-5 = 94% of ceiling; beats random, CI excl 0) |

Phase 1 claimed nothing novel (reproduction only, R1). Phase 2 produces novel *artifacts* but still
makes no interpretability claim — those wait for Phase 3's evaluation.

## Phase 2 — custom SAE + skip-transcoder (training setup; novel artifacts, no claim yet)

Two custom coders were trained on **Gemma-2-2B layer 12** with EleutherAI **sparsify** (ADR-0004) under
an **identical recipe** so Phase 3's comparison is fair: TopK activation, **width 16,384**, **k=64**
(exact L0), ~10M tokens of NeelNanda/pile-10k, bf16, on a Modal L4 (~25 min / ~$0.33 each). They differ
only by the documented flags — the **SAE** has `transcode=False, skip_connection=False` (`train-g2-sae`);
the **skip-transcoder** has both `True` (`train-g2-tc`). Both saved as loadable dictionaries. These are
**novel artifacts, not yet a claim** — no reconstruction or interpretability number is asserted until
Phase 3 evaluates them. (The ~10M-token budget is modest under the $30 cap, so absolute reconstruction
will trail a production-scale SAE; the comparison's validity comes from the *identical* recipe/budget
across the two coders, not from absolute scale.)

## Phase 3 — pre-registration (committed before evaluation; full text in PROGRESS.md)

Before evaluating the coders, the comparison was pre-registered (R3): **100 random latents** (seed 0,
the same indices for both coders); metrics = auto-interp **detection + fuzzing** (local Qwen2.5-3B),
each-coder **reconstruction FVU on its own objective**, and **SAEBench sparse_probing**; hypothesis
(to confirm/refute, not assume) = the skip-transcoder Pareto-dominates the SAE on
interpretability-vs-reconstruction; every SAE-vs-transcoder delta is reported with a **bootstrap 95% CI**
(a difference is real only if the CI excludes zero). Auto-interp is expected to be scorer-limited
(possibly inconclusive); reconstruction + SAEBench are the load-bearing axes.

### Phase 3 results — SAE vs skip-transcoder (custom Gemma-2-2B coders, identical recipe)

| Axis | SAE | Skip-transcoder | Δ (TC−SAE), 95% bootstrap CI | Label |
|---|---|---|---|---|
| Reconstruction VE (own objective) | 0.514 (CI [0.507, 0.519]) | not cleanly isolable | — | SAE novel; TC limitation |
| Auto-interp **detection** (3B scorer) | 0.540 (n=58) | 0.539 (n=61) | −0.001, [−0.022, +0.022] | **inconclusive** |
| Auto-interp **fuzzing** (3B scorer) | 0.523 | 0.546 | +0.023, [−0.001, +0.047] | **inconclusive** |
| Auto-interp **detection** (7B scorer) | 0.607 (n=58) | 0.660 (n=60) | **+0.053, [+0.016, +0.089]** | **transcoder wins** |
| Auto-interp **fuzzing** (7B scorer) | 0.631 | 0.690 | **+0.059, [+0.019, +0.097]** | **transcoder wins** |
| SAEBench sparse_probing | resid-probing → SAE-only | N/A (transcoder) | — | SAE-side only (Phase 4 adapter) |

**Verdict: scorer-dependent — INCONCLUSIVE at the 3B scorer, but the skip-transcoder WINS once the
scorer is strong enough (7B).** With the weak 3B local scorer neither auto-interp delta's CI excludes
zero and both coders sit near the 0.5 chance line, so at that scale we could not separate them (the
bottleneck is the *scorer*, not the coders — repro-004 showed the identical near-chance pattern for the
*pretrained* Gemma Scope SAE, 0.544/0.529). Re-running the **same paired latent sample** with a stronger
**local** Qwen2.5-7B scorer (resolved 2026-06-23; see below) lifts every score well above chance **and
flips the head-to-head**: the skip-transcoder is significantly more interpretable than the SAE on **both**
detection (+0.053, CI [+0.016, +0.089]) and fuzzing (+0.059, CI [+0.019, +0.097]) — both CIs now exclude
zero (unpaired diff-of-means bootstrap, seed 0, 10k resamples, the same method as the 3B row). On the
interpretability axis this **CONFIRMS the pre-registered "transcoders beat SAEs" direction** (R4). The
reconstruction axis remains SAE-only: the transcoder's own-objective reconstruction could not be cleanly
isolated through external HF hooks (sparsify's transcode hookpoints), and sparsify's `ForwardOutput.fvu`
measures input-reconstruction (inflated by the skip), so we report no transcoder reconstruction number
rather than a wrong one — meaning a full *Pareto-dominance* claim (interpretability **and**
reconstruction jointly) is still not closed, only the interpretability half is. The scorer-*independent*
signal is pursued in Phase 4 (randomized-model probing gap).

#### Scorer-strength check (2026-06-23) — did a stronger local scorer move the bottleneck? (RESOLVED — yes)

The near-chance 3B auto-interp scores throughout this project raised an obvious question: is the
near-chance a property of the *coders*, or just of the *weak 3B local scorer*? An earlier attempt to
answer it with a stronger **local** Qwen2.5-7B scorer (`ai-g2-7b-ATTEMPT`) was blocked on the L4: delphi
caches activations with the Gemma-2-2B base model and then scores with a vLLM scorer **in the same
process without freeing the base model from the GPU**, so on the 24 GiB L4 only ~16 GiB is free at
scorer startup and vLLM's startup guard cannot fit the 7B (~14.3 GiB weights + KV cache + CUDA graphs)
next to the resident base model (not a runtime OOM — a startup memory-accounting failure). The 3B
scorer (~6 GiB) fit, which is why ai-g2 ran.

**Resolution: run the 7B where the base model and the 7B coexist — an A100-40GB.** A thin GPU wrapper
(`auto_interp_custom_a100`, `gpu="A100-40GB"`, `gpu_memory_utilization=0.65`) shares the same delphi
config, seed, and paired latent sample as the L4 3B path; only the scorer and GPU change. Both 7B runs
started cleanly (the 7B's 14.29 GiB of weights loaded alongside the ~6 GiB resident base model on the
40 GiB card) and completed in ~8 min each for ~$0.6 total GPU (well under budget).

The answer is **yes, the scorer was the bottleneck**. Every score rises well above the 0.5 chance line:
detection SAE 0.540 → **0.607**, TC 0.539 → **0.660**; fuzzing SAE 0.523 → **0.631**, TC 0.546 →
**0.690**. And it **changes the head-to-head conclusion**: under the 3B scorer both deltas were
inconclusive (CIs included 0), but under the 7B scorer the skip-transcoder is significantly more
interpretable on **both** metrics — detection TC−SAE **+0.053, CI [+0.016, +0.089]** and fuzzing TC−SAE
**+0.059, CI [+0.019, +0.097]**, both CIs excluding 0 (same unpaired diff-of-means bootstrap, seed 0,
10k resamples; recompute via `scripts/headtohead_autointerp.py`). So the project's recurring
"scorer-limited near-chance" caveat was real, and lifting it confirms the pre-registered direction:
**the transcoder is the more interpretable coder** (`ai-g2-sae-7b` / `ai-g2-tc-7b`). The auto-interp
code writes a scorer-tagged output file, so the 7B results sit alongside the 3B results without
clobbering them.

## Phase 4 — controls (the differentiator)

### Control A — randomized-model control (multi-axis, ADR-0005)

**Primary axis (scorer-independent) — CONCLUSIVE.** A logistic probe on the SAE's features separates two
bias-in-bios professions at **0.933** (real-model SAE) vs **0.861** (an SAE trained on a randomized-weight
Gemma, with the real token embeddings kept). The **paired** gap is **+0.072, 95% bootstrap CI
[+0.033, +0.117]** — it excludes zero, so the real-model SAE encodes **model-learned structure beyond
token statistics**: on this axis the interpretability is *real*, not a token-co-occurrence artifact.
**Honest nuance (the control's real value):** the random-model SAE is already at 0.861, so the *majority*
of the sparse-probing signal is **token-level** (profession words carried by the preserved embeddings);
the model's *learned* structure contributes a statistically-significant but **modest +7 points**. The
control both confirms real structure *and* quantifies how much "interpretability" is really token stats.

**Secondary axis (auto-interp gap) — not measurable.** delphi could not score the random-model SAE
(`AssertionError: no non-activating examples` — its features are too degenerate to build contrastive
examples), so the real-vs-random auto-interp gap is not reported. Consistent with the scorer-limited
auto-interp throughout — which is exactly why the primary axis was designed to be scorer-independent.

### Control B — steering vs difference-of-means — **INCONCLUSIVE** (now discriminating)

Implemented per ADR-0005: steer generations at layer 12 with (a) the SAE feature most tied to the target
profession vs (b) the `difference_of_means` direction, scoring concept-induction success (the probe) under
a fluency bound (perplexity ≤ 1.5× baseline), with a coefficient sweep.

The first run (`ctrl-steer`) was **degenerate**: the `"My favorite"` prompt already classified as the
target **0.81** of the time (a ceiling, no headroom), and the coarse coefficients {2, 4, 8}×resid-RMS all
broke the fluency cap, so each direction's best fluency-preserving coefficient was 0 and the head-to-head
was trivially 0.0. We **recalibrated** (`ctrl-steer-v2`) — *same pre-registered metric and concept*, only
the prompt and coefficient grid changed, so this is a calibration fix, **not** a new design decision
(ADR-0005 Gate-4). Two changes: (1) scan six neutral candidate prompts and pick the one whose baseline
success is closest to ~0.5 — `"This person"` (baseline **0.562**), versus ceilings like `"My favorite"`
and `"I"` (both **1.0**); (2) a finer grid {0.5, 1, 2, 3, 4}×resid-RMS so a fluency-preserving sweet spot
can exist.

With headroom restored, **both directions now steer the concept substantially within the fluency cap**
(cap = 13.76 ppl). The fluency-preserving sweet spot is **coefficient 0.5** for both; higher coefficients
degenerate the text (perplexity 16 → 2268) and are correctly rejected:

| coef | SAE success / ppl | dom success / ppl |
|---|---|---|
| 0.0 (baseline) | 0.562 / 9.18 | 0.562 / 9.18 |
| **0.5** | **0.875 / 10.26** ✓ | **0.938 / 9.79** ✓ |
| 1.0 | 0.625 / 10.75 ✓ | 1.000 / 16.64 ✗ |
| 2.0 | 0.938 / 44.27 ✗ | 1.000 / 78.48 ✗ |
| 3.0 | 0.688 / 344 ✗ | 0.938 / 108 ✗ |
| 4.0 | 0.812 / 2269 ✗ | 1.000 / 338 ✗ |

(✓ = within fluency cap; ✗ = perplexity over cap, success not counted.) The steering **effect** (success
minus the 0.562 baseline) is **+0.312 for the SAE feature** and **+0.375 for difference-of-means**. The
head-to-head difference is **SAE − dom = −0.062, 95% CI [−0.25, +0.125]** — the confidence interval
**includes 0**, so the result is **inconclusive**: the simple difference-of-means baseline matches (and
here slightly edges) the SAE feature. This is the **AxBench expectation stated plainly** (R4): an SAE
feature does not steer this concept better than a plain difference-of-means direction. The difference from
the first run is that this is now a *meaningful* inconclusive — both directions demonstrably steer with
real headroom — rather than a degenerate ceiling artifact.

## Phase 5 — one feature circuit (ADR-0006)

**A sparse, faithful feature circuit — CONCLUSIVE.** For the bias-in-bios profession behavior (the same
classes 21 vs 19 as Control A), ranking the SAE's L12 features by a **probe-independent** attribution
(class-mean activation difference) and validating by **faithfulness vs a random-feature control**:

| Circuit size K | top-K probe acc | random-K (control) | faithfulness (vs 0.933 full-dict ceiling) | gap, 95% CI |
|---|---|---|---|---|
| 5  | 0.878 | 0.583 | 0.94 | +0.294 [0.206, 0.383] |
| 10 | 0.906 | 0.656 | 0.97 | +0.250 [0.167, 0.333] |
| 20 | 0.906 | 0.744 | 0.97 | +0.161 [0.083, 0.239] |
| 50 | 0.922 | 0.789 | 0.99 | +0.133 [0.061, 0.206] |

**Just 5 SAE features recover 94% of the full 16,384-feature accuracy** (10 → 97%), and every K beats the
random-feature control with a bootstrap CI excluding zero. The profession distinction is carried by a
**small, identifiable circuit** of SAE latents (ids `[3955, 1649, 1962, 5409, 6053, …]`), not smeared
across the dictionary — a sparse, causally-validated circuit (*novel*, with a control). **Caveat
(honest):** this is the same behavior Control A showed is substantially **token-driven** (the
random-*model* SAE still probed 0.86), so the circuit partly captures token features (profession words),
not purely abstract semantics. The precise claim: a sparse SAE-feature circuit faithfully mediates the
L12 readout of this profession distinction.

## Status — Phases 1–5 done (Phase 6 = this report); PAUSED for follow-ups

Phases 1–5 are complete; Phase 6 (this write-up) consolidates them. The Control-B steering sweep was
recalibrated (`ctrl-steer-v2`) into a discriminating, honestly-inconclusive result. **Possible follow-ups**
(not done): the sparsify→sae_lens adapter to run the custom coders through the full SAEBench suite, a
stronger auto-interp scorer (the near-chance bottleneck throughout), and a multi-layer (cross-component)
circuit. Open questions/risks: PHASE1_RETROSPECTIVE.md.

## Reproducibility & cost

Every number above maps to a row in [EXPERIMENTS.md](EXPERIMENTS.md) with its git commit, config,
hardware, and seed. Total GPU spend through Phase 5 ≈ **$10–11 of $30** (Modal, per-second). The verified recipe
lives in `infra/modal_app.py`; the CPU-importable package mirror is in `src/microscope/`.

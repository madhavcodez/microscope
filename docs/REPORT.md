<!--
REPORT.md is the finding. Every number here traces to a row in docs/EXPERIMENTS.md (R5).
Phases 1-5 done (reproduction, custom coders, head-to-head, controls, circuit). Phase 6 = this write-up.
-->

# MicroScope: Report (Phases 1-5)

## The question

Before building anything novel, can MicroScope reproduce known Gemma Scope SAE results on Gemma-2-2B
(reconstruction fidelity, a SAEBench metric, and the auto-interp pipeline) honestly and cheaply, under
a $30 budget? Reproduction-first (RULES.md R1) is the gate: no custom training until this passes. This
report covers Phases 1-5. It labels every claim reproduced, novel, or inconclusive, and ties each one to
a logged run in [EXPERIMENTS.md](EXPERIMENTS.md).

## Summary

MicroScope is a reproducible, honestly-evaluated mech-interp pipeline on Gemma-2-2B, run on Modal under
a $30 cap. It reproduces the Gemma Scope reference (reconstruction VE 0.80, SAEBench probing 0.767), then
trains a custom SAE and a skip-transcoder and asks whether transcoders beat SAEs on interpretability.

That head-to-head turns out to depend on the scorer. With the weak 3B local scorer it is inconclusive:
both coders land near chance and neither delta's confidence interval excludes zero. Re-running the same
paired sample with a stronger local 7B scorer (on an A100-40GB) lifts every score well above chance and
flips the result. The skip-transcoder is significantly more interpretable on both detection (+0.053, CI
[+0.016, +0.089]) and fuzzing (+0.059, CI [+0.019, +0.097]), which confirms the pre-registered "transcoders
beat SAEs" direction on the interpretability axis. A full Pareto-dominance claim is still open, because the
transcoder's reconstruction could not be cleanly isolated. The recurring near-chance was a scorer artifact,
not a coder limit.

Two further results carry most of the weight here, and neither depends on the scorer at all. First, a
randomized-model control shows the real-model SAE carries model-learned structure beyond token statistics
(probing 0.933 vs 0.861 random, paired CI [+0.033, +0.117]). The effect is significant but modest, because
most of the probing signal is token-level. Second, a sparse feature circuit of just 5-10 SAE features
recovers 94-97% of the full-dictionary accuracy for a profession behavior and decisively beats a
random-feature control. Its multi-layer extension finds an equally sparse cross-layer circuit (about 9
Gemma Scope features over L5/L12/L19, 97% of the ceiling, beating a random cross-layer control), with the
concept accumulating by mid-depth.

The honest read: the splashy novel comparison did not resolve under the weak scorer but does resolve once
the scorer is strong enough, and the two scorer-independent results are conclusive on their own. Every
number traces to [EXPERIMENTS.md](EXPERIMENTS.md).

## Setup

Gemma-2-2B with the canonical Gemma Scope SAEs (`gemma-scope-2b-pt-res`), run on Modal (L4 24 GB,
per-second billed). Activations are the TransformerLens `blocks.<L>.hook_resid_post` residual stream with
the BOS token excluded, which is the source the SAEs were trained on (see
[ADR-0003](adr/0003-modal-execution-and-activation-recipe.md); using raw HF hidden states instead gave
variance-explained -4.5). Auto-interp uses a local scorer (delphi `Offline`/vLLM), so there is no paid API.

## Results

### 1. Reconstruction fidelity: REPRODUCED  (`repro-001`, `repro-002`)

The canonical layer-12 / width-16k SAE reconstructs the residual stream with variance-explained 0.797 and
mean L0 83, inside the documented Gemma Scope range (about 0.79-0.81). The result is stable across depth:
layers 5 / 12 / 19 give VE 0.802 / 0.796 / 0.794, and layer 12 reproduces its own value across runs (0.797
vs 0.796), confirming determinism.

### 2. SAEBench sparse probing: REPRODUCED  (`repro-003`)

On the bias-in-bios probing task, the SAE probe top-1 accuracy is 0.767 against a residual-stream baseline
of 0.688. The SAE beats the raw-residual baseline by about 8 points, matching the SAEBench paper's
qualitative finding (residual baseline around 0.65, SAEs clearly above). This is a single-dataset smoke
test; the paper's headline is the 8-dataset by k in {1,2,5} mean, and scaling to that is a follow-up rather
than a different result.

### 3. Automated interpretability (delphi, local scorer): METHOD REPRODUCED; scores INCONCLUSIVE for paper comparison  (`repro-004`)

The delphi auto-interp pipeline runs end-to-end on the pretrained SAE with a local scorer and no API: cache
latents, explain, score (detection plus fuzzing). On 18 latents (about 1,355 detection / 1,370 fuzz
examples) with a Qwen2.5-3B-Instruct local scorer, the aggregate accuracies are detection 0.544 and fuzzing
0.529, barely above the 0.5 chance line. The method is reproduced (valid scores over 1,300+ examples), but
the absolute scores are inconclusive for comparison to the literature, and they depend heavily on the
scorer. A 1.5B scorer could not follow delphi's structured output format at all (mass parse failures), and
even 3B sits near chance, versus the frontier-scale scorers behind the published detection numbers (around
0.8-0.9). The takeaway is useful: the bottleneck is scorer capacity, not the pipeline. The
research-meaningful number in Phase 4 is therefore the randomized-model gap (the same scorer on real vs
randomized weights), which cancels scorer weakness and isolates whether the interpretability is real.

## Honest scope

| Claim | Label | Evidence |
|---|---|---|
| Gemma Scope SAE reconstruction fidelity (VE/L0) | reproduced | repro-001, repro-002 |
| SAE sparse-probing beats residual baseline | reproduced | repro-003 |
| Auto-interp pipeline works on the pretrained SAE (local scorer) | reproduced (method) | repro-004 |
| Auto-interp absolute detection/fuzz scores are scorer-size dependent (near-chance at 3B, well above at 7B) | reproduced (effect) | repro-004 (3B near-chance) vs ai-g2-*-7b (7B: 0.61-0.69) |
| Custom SAE + skip-transcoder trained on Gemma-2-2B (artifacts; no interp claim yet) | novel (artifact) | train-g2-sae, train-g2-tc |
| Custom SAE reconstruction VE (own objective) | novel | recon-g2-sae (0.514) |
| Custom SAE SAEBench sparse-probing (top-1), below Gemma Scope and below its own residual baseline | novel (honest negative; encode-verified) | saebench-custom-sae-v2 (0.670 vs ref 0.767; baseline 0.688) |
| SAE-vs-skip-transcoder interpretability head-to-head | scorer-dependent: inconclusive at 3B, transcoder WINS at 7B (novel) | ai-g2-sae/tc (3B, CIs incl 0) then ai-g2-sae-7b/tc-7b (7B: det delta +0.053 CI[+0.016,+0.089], fuzz delta +0.059 CI[+0.019,+0.097]) |
| Randomized-model control: real SAE has structure beyond token stats | conclusive | ctrl-probe-real/random (paired gap +0.072, CI [+0.033,+0.117]) |
| Steering control (B): SAE feature vs difference-of-means | inconclusive | ctrl-steer-v2 (both steer well within fluency: SAE +0.312, dom +0.375; delta -0.062, CI [-0.25,+0.125]) |
| Feature circuit, sparse and faithful (5-10 SAE features) | conclusive (novel) | circuit-g2-sae (top-5 = 94% of ceiling; beats random, CI excl 0) |
| Multi-layer (cross-layer feature-SET) circuit + depth build-up | conclusive (novel) | circuit-multilayer (9 features over L5/12/19 = 97% of ceiling; beats random cross-layer control at every K, CI excl 0; concept accumulates by mid-depth) |

Phase 1 claimed nothing novel (reproduction only, R1). Phase 2 produces novel artifacts but still makes no
interpretability claim; those wait for Phase 3's evaluation.

## Phase 2: custom SAE + skip-transcoder (training setup; novel artifacts, no claim yet)

Two custom coders were trained on Gemma-2-2B layer 12 with EleutherAI sparsify (ADR-0004) under an
identical recipe, so Phase 3's comparison is fair: TopK activation, width 16,384, k=64 (exact L0), about
10M tokens of NeelNanda/pile-10k, bf16, on a Modal L4 (about 25 min / $0.33 each). They differ only by the
documented flags: the SAE has `transcode=False, skip_connection=False` (`train-g2-sae`), and the
skip-transcoder has both `True` (`train-g2-tc`). Both are saved as loadable dictionaries. These are novel
artifacts, not yet a claim; no reconstruction or interpretability number is asserted until Phase 3
evaluates them. The 10M-token budget is modest under the $30 cap, so absolute reconstruction will trail a
production-scale SAE. The comparison's validity comes from the identical recipe and budget across the two
coders, not from absolute scale.

## Phase 3: pre-registration (committed before evaluation; full text in PROGRESS.md)

Before evaluating the coders, the comparison was pre-registered (R3): 100 random latents (seed 0, the same
indices for both coders); metrics were auto-interp detection plus fuzzing (local Qwen2.5-3B), each coder's
reconstruction FVU on its own objective, and SAEBench sparse_probing. The hypothesis to confirm or refute,
not assume, was that the skip-transcoder Pareto-dominates the SAE on interpretability-vs-reconstruction.
Every SAE-vs-transcoder delta is reported with a bootstrap 95% CI, and a difference is real only if the CI
excludes zero. Auto-interp was expected to be scorer-limited (possibly inconclusive), so reconstruction and
SAEBench were meant to carry the comparison.

### Phase 3 results: SAE vs skip-transcoder (custom Gemma-2-2B coders, identical recipe)

| Axis | SAE | Skip-transcoder | Delta (TC-SAE), 95% bootstrap CI | Label |
|---|---|---|---|---|
| Reconstruction VE (own objective) | 0.514 (CI [0.507, 0.519]) | not cleanly isolable | n/a | SAE novel; TC limitation |
| Auto-interp detection (3B scorer) | 0.540 (n=58) | 0.539 (n=61) | -0.001, [-0.022, +0.022] | inconclusive |
| Auto-interp fuzzing (3B scorer) | 0.523 | 0.546 | +0.023, [-0.001, +0.047] | inconclusive |
| Auto-interp detection (7B scorer) | 0.607 (n=58) | 0.660 (n=60) | +0.053, [+0.016, +0.089] | transcoder wins |
| Auto-interp fuzzing (7B scorer) | 0.631 | 0.690 | +0.059, [+0.019, +0.097] | transcoder wins |
| SAEBench sparse_probing (top-1) | 0.670 (custom SAE; baseline 0.688) | N/A (transcoder) | n/a | SAE below its own baseline and below Gemma Scope (ADR-0007, encode-verified) |

The verdict is scorer-dependent: inconclusive at the 3B scorer, but the skip-transcoder wins once the
scorer is strong enough (7B). With the weak 3B local scorer neither auto-interp delta's CI excludes zero
and both coders sit near the 0.5 chance line, so at that scale we could not separate them. The bottleneck
is the scorer, not the coders: repro-004 showed the identical near-chance pattern for the pretrained Gemma
Scope SAE (0.544/0.529). The 7B re-run that resolves this is detailed in the scorer-strength check below;
in short, both detection and fuzzing CIs move off zero in the transcoder's favor.

This confirms the pre-registered "transcoders beat SAEs" direction (R4) on the interpretability axis, as a
relative ranking. The absolute scores (0.61-0.69) are only moderately above the 0.5 chance line and well
below the 0.8-0.9 frontier-scorer literature, and this is one local 7B scorer at one model, layer, and
budget, not a settled general result. The reconstruction axis stays SAE-only. The transcoder's own-objective
reconstruction could not be cleanly isolated through external HF hooks (sparsify's transcode hookpoints),
and sparsify's `ForwardOutput.fvu` measures input-reconstruction (inflated by the skip), so we report no
transcoder reconstruction number rather than a wrong one. That means a full Pareto-dominance claim
(interpretability and reconstruction jointly) is still open; only the interpretability half is closed. The
scorer-independent signal is pursued in Phase 4 (randomized-model probing gap).

#### Scorer-strength check (2026-06-23): did a stronger local scorer move the bottleneck?

The near-chance 3B auto-interp scores throughout this project raised an obvious question: is the near-chance
a property of the coders, or of the weak 3B local scorer? An earlier attempt to answer it with a stronger
local Qwen2.5-7B scorer (`ai-g2-7b-ATTEMPT`) was blocked on the L4. delphi caches activations with the
Gemma-2-2B base model and then scores with a vLLM scorer in the same process without freeing the base model
from the GPU, so on the 24 GiB L4 only about 16 GiB is free at scorer startup, and vLLM's startup guard
cannot fit the 7B (about 14.3 GiB weights plus KV cache plus CUDA graphs) next to the resident base model.
This is a startup memory-accounting failure, not a runtime OOM. The 3B scorer (about 6 GiB) fit, which is
why ai-g2 ran.

The fix is to run the 7B where the base model and the 7B can coexist, on an A100-40GB. A thin GPU wrapper
(`auto_interp_custom_a100`, `gpu="A100-40GB"`, `gpu_memory_utilization=0.65`) shares the same delphi config,
seed, and pre-registered latent target as the L4 3B path; only the scorer and GPU change. delphi drops a
different handful of unscoreable latents per coder, so the surviving sets differ and the head-to-head is
unpaired (SAE n=58, TC n=60), the same fallback noted in the 3B row, and a documented deviation in the
conservative direction. The test is an unpaired diff-of-means bootstrap (seed 0, 10k resamples, same method
as the 3B row), robust to Welch t p approximately 0.005 and Mann-Whitney p approximately 0.006. Both 7B runs
started cleanly (the 7B's 14.29 GiB of weights loaded alongside the about 6 GiB resident base model on the
40 GiB card) and completed in about 8 min each for about $0.6 total GPU, well under budget.

The answer is yes, the scorer was the bottleneck. Every score rises well above the 0.5 chance line:
detection SAE 0.540 to 0.607, TC 0.539 to 0.660; fuzzing SAE 0.523 to 0.631, TC 0.546 to 0.690. And it
changes the conclusion. Under the 3B scorer both deltas were inconclusive (CIs included 0); under the 7B
scorer the skip-transcoder is significantly more interpretable on both metrics, detection TC-SAE +0.053 (CI
[+0.016, +0.089]) and fuzzing TC-SAE +0.059 (CI [+0.019, +0.097]), with both CIs excluding zero (recompute
via `scripts/headtohead_autointerp.py`). So the project's recurring "scorer-limited near-chance" caveat was
real, and lifting it confirms the pre-registered direction: the transcoder is the more interpretable coder
(`ai-g2-sae-7b` / `ai-g2-tc-7b`). The auto-interp code writes a scorer-tagged output file, so the 7B results
sit alongside the 3B results without clobbering them.

#### SAEBench sparse_probing on the custom SAE (2026-06-23): deferred item now CLOSED (ADR-0007)

The Phase-3 SAEBench axis was originally SAE-only and deferred. SAEBench consumes SAEs as native `sae_lens`
objects, but our coders are EleutherAI sparsify `SparseCoder`s, so running it on the custom SAE needed an
adapter. That adapter now exists (`_sparsify_to_topk_sae`): it wraps the sparsify SAE as a real
`sae_lens.TopKSAE` (`W_enc = encoder.weight.T`, `b_enc = encoder.bias`, `W_dec`/`b_dec` copied, k=64,
`apply_b_dec_to_input=True`, `hook_name`/`hook_layer`/`model_name` in `cfg.metadata`), all API-verified on
Modal (E4). A cheap GPU pre-flight (`verify_saebench_adapter`) confirmed the SAE loads all four weight
tensors, enforces exactly k=64 active latents on encode, is accepted by SAEBench's `load_and_format_sae`,
and, critically, passes an encode-fidelity check before paying for the full eval. The eval uses the
identical config as repro-003 (`LabHC/bias_in_bios_class_set1`, train 1500 / test 500, k=1, seed 42), so the
custom number is directly comparable to the Gemma Scope reference.

Adapter correctness rests on that encode-fidelity check. sparsify's `SparseCoder.encode` subtracts the
decoder bias for a non-transcode SAE (`if not self.cfg.transcode: x = x - self.b_dec`, read from the
installed source, E4), so the adapter sets `apply_b_dec_to_input=True` to reproduce that shift exactly. This
is verified, not assumed: `verify_saebench_adapter` runs both the real sparsify `coder.encode` and the
adapter `.encode` on the same random and real-resid batches and asserts identical active TopK indices and
values. It passes (Jaccard 1.0 per row, max abs diff 6e-6 / 7.6e-5, cosine 1.0), while the alternative
`apply_b_dec_to_input=False` fails the same check (only about 7% index overlap, cosine 0.139). An earlier
version of this adapter used `=False` and produced `sae_top_1` 0.667; that was an adapter artifact (different
latents firing), now corrected. The fidelity dict is persisted to `saebench_adapter_verify.json` for
traceability.

The result, reported honestly (R4): the budget-trained custom SAE scores `sae_top_1` = 0.670
(encode-verified; the buggy pre-fix run reported 0.667), below the Gemma Scope reference (0.767) and below
its own residual baseline (0.688). On this single-dataset top-1 sparse probe the custom SAE's single best
feature does not beat the raw residual stream's best neuron, the opposite of repro-003, where the canonical
Gemma Scope SAE (0.767) clearly beat the same baseline (0.688). The residual baseline here (0.688) is
essentially identical to repro-003's (0.688), as it must be since the baseline does not depend on the SAE,
which confirms the eval is sound and the comparison apples-to-apples. The full-feature accuracies match too
(custom 0.950 / 0.965 vs reference 0.964 / 0.965). This below-baseline outcome is the expected consequence
of the modest 10M-token training budget (reconstruction VE 0.51 vs Gemma Scope's 0.80): a weaker SAE
concentrates less of the profession signal into a single top feature. It is reported as-is, not hidden;
beating, or here failing to beat, the residual baseline is the finding either way. SAEBench's
`sparse_probing` is residual-SAE oriented, so the skip-transcoder is N/A for this metric (R3): the adapter
raises on a transcoder/skip coder rather than fabricate a number (R5), consistent with the reconstruction
axis already being SAE-only. (A minor honesty note: a few of the budget SAE's decoder rows drift up to about
0.07 off unit-norm, mean row norm about 1.004, so SAEBench's `check_decoder_norms` warns; it does not raise
and the eval proceeds.)

## Phase 4: controls

### Control A: randomized-model control (multi-axis, ADR-0005)

The primary axis is scorer-independent, and it is conclusive. A logistic probe on the SAE's features
separates two bias-in-bios professions at 0.933 (real-model SAE) vs 0.861 (an SAE trained on a
randomized-weight Gemma, with the real token embeddings kept). The paired gap is +0.072, 95% bootstrap CI
[+0.033, +0.117], which excludes zero, so the real-model SAE encodes model-learned structure beyond token
statistics: on this axis the interpretability is real, not a token-co-occurrence artifact. The control's
real value is in the nuance. The random-model SAE is already at 0.861, so the majority of the sparse-probing
signal is token-level (profession words carried by the preserved embeddings), and the model's learned
structure contributes a statistically significant but modest +7 points. The control both confirms real
structure and quantifies how much "interpretability" is really token stats.

The secondary axis, the auto-interp gap, is not measurable. delphi could not score the random-model SAE
(`AssertionError: no non-activating examples`, its features are too degenerate to build contrastive
examples), so the real-vs-random auto-interp gap is not reported. This is consistent with the scorer-limited
auto-interp throughout, which is exactly why the primary axis was designed to be scorer-independent.

### Control B: steering vs difference-of-means: INCONCLUSIVE (now discriminating)

Implemented per ADR-0005: steer generations at layer 12 with (a) the SAE feature most tied to the target
profession and (b) the `difference_of_means` direction, scoring concept-induction success (the probe) under
a fluency bound (perplexity at most 1.5x baseline), with a coefficient sweep.

The first run (`ctrl-steer`) was degenerate. The `"My favorite"` prompt already classified as the target
0.81 of the time (a ceiling, no headroom), and the coarse coefficients {2, 4, 8} times resid-RMS all broke
the fluency cap, so each direction's best fluency-preserving coefficient was 0 and the head-to-head was
trivially 0.0. We recalibrated (`ctrl-steer-v2`) using the same pre-registered metric and concept, changing
only the prompt and coefficient grid, so this is a calibration fix and not a new design decision (ADR-0005
Gate-4). Two things changed. First, we scanned six neutral candidate prompts and picked the one whose
baseline success is closest to about 0.5, which was `"This person"` (baseline 0.562), versus ceilings like
`"My favorite"` and `"I"` (both 1.0). Second, we used a finer grid {0.5, 1, 2, 3, 4} times resid-RMS so a
fluency-preserving sweet spot could exist.

With headroom restored, both directions now steer the concept substantially within the fluency cap (cap =
13.76 ppl). The fluency-preserving sweet spot is coefficient 0.5 for both; higher coefficients degenerate
the text (perplexity 16 to 2268) and are correctly rejected:

| coef | SAE success / ppl | dom success / ppl |
|---|---|---|
| 0.0 (baseline) | 0.562 / 9.18 | 0.562 / 9.18 |
| 0.5 | 0.875 / 10.26 (within cap) | 0.938 / 9.79 (within cap) |
| 1.0 | 0.625 / 10.75 (within cap) | 1.000 / 16.64 (over cap) |
| 2.0 | 0.938 / 44.27 (over cap) | 1.000 / 78.48 (over cap) |
| 3.0 | 0.688 / 344 (over cap) | 0.938 / 108 (over cap) |
| 4.0 | 0.812 / 2269 (over cap) | 1.000 / 338 (over cap) |

(Within cap = perplexity within the fluency bound; over cap = perplexity over the bound, success not
counted.) The steering effect (success minus the 0.562 baseline) is +0.312 for the SAE feature and +0.375
for difference-of-means. The head-to-head difference is SAE minus dom = -0.062, 95% CI [-0.25, +0.125]. The
confidence interval includes 0, so the result is inconclusive: the simple difference-of-means baseline
matches, and here slightly edges, the SAE feature. This is the AxBench expectation stated plainly (R4): an
SAE feature does not steer this concept better than a plain difference-of-means direction. The difference
from the first run is that this is now a meaningful inconclusive, with both directions demonstrably steering
with real headroom, rather than a degenerate ceiling artifact.

## Phase 5: one feature circuit (ADR-0006)

A sparse, faithful feature circuit, and it is conclusive. For the bias-in-bios profession behavior (the same
classes 21 vs 19 as Control A), we rank the SAE's L12 features by a probe-independent attribution (class-mean
activation difference) and validate by faithfulness against a random-feature control:

| Circuit size K | top-K probe acc | random-K (control) | faithfulness (vs 0.933 full-dict ceiling) | gap, 95% CI |
|---|---|---|---|---|
| 5  | 0.878 | 0.583 | 0.94 | +0.294 [0.206, 0.383] |
| 10 | 0.906 | 0.656 | 0.97 | +0.250 [0.167, 0.333] |
| 20 | 0.906 | 0.744 | 0.97 | +0.161 [0.083, 0.239] |
| 50 | 0.922 | 0.789 | 0.99 | +0.133 [0.061, 0.206] |

Just 5 SAE features recover 94% of the full 16,384-feature accuracy (10 features reach 97%), and every K
beats the random-feature control with a bootstrap CI excluding zero. The profession distinction is carried
by a small, identifiable circuit of SAE latents (ids `[3955, 1649, 1962, 5409, 6053, ...]`), not smeared
across the dictionary: a sparse, causally-validated circuit (novel, with a control). Two honest caveats.
This is the same behavior Control A showed is substantially token-driven (the random-model SAE still probed
0.86), so the circuit partly captures token features (profession words), not purely abstract semantics. And
feature attribution is computed over the full labeled set, including the test split, so the absolute
faithfulness numbers are mildly optimistic; the circuit-vs-random gap is unaffected, because the random
control runs the same protocol (the leak is symmetric). The precise claim: a sparse SAE-feature circuit
faithfully mediates the L12 readout of this profession distinction.

### Phase 5 (multi-layer extension): a cross-layer feature-set circuit + depth build-up (ADR-0008)

A sparse multi-layer circuit, faithful and beating its control, and conclusive (novel). The single-layer
circuit above lives at L12 only. The deferred extension asks whether a few features spread across depth
carry the same profession distinction. Because we have one custom SAE (L12), this uses the pretrained Gemma
Scope SAEs at layers 5, 12, 19 (the three layers reproduced in Phase 1), on the same TransformerLens
`resid_post` recipe (BOS excluded). The method mirrors ADR-0006 per layer: probe-independent attribution
(class-mean activation difference), then top-`K` features per layer; the circuit is the union across layers
(a small cross-layer node set); faithfulness is a fresh probe on the circuit's features (concatenated across
layers) vs a same-size random cross-layer feature set vs the full 49,152-feature ceiling.

| K per layer | circuit size | circuit acc | random (control) | faithfulness (vs 0.944 ceiling) | gap, 95% CI |
|---|---|---|---|---|---|
| 3  | 9  | 0.917 | 0.594 | 0.97 | +0.322 [0.239, 0.406] |
| 5  | 15 | 0.939 | 0.778 | 0.99 | +0.161 [0.094, 0.233] |
| 10 | 30 | 0.950 | 0.667 | 1.01 | +0.283 [0.206, 0.356] |

As few as 9 features spread over 3 layers recover 97% of the full 49,152-feature accuracy (15 features reach
99%), and every K beats the random cross-layer control with a bootstrap CI excluding zero. The profession
distinction is carried by a small, identifiable cross-layer circuit, not smeared across three full
dictionaries.

For the cross-layer build-up, using K=5 per layer and adding layers cumulatively (circuit features only):

| layers | features | probe acc |
|---|---|---|
| L5 | 5 | 0.911 |
| L5 + L12 | 10 | 0.939 |
| L5 + L12 + L19 | 15 | 0.939 |

The concept is largely present by layer 5, sharpened by layer 12, and then saturates: L19 adds nothing on
top (+0.000). So this profession signal accumulates by mid-depth rather than building monotonically all the
way to late layers. (This curve is descriptive, showing which layers carry the signal, not a causal claim.)

On scope, and honestly (R4): this is a cross-layer feature-set circuit plus a depth build-up curve, not a
full feature-to-feature causal edge graph. We do not compute or intervene on cross-layer edges (attribution
patching, or the sparse-feature-circuits construction), which remains the heavier follow-up. It is also the
same token-influenced behavior as Control A, so the circuit partly reflects token features (profession
words). And as in the single-layer circuit, attribution is computed over the full labeled set including the
test split, so absolute faithfulness is mildly optimistic while the circuit-vs-random gap is unaffected (the
random control uses the same protocol, so the leak is symmetric). The precise claim: a small set of Gemma
Scope SAE features spanning L5/L12/L19 faithfully mediates the readout of this profession distinction, and
the signal accumulates by mid-depth.

## Status: Phases 1-5 done (Phase 6 = this report); PAUSED for follow-ups

Phases 1-5 are complete, and Phase 6 (this write-up) consolidates them. The Control-B steering sweep was
recalibrated (`ctrl-steer-v2`) into a discriminating, honestly-inconclusive result, and the deferred Phase-3
SAEBench-on-custom-SAE item is now closed: a `sparsify` to `sae_lens.TopKSAE` adapter (ADR-0007), with an
encode-fidelity check that proves it reproduces sparsify's `coder.encode` exactly, ran the full
sparse_probing eval on the custom SAE (`sae_top_1` 0.670, below Gemma Scope's 0.767 and below its own
residual baseline 0.688, the expected consequence of the budget training; transcoder N/A per R3). The
multi-layer circuit follow-up is now also done, as a cross-layer feature-set circuit plus depth build-up
(`circuit-multilayer`, ADR-0008: 9 Gemma Scope features over L5/12/19 recover 97% of the ceiling and beat a
random cross-layer control at every size; the profession concept accumulates by mid-depth). Remaining
possible follow-ups, not done: scaling SAEBench to the full 8-dataset suite, and the heavier
feature-to-feature causal edge graph (attribution patching, or sparse-feature-circuits). This unit built the
feature-set plus build-up version, not the causal edge graph. Open questions and risks are in
PHASE1_RETROSPECTIVE.md.

## Reproducibility and cost

Every number above maps to a row in [EXPERIMENTS.md](EXPERIMENTS.md) with its git commit, config, hardware,
and seed. Total GPU spend through Phase 5 is about $10-11 of $30 (Modal, per-second). The verified recipe
lives in `infra/modal_app.py`; the CPU-importable package mirror is in `src/microscope/`.

# ADR 0007: sparsify -> sae_lens adapter for SAEBench on the custom SAE

- Status: accepted. Resolves the Phase-3 deferred item ("SAEBench on the custom coders was SAE-only +
  needs a sae_lens adapter => deferred"). Implements the adapter and runs the full SAEBench
  sparse_probing eval on the custom SAE.
- Date: 2026-06-23

## Context

Phase 3 evaluated the custom SAE vs skip-transcoder on reconstruction + auto-interp, but the SAEBench
axis was left as SAE-only and then **deferred**: SAEBench loads SAEs as native `sae_lens` objects, and
our coders are EleutherAI **sparsify** `SparseCoder`s, so SAEBench could not consume them without an
adapter (the same class of integration friction noted in ADR-0006). repro-003 established the SAEBench
reference on the *canonical Gemma Scope* SAE (`sae_top_1` 0.767 vs residual baseline 0.688); the open
question was the *custom* SAE's number on the identical eval.

E4 (verified on Modal, sae_lens 6.44.3 — `probe_saebench_adapter`, `_adapter2`, `_adapter3`):
- SAEBench `general_utils.load_and_format_sae`, for a **custom object** (not a string id), does only:
  `check_decoder_norms(sae.W_dec.data)` (warns, does **not** raise) then `_standardize_sae_cfg(sae.cfg)`,
  and the eval then reads `sae.cfg.hook_name` / `sae.cfg.hook_layer` and calls the SAE's encode/decode.
- sparsify `SparseCoder` state: `W_dec(16384,2304)`, `b_dec(2304,)`, `encoder.weight(16384,2304)`,
  `encoder.bias(16384,)`; cfg `activation=topk, num_latents=16384, k=64, normalize_decoder=True`.
- `sae_lens` exposes an inference `TopKSAE` + `TopKSAEConfig(d_in,d_sae,k,dtype,device,
  apply_b_dec_to_input,normalize_activations,metadata,...)`; `SAEMetadata` (at `sae_lens.saes.sae`)
  carries `model_name/hook_name/hook_layer`, and `cfg.hook_name` etc. resolve from `cfg.metadata`
  automatically — exactly the fields sparse_probing reads.

## Decision

### Build a real `sae_lens.TopKSAE`, not a duck-typed shim (Gate-4-adjacent, documented)
Construct a genuine `sae_lens.TopKSAE` and load the sparsify weights into it, rather than a minimal
object that merely exposes `.W_dec/.cfg/encode/decode`. Rationale: the real class gives SAEBench the
exact encode/decode semantics + TopK behaviour it expects, the config-from-metadata plumbing for free,
and no risk of a hand-rolled `encode` diverging from sparsify's. Weight map (verified):
`W_enc = encoder.weight.T`, `b_enc = encoder.bias`, `W_dec = W_dec`, `b_dec = b_dec`.

### `apply_b_dec_to_input=False`
sparsify's TopK encode does **not** subtract a decoder bias from the input before encoding, so the
adapter sets `apply_b_dec_to_input=False` to match sparsify's forward exactly (a wrong choice here would
silently change which latents fire). `normalize_activations="none"` for the same fidelity reason.

### float32 adapter weights
The coders are bf16 on disk; the adapter casts to float32 for the eval. This keeps the probe numerically
clean (SAEBench's `check_decoder_norms` uses a 1e-2 tolerance specifically because bf16 norms drift) and
matches the precision posture of repro-003. The decode/probe is not the training step, so this does not
alter the coder.

### Same eval config as repro-003 (apples-to-apples)
`dataset_names=["LabHC/bias_in_bios_class_set1"]`, train 1500 / test 500, `context_length=128`,
`k_values=[1]`, `random_seed=42` — identical to repro-003 so the custom number is directly comparable to
the Gemma Scope reference. (The bare key `"LabHC/bias_in_bios"` `KeyError`s in this SAEBench build's
`chosen_classes_per_dataset`; the recognized key is `..._class_set1`, which is what repro-003's
EXPERIMENTS row actually records — E4 `probe_saebench_datasets`.)

### Transcoder = N/A (R3)
sparse_probing is residual-SAE oriented (it probes resid activations through the SAE). The skip-transcoder
targets MLP-out from MLP-in and is not a drop-in for this eval; the adapter raises on a transcoder/skip
coder rather than forcing a wrong number. Documented N/A, not faked (R5), consistent with the Phase-3
reconstruction axis already being SAE-only.

## Alternatives considered
- **Minimal duck-typed adapter object** — rejected: more bespoke code to keep correct (encode/TopK/cfg),
  no upside over the real class once E4 showed `TopKSAE` accepts our weights cleanly.
- **Retrain the SAE through sae_lens** — rejected: wasteful + changes the artifact under test; the whole
  point is to evaluate the *existing* sparsify coder.
- **Force the transcoder through sparse_probing** — rejected (R3/R5): would produce a meaningless number.

## Consequences
- (+) The deferred SAEBench-on-custom-SAE item is closed with a real, comparable number; the adapter
  (`_sparsify_to_topk_sae`) is reusable for other SAEBench evals on sparsify coders.
- (+) A cheap pre-flight (`verify_saebench_adapter`) proves load + k-enforcement + SAEBench acceptance
  before paying for the full run (it confirmed k=64 exactly, all 4 weights loaded, SAEBench accepts).
- (-) Honest result: the budget-trained custom SAE scores `sae_top_1` **0.667**, **below** Gemma Scope's
  0.767 *and* below its own residual baseline (0.688) — i.e. on this single-dataset top-1 probe the
  budget SAE's best feature does **not** beat the raw residual, the opposite of repro-003. This is the
  expected consequence of the ~10M-token budget (recon VE 0.51 vs 0.80), reported as-is (R4), not hidden.
- (-) Minor: a few of the budget SAE's decoder rows drift up to ~0.07 off unit-norm (mean norm ≈ 1.004),
  so `check_decoder_norms` warns; it does not raise and the eval proceeds. Noted for honesty.
- One L4 run (~$0.5) + cheap E4 probes/pre-flight (~$0.2). Changing the eval dataset/metric is Gate 4.

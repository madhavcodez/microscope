# ADR 0007: sparsify -> sae_lens adapter for SAEBench on the custom SAE

- Status: accepted (AMENDED 2026-06-23, `apply_b_dec_to_input` corrected from False to True after a review
  finding; see "Correction" below). Resolves the Phase-3 deferred item ("SAEBench on the custom coders
  was SAE-only + needs a sae_lens adapter => deferred"). Implements the adapter and runs the full
  SAEBench sparse_probing eval on the custom SAE.
- Date: 2026-06-23

## Correction (2026-06-23): apply_b_dec_to_input was wrong (False -> True)

The original decision below set `apply_b_dec_to_input=False` on the premise that "sparsify's TopK encode
does not subtract a decoder bias from the input." **That premise is factually wrong.** The installed
sparsify `SparseCoder.encode` (read verbatim via `probe_sparsify_encode`, E4) is:

```python
def encode(self, x):
    if not self.cfg.transcode:
        x = x - self.b_dec          # <-- non-transcode SAEs DO subtract b_dec
    return fused_encoder(x, self.encoder.weight, self.encoder.bias, self.cfg.k, self.cfg.activation)
```

The coder under test has `cfg.transcode=False`, so sparsify's true encode is `(x - b_dec) @ W_encᵀ + b_enc`
(b_dec norm ≈ 90.7, a large per-latent pre-activation shift), while the buggy adapter computed
`x @ W_encᵀ + b_enc`. That shift changes WHICH TopK latents fire, exactly what sparse_probing consumes -
so the v1 number (`sae_top_1` 0.6668) was an **adapter artifact**, not the SAE's real number.

**Fix:** `apply_b_dec_to_input=True`. b_dec is already copied into the sae_lens SAE, so sae_lens now
performs the identical `(x - b_dec)` shift. This is verified, not assumed: the new ENCODE-FIDELITY check in
`verify_saebench_adapter` (persisted to `saebench_adapter_verify.json`) runs BOTH the real sparsify
`coder.encode` and the adapter `.encode` on the same random AND real-resid batches and asserts identical
active TopK indices + values within tolerance. With `=True` it PASSES (Jaccard 1.0 every row, max abs diff
6e-6 / 7.6e-5, cosine 1.0); the buggy `=False` variant FAILS the same check (Jaccard ≈ 0.07, cosine 0.139).
This fidelity assertion, not the eval-config matching, is the **real correctness evidence** for the adapter,
and is the check that would have caught the bug at pre-flight.

**Corrected result:** `sae_top_1` **0.670** (vs the buggy 0.667). The conclusion is UNCHANGED and now real:
still below Gemma Scope 0.767 and below the residual baseline 0.688, the honest negative stands, encode-verified.
The b_dec fix moved the number only +0.003 because a top-1 best-single-feature probe is robust to which of
the (near-equivalent) budget-SAE latents win; the bug was nonetheless real and could have changed the result.

The rest of this ADR is the original v1 reasoning, retained for history; read `apply_b_dec_to_input=False`
below as **superseded by the correction above**.

## Context

Phase 3 evaluated the custom SAE vs skip-transcoder on reconstruction + auto-interp, but the SAEBench
axis was left as SAE-only and then **deferred**: SAEBench loads SAEs as native `sae_lens` objects, and
our coders are EleutherAI **sparsify** `SparseCoder`s, so SAEBench could not consume them without an
adapter (the same class of integration friction noted in ADR-0006). repro-003 established the SAEBench
reference on the *canonical Gemma Scope* SAE (`sae_top_1` 0.767 vs residual baseline 0.688); the open
question was the *custom* SAE's number on the identical eval.

E4 (verified on Modal, sae_lens 6.44.3, `probe_saebench_adapter`, `_adapter2`, `_adapter3`):
- SAEBench `general_utils.load_and_format_sae`, for a **custom object** (not a string id), does only:
  `check_decoder_norms(sae.W_dec.data)` (warns, does **not** raise) then `_standardize_sae_cfg(sae.cfg)`,
  and the eval then reads `sae.cfg.hook_name` / `sae.cfg.hook_layer` and calls the SAE's encode/decode.
- sparsify `SparseCoder` state: `W_dec(16384,2304)`, `b_dec(2304,)`, `encoder.weight(16384,2304)`,
  `encoder.bias(16384,)`; cfg `activation=topk, num_latents=16384, k=64, normalize_decoder=True`.
- `sae_lens` exposes an inference `TopKSAE` + `TopKSAEConfig(d_in,d_sae,k,dtype,device,
  apply_b_dec_to_input,normalize_activations,metadata,...)`; `SAEMetadata` (at `sae_lens.saes.sae`)
  carries `model_name/hook_name/hook_layer`, and `cfg.hook_name` etc. resolve from `cfg.metadata`
  automatically, exactly the fields sparse_probing reads.

## Decision

### Build a real `sae_lens.TopKSAE`, not a duck-typed shim (Gate-4-adjacent, documented)
Construct a genuine `sae_lens.TopKSAE` and load the sparsify weights into it, rather than a minimal
object that merely exposes `.W_dec/.cfg/encode/decode`. Rationale: the real class gives SAEBench the
exact encode/decode semantics + TopK behaviour it expects, the config-from-metadata plumbing for free,
and no risk of a hand-rolled `encode` diverging from sparsify's. Weight map (verified):
`W_enc = encoder.weight.T`, `b_enc = encoder.bias`, `W_dec = W_dec`, `b_dec = b_dec`.

### `apply_b_dec_to_input=False` , SUPERSEDED (see "Correction" above; the correct value is True)
~~sparsify's TopK encode does **not** subtract a decoder bias from the input before encoding, so the
adapter sets `apply_b_dec_to_input=False` to match sparsify's forward exactly~~, this premise is WRONG.
sparsify's non-transcode `encode` DOES subtract b_dec (`if not self.cfg.transcode: x = x - self.b_dec`),
so the correct setting is `apply_b_dec_to_input=True`. The original note that "a wrong choice here would
silently change which latents fire" was exactly right about the risk, and the wrong choice was made; the
encode-fidelity check now guards it. `normalize_activations="none"` is unaffected (sparsify does not
normalize activations) and remains correct.

### float32 adapter weights
The coders are bf16 on disk; the adapter casts to float32 for the eval. This keeps the probe numerically
clean (SAEBench's `check_decoder_norms` uses a 1e-2 tolerance specifically because bf16 norms drift) and
matches the precision posture of repro-003. The decode/probe is not the training step, so this does not
alter the coder.

### Same eval config as repro-003 (apples-to-apples)
`dataset_names=["LabHC/bias_in_bios_class_set1"]`, train 1500 / test 500, `context_length=128`,
`k_values=[1]`, `random_seed=42`, identical to repro-003 so the custom number is directly comparable to
the Gemma Scope reference. (The bare key `"LabHC/bias_in_bios"` `KeyError`s in this SAEBench build's
`chosen_classes_per_dataset`; the recognized key is `..._class_set1`, which is what repro-003's
EXPERIMENTS row actually records, E4 `probe_saebench_datasets`.)

### Transcoder = N/A (R3)
sparse_probing is residual-SAE oriented (it probes resid activations through the SAE). The skip-transcoder
targets MLP-out from MLP-in and is not a drop-in for this eval; the adapter raises on a transcoder/skip
coder rather than forcing a wrong number. Documented N/A, not faked (R5), consistent with the Phase-3
reconstruction axis already being SAE-only.

## Alternatives considered
- **Minimal duck-typed adapter object**, rejected: more bespoke code to keep correct (encode/TopK/cfg),
  no upside over the real class once E4 showed `TopKSAE` accepts our weights cleanly.
- **Retrain the SAE through sae_lens**, rejected: wasteful + changes the artifact under test; the whole
  point is to evaluate the *existing* sparsify coder.
- **Force the transcoder through sparse_probing**, rejected (R3/R5): would produce a meaningless number.

## Consequences
- (+) The deferred SAEBench-on-custom-SAE item is closed with a real, comparable number; the adapter
  (`_sparsify_to_topk_sae`) is reusable for other SAEBench evals on sparsify coders.
- (+) A cheap pre-flight (`verify_saebench_adapter`) proves load + k-enforcement + SAEBench acceptance
  + ENCODE-FIDELITY before paying for the full run (it confirmed k=64 exactly, all 4 weights loaded,
  SAEBench accepts, and, post-correction, that the adapter's encode reproduces sparsify's `coder.encode`
  exactly: same TopK indices, max abs diff 6e-6/7.6e-5, cosine 1.0; the buggy False variant fails this).
  The fidelity check is the real adapter-correctness evidence and is persisted to
  `saebench_adapter_verify.json`.
- (-) Honest result (CORRECTED, encode-verified): the budget-trained custom SAE scores `sae_top_1`
  **0.670** (the buggy v1 reported 0.667), **below** Gemma Scope's 0.767 *and* below its own residual
  baseline (0.688), i.e. on this single-dataset top-1 probe the budget SAE's best feature does **not**
  beat the raw residual, the opposite of repro-003. This is the expected consequence of the ~10M-token
  budget (recon VE 0.51 vs 0.80), reported as-is (R4), not hidden.
- (-) Minor: a few of the budget SAE's decoder rows drift up to ~0.07 off unit-norm (mean norm ≈ 1.004),
  so `check_decoder_norms` warns; it does not raise and the eval proceeds. Noted for honesty.
- One L4 run (~$0.5) + cheap E4 probes/pre-flight (~$0.2). Changing the eval dataset/metric is Gate 4.

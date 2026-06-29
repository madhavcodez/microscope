# PROGRESS

## Current phase
**PHASES 1-6 COMPLETE (2026-06-22); Control-B steering recalibrated + 7B scorer head-to-head RESOLVED 2026-06-23;
SAEBench-custom-SAE adapter b_dec bug FIXED + re-run (encode-verified) 2026-06-23; MULTI-LAYER circuit follow-up DONE 2026-06-23;
SOLIDIFICATION (multi-concept n=5 + leak-free attribution + permutation/paired stats + paper-grade 8-dataset SAEBench) DONE 2026-06-28 (ADR-0009).**
SOLIDIFICATION verdicts (2026-06-28, R4): the sparse SAE-feature circuit REPLICATES across all 5 profession concepts
under leak-free (train-only) attribution + a permutation null (conclusive; min holdout faithfulness 0.873 at K=10; the
leak fix lowers but does not erase the effect, so the ~0.97 single-concept headline SURVIVES); the real>random control
PARTIALLY replicates (4/5 concepts' CIs exclude 0 + survive Holm, all 5 positive, sign-test p=0.0625 borderline, the
original 21/19 the lone non-significant one); paper-grade SAEBench sparse-probing REPRODUCED at 8-dataset x k{1,2,5}
(SAE top-1 0.772 vs 0.679 baseline, +0.094 mean, SAE wins 7/8, ag_news the exception). New fns circuit_multi_eval /
probing_multi_eval / saebench_sparse_probing_paper (originals untouched, D1); CPU analysis scripts/aggregate_controls.py.
CORE PROJECT DONE: reproduce -> train -> head-to-head -> controls -> circuit -> write-up. Spend ≈ $12.2 of $30
(7B A100 unit ≈ $0.6; adapter-fix verify+eval ≈ $0.2; multi-layer circuit ≈ $0.35). FOUR conclusive results now: THREE scorer-independent
(randomized-model control + single-layer sparse feature circuit + multi-layer cross-layer feature-set circuit)
PLUS the novel SAE-vs-transcoder head-to-head, which RESOLVED once a strong-enough
scorer was used, inconclusive at the 3B scorer (both CIs incl 0) but the **skip-transcoder WINS at the 7B
scorer** on both detection (Δ+0.053 CI[+0.016,+0.089]) and fuzzing (Δ+0.059 CI[+0.019,+0.097]); the
recurring near-chance was a SCORER artifact (3B), not a coder limit (ai-g2-sae-7b/tc-7b). Multi-layer circuit
(circuit-multilayer, ADR-0008): ≈9 Gemma Scope features over L5/12/19 recover 97% of the 0.944 ceiling and beat
a random cross-layer control at every K (CI excl 0); concept accumulates by mid-depth (L5->L12, L19 adds +0.000);
scoped honestly as a feature-SET circuit + build-up, NOT a causal edge graph. Repo clean: ruff
+ 170 tests pass, pip install -e . + CLI work. REPORT.md is the finding (Phases 1-5 + abstract); README refreshed.

- **Phase 2 DONE:** sparsify wrapper (SAE=transcode/skip F/F, transcoder T/T, shared width/k) + Pythia smokes
  + both Gemma-2-2B custom coders trained @ L12 (width 16384, k=64, ~10M tokens, bf16): train-g2-sae,
  train-g2-tc, saved on the artifacts Volume (identical recipe => fair head-to-head). Gemma OOM (fp32) fixed bf16+batch8.
- **Phase 3 DONE (head-to-head = INCONCLUSIVE, pre-registered):** reconstruction SAE VE=0.514 (CI [.507,.519]);
  transcoder own-objective recon NOT cleanly isolable externally (limitation, documented, sparsify transcode
  hooks; ForwardOutput.fvu is input-recon inflated by skip). Auto-interp (delphi, custom dicts; fixed cpu->cuda
  + fp32->bf16 loader bugs): SAE det 0.540/fuzz 0.523 (n=58) vs TC det 0.539/fuzz 0.546 (n=61); both Δ bootstrap
  CIs include 0 => no significant difference (scorer-limited, matches repro-004). SAEBench on custom coders:
  the deferred sae_lens adapter is now BUILT + encode-verified (ADR-0007, saebench-custom-sae-v2), custom
  SAE sae_top_1=0.670, BELOW Gemma Scope 0.767 AND below its own residual baseline 0.688 (honest
  budget-training result, R4); transcoder N/A (R3, resid-probing oriented). [v1 reported 0.667 with an
  apply_b_dec_to_input bug, adapter artifact, corrected 2026-06-23; see Coder log + ADR-0007 Correction.]
- **Phase 4 DONE (controls = the differentiator):**
  - Control A randomized-model (ADR-0005): PRIMARY (scorer-independent) = CONCLUSIVE. SAE-feature linear probe
    on bias_in_bios professions: real-model SAE 0.933 vs random-model SAE 0.861; PAIRED gap +0.072, CI95
    [+0.033,+0.117] EXCLUDES 0 => real SAE has model-learned structure beyond token stats. Honest nuance: random
    already 0.861 => most probing signal is TOKEN-level (preserved embeddings); learned adds modest +7pts.
    SECONDARY auto-interp gap = N/A (delphi can't score degenerate random-SAE features).
  - Control B steering (SAE-feat vs difference_of_means): RECALIBRATED 2026-06-23 (ctrl-steer-v2), now
    DISCRIMINATING + honestly INCONCLUSIVE. v1 (ctrl-steer) was degenerate: "My favorite" baseline 0.81
    (ceiling) + coarse coefs [2,4,8] all broke fluency => diff 0.0. v2 = CALIBRATION fix only (same ADR-0005
    metric/concept, NOT a new Gate-4): scan 6 prompts -> neutral 'This person' (baseline 0.562); finer grid
    [0.5,1,2,3,4]. Both directions steer well within fluency at coef 0.5: SAE success 0.875 (effect +0.312),
    dom 0.938 (effect +0.375); coefs>=1-2 break fluency (ppl 16->2268). SAE-dom=-0.062 CI95[-0.25,+0.125]
    (incl 0) => dom matches/slightly beats SAE = AxBench expectation, stated plainly (R4).
- **Phase 5 DONE (circuit = CONCLUSIVE, novel; ADR-0006):** single-layer L12 SAE-feature circuit for the
  bias_in_bios profession behavior. Probe-independent attribution (class-mean act diff) -> top-K; faithfulness
  vs random-K control. top-5 features = 0.878 (94% of 0.933 full-dict ceiling) vs random-5 0.583, gap +0.294
  CI[0.206,0.383]; top-10 = 0.906 (97%); ALL K beat random (CI excl 0). Circuit ids [3955,1649,1962,5409,...].
  Caveat: same token-influenced behavior as Control A. (circuit-g2-sae)
- **Phase 6 DONE (write-up):** REPORT.md = full finding (abstract + Phases 1-5 + honest-scope table); README
  refreshed (status, results table, Modal reproduce commands, $30 cap). Repo verified clean+installable.
- Infra fns: recon_eval, auto_interp_custom, probe_coder_fvu, probe_saebench_adapter, probing_eval,
  steer_eval, circuit_eval, train --randomize. ADRs 0005 (controls) + 0006 (circuit) added.

## Possible follow-ups (NOT started, optional, beyond core scope)
- ~~Calibrated Control-B steering sweep~~ DONE 2026-06-23 (ctrl-steer-v2): neutral prompt + finer grid =>
  discriminating, honestly-inconclusive (dom matches/slightly beats SAE, CI incl 0).
- ~~sparsify->sae_lens adapter for SAEBench on the custom coders (Phase-3 SAEBench was SAE-only)~~ DONE
  2026-06-23 (ADR-0007, saebench-custom-sae-v2): `_sparsify_to_topk_sae` wraps the sparsify SAE as a native
  sae_lens.TopKSAE; full sparse_probing ran on the custom SAE => sae_top_1=0.670 (< Gemma Scope 0.767 AND
  < its own residual baseline 0.688; honest budget-training result, R4, encode-verified). Transcoder N/A
  (R3). Adapter verified before the paid run (verify_saebench_adapter: 4/4 weights, k=64 enforced, SAEBench
  accepts, AND encode-fidelity vs sparsify coder.encode, Jaccard 1.0, cosine 1.0; the buggy =False variant
  fails it). [v1 had an apply_b_dec_to_input=False bug => 0.667 artifact; FIXED 2026-06-23.] Remaining:
  scale to the full 8-dataset SAEBench suite.
- ~~Stronger auto-interp scorer (the near-chance bottleneck throughout)~~ RESOLVED 2026-06-23
  (ai-g2-sae-7b / ai-g2-tc-7b). The L4 ATTEMPT (ai-g2-7b-ATTEMPT) was blocked because delphi keeps the
  Gemma base model resident through scoring (only ~16/22 GiB free, 7B can't start). FIX = run the 7B on an
  **A100-40GB** where base (~6 GiB) + 7B (~14.3 GiB) coexist: new `auto_interp_custom_a100` wrapper
  (gpu="A100-40GB", max_memory=0.65) + `autointerp_main` entrypoint (--gpu a100|l4). Both 7B runs started
  cleanly + completed (~8 min each, ~$0.6 total). RESULT: every score rises WELL above 3B near-chance
  (det SAE 0.540->0.607, TC 0.539->0.660; fuzz SAE 0.523->0.631, TC 0.546->0.690) => the near-chance was a
  SCORER artifact, not a coder limit. HEAD-TO-HEAD FLIPS: 3B both CIs incl 0 (inconclusive) -> 7B
  transcoder WINS on BOTH (det TC-SAE +0.053 CI[+0.016,+0.089]; fuzz +0.059 CI[+0.019,+0.097], both excl 0;
  same unpaired diff-of-means bootstrap seed0/10k as ai-g2). Confirms the pre-registered "transcoders beat
  SAEs" direction on the interpretability axis (full Pareto-dominance still open, TC reconstruction not
  isolable). Recompute: scripts/headtohead_autointerp.py.
- ~~Multi-layer (cross-component) circuit via sparse-feature-circuits.~~ DONE 2026-06-23
  (circuit-multilayer, ADR-0008) as a **cross-layer feature-SET circuit + depth build-up** (NOT the
  heavier feature->feature causal edge graph, which stays the follow-up). Uses PRETRAINED Gemma Scope
  SAEs at L5/12/19 (the layers reproduced in repro-002) on the TransformerLens resid_post recipe
  (BOS excluded). Probe-independent attribution per layer -> union of per-layer top-K = circuit; fresh
  probe on circuit features (concat across layers) vs same-size RANDOM cross-layer set vs full ceiling
  (49152 feats), bootstrap CI on the gap (R2/R3). RESULT: a small cross-layer set is FAITHFUL and beats
  random at every K (CI excl 0): K/layer=3 (9 nodes) circuit 0.917 (97% of 0.944 ceiling) vs random
  0.594 gap +0.322 CI[0.239,0.406]; K/layer=5 (15) 0.939 (99%) vs 0.778 gap +0.161 CI[0.094,0.233];
  K/layer=10 (30) 0.950 (101%) vs 0.667 gap +0.283 CI[0.206,0.356]. Build-up (K/layer=5): L5 0.911,
  L5+L12 0.939, L5+L12+L19 0.939 => concept accumulates by mid-depth (L5->L12) and saturates (L19 adds
  +0.000). Caveat: same token-influence as Control A. ~$0.35 GPU (1 E4 probe + 1 import-fail + 1 real
  run, all L4, ~13 min). infra fn: multilayer_circuit_eval + multilayer_circuit_main entrypoint.
- Feature->feature causal cross-layer EDGE graph (attribution patching / sparse-feature-circuits), the
  heavier version; this unit built the feature-set + build-up, not the edge graph (R4).

## Phase 3 pre-registration (R3, COMMITTED 2026-06-22 BEFORE any eval run; do not change post-hoc)
- **Coders compared:** train-g2-sae vs train-g2-tc (Gemma-2-2B L12, width 16384, k=64, identical recipe).
- **Feature sample:** 100 latents, drawn uniformly without replacement from range(16384) via
  `numpy.random.default_rng(0).choice(16384, 100, replace=False)`. The SAME 100 indices are used for
  BOTH coders (paired comparison). 100 ≤ 500 cap (C3); enough for a bootstrap CI.
- **Auto-interp metrics (delphi, LOCAL Qwen2.5-3B-Instruct scorer, no API):** detection accuracy +
  fuzzing accuracy, aggregate over the 100 latents. (Intruder = optional stretch only if it wires
  cleanly + budget allows; not committed.) Reported with bootstrap 95% CIs.
- **Reconstruction metric:** each coder's FVU / variance-explained on ITS OWN objective, SAE on
  resid_post@L12, transcoder on MLP-out@L12 (from MLP-in). Labeled as each-on-own-target (they are not
  the same target; this is the Transcoders-Beat-SAEs framing, not a like-for-like resid comparison).
- **SAEBench:** sparse_probing (validated in repro-003). absorption ONLY if it runs cleanly on the
  custom coder. NOTE: a SAEBench eval may be SAE-only; if it cannot evaluate the transcoder, that metric
  is N/A for the transcoder (documented, not faked) and the head-to-head for it leans on auto-interp+recon.
- **Hypothesis (reference, to CONFIRM or REFUTE, NOT assumed):** the skip-transcoder Pareto-dominates
  the SAE on the interpretability-vs-reconstruction frontier (Transcoders-Beat-SAEs).
- **Significance:** every reported SAE-vs-transcoder delta gets a bootstrap 95% CI over the 100-latent
  sample; a difference is only called real if the CI excludes zero (recall repro-004's 0.544 vs 0.529
  was within noise).
- **Honest expectation (stated up front):** auto-interp absolute scores will likely be near-chance
  (scorer-limited, repro-004), so the interpretability axis may be INCONCLUSIVE. The salvageable signals
  are the reconstruction axis and any relative ordering that survives the CI test.
Phase 1 COMPLETE earlier (R1 gate: repro-001/002 recon, repro-003 SAEBench, repro-004 auto-interp;
REPORT.md + PHASE1_RETROSPECTIVE.md).

## Done
- Repo created at the repository root; `git init`; local author `madhavcodez
  <madhavcbusiness@gmail.com>`; gpgsign off. GitHub: https://github.com/madhavcodez/microscope
  (private), `main` pushed.
- All project docs (docs/RULES.md, PROGRESS.md, EXPERIMENTS.md, REPORT.md, adr/0000),
  ADR-0001 (model+tooling), ADR-0002 (Python 3.11 pin + GPU-host gate), SETUP_GPU.md runbook.
- pyproject.toml (base installs clean on CPU; ML stack in [gpu] extra), README, LICENSE,
  .gitattributes (LF for demo.sh), example configs, scripts/demo.sh.
- src/microscope: config.py (seeds/determinism/config-hash/RunRecord+EXPERIMENTS logging), Typer
  cli.py (7 stages), stage contracts as honest E4 stubs, real difference-of-means baseline.
- FOUNDATION UNIT (tested + integrity-checked + merged):
  74 tests, 95% coverage, ruff clean, ruff-format clean, mypy clean. Verified: pip install -e .,
  microscope info/--help, determinism, stable config hash.

## In progress
- (nothing, PAUSED after Phase 1 per user instruction)

## Blocked / needs human
- **GATE #1, RESOLVED.** Platform = **Modal** (existing creds + hf-token secret + credits). GPU =
  L4 24GB (~$0.80/hr, per-second billed → no idle burn). HARD cap **$30**; tightened run-gate
  (~$5/90min). Local scorer. Gemma-2-2B license accepted. ADR-0003. Spend ≈ $3.
- (nothing currently blocking)

## Next (Phase 2, when resuming; NOT started)
1. **First answer the open question (PHASE1_RETROSPECTIVE §4.2):** does dictionary_learning 0.1.0
   support a skip-transcoder? Introspect on Modal before committing to the SAE+transcoder deliverable.
2. Phase 2: train a custom SAE (+ transcoder if supported). SMOKE on Pythia-70M first (cheap), then a
   single solid Gemma-2-2B config. Log cost_est; stay ≤ $30 (~$27 remains).
3. Optional polish to fully close Phase 1 to paper-grade: scale SAEBench to 8 datasets × k{1,2,5}
   (repro-003 was a single-dataset smoke); larger auto-interp feature sample + stronger scorer if the
   absolute auto-interp number matters (it is currently labeled inconclusive by design).
4. Then Phases 3-6 through the implement->test->integrity-check loop, logging cost_est each run (≤$30).

## Current task spec
- (filled in per unit of work)

## Test log
- 2026-06-21: Wrote the CPU-verifiable pytest suite under tests/ (8 files, 71 tests,
  AAA style). Covers config.set_seed determinism (numpy + python random reproduce on same seed,
  diverge on different), config_hash (stable / key-order-insensitive / value-sensitive / 12-char
  hex), load_config (real config / FileNotFoundError / ValueError on non-mapping+empty+scalar YAML),
  git_commit + hardware_info (keys platform/python/cpu_count/gpu), RunRecord.as_row_cells (18 cells,
  pipe + newline escaping), append_experiment_row (header creation + multi-row append + parent mkdir,
  via tmp_path), steering.difference_of_means (known input, unit norm, all ValueError guards incl.
  zero-vector), _pending.pending (GpuImplementationPending subclass of NotImplementedError, names
  stage), and CLI (info exit 0; all GPU-bound commands surface the E4 gate as exit code 2). GPU stubs
  asserted to raise only (per RULES.md E4), not implemented.
- Command: `python -m pytest --cov=microscope --cov-report=term-missing`
- Result: 71 passed, 0 failed. Coverage: microscope total 94% (config.py 88%, cli.py 97%,
  steering/baselines.py 100%, _pending.py 100%, all GPU-stub modules 100%). Remaining misses are
  environment-gated branches that cannot fire on this CPU box (numpy/torch ImportError fallbacks,
  CUDA-present branch in hardware_info, git OSError catch) plus cli.py's green-success path and
  __main__ guard. No source bugs found; the CPU-verifiable code behaves exactly as documented.

## Quality-check log
- 2026-06-21 (research-integrity + python code-quality review):
  Integrity verdict APPROVE; code-quality verdict REQUEST-CHANGES (lint/type-hint only). Findings
  and resolution:
  - [HIGH, integrity] C3 ≤500-feature cap was NOT enforced on the randomized-model control path
    (unbounded-spend back door once implemented). FIXED: eval/controls.randomized_model_control now
    enforces MAX_FEATURES_PER_RUN with a ValueError + regression test added.
  - [MEDIUM, integrity] R1 reproduction-first is documented/sequenced but not mechanically gated in
    code. DEFERRED with explicit note (accepted in review): once Phase-1 stages are implemented on
    the host, add a check that a 'reproduced' row exists in EXPERIMENTS.md before `train` runs.
    Tracked here so it is not forgotten.
  - [code-quality] B008 (Typer idiom), UP035, I001, E501 (37 nits): FIXED, collections.abc imports,
    `_run_stage` typed, per-file B008 ignore for cli.py, `ruff check --fix` + `ruff format`, banner
    comments stripped, prose wrapped. ruff now passes; mypy clean.
  - [coverage] project_root cwd-fallback + git_commit OSError branch were untested (CPU-reachable):
    tests added; config.py 88%->91%, total 95%.
  - Re-verified post-fix: 74 passed, 95% coverage, ruff/format/mypy all clean.

## Work log
- 2026-06-22: Implemented the previously-DEFERRED R1 mechanical gate (closes the MEDIUM
  integrity item in the quality-check log above). Pure-CPU, stdlib-only, no new deps.
  - config.py: added `reproduction_logged(path: Path | None = None) -> bool`, parses
    EXPERIMENTS.md, finds the `| run_id` header, locates the column whose header contains 'label',
    scans data rows (skips the `|---` separator), returns True iff any label cell contains
    'reproduced' (case-insensitive, substring). Missing file / no data rows -> False (fail closed).
    Helper `_split_markdown_row` tolerates leading/trailing pipes + whitespace. Also added
    `class ReproductionGateError(RuntimeError)` (R1 gate; distinct from _pending GPU gates).
  - cli.py: `train` now calls `reproduction_logged()` BEFORE `_prepare`/`_run_stage`; if False it
    prints a red message naming RULES.md R1 and `raise typer.Exit(code=3)`. Exit code 3 = R1 gate
    (distinct from code 2 = GPU/E4 gate). Imported `reproduction_logged` from `.config`.
  - Gate currently PASSES (EXPERIMENTS.md has reproduced rows repro-001/002/003), verified
    `reproduction_logged()` returns True. Verified via Typer CliRunner: train with an empty/no-repro
    table -> exit 3 + R1 message + `_prepare` NOT reached; train with real table -> R1 passes
    through to the existing exit-2 GPU/E4 gate (unregressed).
  - Checks: `python -c "import microscope.cli"` OK; full pytest 94 passed; ruff check + ruff format
    + mypy all clean on config.py + cli.py. NOTE: unit tests still to add. Coverage
    of the new branches is currently exercised only by an ad-hoc CliRunner check, not the committed
    suite; regression tests still to add (see handoff). Did not commit (per instruction).
- 2026-06-22: Phase 2 unit 1, sparsify training wrapper + training YAMLs (ADR-0004). No
  training run here (that is unit 2 on Modal). Heavy imports kept lazy; package still imports on CPU.
  - src/microscope/saes/train.py REWRITTEN (was a `pending` stub):
    * `coder_config_dict(config, kind) -> dict[str, Any]`, PURE (no sparsify/torch import); the
      TESTABLE CORE. Maps RunConfig (+extra='allow' keys width/k/activation/batch_size/lr/save_dir/
      run_name) to a flat sparsify-settings dict. THE KEY INVARIANT: SAE => transcode=False &
      skip_connection=False; transcoder => both True; width(=>num_latents) and k(=>TopK L0) come from
      the SAME config so SAE vs skip-transcoder is a fair head-to-head. Validates: kind in
      {sae,transcoder}, width & k present + positive int (coerces "64"/64.0), activation in
      {topk,groupmax}, layer not None, lr/batch_size sane, all raise ValueError (fail fast).
    * `train_coder(config, kind) -> dict[str, Any]`, GPU-only. Validates via coder_config_dict
      FIRST (fail fast on CPU before the GPU import), then lazy `import sparsify`; on ImportError
      raises `GpuStackUnavailable` naming the Modal [gpu] image (mirrors
      activations.harvest_resid_activations, NOT GpuImplementationPending). Builds SaeConfig +
      TrainConfig from the flat dict, loads HF model (lazy `from transformers import AutoModel`) +
      dataset (lazy `from datasets import load_dataset`, streaming), constructs
      `Trainer(cfg, dataset, model)`, then delegates to helper `_launch_train_and_save`. The ONE
      unresolved E4 item (sparsify launch method `.fit()` vs `.train()`, per ADR-0004 "confirmed on
      Modal in unit 2") is isolated in that helper as a clearly-commented TODO + a GpuStackUnavailable
      raise; the post-launch save (`Sae.save_to_disk`)/return-metrics shape is documented there for
      unit 2. Type hints + docstrings (E5).
  - experiments/configs/: NEW `train_pythia70m_smoke.yaml` (Pythia-70m-deduped, layer 3, width 4096,
    k 32, n_tokens 500k) and `train_gemma2_2b_l12.yaml` (gemma-2-2b, layer 12, width 16384, k 64,
    n_tokens 10M with a comment that the exact budget is cost-gated in unit 3). One YAML per model;
    `--kind` flips SAE/transcoder (DRY; same width/k = fair). Both point to ADR-0004. (Left the OLD
    pythia70m_smoke.yaml / gemma2_2b_reproduce.yaml untouched, those are the Phase-1 reproduce
    configs.)
  - Checks: both required CPU imports OK (`import microscope.cli`,
    `from microscope.saes.train import coder_config_dict`); ruff + ruff format + mypy CLEAN on
    train.py; ad-hoc CPU runs confirm the invariant (SAE F/F, transcoder T/T, shared width/k), every
    validation path raises ValueError, both YAMLs load + hash + flow through coder_config_dict, and
    train_coder raises GpuStackUnavailable (sparsify absent), and ValueError before the gate on a
    config missing width/k. Did NOT add/modify tests (separate step). Did not commit (per instruction).
  - HANDOFF / tests to update: 5 pre-existing tests that pinned the OLD stub contract (these are
    EXPECTED contract changes, not source bugs, same transition reproduce/harvest already made):
    (1) tests/test_pending.py::test_train_coder_stub_raises_pending, train_coder no longer raises
    GpuImplementationPending; it now raises GpuStackUnavailable (and ValueError first on a config
    without width/k). (2-5) tests/test_cli.py::test_train_with_valid_config_surfaces_gpu_gate and
    tests/test_reproduction_gate.py::{test_train_passes_r1_then_hits_gpu_gate_exit_2_via_real_parser,
    test_train_passes_r1_exit_2_via_function_monkeypatch, test_train_r1_and_gpu_exit_codes_are_distinct}
    all invoke `train` with the OLD pythia70m_smoke.yaml (no `k`) expecting exit 2 (GPU gate); now
    train_coder validates first and raises ValueError(missing k) -> different path. Fix: point those
    train invocations at the NEW train_pythia70m_smoke.yaml (has width+k) so they pass validation and
    reach the exit-2 GPU gate, OR assert the new validation behaviour. The R1 exit-3 tests
    (gate-shut) are unaffected and still pass (R1 fires before train_coder).
- 2026-06-23: RECALIBRATED Control-B steering (`steer_eval` in infra/modal_app.py) so it
  discriminates instead of returning the degenerate baseline-ceiling result. CALIBRATION FIX ONLY -
  same ADR-0005 pre-registered metric (success-rate-under-fluency) + concept (bias_in_bios prof 21 vs
  19, steer->19), so NOT a new Gate-4 decision (stated in docstring + EXPERIMENTS notes).
  - steer_eval edited in place: (1) NEUTRAL-prompt scan over 6 candidates ["I","The","This person",
    "They said","Yesterday","My favorite"], prints each one's unsteered baseline success, picks the
    one closest to 0.5 (new `n_scan=8` param keeps the scan cheap; chosen prompt's baseline is
    re-measured at full n_gen). (2) FINER grid coefs=[0.5,1,2,3,4]xresid_rms (was [2,4,8]). (3) Reports
    success AND ppl at EVERY coef for BOTH directions (full sweep incl coef0), each direction's best
    FLUENCY-PRESERVING coef, the steering EFFECT (success-baseline), and bootstrap CI95 on SAE-minus-dom
    with an explicit R4 honest verdict. E1: logs+sets seed (np/torch/cuda), `seed` param, default 0.
  - RAN on Modal L4 (PYTHONUTF8=1, ~30 min, ~$0.45, 1 GPU run, exit 0). RESULT (ctrl-steer-v2):
    chosen prompt 'This person' baseline_success=0.562 (vs 'My favorite'/'I' = 1.0 ceilings), ppl 9.18,
    fluency cap 13.76. SAE feature best=coef0.5 success 0.875 (effect +0.312); dom best=coef0.5 success
    0.938 (effect +0.375); coefs>=1-2 break fluency (ppl 16.6->2268). SAE-dom=-0.062 CI95[-0.25,+0.125]
    (incl 0) => INCONCLUSIVE but now meaningful: both steer with real headroom; dom matches/slightly
    beats SAE = the AxBench expectation, stated plainly (R4). top_feature=795, resid_rms=104.4.
  - Checks: py_compile OK; ruff on my edited range adds only E501 (file already has 89 pre-existing
    E501 + B905/E702/E731/F401 in unmodified code; infra/modal_app.py is not held to the src/ lint
    bar). No new callers/tests reference steer_eval's signature. Updated EXPERIMENTS.md (annotated old
    ctrl-steer as SUPERSEDED + added ctrl-steer-v2 row), REPORT.md (Control B section + summary table),
    README (steering row), PROGRESS.md. COMMITTED on main.
  - HANDOFF / verification: this is a Modal-GPU control fn (not CPU-unit-testable end-to-end). Verify
    (a) py_compile/import of infra/modal_app.py; (b) the numbers in the docs match output line 54-55 of
    the run (STEER RESULT + BY DIR) and /root/outputs/steering.json on the artifacts volume; (c) the
    metric/concept are unchanged vs ADR-0005 (only prompt + coef grid + reporting changed); (d) the CI
    [-0.25,+0.125] includes 0 => 'inconclusive' label is correct (R4). A re-run with the same seed
    should reproduce (E1), modulo any nondeterminism in CUDA sampling.
- 2026-06-23: STRONGER-SCORER attempt for the auto-interp head-to-head (the near-chance
  bottleneck). Tried a LOCAL Qwen2.5-7B-Instruct on auto_interp_custom for the SAE coder; STOPPED with
  NO RESULT after 2 GPU startup failures (retry cap; ~$0.15, both died at vLLM engine init before any
  scoring). The 3B head-to-head is UNCHANGED and remains the reported Phase-3 result (R4: the
  scorer-strength question is still open, not answered).
  - DIAGNOSIS (not an OOM): delphi caches activations with the Gemma-2-2B base model and then scores
    with a vLLM scorer in ONE process WITHOUT freeing the base model from the GPU, so at scorer startup
    only ~16/22 GiB is free. vLLM's request_memory guard rejects gpu_memory_utilization 0.9
    (ValueError: 19.83 GiB requested > 16.05 GiB free); 0.5 underfits the 7B's ~14.3 GiB weights + KV
    cache + CUDA graphs. No max_memory fraction satisfies both while the base model is resident. The 3B
    (~6 GiB) coexists with it, which is why ai-g2 ran. E4: delphi's GPU lifecycle differs from the
    "base model freed after caching" assumption in the old code comment (now corrected).
  - CODE (infra/modal_app.py auto_interp_custom, backward-compatible, CPU py_compile + ruff clean on my
    lines): (1) added `max_memory: float = 0.5` param (was hard-coded 0.5) so the vLLM budget is tunable
    for the eventual fix; (2) anti-clobber output filename, derive a scorer_tag from scorer_model; the
    default 3B writes the historical autointerp_<tag>.json, a non-default scorer writes
    autointerp_<tag>_<scorer_tag>.json (verified: 3B->autointerp_sae.json, 7B->autointerp_sae_7b.json),
    so a future 7B run will NOT overwrite the 3B results on the volume; (3) logged seed=0 + scorer_tag +
    max_memory into the result dict (E3); (4) corrected the docstring + comment to document the KNOWN
    BLOCKER honestly (R4) instead of the wrong "base model freed" claim. No production scorer switch (still
    local, C1). Updated EXPERIMENTS.md (ai-g2-7b-ATTEMPT row, label inconclusive(no result), no fake
    scores per R5), REPORT.md (Phase-3 "Scorer-strength check" note), PROGRESS.md.
  - HANDOFF / verification: CPU-verifiable parts only. Verify (a) py_compile infra/modal_app.py; (b) the
    filename derivation: scorer_model="Qwen/Qwen2.5-3B-Instruct" -> out_suffix="" (autointerp_<tag>.json);
    any other scorer -> "_<scorer_tag>" (7B -> _7b); (c) the 3B results on the volume are untouched
    (autointerp_sae.json/autointerp_tc.json still present, scorer=Qwen2.5-3B); (d) no fabricated 7B
    scores anywhere in the docs (R5). The 7B run itself is NOT reproducible until the base-model-free fix
    lands, it is a documented Gate (free base model from cuda before scoring, or a >24 GiB GPU).
- 2026-06-23: RESOLVED the stronger-scorer question by running the 7B on an A100-40GB (where the
  resident Gemma base model + 7B vLLM scorer coexist), which the L4 could not fit. This ANSWERS the
  scorer-strength question the ATTEMPT left open, and FLIPS the Phase-3 head-to-head.
  - CODE (infra/modal_app.py): refactored auto_interp_custom's body into a GPU-agnostic helper
    `_auto_interp_impl(...)`; kept `auto_interp_custom` (gpu="L4", 3B scorer, max_memory=0.5) as a thin
    wrapper (backward-compatible, historical ai-g2 path unchanged); added `auto_interp_custom_a100`
    (gpu="A100-40GB", default scorer Qwen2.5-7B, max_memory=0.65) calling the same helper; added an
    `autointerp_main` local_entrypoint (--gpu a100|l4, config-driven, enforces the C3 <=500 cap). Verified
    Modal's A100-40GB GPU string against the installed `modal` 1.4.2 (parse_gpu_config uppercases + sends
    to backend; docs confirm "A100-40GB" = the 40 GiB variant, bare "A100" may auto-upgrade to 80GB so I
    pinned -40GB). `Function.with_options` does NOT exist in 1.4.2 (only Cls), so a second decorated fn is
    the clean config-driven path, not a call-time override. py_compile + ruff clean on my lines (only the
    file's pre-existing E501 etc remain); 170 CPU tests still pass (infra not imported by the package).
  - RAN (E1, seed 0 logged): both coders, scorer Qwen/Qwen2.5-7B-Instruct, max_latents=100, A100-40GB,
    PYTHONUTF8=1. The 7B's 14.29 GiB weights loaded next to the ~6 GiB resident base model and both runs
    completed cleanly (~8 min each, ~$0.6 total GPU, under the unit's $5 cap). Pulled
    autointerp_sae_7b.json / autointerp_tc_7b.json via `modal volume get`; verified the 3B jsons
    (autointerp_sae/tc.json) are UNTOUCHED (no clobber) and the 7B jsons log seed/scorer_tag/max_memory (E3).
  - RESULT (real, R5, no fabrication): SAE det 0.6072/fuzz 0.6309 (n=58); TC det 0.6602/fuzz 0.6895 (n=60).
    Both far above the 3B near-chance => scorer artifact confirmed. Recomputed the head-to-head with
    scripts/headtohead_autointerp.py (unpaired diff-of-means bootstrap, seed 0, 10k resamples, the SAME
    method as ai-g2; validated by reproducing the 3B CIs exactly before adding 7B). 3B: det Δ(TC-SAE)+0.001
    CI[-0.022,+0.022], fuzz +0.023 CI[-0.001,+0.047] (both incl 0). 7B: det +0.053 CI[+0.016,+0.089], fuzz
    +0.059 CI[+0.019,+0.097] (both EXCLUDE 0) => transcoder significantly more interpretable on both.
    VERDICT (R4): 7B raised scores above chance AND changed the conclusion (inconclusive -> transcoder wins);
    confirms the pre-registered Transcoders-Beat-SAEs direction on the interpretability axis (full
    Pareto-dominance still open, TC own-objective reconstruction not externally isolable).
  - DOCS: EXPERIMENTS.md (ai-g2-sae-7b + ai-g2-tc-7b rows with real scores; ai-g2-7b-ATTEMPT annotated
    RESOLVED), REPORT.md (Phase-3 table + verdict + scorer-strength note + abstract + summary table all
    updated to scorer-dependent/transcoder-wins), README (results table + bottom line + Phase-3 commands),
    PROGRESS (this). New scripts/headtohead_autointerp.py (CPU, numpy-only). .gitignore: artifacts_pull/.
    COMMITTED on main.
  - HANDOFF / verification: (a) py_compile infra/modal_app.py + the 3 new fns exist (impl + 2 wrappers + entrypoint);
    (b) `python scripts/headtohead_autointerp.py --dir artifacts_pull` reproduces 3B CIs exactly (matches the
    ai-g2 row) AND the 7B CIs both exclude 0; (c) the numbers in EXPERIMENTS/REPORT/README match the JSONs on
    the volume (modal volume ls microscope-artifacts | autointerp_*_7b.json) and the FINAL AUTO-INTERP RESULT
    lines; (d) 3B jsons untouched (no clobber); (e) no fabricated numbers (R5); (f) ruff adds no NEW non-E501
    issues vs the pre-existing baseline; 170 tests still pass. A 7B re-run with the same seed should
    reproduce modulo CUDA-sampling nondeterminism (E1).
- 2026-06-23: Built the sparsify->sae_lens adapter + ran the FULL SAEBench sparse_probing on the
  CUSTOM SAE, the last deferred Phase-3 item (ADR-0007). E4-first: ran probe_saebench_adapter (already in
  the file) + added probe_saebench_adapter2/_adapter3/_datasets to pin the exact API on Modal before any
  code (sae_lens 6.44.3: TopKSAE + TopKSAEConfig + SAEMetadata at sae_lens.saes.sae; cfg.hook_name resolves
  from metadata; load_and_format_sae for a custom object only check_decoder_norms[warns]+_standardize_sae_cfg).
  - CODE (infra/modal_app.py, all my lines ruff-CLEAN incl. no new E501): `_sparsify_to_topk_sae(coder_dir,
    layer, device, dtype='float32')` helper, loads the sparsify SparseCoder, raises on transcode/skip
    (R3), builds a real sae_lens.TopKSAE (W_enc=encoder.weight.T, b_enc=encoder.bias, W_dec/b_dec copied,
    k=64, apply_b_dec_to_input=False to match sparsify, metadata=hook_name/layer/model). `verify_saebench_adapter`
    (cheap GPU pre-flight, E4) + `saebench_sparse_probing_custom` (full eval, same config as repro-003,
    seed 42 set+logged E1). Plus 3 CPU E4 probes (adapter2/3/datasets).
  - RAN (Modal, PYTHONUTF8=1): (1) verify_saebench_adapter L4 ~1 min, 4/4 weights load, k=64 enforced
    exactly, SAEBench load_and_format_sae ACCEPTS the object, cfg.hook_name/layer/model all resolve. (2)
    First full run KeyError'd on bare 'LabHC/bias_in_bios' (SAEBench build keys it '..._class_set1', E4
    probe_saebench_datasets); fixed the dataset key (matches repro-003's actual key). (3) Re-ran full eval
    L4 ~6 min, exit 0. ~3 GPU iterations total (verify + 2 eval), ~$0.7 GPU all-in, under the $4 cap.
  - RESULT (real, R5, no fabrication): sae_top_1=0.6668; residual(llm) baseline top_1=0.6876; full-feat
    sae=0.9532/llm=0.9648. HONEST (R4): budget SAE 0.667 < Gemma Scope 0.767 AND < its own residual baseline
    0.688, on this single-dataset top-1 probe the budget SAE's best feature does NOT beat the raw residual
    (opposite of repro-003). Expected from the ~10M-token budget (recon VE 0.51). Baseline 0.688 ==
    repro-003's 0.688 (SAE-independent) => eval sound + apples-to-apples. Transcoder N/A (R3). Decoder-norm
    note: mean row norm ~1.004, a few rows ~0.07 off => check_decoder_norms warns (does not raise).
  - DOCS: ADR-0007 (new), EXPERIMENTS.md (saebench-custom-sae row), REPORT.md (Phase-3 table row + new
    SAEBench subsection + scope-table row + follow-ups status), PROGRESS (this + Phase-3 bullet + follow-up).
    result json: /root/outputs/saebench_custom_sae.json on the microscope-artifacts volume. COMMITTED on main.
  - HANDOFF / verification: (a) py_compile infra/modal_app.py; the 5 new fns exist (_sparsify_to_topk_sae helper,
    verify_saebench_adapter, saebench_sparse_probing_custom, + CPU probes adapter2/3/datasets); (b) ruff
    adds no NEW non-E501 issue on my lines (>=line ~1907) vs baseline; 170 CPU tests still pass (infra not
    imported by the package); (c) the numbers in EXPERIMENTS/REPORT/ADR-0007 match saebench_custom_sae.json
    on the volume (`modal volume get microscope-artifacts saebench_custom_sae.json -`) AND the SAEBENCH
    CUSTOM-SAE RESULT stdout line; (d) the residual baseline (0.688) matches repro-003's (SAE-independent
    sanity check); (e) no fabricated numbers (R5); the 0.667<0.688 below-baseline result is reported as-is.
    A re-run with seed 42 + same config should reproduce (modulo minor GPU nondeterminism, E1).
- 2026-06-23: FIXED a CRITICAL adapter encode bug found in review + RE-RAN. ADR-0007 v1 set
  `apply_b_dec_to_input=False` in `_sparsify_to_topk_sae` on the premise that sparsify's TopK encode does
  not subtract b_dec. That premise was FACTUALLY WRONG, so v1's sae_top_1=0.667 was an adapter artifact.
  - E4 FIRST (probe_sparsify_encode, NEW): read the INSTALLED sparsify `SparseCoder.encode` verbatim on
    Modal => `if not self.cfg.transcode: x = x - self.b_dec` then fused_encoder. Coder under test has
    cfg.transcode=False (b_dec norm ≈ 90.7), so sparsify's true encode is `(x-b_dec)@Wencᵀ+b_enc` while the
    buggy adapter did `x@Wencᵀ+b_enc`. Confirmed `- self.b_dec` present in source.
  - FIX (infra/modal_app.py): (1) `_sparsify_to_topk_sae` now sets `apply_b_dec_to_input=True` (b_dec is
    copied into the SAE, so sae_lens applies the same shift) + corrected comment. (2) NEW encode-fidelity
    check in `verify_saebench_adapter` (helpers `_sparsify_dense_acts`/`_encode_fidelity`/`_fidelity_with_b_dec_flag`):
    runs BOTH the real sparsify coder.encode and the adapter.encode on the same random AND real-resid
    batches, asserts identical active TopK indices (per-row Jaccard) + values within tol; HARD-FAILS on
    random mismatch; ALSO computes the apply_b_dec=False variant to document the contrast; persists the
    verify dict to /root/outputs/saebench_adapter_verify.json (H2). (3) result-dict dataset label fixed to
    "LabHC/bias_in_bios_class_set1" (M1; was dropping the suffix).
  - RAN (Modal L4, PYTHONUTF8=1, seed logged E1): verify_saebench_adapter => encode-fidelity PASSES with
    True (random: Jaccard 1.0 all 8 rows, max abs diff 6e-6, cosine 1.0; real-resid: Jaccard 1.0 all 16
    rows, max abs diff 7.6e-5, cosine 1.0) and the False variant FAILS the SAME check (Jaccard ≈ 0.07,
    cosine 0.139), this is the check that would have caught the bug. Then saebench_sparse_probing_custom
    => CORRECTED sae_top_1=0.670 (vs buggy 0.667), baseline 0.6876 UNCHANGED (SAE-independent, == repro-003),
    full-feat 0.9496. ~$0.2 GPU total (1 verify + 1 eval), under the $3 cap. The +0.003 move shows a top-1
    best-single-feature probe is robust to which near-equivalent budget latents win; the result now rests on
    a verified-correct encode. CONCLUSION UNCHANGED + now REAL: 0.670 < baseline 0.688 < Gemma Scope 0.767,
    the honest negative STANDS (encode-verified, R4).
  - DOCS: ADR-0007 (Correction section + premise corrected + fidelity = real correctness evidence),
    EXPERIMENTS.md (v1 row annotated BUGGY/superseded + new saebench-custom-sae-v2 row), REPORT.md (table
    rows + SAEBench subsection + adapter-correctness paragraph + status), README, PROGRESS (this). COMMITTED on main.
  - HANDOFF / verification: (a) py_compile infra/modal_app.py; the new helpers + probe_sparsify_encode exist and
    are NOT @app.function-decorated (helpers must stay plain, a misplaced decorator caused a first-run
    'Function not callable', since fixed); (b) ruff adds no NEW non-E501 issue on my lines vs baseline;
    170 CPU tests still pass (infra not imported by the package); (c) numbers in EXPERIMENTS/REPORT/README/
    ADR-0007 match saebench_custom_sae.json (sae_top_1 0.670) AND saebench_adapter_verify.json
    (encode_fidelity_PASS True) on the volume; (d) the False-variant contrast in the verify json FAILS
    (proves the check is real); (e) no fabricated numbers (R5). Re-run with seed 42 reproduces modulo GPU
    nondeterminism (E1).
- 2026-06-23: Built the MULTI-LAYER circuit, the deferred extension of the single-layer
  circuit-g2-sae (ADR-0008, NEW). Pre-registered the method in ADR-0008 BEFORE running (R3).
  - SCOPE (R4, stated up front): a cross-layer feature-SET circuit + depth build-up, NOT a
    feature->feature causal EDGE graph (the heavier attribution-patching / sparse-feature-circuits
    version stays a follow-up). Same bias_in_bios prof 21v19 contrast as Phase 5 / Control A (continuity).
  - E4 FIRST (probe_gemma_scope_multilayer, NEW, ~3 min L4): confirmed the PRETRAINED Gemma Scope SAEs
    load at L5/12/19 (width_16k/canonical, d_sae=16384 each) AND that `sae_lens SAE.encode` returns a
    DENSE (n_tokens, 16384) tensor (NOT the sparsify TopK tuple) on the TransformerLens resid_post recipe
    (BOS excluded). This is why the fn uses the reproduce_recon recipe, NOT circuit_eval's sparsify path
    (raw HF acts gave VE -4.5 for Gemma Scope, ADR-0003).
  - CODE (infra/modal_app.py): `multilayer_circuit_eval` (image=pkg_image, needs BOTH transformer_lens
    [in base] AND sklearn [in the full interp image; base lacks it, first run crashed at sklearn import,
    fixed by switching base_image->pkg_image]) + `multilayer_circuit_main` local entrypoint. Per layer:
    run_with_cache(resid_post, stop_at_layer=L+1), drop BOS, sae.encode (dense), mean-pool per example.
    Attribution = |mean_act(c1)-mean_act(c0)| per layer (probe-independent); circuit = union of per-layer
    top-K; faithfulness = fresh logistic probe on circuit cols (concat across layers) vs same-size RANDOM
    cross-layer set vs full 49152-feat ceiling, bootstrap CI on the gap; single fixed split shared by all
    probes (paired); build-up curve L5/L5+L12/L5+L12+L19. Seeds set+logged (E1, seed=0). py_compile OK;
    ruff adds no NEW non-E501 issue on my lines (added strict=False to my one zip to avoid a B905).
  - RAN (Modal L4, PYTHONUTF8=1, seed 0, ~13 min, ~$0.2; total unit ~$0.35 incl E4 probe + 1 import-fail,
    under the $3 cap; 3 GPU iterations = probe + fail + real run, the budget-stated limit). RESULT (real,
    R5): ceiling(49152)=0.9444; K/layer=3 (9 nodes) circuit=0.9167 (97.1%) vs random=0.5944 gap+0.322
    CI[0.239,0.406]; K/layer=5 (15) 0.9389 (99.4%) vs 0.7778 gap+0.161 CI[0.094,0.233]; K/layer=10 (30)
    0.9500 (100.6%) vs 0.6667 gap+0.283 CI[0.206,0.356]; ALL K beat random (CI excl 0). Build-up
    (K/layer=5): L5=0.9111, L5+L12=0.9389, L5+L12+L19=0.9389. VERDICT (R4): sparse multi-layer
    (cross-layer feature-set) circuit, faithful + beats the random cross-layer control (novel, with
    control); concept accumulates by mid-depth (built L5->L12, L19 adds +0.000). Top features per layer
    (K/layer=5): L5 [12872,5411,14908,28,807], L12 [6810,23,5364,1041,10603], L19 [4346,10992,12025,7663,14180].
    result: /root/outputs/circuit_multilayer.json on the microscope-artifacts volume.
  - DOCS: ADR-0008 (new), EXPERIMENTS.md (circuit-multilayer row), REPORT.md (Phase-5 multi-layer
    subsection + scope-table row + summary sentence + status follow-up update), README (results row +
    Phase-5 command), PROGRESS (this + follow-up marked done). COMMITTED on main.
  - HANDOFF / verification: (a) py_compile infra/modal_app.py; the 2 new fns exist (multilayer_circuit_eval +
    probe_gemma_scope_multilayer) + the multilayer_circuit_main entrypoint; (b) ruff adds no NEW non-E501
    issue on my lines (>=~2376) vs baseline (114 E501 are pre-existing/not held to the src bar); 170 CPU
    tests still pass (infra not imported by the package); (c) the numbers in EXPERIMENTS/REPORT/README/
    ADR-0008 match circuit_multilayer.json on the volume (`modal volume get microscope-artifacts
    circuit_multilayer.json <path>`, pull to a FILE, not '-': stdout '-' prepends Modal log lines and
    corrupts the JSON) AND the FINAL MULTILAYER CIRCUIT RESULT stdout line; (d) the scope label is
    'feature-SET circuit + build-up', NOT a causal edge graph (R4); (e) no fabricated numbers (R5). A
    re-run with seed 0 reproduces modulo minor GPU nondeterminism (E1). NOTE: multilayer_circuit_eval uses
    pkg_image (base_image lacks sklearn); probe_gemma_scope_multilayer uses base_image (only needs TL+sae_lens).
- 2026-06-28: SOLIDIFICATION follow-up, multi-concept replication + leak-free attribution +
  upgraded statistics (ADR-0008 -> ADR-0009, NEW, pre-registered BEFORE the run, R3). Extends the Phase-5
  single-layer circuit (circuit-g2-sae, ADR-0006) and the Phase-4 randomized-model control. D1
  conservative/reversible: ALL changes land in NEW backward-compatible functions, the original
  circuit_eval / probing_eval and their logged artifacts are UNTOUCHED so the old-vs-new delta stays
  checkable.
  - WHY: every conclusive novel claim so far rested on ONE concept (bias_in_bios prof 21 vs 19), so
    generalization was unproven; AND circuit_eval had an ASYMMETRIC test-leak (attribution
    |mean_act(c1)-mean_act(c0)| computed over the FULL labeled set incl. the test split before top-K
    selection => only the circuit's selection saw test-label-informed attribution, the random control drew
    blind, so the leak could inflate the circuit-vs-random GAP the claim turns on).
  - CODE (infra/modal_app.py, 3 new fns + 3 entrypoints, backward-compatible): `circuit_multi_eval`
    (+ `circuit_multi_main`), `probing_multi_eval` (+ `probing_multi_main`), `saebench_sparse_probing_paper`
    (+ `saebench_paper_main`). Helper `_profession_contrasts` derives a DETERMINISTIC set of distinct binary
    contrasts from the top-K most-frequent professions; first pair = original {21,19} (continuity), then
    {21,2},{19,2},{2,18},{21,11}. (a) replicate circuit + control across all 5 (n=1 -> n=5, every contrast
    reported, none dropped, R3); (b) attribution on the TRAIN split ONLY (held-out), top-K from train-only
    attribution, probe fit+scored on the held-out test split (full-set attribution computed alongside ONLY
    to quantify the leak); (c) R=100 random-K permutation null + genuinely paired bootstrap (10000 iters,
    ONE shared resample index per iter, fixing the prior independent-index "paired" bootstrap) + Holm-
    Bonferroni across concepts; saebench_sparse_probing_paper widens SAEBench sparse_probing to the
    8-dataset x k{1,2,5} paper headline. CPU analysis: scripts/aggregate_controls.py (numpy-only; computes,
    does not fabricate; writes solidification_summary.json). Seeds set+logged (E1, seed 0; rng(0),
    sklearn random_state=0; one fixed split per concept reused across arms = paired). py_compile OK.
  - RAN (Modal L4, seed 0, n_per_class 250, 2026-06-28; custom L12 SAE already on the artifacts volume):
    circuit_multi_eval + probing_multi_eval (real + randomize) + saebench_sparse_probing_paper. Pulled
    circuit_multi.json / probing_multi_real.json / probing_multi_random.json / saebench_paper.json; ran
    scripts/aggregate_controls.py -> solidification_summary.json. ~$0.3 (circuit) + ~$0.5 (control,
    real+random) + ~$0.7 (saebench-8) est GPU, under cap.
  - RESULT (real, R5, no fabrication):
    * CIRCUIT (leak-free, 5 concepts). Primary K=10 holdout/fullset/gap CI95/perm-p/Holm: 21/19
      0.993/1.022 +0.2553[0.2145,0.2927] 0.0099 survive; 21/2 0.957/0.957 +0.3100[0.2633,0.3528] survive;
      19/2 0.889/0.896 +0.2013[0.1539,0.2458] survive; 2/18 0.924/0.944 +0.2799[0.2321,0.3230] survive;
      21/11 0.873/0.880 +0.2389[0.1803,0.2945] survive. Cross-K aggregate mean faithfulness K=5/10/20/50 =
      0.885/0.927/0.971/0.983 (min 0.845/0.873/0.944/0.944); ALL 5 of 5 beat random + survive Holm at every
      K. leak_delta=fullset-holdout <=+0.029 (21/19 fullset 1.022>1.0 = over-selection rounding artifact of
      leaked attribution; holdout 0.993 honest) => the leak fix lowers faithfulness only marginally.
    * CONTROL real vs random SAE (paired bootstrap, 5 concepts). real/random/gap CI95/p/Holm: 21/19
      0.9267/0.9000 +0.0267[-0.0133,+0.0667] 0.1302 NO; 21/2 0.9400/0.8867 +0.0533[+0.0067,+0.1067] survive;
      19/2 0.9600/0.9133 +0.0467[+0.0133,+0.0800] survive; 2/18 0.9600/0.9067 +0.0533[+0.0067,+0.1000]
      survive; 21/11 0.9467/0.8333 +0.1134[+0.0600,+0.1733] survive. mean gap +0.0587; 4/5 CIs exclude 0;
      5/5 positive; sign-test p=0.0625; 4/5 survive Holm.
    * SAEBENCH-8 (paper headline). 8-dataset mean SAE top-1=0.772, top-2=0.809, top-5=0.876 vs residual(LLM)
      baseline top-1=0.679; SAE beats baseline top-1 by +0.094 on the mean and on 7/8 datasets; set1
      reproduces repro-003 (0.767/0.688, deterministic); exception ag_news (top-1 0.694<0.732, rises to
      0.84 at top-5).
  - VERDICTS (R4): (1) sparse SAE-feature circuit REPLICATES across all 5 concepts (conclusive) under
    leak-free attribution + permutation null; min holdout faithfulness 0.873 at K=10; the test-leak fix
    lowers but does not erase the effect => STRONGER than the original n=1 claim. (2) real>random control
    PARTIALLY replicates (4/5 CIs exclude 0 + survive Holm, all 5 positive, sign-test p=0.0625 borderline;
    the original 21/19 is the lone non-significant one at n_per_class=250); reported honestly, NOT a clean
    win. (3) paper-grade SAEBench sparse-probing REPRODUCED at the 8-dataset x k{1,2,5} scale, upgrading
    repro-003's single-dataset smoke.
  - DOCS: ADR-0009 (new, pre-registered), EXPERIMENTS.md (circuit-multi + ctrl-probe-multi + saebench-paper
    rows), REPORT.md (Solidification section: per-concept circuit table + control table + SAEBench-8 table +
    3 verdicts; honest-scope rows updated to n=5 + leak-free; status updated), README (results rows +
    reproduce commands), PROGRESS (this), portfolio microscope.mdx (current status). On branch
    solidify/multiconcept-leakfree (NOT main).
  - HANDOFF / verification: (a) py_compile infra/modal_app.py + scripts/aggregate_controls.py; the 3 new fns
    + 3 entrypoints exist; (b) 170 CPU tests still pass (infra not imported by the package); (c) the numbers
    in EXPERIMENTS/REPORT/README/portfolio match circuit_multi.json / probing_multi_real.json /
    probing_multi_random.json / saebench_paper.json + solidification_summary.json (R5, no fabrication);
    (d) attribution is TRAIN-only in circuit_multi_eval (leak fix), bootstrap uses ONE shared index per iter
    (genuinely paired); (e) original circuit_eval/probing_eval + their artifacts untouched (D1). Re-run with
    seed 0 reproduces modulo minor GPU nondeterminism (E1).

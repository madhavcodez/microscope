# PROGRESS

## Current phase
**PHASES 1-6 COMPLETE (2026-06-22); Control-B steering recalibrated 2026-06-23.** CORE PROJECT DONE:
reproduce -> train -> head-to-head -> controls -> circuit -> write-up. Spend ≈ $11 of $30. Two CONCLUSIVE scorer-independent results (randomized-model
control + sparse feature circuit); the novel SAE-vs-transcoder head-to-head is INCONCLUSIVE (honest,
scorer/budget-limited). Repo clean: ruff + 170 tests pass, pip install -e . + CLI work. REPORT.md is the
finding (Phases 1-5 + abstract); README refreshed.

- **Phase 2 DONE:** sparsify wrapper (SAE=transcode/skip F/F, transcoder T/T, shared width/k) + Pythia smokes
  + both Gemma-2-2B custom coders trained @ L12 (width 16384, k=64, ~10M tokens, bf16): train-g2-sae,
  train-g2-tc, saved on the artifacts Volume (identical recipe => fair head-to-head). Gemma OOM (fp32) fixed bf16+batch8.
- **Phase 3 DONE (head-to-head = INCONCLUSIVE, pre-registered):** reconstruction SAE VE=0.514 (CI [.507,.519]);
  transcoder own-objective recon NOT cleanly isolable externally (limitation, documented — sparsify transcode
  hooks; ForwardOutput.fvu is input-recon inflated by skip). Auto-interp (delphi, custom dicts; fixed cpu->cuda
  + fp32->bf16 loader bugs): SAE det 0.540/fuzz 0.523 (n=58) vs TC det 0.539/fuzz 0.546 (n=61); both Δ bootstrap
  CIs include 0 => no significant difference (scorer-limited, matches repro-004). SAEBench on custom coders =
  SAE-only (resid-probing; transcoder N/A) + needs sae_lens adapter => deferred.
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

## Possible follow-ups (NOT started — optional, beyond core scope)
- ~~Calibrated Control-B steering sweep~~ DONE 2026-06-23 (ctrl-steer-v2): neutral prompt + finer grid =>
  discriminating, honestly-inconclusive (dom matches/slightly beats SAE, CI incl 0).
- sparsify->sae_lens adapter for the full SAEBench suite on the custom coders (Phase-3 SAEBench was SAE-only).
- Stronger auto-interp scorer (the near-chance bottleneck throughout): ATTEMPTED 2026-06-23 with a LOCAL
  Qwen2.5-7B-Instruct (ai-g2-7b-ATTEMPT). BLOCKED, no result — delphi keeps the Gemma base model resident
  on the GPU through scoring, leaving only ~16/22 GiB free; vLLM's startup guard rejects max_memory 0.9
  (19.83>16.05 GiB) and 0.5 underfits the 7B (~14.3 GiB weights + KV cache). Not an OOM; no memory
  fraction works while the base model is resident. Stopped after 2 startup failures (~$0.15, retry cap).
  3B head-to-head UNCHANGED + remains the reported result; scorer-strength question still OPEN. Code now
  parameterizes scorer_model + max_memory and scorer-tags the output json (no clobber). FIX (needs a Gate):
  free the base model from cuda before scoring (split cache vs score into two GPU calls) or use a >24 GiB GPU.
- Multi-layer (cross-component) circuit via sparse-feature-circuits.

## Phase 3 pre-registration (R3 — COMMITTED 2026-06-22 BEFORE any eval run; do not change post-hoc)
- **Coders compared:** train-g2-sae vs train-g2-tc (Gemma-2-2B L12, width 16384, k=64, identical recipe).
- **Feature sample:** 100 latents, drawn uniformly without replacement from range(16384) via
  `numpy.random.default_rng(0).choice(16384, 100, replace=False)`. The SAME 100 indices are used for
  BOTH coders (paired comparison). 100 ≤ 500 cap (C3); enough for a bootstrap CI.
- **Auto-interp metrics (delphi, LOCAL Qwen2.5-3B-Instruct scorer, no API):** detection accuracy +
  fuzzing accuracy, aggregate over the 100 latents. (Intruder = optional stretch only if it wires
  cleanly + budget allows; not committed.) Reported with bootstrap 95% CIs.
- **Reconstruction metric:** each coder's FVU / variance-explained on ITS OWN objective — SAE on
  resid_post@L12, transcoder on MLP-out@L12 (from MLP-in). Labeled as each-on-own-target (they are not
  the same target; this is the Transcoders-Beat-SAEs framing, not a like-for-like resid comparison).
- **SAEBench:** sparse_probing (validated in repro-003). absorption ONLY if it runs cleanly on the
  custom coder. NOTE: a SAEBench eval may be SAE-only; if it cannot evaluate the transcoder, that metric
  is N/A for the transcoder (documented, not faked) and the head-to-head for it leans on auto-interp+recon.
- **Hypothesis (reference, to CONFIRM or REFUTE — NOT assumed):** the skip-transcoder Pareto-dominates
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
- Repo created at C:\Users\madha\microscope; `git init`; local author `madhavcodez
  <madhavcbusiness@gmail.com>`; gpgsign off. GitHub: https://github.com/madhavcodez/microscope
  (private), `main` pushed.
- All memory files (CLAUDE.md, docs/RULES.md, PROGRESS.md, EXPERIMENTS.md, REPORT.md, adr/0000),
  ADR-0001 (model+tooling), ADR-0002 (Python 3.11 pin + GPU-host gate), SETUP_GPU.md runbook.
- Agent definitions: .claude/agents/{coder,tester,quality-checker}.md.
- pyproject.toml (base installs clean on CPU; ML stack in [gpu] extra), README, LICENSE,
  .gitattributes (LF for demo.sh), example configs, scripts/demo.sh.
- src/microscope: config.py (seeds/determinism/config-hash/RunRecord+EXPERIMENTS logging), Typer
  cli.py (7 stages), stage contracts as honest E4 stubs, real difference-of-means baseline.
- FOUNDATION UNIT through the agent loop (tester -> 2x quality-checker -> orchestrator merge):
  74 tests, 95% coverage, ruff clean, ruff-format clean, mypy clean. Verified: pip install -e .,
  microscope info/--help, determinism, stable config hash.

## In progress
- (nothing — PAUSED after Phase 1 per user instruction)

## Blocked / needs human
- **GATE #1 — RESOLVED.** Platform = **Modal** (existing creds + hf-token secret + credits). GPU =
  L4 24GB (~$0.80/hr, per-second billed → no idle burn). HARD cap **$30**; tightened run-gate
  (~$5/90min). Local scorer. Gemma-2-2B license accepted (account madhavc123). ADR-0003. Spend ≈ $3.
- (nothing currently blocking)

## Next (Phase 2 — when resuming; NOT started)
1. **First answer the open question (PHASE1_RETROSPECTIVE §4.2):** does dictionary_learning 0.1.0
   support a skip-transcoder? Introspect on Modal before committing to the SAE+transcoder deliverable.
2. Phase 2: train a custom SAE (+ transcoder if supported). SMOKE on Pythia-70M first (cheap), then a
   single solid Gemma-2-2B config. Log cost_est; stay ≤ $30 (~$27 remains).
3. Optional polish to fully close Phase 1 to paper-grade: scale SAEBench to 8 datasets × k{1,2,5}
   (repro-003 was a single-dataset smoke); larger auto-interp feature sample + stronger scorer if the
   absolute auto-interp number matters (it is currently labeled inconclusive by design).
4. Then Phases 3-6 through the coder->tester->quality-checker loop, logging cost_est each run (≤$30).

## Current task spec
- (orchestrator fills this per unit of work)

## Test log
- 2026-06-21 (tester): Wrote the CPU-verifiable pytest suite under tests/ (8 files, 71 tests,
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

## QC log
- 2026-06-21 (quality-checker x2 — research-integrity + python code-quality, run via workflow):
  Integrity verdict APPROVE; code-quality verdict REQUEST-CHANGES (lint/type-hint only). Findings
  and orchestrator resolution:
  - [HIGH, integrity] C3 ≤500-feature cap was NOT enforced on the randomized-model control path
    (unbounded-spend back door once implemented). FIXED: eval/controls.randomized_model_control now
    enforces MAX_FEATURES_PER_RUN with a ValueError + regression test added.
  - [MEDIUM, integrity] R1 reproduction-first is documented/sequenced but not mechanically gated in
    code. DEFERRED with explicit note (QC agreed acceptable): once Phase-1 stages are implemented on
    the host, add a check that a 'reproduced' row exists in EXPERIMENTS.md before `train` runs.
    Tracked here so it is not forgotten.
  - [code-quality] B008 (Typer idiom), UP035, I001, E501 (37 nits): FIXED — collections.abc imports,
    `_run_stage` typed, per-file B008 ignore for cli.py, `ruff check --fix` + `ruff format`, banner
    comments stripped, prose wrapped. ruff now passes; mypy clean.
  - [coverage] project_root cwd-fallback + git_commit OSError branch were untested (CPU-reachable):
    tests added; config.py 88%->91%, total 95%.
  - Re-verified post-fix: 74 passed, 95% coverage, ruff/format/mypy all clean.

## Coder log
- 2026-06-22 (coder): Implemented the previously-DEFERRED R1 mechanical gate (closes the MEDIUM
  integrity item in the QC log above). Pure-CPU, stdlib-only, no new deps.
  - config.py: added `reproduction_logged(path: Path | None = None) -> bool` — parses
    EXPERIMENTS.md, finds the `| run_id` header, locates the column whose header contains 'label',
    scans data rows (skips the `|---` separator), returns True iff any label cell contains
    'reproduced' (case-insensitive, substring). Missing file / no data rows -> False (fail closed).
    Helper `_split_markdown_row` tolerates leading/trailing pipes + whitespace. Also added
    `class ReproductionGateError(RuntimeError)` (R1 gate; distinct from _pending GPU gates).
  - cli.py: `train` now calls `reproduction_logged()` BEFORE `_prepare`/`_run_stage`; if False it
    prints a red message naming RULES.md R1 and `raise typer.Exit(code=3)`. Exit code 3 = R1 gate
    (distinct from code 2 = GPU/E4 gate). Imported `reproduction_logged` from `.config`.
  - Gate currently PASSES (EXPERIMENTS.md has reproduced rows repro-001/002/003) — verified
    `reproduction_logged()` returns True. Verified via Typer CliRunner: train with an empty/no-repro
    table -> exit 3 + R1 message + `_prepare` NOT reached; train with real table -> R1 passes
    through to the existing exit-2 GPU/E4 gate (unregressed).
  - Checks: `python -c "import microscope.cli"` OK; full pytest 94 passed; ruff check + ruff format
    + mypy all clean on config.py + cli.py. NOTE: did NOT add unit tests (tester's unit). Coverage
    of the new branches is currently exercised only by my ad-hoc CliRunner check, not the committed
    suite — tester must add regression tests (see handoff). Did not commit (per instruction).
- 2026-06-22 (coder): Phase 2 unit 1 — sparsify training wrapper + training YAMLs (ADR-0004). No
  training run here (that is unit 2 on Modal). Heavy imports kept lazy; package still imports on CPU.
  - src/microscope/saes/train.py REWRITTEN (was a `pending` stub):
    * `coder_config_dict(config, kind) -> dict[str, Any]` — PURE (no sparsify/torch import); the
      TESTABLE CORE. Maps RunConfig (+extra='allow' keys width/k/activation/batch_size/lr/save_dir/
      run_name) to a flat sparsify-settings dict. THE KEY INVARIANT: SAE => transcode=False &
      skip_connection=False; transcoder => both True; width(=>num_latents) and k(=>TopK L0) come from
      the SAME config so SAE vs skip-transcoder is a fair head-to-head. Validates: kind in
      {sae,transcoder}, width & k present + positive int (coerces "64"/64.0), activation in
      {topk,groupmax}, layer not None, lr/batch_size sane — all raise ValueError (fail fast).
    * `train_coder(config, kind) -> dict[str, Any]` — GPU-only. Validates via coder_config_dict
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
    pythia70m_smoke.yaml / gemma2_2b_reproduce.yaml untouched — those are the Phase-1 reproduce
    configs.)
  - Checks: both required CPU imports OK (`import microscope.cli`,
    `from microscope.saes.train import coder_config_dict`); ruff + ruff format + mypy CLEAN on
    train.py; ad-hoc CPU runs confirm the invariant (SAE F/F, transcoder T/T, shared width/k), every
    validation path raises ValueError, both YAMLs load + hash + flow through coder_config_dict, and
    train_coder raises GpuStackUnavailable (sparsify absent) — and ValueError before the gate on a
    config missing width/k. Did NOT add/modify tests (tester's unit). Did not commit (per instruction).
  - HANDOFF / tester must update 5 pre-existing tests that pinned the OLD stub contract (these are
    EXPECTED contract changes, not source bugs — same transition reproduce/harvest already made):
    (1) tests/test_pending.py::test_train_coder_stub_raises_pending — train_coder no longer raises
    GpuImplementationPending; it now raises GpuStackUnavailable (and ValueError first on a config
    without width/k). (2-5) tests/test_cli.py::test_train_with_valid_config_surfaces_gpu_gate and
    tests/test_reproduction_gate.py::{test_train_passes_r1_then_hits_gpu_gate_exit_2_via_real_parser,
    test_train_passes_r1_exit_2_via_function_monkeypatch, test_train_r1_and_gpu_exit_codes_are_distinct}
    all invoke `train` with the OLD pythia70m_smoke.yaml (no `k`) expecting exit 2 (GPU gate); now
    train_coder validates first and raises ValueError(missing k) -> different path. Fix: point those
    train invocations at the NEW train_pythia70m_smoke.yaml (has width+k) so they pass validation and
    reach the exit-2 GPU gate, OR assert the new validation behaviour. The R1 exit-3 tests
    (gate-shut) are unaffected and still pass (R1 fires before train_coder).
- 2026-06-23 (coder): RECALIBRATED Control-B steering (`steer_eval` in infra/modal_app.py) so it
  discriminates instead of returning the degenerate baseline-ceiling result. CALIBRATION FIX ONLY —
  same ADR-0005 pre-registered metric (success-rate-under-fluency) + concept (bias_in_bios prof 21 vs
  19, steer->19), so NOT a new Gate-4 decision (stated in docstring + EXPERIMENTS notes).
  - steer_eval edited in place: (1) NEUTRAL-prompt scan over 6 candidates ["I","The","This person",
    "They said","Yesterday","My favorite"], prints each one's unsteered baseline success, picks the
    one closest to 0.5 (new `n_scan=8` param keeps the scan cheap; chosen prompt's baseline is
    re-measured at full n_gen). (2) FINER grid coefs=[0.5,1,2,3,4]xresid_rms (was [2,4,8]). (3) Reports
    success AND ppl at EVERY coef for BOTH directions (full sweep incl coef0), each direction's best
    FLUENCY-PRESERVING coef, the steering EFFECT (success-baseline), and bootstrap CI95 on SAE-minus-dom
    with an explicit R4 honest verdict. E1: logs+sets seed (np/torch/cuda) — `seed` param, default 0.
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
  - HANDOFF / tester: this is a Modal-GPU control fn (not CPU-unit-testable end-to-end). Verify
    (a) py_compile/import of infra/modal_app.py; (b) the numbers in the docs match output line 54-55 of
    the run (STEER RESULT + BY DIR) and /root/outputs/steering.json on the artifacts volume; (c) the
    metric/concept are unchanged vs ADR-0005 (only prompt + coef grid + reporting changed); (d) the CI
    [-0.25,+0.125] includes 0 => 'inconclusive' label is correct (R4). A re-run with the same seed
    should reproduce (E1), modulo any nondeterminism in CUDA sampling.
- 2026-06-23 (coder): STRONGER-SCORER attempt for the auto-interp head-to-head (the near-chance
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
    for the eventual fix; (2) anti-clobber output filename — derive a scorer_tag from scorer_model; the
    default 3B writes the historical autointerp_<tag>.json, a non-default scorer writes
    autointerp_<tag>_<scorer_tag>.json (verified: 3B->autointerp_sae.json, 7B->autointerp_sae_7b.json),
    so a future 7B run will NOT overwrite the 3B results on the volume; (3) logged seed=0 + scorer_tag +
    max_memory into the result dict (E3); (4) corrected the docstring + comment to document the KNOWN
    BLOCKER honestly (R4) instead of the wrong "base model freed" claim. No production scorer switch (still
    local, C1). Updated EXPERIMENTS.md (ai-g2-7b-ATTEMPT row, label inconclusive(no result), no fake
    scores per R5), REPORT.md (Phase-3 "Scorer-strength check" note), PROGRESS.md.
  - HANDOFF / tester: CPU-verifiable parts only. Verify (a) py_compile infra/modal_app.py; (b) the
    filename derivation: scorer_model="Qwen/Qwen2.5-3B-Instruct" -> out_suffix="" (autointerp_<tag>.json);
    any other scorer -> "_<scorer_tag>" (7B -> _7b); (c) the 3B results on the volume are untouched
    (autointerp_sae.json/autointerp_tc.json still present, scorer=Qwen2.5-3B); (d) no fabricated 7B
    scores anywhere in the docs (R5). The 7B run itself is NOT reproducible until the base-model-free fix
    lands — it is a documented Gate (free base model from cuda before scoring, or a >24 GiB GPU).

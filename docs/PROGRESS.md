# PROGRESS

## Current phase
**PHASE 2 IN PROGRESS (resume prompt: Phases 2-4).** Course corrections 0a/0b/0c DONE: ADR-0002 fixed;
R1 mechanical gate merged (train exits 3 without a reproduced row, QC APPROVE); ADR-0004 (sparsify) +
sparsify 1.3.0 API verified on Modal. **PHASE 2 COMPLETE** — sparsify wrapper (SAE=transcode/skip F/F, transcoder T/T, shared width/k) +
Pythia smokes + both Gemma-2-2B custom coders trained on layer 12 (width 16384, k=64, ~10M tokens, bf16):
train-g2-sae + train-g2-tc, saved/loadable on the artifacts Volume (identical recipe => fair head-to-head).
First Gemma runs OOM'd (fp32) => fixed bf16+batch8. Phase-3 glue de-risked (delphi native sparsify loader;
SparseCoder reloads with sae_lens-like interface => SAEBench needs only a thin cfg adapter). NOW STARTING
PHASE 3 (auto-interp + SAEBench head-to-head). Spend ≈ $4-5 of $30.
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

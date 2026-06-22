# PROGRESS

## Current phase
**Phase 1 COMPLETE — PAUSED here per user (resume at Phase 2 later).** R1 gate satisfied:
reconstruction REPRODUCED (repro-001/002, VE ~0.80 / L0 ~83), SAEBench sparse-probing REPRODUCED
(repro-003, SAE 0.767 > residual 0.688), auto-interp pipeline reproduced (repro-004, detection 0.544 /
fuzz 0.529 — method works; absolute scores inconclusive, 3B local scorer near chance). Finding written
in REPORT.md; decisions/findings/open-questions in PHASE1_RETROSPECTIVE.md. Spend ≈ $3 of $30.
Phase 2 (custom SAE + transcoder training) NOT started.

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

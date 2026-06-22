# PROGRESS

## Current phase
Phase 1 (reproduction) IN PROGRESS on Modal. Reconstruction reproduced (VE 0.797 / L0 83);
auto-interp + SAEBench parts of the R1 gate still to do. Gate #1 fully resolved (Modal, $30).

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
- Phase 1 reproduction on Modal. DONE: pretrained Gemma Scope SAE reconstruction reproduced
  (repro-001: VE 0.797 / L0 83 @ layer_12/width_16k, documented ballpark). TODO to fully clear the
  R1 gate: (a) delphi auto-interp detection/fuzzing/intruder scores with the local Offline scorer,
  (b) a SAEBench metric (absorption / sparse_probing) — both on the same pretrained SAE.

## Blocked / needs human
- **GATE #1 — RESOLVED.** Platform = **Modal** (existing creds + hf-token secret + credits). GPU =
  L4 24GB (~$0.80/hr, per-second billed → no idle burn). HARD cap **$30**; tightened run-gate
  (~$5/90min). Local scorer. Gemma-2-2B license accepted by the user (account madhavc123). See
  ADR-0003. Spend so far ≈ $0.25.
- (nothing currently blocking — running autonomously on Modal within the $30 cap)

## Next
1. Phase 1 (finish R1 gate): wire delphi auto-interp (Offline local scorer) + one SAEBench eval on
   the pretrained Gemma Scope SAE; reproduce documented-ballpark scores; log to EXPERIMENTS.md.
2. Refactor the proven Modal reproduction recipe into src/microscope/{reproduce,activations,eval,
   autointerp} via coder→tester→quality-checker, replacing the E4 stubs; point the CLI at Modal.
3. Phase 2: train a custom SAE + skip-transcoder (smoke on Pythia-70M first). Then Phases 3-6.
   Log cost_est every run; stay ≤ $30.
4. Then Phases 2-6 through the coder->tester->quality-checker loop, logging cost_est each run (≤$30).

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

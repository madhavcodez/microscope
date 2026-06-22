# PROGRESS

## Current phase
Phase 0 — scaffolding COMPLETE + foundation unit merged. Phase 1 BLOCKED on Gate #1 (GPU host).

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
- (nothing active — awaiting Gate #1 decision before Phase 1)

## Blocked / needs human
- **GATE #1 — GPU host + spend (BLOCKS Phase 1 onward).** Local GPU is a GTX 1660 SUPER (6 GB),
  which is NOT enough for Gemma-2-2B SAE training / SAEBench (spec requires ~24 GB / RTX 3090).
  Need the human to provision a GPU host (RunPod / Vast / Lambda / Modal) and confirm a spend cap
  (target ≤ $80 total; any single run > $15 or > 2 h needs explicit approval per C2). Until then,
  only CPU-verifiable scaffolding/code can be built (config system, CLI, module contracts, tests).
- **Note — Python runtime.** Local interpreter is 3.13.14. The interp stack (dictionary_learning,
  delphi, sae-bench, nnsight) is validated on 3.10/3.11. The GPU host should pin 3.10 or 3.11.
  Captured for the human in ADR-0002 (proposed).

## Next
- Finish Phase 0 commit + push.
- Build CPU-verifiable foundation through the agent loop: config.py (pydantic configs + seed/
  determinism + run-metadata logging) -> tester -> quality-checker -> merge.
- Lay down module contracts (typed stubs + docstrings) for reproduce/activations/saes/autointerp/
  eval/circuits/steering, each marked "E4: verify library API on GPU host before implementing".
- Phase 1 reproduction begins once GATE #1 is resolved.

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

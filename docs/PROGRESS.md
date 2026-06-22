# PROGRESS

## Current phase
Phase 0 — scaffolding (in progress, orchestrator)

## Done
- Repo created at C:\Users\madha\microscope; `git init`; local author set to
  `madhavcodez <madhavcbusiness@gmail.com>`; gpgsign off.
- Memory files written: CLAUDE.md, docs/RULES.md, docs/PROGRESS.md, docs/EXPERIMENTS.md,
  docs/REPORT.md, docs/adr/0000-template.md.
- Agent definitions written: .claude/agents/{coder,tester,quality-checker}.md.

## In progress
- ADR-0001 (model + tooling choices), pyproject.toml, README stub, package skeleton, demo.sh.
- First commit + GitHub repo creation + push.

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
- (tester appends)

## QC log
- (quality-checker appends)

# CLAUDE.md — MicroScope project memory (always loaded)

## Mission
MicroScope is a reproducible mechanistic-interpretability toolkit. On a small open model
(Gemma-2-2B primary; Pythia-70M for smoke tests) it trains SAEs and transcoders, auto-interprets
and scores their features, evaluates with SAEBench, finds one feature circuit, and includes
adversarial controls (randomized-model baseline, simple-vector steering baseline) that test whether
the interpretability is real. Deliverable: a clean installable repo + a written report framed as a
finding. The defining property is HONEST EVALUATION.

## Non-negotiable rules (full text in docs/RULES.md)
- Research integrity: reproduce before novelty; controls mandatory; no cherry-picking; label every
  claim reproduced/novel/inconclusive; every claim traces to a logged number.
- Reproducibility: seeds set+logged; config-driven YAML runs; full run metadata logged; VERIFY library
  APIs before coding; types + docstrings.
- Cost: GPU ≤ $80 total; ask human before any run > $15 or > 2h; auto-interp uses a LOCAL scorer
  (no paid API by default); ≤ 500 features per auto-interp run; smallest model that proves the point.
- Decisions: no silent scope/product decisions — write an ADR, pick the conservative option, flag if
  it affects scope/validity. Never run destructive git/rm. Work on branches; commit often.

## Multi-agent workflow
Main session = orchestrator. Subagents (coder, tester, quality-checker) are one level deep and CANNOT
spawn subagents. Per unit of work: orchestrator writes a task spec -> coder -> tester -> quality-checker
-> merge. Up to ~3 parallel coders for independent modules; each still passes tester + quality-checker.

## How memory works here
Subagents start fresh every invocation. Durable memory = files. EVERY subagent reads CLAUDE.md,
docs/RULES.md, docs/PROGRESS.md, and relevant ADRs + the task spec FIRST, and updates the relevant
log file(s) LAST before returning.

## State pointers
- Current state & task: docs/PROGRESS.md
- Experiment log: docs/EXPERIMENTS.md
- Decisions: docs/adr/
- Report (the finding): docs/REPORT.md

## Conventions
- Python >= 3.10, Pydantic configs, Typer CLI, pytest tests.
- One YAML per run in experiments/configs/; log its hash with results.
- Set seeds via microscope.config helpers; log seed + git commit + hardware on every run.
- Primary tooling: dictionary_learning (train SAE/transcoder), delphi (auto-interp, LOCAL scorer),
  sae-bench (eval), sparse-feature-circuits (circuits), nnsight (activations), sae_lens or HF
  (load pretrained Gemma Scope SAEs for the Phase-1 reproduction).

# ADR 0002: Python runtime pin and GPU host
- Status: accepted (2026-06-21); host choice SUPERSEDED IN PART by ADR-0003, execution moved to
  **Modal serverless** (not a rented RunPod/Vast pod). The Python 3.11 pin and the $30 hard cap +
  tightened per-run gate (~$5/90 min) below still stand.
- Date: 2026-06-21

## Context
Discovered at scaffolding time on the local development machine:
- **Local Python is 3.13.14.** The interpretability stack (dictionary_learning, delphi, sae-bench,
  nnsight, sae_lens) is developed and validated on Python 3.10/3.11. 3.13 is new enough that wheels and
  pinned deps for parts of this stack may be missing or untested.
- **Local GPU is a GTX 1660 SUPER, 6 GB.** The spec and SAEBench docs call for ~24 GB (RTX 3090) to run
  Gemma-2-2B SAE training + SAEBench. 6 GB cannot hold Gemma-2-2B work; it can host Pythia-70M smoke
  tests only.

Per C4 and R1, the real results (Phase 1 reproduction onward) must run on Gemma-2-2B, which means a GPU
host with ≥ 24 GB VRAM. Choosing and paying for that host is a research/cost decision reserved for the
human (Human-Decision Gate #1).

## Decision
- **Pin the runtime to Python 3.11 on the GPU host** (3.10 acceptable). Keep `requires-python = ">=3.10"`
  in pyproject so the package metadata matches the spec, but the actual environment that runs training/
  eval/auto-interp is 3.11 to match the libraries' validated range.
- **GPU host: DECIDED, rented cloud, a single ~24 GB spot instance** (RTX 3090/4090 on
  RunPod/Vast, ~$0.25/hr). No A100/H100 (unjustified at this budget; would be a Gate).
- **Budget: HARD CAP $30 total** (overrides the spec's ≤$80 target downward; the user set $30).
  Because $30 is small, the cost gate from RULES.md C2 is tightened for this project: pause for human
  approval before any single run expected to exceed **~$5 or ~90 minutes**, hard-stop with a ~$5
  buffer (i.e. stop work near ~$25 spent), and log an estimated `cost_est` per run in EXPERIMENTS.md.
- **Budget discipline (how $30 is made to fit):** (1) develop + debug every library wrapper on
  Pythia-70M first, only spend Gemma-2-2B GPU time on code that already works; (2) local scorer only
  (no paid API); (3) modest token budgets, subset evals, ≤200 auto-interp features; (4) stop/terminate
  the pod whenever idle (idle burn, not compute, is the main risk); (5) avoid large activation caches
  (C3). If a run threatens the cap, STOP and tell the human: trim scope (SAE-only, fewer SAEBench
  evals) or top up.

## Alternatives considered
- **Use local Python 3.13 + local 6 GB GPU for everything**, rejected. Cannot fit Gemma-2-2B; library
  compatibility on 3.13 is unverified. Usable only for Pythia-70M smoke tests and CPU-side unit tests.
- **Pin Python 3.10**, acceptable alternative; 3.11 chosen as the newer of the two validated versions
  for better performance with equally broad wheel availability.
- **Commit to a specific cloud (RunPod/Vast/Lambda/Modal) now**, rejected as a silent decision (D1);
  it has cost + validity implications, so it is escalated as Gate #1 rather than chosen here.

## Consequences
- (+) Local machine is still useful: it builds and unit-tests all CPU-verifiable code and can run
  Pythia-70M smoke tests, so development proceeds while the GPU host is decided.
- (+) Pinning 3.11 on the host avoids burning GPU time on dependency-resolution failures.
- (-) Phase 1+ is blocked until the human resolves Gate #1 (host + spend cap).
- This ADR is `proposed` until the human picks a host; flip to `accepted` and record the host + the
  agreed spend cap at that point.

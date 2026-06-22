# MicroScope

A reproducible mechanistic-interpretability toolkit. On a small open model (**Gemma-2-2B** primary;
**Pythia-70M** for smoke tests) it trains sparse autoencoders (SAEs) and transcoders, automatically
interprets and scores their features, evaluates them with **SAEBench**, discovers one feature circuit,
and — critically — runs **adversarial controls** that test whether the interpretability is real:

- a **randomized-model baseline** (same pipeline on randomized weights — how much of the score is just
  token statistics?), and
- a **simple-vector steering baseline** (does an SAE feature steer better than a plain
  difference-of-means direction?).

The deliverable is a clean, installable repo **plus a written report framed as a finding** that honestly
labels what was *reproduced*, what is *novel*, and what is *inconclusive*. The defining property of this
project is **honest evaluation** — see [`docs/RULES.md`](docs/RULES.md).

> **Status:** Phase 0 (scaffolding) complete. Phase 1 (reproduction) is gated on a GPU host — see
> [`docs/PROGRESS.md`](docs/PROGRESS.md) and [`docs/adr/0002-runtime-and-gpu-host.md`](docs/adr/0002-runtime-and-gpu-host.md).

## Install

```bash
# CPU dev box (config + CLI + tests; no heavy ML stack):
pip install -e .

# With dev tooling:
pip install -e ".[dev]"
```

`pip install -e .` works anywhere. The CLI (`microscope --help`) and the config/determinism layer
run without a GPU. The heavy ML runtime is an opt-in extra installed on the GPU host (below).

## GPU host setup (Phase 1+)

The real results need **~24 GB VRAM** (e.g. RTX 3090/4090 or A10; the spec follows SAEBench's
documented 24 GB target for Gemma-2-2B). The local dev box (6 GB) only covers Pythia-70M smoke tests.
**Pin Python 3.11** on the host (the interp stack is validated on 3.10/3.11; see ADR-0002).

```bash
pip install -e ".[gpu]"
# Then install the source-only interpretability packages (verify exact API per RULES.md E4):
#   dictionary_learning, delphi (auto-interp), sae-bench (SAEBench), sparse-feature-circuits
# Exact install commands are pinned in ADR/PROGRESS once verified on the host.
```

## How to reproduce each result

_Filled in as each phase lands. The reproduction-first gate (Phase 1) must pass before any custom
training — see [`docs/RULES.md`](docs/RULES.md) R1._

```bash
microscope --help            # list all stages
# microscope reproduce ...   # Phase 1: load pretrained Gemma Scope SAE, reproduce auto-interp+SAEBench
# microscope train ...       # Phase 2: train SAE / skip-transcoder
# microscope autointerp ...  # Phase 3: explanations + detection/fuzzing/intruder scores (local scorer)
# microscope eval ...        # Phase 3: SAEBench scorecard
# microscope control ...     # Phase 4: randomized-model + steering-vs-simple-baseline controls
# microscope circuit ...     # Phase 5: discover + validate one feature circuit
```

## Hardware & cost notes
- GPU budget target: **≤ $80 total**. Any single run expected to exceed **$15 or 2 h** stops for human
  approval (RULES.md C2). Auto-interp uses a **local scorer** — no paid API by default (C1).
- Auto-interp is capped at **≤ 500 features/run** (C3). Activation caches can reach ~100 GB — the
  pipeline prefers the in-memory / small-token-budget path and cleans up.

## Honest scope
This is a research-engineering artifact, not a product demo. It reproduces published interpretability
results first, then builds custom training + controls + a circuit on that verified foundation. Results
— including inconclusive ones — are reported as a finding in [`docs/REPORT.md`](docs/REPORT.md), with
every number traceable to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Repo layout
```
src/microscope/   config, activations, saes/, autointerp/, eval/, circuits/, steering/, reproduce/, cli
docs/             RULES, PROGRESS, EXPERIMENTS, REPORT, adr/
experiments/      configs/ (one YAML per run), results/
scripts/          demo.sh
.claude/agents/   coder, tester, quality-checker (the build loop)
```

# MicroScope

A reproducible mechanistic-interpretability toolkit. On a small open model (**Gemma-2-2B** primary;
**Pythia-70M** for smoke tests) it trains sparse autoencoders (SAEs) and transcoders, automatically
interprets and scores their features, evaluates them with a probing/SAEBench setup, discovers one
feature circuit, and — critically — runs **adversarial controls** that test whether the interpretability
is real:

- a **randomized-model baseline** (same pipeline on randomized weights — how much of the score is just
  token statistics?), and
- a **simple-vector steering baseline** (does an SAE feature steer better than a plain
  difference-of-means direction?).

The deliverable is a clean, installable repo **plus a written report framed as a finding** that honestly
labels what was *reproduced*, what is *novel*, and what is *inconclusive*. The defining property of this
project is **honest evaluation** — see [`docs/RULES.md`](docs/RULES.md).

> **Status:** Phases 1–5 complete (reproduction → custom coders → head-to-head → controls → circuit),
> run on **Modal** under a **$30** GPU cap (~$11 spent). The finding is in [`docs/REPORT.md`](docs/REPORT.md);
> every number traces to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Phase 6 = this write-up.

## Results (the finding)

| Result | Verdict | Evidence |
|---|---|---|
| Reproduce Gemma Scope: reconstruction VE 0.80 / L0 83 | **reproduced** | `repro-001/002` |
| Reproduce SAEBench sparse probing (0.767 vs 0.688 baseline) | **reproduced** | `repro-003` |
| Auto-interp pipeline (local scorer) | **method reproduced**; scores near-chance | `repro-004` |
| Custom SAE vs skip-transcoder interpretability head-to-head | **inconclusive** (scorer/budget) | `ai-g2-sae/tc` |
| Randomized-model control: real-model SAE > randomized-model SAE | **conclusive** (+0.072, CI [0.033, 0.117]) | `ctrl-probe-*` |
| Steering: SAE feature vs difference-of-means | **inconclusive** (baseline ceiling) | `ctrl-steer` |
| Feature circuit: 5–10 SAE features = 94–97% of full accuracy | **conclusive (novel)** | `circuit-g2-sae` |

**Honest bottom line:** the novel transcoder-vs-SAE comparison didn't resolve under a $30 budget + a weak
local scorer, but two **scorer-independent** results are conclusive — the SAE encodes real structure
beyond token statistics (modestly; most probing signal is token-level), and a **sparse feature circuit**
faithfully mediates a concept. Full narrative: [`docs/REPORT.md`](docs/REPORT.md).

## Install

```bash
pip install -e .            # CPU dev box: config + CLI + tests; no heavy ML stack
pip install -e ".[dev]"     # + dev tooling (ruff, pytest, mypy)
```

`pip install -e .` works anywhere; the CLI (`microscope --help`) and the config/determinism layer run
without a GPU. The heavy ML runtime lives on Modal (below).

## How to reproduce

Experiments run on **Modal** (serverless GPU, per-second billed; [ADR-0003](docs/adr/0003-modal-execution-and-activation-recipe.md)).
The CPU-importable `microscope` package is the config/determinism layer + CLI skeleton; the **verified
runs are Modal functions** in [`infra/modal_app.py`](infra/modal_app.py), one per phase:

```bash
# Phase 1 — reproduction (the R1 gate)
modal run infra/modal_app.py::reproduce_recon                 # Gemma Scope reconstruction VE / L0
modal run infra/modal_app.py::saebench_sparse_probing         # SAEBench sparse probing
modal run infra/modal_app.py::auto_interp                     # auto-interp, local scorer (no API)

# Phase 2 — custom coders (identical recipe; fair head-to-head)
modal run infra/modal_app.py::train_main --config experiments/configs/train_gemma2_2b_l12.yaml --kind sae
modal run infra/modal_app.py::train_main --config experiments/configs/train_gemma2_2b_l12.yaml --kind transcoder

# Phase 3 — head-to-head (auto-interp + reconstruction)
modal run infra/modal_app.py::auto_interp_custom --run-name train_gemma2_2b_l12-sae
modal run infra/modal_app.py::auto_interp_custom --run-name train_gemma2_2b_l12-transcoder
modal run infra/modal_app.py::recon_eval --run-name train_gemma2_2b_l12-sae --kind sae

# Phase 4 — controls
modal run infra/modal_app.py::train_main --config experiments/configs/train_gemma2_2b_l12.yaml --kind sae --randomize
modal run infra/modal_app.py::probing_eval --run-name train_gemma2_2b_l12-sae               # control A (real)
modal run infra/modal_app.py::probing_eval --run-name train_gemma2_2b_l12-sae-random --randomize  # control A (random)
modal run infra/modal_app.py::steer_eval                      # control B (steering vs diff-of-means)

# Phase 5 — feature circuit
modal run infra/modal_app.py::circuit_eval
```

Run `PYTHONUTF8=1` on Windows. Each run logs a row to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) with
its config, seed, hardware, and cost.

## Hardware & cost

- Spec target ≤ $80; **this project ran under a hard $30 cap** (~$11 spent). Any single run expected to
  exceed ~$5 / 90 min stops for approval (RULES.md C2, tightened).
- Auto-interp uses a **local scorer** (delphi + vLLM) — no paid API (C1) — capped at ≤ 500 features/run (C3).
- Modal L4 (24 GB) is the execution target; the per-second billing means no idle burn.

## Honest scope

This is a research-engineering artifact, not a product demo. It reproduces published interpretability
results first (R1 gate), then builds custom training + controls + a circuit on that verified foundation.
Results — including the inconclusive ones — are reported as a finding in [`docs/REPORT.md`](docs/REPORT.md),
with every number traceable to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Decisions are recorded as
ADRs in [`docs/adr/`](docs/adr/).

## Repo layout

```
src/microscope/   config, activations, saes/, autointerp/, eval/, circuits/, steering/, reproduce/, cli
infra/modal_app.py  the verified GPU runs (one function per phase)
docs/             RULES, PROGRESS, EXPERIMENTS, REPORT, adr/ (0001–0006)
experiments/      configs/ (one YAML per run), results/
.claude/agents/   coder, tester, quality-checker (the build loop)
```

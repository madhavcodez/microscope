# MicroScope

A reproducible mechanistic-interpretability toolkit. On a small open model (Gemma-2-2B primary; Pythia-70M
for smoke tests) it trains sparse autoencoders (SAEs) and transcoders, automatically interprets and scores
their features, evaluates them with a probing/SAEBench setup, discovers one feature circuit, and, most
importantly, runs adversarial controls that test whether the interpretability is real:

- a randomized-model baseline (the same pipeline on randomized weights: how much of the score is just
  token statistics?), and
- a simple-vector steering baseline (does an SAE feature steer better than a plain difference-of-means
  direction?).

The deliverable is a clean, installable repo plus a written report framed as a finding that honestly labels
what was reproduced, what is novel, and what is inconclusive. The defining property of this project is
honest evaluation; see [`docs/RULES.md`](docs/RULES.md).

> Status: Phases 1-5 complete (reproduction, custom coders, head-to-head, controls, circuit), run on Modal
> under a $30 GPU cap (about $11 spent). The finding is in [`docs/REPORT.md`](docs/REPORT.md), and every
> number traces to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Phase 6 = this write-up.

## Results (the finding)

| Result | Verdict | Evidence |
|---|---|---|
| Reproduce Gemma Scope: reconstruction VE 0.80 / L0 83 | reproduced | `repro-001/002` |
| Reproduce SAEBench sparse probing (0.767 vs 0.688 baseline) | reproduced | `repro-003` |
| Auto-interp pipeline (local scorer) | method reproduced; 3B near-chance, 7B well above chance | `repro-004`, `ai-g2-*-7b` |
| Custom SAE vs skip-transcoder interpretability head-to-head | scorer-dependent: inconclusive at 3B, transcoder WINS at 7B (novel) | `ai-g2-sae/tc` (3B) then `ai-g2-sae-7b/tc-7b` |
| Custom SAE SAEBench sparse probing (sparsify to sae_lens adapter, encode-verified) | novel (honest negative): 0.670 < Gemma Scope 0.767 and < its own residual baseline 0.688 (budget training) | `saebench-custom-sae-v2` |
| Randomized-model control: real-model SAE > randomized-model SAE | conclusive (+0.072, CI [0.033, 0.117]) | `ctrl-probe-*` |
| Steering: SAE feature vs difference-of-means | inconclusive (both steer; dom matches SAE, CI incl 0) | `ctrl-steer-v2` |
| Feature circuit: 5-10 SAE features = 94-97% of full accuracy | conclusive (novel) | `circuit-g2-sae` |
| Multi-layer (cross-layer) circuit: about 9 features over L5/12/19 = 97% of full accuracy; concept accumulates by mid-depth | conclusive (novel) | `circuit-multilayer` |

In short: the novel transcoder-vs-SAE comparison was inconclusive under the weak 3B scorer, but a stronger
local 7B scorer (on an A100-40GB) lifts every score above chance and resolves it. The skip-transcoder is
significantly more interpretable on both auto-interp metrics (detection +0.053, CI [+0.016, +0.089];
fuzzing +0.059, CI [+0.019, +0.097]), confirming the pre-registered direction on the interpretability axis.
Two scorer-independent results are also conclusive: the SAE encodes real structure beyond token statistics
(modestly, since most probing signal is token-level), and a sparse feature circuit faithfully mediates a
concept. Full narrative: [`docs/REPORT.md`](docs/REPORT.md).

## Install

```bash
pip install -e .            # CPU dev box: config + CLI + tests; no heavy ML stack
pip install -e ".[dev]"     # + dev tooling (ruff, pytest, mypy)
```

`pip install -e .` works anywhere; the CLI (`microscope --help`) and the config/determinism layer run
without a GPU. The heavy ML runtime lives on Modal (below).

## How to reproduce

Experiments run on Modal (serverless GPU, per-second billed; [ADR-0003](docs/adr/0003-modal-execution-and-activation-recipe.md)).
The CPU-importable `microscope` package is the config/determinism layer plus CLI skeleton; the verified
runs are Modal functions in [`infra/modal_app.py`](infra/modal_app.py), one per phase:

```bash
# Phase 1: reproduction (the R1 gate)
modal run infra/modal_app.py::reproduce_recon                 # Gemma Scope reconstruction VE / L0
modal run infra/modal_app.py::saebench_sparse_probing         # SAEBench sparse probing
modal run infra/modal_app.py::auto_interp                     # auto-interp, local scorer (no API)

# Phase 2: custom coders (identical recipe; fair head-to-head)
modal run infra/modal_app.py::train_main --config experiments/configs/train_gemma2_2b_l12.yaml --kind sae
modal run infra/modal_app.py::train_main --config experiments/configs/train_gemma2_2b_l12.yaml --kind transcoder

# Phase 3: head-to-head (auto-interp + reconstruction)
# 3B scorer on L4 (the historical near-chance result):
modal run infra/modal_app.py::auto_interp_custom --run-name train_gemma2_2b_l12-sae
modal run infra/modal_app.py::auto_interp_custom --run-name train_gemma2_2b_l12-transcoder
# Stronger 7B scorer on A100-40GB (resolves the near-chance bottleneck; transcoder wins):
modal run infra/modal_app.py::autointerp_main --run-name train_gemma2_2b_l12-sae --gpu a100
modal run infra/modal_app.py::autointerp_main --run-name train_gemma2_2b_l12-transcoder --gpu a100
python scripts/headtohead_autointerp.py --dir artifacts_pull   # recompute the head-to-head + CIs
modal run infra/modal_app.py::recon_eval --run-name train_gemma2_2b_l12-sae --kind sae
# SAEBench sparse_probing on the CUSTOM SAE (sparsify->sae_lens adapter, ADR-0007):
modal run infra/modal_app.py::verify_saebench_adapter          # cheap pre-flight: adapter loads + accepted + encode-fidelity vs sparsify
modal run infra/modal_app.py::saebench_sparse_probing_custom   # full eval on the custom SAE

# Phase 4: controls
modal run infra/modal_app.py::train_main --config experiments/configs/train_gemma2_2b_l12.yaml --kind sae --randomize
modal run infra/modal_app.py::probing_eval --run-name train_gemma2_2b_l12-sae               # control A (real)
modal run infra/modal_app.py::probing_eval --run-name train_gemma2_2b_l12-sae-random --randomize  # control A (random)
modal run infra/modal_app.py::steer_eval                      # control B (steering vs diff-of-means)

# Phase 5: feature circuit (single-layer L12 custom SAE)
modal run infra/modal_app.py::circuit_eval
# Phase 5: multi-layer (cross-layer) circuit: pretrained Gemma Scope SAEs at L5/12/19 (ADR-0008)
modal run infra/modal_app.py::multilayer_circuit_main
```

Run `PYTHONUTF8=1` on Windows. Each run logs a row to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) with
its config, seed, hardware, and cost.

## Hardware & cost

- Spec target was $80 or less; this project ran under a hard $30 cap (about $11 spent). Any single run
  expected to exceed about $5 / 90 min stops for approval (RULES.md C2, tightened).
- Auto-interp uses a local scorer (delphi + vLLM), so there is no paid API (C1), capped at 500 or fewer
  features per run (C3).
- Modal L4 (24 GB) is the execution target; the per-second billing means no idle burn.

## Honest scope

This is a research-engineering artifact, not a product demo. It reproduces published interpretability
results first (R1 gate), then builds custom training, controls, and a circuit on that verified foundation.
Results, including the inconclusive ones, are reported as a finding in [`docs/REPORT.md`](docs/REPORT.md),
with every number traceable to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Decisions are recorded as ADRs
in [`docs/adr/`](docs/adr/).

## Repo layout

```
src/microscope/   config, activations, saes/, autointerp/, eval/, circuits/, steering/, reproduce/, cli
infra/modal_app.py  the verified GPU runs (one function per phase)
docs/             RULES, PROGRESS, EXPERIMENTS, REPORT, adr/
experiments/      configs/ (one YAML per run), results/
```

# GPU host setup runbook (Phase 1+)

This is the turnkey path to unblock **Gate #1** (docs/PROGRESS.md). The local dev box (GTX 1660 SUPER,
6 GB) only runs Pythia-70M smoke tests + CPU unit tests; the real Gemma-2-2B work needs **~24 GB VRAM**.

## 1. Pick a host (the decision reserved for you, Gate #1)

| Option | Card | Rough $/hr | Notes |
|--------|------|-----------|-------|
| **RunPod** (community/spot) | RTX 3090/4090 24 GB | ~$0.20-0.45 | Cheapest for the bulk of the work; recommended default. |
| **Vast.ai** | RTX 3090/4090 24 GB | ~$0.20-0.40 | Cheapest absolute, more variance in reliability. |
| **Lambda** | A10 24 GB / A100 40-80 GB | ~$0.75 (A10) / ~$1.10+ (A100) | Cleaner UX; A100 only if a run truly needs it. |
| **Modal** | A10G / A100 | usage-based | Best if you want serverless/scripted spin-up; per-second billing. |

Budget frame (DECIDED, ADR-0002): **HARD cap $30 total.** A 24 GB RTX 3090 spot at ~$0.25/hr ≈ 100+
GPU-hours, enough for Phase 1 reproduction + one SAE + one transcoder + auto-interp/SAEBench on a
subset + both controls + one circuit, IF disciplined. Tightened gate for this small budget: pause for
approval before any single run expected to exceed **~$5 or ~90 min**; hard-stop near ~$25 (keep a $5
buffer); log `cost_est` per run in EXPERIMENTS.md. The real budget risk is idle time, not compute -
**stop/terminate the pod whenever idle.**

**Make $30 fit:** develop + debug every wrapper on **Pythia-70M first** (tiny/fast/cheap) and spend
Gemma-2-2B GPU time only on code that already works; local scorer only (no paid API); modest token
budgets, subset evals, ≤200 auto-interp features; no large activation caches.

**Host:** a single 24 GB RTX 3090/4090 spot instance on RunPod (or Vast). No A100/H100 at this budget.

## 2. Bring up the environment (Python 3.11, ADR-0002)

```bash
git clone https://github.com/madhavcodez/microscope && cd microscope
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu,dev]"        # base + torch/transformers/nnsight/sae-lens + test tooling
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
microscope info                    # should now report the 24 GB GPU
pytest -q                          # CPU foundation suite should pass on the host too
```

## 3. Install the source-only interpretability packages, VERIFY THE API (RULES.md E4)

These are NOT pinned in pyproject because their exact install + API must be confirmed against the live
package before any wrapper is written. Canonical sources (confirm current install command on the host):

- **dictionary_learning**, SAE/transcoder training (ActivationBuffer + trainers). Saprmarks/EleutherAI.
- **delphi** (formerly `sae-auto-interp`), auto-interp with a **local** scorer model.
- **SAEBench** (`sae-bench`), the evaluation scorecard.
- **sparse-feature-circuits**, the circuit discovery + validation (same ecosystem as dictionary_learning).

For each, before writing the wrapper (E4): read the installed package source / run
`python -c` probes / check `--help` + docstrings, and record any divergence from the spec in a new ADR.
Pick a **local scorer model** delphi supports (a small instruct model that fits alongside Gemma-2-2B on
24 GB), this keeps auto-interp at $0 API spend (C1).

## 4. Run the pipeline (each stage already wired in the CLI)

```bash
# Phase 1, HARD GATE (reproduce before any custom training, R1):
microscope reproduce --config experiments/configs/gemma2_2b_reproduce.yaml
#   -> auto-interp (detection/fuzzing) + SAEBench on a pretrained Gemma Scope SAE must land in the
#      documented ballpark; logged to docs/EXPERIMENTS.md labelled 'reproduced'. If far off => bug to fix.

# Phase 2, smoke first (cheap), then real:
microscope train --config experiments/configs/pythia70m_smoke.yaml --kind sae       # converge check
microscope train --config experiments/configs/gemma2_2b_<...>.yaml --kind sae
microscope train --config experiments/configs/gemma2_2b_<...>.yaml --kind transcoder

# Phase 3, auto-interp + SAEBench on YOUR coders (local scorer, <=500 features, aggregates only):
microscope autointerp --config <...> --n-features 200 --scorer-model <local-scorer-id>
microscope eval --config <...>

# Phase 4, mandatory controls (R2):
microscope control --config <...> --kind randomized --n-features 200 --scorer-model <local-scorer-id>
microscope control --config <...> --kind steering

# Phase 5, one validated feature circuit:
microscope circuit --config <...> --task bias_in_bios_profession_classification
```

## 5. Cost hygiene on the host
- Stop/terminate the instance when idle, spot billing is per-second/minute.
- Avoid large on-disk activation caches (~100 GB possible); prefer in-memory / small token budgets and
  clean up between runs (C3).
- Log every run to docs/EXPERIMENTS.md with the real `cost_est` and `wall_clock` (E3) so the $80 budget
  stays auditable.

## 6. How the build proceeds on the host
Each stage follows the same loop: write a task spec → implement against the **verified** library API →
run determinism + metric-correctness checks (and ballpark checks for Phase 1) → check research integrity
→ merge. Reproduction (Phase 1) must go green before Phase 2 (R1).

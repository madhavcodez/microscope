# ADR 0004: Training library for the custom SAE + skip-transcoder — EleutherAI `sparsify`
- Status: accepted (decision); sparsify API to be E4-verified on Modal before any wrapper
- Date: 2026-06-22

## Context
Phase 2 must train BOTH a custom SAE and a custom **skip-transcoder** on Gemma-2-2B so Phase 3 can run a
fair head-to-head. ADR-0001 picked `dictionary_learning` for training. But the Phase-1 E4 probe found
`dictionary_learning` 0.1.0 exposes SAEs + trainers (TopK/JumpReLU/Standard/BatchTopK) and **no obvious
skip-transcoder** (PHASE1_RETROSPECTIVE §4.2). Skip-transcoders are the central object of the
"Transcoders Beat SAEs" line of work, where skip-transcoders Pareto-dominate SAEs on the
interpretability-vs-reconstruction frontier — reproducing/​testing that claim requires training the SAE
and the transcoder in the **same library and recipe**, or the comparison is confounded.

## Decision
Train both the SAE and the skip-transcoder with **EleutherAI `sparsify`**
(github.com/EleutherAI/sparsify):
- CLI: `python -m sparsify <model> [data] [--transcode]`; programmatic: `from sparsify import
  SaeConfig, Trainer, TrainConfig`. TopK activation; activations computed on-the-fly (no disk cache,
  good for the $30 budget / C3).
- The SAE and the skip-transcoder differ by the transcoder/skip flag (likely `--transcode` + a skip
  option) on the same trainer + config, holding width / k / tokens / layer / seed fixed → a fair Phase-3
  head-to-head.

## Why (and what it costs)
- `dictionary_learning` lacks the skip-transcoder; `sparsify` has it. Same-library training is the
  methodology the transcoder claim demands.
- **This supersedes part of ADR-0001:** the "single coherent ecosystem (dictionary_learning + delphi +
  sae-bench + sparse-feature-circuits)" no longer holds for training. `sparsify` dictionaries must be
  made to feed **delphi** (auto-interp) and **SAEBench** (eval). That format glue is a real, explicit
  Phase-3 task (verified, not assumed) — it is literature-plausible since sparsify / delphi / SAEBench
  are all EleutherAI and delphi already has a `sparsify` loader path (`delphi.sparse_coders.load_sparsify`
  was seen in the Phase-1 probe), but the adapter is verified on Pythia-70M dictionaries before any
  Gemma-2-2B spend.
- `dictionary_learning` is **retained only for the deferred Phase-5 circuit work**
  (`sparse-feature-circuits` is built on it); whether the Phase-5 circuit can consume a sparsify
  dictionary is a later question, out of scope for Phases 2–4.

## Alternatives considered
- **Hand-roll a skip-transcoder in `dictionary_learning`** — rejected: re-implementing the core novel
  object loses the known-good reference (violates the spirit of R1) and adds metric-bug surface.
- **SAE-only Phase 2 (skip the transcoder)** — rejected: the SAE-vs-transcoder head-to-head is the
  point of Phases 2–3; dropping it would gut the contribution.
- **Use `sparsify` for SAE but a different transcoder repo** — rejected: different libraries/recipes
  confound the comparison.

## E4 — verify on Modal before writing any wrapper (record divergences in a follow-up note)
Introspect the **installed** `sparsify` on the Modal image and confirm: (1) `SaeConfig` / `TrainConfig`
/ `Trainer` exist with the fields this ADR assumes; (2) the exact **skip-transcoder** flag (is the skip
connection a `--transcode` sub-option, a separate config field, or a distinct class?); (3) the
activation / hookpoint specification (which layer + which stream; for the transcoder, MLP-in→MLP-out is
internal); (4) the saved **dictionary format** + how to save/load a trained dictionary; (5) the
`delphi.sparse_coders.load_sparsify` loader contract and whether SAEBench can consume the same artifact
(or needs an adapter). Pin the verified install command + version here once confirmed.

## E4 verification results (2026-06-22, sparsify **1.3.0** on the Modal image)
Confirmed it is EleutherAI's sparsify (not Neural Magic's): module at
`site-packages/sparsify/__init__.py`, exports `Sae, SaeConfig, SparseCoder, SparseCoderConfig,
TranscoderConfig, Trainer, SaeTrainer, TrainConfig`.
- **SAE vs skip-transcoder = two config flags on the SAME `SaeConfig`/`Trainer`** (the fair-comparison
  property this ADR needed): `SaeConfig` fields = `activation, expansion_factor, normalize_decoder,
  num_latents, k, multi_topk, skip_connection, transcode`. → **SAE**: `transcode=False,
  skip_connection=False`; **skip-transcoder**: `transcode=True, skip_connection=True`.
- **`TrainConfig`** fields = `sae (SaeConfig), batch_size, grad_acc_steps, micro_acc_steps, loss_fn
  ('ce'|'fvu'|'kl'), optimizer ('adam'|'muon'|'signum'), lr, lr_warmup_steps, k_decay_steps,
  auxk_alpha, dead_feature_threshold, exclude_tokens, hookpoints, init_seeds, layers, layer_stride,
  distribute_modules, save_every, save_best, finetune, log_to_wandb, run_name, wandb_log_frequency,
  save_dir`. Hookpoint via `hookpoints=[...]` or `layers=[N]` (+`layer_stride`).
- **Entry point**: `Trainer(cfg: TrainConfig, dataset, model: PreTrainedModel)` — activations computed
  on-the-fly from the HF model (no disk cache, good for C3). (Confirm the train-launch method name —
  `.fit()` vs `.train()` — when writing the wrapper.)
- **Dictionary I/O**: `Sae`/`SparseCoder` expose `encode, decode, save_to_disk, load_from_disk,
  load_from_hub, load_many, set_decoder_norm_to_unit_norm`. So save with `save_to_disk`, reload with
  `load_from_disk` → satisfies "saved and loadable".
- **Width/sparsity mapping**: width via `num_latents` (absolute; 16384 ≈ Gemma Scope 16k) or
  `expansion_factor` (×d_model=2304). TopK `activation="topk"` with `k` = the exact L0 → pick `k≈64–80`
  to be comparable to the Gemma Scope L0≈82 reference; SAE and transcoder MUST share `k`/width.
- **CLI confirms**: `--transcode bool`, `--skip_connection bool`, `--activation {groupmax,topk}`,
  `--expansion_factor`, `--hookpoints`, `--layers`, `--k`, `--data_args`.
- **delphi glue exists**: `delphi.sparse_coders.load_sparsify` is present (a module) — the exact loader
  function + its contract is the Phase-3 integration to verify (don't assume the artifact is drop-in).
- No divergence from the decision above; install pinned: `pip install
  git+https://github.com/EleutherAI/sparsify.git` (resolved to 1.3.0).

## Consequences
- (+) A methodologically fair SAE-vs-skip-transcoder comparison in the same recipe.
- (−) Two training runs on Gemma-2-2B (SAE + transcoder) — both are cost-gated (~$5/90 min each; if
  either threatens the $30 cap, STOP and present options as a Gate).
- (−) New integration surface: sparsify→delphi and sparsify→SAEBench glue (Phase 3), verified on
  Pythia-70M first.
- Commits us to: smoke every sparsify pipeline on Pythia-70M before Gemma-2-2B; one solid config per
  coder (on-the-fly activations recompute per config, so no cheap sweeps).

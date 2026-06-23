"""Modal execution layer for MicroScope (the $30 GPU host — ADR-0003).

Modal is serverless + per-second billed, so there is no idle burn: GPU cost accrues only while a
function runs. This file builds the GPU image and exposes verification entrypoints. Stage wrappers
(reproduce/train/autointerp/eval/controls/circuit) are added here once their library APIs are
verified on this image (RULES.md E4).

Run:
  modal run infra/modal_app.py::probe       # CPU — import + dump library APIs (E4 verification)
  modal run infra/modal_app.py::gpu_smoke   # GPU — nvidia-smi, torch.cuda, gated-HF access check

Budget (ADR-0002): hard cap $30. Default GPU = L4 (24 GB, ~$0.80/hr) — cheapest 24 GB option that
fits Gemma-2-2B. Keep functions short; nothing here should run more than a few minutes.
"""

from __future__ import annotations

import modal

# Base image: the PyPI-resolvable core. Heavy interp libs (dictionary_learning, delphi, sae-bench,
# sparse-feature-circuits) are layered on AFTER this base builds + probes cleanly (E4 discipline).
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "datasets>=2.20",
        "nnsight>=0.3",
        "sae-lens>=4.0",
        "einops>=0.7",
        "safetensors>=0.4",
        "huggingface_hub>=0.24",
        "tqdm>=4.66",
        "numpy>=1.24",
    )
)

# Full image: base + the source-only interp libraries. Each install is `|| true` so a single bad
# URL cannot abort the whole build — probe_interp then reports exactly which imported and their APIs
# (E4 discovery). The verified install commands get pinned into this file once confirmed.
full_image = base_image.run_commands(
    "pip install 'git+https://github.com/saprmarks/dictionary_learning.git' || true",
    "pip install sae-bench || pip install 'git+https://github.com/adamkarvonen/SAEBench.git' || true",
    "pip install 'git+https://github.com/EleutherAI/delphi.git' || true",
    "pip install 'git+https://github.com/saprmarks/feature-circuits.git' || true",
    # EleutherAI sparsify — trains the custom SAE + skip-transcoder (ADR-0004). Install from the
    # EleutherAI repo (NOT PyPI 'sparsify', which is Neural Magic's unrelated package).
    "pip install 'git+https://github.com/EleutherAI/sparsify.git' || true",
    # torchvision is an unused transitive dep whose video API breaks the HF `datasets` torch
    # formatter (ImportError: VideoReader) during delphi's text caching — remove it.
    "pip uninstall -y torchvision || true",
    # flashinfer JIT-compiles CUDA kernels at runtime, but this image has the CUDA runtime, not the
    # toolkit (no nvcc/CUDA_HOME) — its sampler build fails in vLLM. Remove it so vLLM falls back to
    # prebuilt FlashAttention + the native PyTorch sampler (no compilation needed). (ADR-0003)
    "pip uninstall -y flashinfer-python flashinfer || true",
)

# Image variant that also bundles the local `microscope` package, so Modal training/eval functions can
# call the verified package wrappers (microscope.saes.train, etc.) instead of duplicating logic.
pkg_image = full_image.add_local_python_source("microscope")

app = modal.App("microscope-infra")

HF_SECRET = modal.Secret.from_name("hf-token")

# Persistent HF cache so the ~5 GB Gemma-2-2B + SAE download once, not every run (saves GPU $).
hf_cache = modal.Volume.from_name("microscope-hf-cache", create_if_missing=True)
CACHE = {"/root/.cache/huggingface": hf_cache}
CACHE_ENV = {"HF_HOME": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"}

# Persistent store for trained dictionaries (Phase 2) so Phase 3 eval can reload them across runs.
artifacts_vol = modal.Volume.from_name("microscope-artifacts", create_if_missing=True)


@app.function(
    image=base_image, gpu="L4", secrets=[HF_SECRET], volumes=CACHE, timeout=1800, retries=0
)
def reproduce_recon(layer: int = 12, width: str = "16k", n_docs: int = 128, seq_len: int = 128) -> dict:
    """Phase-1 reproduction (step 1): load the canonical Gemma Scope SAE and measure its
    variance-explained + mean L0 on Gemma-2-2B residual activations. Reproduces a KNOWN property of
    a pretrained SAE before any custom training (RULES.md R1). Streaming stats => O(1) memory."""
    import os

    import torch
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dev = "cuda"
    hook = f"blocks.{layer}.hook_resid_post"

    # Canonical Gemma Scope recipe: TransformerLens residual stream (what the SAE was trained on).
    model = HookedTransformer.from_pretrained("gemma-2-2b", dtype="bfloat16").to(dev)
    loaded = SAE.from_pretrained(
        "gemma-scope-2b-pt-res-canonical", f"layer_{layer}/width_{width}/canonical",
        device=dev, dtype="bfloat16",
    )
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    print("SAE cfg:", {k: str(v)[:50] for k, v in vars(sae.cfg).items()})

    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{n_docs}]")
    texts = [t for t in ds["text"] if t and t.strip()]

    sum_x = sum_x2 = sse = 0.0
    n_tok = 0
    l0_total = xn = rn = 0.0
    with torch.no_grad():
        for text in texts:
            toks = model.to_tokens(text)[:, :seq_len]  # prepends BOS
            _, cache = model.run_with_cache(
                toks, names_filter=hook, stop_at_layer=layer + 1, return_type=None
            )
            x = cache[hook][0, 1:].float()  # drop BOS position; [seq-1, d_model]
            if x.shape[0] == 0:
                continue
            feats = sae.encode(x.to(torch.bfloat16))
            recon = sae.decode(feats).float()
            sum_x += x.sum(0).double()
            sum_x2 += (x * x).sum(0).double()
            sse += ((x - recon) ** 2).sum().double()
            l0_total += (feats > 0).float().sum().item()
            xn += x.norm(dim=-1).sum().item()
            rn += recon.norm(dim=-1).sum().item()
            n_tok += x.shape[0]

    var = (sum_x2 - sum_x.pow(2) / n_tok).sum()
    fvu = (sse / var).item()
    out = {
        "sae_id": f"layer_{layer}/width_{width}/canonical",
        "hook": hook,
        "n_tokens": n_tok,
        "variance_explained": round(1.0 - fvu, 4),
        "mean_l0": round(l0_total / n_tok, 1),
        "mean_norm_x": round(xn / n_tok, 1),
        "mean_norm_recon": round(rn / n_tok, 1),
    }
    for k, v in out.items():
        print(f"{k:20s} {v}")
    return out


@app.function(image=full_image, secrets=[HF_SECRET], timeout=1200)
def probe_sparsify2() -> dict[str, str]:
    """E4: sparsify training flow — data tokenization helper + Trainer launch method + dataset type."""
    import inspect

    import sparsify

    out: dict[str, str] = {}
    # data submodule: how to tokenize/chunk a HF dataset for the Trainer
    try:
        from sparsify import data as sdata

        out["sparsify.data.public"] = ", ".join(n for n in dir(sdata) if not n.startswith("_"))[:400]
        for fn in ("chunk_and_tokenize", "MemmapDataset"):
            obj = getattr(sdata, fn, None)
            if obj is not None:
                try:
                    out[f"data.{fn}()"] = f"{fn}{inspect.signature(obj)}"[:320]
                except (ValueError, TypeError):
                    out[f"data.{fn}"] = f"(type {type(obj).__name__})"
    except Exception as exc:  # noqa: BLE001
        out["sparsify.data"] = f"FAIL: {type(exc).__name__}: {str(exc)[:120]}"

    # Trainer: launch method (.fit / .train / .run?) + __init__ signature
    tr = sparsify.Trainer
    out["Trainer.methods"] = ", ".join(
        m for m in dir(tr) if not m.startswith("_")
    )[:300]
    for m in ("fit", "train", "run"):
        meth = getattr(tr, m, None)
        if callable(meth):
            try:
                out[f"Trainer.{m}()"] = f"{m}{inspect.signature(meth)}"[:200]
            except (ValueError, TypeError):
                out[f"Trainer.{m}"] = "exists"
    try:
        out["Trainer.__init__"] = f"{inspect.signature(tr.__init__)}"[:300]
    except (ValueError, TypeError):
        pass

    for k, v in out.items():
        print(f"{k}\n    {v}")
    return out


@app.function(image=pkg_image, timeout=600)
def pkg_smoke() -> dict[str, str]:
    """Verify the local microscope package is importable inside Modal (the unit-2 integration path)."""
    import microscope
    from microscope.config import reproduction_logged  # exists today; confirms package code shipped

    return {"version": microscope.__version__, "rl_importable": str(callable(reproduction_logged))}


@app.function(
    image=pkg_image,
    gpu="L4",
    secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol},
    timeout=5400,
    retries=0,
)
def train_coder_modal(config_dict: dict, kind: str = "sae", randomize: bool = False) -> dict:
    """Phase 2: train an SAE or skip-transcoder on Modal via the verified microscope.saes.train wrapper.

    Persists the trained dictionary to the artifacts Volume so Phase-3 eval can reload it. Returns the
    metrics dict + a listing of the saved files (the smoke's save/load confirmation)."""
    import glob
    import os

    from microscope.config import RunConfig
    from microscope.saes.train import train_coder

    # Redirect the dictionary save into the mounted persistent Volume (overrides the YAML save_dir).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    overrides = {"save_dir": "/root/outputs/coders"}
    if randomize:  # randomized-model control (ADR-0005): random transformer, real embeddings
        overrides["randomize_model"] = True
        overrides["run_name"] = f"{config_dict.get('name', 'run')}-{kind}-random"
    cfg = RunConfig(**{**config_dict, **overrides})
    result = train_coder(cfg, kind)  # type: ignore[arg-type]

    save_path = result.get("save_path", "")
    all_paths = sorted(glob.glob(save_path + "/**", recursive=True))
    result["saved_files"] = [p.replace("/root/outputs/", "") for p in all_paths if os.path.isfile(p)][:20]
    result["n_saved_files"] = sum(1 for p in all_paths if os.path.isfile(p))
    artifacts_vol.commit()
    print("TRAIN RESULT:", result)
    return result


@app.local_entrypoint()
def train_main(config: str, kind: str = "sae", randomize: bool = False) -> None:
    """Run a training job: modal run infra/modal_app.py::train_main --config <yaml> --kind sae|transcoder."""
    import yaml

    with open(config) as fh:
        config_dict = yaml.safe_load(fh)
    result = train_coder_modal.remote(config_dict, kind, randomize)
    print("FINAL TRAIN RESULT:", result)


@app.function(image=full_image, secrets=[HF_SECRET], timeout=1200)
def probe_sparsify() -> dict[str, str]:
    """E4 (ADR-0004): introspect the installed EleutherAI sparsify — SAE/transcoder config + API."""
    import dataclasses
    import importlib
    import inspect
    import subprocess

    out: dict[str, str] = {}
    try:
        import sparsify
    except Exception as exc:  # noqa: BLE001
        out["sparsify"] = f"IMPORT FAILED: {type(exc).__name__}: {str(exc)[:200]}"
        for k, v in out.items():
            print(f"{k}\n    {v}")
        return out

    out["sparsify.version"] = getattr(sparsify, "__version__", "?")
    out["sparsify.file"] = str(getattr(sparsify, "__file__", "?"))  # confirm EleutherAI, not Neural Magic
    out["sparsify.public"] = ", ".join(n for n in dir(sparsify) if not n.startswith("_"))[:400]

    def _resolve(name: str) -> object:
        obj = getattr(sparsify, name, None)
        if obj is not None:
            return obj
        for sub in ("config", "trainer", "sae", "sparse_coder", "sparsecoder"):
            try:
                m = importlib.import_module(f"sparsify.{sub}")
                if hasattr(m, name):
                    return getattr(m, name)
            except Exception:  # noqa: BLE001
                continue
        return None

    for name in ("SaeConfig", "TrainConfig", "Trainer", "Sae", "SparseCoder"):
        obj = _resolve(name)
        if obj is None:
            out[name] = "NOT FOUND"
            continue
        if dataclasses.is_dataclass(obj):
            fields = [f.name for f in dataclasses.fields(obj)]
            out[f"{name}<fields>"] = ", ".join(fields)[:500]
            tc = [f for f in fields if any(s in f.lower() for s in ("transcod", "skip", "mlp"))]
            out[f"{name}<transcode/skip>"] = ", ".join(tc) or "(none in field names)"
        else:
            try:
                out[f"{name}()"] = f"{name}{inspect.signature(obj)}"[:300]
            except (ValueError, TypeError):
                out[name] = f"(type {type(obj).__name__})"
        if name in ("Sae", "SparseCoder"):
            methods = [
                m for m in dir(obj)
                if not m.startswith("_")
                and any(s in m.lower() for s in ("save", "load", "encode", "decode", "pretrained"))
            ]
            out[f"{name}.io_methods"] = ", ".join(methods)[:300]

    # CLI flags (does `python -m sparsify --help` expose --transcode / hookpoint / k / width?)
    try:
        r = subprocess.run(
            ["python", "-m", "sparsify", "--help"], capture_output=True, text=True, timeout=60
        )
        help_txt = (r.stdout or "") + (r.stderr or "")
        out["cli.transcode_in_help"] = str("transcode" in help_txt.lower())
        rel = [
            ln.strip() for ln in help_txt.splitlines()
            if any(s in ln.lower() for s in ("transcode", "skip", "hookpoint", "--layer", "--k", "expansion", "--data"))
        ]
        out["cli.relevant_flags"] = " || ".join(rel[:14])[:600]
    except Exception as exc:  # noqa: BLE001
        out["cli"] = f"help FAILED: {type(exc).__name__}: {str(exc)[:100]}"

    # delphi <- sparsify glue (the Phase-3 integration we must verify)
    try:
        from delphi.sparse_coders import load_sparsify

        out["delphi.load_sparsify()"] = f"load_sparsify{inspect.signature(load_sparsify)}"[:300]
    except Exception as exc:  # noqa: BLE001
        out["delphi.load_sparsify"] = f"n/a: {type(exc).__name__}: {str(exc)[:100]}"

    for k, v in out.items():
        print(f"{k}\n    {v}")
    return out


@app.function(image=full_image, secrets=[HF_SECRET], timeout=1200)
def probe_saebench() -> dict[str, str]:
    """E4: introspect the INSTALLED SAEBench sparse_probing module (pip pkg may differ from GitHub)."""
    import dataclasses
    import importlib
    import inspect

    out: dict[str, str] = {}
    for modname in (
        "sae_bench.evals.sparse_probing_sae_probes",
        "sae_bench.evals.sparse_probing",
        "sae_bench.evals.sparse_probing.main",
        "sae_bench.evals.sparse_probing.eval_config",
    ):
        try:
            m = importlib.import_module(modname)
            names = [n for n in dir(m) if not n.startswith("_")]
            out[modname] = ", ".join(names)[:400]
            for n in names:
                obj = getattr(m, n)
                if callable(obj) and any(k in n.lower() for k in ("run", "eval", "main", "probe")):
                    try:
                        out[f"{modname}.{n}()"] = f"{n}{inspect.signature(obj)}"[:320]
                    except (ValueError, TypeError):
                        pass
                if dataclasses.is_dataclass(obj):
                    out[f"{modname}.{n}<fields>"] = ", ".join(
                        f.name for f in dataclasses.fields(obj)
                    )[:320]
        except Exception as exc:  # noqa: BLE001
            out[modname] = f"FAIL: {type(exc).__name__}: {str(exc)[:100]}"

    # the SAE loader SAEBench uses
    try:
        from sae_bench.sae_bench_utils import general_utils

        out["general_utils"] = ", ".join(n for n in dir(general_utils) if not n.startswith("_"))[:300]
        for fn in ("load_and_format_sae", "get_results_filepath"):
            if hasattr(general_utils, fn):
                out[f"general_utils.{fn}()"] = f"{fn}{inspect.signature(getattr(general_utils, fn))}"
    except Exception as exc:  # noqa: BLE001
        out["general_utils"] = f"FAIL: {type(exc).__name__}: {str(exc)[:100]}"

    for k, v in out.items():
        print(f"{k}\n    {v}")
    return out


@app.function(image=full_image, secrets=[HF_SECRET], timeout=1200)
def probe2() -> dict[str, str]:
    """Confirm Gemma access post-license + locate Gemma Scope SAE IDs + deep-probe delphi/sae_bench."""
    import importlib
    import os
    import pkgutil

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out: dict[str, str] = {}

    # 1. Gemma-2-2B gated access (small tokenizer download).
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
        out["gemma2_2b"] = "ACCESS OK"
    except Exception as exc:  # noqa: BLE001
        out["gemma2_2b"] = f"FAIL: {type(exc).__name__}: {str(exc)[:140]}"

    # 2. Gemma Scope repo access (metadata only — list files, no big download).
    from huggingface_hub import HfApi

    api = HfApi()
    for repo in ("google/gemma-scope-2b-pt-res", "google/gemma-scope-2b-pt-res-canonical"):
        try:
            files = api.list_repo_files(repo, token=token)
            out[f"gemma_scope::{repo}"] = f"OK ({len(files)} files), e.g. {files[:3]}"
        except Exception as exc:  # noqa: BLE001
            out[f"gemma_scope::{repo}"] = f"FAIL: {type(exc).__name__}: {str(exc)[:100]}"

    # 3. sae_lens pretrained directory — find the exact gemma-scope release + sae_id strings.
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

        d = get_pretrained_saes_directory()
        gemma = [k for k in d if "gemma-scope-2b-pt-res" in k]
        out["sae_lens.gemma_releases"] = ", ".join(gemma[:8]) or "none found"
        if gemma:
            rel = d[gemma[0]]
            sample = list(rel.saes_map.items())[:3] if hasattr(rel, "saes_map") else "?"
            out["sae_lens.sample_sae_ids"] = f"{gemma[0]} -> {sample}"
    except Exception as exc:  # noqa: BLE001
        out["sae_lens.gemma_releases"] = f"probe failed: {type(exc).__name__}: {str(exc)[:140]}"

    # 4. Deep-probe delphi + sae_bench submodule trees (names only; selective deep dir()).
    for pkg in ("delphi", "sae_bench"):
        try:
            mod = importlib.import_module(pkg)
            subs = [m.name for m in pkgutil.iter_modules(mod.__path__)]
            out[f"{pkg}.submodules"] = ", ".join(subs)
        except Exception as exc:  # noqa: BLE001
            out[f"{pkg}.submodules"] = f"FAIL: {type(exc).__name__}: {str(exc)[:120]}"

    for k, v in out.items():
        print(f"{k}\n    {v}")
    return out


@app.function(image=full_image, secrets=[HF_SECRET], timeout=1200)
def probe3() -> dict[str, str]:
    """Signatures needed to write Phase-1 wrappers: sae_lens loader, delphi classes, sae_bench evals."""
    import importlib
    import inspect

    out: dict[str, str] = {}

    def sig(obj: object, name: str) -> None:
        try:
            out[name] = f"{name}{inspect.signature(obj)}"  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            out[name] = f"sig n/a: {type(exc).__name__}"

    # sae_lens SAE loader + layer-12 width-16k gemma-scope sae_ids
    try:
        from sae_lens import SAE

        sig(SAE.from_pretrained, "SAE.from_pretrained")
    except Exception as exc:  # noqa: BLE001
        out["SAE.from_pretrained"] = f"FAIL: {exc}"
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

        rel = get_pretrained_saes_directory()["gemma-scope-2b-pt-res-canonical"]
        ids = [k for k in rel.saes_map if "layer_12" in k]
        out["gemma_scope_canonical.layer12_ids"] = ", ".join(ids[:12]) or "none"
    except Exception as exc:  # noqa: BLE001
        out["gemma_scope_canonical.layer12_ids"] = f"FAIL: {str(exc)[:120]}"

    # delphi: dump public API of the wiring-relevant submodules
    for sub in ("config", "latents", "explainers", "scorers", "clients", "pipeline"):
        try:
            m = importlib.import_module(f"delphi.{sub}")
            out[f"delphi.{sub}"] = ", ".join(n for n in dir(m) if not n.startswith("_"))[:320]
        except Exception as exc:  # noqa: BLE001
            out[f"delphi.{sub}"] = f"FAIL: {type(exc).__name__}: {str(exc)[:100]}"

    # sae_bench: where do the evals live? try the common paths.
    import pkgutil

    for path in ("sae_bench.evals", "sae_bench"):
        try:
            m = importlib.import_module(path)
            subs = [s.name for s in pkgutil.iter_modules(m.__path__)]
            out[f"{path}.children"] = ", ".join(subs)
        except Exception as exc:  # noqa: BLE001
            out[f"{path}.children"] = f"FAIL: {type(exc).__name__}: {str(exc)[:100]}"

    for k, v in out.items():
        print(f"{k}\n    {v}")
    return out


@app.function(image=full_image, timeout=1200)
def probe_interp() -> dict[str, str]:
    """Report which interp libs installed + a sketch of their public API (E4 verification, CPU)."""
    import importlib
    import inspect

    # Candidate import names per logical library (install name != import name sometimes).
    candidates = {
        "dictionary_learning": ["dictionary_learning"],
        "sae_bench": ["sae_bench", "saebench", "sae_bench_utils"],
        "delphi": ["delphi"],
        "feature_circuits": ["feature_circuits", "circuit", "attribution"],
    }
    report: dict[str, str] = {}
    for logical, names in candidates.items():
        imported = None
        for name in names:
            try:
                imported = importlib.import_module(name)
                report[logical] = f"OK as '{name}' v{getattr(imported, '__version__', '?')}"
                # dump top-level public names to learn the surface
                public = [n for n in dir(imported) if not n.startswith("_")]
                report[f"{logical}.dir"] = ", ".join(public[:40])
                break
            except Exception as exc:  # noqa: BLE001
                report[logical] = f"import '{name}' failed: {type(exc).__name__}: {str(exc)[:120]}"
        # try to surface key submodules/signatures for the ones we care about most
        if imported is not None and logical == "dictionary_learning":
            for sub in ("training", "trainers", "buffer", "dictionary"):
                try:
                    m = importlib.import_module(f"dictionary_learning.{sub}")
                    report[f"dictionary_learning.{sub}"] = ", ".join(
                        n for n in dir(m) if not n.startswith("_")
                    )[:300]
                except Exception as exc:  # noqa: BLE001
                    report[f"dictionary_learning.{sub}"] = f"n/a: {type(exc).__name__}"

    for k, v in report.items():
        print(f"{k}\n    {v}")
    return report


@app.function(image=base_image, timeout=900)
def probe() -> dict[str, str]:
    """Import the core libs and report versions (CPU, ~free). The E4 verification starting point."""
    import importlib

    report: dict[str, str] = {}
    for mod in ("torch", "transformers", "datasets", "nnsight", "sae_lens", "einops"):
        try:
            m = importlib.import_module(mod)
            report[mod] = getattr(m, "__version__", "imported (no __version__)")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the probe
            report[mod] = f"IMPORT FAILED: {type(exc).__name__}: {exc}"
    for k, v in report.items():
        print(f"{k:14s} {v}")
    return report


@app.function(image=base_image, gpu="L4", secrets=[HF_SECRET], timeout=900)
def gpu_smoke() -> dict[str, str]:
    """Confirm the GPU is allocated, torch sees CUDA, and the hf-token can reach gated Gemma-2-2B."""
    import os

    import torch

    out: dict[str, str] = {
        "cuda_available": str(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["vram_total_gib"] = f"{total / 1024**3:.1f}"

    # hf-token secret may expose HF_TOKEN or HUGGING_FACE_HUB_TOKEN — accept either.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out["hf_token_present"] = str(bool(token))
    try:
        from huggingface_hub import HfApi

        who = HfApi().whoami(token=token)
        out["hf_user"] = who.get("name", "?")
    except Exception as exc:  # noqa: BLE001
        out["hf_user"] = f"whoami FAILED: {type(exc).__name__}: {exc}"

    # Confirm gated Gemma-2-2B license is accepted on this account (tokenizer is a tiny download).
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
        out["gemma2_2b_access"] = "OK"
    except Exception as exc:  # noqa: BLE001
        out["gemma2_2b_access"] = f"GATED/FAILED: {type(exc).__name__}: {str(exc)[:160]}"

    for k, v in out.items():
        print(f"{k:18s} {v}")
    return out


@app.function(
    image=base_image, gpu="L4", secrets=[HF_SECRET], volumes=CACHE, timeout=2400, retries=0
)
def multi_layer_recon(
    layers: str = "5,12,19", width: str = "16k", n_docs: int = 96, seq_len: int = 128
) -> dict:
    """Phase-1 reproduction (objective, no LLM): VE + L0 of canonical Gemma Scope SAEs across layers.

    Reproduces the documented Gemma Scope trend across depth. One model load; capture all needed
    resid_post hooks in a single forward per doc. TransformerLens recipe, BOS excluded (ADR-0003)."""
    import torch
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    dev = "cuda"
    layer_list = [int(x) for x in layers.split(",")]
    hooks = {ly: f"blocks.{ly}.hook_resid_post" for ly in layer_list}
    model = HookedTransformer.from_pretrained("gemma-2-2b", dtype="bfloat16").to(dev)
    saes = {}
    for ly in layer_list:
        loaded = SAE.from_pretrained(
            "gemma-scope-2b-pt-res-canonical",
            f"layer_{ly}/width_{width}/canonical",
            device=dev,
            dtype="bfloat16",
        )
        saes[ly] = loaded[0] if isinstance(loaded, tuple) else loaded

    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{n_docs}]")
    texts = [t for t in ds["text"] if t and t.strip()]
    max_layer = max(layer_list)
    stats = {ly: {"sx": 0.0, "sx2": 0.0, "sse": 0.0, "l0": 0.0} for ly in layer_list}
    n_tok = {ly: 0 for ly in layer_list}
    with torch.no_grad():
        for text in texts:
            toks = model.to_tokens(text)[:, :seq_len]
            _, cache = model.run_with_cache(
                toks,
                names_filter=list(hooks.values()),
                stop_at_layer=max_layer + 1,
                return_type=None,
            )
            for ly in layer_list:
                x = cache[hooks[ly]][0, 1:].float()  # drop BOS
                if x.shape[0] == 0:
                    continue
                feats = saes[ly].encode(x.to(torch.bfloat16))
                recon = saes[ly].decode(feats).float()
                s = stats[ly]
                s["sx"] += x.sum(0).double()
                s["sx2"] += (x * x).sum(0).double()
                s["sse"] += ((x - recon) ** 2).sum().double()
                s["l0"] += (feats > 0).float().sum().item()
                n_tok[ly] += x.shape[0]

    results = {}
    for ly in layer_list:
        s = stats[ly]
        var = (s["sx2"] - s["sx"].pow(2) / n_tok[ly]).sum()
        results[f"layer_{ly}"] = {
            "variance_explained": round((1 - s["sse"] / var).item(), 4),
            "mean_l0": round(s["l0"] / n_tok[ly], 1),
            "n_tokens": n_tok[ly],
        }
        print(f"layer_{ly}: {results[f'layer_{ly}']}")
    return {"width": width, "results": results}


@app.function(
    image=full_image, gpu="L4", secrets=[HF_SECRET], volumes=CACHE, timeout=3600, retries=0
)
def auto_interp(
    layer: int = 12,
    width: str = "16k",
    max_latents: int = 20,
    scorer_model: str = "Qwen/Qwen2.5-3B-Instruct",
    n_tokens: int = 200_000,
) -> dict:
    """Phase-1 auto-interp: run delphi (LOCAL Offline scorer, no paid API) on the Gemma Scope SAE.

    Uses delphi's native Gemma Scope path (verified): sparse_model='google/gemma-scope-2b-pt-res',
    hookpoint='layer_<L>/width_<W>/average_l0_<L0>'. Reports aggregate detection + fuzzing accuracy.
    Scores from a small local scorer are NOT directly comparable to papers using frontier scorers —
    label accordingly (R4). Capped at <=500 latents (RULES.md C3)."""
    import os

    # Belt-and-suspenders: also disable vLLM's flashinfer sampler via env (flashinfer is uninstalled
    # from the image because its runtime JIT needs a CUDA toolkit this image lacks — see full_image).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import asyncio
    from pathlib import Path

    from delphi.__main__ import run
    from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
    from delphi.log.result_analysis import get_agg_metrics, load_data
    from huggingface_hub import HfApi

    assert max_latents <= 500, "auto-interp cap is 500 latents (RULES.md C3)"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # delphi needs a concrete average_l0 dir (no 'canonical') — discover a valid one near L0~82.
    files = HfApi().list_repo_files("google/gemma-scope-2b-pt-res", token=token)
    prefix = f"layer_{layer}/width_{width}/average_l0_"
    l0s = sorted({int(f.split(prefix)[1].split("/")[0]) for f in files if f.startswith(prefix)})
    if not l0s:
        raise RuntimeError(f"No average_l0 dirs found for {prefix} in gemma-scope-2b-pt-res")
    l0 = min(l0s, key=lambda v: abs(v - 82))
    hookpoint = f"layer_{layer}/width_{width}/average_l0_{l0}"
    print(f"hookpoint={hookpoint}  available_l0s={l0s}")

    name = f"g2-l{layer}-w{width}-l0{l0}"
    run_cfg = RunConfig(
        name=name,
        model="google/gemma-2-2b",
        sparse_model="google/gemma-scope-2b-pt-res",
        hookpoints=[hookpoint],
        explainer_provider="offline",
        explainer_model=scorer_model,
        explainer_model_max_len=4096,
        explainer="default",
        scorers=["detection", "fuzz"],
        max_latents=max_latents,
        filter_bos=True,
        num_gpus=1,
        max_memory=0.6,  # base model is freed after caching; 0.6 of 22GB fits the 3B scorer + KV
        seed=22,
        verbose=False,  # skip delphi's plotly/kaleido visualization (kaleido absent -> crash)
        hf_token=token,
        cache_cfg=CacheConfig(
            dataset_repo="NeelNanda/pile-10k",
            dataset_split="train",
            dataset_column="text",
            batch_size=8,
            cache_ctx_len=256,
            n_tokens=n_tokens,
            n_splits=5,
        ),
        constructor_cfg=ConstructorConfig(
            example_ctx_len=32,
            min_examples=50,
            n_non_activating=50,
            non_activating_source="random",
        ),
        sampler_cfg=SamplerConfig(
            n_examples_train=20,
            n_examples_test=40,
            n_quantiles=10,
            train_type="quantiles",
            test_type="quantiles",
        ),
    )
    asyncio.run(run(run_cfg))

    # delphi stores scores under the RESOLVED module name (e.g. 'layers.12'), not the gemma-scope
    # params path, so load_data(hookpoints=[params_path]) finds nothing. Read the raw per-latent
    # score files directly and aggregate defensively (schema is also printed for transparency).
    import glob
    import re

    # delphi score files are JSON arrays of per-example records each with one "activating" (ground
    # truth) and one "prediction" (scorer's call). Extract those booleans by regex straight from the
    # raw text — robust to any JSON-parse quirk — and score balanced accuracy = mean(activating==pred).
    act_re = re.compile(r'"activating":\s*(true|false)')
    pred_re = re.compile(r'"prediction":\s*(true|false)')

    out: dict = {"hookpoint": hookpoint, "scorer": scorer_model, "max_latents": max_latents, "scores": {}}
    for scorer in ("detection", "fuzz"):
        sdir = Path.cwd() / "results" / name / "scores" / scorer
        files = glob.glob(str(sdir / "*"))
        per_latent_acc: list[float] = []
        total_correct = total_examples = 0
        for fp in files:
            try:
                text = open(fp).read()
            except OSError:
                continue
            acts = act_re.findall(text)
            preds = pred_re.findall(text)
            n = min(len(acts), len(preds))
            if n == 0:
                continue
            correct = sum(1 for a, p in zip(acts[:n], preds[:n]) if a == p)
            per_latent_acc.append(correct / n)
            total_correct += correct
            total_examples += n
        out["scores"][scorer] = {
            "n_latents_scored": len(per_latent_acc),
            "n_examples": total_examples,
            "mean_accuracy_macro": round(sum(per_latent_acc) / len(per_latent_acc), 4) if per_latent_acc else None,
            "accuracy_micro": round(total_correct / total_examples, 4) if total_examples else None,
        }
    print("AUTO-INTERP RESULT:", out)
    return out


@app.function(
    image=full_image, gpu="L4", secrets=[HF_SECRET], volumes=CACHE, timeout=3600, retries=0
)
def saebench_sparse_probing(
    layer: int = 12, width: str = "16k", train_size: int = 1500, test_size: int = 500
) -> dict:
    """Phase-1 SAEBench eval: sparse-probing accuracy of the canonical Gemma Scope SAE.

    Objective metric (no LLM scorer). Verified API: sae_bench.evals.sparse_probing.main.run_eval with
    selected_saes as native sae_lens (release, sae_id) tuples. Headline = SAE probe top-1 accuracy vs
    the residual-stream baseline (~0.65 documented); the SAE should clearly beat the baseline."""
    import glob
    import json
    import os

    from sae_bench.evals.sparse_probing import main as sp

    sae_id = f"layer_{layer}/width_{width}/canonical"
    selected_saes = [("gemma-scope-2b-pt-res-canonical", sae_id)]
    config = sp.SparseProbingEvalConfig(
        model_name="gemma-2-2b",
        random_seed=42,
        llm_batch_size=32,
        llm_dtype="bfloat16",
        dataset_names=["LabHC/bias_in_bios"],  # single dataset = fast smoke
        probe_train_set_size=train_size,
        probe_test_set_size=test_size,
        context_length=128,
        k_values=[1],
    )
    out_path = "eval_results/sparse_probing"
    sp.run_eval(
        config, selected_saes, "cuda", out_path,
        force_rerun=True, clean_up_activations=True, save_activations=False,
    )

    files = glob.glob(os.path.join(out_path, "*_eval_results.json"))
    if not files:
        raise RuntimeError(f"no SAEBench result json written to {out_path}")
    res = json.load(open(files[0]))
    metrics = res.get("eval_result_metrics", {})
    sae_m = metrics.get("sae", {})
    llm_m = metrics.get("llm", {})
    out = {
        "sae_id": sae_id,
        "dataset": "LabHC/bias_in_bios",
        "sae_top_1_test_accuracy": sae_m.get("sae_top_1_test_accuracy"),
        "sae_test_accuracy": sae_m.get("sae_test_accuracy"),
        "llm_top_1_test_accuracy": llm_m.get("llm_top_1_test_accuracy"),
        "llm_test_accuracy": llm_m.get("llm_test_accuracy"),
    }
    print("SAEBENCH SPARSE-PROBING RESULT:", out)
    return out


@app.function(
    image=pkg_image, secrets=[HF_SECRET], volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=900
)
def probe_phase3_glue() -> dict:
    """E4 (Phase 3): verify how a sparsify-trained dict feeds delphi (load_sparsify) + assess SAEBench.

    Reads the trained Gemma SAE from the artifacts Volume and inspects the delphi loader contract +
    the sparsify coder's interface (to decide whether SAEBench needs an adapter)."""
    import glob
    import importlib
    import inspect
    import os

    out: dict = {}
    sae_path = "/root/outputs/coders/train_gemma2_2b_l12-sae"
    out["sae_dir_exists"] = str(os.path.isdir(sae_path))
    out["sae_files"] = [
        p.replace("/root/outputs/", "")
        for p in glob.glob(sae_path + "/**", recursive=True)
        if os.path.isfile(p)
    ][:10]

    # 1) delphi's sparsify loader (the auto-interp glue)
    try:
        sc = importlib.import_module("delphi.sparse_coders")
        for fn in ("load_sparse_coders", "load_hooks_sparse_coders", "load_sparsify"):
            obj = getattr(sc, fn, None)
            if obj is not None and callable(obj):
                try:
                    out[f"delphi.sparse_coders.{fn}()"] = f"{fn}{inspect.signature(obj)}"[:260]
                except (ValueError, TypeError):
                    out[f"delphi.sparse_coders.{fn}"] = "callable (no sig)"
        mod = importlib.import_module("delphi.sparse_coders.load_sparsify")
        out["delphi.load_sparsify.module"] = ", ".join(
            n for n in dir(mod) if not n.startswith("_")
        )[:300]
    except Exception as exc:  # noqa: BLE001
        out["delphi.sparse_coders"] = f"FAIL: {type(exc).__name__}: {str(exc)[:140]}"

    # 2) load the trained sparsify dict + inspect its interface (for SAEBench compatibility)
    try:
        import sparsify

        coder = sparsify.SparseCoder.load_from_disk(sae_path + "/layers.12", device="cpu")
        out["loaded_coder_type"] = type(coder).__name__
        out["coder_cfg_type"] = type(getattr(coder, "cfg", None)).__name__
        out["coder_attrs"] = ", ".join(
            a for a in dir(coder)
            if not a.startswith("_")
            and any(s in a.lower() for s in ("encode", "decode", "cfg", "w_dec", "w_enc", "d_in", "num_latents", "forward"))
        )[:300]
    except Exception as exc:  # noqa: BLE001
        out["load_coder"] = f"FAIL: {type(exc).__name__}: {str(exc)[:180]}"

    for k, v in out.items():
        print(f"{k}\n    {v}")
    return out


@app.function(
    image=pkg_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=1800, retries=0,
)
def recon_eval(run_name: str, kind: str, layer: int = 12, n_docs: int = 96, seq_len: int = 128) -> dict:
    """Phase 3: each-coder reconstruction variance-explained on its OWN objective, on HF activations
    (matching how sparsify trained the coders), with a bootstrap 95% CI over documents.

    SAE target = resid_post@L (output of model.layers[L]); transcoder target = MLP-out@L (from MLP-in).
    Uses coder.forward(input) for the reconstruction so the transcoder's skip connection is included."""
    import os

    import numpy as np
    import sparsify
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dev = "cuda"
    coder = sparsify.SparseCoder.load_from_disk(
        f"/root/outputs/coders/{run_name}/layers.{layer}", device=dev
    )
    model = (
        AutoModel.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, token=token)
        .to(dev).eval()
    )
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{n_docs}]")
    texts = [t for t in ds["text"] if t and t.strip()]

    layer_mod = model.layers[layer]
    mlp_mod = layer_mod.mlp
    cap: dict = {}

    def _t(x):  # extract a tensor from a tensor / tuple / structured output
        if torch.is_tensor(x):
            return x
        if isinstance(x, (tuple, list)) and torch.is_tensor(x[0]):
            return x[0]
        for a in ("sae_out", "output", "recon", "reconstruction", "out", "y"):
            v = getattr(x, a, None)
            if torch.is_tensor(v):
                return v
        return None

    per_doc = []
    printed = False
    with torch.no_grad():
        for text in texts:
            cap.clear()
            ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(dev)
            if kind == "sae":
                h = layer_mod.register_forward_hook(lambda m, i, o: cap.__setitem__("y", _t(o)))
                model(ids)
                h.remove()
                src = tgt = cap["y"][0, 1:].float()
            else:
                h1 = mlp_mod.register_forward_pre_hook(lambda m, i: cap.__setitem__("in", i[0]))
                h2 = mlp_mod.register_forward_hook(lambda m, i, o: cap.__setitem__("out", _t(o)))
                model(ids)
                h1.remove()
                h2.remove()
                src = cap["in"][0, 1:].float()
                tgt = cap["out"][0, 1:].float()
            if tgt.shape[0] == 0:
                continue
            fwd = coder(src.to(torch.bfloat16))
            recon = _t(fwd)
            if recon is None:
                raise RuntimeError(f"could not extract recon from coder output: {type(fwd)} {dir(fwd)}")
            recon = recon.float()
            if not printed:
                print(f"coder fwd type={type(fwd).__name__} recon_shape={tuple(recon.shape)} tgt={tuple(tgt.shape)}")
                printed = True
            sse = ((tgt - recon) ** 2).sum().item()
            var = ((tgt - tgt.mean(0)) ** 2).sum().item()
            per_doc.append((sse, var, tgt.shape[0]))

    sse = np.array([d[0] for d in per_doc])
    var = np.array([d[1] for d in per_doc])
    ve = 1.0 - sse.sum() / var.sum()
    rng = np.random.default_rng(0)
    n = len(per_doc)
    boot = [1.0 - sse[i].sum() / var[i].sum() for i in (rng.integers(0, n, n) for _ in range(1000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {
        "run_name": run_name, "kind": kind,
        "target": "resid_post" if kind == "sae" else "mlp_out",
        "variance_explained": round(float(ve), 4),
        "ve_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "k_L0": int(getattr(coder.cfg, "k", 0)), "n_docs": n,
    }
    print("RECON RESULT:", out)
    return out


def _auto_interp_impl(run_name: str, layer: int = 12, max_latents: int = 100,
                      scorer_model: str = "Qwen/Qwen2.5-3B-Instruct",
                      max_memory: float = 0.5) -> dict:
    """Shared delphi auto-interp body for a CUSTOM sparsify-trained dict (sparsify branch, local ckpt).

    GPU-agnostic core called by the L4 wrapper :func:`auto_interp_custom` (3B scorer) and the
    A100-40GB wrapper :func:`auto_interp_custom_a100` (stronger 7B scorer). delphi routes to its
    gemma-scope loader if 'gemma' appears in sparse_model, so the checkpoint is copied to a gemma-free
    path first. Same LOCAL scorer + seed + sample as repro-004 so the SAE and transcoder runs are
    paired (max_latents + seed fixed => same latent selection per coder). The scorer is configurable
    and a non-default scorer writes autointerp_<tag>_<scorer_tag>.json so it does NOT clobber the 3B
    results (C1: local scorer only).

    GPU SIZING (resolved 2026-06-23). delphi runs caching (the Gemma-2-2B base model) and scoring
    (a vLLM scorer) in ONE process and does NOT free the base model from the GPU between phases, so
    at vLLM startup only (total - ~6 GiB base) is free for the scorer. Consequences:
      * 3B scorer (~6 GiB weights): coexists with the resident base model on L4 (24 GiB) ->
        :func:`auto_interp_custom` on gpu='L4', max_memory=0.5. This is the historical ai-g2 path.
      * 7B scorer (~14.3 GiB weights + KV cache + CUDA graphs, ~16-18 GiB): does NOT fit on L4 next
        to the resident base model (only ~16/22 GiB free; the earlier ai-g2-7b-ATTEMPT failed at
        vLLM startup, not a runtime OOM). It DOES fit on an A100-40GB: ~6 GiB base + ~16-18 GiB
        scorer of 40 GiB, so the vLLM startup guard passes at gpu_memory_utilization ~0.6-0.7
        (~24-28 GiB of the 40 GiB budget; comfortably above the 7B's needs, below 40 minus base) ->
        :func:`auto_interp_custom_a100` on gpu='A100-40GB', max_memory=0.65."""
    import asyncio
    import glob
    import os
    import re
    import shutil

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from pathlib import Path

    from delphi.__main__ import run
    from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # Copy to a gemma-FREE path so delphi uses the sparsify loader, not the gemma-scope loader.
    safe_src = f"/root/outputs/coders/{run_name}"
    tag = "sae" if run_name.endswith("-sae") else "tc"
    # Scorer tag keeps multiple-scorer results from clobbering each other on the volume. The default
    # 3B scorer writes the historical autointerp_<tag>.json (backward-compatible); any non-default
    # scorer (e.g. the stronger 7B) writes autointerp_<tag>_<scorer_tag>.json so both coexist.
    _scorer_short = scorer_model.rsplit("/", 1)[-1].lower()
    scorer_tag = "3b" if "3b" in _scorer_short else ("7b" if "7b" in _scorer_short else _scorer_short)
    out_suffix = "" if scorer_model == "Qwen/Qwen2.5-3B-Instruct" else f"_{scorer_tag}"
    safe_dst = f"/root/coder_{tag}"
    if not os.path.isdir(safe_dst):
        shutil.copytree(safe_src, safe_dst)

    # delphi loads the sparsify coder on CPU by default while the model runs on CUDA -> device
    # mismatch inside encode (x - b_dec). Wrap the loader delphi calls to force the coder onto CUDA.
    import sparsify
    import torch as _torch

    def _force_cuda(orig):
        def _w(*a, **k):
            k["device"] = "cuda"  # FORCE onto cuda + bf16 to match the bf16 model activations delphi feeds
            r = orig(*a, **k)
            return r.to("cuda", _torch.bfloat16) if hasattr(r, "to") else r
        return _w

    for _name in ("load_from_disk", "load_many"):
        _o = getattr(sparsify.SparseCoder, _name, None)
        if _o is not None:
            setattr(sparsify.SparseCoder, _name, _force_cuda(_o))

    name = f"ai-{tag}"
    run_cfg = RunConfig(
        name=name,
        model="google/gemma-2-2b",
        sparse_model=safe_dst,
        hookpoints=[f"layers.{layer}"],
        explainer_provider="offline",
        explainer_model=scorer_model,
        explainer_model_max_len=4096,
        explainer="default",
        scorers=["detection", "fuzz"],
        max_latents=max_latents,
        filter_bos=True,
        num_gpus=1,
        # vLLM gpu_memory_utilization for the scorer. The Gemma base model stays resident through
        # scoring (~6 GiB), so the usable budget is (total - ~6 GiB). 0.5 fits the 3B (~6 GiB) on L4.
        # The 7B (~16-18 GiB) needs the A100-40GB wrapper with ~0.65 (see docstring GPU SIZING).
        max_memory=max_memory,
        seed=0,
        verbose=False,
        hf_token=token,
        cache_cfg=CacheConfig(
            dataset_repo="NeelNanda/pile-10k", dataset_split="train", dataset_column="text",
            batch_size=8, cache_ctx_len=256, n_tokens=200_000, n_splits=5,
        ),
        constructor_cfg=ConstructorConfig(
            example_ctx_len=32, min_examples=50, n_non_activating=50, non_activating_source="random",
        ),
        sampler_cfg=SamplerConfig(
            n_examples_train=20, n_examples_test=40, n_quantiles=10,
            train_type="quantiles", test_type="quantiles",
        ),
    )
    asyncio.run(run(run_cfg))

    act_re = re.compile(r'"activating":\s*(true|false)')
    pred_re = re.compile(r'"prediction":\s*(true|false)')
    out: dict = {"run_name": run_name, "tag": tag, "scorer": scorer_model,
                 "scorer_tag": scorer_tag, "seed": 0, "max_memory": max_memory, "scores": {}}
    for scorer in ("detection", "fuzz"):
        sdir = Path.cwd() / "results" / name / "scores" / scorer
        files = glob.glob(str(sdir / "*"))
        per_latent = []
        latent_ids = []
        for fp in files:
            try:
                text = open(fp).read()
            except OSError:
                continue
            a = act_re.findall(text)
            p = pred_re.findall(text)
            n = min(len(a), len(p))
            if n:
                per_latent.append(sum(1 for x, y in zip(a[:n], p[:n]) if x == y) / n)
                latent_ids.append(os.path.basename(fp))
        out["scores"][scorer] = {
            "n_latents": len(per_latent),
            "mean_accuracy": round(sum(per_latent) / len(per_latent), 4) if per_latent else None,
            "per_latent": [round(v, 4) for v in per_latent],
        }
    out["latent_sample"] = latent_ids[:5]  # a few scored-latent filenames, for reference only
    import json as _json
    with open(f"/root/outputs/autointerp_{tag}{out_suffix}.json", "w") as _fh:
        _json.dump(out, _fh)
    artifacts_vol.commit()
    print("AUTO-INTERP CUSTOM RESULT:", {k: (v if k != "scores" else {s: {kk: vv for kk, vv in d.items() if kk != 'per_latent'} for s, d in v.items()}) for k, v in out.items()})
    return out


@app.function(
    image=pkg_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=5400, retries=0,
)
def auto_interp_custom(run_name: str, layer: int = 12, max_latents: int = 100,
                       scorer_model: str = "Qwen/Qwen2.5-3B-Instruct",
                       max_memory: float = 0.5) -> dict:
    """L4 wrapper (the historical ai-g2 path): delphi auto-interp with the 3B local scorer.

    The 3B scorer (~6 GiB) coexists with the resident Gemma base model on the L4's 24 GiB. For the
    stronger 7B scorer use :func:`auto_interp_custom_a100` (40 GiB). See :func:`_auto_interp_impl`."""
    return _auto_interp_impl(run_name, layer, max_latents, scorer_model, max_memory)


@app.function(
    image=pkg_image, gpu="A100-40GB", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=5400, retries=0,
)
def auto_interp_custom_a100(run_name: str, layer: int = 12, max_latents: int = 100,
                            scorer_model: str = "Qwen/Qwen2.5-7B-Instruct",
                            max_memory: float = 0.65) -> dict:
    """A100-40GB wrapper: delphi auto-interp with a STRONGER local scorer (default Qwen2.5-7B).

    Resolves the ai-g2-7b-ATTEMPT blocker by giving the 7B room to start NEXT TO the resident Gemma
    base model. Budget on 40 GiB: ~6 GiB base + ~16-18 GiB 7B (weights + KV cache + CUDA graphs); at
    max_memory=0.65 vLLM reserves ~26 GiB, which clears its startup guard and the 7B's needs while
    staying under (40 - base). Same delphi config/seed/sample as the L4 3B path so the SAE vs
    transcoder runs stay paired and directly comparable to ai-g2 (only the scorer + GPU change).
    Writes the scorer-tagged autointerp_<tag>_7b.json (no clobber of the 3B json). C1: local scorer
    only (no paid API). See :func:`_auto_interp_impl` for the GPU-sizing rationale."""
    return _auto_interp_impl(run_name, layer, max_latents, scorer_model, max_memory)


@app.local_entrypoint()
def autointerp_main(run_name: str, scorer_model: str = "Qwen/Qwen2.5-7B-Instruct",
                    gpu: str = "a100", max_latents: int = 100, max_memory: float = 0.65,
                    layer: int = 12) -> None:
    """Drive a custom-coder auto-interp run on either GPU (config-driven; default = 7B on A100-40GB).

    modal run infra/modal_app.py::autointerp_main --run-name train_gemma2_2b_l12-sae --gpu a100
    --gpu a100 -> 7B scorer on A100-40GB (the scorer-strength resolution; max_memory ~0.65).
    --gpu l4   -> routes to the L4 3B wrapper (historical ai-g2; pass --scorer-model Qwen/Qwen2.5-3B-Instruct
                  + --max-memory 0.5). max_latents <= 500 (RULES.md C3)."""
    if max_latents > 500:
        raise ValueError(f"max_latents={max_latents} exceeds the C3 cap of 500 (RULES.md).")
    target = gpu.strip().lower()
    if target in ("a100", "a100-40gb", "a100_40gb"):
        fn = auto_interp_custom_a100
    elif target == "l4":
        fn = auto_interp_custom
    else:
        raise ValueError(f"Unknown gpu={gpu!r}; use 'a100' (A100-40GB, 7B scorer) or 'l4' (3B scorer).")
    result = fn.remote(run_name=run_name, layer=layer, max_latents=max_latents,
                       scorer_model=scorer_model, max_memory=max_memory)
    print("FINAL AUTO-INTERP RESULT:", result)


@app.function(
    image=pkg_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=900, retries=0,
)
def probe_coder_fvu(layer: int = 12, n_docs: int = 16) -> dict:
    """E4: does sparsify ForwardOutput carry .fvu? Find the transcoder's correct input (mlp_in vs resid)."""
    import os
    import statistics as st

    import sparsify
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dev = "cuda"
    model = AutoModel.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, token=token).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{n_docs}]")
    texts = [t for t in ds["text"] if t and t.strip()]
    base = "/root/outputs/coders"
    sae = sparsify.SparseCoder.load_from_disk(f"{base}/train_gemma2_2b_l12-sae/layers.{layer}", device=dev)
    tc = sparsify.SparseCoder.load_from_disk(f"{base}/train_gemma2_2b_l12-transcoder/layers.{layer}", device=dev)
    lm = model.layers[layer]
    cap: dict = {}
    tt = lambda o: o[0] if isinstance(o, tuple) else o
    out: dict = {}
    sae_f, tc_mlpin, tc_resid = [], [], []

    def fvu(coder, x):
        o = coder(x.to(torch.bfloat16))
        v = getattr(o, "fvu", None)
        return float(v) if v is not None else float("nan")

    with torch.no_grad():
        for text in texts:
            cap.clear()
            ids = tok(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(dev)
            h0 = lm.register_forward_hook(lambda m, i, o: cap.__setitem__("resid", tt(o)))
            h1 = lm.mlp.register_forward_pre_hook(lambda m, i: cap.__setitem__("mlp_in", i[0]))
            model(ids)
            h0.remove()
            h1.remove()
            resid = cap["resid"][0, 1:]
            mlp_in = cap["mlp_in"][0, 1:]
            if not out:
                o = sae(resid.to(torch.bfloat16))
                out["ForwardOutput_fields"] = ", ".join(a for a in dir(o) if not a.startswith("_"))[:300]
            sae_f.append(fvu(sae, resid))
            tc_mlpin.append(fvu(tc, mlp_in))
            tc_resid.append(fvu(tc, resid))
    out["sae_fvu(resid)"] = round(st.mean(sae_f), 4)
    out["tc_fvu(mlp_in)"] = round(st.mean(tc_mlpin), 4)
    out["tc_fvu(resid)"] = round(st.mean(tc_resid), 4)
    print("CODER FVU PROBE:", out)
    return out


@app.function(
    image=full_image, secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=600, retries=0,
)
def probe_saebench_adapter() -> dict:
    """E4 (Phase 4 dep): requirements to wrap a sparsify SAE as a sae_lens SAE SAEBench accepts."""
    import dataclasses
    import inspect

    import sparsify

    out: dict = {}
    sae = sparsify.SparseCoder.load_from_disk(
        "/root/outputs/coders/train_gemma2_2b_l12-sae/layers.12", device="cpu"
    )
    sd = sae.state_dict()
    out["sparsify_state_keys"] = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sd.items())[:400]
    try:
        out["sparsify_cfg"] = ", ".join(f"{k}={v}" for k, v in vars(sae.cfg).items())[:300]
    except Exception as exc:  # noqa: BLE001
        out["sparsify_cfg"] = f"n/a: {exc}"

    try:
        import sae_lens

        out["sae_lens.version"] = getattr(sae_lens, "__version__", "?")
        out["sae_lens.public"] = ", ".join(n for n in dir(sae_lens) if not n.startswith("_"))[:300]
        for cfgname in ("SAEConfig", "SAEConfigLoadOptions"):
            obj = getattr(sae_lens, cfgname, None)
            if obj is not None and dataclasses.is_dataclass(obj):
                out[f"{cfgname}_fields"] = ", ".join(f.name for f in dataclasses.fields(obj))[:400]
    except Exception as exc:  # noqa: BLE001
        out["sae_lens"] = f"FAIL: {exc}"

    try:
        from sae_bench.sae_bench_utils import general_utils

        out["load_and_format_sae_src"] = inspect.getsource(general_utils.load_and_format_sae)[:1200]
    except Exception as exc:  # noqa: BLE001
        out["load_and_format_sae_src"] = f"FAIL: {exc}"

    for k, v in out.items():
        print(f"=== {k} ===\n{v}\n")
    return out


@app.function(
    image=full_image, secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=600, retries=0,
)
def probe_saebench_adapter2() -> dict:
    """E4 (Phase 4 dep, deeper): pin the exact sae_lens inference TopK SAE class + config + constructor,
    SAEBench's check_decoder_norms / _standardize_sae_cfg requirements, and which sae.cfg/metadata
    fields sparse_probing reads. Everything needed to construct the adapter without guessing."""
    import dataclasses
    import inspect

    import sae_lens

    out: dict = {}
    # Full public class list (the first probe truncated at 300 chars).
    out["sae_lens.public_full"] = ", ".join(n for n in dir(sae_lens) if not n.startswith("_"))

    # Inference (non-Training) SAE classes + their config classes + constructor signatures.
    for name in (
        "SAE", "StandardSAE", "TopKSAE", "TopK", "BatchTopKSAE", "JumpReLUSAE", "GatedSAE",
        "StandardSAEConfig", "TopKSAEConfig", "BatchTopKSAEConfig",
    ):
        obj = getattr(sae_lens, name, None)
        if obj is None:
            continue
        if dataclasses.is_dataclass(obj):
            fields = []
            for f in dataclasses.fields(obj):
                d = "" if f.default is dataclasses.MISSING else f"={f.default!r}"
                fields.append(f"{f.name}:{getattr(f.type, '__name__', f.type)}{d}")
            out[f"{name}<dataclass fields>"] = ", ".join(fields)[:600]
        else:
            try:
                out[f"{name}.__init__"] = f"{name}{inspect.signature(obj.__init__)}"[:300]
            except (ValueError, TypeError):
                out[f"{name}"] = "class (no init sig)"
            # Does this class expose a topk architecture marker / get_sae_config helper?
            for attr in ("architecture", "get_sae_config_class", "cfg_class"):
                if hasattr(obj, attr):
                    out[f"{name}.{attr}"] = str(getattr(obj, attr))[:120]

    # The base SAEConfig + how to build a TopK metadata; metadata class fields.
    for cfgname in ("SAEConfig", "SAEMetadata"):
        obj = getattr(sae_lens, cfgname, None)
        if obj is not None and dataclasses.is_dataclass(obj):
            out[f"{cfgname}_fields"] = ", ".join(f.name for f in dataclasses.fields(obj))[:500]

    # SAEBench: check_decoder_norms + _standardize_sae_cfg source (what cfg shape they require).
    try:
        from sae_bench.sae_bench_utils import general_utils

        for fn in ("check_decoder_norms", "_standardize_sae_cfg"):
            f = getattr(general_utils, fn, None)
            if f is not None:
                out[f"src::{fn}"] = inspect.getsource(f)[:900]
    except Exception as exc:  # noqa: BLE001
        out["general_utils"] = f"FAIL: {exc}"

    # sparse_probing main: how does it pull activations from the SAE? (encode? cfg.metadata.hook_name?)
    try:
        from sae_bench.evals.sparse_probing import main as sp

        src = inspect.getsource(sp)
        for needle in ("hook_name", "hook_layer", ".encode(", "load_and_format_sae", "metadata"):
            idx = src.find(needle)
            out[f"sp_uses::{needle}"] = (
                "NOT FOUND" if idx < 0 else src[max(0, idx - 80): idx + 80].replace("\n", " ")
            )
    except Exception as exc:  # noqa: BLE001
        out["sparse_probing.main"] = f"FAIL: {exc}"

    for k, v in out.items():
        print(f"=== {k} ===\n{v}\n")
    return out


@app.function(
    image=full_image, secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=600, retries=0,
)
def probe_saebench_adapter3() -> dict:
    """E4 final: cfg<->metadata duality (does sae.cfg.hook_name exist as a top-level property?),
    SAEMetadata fields, how SAE.encode behaves for TopK, and how SAEBench extracts activations
    (does it call sae.encode, or run the model at hook_name and then sae.encode_standard?)."""
    import dataclasses
    import inspect

    import sae_lens
    import torch

    out: dict = {}

    # Where does SAEMetadata live + its fields?
    from sae_lens.saes.sae import SAEMetadata  # v6 location

    out["SAEMetadata_module"] = SAEMetadata.__module__
    if dataclasses.is_dataclass(SAEMetadata):
        out["SAEMetadata_fields"] = ", ".join(f.name for f in dataclasses.fields(SAEMetadata))[:600]
    # Does SAEConfig expose hook_name/hook_layer/model_name as top-level attrs (properties)?
    cfgcls = sae_lens.SAEConfig
    out["SAEConfig_props"] = ", ".join(
        n for n in dir(cfgcls) if not n.startswith("_") and isinstance(getattr(cfgcls, n, None), property)
    )[:300]

    # Full _standardize_sae_cfg source (was truncated): does it WRITE hook_name/hook_layer onto cfg?
    try:
        from sae_bench.sae_bench_utils import general_utils

        out["full_standardize_src"] = inspect.getsource(general_utils._standardize_sae_cfg)[:2000]
        out["get_cfg_meta_field_src"] = inspect.getsource(general_utils._get_cfg_meta_field)[:800]
    except Exception as exc:  # noqa: BLE001
        out["standardize_src"] = f"FAIL: {exc}"

    # Build a tiny real TopKSAE and check: does cfg.hook_name / cfg.hook_layer resolve from metadata?
    try:
        TopKSAEConfig = sae_lens.TopKSAEConfig
        TopKSAE = sae_lens.TopKSAE
        md = SAEMetadata(
            model_name="gemma-2-2b", hook_name="blocks.12.hook_resid_post", hook_layer=12
        )
        cfg = TopKSAEConfig(d_in=8, d_sae=16, k=4, dtype="float32", device="cpu", metadata=md)
        sae = TopKSAE(cfg)
        # After standardize, does cfg.hook_name become a top-level attr?
        try:
            from sae_bench.sae_bench_utils import general_utils as _gu

            _gu._standardize_sae_cfg(cfg)
            out["post_standardize.cfg.hook_name"] = repr(getattr(cfg, "hook_name", "MISSING"))
            out["post_standardize.cfg.hook_layer"] = repr(getattr(cfg, "hook_layer", "MISSING"))
            out["post_standardize.cfg.model_name"] = repr(getattr(cfg, "model_name", "MISSING"))
        except Exception as exc:  # noqa: BLE001
            out["post_standardize"] = f"FAIL: {type(exc).__name__}: {str(exc)[:200]}"
        out["built_topk"] = type(sae).__name__
        for attr in ("hook_name", "hook_layer", "model_name"):
            out[f"cfg.{attr}"] = repr(getattr(cfg, attr, "MISSING"))
            out[f"cfg.metadata.{attr}"] = repr(getattr(md, attr, "MISSING"))
        # encode/decode round-trip shape + dtype.
        x = torch.randn(3, 8)
        enc = sae.encode(x)
        dec = sae.decode(enc)
        out["encode_shape"] = str(tuple(enc.shape))
        out["decode_shape"] = str(tuple(dec.shape))
        out["encode_nonzeros_per_row"] = str(int((enc[0] != 0).sum()))
        out["W_dec_shape"] = str(tuple(sae.W_dec.shape))
        out["W_dec_row_norms~1"] = str(
            bool(torch.allclose(sae.W_dec.norm(dim=1), torch.ones(16), atol=1e-4))
        )
        # state_dict keys (so we know what to load sparsify weights into).
        out["topk_state_keys"] = ", ".join(
            f"{k}{tuple(v.shape)}" for k, v in sae.state_dict().items()
        )[:400]
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["build_topk"] = f"FAIL: {type(exc).__name__}: {exc}\n{traceback.format_exc()[:500]}"

    # How does SAEBench's sparse_probing get activations? Inspect the activation-collection fn.
    try:
        from sae_bench.evals.sparse_probing import probe_training

        src = inspect.getsource(probe_training)
        for needle in (".encode(", "encode_standard", "run_with_cache", "hook_name", "def "):
            idx = src.find(needle)
            out[f"probe_training::{needle}"] = (
                "NOT FOUND" if idx < 0 else src[max(0, idx - 60): idx + 90].replace("\n", " ")
            )
    except Exception as exc:  # noqa: BLE001
        out["probe_training"] = f"n/a: {type(exc).__name__}: {str(exc)[:120]}"

    for k, v in out.items():
        print(f"=== {k} ===\n{v}\n")
    return out


@app.function(
    image=pkg_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=2400, retries=0,
)
def probing_eval(run_name: str, randomize: bool = False, layer: int = 12,
                 n_per_class: int = 300, seq_len: int = 64) -> dict:
    """Scorer-INDEPENDENT control signal (ADR-0005 Control A primary): linear-probe accuracy of an
    SAE's features on a bias-in-bios profession contrast. Run for the real-model SAE and the
    randomized-model SAE; the GAP (real > random) = the real SAE encodes model-learned structure.

    A coder's features are probed on ITS OWN model's resid@L (real SAE -> real gemma; random SAE ->
    the same seeded randomized gemma). Minimal logistic probe (same code both sides) rather than the
    full SAEBench adapter; the GAP is the signal (repro-003 established the SAEBench reference)."""
    import os

    import numpy as np
    import sparsify
    import torch
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dev = "cuda"
    coder = sparsify.SparseCoder.load_from_disk(
        f"/root/outputs/coders/{run_name}/layers.{layer}", device=dev
    ).to(dev, torch.bfloat16)
    d_sae = int(coder.cfg.num_latents)

    model = AutoModel.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, token=token).to(dev)
    if randomize:  # rebuild the SAME seeded randomized model the random SAE was trained on (ADR-0005)
        torch.manual_seed(0)
        emb = model.get_input_embeddings().weight.data.clone()
        model = AutoModel.from_config(AutoConfig.from_pretrained("google/gemma-2-2b", token=token)).to(
            dev, torch.bfloat16
        )
        model.get_input_embeddings().weight.data.copy_(emb)
    model.eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
    lm = model.layers[layer]

    ds = load_dataset("LabHC/bias_in_bios", split="train")
    cols = ds.column_names
    text_col = "text" if "text" in cols else ("hard_text" if "hard_text" in cols else cols[0])
    label_col = "label" if "label" in cols else ("profession" if "profession" in cols else cols[-1])
    labels_all = ds[label_col]
    from collections import Counter

    top2 = [c for c, _ in Counter(labels_all).most_common(2)]
    rows = {c: [] for c in top2}
    for tx, lb in zip(ds[text_col], labels_all):
        if lb in rows and len(rows[lb]) < n_per_class and tx and tx.strip():
            rows[lb].append(tx)
    texts = rows[top2[0]] + rows[top2[1]]
    y = np.array([0] * len(rows[top2[0]]) + [1] * len(rows[top2[1]]))

    cap: dict = {}

    def dense_feats(x):
        enc = coder.encode(x)
        if torch.is_tensor(enc) and enc.shape[-1] == d_sae:
            return enc.float()
        ta = getattr(enc, "top_acts", None)
        ti = getattr(enc, "top_indices", None)
        if ta is None and isinstance(enc, (tuple, list)):
            ta, ti = enc[0], enc[1]
        out = torch.zeros(x.shape[0], d_sae, device=x.device, dtype=torch.float32)
        out.scatter_(1, ti.long(), ta.float())
        return out

    X = []
    with torch.no_grad():
        for tx in texts:
            ids = tok(tx, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(dev)
            h = lm.register_forward_hook(lambda m, i, o: cap.__setitem__("y", o[0] if isinstance(o, tuple) else o))
            model(ids)
            h.remove()
            xx = cap["y"][0, 1:].to(torch.bfloat16)
            X.append(dense_feats(xx).mean(0).cpu().numpy())
    X = np.array(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float((pred == yte).mean())
    rng = np.random.default_rng(0)
    boot = [float((pred[i] == yte[i]).mean()) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(2000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {
        "run_name": run_name, "randomize": randomize, "classes": [int(c) for c in top2],
        "n_examples": len(texts), "sae_probe_acc": round(acc, 4),
        "acc_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "d_sae": d_sae, "text_col": text_col, "label_col": label_col,
        "yte": [int(v) for v in yte], "pred": [int(v) for v in pred],
    }
    import json as _json

    with open(f"/root/outputs/probing_{'random' if randomize else 'real'}.json", "w") as fh:
        _json.dump(out, fh)
    artifacts_vol.commit()
    print("PROBING RESULT:", out)
    return out


@app.function(
    image=pkg_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=2400, retries=0,
)
def steer_eval(
    layer: int = 12,
    n_per_class: int = 200,
    n_gen: int = 16,
    n_scan: int = 8,
    max_new: int = 32,
    seed: int = 0,
) -> dict:
    """Control B (ADR-0005), RECALIBRATED: SAE-feature steering vs difference-of-means.

    Same pre-registered metric/concept as ADR-0005 (success-rate-under-fluency on the two most-frequent
    bias_in_bios professions, steering toward class top2[1]); this is a CALIBRATION fix only (neutral
    prompt + finer coefficient grid + full per-coef reporting), NOT a metric/concept change, so it is
    not a new Gate-4 decision.

    The first run (ctrl-steer) was degenerate: the "My favorite" prompt already classified as the target
    ~0.81 of the time (a ceiling, no headroom) and the coarse coefs [2,4,8]xresid_rms all broke the
    fluency cap, so the best fluency-preserving coef was 0 for both directions and the head-to-head was
    trivially 0.0. Here we (1) choose the prompt whose baseline success is closest to ~0.5 from a neutral
    candidate set, (2) sweep a finer/smaller grid so a fluency-preserving sweet spot can exist, and
    (3) report success AND perplexity at every coef for both directions plus the steering EFFECT
    (success minus baseline) with a bootstrap CI on the SAE-minus-dom difference.

    Metric (unchanged): fraction of generations the concept probe labels as the target class, at the
    strongest steering coefficient that keeps mean perplexity <= 1.5x the unsteered baseline (fluency
    bound). Head-to-head SAE-feature vs diff-of-means with a bootstrap CI on the success-rate difference.
    """
    import os

    import numpy as np
    import sparsify
    import torch
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # E1 determinism: log + set all seeds (generation uses do_sample=True, so this fixes the sample).
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dev = "cuda"
    sae = sparsify.SparseCoder.load_from_disk(
        f"/root/outputs/coders/train_gemma2_2b_l12-sae/layers.{layer}", device=dev
    ).to(dev, torch.bfloat16)
    d_sae = int(sae.cfg.num_latents)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", torch_dtype=torch.bfloat16, token=token
    ).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
    layers = model.model.layers

    # --- collect labelled resid@L (mean-pooled) for the two professions (concept = class A vs B) ---
    ds = load_dataset("LabHC/bias_in_bios", split="train")
    from collections import Counter
    top2 = [c for c, _ in Counter(ds["profession"]).most_common(2)]
    buf = {c: [] for c in top2}
    cap: dict = {}

    def grab_resid(ids):
        h = layers[layer].register_forward_hook(
            lambda m, i, o: cap.__setitem__("y", (o[0] if isinstance(o, tuple) else o))
        )
        model(ids)
        h.remove()
        return cap["y"][0, 1:].float()

    def dense(x):
        enc = sae.encode(x.to(torch.bfloat16))
        if torch.is_tensor(enc) and enc.shape[-1] == d_sae:
            return enc.float()
        ta, ti = getattr(enc, "top_acts", enc[0]), getattr(enc, "top_indices", enc[1])
        out = torch.zeros(x.shape[0], d_sae, device=x.device); out.scatter_(1, ti.long(), ta.float())
        return out

    feats, raws, ys = [], [], []
    with torch.no_grad():
        for tx, lb in zip(ds["hard_text"], ds["profession"]):
            if lb in buf and len(buf[lb]) < n_per_class and tx and tx.strip():
                buf[lb].append(1)
                ids = tok(tx, return_tensors="pt", truncation=True, max_length=64).input_ids.to(dev)
                r = grab_resid(ids)
                feats.append(dense(r).mean(0).cpu().numpy())
                raws.append(r.mean(0).cpu().numpy())
                ys.append(0 if lb == top2[0] else 1)
    feats, raws, ys = np.array(feats), np.array(raws), np.array(ys)
    target = 1  # steer toward class top2[1]
    probe = LogisticRegression(max_iter=2000, C=0.5).fit(feats, ys)

    # --- directions: SAE feature most predictive of target, and difference-of-means ---
    top_feat = int(np.argmax(probe.coef_[0]))  # SAE latent most positively tied to target
    sae_dir = sae.W_dec[top_feat].float().detach()
    sae_dir = sae_dir / sae_dir.norm()
    dom = torch.tensor(raws[ys == 1].mean(0) - raws[ys == 0].mean(0), device=dev, dtype=torch.float32)
    dom = dom / dom.norm()
    resid_rms = float(np.linalg.norm(raws, axis=1).mean())

    def gen_with(prompt_ids, direction, coef, n=None):
        """Generate n continuations of prompt_ids, optionally steered by coef*resid_rms*direction."""
        n = n_gen if n is None else n
        vec = (coef * resid_rms) * direction.to(torch.bfloat16)
        def hook(m, i, o):
            if isinstance(o, tuple):
                return (o[0] + vec, *o[1:])
            return o + vec
        outs = []
        handle = None if coef == 0 else layers[layer].register_forward_hook(hook)
        try:
            for _ in range(n):
                g = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=True,
                                   temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
                outs.append(g)
        finally:
            if handle is not None:
                handle.remove()
        return outs

    def evaluate(gens):
        succ, ppls = [], []
        with torch.no_grad():
            for g in gens:
                txt_ids = g
                # perplexity (unsteered model loss on the full sequence)
                loss = model(txt_ids, labels=txt_ids).loss
                ppls.append(float(torch.exp(loss)))
                # concept class via probe on resid@L features
                r = grab_resid(txt_ids)
                pc = int(probe.predict(dense(r).mean(0).cpu().numpy()[None])[0])
                succ.append(1 if pc == target else 0)
        return np.array(succ), np.array(ppls)

    # --- (1) NEUTRAL-PROMPT selection: pick the candidate whose unsteered baseline success is closest
    # to ~0.5 (no lean toward the target class) so the sweep has headroom. The old run used the single
    # prompt "My favorite", whose baseline was 0.81 (a ceiling). We print every candidate's baseline. ---
    candidate_prompts = ["I", "The", "This person", "They said", "Yesterday", "My favorite"]
    prompt_scan = []
    for p in candidate_prompts:
        pid = tok(p, return_tensors="pt").input_ids.to(dev)
        s, pp = evaluate(gen_with(pid, sae_dir, 0.0, n=n_scan))
        prompt_scan.append({"prompt": p, "baseline_success": round(float(s.mean()), 3),
                            "baseline_ppl": round(float(pp.mean()), 2), "n_scan": n_scan})
        print(f"PROMPT-SCAN {p!r}: baseline_success={s.mean():.3f} baseline_ppl={pp.mean():.2f}")
    # closest to 0.5 = most neutral (no lean toward the target class); the chosen prompt's baseline is
    # re-measured below at the full n_gen so the head-to-head uses a full-precision baseline.
    chosen = min(prompt_scan, key=lambda r: abs(r["baseline_success"] - 0.5))
    prompt = chosen["prompt"]
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    print(f"CHOSEN NEUTRAL PROMPT: {prompt!r} (scan baseline_success={chosen['baseline_success']})")

    base_succ, base_ppl = evaluate(gen_with(prompt_ids, sae_dir, 0.0))
    base_ppl_mean = float(base_ppl.mean())
    base_succ_mean = float(base_succ.mean())
    fluency_cap = 1.5 * base_ppl_mean

    results = {
        "seed": seed,
        "chosen_prompt": prompt,
        "prompt_scan": prompt_scan,
        "n_gen": n_gen,
        "baseline_ppl": round(base_ppl_mean, 2),
        "baseline_success": round(base_succ_mean, 3),
        "fluency_cap_ppl": round(fluency_cap, 2),
        "target_class": int(top2[target]),
        "top_feature": top_feat,
        "resid_rms": round(resid_rms, 1),
        "coef_grid": [],
        "by_direction": {},
    }

    # --- (2) FINER, smaller coefficient grid so a fluency-preserving sweet spot can exist. The old grid
    # [2,4,8] all blew the fluency cap. (3) Report success AND perplexity at EVERY coef for BOTH
    # directions, and pick each direction's best FLUENCY-PRESERVING coef (max success s.t. ppl<=cap). ---
    coefs = [0.5, 1.0, 2.0, 3.0, 4.0]
    results["coef_grid"] = coefs
    for name, d in (("sae_feature", sae_dir), ("difference_of_means", dom)):
        # full sweep, coef 0 (baseline) included for a complete success-vs-coef curve
        sweep = [{"coef": 0.0, "success": round(base_succ_mean, 3), "ppl": round(base_ppl_mean, 2),
                  "fluent": True, "succ_arr": base_succ.tolist()}]
        for c in coefs:
            s, pp = evaluate(gen_with(prompt_ids, d, c))
            fluent = bool(pp.mean() <= fluency_cap)
            sweep.append({"coef": c, "success": round(float(s.mean()), 3), "ppl": round(float(pp.mean()), 2),
                          "fluent": fluent, "succ_arr": s.tolist()})
            print(f"SWEEP {name} coef={c}: success={s.mean():.3f} ppl={pp.mean():.2f} fluent={fluent}")
        # best fluency-preserving coef = highest success among fluent points (ties -> smaller coef)
        fluent_pts = [pt for pt in sweep if pt["fluent"]]
        best = max(fluent_pts, key=lambda pt: (pt["success"], -pt["coef"]))
        results["by_direction"][name] = {
            "sweep": [{k: v for k, v in pt.items() if k != "succ_arr"} for pt in sweep],
            "best_coef": best["coef"],
            "best_success": best["success"],
            "best_ppl": best["ppl"],
            # steering EFFECT at the best fluency-preserving coef = success minus the unsteered baseline
            "steering_effect": round(best["success"] - base_succ_mean, 3),
            "_best_succ_arr": best["succ_arr"],
        }

    # --- (3) head-to-head at each direction's best fluency-preserving coef + bootstrap 95% CI on the
    # SAE-minus-dom success difference. Honest label (R4): if dom matches/beats SAE, say so plainly. ---
    a = np.array(results["by_direction"]["sae_feature"].pop("_best_succ_arr"), float)
    b = np.array(results["by_direction"]["difference_of_means"].pop("_best_succ_arr"), float)
    rng = np.random.default_rng(seed)
    boot = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sae_eff = results["by_direction"]["sae_feature"]["steering_effect"]
    dom_eff = results["by_direction"]["difference_of_means"]["steering_effect"]
    results["sae_minus_dom"] = round(float(a.mean() - b.mean()), 3)
    results["sae_minus_dom_ci95"] = [round(float(lo), 3), round(float(hi), 3)]
    if lo > 0:
        verdict = "SAE feature steers better than difference-of-means (CI excludes 0)"
    elif hi < 0:
        verdict = "difference-of-means matches/beats SAE feature (CI excludes 0)"
    else:
        verdict = "inconclusive: no significant SAE-vs-baseline difference (CI includes 0)"
    results["verdict"] = verdict
    results["effect_note"] = (
        f"steering_effect SAE={sae_eff:+.3f} dom={dom_eff:+.3f} (success minus baseline {base_succ_mean:.3f})"
    )

    import json as _json
    with open("/root/outputs/steering.json", "w") as fh:
        _json.dump(results, fh)
    artifacts_vol.commit()
    print("STEER RESULT:", {k: v for k, v in results.items() if k not in ("by_direction", "prompt_scan")})
    print("BY DIR:", {n: {k: v for k, v in d.items() if k != "sweep"} for n, d in results["by_direction"].items()})
    return results


@app.function(
    image=pkg_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=2400, retries=0,
)
def circuit_eval(run_name: str = "train_gemma2_2b_l12-sae", layer: int = 12,
                 n_per_class: int = 300, seq_len: int = 64) -> dict:
    """Phase 5 (ADR-0006): a single-layer SAE-feature circuit for the bias-in-bios profession behavior.

    Attribution = per-feature |mean_act(class1) - mean_act(class0)| (probe-independent). Circuit = top-K.
    Faithfulness = fresh probe on ONLY top-K (sufficiency) vs fresh probe on K RANDOM features (control)
    vs full-feature ceiling, with a paired bootstrap CI on the top-K - random-K gap (R2/R3)."""
    import os

    import numpy as np
    import sparsify
    import torch
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from transformers import AutoModel, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dev = "cuda"
    sae = sparsify.SparseCoder.load_from_disk(
        f"/root/outputs/coders/{run_name}/layers.{layer}", device=dev
    ).to(dev, torch.bfloat16)
    d_sae = int(sae.cfg.num_latents)
    model = AutoModel.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, token=token).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=token)
    lm = model.layers[layer]

    ds = load_dataset("LabHC/bias_in_bios", split="train")
    from collections import Counter
    top2 = [c for c, _ in Counter(ds["profession"]).most_common(2)]
    buf = {c: [] for c in top2}
    cap: dict = {}

    def dense(x):
        enc = sae.encode(x.to(torch.bfloat16))
        if torch.is_tensor(enc) and enc.shape[-1] == d_sae:
            return enc.float()
        ta, ti = getattr(enc, "top_acts", enc[0]), getattr(enc, "top_indices", enc[1])
        out = torch.zeros(x.shape[0], d_sae, device=x.device); out.scatter_(1, ti.long(), ta.float())
        return out

    X, y = [], []
    with torch.no_grad():
        for tx, lb in zip(ds["hard_text"], ds["profession"]):
            if lb in buf and len(buf[lb]) < n_per_class and tx and tx.strip():
                buf[lb].append(1)
                ids = tok(tx, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(dev)
                h = lm.register_forward_hook(lambda m, i, o: cap.__setitem__("y", o[0] if isinstance(o, tuple) else o))
                model(ids); h.remove()
                X.append(dense(cap["y"][0, 1:].to(torch.bfloat16)).mean(0).cpu().numpy())
                y.append(0 if lb == top2[0] else 1)
    X, y = np.array(X), np.array(y)

    # attribution (probe-independent): per-feature class-mean activation difference
    attr = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0))
    order = np.argsort(-attr)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    def probe_acc(cols):
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr[:, cols], ytr)
        return (clf.predict(Xte[:, cols]) == yte).astype(float)

    full_correct = probe_acc(np.arange(d_sae))
    ceiling = float(full_correct.mean())
    rng = np.random.default_rng(0)
    per_k = []
    for K in (5, 10, 20, 50):
        top = order[:K]
        rnd = rng.choice(d_sae, K, replace=False)
        ct, cr = probe_acc(top), probe_acc(rnd)
        n = len(yte)
        boot = np.array([ct[i].mean() - cr[j].mean()
                         for i, j in ((rng.integers(0, n, n), rng.integers(0, n, n)) for _ in range(5000))])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        per_k.append({
            "K": K, "acc_topK": round(float(ct.mean()), 4), "acc_randomK": round(float(cr.mean()), 4),
            "faithfulness_vs_ceiling": round(float(ct.mean() / ceiling), 3),
            "gap_topK_minus_randomK": round(float(ct.mean() - cr.mean()), 4),
            "gap_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "circuit_beats_random": bool(lo > 0),
        })
    out = {
        "run_name": run_name, "behavior": f"bias_in_bios professions {top2}",
        "d_sae": d_sae, "n_examples": len(y), "ceiling_acc_full": round(ceiling, 4),
        "top10_feature_ids": [int(i) for i in order[:10]],
        "top10_attribution": [round(float(attr[i]), 4) for i in order[:10]],
        "by_K": per_k,
    }
    import json as _json
    with open("/root/outputs/circuit.json", "w") as fh:
        _json.dump(out, fh)
    artifacts_vol.commit()
    print("CIRCUIT RESULT:", {k: v for k, v in out.items() if k != "top10_attribution"})
    return out


def _sparsify_to_topk_sae(coder_dir: str, layer: int, device: str, dtype: str = "float32"):
    """Adapter: load an EleutherAI sparsify TopK SparseCoder and return a native sae_lens TopKSAE
    holding the same weights, so SAEBench's load_and_format_sae accepts it as a custom SAE object.

    Verified on Modal (E4: probe_saebench_adapter / _adapter2 / _adapter3, sae_lens 6.44.3):
      * sparsify state: W_dec(d_sae,d_in), b_dec(d_in,), encoder.weight(d_sae,d_in),
        encoder.bias(d_sae,)
      * sae_lens TopKSAE state: W_enc(d_in,d_sae), b_enc(d_sae,), W_dec(d_sae,d_in), b_dec(d_in,)
        => W_enc = encoder.weight.T ; b_enc = encoder.bias ; W_dec/b_dec copy across.
      * TopKSAEConfig carries k; cfg.hook_name/hook_layer/model_name resolve from cfg.metadata
        (SAEMetadata at sae_lens.saes.sae) -- exactly the fields sparse_probing reads.
    Returns (TopKSAE in eval mode on `device`, info dict)."""
    import sparsify
    import torch
    from sae_lens import TopKSAE, TopKSAEConfig
    from sae_lens.saes.sae import SAEMetadata

    coder = sparsify.SparseCoder.load_from_disk(coder_dir, device=device)
    if bool(getattr(coder.cfg, "transcode", False)) or bool(
        getattr(coder.cfg, "skip_connection", False)
    ):
        raise ValueError(
            "sparse_probing adapter is for residual SAEs only; this coder is a transcoder/skip "
            f"(transcode={coder.cfg.transcode}, skip={coder.cfg.skip_connection}). Mark N/A (R3)."
        )
    sd = coder.state_dict()
    d_sae = int(coder.cfg.num_latents)
    k = int(coder.cfg.k)
    d_in = int(sd["W_dec"].shape[1])

    torch_dtype = getattr(torch, dtype)
    cfg = TopKSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        dtype=dtype,
        device=device,
        apply_b_dec_to_input=False,  # sparsify TopK does not subtract b_dec before encode
        normalize_activations="none",
        metadata=SAEMetadata(
            model_name="gemma-2-2b",
            hook_name=f"blocks.{layer}.hook_resid_post",
            hook_layer=layer,
        ),
    )
    sae = TopKSAE(cfg)
    new_state = {
        "W_enc": sd["encoder.weight"].T.contiguous().to(torch_dtype),
        "b_enc": sd["encoder.bias"].to(torch_dtype),
        "W_dec": sd["W_dec"].to(torch_dtype),
        "b_dec": sd["b_dec"].to(torch_dtype),
    }
    missing, unexpected = sae.load_state_dict(new_state, strict=False)
    # TopKSAE may carry extra non-persistent buffers (e.g. threshold); enc/dec weights must load.
    if {"W_enc", "b_enc", "W_dec", "b_dec"} & set(missing):
        raise RuntimeError(f"adapter failed to load encoder/decoder weights: missing={missing}")
    sae = sae.to(device=device, dtype=torch_dtype).eval()
    return sae, {
        "d_in": d_in, "d_sae": d_sae, "k": k,
        "missing": list(missing), "unexpected": list(unexpected),
    }


@app.function(
    image=full_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=900, retries=0,
)
def verify_saebench_adapter(run_name: str = "train_gemma2_2b_l12-sae", layer: int = 12) -> dict:
    """E4 pre-flight (cheap, ~1-2 min GPU): build the sparsify->sae_lens TopKSAE adapter and prove
    it (a) loads weights, (b) encode/decode round-trips with exactly k active latents, (c) passes
    SAEBench's load_and_format_sae (check_decoder_norms + _standardize_sae_cfg) WITHOUT error,
    before paying for the full sparse_probing run. Verifies the adapter, not the eval."""
    import torch

    dev = "cuda"
    coder_dir = f"/root/outputs/coders/{run_name}/layers.{layer}"
    sae, info = _sparsify_to_topk_sae(coder_dir, layer, dev)
    out: dict = {"run_name": run_name, **info, "sae_type": type(sae).__name__}

    # (b) encode/decode round-trip on random activations of the right width.
    x = torch.randn(4, info["d_in"], device=dev, dtype=sae.W_dec.dtype)
    enc = sae.encode(x)
    dec = sae.decode(enc)
    out["encode_shape"] = str(tuple(enc.shape))
    out["decode_shape"] = str(tuple(dec.shape))
    out["active_latents_per_row"] = [int((enc[i] != 0).sum()) for i in range(enc.shape[0])]
    out["k_enforced"] = bool(all(n == info["k"] for n in out["active_latents_per_row"]))
    out["W_dec_row_norm_mean"] = round(float(sae.W_dec.norm(dim=1).mean()), 4)

    # (c) does SAEBench accept this object? load_and_format_sae runs check_decoder_norms + standard.
    from sae_bench.sae_bench_utils import general_utils

    sae_id, formatted, _sparsity = general_utils.load_and_format_sae("custom_sae", sae, dev)
    out["saebench_accepts"] = True
    out["saebench_sae_id"] = sae_id
    out["cfg.hook_name"] = str(getattr(formatted.cfg, "hook_name", "MISSING"))
    out["cfg.hook_layer"] = str(getattr(formatted.cfg, "hook_layer", "MISSING"))
    out["cfg.model_name"] = str(getattr(formatted.cfg, "model_name", "MISSING"))
    out["decoder_norms_ok"] = bool(general_utils.check_decoder_norms(formatted.W_dec.data))

    print("ADAPTER VERIFY:", out)
    return out


@app.function(
    image=full_image, gpu="L4", secrets=[HF_SECRET],
    volumes={**CACHE, "/root/outputs": artifacts_vol}, timeout=3600, retries=0,
)
def saebench_sparse_probing_custom(
    run_name: str = "train_gemma2_2b_l12-sae",
    layer: int = 12,
    train_size: int = 1500,
    test_size: int = 500,
    seed: int = 42,
) -> dict:
    """Phase-3 (deferred item): full SAEBench sparse_probing on the CUSTOM sparsify SAE, via the
    sparsify->sae_lens TopKSAE adapter (_sparsify_to_topk_sae). Same eval config as repro-003 (the
    canonical Gemma Scope reference: SAE top-1 0.767 vs residual baseline 0.688), so the numbers are
    directly comparable. SAEBench is residual-SAE oriented; the transcoder is N/A (R3) -- see docs.

    E1: seed set+logged. E4: API verified on Modal (probe_saebench_adapter*). Budget ~$0.5 L4."""
    import glob
    import json
    import os
    import random

    import numpy as np
    import torch
    from sae_bench.evals.sparse_probing import main as sp

    # E1 determinism: set + log all seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = "cuda"
    coder_dir = f"/root/outputs/coders/{run_name}/layers.{layer}"
    sae, info = _sparsify_to_topk_sae(coder_dir, layer, dev)
    # SAEBench custom path: selected_saes = [(unique_id, sae_object)]; the object goes through
    # load_and_format_sae (verified: check_decoder_norms warns-only, _standardize_sae_cfg reads
    # hook_name/hook_layer/model_name from cfg.metadata).
    selected_saes = [(f"custom-{run_name}", sae)]

    config = sp.SparseProbingEvalConfig(
        model_name="gemma-2-2b",
        random_seed=seed,
        llm_batch_size=32,
        llm_dtype="bfloat16",
        # Same single dataset as repro-003 (its EXPERIMENTS row = bias_in_bios_class_set1). The
        # SAEBench build keys this dataset as '..._class_set1' (E4: probe_saebench_datasets); the
        # bare 'LabHC/bias_in_bios' KeyErrors in chosen_classes_per_dataset.
        dataset_names=["LabHC/bias_in_bios_class_set1"],
        probe_train_set_size=train_size,
        probe_test_set_size=test_size,
        context_length=128,
        k_values=[1],
    )
    out_path = "eval_results/sparse_probing_custom"
    sp.run_eval(
        config, selected_saes, dev, out_path,
        force_rerun=True, clean_up_activations=True, save_activations=False,
    )

    files = glob.glob(os.path.join(out_path, "*_eval_results.json"))
    if not files:
        raise RuntimeError(f"no SAEBench result json written to {out_path}")
    res = json.load(open(files[0]))
    metrics = res.get("eval_result_metrics", {})
    sae_m = metrics.get("sae", {})
    llm_m = metrics.get("llm", {})
    out = {
        "run_name": run_name,
        "adapter": "sparsify->sae_lens.TopKSAE",
        "d_in": info["d_in"], "d_sae": info["d_sae"], "k": info["k"],
        "seed": seed,
        "dataset": "LabHC/bias_in_bios",
        "sae_top_1_test_accuracy": sae_m.get("sae_top_1_test_accuracy"),
        "sae_test_accuracy": sae_m.get("sae_test_accuracy"),
        "llm_top_1_test_accuracy": llm_m.get("llm_top_1_test_accuracy"),
        "llm_test_accuracy": llm_m.get("llm_test_accuracy"),
        "repro003_reference": {"sae_top_1": 0.767, "llm_top_1": 0.688},
    }
    with open("/root/outputs/saebench_custom_sae.json", "w") as fh:
        json.dump(out, fh)
    artifacts_vol.commit()
    print("SAEBENCH CUSTOM-SAE SPARSE-PROBING RESULT:", out)
    return out


@app.function(image=full_image, secrets=[HF_SECRET], timeout=600, retries=0)
def probe_saebench_datasets() -> dict:
    """E4: list the dataset keys SAEBench sparse_probing recognizes (chosen_classes_per_dataset), so
    the eval uses a key that exists (bare 'LabHC/bias_in_bios' KeyErrors; the recognized key is
    '..._class_set1', which is what repro-003 actually used)."""
    import inspect

    out: dict = {}
    from sae_bench.evals.sparse_probing import main as sp

    out["run_eval_sig"] = f"run_eval{inspect.signature(sp.run_eval)}"[:300]
    out["SparseProbingEvalConfig_default_dataset_names"] = str(
        getattr(sp.SparseProbingEvalConfig(), "dataset_names", "MISSING")
    )[:400]
    try:
        from sae_bench.evals.sparse_probing import dataset_info as di

        keys = list(di.chosen_classes_per_dataset.keys())
        out["chosen_classes_keys"] = ", ".join(keys)[:600]
        bib = [k for k in keys if "bias" in k.lower() or "bios" in k.lower()]
        out["bias_in_bios_keys"] = ", ".join(bib) or "NONE"
    except Exception as exc:  # noqa: BLE001
        out["dataset_info"] = f"FAIL: {type(exc).__name__}: {str(exc)[:160]}"
    for k, v in out.items():
        print(f"=== {k} ===\n{v}\n")
    return out

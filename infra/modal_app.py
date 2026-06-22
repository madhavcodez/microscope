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
def train_coder_modal(config_dict: dict, kind: str = "sae") -> dict:
    """Phase 2: train an SAE or skip-transcoder on Modal via the verified microscope.saes.train wrapper.

    Persists the trained dictionary to the artifacts Volume so Phase-3 eval can reload it. Returns the
    metrics dict + a listing of the saved files (the smoke's save/load confirmation)."""
    import glob
    import os

    from microscope.config import RunConfig
    from microscope.saes.train import train_coder

    # Redirect the dictionary save into the mounted persistent Volume (overrides the YAML save_dir).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = RunConfig(**{**config_dict, "save_dir": "/root/outputs/coders"})
    result = train_coder(cfg, kind)  # type: ignore[arg-type]

    save_path = result.get("save_path", "")
    all_paths = sorted(glob.glob(save_path + "/**", recursive=True))
    result["saved_files"] = [p.replace("/root/outputs/", "") for p in all_paths if os.path.isfile(p)][:20]
    result["n_saved_files"] = sum(1 for p in all_paths if os.path.isfile(p))
    artifacts_vol.commit()
    print("TRAIN RESULT:", result)
    return result


@app.local_entrypoint()
def train_main(config: str, kind: str = "sae") -> None:
    """Run a training job: modal run infra/modal_app.py::train_main --config <yaml> --kind sae|transcoder."""
    import yaml

    with open(config) as fh:
        config_dict = yaml.safe_load(fh)
    result = train_coder_modal.remote(config_dict, kind)
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
        dataset_names=["LabHC/bias_in_bios_class_set1"],  # single dataset = fast smoke
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
        "dataset": "LabHC/bias_in_bios_class_set1",
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

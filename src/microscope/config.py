"""Configuration, determinism, and run-metadata logging for MicroScope.

This is the CPU-verifiable foundation every stage depends on. It deliberately does **not** import
``torch`` at module-import time: torch-dependent helpers degrade gracefully when torch is absent, so
the base install (``pip install -e .``) and the CLI work on a machine with no ML stack.

Responsibilities (RULES.md E1/E2/E3):
- Seed + determinism control (Python / NumPy / PyTorch / CUDA).
- Pydantic config models + YAML loading.
- Stable config hashing (so a run's config is identified by content, not by filename).
- Capturing run metadata (git commit, hardware) and appending a row to ``docs/EXPERIMENTS.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Gates (research integrity)


class ReproductionGateError(RuntimeError):
    """Raised when RULES.md R1 is violated: custom training attempted before a known result.

    R1 ("reproduction before novelty") forbids training a custom SAE/transcoder until the pipeline
    has reproduced a known Gemma Scope result and logged it in ``docs/EXPERIMENTS.md``. This is the
    research-integrity gate; it is deliberately distinct from the GPU/library gate exceptions in
    :mod:`microscope._pending` (``GpuImplementationPending`` / ``GpuStackUnavailable``), which are
    about environment availability, not research validity.
    """


# Project paths


def project_root(start: Path | None = None) -> Path:
    """Return the repo root by walking up until a ``pyproject.toml`` is found.

    Falls back to the current working directory if no marker is located (e.g. installed wheel).
    """
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "docs").exists():
            return parent
    return Path.cwd()


# Resolved eagerly at import for the default log target. Helpers that take a `path` arg (e.g.
# append_experiment_row) call project_root() lazily so tests can monkeypatch it; this constant
# is just the convenience default, not the source of truth.
EXPERIMENTS_PATH = project_root() / "docs" / "EXPERIMENTS.md"


# Determinism (E1)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch + CUDA for reproducible runs.

    Args:
        seed: The integer seed to apply across all RNGs.
        deterministic: If True, also request deterministic algorithms from torch/cuDNN and set the
            CUBLAS workspace env var. This trades some speed for exact reproducibility (E1).

    torch is imported lazily so this function is usable on a CPU-only install with no torch present.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Must be set before the first CUDA op for cuBLAS determinism.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Config models (E2/E5) + loading


class RunConfig(BaseModel):
    """Base run configuration. Stage-specific YAMLs may add fields (``extra='allow'``).

    Common fields are typed; anything a particular stage needs (width, l0_target, trainer, etc.)
    can be supplied in the YAML and is preserved for hashing + logging.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Human-readable run name, also used in experiment notes.")
    model: str = Field(description="Model id, e.g. 'google/gemma-2-2b' or 'EleutherAI/pythia-70m'.")
    layer: int | None = Field(default=None, description="Residual/MLP layer index, if applicable.")
    hookpoint: str | None = Field(
        default=None, description="nnsight/TransformerLens hookpoint name, if applicable."
    )
    dataset: str | None = Field(default=None, description="Dataset id used to harvest activations.")
    n_tokens: int | None = Field(default=None, description="Token budget for the run.")
    seed: int = Field(default=0, description="RNG seed (E1).")


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML run config from ``experiments/configs/`` into a :class:`RunConfig`.

    Raises:
        FileNotFoundError: if the path does not exist (fail fast — RULES.md input validation).
        ValueError: if the YAML is empty or not a mapping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}: {p}")
    return RunConfig(**raw)


def config_hash(config: Mapping[str, Any] | BaseModel, *, length: int = 12) -> str:
    """Return a stable short hash of a config's content (E2).

    The hash is over canonical JSON (sorted keys), so the same config produces the same hash
    regardless of key order or source filename.
    """
    if isinstance(config, BaseModel):
        payload = config.model_dump(mode="json")
    else:
        payload = dict(config)
    canonical = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


# Run metadata (E3)


def git_commit(short: bool = True) -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo / git missing.

    Never raises — metadata capture must not crash a run.
    """
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, cwd=project_root(), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def hardware_info() -> dict[str, str]:
    """Capture a compact hardware fingerprint for the experiment log (E3).

    Reports the GPU name + total memory if torch+CUDA are available, otherwise 'cpu'. Lazy torch
    import keeps this usable on a CPU-only install.
    """
    info: dict[str, str] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": str(os.cpu_count() or "?"),
        "gpu": "cpu",
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gib = props.total_memory / (1024**3)
            info["gpu"] = f"{props.name} ({total_gib:.0f}GiB)"
            info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


# Columns must stay in sync with the header in docs/EXPERIMENTS.md.
EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "config_hash",
    "git_commit",
    "model",
    "layer/hookpoint",
    "coder_type (SAE/transcoder/eval)",
    "width",
    "sparsity/L0",
    "dataset",
    "tokens",
    "seed",
    "hardware",
    "wall_clock",
    "cost_est",
    "key_results",
    "label (repro/novel/inconclusive)",
    "notes",
)


class RunRecord(BaseModel):
    """One row of docs/EXPERIMENTS.md (E3 + R5). Every reported claim must map back to a row."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    date: str
    config_hash: str
    git_commit: str
    model: str
    layer_or_hookpoint: str = ""
    kind: str = ""  # SAE / transcoder / eval / control / circuit / reproduce
    width: str = ""
    sparsity_l0: str = ""
    dataset: str = ""
    tokens: str = ""
    seed: str = ""
    hardware: str = ""
    wall_clock: str = ""
    cost_est: str = ""
    key_results: str = ""
    label: str = ""  # reproduced / novel / inconclusive  (R4)
    notes: str = ""

    def as_row_cells(self) -> list[str]:
        """Return the 18 cell values in EXPERIMENT_COLUMNS order, pipes escaped for markdown."""
        ordered = [
            self.run_id,
            self.date,
            self.config_hash,
            self.git_commit,
            self.model,
            self.layer_or_hookpoint,
            self.kind,
            self.width,
            self.sparsity_l0,
            self.dataset,
            self.tokens,
            self.seed,
            self.hardware,
            self.wall_clock,
            self.cost_est,
            self.key_results,
            self.label,
            self.notes,
        ]
        return [str(c).replace("|", "\\|").replace("\n", " ").strip() for c in ordered]


def append_experiment_row(record: RunRecord, *, path: Path | None = None) -> None:
    """Append a run record as a markdown table row to docs/EXPERIMENTS.md (E3/R5).

    Creates the file with a header if it does not yet exist. Idempotency is the caller's concern;
    this always appends (one row per run).
    """
    target = path or EXPERIMENTS_PATH
    cells = record.as_row_cells()
    row = "| " + " | ".join(cells) + " |\n"

    if not target.exists():
        header = "# EXPERIMENTS\n\n"
        header += "| " + " | ".join(EXPERIMENT_COLUMNS) + " |\n"
        header += "|" + "|".join(["---"] * len(EXPERIMENT_COLUMNS)) + "|\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(header, encoding="utf-8")

    with target.open("a", encoding="utf-8") as fh:
        fh.write(row)


# R1 reproduction gate


def _split_markdown_row(line: str) -> list[str]:
    """Split one markdown table line into stripped cells, tolerant of leading/trailing pipes.

    ``| a | b |`` and ``a | b`` both yield ``['a', 'b']``. Empty edge cells produced by the
    surrounding pipes are dropped; interior empty cells are preserved so columns stay aligned.
    """
    stripped = line.strip()
    parts = [cell.strip() for cell in stripped.split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def reproduction_logged(path: Path | None = None) -> bool:
    """Return True iff ``docs/EXPERIMENTS.md`` has a data row labelled 'reproduced' (RULES.md R1).

    Parses the EXPERIMENTS.md markdown table, finds the header row (the line starting with
    ``| run_id``), locates the column whose header contains ``label`` (the full header is
    ``label (repro/novel/inconclusive)``), then scans every data row and checks whether that
    column's cell — stripped and lowercased — contains ``reproduced``.

    This is the mechanical R1 gate: it answers "has the pipeline reproduced a known result yet?"
    so callers (e.g. the ``train`` command) can refuse to start custom training until it has.

    Args:
        path: EXPERIMENTS.md to inspect. Defaults to the module's :data:`EXPERIMENTS_PATH`.

    Returns:
        True if at least one data row's label cell contains 'reproduced', else False. A missing
        file, an absent header, or a table with no data rows all return False (fail closed — the
        gate stays shut when there is no positive evidence of a reproduction).
    """
    target = path or EXPERIMENTS_PATH
    if not target.exists():
        return False

    lines = target.read_text(encoding="utf-8").splitlines()

    header_index: int | None = None
    label_col: int | None = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| run_id"):
            cells = _split_markdown_row(line)
            for col, name in enumerate(cells):
                if "label" in name.strip().lower():
                    label_col = col
                    break
            header_index = i
            break

    if header_index is None or label_col is None:
        return False

    # Data rows are the pipe-prefixed lines after the header, skipping the '|---' separator row.
    for line in lines[header_index + 1 :]:
        if not line.strip().startswith("|"):
            continue
        if set(line.strip()) <= {"|", "-", ":", " "}:
            continue  # separator row (e.g. |---|---|)
        cells = _split_markdown_row(line)
        if label_col < len(cells) and "reproduced" in cells[label_col].strip().lower():
            return True

    return False

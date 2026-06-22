"""MicroScope CLI — one command per pipeline stage (Typer).

`microscope info` is fully functional on CPU and exercises the config/determinism/metadata layer.
The GPU-bound stages (reproduce/train/autointerp/eval/control/circuit) load + hash their config,
set the seed, then dispatch to the stage contract. Until the GPU host exists (docs/PROGRESS.md
Gate #1) those contracts raise a clear, documented 'pending' message rather than failing obscurely.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from ._pending import GpuImplementationPending, GpuStackUnavailable
from .config import RunConfig, config_hash, git_commit, hardware_info, load_config, set_seed

app = typer.Typer(
    add_completion=False,
    help="MicroScope — reproducible mechanistic-interpretability toolkit.",
    no_args_is_help=True,
)
console = Console()


def _prepare(config_path: Path) -> RunConfig:
    """Load a config, log its identity (hash/commit/hardware), set the seed. Shared prelude."""
    config = load_config(config_path)
    chash = config_hash(config)
    console.print(
        f"[bold]config[/bold]={config_path}  [bold]hash[/bold]={chash}  "
        f"[bold]commit[/bold]={git_commit()}  [bold]seed[/bold]={config.seed}"
    )
    set_seed(config.seed)
    return config


def _run_stage(label: str, fn: Callable[[], Any]) -> None:
    """Execute a stage callable, rendering the GPU/E4 gate cleanly instead of a raw traceback."""
    try:
        result = fn()
        console.print(f"[green]{label} complete[/green]: {result}")
    except (GpuImplementationPending, GpuStackUnavailable) as exc:
        console.print(f"[yellow]GATED[/yellow] {label}: {exc}")
        raise typer.Exit(code=2) from None


@app.command()
def info() -> None:
    """Show version, git commit, and hardware (verifies the config/metadata layer works)."""
    hw = hardware_info()
    table = Table(title=f"MicroScope v{__version__}", show_header=False)
    table.add_row("git commit", git_commit())
    for key, value in hw.items():
        table.add_row(key, value)
    console.print(table)


@app.command()
def reproduce(
    config: Path = typer.Option(..., help="Path to a Phase-1 reproduction YAML config."),
) -> None:
    """Phase 1 (HARD GATE): reproduce a known Gemma Scope auto-interp + SAEBench result."""
    from .reproduce.gemma_scope import reproduce as _reproduce

    cfg = _prepare(config)
    _run_stage("reproduce", lambda: _reproduce(cfg))


@app.command()
def train(
    config: Path = typer.Option(..., help="Path to a Phase-2 training YAML config."),
    kind: str = typer.Option("sae", help="'sae' or 'transcoder' (skip-transcoder)."),
) -> None:
    """Phase 2: train a custom SAE or skip-transcoder (smoke-test on Pythia-70M first)."""
    from .saes.train import train_coder

    cfg = _prepare(config)
    _run_stage(f"train:{kind}", lambda: train_coder(cfg, kind))  # type: ignore[arg-type]


@app.command()
def autointerp(
    config: Path = typer.Option(..., help="Path to an auto-interp YAML config."),
    n_features: int = typer.Option(100, help="Features to interpret (<=500, RULES.md C3)."),
    scorer_model: str = typer.Option(..., help="HF id of the LOCAL scorer model (no paid API)."),
) -> None:
    """Phase 1/3: explanations + detection/fuzzing/intruder scores via delphi (local scorer)."""
    from .autointerp.run import run_autointerp

    cfg = _prepare(config)
    _run_stage(
        "autointerp",
        lambda: run_autointerp(cfg, sae=None, n_features=n_features, scorer_model=scorer_model),
    )


@app.command(name="eval")
def eval_(config: Path = typer.Option(..., help="Path to an eval YAML config.")) -> None:
    """Phase 1/3: SAEBench scorecard for an SAE/transcoder."""
    from .eval.saebench import run_saebench

    cfg = _prepare(config)
    _run_stage("eval", lambda: run_saebench(cfg, sae=None))


@app.command()
def control(
    config: Path = typer.Option(..., help="Path to a controls YAML config."),
    kind: str = typer.Option(
        "randomized", help="'randomized' (model control) or 'steering' baseline."
    ),
    n_features: int = typer.Option(100, help="Features for the randomized-model control (<=500)."),
    scorer_model: str = typer.Option("", help="LOCAL scorer model id (randomized control)."),
) -> None:
    """Phase 4 (mandatory, RULES.md R2): randomized-model control + steering-vs-simple-baseline."""
    cfg = _prepare(config)
    if kind == "randomized":
        from .eval.controls import randomized_model_control

        _run_stage(
            "control:randomized",
            lambda: randomized_model_control(cfg, n_features=n_features, scorer_model=scorer_model),
        )
    elif kind == "steering":
        from .steering.baselines import steer_with_sae_feature

        _run_stage("control:steering", lambda: steer_with_sae_feature(cfg, None, 0, 1.0))
    else:
        raise typer.BadParameter("kind must be 'randomized' or 'steering'")


@app.command()
def circuit(
    config: Path = typer.Option(..., help="Path to a circuit YAML config."),
    task: str = typer.Option("bias_in_bios_profession_classification", help="Behavior to explain."),
) -> None:
    """Phase 5: discover + validate one editable feature circuit; save the graph artifact."""
    from .circuits.discover import discover_circuit

    cfg = _prepare(config)
    _run_stage("circuit", lambda: discover_circuit(cfg, sae=None, task=task))


if __name__ == "__main__":
    app()

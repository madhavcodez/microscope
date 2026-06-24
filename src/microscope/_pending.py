"""Shared sentinel for stages whose library wrapper is intentionally not yet implemented.

Per RULES.md E4 ("verify the library API before you code"), the wrappers around dictionary_learning,
delphi, sae-bench, sparse-feature-circuits, nnsight, and sae_lens must be written against the live
installed packages on the GPU host, not against remembered/assumed APIs. Until that environment
exists (docs/PROGRESS.md Gate #1), these stages raise :class:`GpuImplementationPending` so the
failure is explicit and self-documenting rather than a silent stub or a fabricated call.
"""

from __future__ import annotations


class GpuImplementationPending(NotImplementedError):
    """A stage that must be implemented on the GPU host after verifying the library API (E4)."""


class GpuStackUnavailable(RuntimeError):
    """An IMPLEMENTED GPU stage was invoked where the interp stack is absent.

    Distinct from :class:`GpuImplementationPending` (which means "not written yet"): this means the
    code exists and is correct, but ``sae_lens`` / ``transformer_lens`` etc. are not installed -
    they live only on the Modal ``[gpu]`` image (ADR-0003). Subclasses ``RuntimeError`` so callers
    that catch ``RuntimeError`` still work; the CLI catches it to render the gate cleanly.
    """


def pending(stage: str, lib: str, phase: str) -> GpuImplementationPending:
    """Build a consistent, actionable 'not yet implemented' error for a GPU/library-bound stage."""
    return GpuImplementationPending(
        f"[{phase}] '{stage}' is not implemented yet. Per RULES.md E4, wire it "
        f"against the installed '{lib}' API on the GPU host "
        f"(see docs/PROGRESS.md Gate #1 and docs/adr/0002). "
        f"This is a deliberate, documented stub, not a bug."
    )

"""Shared sentinel for stages whose library wrapper is intentionally not yet implemented.

Per RULES.md E4 ("verify the library API before you code"), the wrappers around dictionary_learning,
delphi, sae-bench, sparse-feature-circuits, nnsight, and sae_lens must be written against the live
installed packages on the GPU host — not against remembered/assumed APIs. Until that environment
exists (docs/PROGRESS.md Gate #1), these stages raise :class:`GpuImplementationPending` so the failure
is explicit and self-documenting rather than a silent stub or a fabricated call.
"""

from __future__ import annotations


class GpuImplementationPending(NotImplementedError):
    """A stage that must be implemented on the GPU host after verifying the library API (E4)."""


def pending(stage: str, lib: str, phase: str) -> GpuImplementationPending:
    """Build a consistent, actionable 'not yet implemented' error for a GPU/library-bound stage."""
    return GpuImplementationPending(
        f"[{phase}] '{stage}' is not implemented yet. Per RULES.md E4, wire it against the installed "
        f"'{lib}' API on the GPU host (see docs/PROGRESS.md Gate #1 and docs/adr/0002). "
        f"This is a deliberate, documented stub — not a bug."
    )

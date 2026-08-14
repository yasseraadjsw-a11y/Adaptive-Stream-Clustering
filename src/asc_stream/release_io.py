from __future__ import annotations

from pathlib import Path


PROTECTED_RESULTS_PARTS = ("results", "main_results")


def ensure_not_protected_output(root: Path, output: Path) -> Path:
    """Resolve an output path and reject writes inside protected study results.

    The immutable manuscript-facing results live under ``results/main_results``.
    Fresh executions must be written elsewhere (normally ``results/execution_runs``).
    This guard is deliberately used by direct experiment scripts as well as main.py.
    """
    root = Path(root).resolve()
    out = Path(output)
    resolved = (root / out).resolve() if not out.is_absolute() else out.resolve()
    protected = (root / "results" / "main_results").resolve()
    try:
        resolved.relative_to(protected)
    except ValueError:
        return resolved
    raise ValueError(
        f"Refusing to write into protected manuscript results: {resolved}. "
        "Use results/execution_runs/ for fresh executions."
    )


def ensure_execution_output(root: Path, output: Path) -> Path:
    """Require a fresh output to live inside results/execution_runs."""
    root = Path(root).resolve()
    resolved = ensure_not_protected_output(root, output)
    execution_root = (root / "results" / "execution_runs").resolve()
    try:
        resolved.relative_to(execution_root)
    except ValueError as exc:
        raise ValueError(
            f"Fresh execution output must be inside {execution_root}: {resolved}"
        ) from exc
    return resolved

"""Centralized MLX import guard and backend utilities.

All modules that optionally use MLX should import from here instead
of duplicating the try/except import pattern.
"""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


def to_mx(arr: np.ndarray) -> mx.array:
    """Convert a NumPy array to an MLX array (float32)."""
    if not HAS_MLX:
        raise ImportError("MLX is not installed. Install with: pip install 'helicon[mlx]'")
    return mx.array(arr.astype(np.float32))


def to_np(arr: mx.array) -> np.ndarray:
    """Convert an MLX array back to a NumPy float64 array."""
    return np.array(arr).astype(np.float64)


def resolve_backend(backend: str) -> str:
    """Resolve ``"auto"`` to ``"mlx"`` or ``"numpy"`` based on availability.

    Parameters
    ----------
    backend : str
        One of ``"auto"``, ``"mlx"``, or ``"numpy"``.

    Returns
    -------
    str
        ``"mlx"`` or ``"numpy"``.
    """
    if backend == "auto":
        return "mlx" if HAS_MLX else "numpy"
    if backend not in ("mlx", "numpy"):
        raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'mlx', or 'numpy'.")
    if backend == "mlx" and not HAS_MLX:
        raise ImportError("MLX is not installed. Install with: pip install 'helicon[mlx]'")
    return backend

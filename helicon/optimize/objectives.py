"""Differentiable field-geometry objectives for MLX-based coil optimization.

These objectives map coil parameters → applied B-field → scalar metric,
and are fully differentiable via ``mlx.core.grad``.  They do **not**
require a WarpX simulation, so gradient-based optimization runs in seconds
on Apple Silicon.

Typical use cases
-----------------
- Maximize throat-to-exit B ratio (stronger magnetic confinement)
- Minimize peak conductor field (structural/thermal stress)
- Control on-axis field uniformity

For objectives that require WarpX evaluation (e.g. detachment efficiency),
use :class:`helicon.optimize.surrogate.BayesianOptimizer` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _require_mlx() -> None:
    if not HAS_MLX:
        raise ImportError(
            "MLX is required for gradient-based coil optimization. "
            "Install with: pip install 'helicon[mlx]'"
        )


@dataclass
class OptimizationResult:
    """Result from a gradient-based coil optimization run.

    Attributes
    ----------
    coil_params : np.ndarray, shape (N_coils, 3)
        Optimized coil parameters ``[z, r, I]``.
    history : list of float
        Objective value recorded at each gradient step.
    n_steps : int
        Number of steps executed.
    converged : bool
        True if the convergence tolerance was reached.
    """

    coil_params: np.ndarray
    history: list[float]
    n_steps: int
    converged: bool


def throat_ratio_objective(
    coil_params: mx.array,
    grid_r: mx.array,
    grid_z: mx.array,
    *,
    n_phi: int = 64,
) -> mx.array:
    """On-axis throat-to-exit B_z ratio (maximize for confinement).

    Computes B_z along the axis (r=0), then returns B_z_max / B_z_min_nonzero.
    Fully differentiable w.r.t. ``coil_params``.

    Parameters
    ----------
    coil_params : mx.array, shape (N_coils, 3)
        ``[z_coil, radius, current]`` per coil.
    grid_r, grid_z : mx.array, shape (N_pts,)
        Evaluation points — typically the axis (r=0).
    n_phi : int
        Azimuthal quadrature resolution.

    Returns
    -------
    ratio : scalar mx.array
    """
    _require_mlx()
    from helicon.fields.biot_savart import compute_bfield_mlx_differentiable

    _, Bz = compute_bfield_mlx_differentiable(coil_params, grid_r, grid_z, n_phi=n_phi)
    Bz_max = mx.max(Bz)
    # Guard against zero/negative field
    Bz_min = mx.maximum(mx.min(mx.abs(Bz) + 1e-10), mx.array(1e-10))
    return Bz_max / Bz_min


def peak_field_objective(
    coil_params: mx.array,
    grid_r: mx.array,
    grid_z: mx.array,
    *,
    n_phi: int = 64,
) -> mx.array:
    """Negative peak |B| on the grid (minimize conductor stress).

    Returns ``-max(|B|)`` so that maximizing this objective minimizes the
    peak field (consistent with the maximization convention used by
    :class:`~helicon.optimize.surrogate.BayesianOptimizer`).

    Fully differentiable w.r.t. ``coil_params``.
    """
    _require_mlx()
    from helicon.fields.biot_savart import compute_bfield_mlx_differentiable

    Br, Bz = compute_bfield_mlx_differentiable(coil_params, grid_r, grid_z, n_phi=n_phi)
    B_mag = mx.sqrt(Br * Br + Bz * Bz)
    return -mx.max(B_mag)


def optimize_coils_mlx(
    coil_params_init: np.ndarray,
    objective_fn: callable,
    *,
    n_steps: int = 200,
    lr: float = 1e-3,
    bounds: np.ndarray | None = None,
    tol: float = 1e-6,
) -> OptimizationResult:
    """Gradient ascent on a differentiable field-geometry objective via MLX.

    Runs Adam optimizer on ``objective_fn(coil_params)`` with optional
    per-parameter clipping to ``bounds``.

    Parameters
    ----------
    coil_params_init : np.ndarray, shape (N_coils, 3)
        Starting coil parameters ``[z, r, I]``.
    objective_fn : callable
        ``(coil_params: mx.array) -> scalar mx.array``.
        Must be differentiable w.r.t. its argument.
    n_steps : int
        Maximum gradient steps.
    lr : float
        Adam learning rate.
    bounds : np.ndarray, shape (N_coils, 3, 2), optional
        ``[[[z_lo, z_hi], [r_lo, r_hi], [I_lo, I_hi]], ...]``.
        Applied via clipping after each step.
    tol : float
        Stop early if |Δobjective| < tol between consecutive steps.

    Returns
    -------
    OptimizationResult
    """
    _require_mlx()

    params = mx.array(coil_params_init.astype(np.float32))
    history: list[float] = []

    # Adam state
    m = mx.zeros_like(params)
    v = mx.zeros_like(params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    grad_fn = mx.grad(objective_fn)
    converged = False

    for step in range(n_steps):
        obj_val = objective_fn(params)
        g = grad_fn(params)
        mx.eval(obj_val, g)

        history.append(float(np.array(obj_val)))

        # Adam ascent (maximize objective)
        t = step + 1
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        params = params + lr * m_hat / (mx.sqrt(v_hat) + eps_adam)

        if bounds is not None:
            bounds_arr = np.asarray(bounds, dtype=np.float32)
            lo = mx.array(bounds_arr[:, :, 0])
            hi = mx.array(bounds_arr[:, :, 1])
            params = mx.clip(params, lo, hi)

        mx.eval(params)

        if step > 0 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break

    return OptimizationResult(
        coil_params=np.array(params),
        history=history,
        n_steps=len(history),
        converged=converged,
    )

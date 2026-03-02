"""MLX gradient-based coil optimizer for magnetic nozzle design.

Implements gradient descent with Adam-style momentum using MLX's automatic
differentiation.  Designed to work with the differentiable Biot-Savart
backend in :mod:`helicon.fields.biot_savart`.

When MLX is unavailable, importing the optimizer classes still works, but
calling :meth:`GradientOptimizer.run` raises :class:`ImportError` with a
clear message pointing to the installation instructions.

Typical usage
-------------
::

    from helicon.fields.biot_savart import Grid
    from helicon.optimize.gradient import (
        GradientOptimizer,
        GradientOptimizerConfig,
        optimize_mirror_ratio,
    )

    grid = Grid(z_min=-0.5, z_max=0.5, r_max=0.3, nz=64, nr=32)
    result = optimize_mirror_ratio(
        base_coil_params=coil_params_np,
        grid=grid,
        target_mirror_ratio=5.0,
        n_steps=300,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Optional MLX import
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _require_mlx() -> None:
    """Raise a descriptive ImportError when MLX is not available."""
    if not HAS_MLX:
        raise ImportError(
            "MLX is required for gradient-based coil optimization. "
            "Install with: pip install 'helicon[mlx]'"
        )


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class GradientOptimizerConfig:
    """Hyper-parameters for :class:`GradientOptimizer`.

    Attributes
    ----------
    n_steps : int
        Maximum number of gradient steps.  Default: 200.
    learning_rate : float
        Adam base learning rate.  Default: 1e-3.
    beta1 : float
        Adam first-moment decay.  Default: 0.9.
    beta2 : float
        Adam second-moment decay.  Default: 0.999.
    eps_adam : float
        Adam numerical stability term.  Default: 1e-8.
    tol : float
        Convergence tolerance on gradient L2 norm.  Optimization stops
        early when ``||grad||_2 < tol``.  Default: 1e-6.
    history_every : int
        Record ``coil_params`` snapshot every this many steps.  Default: 10.
    n_phi : int
        Azimuthal quadrature points for the MLX Biot-Savart backend.
        Default: 64.
    """

    n_steps: int = 200
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps_adam: float = 1e-8
    tol: float = 1e-6
    history_every: int = 10
    n_phi: int = 64


@dataclass
class GradientResult:
    """Result from a :class:`GradientOptimizer` run.

    Attributes
    ----------
    coil_params_history : list of np.ndarray
        Parameter snapshots recorded every ``config.history_every`` steps.
        Each array has the same shape as the initial parameter array.
    objective_history : list of float
        Scalar objective value at each gradient step.
    final_coil_params : np.ndarray
        Optimized parameters at the last completed step.
    n_steps_run : int
        Actual number of steps executed (may be less than ``n_steps`` if
        converged early).
    converged : bool
        True if gradient norm dropped below ``config.tol``.
    """

    coil_params_history: list[np.ndarray]
    objective_history: list[float]
    final_coil_params: np.ndarray
    n_steps_run: int
    converged: bool


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------
class GradientOptimizer:
    """Gradient descent optimizer backed by ``mlx.core.grad``.

    Parameters
    ----------
    grid : Grid
        Axisymmetric computation grid — passed through to the objective
        function during :meth:`run` if needed.  Not used internally by the
        optimizer itself; stored for user convenience.
    objective_fn : callable
        ``(coil_params: mx.array) -> mx.array`` — must return a *scalar*
        MLX array representing the loss to **minimise**.  The function must
        be differentiable with respect to its argument via
        ``mlx.core.grad``.
    config : GradientOptimizerConfig, optional
        Optimizer hyper-parameters.  Defaults to
        :class:`GradientOptimizerConfig` with all default values.

    Notes
    -----
    The optimizer performs *gradient descent* (minimisation).  If you want
    to *maximise* a metric, negate it inside ``objective_fn``.
    """

    def __init__(
        self,
        grid,
        objective_fn: Callable,
        config: GradientOptimizerConfig | None = None,
    ) -> None:
        self.grid = grid
        self.objective_fn = objective_fn
        self.config = config if config is not None else GradientOptimizerConfig()

    # ------------------------------------------------------------------
    def run(self, initial_params: np.ndarray) -> GradientResult:
        """Execute the optimization loop.

        Parameters
        ----------
        initial_params : np.ndarray
            Starting coil parameters.  Typically shape ``(N_coils, 3)`` for
            ``[z, r, I]`` parameterisation, but any shape is accepted as
            long as ``objective_fn`` handles it.

        Returns
        -------
        GradientResult
        """
        _require_mlx()

        cfg = self.config
        params = mx.array(initial_params.astype(np.float32))

        # Adam state
        m = mx.zeros_like(params)
        v = mx.zeros_like(params)

        objective_history: list[float] = []
        coil_params_history: list[np.ndarray] = []
        converged = False

        grad_fn = mx.grad(self.objective_fn)

        for step in range(cfg.n_steps):
            obj_val = self.objective_fn(params)
            g = grad_fn(params)
            mx.eval(obj_val, g)

            obj_scalar = float(np.array(obj_val))
            objective_history.append(obj_scalar)

            # Record parameter snapshot every history_every steps (and step 0)
            if step % cfg.history_every == 0:
                coil_params_history.append(np.array(params))

            # Adam update (descent: subtract update)
            t = step + 1
            m = cfg.beta1 * m + (1.0 - cfg.beta1) * g
            v = cfg.beta2 * v + (1.0 - cfg.beta2) * g * g
            m_hat = m / (1.0 - cfg.beta1**t)
            v_hat = v / (1.0 - cfg.beta2**t)
            params = params - cfg.learning_rate * m_hat / (mx.sqrt(v_hat) + cfg.eps_adam)
            mx.eval(params)

            # Convergence check: gradient L2 norm
            g_np = np.array(g)
            grad_norm = float(np.sqrt(np.sum(g_np**2)))
            if grad_norm < cfg.tol:
                converged = True
                # Record final snapshot on convergence
                coil_params_history.append(np.array(params))
                break

        final_params = np.array(params)

        return GradientResult(
            coil_params_history=coil_params_history,
            objective_history=objective_history,
            final_coil_params=final_params,
            n_steps_run=len(objective_history),
            converged=converged,
        )


# ---------------------------------------------------------------------------
# Convenience function: optimize mirror ratio
# ---------------------------------------------------------------------------
def optimize_mirror_ratio(
    base_coil_params: np.ndarray,
    grid,
    *,
    target_mirror_ratio: float = 5.0,
    n_steps: int = 200,
    learning_rate: float = 1e-3,
    backend: str = "auto",
) -> GradientResult:
    """Optimize coil currents to maximize mirror ratio via gradient descent.

    Builds an ``objective_fn`` that calls
    :func:`~helicon.fields.biot_savart.compute_bfield_mlx_differentiable`
    and returns the *negative* mirror ratio (so minimising = maximising
    R_B = B_max / B_exit).

    Parameters
    ----------
    base_coil_params : np.ndarray, shape (N_coils, 3)
        Initial coil parameters ``[z_coil, radius, current]`` in SI units.
    grid : Grid
        Axisymmetric computation grid used to evaluate the field.
    target_mirror_ratio : float
        Currently unused — reserved for future penalised objectives.
        The optimization maximises R_B unconditionally.
    n_steps : int
        Maximum gradient steps.  Default: 200.
    learning_rate : float
        Adam learning rate.  Default: 1e-3.
    backend : str
        Backend selector — ``"auto"`` or ``"mlx"``.  ``"numpy"`` is not
        supported because the objective must be differentiable; if
        ``"numpy"`` or any unsupported backend is requested, an
        :class:`ImportError` is raised.  ``"auto"`` selects MLX when
        available, otherwise raises :class:`ImportError`.

    Returns
    -------
    GradientResult

    Raises
    ------
    ImportError
        When MLX is not available or an incompatible backend is requested.
    """
    from helicon.fields.biot_savart import (
        HAS_MLX as _HAS_MLX,
    )
    from helicon.fields.biot_savart import (
        compute_bfield_mlx_differentiable,
    )

    # Resolve backend
    if backend == "auto":
        if not _HAS_MLX:
            raise ImportError(
                "MLX is required for gradient-based mirror ratio optimization. "
                "Install with: pip install 'helicon[mlx]'"
            )
    elif backend != "mlx":
        raise ImportError(
            f"Backend {backend!r} does not support automatic differentiation. "
            "Use backend='mlx' or backend='auto' for gradient-based optimization."
        )

    _require_mlx()

    # Build grid arrays once (outside the hot loop)
    np.linspace(0.0, grid.r_max, grid.nr).astype(np.float32)
    z_np = np.linspace(grid.z_min, grid.z_max, grid.nz).astype(np.float32)
    # Use on-axis points only (r=0) for efficiency
    z_axis_np = z_np.copy()
    r_axis_np = np.zeros_like(z_axis_np)

    grid_r_mlx = mx.array(r_axis_np)
    grid_z_mlx = mx.array(z_axis_np)

    def objective_fn(coil_params: mx.array) -> mx.array:
        """Return negative mirror ratio (to minimise = maximise R_B)."""
        _, Bz = compute_bfield_mlx_differentiable(
            coil_params,
            grid_r_mlx,
            grid_z_mlx,
            n_phi=64,
        )
        Bz_abs = mx.abs(Bz)
        B_max = mx.max(Bz_abs)
        # B_exit: last on-axis point; guard against near-zero
        B_exit = mx.maximum(Bz_abs[-1], mx.array(1e-20))
        mirror_ratio_val = B_max / B_exit
        return -mirror_ratio_val  # negate: minimise = maximise R_B

    config = GradientOptimizerConfig(
        n_steps=n_steps,
        learning_rate=learning_rate,
    )
    optimizer = GradientOptimizer(grid, objective_fn, config=config)
    return optimizer.run(base_coil_params)

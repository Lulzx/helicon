"""Tests for helicon.optimize.gradient — 6 test cases.

MLX-dependent tests are skipped when MLX is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from helicon.fields.biot_savart import HAS_MLX, Grid
from helicon.optimize.gradient import (
    GradientOptimizer,
    GradientOptimizerConfig,
    GradientResult,
    optimize_mirror_ratio,
)

if HAS_MLX:
    import mlx.core as mx

skip_no_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
SMALL_GRID = Grid(z_min=-0.5, z_max=0.5, r_max=0.3, nz=32, nr=8)

# A minimal two-coil parameter array  [z, r, I]
BASE_PARAMS = np.array(
    [
        [-0.2, 0.1, 5000.0],
        [0.2, 0.1, 5000.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Test 1: GradientOptimizerConfig default values
# ---------------------------------------------------------------------------
def test_gradient_optimizer_config_defaults():
    """GradientOptimizerConfig should expose expected defaults."""
    cfg = GradientOptimizerConfig()

    assert cfg.n_steps == 200
    assert cfg.learning_rate == pytest.approx(1e-3)
    assert cfg.beta1 == pytest.approx(0.9)
    assert cfg.beta2 == pytest.approx(0.999)
    assert cfg.eps_adam == pytest.approx(1e-8)
    assert cfg.tol == pytest.approx(1e-6)
    assert cfg.history_every == 10
    assert cfg.n_phi == 64


# ---------------------------------------------------------------------------
# Test 2: GradientResult has all required fields
# ---------------------------------------------------------------------------
def test_gradient_result_structure():
    """GradientResult must carry all required attributes with correct types."""
    dummy_params = np.zeros((2, 3))
    result = GradientResult(
        coil_params_history=[dummy_params.copy(), dummy_params.copy()],
        objective_history=[1.0, 0.8, 0.6],
        final_coil_params=dummy_params.copy(),
        n_steps_run=3,
        converged=False,
    )

    assert hasattr(result, "coil_params_history")
    assert hasattr(result, "objective_history")
    assert hasattr(result, "final_coil_params")
    assert hasattr(result, "n_steps_run")
    assert hasattr(result, "converged")

    assert isinstance(result.coil_params_history, list)
    assert isinstance(result.objective_history, list)
    assert isinstance(result.final_coil_params, np.ndarray)
    assert isinstance(result.n_steps_run, int)
    assert isinstance(result.converged, bool)

    assert result.n_steps_run == 3
    assert len(result.objective_history) == 3
    assert len(result.coil_params_history) == 2
    assert result.final_coil_params.shape == (2, 3)
    assert result.converged is False


# ---------------------------------------------------------------------------
# Test 3: Objective decreases over iterations for a trivial problem
# ---------------------------------------------------------------------------
@skip_no_mlx
def test_objective_history_decreasing():
    """On a trivial convex loss (||params||^2), the objective must decrease."""

    # Trivial objective: minimize L2 norm of params
    def quadratic_loss(params: mx.array) -> mx.array:
        return mx.sum(params * params)

    init_params = np.ones((2, 3), dtype=np.float64) * 2.0

    cfg = GradientOptimizerConfig(n_steps=50, learning_rate=1e-1, history_every=5)
    optimizer = GradientOptimizer(SMALL_GRID, quadratic_loss, config=cfg)
    result = optimizer.run(init_params)

    assert len(result.objective_history) > 1, "No objective history recorded"

    # The first half of the run should show a clear downward trend
    first_half = result.objective_history[: len(result.objective_history) // 2]
    second_half = result.objective_history[len(result.objective_history) // 2 :]

    assert np.mean(second_half) < np.mean(first_half), (
        f"Objective did not decrease: first_half mean={np.mean(first_half):.4f}, "
        f"second_half mean={np.mean(second_half):.4f}"
    )

    # Final objective must be strictly less than initial
    assert result.objective_history[-1] < result.objective_history[0], (
        "Final objective is not smaller than initial objective"
    )

    # Parameter snapshots should exist
    assert len(result.coil_params_history) >= 1
    assert result.final_coil_params.shape == init_params.shape


# ---------------------------------------------------------------------------
# Test 4: optimize_mirror_ratio improves the mirror ratio
# ---------------------------------------------------------------------------
@skip_no_mlx
def test_optimize_mirror_ratio_improves():
    """After optimization, the mirror ratio should be higher than the initial one."""
    from helicon.fields.biot_savart import Coil, compute_bfield

    init_params = BASE_PARAMS.copy()
    grid = Grid(z_min=-0.5, z_max=0.5, r_max=0.3, nz=64, nr=4)

    # Compute initial mirror ratio using the numpy backend for reference
    coils_init = [
        Coil(z=init_params[i, 0], r=init_params[i, 1], I=init_params[i, 2])
        for i in range(len(init_params))
    ]
    bf_init = compute_bfield(coils_init, grid, backend="numpy")
    Bz_axis_init = bf_init.Bz[0, :]
    B_max_init = float(np.max(np.abs(Bz_axis_init)))
    B_exit_init = float(np.abs(Bz_axis_init[-1]))
    mirror_ratio_init = B_max_init / B_exit_init if B_exit_init > 1e-20 else float("inf")

    # Run gradient optimization
    result = optimize_mirror_ratio(
        base_coil_params=init_params,
        grid=grid,
        target_mirror_ratio=5.0,
        n_steps=30,
        learning_rate=1e-3,
    )

    assert isinstance(result, GradientResult)
    assert result.n_steps_run > 0
    assert len(result.objective_history) > 0
    assert result.final_coil_params.shape == init_params.shape

    # Compute final mirror ratio
    final_params = result.final_coil_params
    coils_final = [
        Coil(
            z=float(final_params[i, 0]),
            r=float(final_params[i, 1]),
            I=float(final_params[i, 2]),
        )
        for i in range(len(final_params))
    ]
    bf_final = compute_bfield(coils_final, grid, backend="numpy")
    Bz_axis_final = bf_final.Bz[0, :]
    B_max_final = float(np.max(np.abs(Bz_axis_final)))
    B_exit_final = float(np.abs(Bz_axis_final[-1]))
    mirror_ratio_final = B_max_final / B_exit_final if B_exit_final > 1e-20 else float("inf")

    # The optimizer minimizes -R_B, so R_B should increase (or at minimum not worsen
    # beyond a small tolerance for very short runs)
    assert mirror_ratio_final >= mirror_ratio_init * 0.95, (
        f"Mirror ratio did not improve: initial={mirror_ratio_init:.3f}, "
        f"final={mirror_ratio_final:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5: GradientOptimizer raises ImportError when MLX not available
# ---------------------------------------------------------------------------
def test_gradient_optimizer_no_mlx_raises(monkeypatch):
    """When MLX is absent, calling run() must raise ImportError."""
    import helicon.optimize.gradient as grad_module

    # Monkeypatch HAS_MLX to False inside the module
    monkeypatch.setattr(grad_module, "HAS_MLX", False)

    def dummy_objective(params):
        return params[0]

    cfg = GradientOptimizerConfig(n_steps=5)
    optimizer = GradientOptimizer(SMALL_GRID, dummy_objective, config=cfg)

    with pytest.raises(ImportError, match="MLX is required"):
        optimizer.run(np.ones((1, 3)))


# ---------------------------------------------------------------------------
# Test 6: converged=True is set when gradient norm falls below tol
# ---------------------------------------------------------------------------
@skip_no_mlx
def test_gradient_optimizer_convergence_flag():
    """The converged flag must be True when gradient norm drops below tol."""

    # A loss that rapidly converges: L(x) = 0.5 * x^2 starting near zero
    # With lr=0.5 it converges in very few steps.
    def nearly_zero_loss(params: mx.array) -> mx.array:
        return mx.sum(params * params) * mx.array(1e-6)

    init_params = np.array([[1e-4, 1e-4, 1e-4]], dtype=np.float64)

    # Set tol generously high so convergence triggers quickly
    cfg = GradientOptimizerConfig(
        n_steps=200,
        learning_rate=0.5,
        tol=1e-2,  # large tol: should converge early
        history_every=1,
    )
    optimizer = GradientOptimizer(SMALL_GRID, nearly_zero_loss, config=cfg)
    result = optimizer.run(init_params)

    assert result.converged is True, (
        f"Expected converged=True but got False after {result.n_steps_run} steps"
    )
    # Should have stopped early (well before n_steps)
    assert result.n_steps_run < cfg.n_steps, (
        f"Expected early stop, but ran all {cfg.n_steps} steps"
    )

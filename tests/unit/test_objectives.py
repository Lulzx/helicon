"""Tests for helicon.optimize.objectives."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.fields.biot_savart import HAS_MLX

skip_no_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

if HAS_MLX:
    import mlx.core as mx
    from helicon.optimize.objectives import (
        OptimizationResult,
        optimize_coils_mlx,
        peak_field_objective,
        throat_ratio_objective,
    )


@skip_no_mlx
class TestThroatRatioObjective:
    def test_returns_positive(self):
        params = mx.array([[0.0, 0.1, 1000.0]])
        grid_r = mx.zeros((20,))
        grid_z = mx.linspace(-0.3, 0.3, 20)
        ratio = throat_ratio_objective(params, grid_r, grid_z)
        mx.eval(ratio)
        assert float(np.array(ratio)) > 0

    def test_differentiable_wrt_current(self):
        params = mx.array([[0.0, 0.1, 1000.0]])
        grid_r = mx.zeros((20,))
        grid_z = mx.linspace(-0.3, 0.3, 20)
        grad_fn = mx.grad(lambda p: throat_ratio_objective(p, grid_r, grid_z))
        g = grad_fn(params)
        mx.eval(g)
        g_np = np.array(g)
        assert np.all(np.isfinite(g_np))
        assert g_np.shape == (1, 3)

    def test_differentiable_wrt_position(self):
        params = mx.array([[0.05, 0.1, 1000.0]])
        grid_r = mx.zeros((15,))
        grid_z = mx.linspace(-0.2, 0.2, 15)
        grad_fn = mx.grad(lambda p: throat_ratio_objective(p, grid_r, grid_z))
        g = grad_fn(params)
        mx.eval(g)
        assert np.all(np.isfinite(np.array(g)))

    def test_two_coil_case(self):
        params = mx.array([[-0.1, 0.1, 1000.0], [0.1, 0.1, 1000.0]])
        grid_r = mx.zeros((20,))
        grid_z = mx.linspace(-0.3, 0.3, 20)
        ratio = throat_ratio_objective(params, grid_r, grid_z)
        mx.eval(ratio)
        assert np.isfinite(float(np.array(ratio)))

    def test_no_nan_inf(self):
        params = mx.array([[0.0, 0.1, 1000.0]])
        grid_r = mx.zeros((50,))
        grid_z = mx.linspace(-0.5, 0.5, 50)
        val = throat_ratio_objective(params, grid_r, grid_z)
        mx.eval(val)
        assert np.isfinite(float(np.array(val)))


@skip_no_mlx
class TestPeakFieldObjective:
    def test_returns_negative(self):
        params = mx.array([[0.0, 0.1, 1000.0]])
        grid_r = mx.zeros((20,))
        grid_z = mx.linspace(-0.3, 0.3, 20)
        val = peak_field_objective(params, grid_r, grid_z)
        mx.eval(val)
        assert float(np.array(val)) < 0

    def test_larger_current_more_negative(self):
        grid_r = mx.zeros((20,))
        grid_z = mx.linspace(-0.3, 0.3, 20)
        v_low = peak_field_objective(mx.array([[0.0, 0.1, 500.0]]), grid_r, grid_z)
        v_high = peak_field_objective(mx.array([[0.0, 0.1, 2000.0]]), grid_r, grid_z)
        mx.eval(v_low, v_high)
        assert float(np.array(v_high)) < float(np.array(v_low))

    def test_differentiable(self):
        params = mx.array([[0.0, 0.1, 1000.0]])
        grid_r = mx.zeros((20,))
        grid_z = mx.linspace(-0.3, 0.3, 20)
        grad_fn = mx.grad(lambda p: peak_field_objective(p, grid_r, grid_z))
        g = grad_fn(params)
        mx.eval(g)
        assert np.all(np.isfinite(np.array(g)))


@skip_no_mlx
class TestOptimizeCoilsMLX:
    def _setup(self):
        grid_r = mx.zeros((15,))
        grid_z = mx.linspace(-0.2, 0.2, 15)
        init = np.array([[0.0, 0.1, 1000.0]], dtype=np.float32)
        return init, grid_r, grid_z

    def test_returns_optimization_result(self):
        init, grid_r, grid_z = self._setup()
        result = optimize_coils_mlx(
            init,
            lambda p: throat_ratio_objective(p, grid_r, grid_z),
            n_steps=5,
            lr=1.0,
        )
        assert isinstance(result, OptimizationResult)

    def test_output_shape(self):
        init, grid_r, grid_z = self._setup()
        result = optimize_coils_mlx(
            init,
            lambda p: throat_ratio_objective(p, grid_r, grid_z),
            n_steps=5,
        )
        assert result.coil_params.shape == (1, 3)

    def test_history_length(self):
        init, grid_r, grid_z = self._setup()
        result = optimize_coils_mlx(
            init,
            lambda p: throat_ratio_objective(p, grid_r, grid_z),
            n_steps=10,
        )
        assert len(result.history) == 10

    def test_history_finite(self):
        init, grid_r, grid_z = self._setup()
        result = optimize_coils_mlx(
            init,
            lambda p: throat_ratio_objective(p, grid_r, grid_z),
            n_steps=8,
        )
        assert all(np.isfinite(h) for h in result.history)

    def test_bounds_respected(self):
        init, grid_r, grid_z = self._setup()
        bounds = np.array([[[-0.05, 0.05], [0.08, 0.12], [800.0, 1200.0]]])
        result = optimize_coils_mlx(
            init,
            lambda p: peak_field_objective(p, grid_r, grid_z),
            n_steps=30,
            lr=50.0,
            bounds=bounds,
        )
        z, r, I = result.coil_params[0]
        assert -0.05 <= z <= 0.05
        assert 0.08 <= r <= 0.12
        assert 800.0 <= I <= 1200.0

    def test_two_coil_optimization(self):
        grid_r = mx.zeros((15,))
        grid_z = mx.linspace(-0.3, 0.3, 15)
        init = np.array([[-0.1, 0.1, 800.0], [0.1, 0.1, 800.0]], dtype=np.float32)
        result = optimize_coils_mlx(
            init,
            lambda p: throat_ratio_objective(p, grid_r, grid_z),
            n_steps=5,
        )
        assert result.coil_params.shape == (2, 3)

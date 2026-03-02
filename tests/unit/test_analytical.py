"""Tests for helicon.optimize.analytical."""

import math

import numpy as np
import pytest

from helicon.fields.biot_savart import HAS_MLX
from helicon.optimize.analytical import (
    NozzleScreeningResult,
    divergence_half_angle,
    divergence_half_angle_batch,
    thrust_coefficient_batch,
    thrust_coefficient_paraxial,
    thrust_efficiency,
    thrust_efficiency_batch,
)

skip_no_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")


class TestThrustEfficiency:
    def test_zero_for_unity_mirror_ratio(self):
        assert thrust_efficiency(1.0) == 0.0

    def test_one_for_infinite_mirror_ratio(self):
        assert thrust_efficiency(float("inf")) == 1.0

    def test_zero_for_below_unity(self):
        assert thrust_efficiency(0.5) == 0.0

    def test_adiabatic_r2(self):
        # η = 1 - (1/2)^((5/3-1)/(5/3)) = 1 - 0.5^(0.4) ≈ 0.2402
        eta = thrust_efficiency(2.0, gamma=5.0 / 3.0)
        expected = 1.0 - (0.5) ** (2.0 / 5.0)
        assert abs(eta - expected) < 1e-10

    def test_isothermal_r2(self):
        # γ=1: η = 1 - (1/R_B)^0 = 0 ... actually (γ-1)/γ = 0 → R_B^0 = 1 → η=0
        # Use γ=2 instead: η = 1 - (1/2)^(0.5) ≈ 0.293
        eta = thrust_efficiency(2.0, gamma=2.0)
        expected = 1.0 - (0.5) ** 0.5
        assert abs(eta - expected) < 1e-10

    def test_monotonically_increasing(self):
        ratios = [2.0, 5.0, 10.0, 100.0]
        etas = [thrust_efficiency(r) for r in ratios]
        for i in range(len(etas) - 1):
            assert etas[i] < etas[i + 1]

    def test_approaches_one_at_large_rb(self):
        eta = thrust_efficiency(1e6)
        assert eta > 0.99


class TestThrustCoefficientParaxial:
    def test_zero_for_unity_mirror_ratio(self):
        assert thrust_coefficient_paraxial(1.0) == 0.0

    def test_zero_for_below_unity(self):
        assert thrust_coefficient_paraxial(0.5) == 0.0

    def test_finite_for_infinite_rb(self):
        gamma = 5.0 / 3.0
        ct = thrust_coefficient_paraxial(float("inf"), gamma=gamma)
        expected = math.sqrt(2.0 * (gamma + 1.0) / (gamma - 1.0))
        assert abs(ct - expected) < 1e-10

    def test_ct_positive(self):
        ct = thrust_coefficient_paraxial(10.0)
        assert ct > 0.0

    def test_ct_increases_with_rb(self):
        ct2 = thrust_coefficient_paraxial(2.0)
        ct10 = thrust_coefficient_paraxial(10.0)
        ct100 = thrust_coefficient_paraxial(100.0)
        assert ct2 < ct10 < ct100

    def test_ct_formula_check(self):
        # C_T = sqrt(2(γ+1)/(γ-1)) * sqrt(η_T)
        gamma = 5.0 / 3.0
        R_B = 5.0
        eta = thrust_efficiency(R_B, gamma=gamma)
        expected = math.sqrt(2.0 * (gamma + 1.0) / (gamma - 1.0)) * math.sqrt(eta)
        ct = thrust_coefficient_paraxial(R_B, gamma=gamma)
        assert abs(ct - expected) < 1e-10


class TestDivergenceHalfAngle:
    def test_ninety_degrees_at_unity(self):
        assert divergence_half_angle(1.0) == 90.0

    def test_zero_at_infinity(self):
        assert divergence_half_angle(float("inf")) == 0.0

    def test_ninety_degrees_below_unity(self):
        assert divergence_half_angle(0.5) == 90.0

    def test_loss_cone_formula(self):
        # sin(θ) = 1/sqrt(R_B) → θ = arcsin(1/sqrt(R_B))
        R_B = 4.0
        expected_deg = math.degrees(math.asin(1.0 / math.sqrt(R_B)))
        theta = divergence_half_angle(R_B)
        assert abs(theta - expected_deg) < 1e-10

    def test_thirty_degrees_at_rb4(self):
        # sin(30°) = 0.5 → R_B = 4
        theta = divergence_half_angle(4.0)
        assert abs(theta - 30.0) < 1e-10

    def test_monotonically_decreasing(self):
        ratios = [2.0, 4.0, 10.0, 100.0]
        angles = [divergence_half_angle(r) for r in ratios]
        for i in range(len(angles) - 1):
            assert angles[i] > angles[i + 1]

    def test_bounded_0_to_90(self):
        for R_B in [1.5, 2.0, 5.0, 20.0]:
            theta = divergence_half_angle(R_B)
            assert 0.0 <= theta <= 90.0


class TestScreenGeometry:
    """Integration test for screen_geometry using a real Biot-Savart call."""

    def test_screen_geometry_returns_result(self):
        from helicon.fields.biot_savart import Coil
        from helicon.optimize.analytical import screen_geometry

        coils = [Coil(z=0.0, r=0.1, I=1e4)]
        result = screen_geometry(coils, z_min=-0.5, z_max=2.0, n_pts=50)
        assert isinstance(result, NozzleScreeningResult)

    def test_screen_geometry_mirror_ratio_gt1(self):
        from helicon.fields.biot_savart import Coil
        from helicon.optimize.analytical import screen_geometry

        coils = [Coil(z=0.0, r=0.1, I=1e4)]
        result = screen_geometry(coils, z_min=-0.5, z_max=2.0, n_pts=50)
        assert result.mirror_ratio > 1.0

    def test_screen_geometry_metrics_consistent(self):
        from helicon.fields.biot_savart import Coil
        from helicon.optimize.analytical import screen_geometry

        coils = [Coil(z=0.0, r=0.1, I=1e4)]
        result = screen_geometry(coils, z_min=-0.5, z_max=2.0, n_pts=50)
        assert 0.0 <= result.thrust_efficiency <= 1.0
        assert result.thrust_coefficient >= 0.0
        assert 0.0 <= result.divergence_half_angle_deg <= 90.0

    def test_higher_current_higher_mirror_ratio(self):
        """More current → stronger throat field → higher mirror ratio."""
        from helicon.fields.biot_savart import Coil
        from helicon.optimize.analytical import screen_geometry

        coils_low = [Coil(z=0.0, r=0.1, I=1e3)]
        coils_high = [Coil(z=0.0, r=0.1, I=1e5)]
        r_low = screen_geometry(coils_low, z_min=-0.5, z_max=2.0, n_pts=50).mirror_ratio
        r_high = screen_geometry(coils_high, z_min=-0.5, z_max=2.0, n_pts=50).mirror_ratio
        assert r_high > r_low


class TestBatchFunctions:
    """Tests for batch-vectorized analytical functions."""

    def test_efficiency_batch_matches_scalar(self) -> None:
        R_B = np.array([1.0, 2.0, 5.0, 10.0, 100.0])
        batch = thrust_efficiency_batch(R_B, backend="numpy")
        scalar = np.array([thrust_efficiency(r) for r in R_B])
        np.testing.assert_allclose(batch, scalar, rtol=1e-8)

    def test_coefficient_batch_matches_scalar(self) -> None:
        R_B = np.array([1.5, 3.0, 7.0])
        batch = thrust_coefficient_batch(R_B, backend="numpy")
        scalar = np.array([thrust_coefficient_paraxial(r) for r in R_B])
        np.testing.assert_allclose(batch, scalar, rtol=1e-6)

    def test_divergence_batch_matches_scalar(self) -> None:
        R_B = np.array([1.0, 2.0, 4.0, 16.0, float("inf")])
        batch = divergence_half_angle_batch(R_B, backend="numpy")
        scalar = np.array([divergence_half_angle(r) for r in R_B])
        np.testing.assert_allclose(batch, scalar, atol=1e-6)

    @skip_no_mlx
    def test_batch_efficiency_mlx_matches_numpy(self) -> None:
        R_B = np.linspace(1.5, 50.0, 100)
        eta_np = thrust_efficiency_batch(R_B, backend="numpy")
        eta_mlx = thrust_efficiency_batch(R_B, backend="mlx")
        np.testing.assert_allclose(eta_mlx, eta_np, rtol=0.01)

    @skip_no_mlx
    def test_batch_coefficient_mlx_matches_numpy(self) -> None:
        R_B = np.linspace(2.0, 30.0, 50)
        ct_np = thrust_coefficient_batch(R_B, backend="numpy")
        ct_mlx = thrust_coefficient_batch(R_B, backend="mlx")
        np.testing.assert_allclose(ct_mlx, ct_np, rtol=0.01)

    @skip_no_mlx
    def test_batch_divergence_mlx_matches_numpy(self) -> None:
        R_B = np.linspace(1.5, 50.0, 80)
        d_np = divergence_half_angle_batch(R_B, backend="numpy")
        d_mlx = divergence_half_angle_batch(R_B, backend="mlx")
        np.testing.assert_allclose(d_mlx, d_np, atol=0.1)  # degrees


@skip_no_mlx
class TestBreizmanArefievDifferentiable:
    """Tests for the differentiable end-to-end MLX analytical model."""

    def test_breizman_arefiev_ct_runs(self) -> None:
        import mlx.core as mx

        from helicon.optimize.analytical import breizman_arefiev_ct_mlx

        coil_params = mx.array([[0.0, 0.1, 50_000.0]], dtype=mx.float32)
        z_eval = mx.linspace(-0.5, 1.5, 40)
        ct = breizman_arefiev_ct_mlx(coil_params, z_eval, n_phi=32)
        mx.eval(ct)
        assert float(ct) >= 0.0

    def test_breizman_arefiev_differentiable(self) -> None:
        """mx.grad() should work on the differentiable C_T model."""
        import mlx.core as mx

        from helicon.optimize.analytical import breizman_arefiev_ct_mlx

        coil_params = mx.array([[0.0, 0.1, 50_000.0]], dtype=mx.float32)
        z_eval = mx.linspace(-0.5, 1.5, 40)

        def model(cp):
            return breizman_arefiev_ct_mlx(cp, z_eval, n_phi=32)

        grad_fn = mx.grad(model)
        grads = grad_fn(coil_params)
        mx.eval(grads)
        assert grads.shape == coil_params.shape
        # Gradient w.r.t. current should be non-zero (C_T depends on mirror ratio)
        assert float(mx.abs(grads).sum()) > 0.0

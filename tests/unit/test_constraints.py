"""Tests for helicon.optimize.constraints."""

import math

import numpy as np
import pytest

from helicon.optimize.constraints import (
    CoilConstraintResult,
    CoilConstraints,
    evaluate_constraints,
    make_constrained_objective,
)


# Single coil at z=0, r=0.1 m, I=10000 A-turns
SINGLE_COIL = np.array([[0.0, 0.1, 10000.0]])
# Two coils
TWO_COILS = np.array([[0.0, 0.1, 10000.0], [0.2, 0.15, 5000.0]])


class TestCoilConstraints:
    def test_default_values(self):
        c = CoilConstraints()
        assert c.max_total_mass_kg is None
        assert c.max_total_power_W is None
        assert c.max_B_conductor_T is None
        assert c.current_density_Am2 == pytest.approx(1e7)
        assert c.conductor_density_kg_m3 == pytest.approx(8960.0)

    def test_custom_values(self):
        c = CoilConstraints(max_total_mass_kg=50.0, max_B_conductor_T=15.0)
        assert c.max_total_mass_kg == 50.0
        assert c.max_B_conductor_T == 15.0


class TestEvaluateConstraints:
    def test_returns_result(self):
        r = evaluate_constraints(SINGLE_COIL, CoilConstraints())
        assert isinstance(r, CoilConstraintResult)

    def test_coil_count(self):
        r = evaluate_constraints(TWO_COILS, CoilConstraints())
        assert len(r.coil_masses_kg) == 2
        assert len(r.coil_powers_W) == 2
        assert len(r.coil_B_peak_T) == 2

    def test_mass_positive(self):
        r = evaluate_constraints(SINGLE_COIL, CoilConstraints())
        assert r.total_mass_kg > 0.0
        assert all(m > 0 for m in r.coil_masses_kg)

    def test_mass_formula(self):
        # m = rho_Cu * (|I| / J) * 2pi * r
        c = CoilConstraints(current_density_Am2=1e7, conductor_density_kg_m3=8960.0)
        r = evaluate_constraints(SINGLE_COIL, c)
        J = 1e7
        rho = 8960.0
        I, radius = 10000.0, 0.1
        expected = rho * (I / J) * 2 * math.pi * radius
        assert abs(r.coil_masses_kg[0] - expected) < 1e-10

    def test_b_peak_formula(self):
        # B = mu0 * |I| / (2*r)
        r = evaluate_constraints(SINGLE_COIL, CoilConstraints())
        mu0 = 4e-7 * math.pi
        expected = mu0 * 10000.0 / (2 * 0.1)
        assert abs(r.coil_B_peak_T[0] - expected) < 1e-12

    def test_power_positive(self):
        r = evaluate_constraints(SINGLE_COIL, CoilConstraints())
        assert r.total_power_W >= 0.0

    def test_power_zero_for_superconductor(self):
        c = CoilConstraints(conductor_resistivity_Ohm_m=0.0)
        r = evaluate_constraints(SINGLE_COIL, c)
        assert r.total_power_W == pytest.approx(0.0)

    def test_no_violations_unconstrained(self):
        r = evaluate_constraints(SINGLE_COIL, CoilConstraints())
        assert r.violations == {}
        assert r.satisfied
        assert r.penalty == pytest.approx(0.0)

    def test_mass_violation_detected(self):
        c = CoilConstraints(max_total_mass_kg=1e-6)  # impossibly tight
        r = evaluate_constraints(SINGLE_COIL, c)
        assert "total_mass_kg" in r.violations
        assert r.violations["total_mass_kg"] > 0
        assert not r.satisfied

    def test_b_violation_detected(self):
        c = CoilConstraints(max_B_conductor_T=1e-6)  # impossibly tight
        r = evaluate_constraints(SINGLE_COIL, c)
        assert "max_B_conductor_T" in r.violations
        assert not r.satisfied

    def test_power_violation_detected(self):
        c = CoilConstraints(max_total_power_W=1e-6)  # impossibly tight
        r = evaluate_constraints(SINGLE_COIL, c)
        assert "total_power_W" in r.violations
        assert not r.satisfied

    def test_penalty_nonzero_when_violated(self):
        c = CoilConstraints(max_total_mass_kg=1e-10)
        r = evaluate_constraints(SINGLE_COIL, c, penalty_factor=1000.0)
        assert r.penalty > 0

    def test_satisfied_when_mass_below_budget(self):
        r_unconstrained = evaluate_constraints(SINGLE_COIL, CoilConstraints())
        large_budget = r_unconstrained.total_mass_kg * 10
        c = CoilConstraints(max_total_mass_kg=large_budget)
        r = evaluate_constraints(SINGLE_COIL, c)
        assert r.satisfied
        assert r.penalty == pytest.approx(0.0)

    def test_total_mass_sum_of_coils(self):
        r = evaluate_constraints(TWO_COILS, CoilConstraints())
        assert abs(r.total_mass_kg - sum(r.coil_masses_kg)) < 1e-10

    def test_total_power_sum_of_coils(self):
        r = evaluate_constraints(TWO_COILS, CoilConstraints())
        assert abs(r.total_power_W - sum(r.coil_powers_W)) < 1e-10

    def test_mass_scales_with_current(self):
        coil_lo = np.array([[0.0, 0.1, 5000.0]])
        coil_hi = np.array([[0.0, 0.1, 10000.0]])
        c = CoilConstraints()
        r_lo = evaluate_constraints(coil_lo, c)
        r_hi = evaluate_constraints(coil_hi, c)
        assert abs(r_hi.total_mass_kg / r_lo.total_mass_kg - 2.0) < 1e-10

    def test_mass_scales_with_radius(self):
        coil_lo = np.array([[0.0, 0.05, 10000.0]])
        coil_hi = np.array([[0.0, 0.10, 10000.0]])
        c = CoilConstraints()
        r_lo = evaluate_constraints(coil_lo, c)
        r_hi = evaluate_constraints(coil_hi, c)
        assert abs(r_hi.total_mass_kg / r_lo.total_mass_kg - 2.0) < 1e-10

    def test_negative_current_same_as_positive(self):
        coil_pos = np.array([[0.0, 0.1, 10000.0]])
        coil_neg = np.array([[0.0, 0.1, -10000.0]])
        c = CoilConstraints()
        r_pos = evaluate_constraints(coil_pos, c)
        r_neg = evaluate_constraints(coil_neg, c)
        assert abs(r_pos.total_mass_kg - r_neg.total_mass_kg) < 1e-10
        assert abs(r_pos.coil_B_peak_T[0] - r_neg.coil_B_peak_T[0]) < 1e-12


class TestMakeConstrainedObjective:
    """Tests for the MLX-differentiable penalized objective factory."""

    @pytest.fixture(autouse=True)
    def skip_without_mlx(self):
        pytest.importorskip("mlx.core", reason="MLX not available")

    def test_returns_callable(self):
        import mlx.core as mx
        from helicon.optimize.objectives import throat_ratio_objective

        grid_r = mx.array([0.0, 0.05, 0.1])
        grid_z = mx.array([-0.2, 0.0, 0.2])
        objective = lambda cp: throat_ratio_objective(cp, grid_r, grid_z)

        c = CoilConstraints(max_total_mass_kg=1000.0)
        fn = make_constrained_objective(objective, c)
        assert callable(fn)

    def test_satisfied_constraint_no_penalty(self):
        import mlx.core as mx
        from helicon.optimize.objectives import throat_ratio_objective

        grid_r = mx.array([0.0, 0.05, 0.1])
        grid_z = mx.array([-0.2, 0.0, 0.2])
        coil_params = mx.array([[0.0, 0.1, 10000.0]])

        def objective(cp):
            return throat_ratio_objective(cp, grid_r, grid_z)

        # Budget well above actual mass → no penalty
        r_unconstrained = evaluate_constraints(np.array([[0.0, 0.1, 10000.0]]), CoilConstraints())
        large_budget = r_unconstrained.total_mass_kg * 10
        c = CoilConstraints(max_total_mass_kg=large_budget)
        fn_unconstrained = make_constrained_objective(lambda cp: mx.array(1.0), CoilConstraints())
        fn_constrained = make_constrained_objective(lambda cp: mx.array(1.0), c)
        mx.eval(coil_params)
        val_unc = float(fn_unconstrained(coil_params))
        val_con = float(fn_constrained(coil_params))
        assert abs(val_unc - val_con) < 1e-8

    def test_violated_constraint_adds_penalty(self):
        import mlx.core as mx

        coil_params = mx.array([[0.0, 0.1, 10000.0]])
        c = CoilConstraints(max_total_mass_kg=1e-10)  # impossible
        fn = make_constrained_objective(lambda cp: mx.array(0.0), c, penalty_factor=1000.0)
        val = float(fn(coil_params))
        assert val > 0.0

    def test_differentiable(self):
        import mlx.core as mx

        coil_params = mx.array([[0.0, 0.1, 10000.0]])
        c = CoilConstraints(max_total_mass_kg=1e-10)
        fn = make_constrained_objective(lambda cp: mx.array(0.0), c)
        grad_fn = mx.grad(fn)
        g = grad_fn(coil_params)
        mx.eval(g)
        assert g.shape == coil_params.shape

    def test_multiple_constraints_combined(self):
        import mlx.core as mx

        coil_params = mx.array([[0.0, 0.1, 10000.0]])
        # Both mass and B field constraints violated
        c = CoilConstraints(max_total_mass_kg=1e-10, max_B_conductor_T=1e-10)
        fn = make_constrained_objective(lambda cp: mx.array(0.0), c, penalty_factor=1000.0)
        val = float(fn(coil_params))
        # Penalty from two violations should be larger than one
        c_single = CoilConstraints(max_total_mass_kg=1e-10)
        fn_single = make_constrained_objective(lambda cp: mx.array(0.0), c_single, penalty_factor=1000.0)
        val_single = float(fn_single(coil_params))
        assert val > val_single

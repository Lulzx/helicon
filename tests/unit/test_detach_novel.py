"""Tests for v2.5 novel detachment contributions.

Five independently novel scientific contributions:
1. calibration.py  — BCE weight calibration + synthetic data generation
2. kinetic.py      — FLR corrections beyond MHD
3. inverse.py      — closed-form M_A from thrust observables
4. control.py      — Lyapunov-stable feedback controller
5. sheath.py       — non-equilibrium sheath coupling correction
"""

from __future__ import annotations

import math

import pytest

from helicon.detach.calibration import (
    CalibrationRecord,
    CalibrationResult,
    DetachmentCalibrator,
)
from helicon.detach.control import (
    ControlState,
    ControlUpdate,
    LyapunovController,
)
from helicon.detach.invariants import (
    alfven_mach as ideal_mach,
)
from helicon.detach.invariants import (
    bohm_velocity,
    ion_larmor_radius,
    ion_magnetization,
)
from helicon.detach.inverse import (
    InferredState,
    ThrustInverter,
    ThrustObservation,
)
from helicon.detach.kinetic import (
    alfven_mach_kinetic,
    bohm_velocity_full,
    flr_correction_factor,
    ion_inertial_length,
    ion_magnetization_flr,
    larmor_radius_maxwellian,
)
from helicon.detach.model import PlasmaState
from helicon.detach.sheath import (
    SheathCorrectedState,
    apply_sheath_correction,
    debye_length,
    electric_to_mirror_ratio,
    sheath_potential,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plasma_state(n=1e18, Te=30.0, Ti=15.0, B=0.05, dBdz=-1.0, vz=4e4):
    return PlasmaState(
        n_m3=n,
        Te_eV=Te,
        Ti_eV=Ti,
        B_T=B,
        dBdz_T_per_m=dBdz,
        vz_ms=vz,
    )


# ---------------------------------------------------------------------------
# 1. Calibration
# ---------------------------------------------------------------------------


class TestCalibrationRecord:
    def test_fields(self):
        r = CalibrationRecord(
            alfven_mach=1.2,
            electron_beta=0.2,
            ion_magnetization=0.8,
            is_detached=True,
            source="test",
        )
        assert r.alfven_mach == pytest.approx(1.2)
        assert r.is_detached is True
        assert r.source == "test"

    def test_default_source(self):
        r = CalibrationRecord(0.5, 0.05, 0.3, False)
        assert r.source == ""


class TestDetachmentCalibrator:
    def test_synthetic_data_count(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=200, seed=42)
        assert len(records) == 200

    def test_synthetic_data_types(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=10, seed=0)
        for r in records:
            assert isinstance(r.alfven_mach, float)
            assert isinstance(r.electron_beta, float)
            assert isinstance(r.ion_magnetization, float)
            assert isinstance(r.is_detached, bool)
            assert r.source == "synthetic"

    def test_synthetic_data_ranges(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=300, seed=1)
        for r in records:
            assert 0.1 <= r.alfven_mach <= 3.0
            assert 0.001 <= r.electron_beta <= 0.50
            assert 0.0 <= r.ion_magnetization <= 2.0

    def test_synthetic_data_reproducible(self):
        r1 = DetachmentCalibrator.generate_synthetic_data(n_samples=50, seed=7)
        r2 = DetachmentCalibrator.generate_synthetic_data(n_samples=50, seed=7)
        for a, b in zip(r1, r2):
            assert a.alfven_mach == pytest.approx(b.alfven_mach)
            assert a.is_detached == b.is_detached

    def test_synthetic_data_different_seeds(self):
        r1 = DetachmentCalibrator.generate_synthetic_data(n_samples=50, seed=0)
        r2 = DetachmentCalibrator.generate_synthetic_data(n_samples=50, seed=99)
        mach_1 = [r.alfven_mach for r in r1]
        mach_2 = [r.alfven_mach for r in r2]
        assert mach_1 != mach_2

    def test_fit_returns_calibration_result(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=100, seed=0)
        cal = DetachmentCalibrator()
        result = cal.fit(records)
        assert isinstance(result, CalibrationResult)

    def test_fit_weights_sum_to_one(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=100, seed=0)
        result = DetachmentCalibrator().fit(records)
        total = result.w_alfven + result.w_beta + result.w_ion_mag
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_fit_weights_non_negative(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=100, seed=0)
        result = DetachmentCalibrator().fit(records)
        assert result.w_alfven >= 0.0
        assert result.w_beta >= 0.0
        assert result.w_ion_mag >= 0.0

    def test_fit_threshold_in_range(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=100, seed=0)
        result = DetachmentCalibrator().fit(records)
        assert 0.01 <= result.score_threshold <= 0.99

    def test_fit_accuracy_reasonable(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=500, seed=0)
        result = DetachmentCalibrator().fit(records)
        # With 500 samples the calibrator should beat random (>50%)
        assert result.accuracy > 0.5

    def test_fit_log_loss_finite(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=100, seed=0)
        result = DetachmentCalibrator().fit(records)
        assert math.isfinite(result.log_loss)
        assert result.log_loss >= 0.0

    def test_to_model_kwargs(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=100, seed=0)
        result = DetachmentCalibrator().fit(records)
        kwargs = result.to_model_kwargs()
        assert set(kwargs.keys()) == {"w_alfven", "w_beta", "w_ion_mag", "score_detached"}

    def test_summary_str(self):
        records = DetachmentCalibrator.generate_synthetic_data(n_samples=50, seed=0)
        result = DetachmentCalibrator().fit(records)
        s = result.summary()
        assert "CalibrationResult" in s
        assert "w_" in s

    def test_calibrated_model_usable(self):
        from helicon.detach.model import DetachmentOnsetModel

        records = DetachmentCalibrator.generate_synthetic_data(n_samples=200, seed=0)
        result = DetachmentCalibrator().fit(records)
        model = DetachmentOnsetModel(**result.to_model_kwargs())
        state = _plasma_state()
        ds = model.assess(state)
        assert 0.0 <= ds.detachment_score <= 1.0


# ---------------------------------------------------------------------------
# 2. Kinetic FLR corrections
# ---------------------------------------------------------------------------


class TestKineticFLR:
    # larmor_radius_maxwellian
    def test_larmor_maxwellian_positive(self):
        r = larmor_radius_maxwellian(20.0, 0.05, 1.0)
        assert r > 0.0

    def test_larmor_maxwellian_lt_thermal(self):
        r_thermal = ion_larmor_radius(20.0, 0.05, 1.0)
        r_max = larmor_radius_maxwellian(20.0, 0.05, 1.0)
        # Maxwellian-averaged ≈ 0.886 × thermal
        assert r_max < r_thermal
        assert r_max == pytest.approx(r_thermal * math.sqrt(math.pi / 4.0), rel=1e-6)

    def test_larmor_maxwellian_scales_with_sqrt_T(self):
        r1 = larmor_radius_maxwellian(10.0, 0.1, 1.0)
        r4 = larmor_radius_maxwellian(40.0, 0.1, 1.0)
        assert r4 == pytest.approx(r1 * 2.0, rel=1e-5)

    # flr_correction_factor
    def test_flr_factor_at_zero(self):
        assert flr_correction_factor(0.0) == pytest.approx(1.0)

    def test_flr_factor_at_half(self):
        # √(1 + 0.75 × 0.25) = √(1.1875) ≈ 1.0898
        expected = math.sqrt(1.0 + 0.75 * 0.25)
        assert flr_correction_factor(0.5) == pytest.approx(expected, rel=1e-6)

    def test_flr_factor_at_one(self):
        expected = math.sqrt(1.0 + 0.75)
        assert flr_correction_factor(1.0) == pytest.approx(expected, rel=1e-6)

    def test_flr_factor_inf(self):
        assert flr_correction_factor(math.inf) == math.inf

    def test_flr_factor_monotone(self):
        vals = [flr_correction_factor(x) for x in [0.0, 0.5, 1.0, 2.0]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    # ion_magnetization_flr
    def test_ion_mag_flr_gt_standard_large_lambda(self):
        # FLR correction only exceeds cold-beam estimate when Lambda_std > ~0.68
        # (Northrop factor must overcome the Maxwellian √(π/4) reduction).
        # Use high Ti / steep gradient so Lambda_std ~ 4.
        Lambda_std = ion_magnetization(200.0, 0.05, -5.0, 1.0)
        Lambda_flr = ion_magnetization_flr(200.0, 0.05, -5.0, 1.0)
        assert Lambda_std > 0.68  # ensure we're in the correct regime
        assert Lambda_flr >= Lambda_std

    def test_ion_mag_flr_lt_standard_small_lambda(self):
        # For small Lambda (< 0.68) the Maxwellian factor dominates
        Lambda_std = ion_magnetization(20.0, 0.05, -1.0, 1.0)
        Lambda_flr = ion_magnetization_flr(20.0, 0.05, -1.0, 1.0)
        assert Lambda_std < 0.68  # confirm small-Lambda regime
        assert Lambda_flr < Lambda_std

    def test_ion_mag_flr_zero_grad(self):
        # Zero gradient → L_B = inf → Λ₀ = 0 → no demagnetization → returns 0.0
        result = ion_magnetization_flr(10.0, 0.1, 0.0, 1.0)
        assert result == pytest.approx(0.0)

    # alfven_mach_kinetic
    def test_kinetic_mach_lt_ideal(self):
        B, n, m = 0.05, 1e18, 1.0
        vz = 3e4
        M_ideal = ideal_mach(vz, B, n, m)
        M_kinetic = alfven_mach_kinetic(vz, B, n, m, Ti_eV=20.0, dBdz_T_per_m=-1.0)
        # kinetic wave is faster → M_kAW ≤ M_ideal
        assert M_kinetic <= M_ideal

    def test_kinetic_mach_positive(self):
        M = alfven_mach_kinetic(1e4, 0.1, 1e18, 1.0, Ti_eV=10.0, dBdz_T_per_m=-2.0)
        assert M >= 0.0

    def test_kinetic_mach_zero_density(self):
        # n=0 → va_A=inf → v_kAW=inf → M_kAW = vz/inf = 0
        M = alfven_mach_kinetic(1e4, 0.1, 0.0, 1.0, Ti_eV=10.0, dBdz_T_per_m=-2.0)
        assert pytest.approx(0.0) == M

    # bohm_velocity_full
    def test_bohm_full_gt_cold(self):
        c_cold = bohm_velocity(30.0, 1.0)
        c_full = bohm_velocity_full(30.0, 30.0, 1.0)
        assert c_full > c_cold

    def test_bohm_full_ratio(self):
        # T_i = T_e → c_full = √2 × c_cold
        c_cold = bohm_velocity_full(20.0, 0.0, 1.0)
        c_full = bohm_velocity_full(20.0, 20.0, 1.0)
        assert c_full == pytest.approx(math.sqrt(2.0) * c_cold, rel=1e-6)

    # ion_inertial_length
    def test_inertial_length_positive(self):
        d_i = ion_inertial_length(1e18, 1.0)
        assert d_i > 0.0

    def test_inertial_length_zero_density(self):
        d_i = ion_inertial_length(0.0, 1.0)
        assert math.inf == d_i

    def test_inertial_length_scales(self):
        d_i_low = ion_inertial_length(1e17, 1.0)
        d_i_high = ion_inertial_length(1e19, 1.0)
        assert d_i_low > d_i_high  # lower density → longer inertial length


# ---------------------------------------------------------------------------
# 3. Inverse problem
# ---------------------------------------------------------------------------


class TestThrustObservation:
    def test_defaults(self):
        obs = ThrustObservation(
            F_thrust_N=0.1,
            m_dot_kg_s=1e-5,
            B_throat_T=0.05,
            A_throat_m2=1e-4,
        )
        assert obs.mass_amu == pytest.approx(1.0)
        assert obs.Te_eV_nominal == pytest.approx(50.0)
        assert obs.dBdz_T_per_m == pytest.approx(-2.0)


class TestThrustInverter:
    def _typical_obs(self):
        return ThrustObservation(
            F_thrust_N=0.05,
            m_dot_kg_s=1e-5,
            B_throat_T=0.03,
            A_throat_m2=5e-4,
            mass_amu=1.0,
            Te_eV_nominal=30.0,
            dBdz_T_per_m=-1.5,
        )

    def test_bad_mirror_ratio(self):
        with pytest.raises(ValueError):
            ThrustInverter(mirror_ratio=0.5)

    def test_eta_T_formula(self):
        inv = ThrustInverter(mirror_ratio=4.0)
        assert inv.eta_T == pytest.approx(1.0 - 0.5, rel=1e-6)

    def test_invert_returns_inferred_state(self):
        inv = ThrustInverter()
        result = inv.invert(self._typical_obs())
        assert isinstance(result, InferredState)

    def test_invert_vz_positive(self):
        result = ThrustInverter().invert(self._typical_obs())
        assert result.vz_ms > 0.0

    def test_invert_density_positive(self):
        result = ThrustInverter().invert(self._typical_obs())
        assert result.n_m3 > 0.0

    def test_invert_mach_valid(self):
        result = ThrustInverter().invert(self._typical_obs())
        assert result.alfven_mach > 0.0 or result.alfven_mach == -1.0

    def test_invert_score_in_range(self):
        result = ThrustInverter().invert(self._typical_obs())
        assert 0.0 <= result.detachment_score <= 1.0

    def test_invert_confidence_in_range(self):
        result = ThrustInverter().invert(self._typical_obs())
        assert 0.0 <= result.confidence <= 1.0

    def test_invert_residual_near_zero(self):
        # By construction F_model = ṁ × v_ex × η_T = F_thrust exactly
        result = ThrustInverter().invert(self._typical_obs())
        assert result.residual == pytest.approx(0.0, abs=1e-10)

    def test_invert_zero_thrust(self):
        obs = ThrustObservation(0.0, 1e-5, 0.03, 5e-4)
        result = ThrustInverter().invert(obs)
        assert result.n_m3 == pytest.approx(0.0)

    def test_invert_zero_mdot(self):
        obs = ThrustObservation(0.05, 0.0, 0.03, 5e-4)
        result = ThrustInverter().invert(obs)
        assert result.vz_ms == pytest.approx(0.0)

    def test_to_plasma_state(self):
        result = ThrustInverter().invert(self._typical_obs())
        ps = result.to_plasma_state(B_T=0.03, dBdz_T_per_m=-1.5)
        assert isinstance(ps, PlasmaState)
        assert ps.n_m3 == pytest.approx(result.n_m3)

    def test_gradient_test_attached(self):
        obs_lo = ThrustObservation(0.04, 1e-5, 0.02, 5e-4)
        obs_hi = ThrustObservation(0.08, 1e-5, 0.04, 5e-4)
        result = ThrustInverter().gradient_test(obs_lo, obs_hi)
        assert "dF_dB" in result
        assert "regime" in result
        assert "optimal_B_T" in result
        assert result["dF_dB"] > 0.0
        assert result["regime"] == "attached"

    def test_gradient_test_detached(self):
        obs_lo = ThrustObservation(0.08, 1e-5, 0.02, 5e-4)
        obs_hi = ThrustObservation(0.04, 1e-5, 0.04, 5e-4)
        result = ThrustInverter().gradient_test(obs_lo, obs_hi)
        assert result["regime"] == "detached"

    def test_gradient_test_equal_B(self):
        obs_lo = ThrustObservation(0.06, 1e-5, 0.03, 5e-4)
        obs_hi = ThrustObservation(0.06, 1e-5, 0.03, 5e-4)
        result = ThrustInverter().gradient_test(obs_lo, obs_hi)
        assert result["dF_dB"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Lyapunov control
# ---------------------------------------------------------------------------


class TestLyapunovController:
    def test_bad_setpoint_low(self):
        with pytest.raises(ValueError):
            LyapunovController(setpoint=0.0)

    def test_bad_setpoint_high(self):
        with pytest.raises(ValueError):
            LyapunovController(setpoint=1.0)

    def test_bad_decay_rate(self):
        with pytest.raises(ValueError):
            LyapunovController(decay_rate=-1.0)

    def test_bad_dB_dI(self):
        with pytest.raises(ValueError):
            LyapunovController(dB_dI_T_per_A=0.0)

    def test_step_returns_control_update(self):
        ctrl = LyapunovController()
        cs = ControlState(I_coil_A=1000.0)
        u = ctrl.step(_plasma_state(), cs)
        assert isinstance(u, ControlUpdate)

    def test_step_lyapunov_V_non_negative(self):
        ctrl = LyapunovController()
        cs = ControlState(I_coil_A=500.0)
        u = ctrl.step(_plasma_state(), cs)
        assert u.lyapunov_V >= 0.0

    def test_step_lyapunov_dV_dt_non_positive(self):
        ctrl = LyapunovController()
        cs = ControlState(I_coil_A=500.0)
        u = ctrl.step(_plasma_state(), cs)
        assert u.lyapunov_dV_dt <= 1e-12  # should be ≤ 0

    def test_step_updates_ctrl_state(self):
        ctrl = LyapunovController()
        cs = ControlState(I_coil_A=0.0)
        ctrl.step(_plasma_state(), cs, dt_s=0.01)
        assert cs.t_s == pytest.approx(0.01)
        assert len(cs.score_history) == 1
        assert len(cs.error_history) == 1

    def test_step_current_clamped(self):
        ctrl = LyapunovController(I_coil_max_A=1000.0, I_coil_min_A=0.0)
        cs = ControlState(I_coil_A=999.0)
        u = ctrl.step(_plasma_state(), cs)
        assert u.new_I_coil_A <= 1000.0
        assert u.new_I_coil_A >= 0.0

    def test_step_score_in_range(self):
        ctrl = LyapunovController()
        cs = ControlState()
        u = ctrl.step(_plasma_state(), cs)
        assert 0.0 <= u.score <= 1.0

    def test_stability_certificate_keys(self):
        ctrl = LyapunovController()
        cert = ctrl.stability_certificate(_plasma_state())
        expected = {"V", "dV_dt", "is_stable", "convergence_time_s", "grad_S_B", "error"}
        assert expected <= set(cert)

    def test_stability_certificate_V_non_negative(self):
        ctrl = LyapunovController()
        cert = ctrl.stability_certificate(_plasma_state())
        assert cert["V"] >= 0.0

    def test_stability_certificate_convergence_time(self):
        ctrl = LyapunovController(decay_rate=2.0)
        cert = ctrl.stability_certificate(_plasma_state())
        assert cert["convergence_time_s"] == pytest.approx(1.0 / 4.0)

    def test_stability_certificate_is_stable(self):
        ctrl = LyapunovController()
        cert = ctrl.stability_certificate(_plasma_state())
        assert isinstance(cert["is_stable"], bool)
        assert cert["is_stable"]  # should be stable with correct control law

    def test_simulate_length(self):
        ctrl = LyapunovController()
        updates = ctrl.simulate(_plasma_state(), n_steps=10)
        assert len(updates) == 10

    def test_simulate_V_decreasing(self):
        ctrl = LyapunovController()
        updates = ctrl.simulate(_plasma_state(), n_steps=20, dt_s=0.01)
        # Lyapunov V should generally decrease over time
        first_V = updates[0].lyapunov_V
        last_V = updates[-1].lyapunov_V
        assert last_V <= first_V + 1e-6  # allow numerical tolerance

    def test_grad_S_B_negative(self):
        ctrl = LyapunovController()
        ds = ctrl.model.assess(_plasma_state())
        grad = ctrl._grad_S_B(_plasma_state(), ds)
        # More B → lower S, so gradient should be ≤ 0
        assert grad <= 0.0

    def test_simulate_all_control_updates(self):
        ctrl = LyapunovController()
        updates = ctrl.simulate(_plasma_state(), n_steps=5)
        for u in updates:
            assert isinstance(u, ControlUpdate)
            assert u.lyapunov_V >= 0.0


# ---------------------------------------------------------------------------
# 5. Sheath coupling
# ---------------------------------------------------------------------------


class TestSheathPhysics:
    # debye_length
    def test_debye_positive(self):
        lam = debye_length(1e18, 20.0)
        assert lam > 0.0

    def test_debye_zero_density(self):
        assert debye_length(0.0, 20.0) == math.inf

    def test_debye_zero_temperature(self):
        assert debye_length(1e18, 0.0) == math.inf

    def test_debye_scales_with_sqrt_T(self):
        lam_20 = debye_length(1e18, 20.0)
        lam_80 = debye_length(1e18, 80.0)
        assert lam_80 == pytest.approx(lam_20 * 2.0, rel=1e-5)

    def test_debye_scales_inversely_with_sqrt_n(self):
        lam_low = debye_length(1e17, 20.0)
        lam_high = debye_length(1e19, 20.0)
        assert lam_low == pytest.approx(lam_high * 10.0, rel=1e-5)

    # sheath_potential
    def test_sheath_potential_positive_H(self):
        phi = sheath_potential(20.0, 1.0)
        assert phi > 0.0

    def test_sheath_potential_H_approx(self):
        # H⁺: Φ_s ≈ 2.84 T_e/e → for T_e=1eV: ≈ 2.84 V
        phi = sheath_potential(1.0, 1.0)
        assert 2.5 < phi < 3.2

    def test_sheath_potential_Ar_gt_H(self):
        phi_H = sheath_potential(20.0, 1.0)
        phi_Ar = sheath_potential(20.0, 40.0)
        assert phi_Ar > phi_H

    def test_sheath_potential_scales_with_Te(self):
        phi_10 = sheath_potential(10.0, 1.0)
        phi_20 = sheath_potential(20.0, 1.0)
        assert phi_20 == pytest.approx(2.0 * phi_10, rel=1e-6)

    # electric_to_mirror_ratio
    def test_eps_ES_non_negative(self):
        eps = electric_to_mirror_ratio(30.0, 15.0, 0.05, -1.0, 1.0)
        assert eps >= 0.0

    def test_eps_ES_zero_gradient(self):
        # Zero gradient → infinite L_B → Lambda_i → 0 → eps_ES clamped
        eps = electric_to_mirror_ratio(30.0, 15.0, 0.05, 0.0, 1.0)
        assert eps == pytest.approx(0.0)

    def test_eps_ES_formula(self):
        # For known Lambda_i, check formula: ε_ES = (Te/Ti) / Lambda_i²
        Te, Ti, B, dBdz, m = 30.0, 15.0, 0.05, -2.0, 1.0
        Lambda_i = ion_magnetization(Ti, B, dBdz, m)
        expected = (Te / Ti) / (Lambda_i**2)
        eps = electric_to_mirror_ratio(Te, Ti, B, dBdz, m)
        assert eps == pytest.approx(expected, rel=1e-6)

    # apply_sheath_correction
    def test_correction_returns_state(self):
        state = apply_sheath_correction(
            score_raw=0.6,
            n_m3=1e18,
            Te_eV=30.0,
            Ti_eV=15.0,
            B_T=0.05,
            dBdz_T_per_m=-1.0,
            mass_amu=1.0,
        )
        assert isinstance(state, SheathCorrectedState)

    def test_correction_reduces_score(self):
        state = apply_sheath_correction(
            score_raw=0.6,
            n_m3=1e18,
            Te_eV=30.0,
            Ti_eV=15.0,
            B_T=0.05,
            dBdz_T_per_m=-1.0,
            mass_amu=1.0,
        )
        # With ε_ES > 0, S_corr ≤ S_raw
        assert state.score_corrected <= state.score_raw + 1e-10

    def test_correction_score_in_range(self):
        state = apply_sheath_correction(
            score_raw=0.8,
            n_m3=1e18,
            Te_eV=30.0,
            Ti_eV=15.0,
            B_T=0.05,
            dBdz_T_per_m=-1.0,
            mass_amu=1.0,
        )
        assert 0.0 <= state.score_corrected <= 1.0

    def test_correction_zero_coupling(self):
        state = apply_sheath_correction(
            score_raw=0.5,
            n_m3=1e18,
            Te_eV=20.0,
            Ti_eV=10.0,
            B_T=0.05,
            dBdz_T_per_m=-1.0,
            mass_amu=1.0,
            coupling_factor=0.0,
        )
        # No coupling → no correction
        assert state.score_corrected == pytest.approx(0.5)

    def test_correction_fraction_in_range(self):
        state = apply_sheath_correction(
            score_raw=0.7,
            n_m3=1e18,
            Te_eV=30.0,
            Ti_eV=10.0,
            B_T=0.05,
            dBdz_T_per_m=-2.0,
            mass_amu=1.0,
        )
        assert 0.0 <= state.correction_fraction <= 1.0

    def test_correction_zero_score_raw(self):
        state = apply_sheath_correction(
            score_raw=0.0,
            n_m3=1e18,
            Te_eV=30.0,
            Ti_eV=15.0,
            B_T=0.05,
            dBdz_T_per_m=-1.0,
            mass_amu=1.0,
        )
        assert state.score_corrected == pytest.approx(0.0)
        assert state.correction_fraction == pytest.approx(0.0)

    def test_correction_debye_length_stored(self):
        state = apply_sheath_correction(
            score_raw=0.5,
            n_m3=1e18,
            Te_eV=20.0,
            Ti_eV=10.0,
            B_T=0.05,
            dBdz_T_per_m=-1.0,
            mass_amu=1.0,
        )
        assert state.debye_length_m == pytest.approx(debye_length(1e18, 20.0), rel=1e-6)

    def test_correction_sheath_potential_stored(self):
        state = apply_sheath_correction(
            score_raw=0.5,
            n_m3=1e18,
            Te_eV=20.0,
            Ti_eV=10.0,
            B_T=0.05,
            dBdz_T_per_m=-1.0,
            mass_amu=1.0,
        )
        assert state.sheath_potential_V == pytest.approx(sheath_potential(20.0, 1.0), rel=1e-6)

    def test_correction_lambda_i_corrected_le_raw(self):
        state = apply_sheath_correction(
            score_raw=0.6,
            n_m3=1e18,
            Te_eV=40.0,
            Ti_eV=10.0,
            B_T=0.05,
            dBdz_T_per_m=-2.0,
            mass_amu=1.0,
        )
        lambda_raw = ion_magnetization(10.0, 0.05, -2.0, 1.0)
        # Effective Λ after correction should be ≤ raw Λ
        assert state.lambda_i_corrected <= lambda_raw + 1e-10

    def test_high_coupling_more_correction(self):
        state_low = apply_sheath_correction(
            0.8, 1e18, 40.0, 10.0, 0.05, -2.0, 1.0, coupling_factor=0.1
        )
        state_high = apply_sheath_correction(
            0.8, 1e18, 40.0, 10.0, 0.05, -2.0, 1.0, coupling_factor=0.9
        )
        assert state_high.score_corrected <= state_low.score_corrected


# ---------------------------------------------------------------------------
# Integration: chain calibration → kinetic → control
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_calibrate_then_control(self):
        """Calibrated model used in Lyapunov controller works end-to-end."""
        from helicon.detach.model import DetachmentOnsetModel

        records = DetachmentCalibrator.generate_synthetic_data(n_samples=200, seed=42)
        result = DetachmentCalibrator().fit(records)
        model = DetachmentOnsetModel(**result.to_model_kwargs())

        ctrl = LyapunovController(model=model)
        cs = ControlState()
        state = _plasma_state()
        u = ctrl.step(state, cs)
        assert isinstance(u, ControlUpdate)
        assert u.lyapunov_V >= 0.0

    def test_inverse_then_sheath(self):
        """Infer state from thrust, then apply sheath correction."""
        obs = ThrustObservation(
            F_thrust_N=0.05,
            m_dot_kg_s=1e-5,
            B_throat_T=0.03,
            A_throat_m2=5e-4,
        )
        inferred = ThrustInverter().invert(obs)
        state = apply_sheath_correction(
            score_raw=inferred.detachment_score,
            n_m3=max(inferred.n_m3, 1e10),
            Te_eV=obs.Te_eV_nominal,
            Ti_eV=obs.Te_eV_nominal,
            B_T=obs.B_throat_T,
            dBdz_T_per_m=obs.dBdz_T_per_m,
            mass_amu=obs.mass_amu,
        )
        assert 0.0 <= state.score_corrected <= 1.0

    def test_full_pipeline(self):
        """Calibrate → infer from thrust → kinetic correction → Lyapunov control."""
        from helicon.detach.model import DetachmentOnsetModel

        # Step 1: calibrate
        records = DetachmentCalibrator.generate_synthetic_data(100, seed=0)
        cal_result = DetachmentCalibrator().fit(records)
        model = DetachmentOnsetModel(**cal_result.to_model_kwargs())

        # Step 2: infer from thrust
        obs = ThrustObservation(
            F_thrust_N=0.05,
            m_dot_kg_s=1e-5,
            B_throat_T=0.03,
            A_throat_m2=5e-4,
        )
        inferred = ThrustInverter().invert(obs)

        # Step 3: kinetic Alfvén correction
        M_kinetic = alfven_mach_kinetic(
            vz_ms=inferred.vz_ms,
            B_T=obs.B_throat_T,
            n_m3=max(inferred.n_m3, 1e10),
            mass_amu=obs.mass_amu,
            Ti_eV=obs.Te_eV_nominal,
            dBdz_T_per_m=obs.dBdz_T_per_m,
        )

        # Step 4: Lyapunov control
        ctrl = LyapunovController(model=model)
        ps = inferred.to_plasma_state(obs.B_throat_T, obs.dBdz_T_per_m)
        cert = ctrl.stability_certificate(ps)

        assert M_kinetic >= 0.0
        assert cert["is_stable"]

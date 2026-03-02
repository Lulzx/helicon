"""Tests for helicon.mission — throttle curves, trajectory, spacecraft, pulsed."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from helicon.mission.pulsed import (
    PulsedProfile,
    burst_delta_v,
    compute_pulsed_performance,
)
from helicon.mission.spacecraft import (
    compute_backflow,
    compute_magnetic_torque,
    compute_spacecraft_charging,
)
from helicon.mission.throttle import ThrottleMap, generate_throttle_map
from helicon.mission.trajectory import (
    MissionLeg,
    OrbitTransferResult,
    analyze_mission,
    circular_transfer,
    earth_mars_dv,
    edelbaum_dv,
    poliastro_trajectory,
    tsiolkovsky,
)

# ---------------------------------------------------------------------------
# ThrottleMap construction helpers
# ---------------------------------------------------------------------------


def _make_minimal_throttle_map() -> ThrottleMap:
    """Construct a minimal ThrottleMap without needing a full SimConfig."""
    power = np.geomspace(1e3, 1e6, 5)
    mdot = np.geomspace(1e-6, 1e-4, 5)
    g0 = 9.80665
    eta = 0.65
    PP, MM = np.meshgrid(power, mdot, indexing="ij")
    ve = np.sqrt(2.0 * eta * PP / MM)
    thrust = MM * ve * 0.9
    isp = ve * 0.9 / g0
    return ThrottleMap(
        power_grid_W=power,
        mdot_grid_kgs=mdot,
        thrust_N=thrust,
        isp_s=isp,
        eta_d=np.full_like(thrust, 0.75),
        mirror_ratio=3.0,
        eta_thermal=eta,
    )


# ---------------------------------------------------------------------------
# ThrottleMap unit tests
# ---------------------------------------------------------------------------


class TestThrottleMap:
    def test_shape(self):
        tm = _make_minimal_throttle_map()
        assert tm.thrust_N.shape == (5, 5)
        assert tm.isp_s.shape == (5, 5)

    def test_higher_power_higher_isp(self):
        """For fixed mdot, higher power → higher Isp."""
        tm = _make_minimal_throttle_map()
        # Fix mdot at centre; Isp should increase with power
        isp_low = tm.isp_at(1e3, 1e-5)
        isp_high = tm.isp_at(1e6, 1e-5)
        assert isp_high > isp_low

    def test_higher_mdot_higher_thrust(self):
        """For fixed power, higher mdot → higher thrust."""
        tm = _make_minimal_throttle_map()
        t_low = tm.thrust_at(1e5, 1e-6)
        t_high = tm.thrust_at(1e5, 1e-4)
        assert t_high > t_low

    def test_interpolator_roundtrip(self):
        """Interpolate at grid nodes should reproduce grid values."""
        tm = _make_minimal_throttle_map()
        for i in range(len(tm.power_grid_W)):
            for j in range(len(tm.mdot_grid_kgs)):
                p = tm.power_grid_W[i]
                m = tm.mdot_grid_kgs[j]
                assert abs(tm.thrust_at(p, m) - tm.thrust_N[i, j]) < 1e-6

    def test_json_roundtrip(self):
        tm = _make_minimal_throttle_map()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tm.json"
            tm.save_json(path)
            tm2 = ThrottleMap.load_json(path)
        np.testing.assert_allclose(tm2.thrust_N, tm.thrust_N)
        np.testing.assert_allclose(tm2.isp_s, tm.isp_s)
        assert tm2.mirror_ratio == tm.mirror_ratio

    def test_to_dict_keys(self):
        tm = _make_minimal_throttle_map()
        d = tm.to_dict()
        assert "thrust_N" in d
        assert "isp_s" in d
        assert "mirror_ratio" in d


class TestGenerateThrottleMap:
    """Integration test: generate_throttle_map from SimConfig."""

    def _make_config(self):
        from helicon.config.parser import SimConfig

        return SimConfig.model_validate(
            {
                "nozzle": {
                    "type": "solenoid",
                    "coils": [{"z": 0.0, "r": 0.1, "I": 40000}],
                    "domain": {"z_min": -0.3, "z_max": 2.0, "r_max": 0.5},
                },
                "plasma": {
                    "n0": 1e18,
                    "T_i_eV": 100,
                    "T_e_eV": 100,
                    "v_injection_ms": 50000,
                },
            }
        )

    def test_generate_small_grid(self):
        config = self._make_config()
        tm = generate_throttle_map(
            config,
            power_range_W=(1e4, 1e6),
            mdot_range_kgs=(1e-6, 1e-4),
            n_power=3,
            n_mdot=3,
            n_pts_mirror=20,
        )
        assert tm.thrust_N.shape == (3, 3)
        assert tm.mirror_ratio > 1.0
        assert 0.0 < float(tm.eta_d.mean()) <= 1.0
        assert float(tm.isp_s.max()) > 1000  # expect > 1000 s for these parameters

    def test_symmetry_eta_d(self):
        """eta_d is constant over the grid (only geometry-dependent)."""
        config = self._make_config()
        tm = generate_throttle_map(
            config,
            power_range_W=(1e4, 1e5),
            mdot_range_kgs=(1e-6, 1e-5),
            n_power=4,
            n_mdot=4,
            n_pts_mirror=10,
        )
        # All eta_d values should be equal (geometry-only)
        np.testing.assert_allclose(tm.eta_d, tm.eta_d[0, 0], rtol=1e-10)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestTsiolkovsky:
    def test_zero_dv(self):
        prop, wet = tsiolkovsky(0.0, 3000.0, 100.0)
        assert prop == pytest.approx(0.0, abs=1e-10)
        assert wet == pytest.approx(100.0)

    def test_known_value(self):
        """ΔV = v_e * ln(2) → mass ratio of 2."""
        isp = 3000.0
        v_e = isp * 9.80665
        dv = v_e * math.log(2.0)
        prop, wet = tsiolkovsky(dv, isp, 100.0)
        assert wet == pytest.approx(200.0, rel=1e-5)
        assert prop == pytest.approx(100.0, rel=1e-5)


class TestEdelbaumDV:
    def test_circular_in_plane(self):
        """LEO to GEO: expect ~4 km/s ΔV."""
        r_leo = 6371e3 + 400e3
        r_geo = 42164e3
        dv = edelbaum_dv(r_leo, r_geo)
        assert 3500 < dv < 5000

    def test_same_orbit_zero_dv(self):
        r = 6371e3 + 500e3
        dv = edelbaum_dv(r, r)
        assert dv == pytest.approx(0.0, abs=1.0)

    def test_higher_orbit_positive_dv(self):
        r1 = 6371e3 + 300e3
        r2 = 6371e3 + 600e3
        assert edelbaum_dv(r1, r2) > 0


class TestAnalyzeMission:
    def test_single_leg(self):
        tm = _make_minimal_throttle_map()
        legs = [MissionLeg("burn", 1000.0, 1e5, 1e-5)]
        result = analyze_mission(legs, tm, 500.0)
        assert result.total_delta_v_ms == 1000.0
        assert result.propellant_mass_kg > 0
        assert result.burn_time_s > 0
        assert 0 < result.payload_fraction <= 1.0

    def test_multi_leg_total_dv(self):
        tm = _make_minimal_throttle_map()
        legs = [
            MissionLeg("leg1", 500.0, 1e5, 1e-5),
            MissionLeg("leg2", 300.0, 1e4, 5e-6),
        ]
        result = analyze_mission(legs, tm, 200.0)
        assert result.total_delta_v_ms == 800.0
        assert len(result.legs) == 2


class TestCircularTransfer:
    def test_returns_result(self):
        tm = _make_minimal_throttle_map()
        r = circular_transfer(
            6371e3 + 400e3,
            6371e3 + 2000e3,
            tm,
            power_W=1e5,
            mdot_kgs=1e-5,
            dry_mass_kg=500.0,
        )
        assert isinstance(r, OrbitTransferResult)
        assert r.delta_v_ms > 0
        assert r.transfer_time_s > 0
        assert r.propellant_mass_kg > 0

    def test_higher_dv_more_propellant(self):
        tm = _make_minimal_throttle_map()
        r_low = circular_transfer(6371e3 + 400e3, 6371e3 + 500e3, tm, 1e5, 1e-5, 500.0)
        r_high = circular_transfer(6371e3 + 400e3, 6371e3 + 3000e3, tm, 1e5, 1e-5, 500.0)
        assert r_high.propellant_mass_kg > r_low.propellant_mass_kg


class TestEarthMarsDV:
    def test_plausible_range(self):
        """Earth-Mars ΔV should be between 3 and 15 km/s."""
        dv = earth_mars_dv()
        assert 3000 < dv < 15000


class TestPoliastroTrajectory:
    def test_returns_dict(self):
        tm = _make_minimal_throttle_map()
        result = poliastro_trajectory(tm, 1e5, 1e-5, 6371e3 + 400e3, 6371e3 + 2000e3, 500.0)
        assert "method" in result
        assert "delta_v_ms" in result
        assert result["delta_v_ms"] > 0

    def test_method_field(self):
        """Should fall back to 'edelbaum' if poliastro not installed."""
        tm = _make_minimal_throttle_map()
        result = poliastro_trajectory(tm, 1e5, 1e-5, 6371e3 + 400e3, 6371e3 + 2000e3, 500.0)
        assert result["method"] in ("edelbaum", "poliastro+edelbaum")


# ---------------------------------------------------------------------------
# Spacecraft interaction
# ---------------------------------------------------------------------------


class TestBackflow:
    def test_perfect_detachment_no_backflow(self):
        r = compute_backflow(10.0, eta_d=1.0, mdot_kgs=1e-5, v_exhaust_ms=50000.0)
        assert r.backflow_fraction == pytest.approx(0.0)
        assert r.backflow_thrust_loss_N == pytest.approx(0.0)

    def test_zero_detachment_full_backflow(self):
        r = compute_backflow(10.0, eta_d=0.0, mdot_kgs=1e-5, v_exhaust_ms=50000.0)
        assert r.backflow_fraction == pytest.approx(1.0)

    def test_partial_backflow(self):
        r = compute_backflow(10.0, eta_d=0.8, mdot_kgs=1e-5, v_exhaust_ms=50000.0)
        assert 0 < r.backflow_fraction < 1
        assert r.ion_return_current_A > 0


class TestSpacecraftCharging:
    def test_negative_floating_potential(self):
        """Floating potential should be negative (electrons hotter than ions)."""
        r = compute_spacecraft_charging(1e16, 100.0)
        assert r.floating_potential_V < 0

    def test_higher_density_more_current(self):
        r_low = compute_spacecraft_charging(1e14, 50.0)
        r_high = compute_spacecraft_charging(1e18, 50.0)
        assert r_high.electron_current_A > r_low.electron_current_A

    def test_net_current_defined(self):
        r = compute_spacecraft_charging(1e16, 100.0, surface_area_m2=5.0)
        assert r.net_current_A == pytest.approx(r.electron_current_A - r.ion_current_A)


class TestMagneticTorque:
    def _make_coils(self):
        from helicon.fields.biot_savart import Coil

        return [Coil(z=0.0, r=0.1, I=40000.0)]

    def test_torque_positive(self):
        coils = self._make_coils()
        r = compute_magnetic_torque(coils, spacecraft_magnetic_moment_Am2=1.0)
        assert r.torque_Nm >= 0

    def test_larger_moment_more_torque(self):
        coils = self._make_coils()
        r_small = compute_magnetic_torque(coils, spacecraft_magnetic_moment_Am2=0.1)
        r_large = compute_magnetic_torque(coils, spacecraft_magnetic_moment_Am2=10.0)
        assert r_large.torque_Nm > r_small.torque_Nm

    def test_farther_distance_less_field(self):
        coils = self._make_coils()
        r_close = compute_magnetic_torque(coils, spacecraft_distance_m=2.0)
        r_far = compute_magnetic_torque(coils, spacecraft_distance_m=10.0)
        assert r_far.field_at_spacecraft_T < r_close.field_at_spacecraft_T


# ---------------------------------------------------------------------------
# Pulsed mission profiles
# ---------------------------------------------------------------------------


class TestPulsedPerformance:
    def _make_profile(self):
        return PulsedProfile(
            pulse_duration_s=1.0,
            off_duration_s=4.0,
            n_pulses=10,
            power_on_W=1e5,
            mdot_on_kgs=1e-5,
        )

    def test_duty_cycle(self):
        p = self._make_profile()
        tm = _make_minimal_throttle_map()
        result = compute_pulsed_performance(p, tm)
        assert result.duty_cycle == pytest.approx(0.2, rel=1e-5)

    def test_mean_thrust_less_than_peak(self):
        p = self._make_profile()
        tm = _make_minimal_throttle_map()
        result = compute_pulsed_performance(p, tm)
        peak_thrust = tm.thrust_at(p.power_on_W, p.mdot_on_kgs)
        assert result.mean_thrust_N < peak_thrust

    def test_effective_isp_equals_peak_isp(self):
        """For an ideal pulse train, effective Isp = instantaneous Isp."""
        p = self._make_profile()
        tm = _make_minimal_throttle_map()
        result = compute_pulsed_performance(p, tm)
        peak_isp = tm.isp_at(p.power_on_W, p.mdot_on_kgs)
        assert result.effective_isp_s == pytest.approx(peak_isp, rel=1e-5)

    def test_total_propellant(self):
        p = self._make_profile()
        tm = _make_minimal_throttle_map()
        result = compute_pulsed_performance(p, tm)
        expected_prop = p.mdot_on_kgs * p.pulse_duration_s * p.n_pulses
        assert result.total_propellant_kg == pytest.approx(expected_prop)

    def test_burst_delta_v_positive(self):
        p = self._make_profile()
        tm = _make_minimal_throttle_map()
        dv = burst_delta_v(p, tm, dry_mass_kg=500.0)
        assert dv > 0

    def test_more_pulses_more_impulse(self):
        p_few = PulsedProfile(1.0, 4.0, 5, 1e5, mdot_on_kgs=1e-5)
        p_many = PulsedProfile(1.0, 4.0, 20, 1e5, mdot_on_kgs=1e-5)
        tm = _make_minimal_throttle_map()
        r_few = compute_pulsed_performance(p_few, tm)
        r_many = compute_pulsed_performance(p_many, tm)
        assert r_many.total_impulse_Ns > r_few.total_impulse_Ns

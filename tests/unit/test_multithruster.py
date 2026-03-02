"""Tests for helicon.multithruster — multi-thruster array simulation."""

from __future__ import annotations

import math

import pytest

from helicon.multithruster import (
    ArrayConfig,
    ArrayResult,
    PlumeModel,
    ThrusterArray,
    compute_overlap_factor,
    compute_plume_interaction,
)

# ---------------------------------------------------------------------------
# PlumeModel
# ---------------------------------------------------------------------------


def test_plume_model_defaults():
    p = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=15.0)
    assert p.exhaust_velocity_ms == pytest.approx(3000.0 * 9.80665, rel=1e-6)


def test_plume_model_mass_flow():
    p = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=15.0)
    expected = 0.1 / (3000.0 * 9.80665)
    assert p.mass_flow_rate_kgs == pytest.approx(expected, rel=1e-6)


def test_plume_model_radius():
    p = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=45.0)
    # tan(45°) = 1, so radius at z=1 should be 1
    assert p.plume_radius_at(1.0) == pytest.approx(1.0, rel=1e-6)


def test_plume_model_explicit_ve():
    p = PlumeModel(
        thrust_N=0.1, isp_s=3000.0, half_angle_deg=15.0, exhaust_velocity_ms=30000.0
    )
    assert p.exhaust_velocity_ms == 30000.0


# ---------------------------------------------------------------------------
# compute_overlap_factor
# ---------------------------------------------------------------------------


def test_no_overlap():
    # Separation >> sum of radii → no overlap
    overlap = compute_overlap_factor(10.0, 10.0, 10.0, reference_z_m=0.1)
    assert overlap == 0.0


def test_full_containment_overlap():
    # One circle entirely inside other → overlap = smaller/larger area
    overlap = compute_overlap_factor(0.0, 30.0, 15.0, reference_z_m=1.0)
    r_small = math.tan(math.radians(15.0))
    r_large = math.tan(math.radians(30.0))
    expected = (r_small**2) / (r_large**2)
    assert overlap == pytest.approx(expected, rel=1e-3)


def test_partial_overlap_range():
    overlap = compute_overlap_factor(0.3, 20.0, 20.0, reference_z_m=1.0)
    assert 0.0 < overlap < 1.0


def test_identical_same_position_full_overlap():
    # Same position, identical plumes → overlap = 1.0 (smaller = larger)
    overlap = compute_overlap_factor(0.0, 15.0, 15.0, reference_z_m=1.0)
    assert overlap == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_plume_interaction
# ---------------------------------------------------------------------------


def test_no_interaction_far_apart():
    pa = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=10.0)
    pb = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=10.0)
    result = compute_plume_interaction(pa, pb, separation_m=10.0, reference_z_m=0.1)
    assert result.overlap_factor == pytest.approx(0.0, abs=1e-9)
    assert result.thrust_penalty_fraction == pytest.approx(0.0, abs=1e-9)
    assert result.combined_thrust_N == pytest.approx(pa.thrust_N + pb.thrust_N, rel=1e-6)


def test_interaction_penalty_positive():
    pa = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=30.0)
    pb = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=30.0)
    result = compute_plume_interaction(pa, pb, separation_m=0.1, reference_z_m=1.0)
    assert result.thrust_penalty_fraction > 0.0
    assert result.combined_thrust_N < pa.thrust_N + pb.thrust_N


def test_interaction_result_isp_positive():
    pa = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=15.0)
    pb = PlumeModel(thrust_N=0.1, isp_s=3000.0, half_angle_deg=15.0)
    result = compute_plume_interaction(pa, pb, separation_m=0.3, reference_z_m=1.0)
    assert result.combined_isp_s > 0.0


# ---------------------------------------------------------------------------
# ArrayConfig
# ---------------------------------------------------------------------------


def test_array_config_default():
    cfg = ArrayConfig()
    assert cfg.n_thrusters == 2
    assert len(cfg.thrust_N) == 2


def test_array_config_bad_n_thrusters():
    with pytest.raises(ValueError, match="n_thrusters"):
        ArrayConfig(n_thrusters=1, thrust_N=[0.1], isp_s=[3000.0], plume_half_angle_deg=[15.0])


def test_array_config_mismatched_lengths():
    with pytest.raises(ValueError, match="thrust_N"):
        ArrayConfig(
            n_thrusters=2,
            thrust_N=[0.1],
            isp_s=[3000.0, 3000.0],
            plume_half_angle_deg=[15.0, 15.0],
        )


def test_array_config_max_thrusters():
    cfg = ArrayConfig(
        n_thrusters=4,
        separation_m=0.3,
        thrust_N=[0.1] * 4,
        isp_s=[3000.0] * 4,
        plume_half_angle_deg=[15.0] * 4,
    )
    positions = cfg.get_positions()
    assert len(positions) == 4


def test_array_config_linear_positions_symmetric():
    cfg = ArrayConfig(
        n_thrusters=2,
        separation_m=1.0,
        thrust_N=[0.1, 0.1],
        isp_s=[3000.0, 3000.0],
        plume_half_angle_deg=[15.0, 15.0],
    )
    positions = cfg.get_positions()
    # Should be symmetric about 0
    assert positions[0][0] == pytest.approx(-0.5, rel=1e-6)
    assert positions[1][0] == pytest.approx(0.5, rel=1e-6)


def test_array_config_custom_positions():
    cfg = ArrayConfig(
        n_thrusters=2,
        thrust_N=[0.1, 0.1],
        isp_s=[3000.0, 3000.0],
        plume_half_angle_deg=[15.0, 15.0],
        positions_m=[(0.0, 0.0), (0.5, 0.3)],
    )
    positions = cfg.get_positions()
    assert positions[1] == (0.5, 0.3)


# ---------------------------------------------------------------------------
# ThrusterArray
# ---------------------------------------------------------------------------


def test_thruster_array_no_interaction():
    """Far apart thrusters — negligible interaction."""
    cfg = ArrayConfig(
        n_thrusters=2,
        separation_m=5.0,
        thrust_N=[0.1, 0.1],
        isp_s=[3000.0, 3000.0],
        plume_half_angle_deg=[10.0, 10.0],
        reference_z_m=0.1,
    )
    result = ThrusterArray(cfg).compute()
    assert result.total_thrust_N == pytest.approx(0.2, rel=0.01)
    assert result.interaction_penalty == pytest.approx(0.0, abs=0.02)


def test_thruster_array_close_has_penalty():
    """Close thrusters — should have non-trivial penalty."""
    cfg = ArrayConfig(
        n_thrusters=2,
        separation_m=0.1,
        thrust_N=[0.1, 0.1],
        isp_s=[3000.0, 3000.0],
        plume_half_angle_deg=[30.0, 30.0],
        reference_z_m=1.0,
    )
    result = ThrusterArray(cfg).compute()
    assert result.interaction_penalty > 0.05


def test_thruster_array_result_types():
    cfg = ArrayConfig()
    result = ThrusterArray(cfg).compute()
    assert isinstance(result, ArrayResult)
    assert isinstance(result.pair_interactions, list)
    assert len(result.pair_interactions) == 1  # C(2,2) = 1 pair


def test_thruster_array_3_thrusters_pairs():
    cfg = ArrayConfig(
        n_thrusters=3,
        separation_m=0.5,
        thrust_N=[0.1, 0.1, 0.1],
        isp_s=[3000.0, 3000.0, 3000.0],
        plume_half_angle_deg=[15.0, 15.0, 15.0],
    )
    result = ThrusterArray(cfg).compute()
    # C(3,2) = 3 pairs
    assert len(result.pair_interactions) == 3


def test_thruster_array_4_thrusters():
    cfg = ArrayConfig(
        n_thrusters=4,
        separation_m=0.5,
        thrust_N=[0.1] * 4,
        isp_s=[3000.0] * 4,
        plume_half_angle_deg=[15.0] * 4,
    )
    result = ThrusterArray(cfg).compute()
    # C(4,2) = 6 pairs
    assert len(result.pair_interactions) == 6
    assert result.total_thrust_N > 0


def test_thruster_array_isp_consistent():
    cfg = ArrayConfig()
    result = ThrusterArray(cfg).compute()
    # Isp = F / (mdot * g0)
    expected_isp = result.total_thrust_N / (result.total_mass_flow_kgs * 9.80665)
    assert result.effective_isp_s == pytest.approx(expected_isp, rel=1e-4)


def test_thruster_array_per_thruster_sum():
    cfg = ArrayConfig()
    result = ThrusterArray(cfg).compute()
    assert sum(result.per_thruster_thrust_N) == pytest.approx(result.total_thrust_N, rel=1e-10)

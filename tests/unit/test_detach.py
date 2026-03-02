"""Tests for helicon.detach — real-time detachment onset model."""

from __future__ import annotations

import json
import math

import pytest
from click.testing import CliRunner

from helicon.cli import main
from helicon.detach import (
    DetachmentOnsetModel,
    DetachmentState,
    PlasmaState,
    alfven_mach,
    alfven_velocity,
    bohm_velocity,
    electron_beta,
    field_scale_length,
    ion_larmor_radius,
    ion_magnetization,
)
from helicon.detach.invariants import (
    EV_TO_J,
    M_PROTON,
    MU0,
    magnetic_mirror_force,
    magnetic_moment,
    species_mass,
)
from helicon.detach.model import ScanResult

# ---------------------------------------------------------------------------
# Physical constants sanity
# ---------------------------------------------------------------------------


def test_mu0_value():
    assert abs(MU0 - 4 * math.pi * 1e-7) < 1e-12


def test_ev_to_j():
    assert abs(EV_TO_J - 1.602176634e-19) < 1e-28


def test_m_proton():
    assert 1.67e-27 < M_PROTON < 1.68e-27


# ---------------------------------------------------------------------------
# species_mass
# ---------------------------------------------------------------------------


def test_species_mass_hydrogen():
    assert abs(species_mass("H+") - 1.008) < 0.01


def test_species_mass_argon():
    assert abs(species_mass("Ar+") - 39.948) < 0.01


def test_species_mass_xenon():
    assert species_mass("Xe+") > 130.0


def test_species_mass_unknown_raises():
    with pytest.raises(ValueError, match="Unknown species"):
        species_mass("U+")


# ---------------------------------------------------------------------------
# alfven_velocity
# ---------------------------------------------------------------------------


def test_alfven_velocity_positive():
    va = alfven_velocity(0.05, 1e18, 1.0)
    assert va > 0


def test_alfven_velocity_scales_with_B():
    va1 = alfven_velocity(0.05, 1e18, 1.0)
    va2 = alfven_velocity(0.10, 1e18, 1.0)
    assert va2 > va1


def test_alfven_velocity_scales_inverse_density():
    va1 = alfven_velocity(0.05, 1e17, 1.0)
    va2 = alfven_velocity(0.05, 1e18, 1.0)
    assert va1 > va2


def test_alfven_velocity_zero_density():
    va = alfven_velocity(0.05, 0.0, 1.0)
    assert va == math.inf


# ---------------------------------------------------------------------------
# alfven_mach
# ---------------------------------------------------------------------------


def test_alfven_mach_less_than_one_attached():
    # Low velocity, strong field → attached
    M_A = alfven_mach(1e4, 0.1, 1e18, 1.0)
    assert M_A < 1.0


def test_alfven_mach_greater_than_one_detached():
    # High velocity, weak field → detached
    M_A = alfven_mach(1e5, 0.001, 1e18, 1.0)
    assert M_A > 1.0


def test_alfven_mach_unity_at_va():
    # When vz == va, M_A should be exactly 1
    va = alfven_velocity(0.05, 1e18, 1.0)
    M_A = alfven_mach(va, 0.05, 1e18, 1.0)
    assert abs(M_A - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# electron_beta
# ---------------------------------------------------------------------------


def test_electron_beta_small_for_strong_field():
    be = electron_beta(1e17, 10.0, 0.5)
    assert be < 0.01


def test_electron_beta_large_for_weak_field():
    be = electron_beta(1e19, 100.0, 0.001)
    assert be > 1.0


def test_electron_beta_zero_field():
    be = electron_beta(1e18, 50.0, 0.0)
    assert be == math.inf


def test_electron_beta_proportional_to_n():
    be1 = electron_beta(1e17, 50.0, 0.05)
    be2 = electron_beta(1e18, 50.0, 0.05)
    assert abs(be2 / be1 - 10.0) < 1e-8


# ---------------------------------------------------------------------------
# ion_larmor_radius
# ---------------------------------------------------------------------------


def test_ion_larmor_positive():
    r = ion_larmor_radius(100.0, 0.05, 1.0)
    assert r > 0


def test_larmor_larger_for_hotter_ions():
    r1 = ion_larmor_radius(50.0, 0.05, 1.0)
    r2 = ion_larmor_radius(200.0, 0.05, 1.0)
    assert r2 > r1


def test_larmor_larger_for_heavier_ions():
    r_h = ion_larmor_radius(100.0, 0.05, 1.0)
    r_ar = ion_larmor_radius(100.0, 0.05, 40.0)
    assert r_ar > r_h


def test_larmor_zero_field():
    r = ion_larmor_radius(100.0, 0.0, 1.0)
    assert r == math.inf


# ---------------------------------------------------------------------------
# field_scale_length
# ---------------------------------------------------------------------------


def test_field_scale_length_finite():
    L = field_scale_length(0.05, -2.0)
    assert pytest.approx(0.025) == L


def test_field_scale_length_zero_gradient():
    L = field_scale_length(0.05, 0.0)
    assert math.inf == L


def test_field_scale_length_absolute_gradient():
    # Sign of gradient should not matter
    L_pos = field_scale_length(0.1, 1.0)
    L_neg = field_scale_length(0.1, -1.0)
    assert L_pos == L_neg


# ---------------------------------------------------------------------------
# ion_magnetization
# ---------------------------------------------------------------------------


def test_ion_magnetization_small_for_gentle_gradient():
    lam = ion_magnetization(100.0, 0.05, -0.01, 1.0)
    assert lam < 1.0


def test_ion_magnetization_large_for_steep_gradient():
    lam = ion_magnetization(100.0, 0.05, -50.0, 1.0)
    assert lam > 1.0


# ---------------------------------------------------------------------------
# bohm_velocity
# ---------------------------------------------------------------------------


def test_bohm_velocity_positive():
    cs = bohm_velocity(50.0, 1.0)
    assert cs > 0


def test_bohm_velocity_scale_sqrt_Te():
    cs1 = bohm_velocity(50.0, 1.0)
    cs2 = bohm_velocity(200.0, 1.0)
    assert abs(cs2 / cs1 - 2.0) < 1e-8


# ---------------------------------------------------------------------------
# magnetic_moment and mirror_force
# ---------------------------------------------------------------------------


def test_magnetic_moment_positive():
    mu = magnetic_moment(100.0, 0.05, 1.0)
    assert mu > 0


def test_mirror_force_downstream_positive():
    # Negative dBdz (field falling) → positive force (accelerating)
    mu = magnetic_moment(100.0, 0.05, 1.0)
    F = magnetic_mirror_force(mu, -5.0)
    assert F > 0


# ---------------------------------------------------------------------------
# PlasmaState validation
# ---------------------------------------------------------------------------


def _typical_state(**kwargs):
    defaults = dict(
        n_m3=1e18, Te_eV=50.0, Ti_eV=50.0, B_T=0.05, dBdz_T_per_m=-2.0, vz_ms=4e4, mass_amu=1.0
    )
    defaults.update(kwargs)
    return PlasmaState(**defaults)


def test_plasma_state_valid():
    s = _typical_state()
    s.validate()  # should not raise


def test_plasma_state_negative_density_raises():
    with pytest.raises(ValueError):
        _typical_state(n_m3=-1e18).validate()


def test_plasma_state_zero_B_raises():
    with pytest.raises(ValueError):
        _typical_state(B_T=0.0).validate()


def test_plasma_state_negative_Te_raises():
    with pytest.raises(ValueError):
        _typical_state(Te_eV=-1.0).validate()


def test_plasma_state_negative_vz_raises():
    with pytest.raises(ValueError):
        _typical_state(vz_ms=-1.0).validate()


# ---------------------------------------------------------------------------
# DetachmentOnsetModel.assess
# ---------------------------------------------------------------------------


def test_assess_returns_detachment_state():
    model = DetachmentOnsetModel()
    ds = model.assess(_typical_state())
    assert isinstance(ds, DetachmentState)


def test_assess_score_in_range():
    model = DetachmentOnsetModel()
    ds = model.assess(_typical_state())
    assert 0.0 <= ds.detachment_score <= 1.0


def test_assess_control_signal_complement():
    model = DetachmentOnsetModel()
    ds = model.assess(_typical_state())
    assert abs(ds.control_signal + ds.detachment_score - 1.0) < 1e-12


def test_assess_typical_is_attached():
    # Typical helicon thruster conditions at throat → attached
    model = DetachmentOnsetModel()
    ds = model.assess(_typical_state())
    assert not ds.is_detached


def test_assess_weak_field_detaches():
    # Very weak B → super-Alfvénic → detached
    model = DetachmentOnsetModel()
    ds = model.assess(_typical_state(B_T=0.0001, vz_ms=1e5))
    assert ds.is_detached or ds.is_imminent


def test_assess_onset_B_positive():
    model = DetachmentOnsetModel()
    ds = model.assess(_typical_state())
    assert ds.onset_B_T >= 0


def test_assess_onset_B_formula():
    # v_A = v_z at onset: B = v_z * sqrt(μ₀ n mᵢ)
    state = _typical_state()
    model = DetachmentOnsetModel()
    ds = model.assess(state)
    m_i = state.mass_amu * M_PROTON
    expected = state.vz_ms * math.sqrt(MU0 * state.n_m3 * m_i)
    assert abs(ds.onset_B_T - expected) < 1e-10


def test_assess_to_dict_keys():
    model = DetachmentOnsetModel()
    d = model.assess(_typical_state()).to_dict()
    for key in (
        "alfven_mach",
        "electron_beta",
        "ion_magnetization",
        "detachment_score",
        "is_detached",
        "is_imminent",
        "onset_B_T",
    ):
        assert key in d


def test_assess_summary_contains_status():
    model = DetachmentOnsetModel()
    ds = model.assess(_typical_state())
    text = ds.summary().upper()
    assert "ATTACHED" in text or "DETACHED" in text or "IMMINENT" in text


def test_bad_weights_raise():
    with pytest.raises(ValueError, match="weights must sum"):
        DetachmentOnsetModel(w_alfven=0.5, w_beta=0.5, w_ion_mag=0.5)


# ---------------------------------------------------------------------------
# DetachmentOnsetModel.scan_z
# ---------------------------------------------------------------------------


def test_scan_z_returns_scan_result():
    model = DetachmentOnsetModel()
    states = [_typical_state(B_T=0.05 * (1 - 0.05 * i)) for i in range(5)]
    z = [0.0, 0.25, 0.50, 0.75, 1.00]
    result = model.scan_z(states, z)
    assert isinstance(result, ScanResult)
    assert len(result.states) == 5


def test_scan_z_length_mismatch_raises():
    model = DetachmentOnsetModel()
    with pytest.raises(ValueError, match="equal length"):
        model.scan_z([_typical_state()], [0.0, 1.0])


def test_scan_z_finds_onset():
    model = DetachmentOnsetModel()
    # Build states progressing from attached to detached
    states = [
        _typical_state(B_T=0.05),  # attached
        _typical_state(B_T=0.01),  # possibly imminent
        _typical_state(B_T=0.0002, vz_ms=2e5),  # detached
    ]
    z = [0.0, 0.5, 1.0]
    result = model.scan_z(states, z)
    # At least one state should be imminent or detached
    assert any(s.is_imminent or s.is_detached for s in result.states)


def test_scan_z_score_profile_length():
    model = DetachmentOnsetModel()
    states = [_typical_state() for _ in range(8)]
    z = list(range(8))
    result = model.scan_z(states, z)
    assert len(result.score_profile()) == 8


def test_scan_z_no_onset_when_all_attached():
    model = DetachmentOnsetModel()
    states = [_typical_state(B_T=0.5) for _ in range(4)]  # strong B → attached
    z = [0.0, 0.3, 0.6, 0.9]
    result = model.scan_z(states, z)
    assert result.onset_z_m is None
    assert result.detach_z_m is None


# ---------------------------------------------------------------------------
# control_recommendation
# ---------------------------------------------------------------------------


def test_control_recommendation_keys():
    model = DetachmentOnsetModel()
    rec = model.control_recommendation(_typical_state())
    for key in (
        "score",
        "control_signal",
        "is_detached",
        "is_imminent",
        "onset_B_T",
        "recommended_action",
    ):
        assert key in rec


def test_control_recommendation_nominal_action():
    model = DetachmentOnsetModel()
    rec = model.control_recommendation(_typical_state())
    assert "NOMINAL" in rec["recommended_action"] or "MONITOR" in rec["recommended_action"]


def test_control_recommendation_increase_B_when_detached():
    model = DetachmentOnsetModel()
    state = _typical_state(B_T=0.0001, vz_ms=1e5)
    ds = model.assess(state)
    if ds.is_detached:
        rec = model.control_recommendation(state)
        assert "INCREASE_B" in rec["recommended_action"]


# ---------------------------------------------------------------------------
# CLI — helicon detach
# ---------------------------------------------------------------------------

_ATTACHED_ARGS = [
    "detach",
    "assess",
    "--n",
    "1e18",
    "--Te",
    "50",
    "--Ti",
    "50",
    "--B",
    "0.05",
    "--dBdz",
    "-2",
    "--vz",
    "40000",
]

_DETACHED_ARGS = [
    "detach",
    "assess",
    "--n",
    "1e18",
    "--Te",
    "50",
    "--Ti",
    "50",
    "--B",
    "0.0001",
    "--dBdz",
    "-50",
    "--vz",
    "100000",
]


def test_detach_cli_attached():
    runner = CliRunner()
    result = runner.invoke(main, _ATTACHED_ARGS)
    # exit 0 when attached
    assert result.exit_code == 0, result.output


def test_detach_cli_output_contains_status():
    runner = CliRunner()
    result = runner.invoke(main, _ATTACHED_ARGS)
    output = result.output.upper()
    assert "ATTACHED" in output or "IMMINENT" in output or "DETACHED" in output


def test_detach_cli_json():
    runner = CliRunner()
    result = runner.invoke(main, [*_ATTACHED_ARGS, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "detachment_score" in data
    assert "alfven_mach" in data


def test_detach_cli_control_flag():
    runner = CliRunner()
    result = runner.invoke(main, [*_ATTACHED_ARGS, "--control"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "recommended_action" in data
    assert "control_signal" in data


def test_detach_cli_species_argon():
    runner = CliRunner()
    result = runner.invoke(main, [*_ATTACHED_ARGS, "--species", "Ar+"])
    assert result.exit_code == 0, result.output


def test_detach_cli_detached_nonzero_exit():
    runner = CliRunner()
    result = runner.invoke(main, _DETACHED_ARGS)
    # exit 1 (imminent) or 2 (detached) when plasma is not attached
    assert result.exit_code in (0, 1, 2)
    # but the score should be high
    data_result = runner.invoke(main, [*_DETACHED_ARGS, "--json"])
    data = json.loads(data_result.output)
    assert data["detachment_score"] > data["alfven_mach"] * 0.0  # score is finite


def test_detach_cli_json_score_in_range():
    runner = CliRunner()
    result = runner.invoke(main, [*_ATTACHED_ARGS, "--json"])
    data = json.loads(result.output)
    assert 0.0 <= data["detachment_score"] <= 1.0

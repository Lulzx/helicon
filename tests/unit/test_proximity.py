"""Tests for magnozzlex.validate.proximity."""

from __future__ import annotations

import math

import pytest

from magnozzlex.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    ResolutionConfig,
    SimConfig,
)
from magnozzlex.validate.proximity import (
    ProximityResult,
    VALIDATED_REGIONS,
    _MU_0,
    config_proximity,
)


# ---------------------------------------------------------------------------
# Helpers to build SimConfig objects quickly
# ---------------------------------------------------------------------------

def _make_config(
    n0: float,
    T_i_eV: float,
    I_coil: float,
    r_coil: float = 0.10,
) -> SimConfig:
    """Construct a minimal SimConfig for testing."""
    return SimConfig(
        nozzle=NozzleConfig(
            type="solenoid",
            coils=[CoilConfig(z=0.0, r=r_coil, I=I_coil)],
            domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            resolution=ResolutionConfig(nz=128, nr=64),
        ),
        plasma=PlasmaSourceConfig(
            n0=n0,
            T_i_eV=T_i_eV,
            T_e_eV=T_i_eV,
            v_injection_ms=50_000.0,
        ),
        timesteps=10_000,
    )


def _b_from_coil(I: float, r: float) -> float:
    """On-axis field at the coil centre: B = μ₀ I / (2 r)."""
    return _MU_0 * abs(I) / (2.0 * r)


def _coil_current_for_b(B: float, r: float = 0.10) -> float:
    """Invert the above relation."""
    return B * 2.0 * r / _MU_0


# ---------------------------------------------------------------------------
# 1. Structure test
# ---------------------------------------------------------------------------

class TestProximityResultStructure:
    """ProximityResult has all required fields with correct types."""

    def test_proximity_result_structure(self):
        config = _make_config(n0=1e18, T_i_eV=100.0, I_coil=_coil_current_for_b(0.10))
        result = config_proximity(config)

        assert isinstance(result, ProximityResult), "must return ProximityResult"
        assert isinstance(result.nearest_case, str), "nearest_case must be str"
        assert isinstance(result.distance, float), "distance must be float"
        assert isinstance(result.in_validated_region, bool), "in_validated_region must be bool"
        assert isinstance(result.parameter_distances, dict), "parameter_distances must be dict"
        # warning is str or None
        assert result.warning is None or isinstance(result.warning, str)

    def test_result_is_dataclass_instance(self):
        import dataclasses
        config = _make_config(n0=1e18, T_i_eV=100.0, I_coil=_coil_current_for_b(0.10))
        result = config_proximity(config)
        assert dataclasses.is_dataclass(result)


# ---------------------------------------------------------------------------
# 2. Free-expansion close config → distance < 1.0
# ---------------------------------------------------------------------------

class TestProximityFreeExpansionClose:
    """A config matching the free_expansion region centre must be inside it."""

    def test_proximity_free_expansion_close(self):
        region = VALIDATED_REGIONS["free_expansion"]
        n0_centre, _ = region["n0"]
        T_centre, _ = region["T_i_eV"]
        B_centre, _ = region["B_T"]

        I_coil = _coil_current_for_b(B_centre)
        config = _make_config(n0=n0_centre, T_i_eV=T_centre, I_coil=I_coil)
        result = config_proximity(config)

        assert result.distance < 1.0, (
            f"Config at free_expansion centre should have distance < 1.0, "
            f"got {result.distance:.4f}"
        )

    def test_free_expansion_is_nearest(self):
        region = VALIDATED_REGIONS["free_expansion"]
        n0_centre, _ = region["n0"]
        T_centre, _ = region["T_i_eV"]
        B_centre, _ = region["B_T"]

        I_coil = _coil_current_for_b(B_centre)
        config = _make_config(n0=n0_centre, T_i_eV=T_centre, I_coil=I_coil)
        result = config_proximity(config)

        assert result.nearest_case == "free_expansion"


# ---------------------------------------------------------------------------
# 3. Far config → warning is not None
# ---------------------------------------------------------------------------

class TestProximityFarConfigWarning:
    """A config far from every validated case must produce a warning."""

    def test_proximity_far_config_warning(self):
        # Choose parameters that are extreme and unlikely to be near any region:
        # n0 = 1e10 (very low), T = 1e6 eV (very high), B ~ 1e-4 T (very low)
        I_coil = _coil_current_for_b(1e-4)
        config = _make_config(n0=1e10, T_i_eV=1e6, I_coil=I_coil)
        result = config_proximity(config)

        assert result.warning is not None, (
            "Config far from all validated cases must produce a warning string"
        )

    def test_far_config_warning_contains_nearest_case(self):
        I_coil = _coil_current_for_b(1e-4)
        config = _make_config(n0=1e10, T_i_eV=1e6, I_coil=I_coil)
        result = config_proximity(config)

        assert result.warning is not None
        assert result.nearest_case in result.warning

    def test_far_config_distance_exceeds_two(self):
        I_coil = _coil_current_for_b(1e-4)
        config = _make_config(n0=1e10, T_i_eV=1e6, I_coil=I_coil)
        result = config_proximity(config)

        assert result.distance > 2.0, (
            f"Far config should have distance > 2.0, got {result.distance:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. VASIMR-like config → nearest = "vasimr"
# ---------------------------------------------------------------------------

class TestProximityNearestCaseCorrect:
    """A VASIMR-like configuration should identify 'vasimr' as the nearest case."""

    def test_proximity_nearest_case_correct(self):
        region = VALIDATED_REGIONS["vasimr"]
        n0_centre, _ = region["n0"]       # 5e19
        T_centre, _ = region["T_i_eV"]    # 1000
        B_centre, _ = region["B_T"]        # 2.0

        I_coil = _coil_current_for_b(B_centre)
        config = _make_config(n0=n0_centre, T_i_eV=T_centre, I_coil=I_coil)
        result = config_proximity(config)

        assert result.nearest_case == "vasimr", (
            f"VASIMR-like config should map to 'vasimr', got '{result.nearest_case}'"
        )

    def test_vasimr_distance_is_small(self):
        region = VALIDATED_REGIONS["vasimr"]
        I_coil = _coil_current_for_b(region["B_T"][0])
        config = _make_config(
            n0=region["n0"][0],
            T_i_eV=region["T_i_eV"][0],
            I_coil=I_coil,
        )
        result = config_proximity(config)
        assert result.distance < 0.5, (
            f"Config exactly at vasimr centre should be very close, got {result.distance:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Config at region centre → in_validated_region = True
# ---------------------------------------------------------------------------

class TestProximityInValidatedRegion:
    """A config placed at the exact centre of any region must be inside it."""

    @pytest.mark.parametrize("case_name", list(VALIDATED_REGIONS.keys()))
    def test_proximity_in_validated_region(self, case_name: str):
        region = VALIDATED_REGIONS[case_name]
        n0_centre, _ = region["n0"]
        T_centre, _ = region["T_i_eV"]
        B_centre, _ = region["B_T"]

        I_coil = _coil_current_for_b(B_centre)
        config = _make_config(n0=n0_centre, T_i_eV=T_centre, I_coil=I_coil)
        result = config_proximity(config)

        assert result.in_validated_region, (
            f"Config at centre of '{case_name}' must be in_validated_region=True, "
            f"got distance={result.distance:.4f}, per_param={result.parameter_distances}"
        )

    def test_centre_distance_near_zero(self):
        region = VALIDATED_REGIONS["free_expansion"]
        I_coil = _coil_current_for_b(region["B_T"][0])
        config = _make_config(
            n0=region["n0"][0],
            T_i_eV=region["T_i_eV"][0],
            I_coil=I_coil,
        )
        result = config_proximity(config)
        assert result.distance < 1e-6, (
            f"Config exactly at region centre should have distance ~0, "
            f"got {result.distance:.2e}"
        )


# ---------------------------------------------------------------------------
# 6. parameter_distances dict has expected keys
# ---------------------------------------------------------------------------

class TestProximityParameterDistancesDict:
    """result.parameter_distances must contain exactly the expected keys."""

    EXPECTED_KEYS = {"n0", "T_i_eV", "B_T"}

    def test_proximity_parameter_distances_dict(self):
        config = _make_config(n0=1e18, T_i_eV=100.0, I_coil=_coil_current_for_b(0.10))
        result = config_proximity(config)

        assert set(result.parameter_distances.keys()) == self.EXPECTED_KEYS, (
            f"parameter_distances keys should be {self.EXPECTED_KEYS}, "
            f"got {set(result.parameter_distances.keys())}"
        )

    def test_parameter_distances_are_non_negative_floats(self):
        config = _make_config(n0=1e18, T_i_eV=100.0, I_coil=_coil_current_for_b(0.10))
        result = config_proximity(config)

        for key, val in result.parameter_distances.items():
            assert isinstance(val, float), f"{key} distance must be float"
            assert val >= 0.0, f"{key} distance must be non-negative, got {val}"

    def test_parameter_distances_finite(self):
        config = _make_config(n0=1e18, T_i_eV=100.0, I_coil=_coil_current_for_b(0.10))
        result = config_proximity(config)

        for key, val in result.parameter_distances.items():
            assert math.isfinite(val), f"{key} distance must be finite, got {val}"


# ---------------------------------------------------------------------------
# 7. Import from validate package public API
# ---------------------------------------------------------------------------

class TestPublicAPIImport:
    """config_proximity and ProximityResult must be importable from the package."""

    def test_importable_from_validate_package(self):
        from magnozzlex.validate import config_proximity as cp, ProximityResult as PR
        assert cp is config_proximity
        assert PR is ProximityResult

    def test_in_validate_all(self):
        import magnozzlex.validate as validate_pkg
        assert "config_proximity" in validate_pkg.__all__
        assert "ProximityResult" in validate_pkg.__all__

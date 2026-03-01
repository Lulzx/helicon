"""Tests for magnozzlex.validate.cases.vasimr_plume."""

from pathlib import Path

import pytest

from magnozzlex.validate.cases.vasimr_plume import (
    TOLERANCES,
    VASIMR_REFERENCE,
    VASIMRPlumeCase,
    ValidationResult,
)


class TestVASIMRPlumeCase:
    def test_name_and_description(self):
        assert VASIMRPlumeCase.name == "vasimr_plume"
        assert "VX-200" in VASIMRPlumeCase.description or "VASIMR" in VASIMRPlumeCase.description.upper()

    def test_get_config_returns_sim_config(self):
        from magnozzlex.config.parser import SimConfig
        config = VASIMRPlumeCase.get_config()
        assert isinstance(config, SimConfig)

    def test_config_has_two_coils(self):
        config = VASIMRPlumeCase.get_config()
        assert len(config.nozzle.coils) == 2

    def test_config_argon_species(self):
        config = VASIMRPlumeCase.get_config()
        assert "Ar+" in config.plasma.species

    def test_config_throat_field_approx(self):
        import numpy as np
        config = VASIMRPlumeCase.get_config()
        mu0 = 4e-7 * np.pi
        coil = config.nozzle.coils[0]
        B_est = mu0 * coil.I / (2 * coil.r)
        # Should be near 0.5 T (within factor 2)
        assert 0.1 < B_est < 2.0

    def test_config_resolution(self):
        config = VASIMRPlumeCase.get_config()
        assert config.nozzle.resolution.nz >= 256
        assert config.nozzle.resolution.nr >= 128

    def test_evaluate_no_data_returns_failed(self, tmp_path):
        result = VASIMRPlumeCase.evaluate(tmp_path)
        assert isinstance(result, ValidationResult)
        assert not result.passed
        assert result.case_name == "vasimr_plume"

    def test_evaluate_no_data_empty_metrics(self, tmp_path):
        result = VASIMRPlumeCase.evaluate(tmp_path)
        assert result.metrics == {}

    def test_evaluate_tolerances_present(self, tmp_path):
        result = VASIMRPlumeCase.evaluate(tmp_path)
        assert "thrust_efficiency" in result.tolerances
        assert "plume_half_angle_deg" in result.tolerances

    def test_reference_values_match_olsen_2015(self):
        assert VASIMR_REFERENCE["thrust_efficiency"] == pytest.approx(0.69)
        assert VASIMR_REFERENCE["plume_half_angle_deg"] == pytest.approx(31.0)
        assert VASIMR_REFERENCE["thrust_N"] == pytest.approx(5.7)

    def test_tolerances_sensible(self):
        assert TOLERANCES["thrust_efficiency"] == pytest.approx(0.15)
        assert TOLERANCES["plume_half_angle_deg"] == pytest.approx(0.20)

    def test_evaluate_result_case_name(self, tmp_path):
        result = VASIMRPlumeCase.evaluate(tmp_path)
        assert result.case_name == "vasimr_plume"


class TestVASIMRIntegration:
    """Test that VASIMR case appears in validation suite."""

    def test_in_all_cases(self):
        from magnozzlex.validate.runner import ALL_CASES
        names = [c.name for c in ALL_CASES]
        assert "vasimr_plume" in names

    def test_importable_from_cases_init(self):
        from magnozzlex.validate.cases import VASIMRPlumeCase as Imported
        assert Imported is VASIMRPlumeCase

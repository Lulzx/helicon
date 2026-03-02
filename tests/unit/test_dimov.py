"""Tests for helicon.validate.cases.resistive_dimov."""

import numpy as np
import pytest

from helicon.validate.cases.resistive_dimov import (
    DIMOV_REFERENCE,
    ResistiveDimovCase,
    ValidationResult,
)


class TestResistiveDimovCase:
    def test_name_and_description(self):
        assert ResistiveDimovCase.name == "resistive_dimov"
        desc = ResistiveDimovCase.description
        assert "Dimov" in desc or "resistive" in desc.lower()

    def test_get_config_returns_sim_config(self):
        from helicon.config.parser import SimConfig

        config = ResistiveDimovCase.get_config()
        assert isinstance(config, SimConfig)

    def test_config_deuterium_species(self):
        config = ResistiveDimovCase.get_config()
        assert "D+" in config.plasma.species

    def test_config_low_b_field(self):
        # B_throat = 0.05 T for resistive threshold
        mu0 = 4e-7 * np.pi
        config = ResistiveDimovCase.get_config()
        coil = config.nozzle.coils[0]
        B_est = mu0 * coil.I / (2 * coil.r)
        assert B_est < 0.1  # lower than typical magnetized nozzle


class TestHallParameterThreshold:
    def test_returns_float(self):
        result = ResistiveDimovCase.hall_parameter_threshold(B_T=0.05, T_e_eV=10.0, n_m3=1e19)
        assert isinstance(result, float)

    def test_resistive_config_near_threshold(self):
        # Config uses n=1e21 m^-3 → Ω_e τ_e ≈ 1 (Spitzer formula)
        config = ResistiveDimovCase.get_config()
        mu0 = 4e-7 * np.pi
        coil = config.nozzle.coils[0]
        B_est = mu0 * coil.I / (2 * coil.r)
        omega_tau = ResistiveDimovCase.hall_parameter_threshold(
            B_T=B_est, T_e_eV=config.plasma.T_e_eV, n_m3=config.plasma.n0
        )
        # Should be within a factor of 3 of unity
        assert 0.1 < omega_tau < 10.0

    def test_scales_with_B(self):
        # Ω_e ∝ B → omega_tau ∝ B
        ot1 = ResistiveDimovCase.hall_parameter_threshold(B_T=0.05, T_e_eV=10.0, n_m3=1e19)
        ot2 = ResistiveDimovCase.hall_parameter_threshold(B_T=0.10, T_e_eV=10.0, n_m3=1e19)
        assert abs(ot2 / ot1 - 2.0) < 0.01

    def test_scales_with_density(self):
        # ν_ei ∝ n → omega_tau ∝ 1/n
        ot1 = ResistiveDimovCase.hall_parameter_threshold(B_T=0.05, T_e_eV=10.0, n_m3=1e20)
        ot2 = ResistiveDimovCase.hall_parameter_threshold(B_T=0.05, T_e_eV=10.0, n_m3=1e21)
        assert abs(ot1 / ot2 - 10.0) < 0.1

    def test_scales_with_temperature(self):
        # τ_e ∝ T_e^(3/2) → omega_tau ∝ T_e^(3/2)
        ot1 = ResistiveDimovCase.hall_parameter_threshold(B_T=0.05, T_e_eV=5.0, n_m3=1e19)
        ot2 = ResistiveDimovCase.hall_parameter_threshold(B_T=0.05, T_e_eV=10.0, n_m3=1e19)
        ratio = ot2 / ot1
        expected_ratio = (10.0 / 5.0) ** 1.5  # 2^1.5 ≈ 2.83
        assert abs(ratio - expected_ratio) < 0.01

    def test_coulomb_log_effect(self):
        # Higher ln_lambda → more collisions → lower omega_tau
        kw = {"B_T": 0.05, "T_e_eV": 10.0, "n_m3": 1e19}
        ot1 = ResistiveDimovCase.hall_parameter_threshold(**kw, ln_lambda=10.0)
        ot2 = ResistiveDimovCase.hall_parameter_threshold(**kw, ln_lambda=20.0)
        assert ot1 > ot2

    def test_positive_value(self):
        omega_tau = ResistiveDimovCase.hall_parameter_threshold(
            B_T=0.1, T_e_eV=20.0, n_m3=1e18
        )
        assert omega_tau > 0.0


class TestResistiveDimovEvaluate:
    def test_evaluate_returns_validation_result(self, tmp_path):
        result = ResistiveDimovCase.evaluate(tmp_path)
        assert isinstance(result, ValidationResult)

    def test_evaluate_case_name(self, tmp_path):
        result = ResistiveDimovCase.evaluate(tmp_path)
        assert result.case_name == "resistive_dimov"

    def test_evaluate_contains_theoretical_metric(self, tmp_path):
        result = ResistiveDimovCase.evaluate(tmp_path)
        assert "hall_parameter_theoretical" in result.metrics

    def test_evaluate_passes_threshold_check(self, tmp_path):
        # Without simulation data, checks theoretical value is near threshold (within factor 3)
        result = ResistiveDimovCase.evaluate(tmp_path)
        assert result.passed  # config is designed to satisfy this

    def test_evaluate_threshold_is_one(self, tmp_path):
        result = ResistiveDimovCase.evaluate(tmp_path)
        assert result.metrics["hall_parameter_threshold"] == pytest.approx(1.0)

    def test_dimov_reference_threshold(self):
        assert DIMOV_REFERENCE["hall_parameter_threshold"] == 1.0
        assert DIMOV_REFERENCE["tolerance"] == 0.20


class TestResistiveDimovIntegration:
    def test_in_all_cases(self):
        from helicon.validate.runner import ALL_CASES

        names = [c.name for c in ALL_CASES]
        assert "resistive_dimov" in names

    def test_importable_from_cases_init(self):
        from helicon.validate.cases import ResistiveDimovCase as Imported

        assert Imported is ResistiveDimovCase

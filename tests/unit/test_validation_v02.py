"""Tests for v0.2 validation cases (Merino-Ahedo & MN1D)."""

from __future__ import annotations

from helicon.validate.cases.merino_ahedo import (
    REFERENCE_BETA_ETA,
    MerinoAhedoCase,
)
from helicon.validate.cases.mn1d_comparison import (
    MN1D_REFERENCE,
    MN1DComparisonCase,
)


class TestMerinoAhedoCase:
    """Tests for the Merino & Ahedo validation case definition."""

    def test_name(self) -> None:
        assert MerinoAhedoCase.name == "merino_ahedo_2016"

    def test_get_config_returns_simconfig(self) -> None:
        config = MerinoAhedoCase.get_config()
        assert config.nozzle.type == "converging_diverging"
        assert len(config.plasma.species) == 2

    def test_get_configs_three_betas(self) -> None:
        configs = MerinoAhedoCase.get_configs()
        assert len(configs) == 3
        for beta in REFERENCE_BETA_ETA:
            assert beta in configs

    def test_coil_current_increases_with_lower_beta(self) -> None:
        """Lower β → higher B → higher coil current."""
        configs = MerinoAhedoCase.get_configs()
        betas = sorted(REFERENCE_BETA_ETA.keys())
        currents = [configs[b].nozzle.coils[0].I for b in betas]
        # Current should decrease as β increases
        assert currents[0] > currents[-1]

    def test_evaluate_no_data(self, tmp_path) -> None:
        result = MerinoAhedoCase.evaluate(tmp_path)
        assert not result.passed
        assert result.case_name == "merino_ahedo_2016"

    def test_reference_eta_increases_with_beta(self) -> None:
        """Higher β → stronger ion detachment → higher η_d (physical expectation)."""
        betas = sorted(REFERENCE_BETA_ETA.keys())
        etas = [REFERENCE_BETA_ETA[b] for b in betas]
        # Validate physically: eta must strictly increase with beta
        for i in range(len(etas) - 1):
            assert etas[i] < etas[i + 1], (
                f"η_d({betas[i]}) = {etas[i]} not < η_d({betas[i+1]}) = {etas[i+1]}"
            )

    def test_reference_eta_in_valid_range(self) -> None:
        """All reference efficiencies must be in (0, 1]."""
        for beta, eta in REFERENCE_BETA_ETA.items():
            assert 0.0 < eta <= 1.0, f"η_d({beta}) = {eta} outside (0, 1]"

    def test_reference_data_used_by_evaluate(self, tmp_path) -> None:
        """evaluate() uses REFERENCE_BETA_ETA — tolerance must cover all betas."""
        result = MerinoAhedoCase.evaluate(tmp_path)
        assert result.case_name == "merino_ahedo_2016"
        # A relative error tolerance must be defined (used against all betas)
        assert "eta_d_relative_error" in result.tolerances
        tol = result.tolerances["eta_d_relative_error"]
        assert 0.0 < tol <= 1.0  # sensible tolerance (e.g. 10%)


class TestMN1DComparisonCase:
    """Tests for the MN1D comparison validation case."""

    def test_name(self) -> None:
        assert MN1DComparisonCase.name == "mn1d_comparison"

    def test_get_config_returns_simconfig(self) -> None:
        config = MN1DComparisonCase.get_config()
        assert config.nozzle.type == "solenoid"

    def test_get_configs_three_betas(self) -> None:
        configs = MN1DComparisonCase.get_configs()
        assert len(configs) == 3

    def test_reference_mach_starts_at_one(self) -> None:
        """All MN1D reference profiles should start at Mach 1 (sonic throat)."""
        for ref in MN1D_REFERENCE.values():
            assert ref["Mach_z"][0] == 1.0

    def test_reference_mach_increases_downstream(self) -> None:
        """Mach number should increase monotonically downstream."""
        for ref in MN1D_REFERENCE.values():
            mach = ref["Mach_z"]
            assert mach == sorted(mach)

    def test_evaluate_no_data(self, tmp_path) -> None:
        result = MN1DComparisonCase.evaluate(tmp_path)
        assert not result.passed
        assert result.case_name == "mn1d_comparison"

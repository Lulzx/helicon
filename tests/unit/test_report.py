"""Tests for helicon.postprocess.report module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helicon.postprocess.report import RunReport, generate_report, load_report, save_report


def _make_report(**kwargs) -> RunReport:
    defaults = dict(
        helicon_version="0.4.0",
        config_hash=None,
        thrust_N=None,
        isp_s=None,
        exhaust_velocity_ms=None,
        mass_flow_rate_kgs=None,
        detachment_momentum=None,
        detachment_particle=None,
        detachment_energy=None,
        plume_half_angle_deg=None,
        beam_efficiency=None,
        thrust_coefficient=None,
        radial_loss_fraction=None,
    )
    defaults.update(kwargs)
    return RunReport(**defaults)


class TestRunReport:
    """Tests for RunReport dataclass."""

    def test_create_report(self) -> None:
        report = RunReport(
            helicon_version="0.2.0",
            config_hash="abc123",
            thrust_N=1.5,
            isp_s=3000.0,
            exhaust_velocity_ms=29400.0,
            mass_flow_rate_kgs=5.1e-5,
            detachment_momentum=0.78,
            detachment_particle=0.85,
            detachment_energy=0.71,
            plume_half_angle_deg=12.3,
            beam_efficiency=0.92,
            thrust_coefficient=1.05,
            radial_loss_fraction=0.03,
        )
        assert report.thrust_N == 1.5
        assert report.isp_s == 3000.0

    def test_none_fields(self) -> None:
        report = RunReport(
            helicon_version="0.2.0",
            config_hash=None,
            thrust_N=None,
            isp_s=None,
            exhaust_velocity_ms=None,
            mass_flow_rate_kgs=None,
            detachment_momentum=None,
            detachment_particle=None,
            detachment_energy=None,
            plume_half_angle_deg=None,
            beam_efficiency=None,
            thrust_coefficient=None,
            radial_loss_fraction=None,
        )
        assert report.thrust_N is None
        assert report.config_hash is None


class TestSaveLoadReport:
    """Tests for save_report and load_report."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        report = RunReport(
            helicon_version="0.2.0",
            config_hash="test_hash",
            thrust_N=2.5,
            isp_s=4000.0,
            exhaust_velocity_ms=39200.0,
            mass_flow_rate_kgs=6.4e-5,
            detachment_momentum=0.80,
            detachment_particle=0.88,
            detachment_energy=0.73,
            plume_half_angle_deg=10.5,
            beam_efficiency=0.94,
            thrust_coefficient=1.10,
            radial_loss_fraction=0.02,
        )
        path = tmp_path / "report.json"
        save_report(report, path)

        loaded = load_report(path)
        assert loaded["results"]["thrust_N"] == 2.5
        assert loaded["results"]["isp_s"] == 4000.0
        assert loaded["config_hash"] == "test_hash"
        assert loaded["results"]["detachment_efficiency"]["momentum_based"] == 0.80

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        report = RunReport(
            helicon_version="0.2.0",
            config_hash=None,
            thrust_N=None,
            isp_s=None,
            exhaust_velocity_ms=None,
            mass_flow_rate_kgs=None,
            detachment_momentum=None,
            detachment_particle=None,
            detachment_energy=None,
            plume_half_angle_deg=None,
            beam_efficiency=None,
            thrust_coefficient=None,
            radial_loss_fraction=None,
        )
        path = tmp_path / "deep" / "nested" / "report.json"
        save_report(report, path)
        assert path.exists()

    def test_json_format(self, tmp_path: Path) -> None:
        report = RunReport(
            helicon_version="0.2.0",
            config_hash=None,
            thrust_N=1.0,
            isp_s=None,
            exhaust_velocity_ms=None,
            mass_flow_rate_kgs=None,
            detachment_momentum=None,
            detachment_particle=None,
            detachment_energy=None,
            plume_half_angle_deg=None,
            beam_efficiency=None,
            thrust_coefficient=None,
            radial_loss_fraction=None,
        )
        path = tmp_path / "report.json"
        save_report(report, path)
        data = json.loads(path.read_text())
        assert "helicon_version" in data
        assert "thrust_N" in data["results"]


class TestGenerateReport:
    """Tests for generate_report (graceful failure with no data)."""

    def test_empty_dir(self, tmp_path: Path) -> None:
        """All metrics should be None when no simulation output exists."""
        report = generate_report(tmp_path)
        assert report.thrust_N is None
        assert report.isp_s is None
        assert report.detachment_momentum is None
        assert report.plume_half_angle_deg is None
        assert report.beam_efficiency is None

    def test_version_populated(self, tmp_path: Path) -> None:
        report = generate_report(tmp_path)
        assert report.helicon_version is not None
        assert len(report.helicon_version) > 0

    def test_config_hash_forwarded(self, tmp_path: Path) -> None:
        report = generate_report(tmp_path, config_hash="abc123")
        assert report.config_hash == "abc123"


class TestMassRatioReduced:
    """Tests for mass_ratio_reduced flag (spec §14)."""

    def test_default_false(self) -> None:
        report = _make_report()
        assert report.mass_ratio_reduced is False

    def test_set_true(self) -> None:
        report = _make_report(mass_ratio_reduced=True)
        assert report.mass_ratio_reduced is True

    def test_in_spec_dict(self) -> None:
        report = _make_report(mass_ratio_reduced=True)
        d = report.to_spec_dict()
        assert d["mass_ratio_reduced"] is True

    def test_false_in_spec_dict(self) -> None:
        report = _make_report(mass_ratio_reduced=False)
        d = report.to_spec_dict()
        assert d["mass_ratio_reduced"] is False

    def test_generate_report_with_reduced_ratio_config(self, tmp_path: Path) -> None:
        from helicon.config.parser import (
            CoilConfig,
            DomainConfig,
            NozzleConfig,
            PlasmaSourceConfig,
            SimConfig,
        )

        config = SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=0.1, I=10000)],
                domain=DomainConfig(z_min=0.0, z_max=1.0, r_max=0.5),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e19, T_i_eV=100.0, T_e_eV=50.0,
                v_injection_ms=50000.0, mass_ratio=100.0,
            ),
        )
        report = generate_report(tmp_path, config=config)
        assert report.mass_ratio_reduced is True

    def test_generate_report_full_ratio_not_flagged(self, tmp_path: Path) -> None:
        from helicon.config.parser import (
            CoilConfig,
            DomainConfig,
            NozzleConfig,
            PlasmaSourceConfig,
            SimConfig,
        )

        config = SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=0.1, I=10000)],
                domain=DomainConfig(z_min=0.0, z_max=1.0, r_max=0.5),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e19, T_i_eV=100.0, T_e_eV=50.0,
                v_injection_ms=50000.0, mass_ratio=None,
            ),
        )
        report = generate_report(tmp_path, config=config)
        assert report.mass_ratio_reduced is False


class TestValidationProximityInReport:
    """Tests for validation_proximity in spec §6.3 output (spec §7.3)."""

    def test_present_in_spec_dict_with_config(self) -> None:
        from helicon.config.parser import (
            CoilConfig,
            DomainConfig,
            NozzleConfig,
            PlasmaSourceConfig,
            SimConfig,
        )

        config = SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=0.1, I=10000)],
                domain=DomainConfig(z_min=0.0, z_max=1.0, r_max=0.5),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e18, T_i_eV=100.0, T_e_eV=50.0, v_injection_ms=50000.0,
            ),
        )
        report = _make_report()
        d = report.to_spec_dict(config=config)
        assert "validation_proximity" in d
        prox = d["validation_proximity"]
        assert prox is not None
        assert "nearest_case" in prox
        assert "distance" in prox
        assert "in_validated_region" in prox

    def test_none_without_config(self) -> None:
        report = _make_report()
        d = report.to_spec_dict()
        # Without config, validation_proximity should be None
        assert d["validation_proximity"] is None

    def test_persists_if_already_set(self) -> None:
        prox_data = {"nearest_case": "vasimr", "distance": 0.5, "in_validated_region": True, "parameter_distances": {}, "warning": None}
        report = _make_report(validation_proximity=prox_data)
        d = report.to_spec_dict()
        assert d["validation_proximity"]["nearest_case"] == "vasimr"


class TestSpecDictResultsFields:
    """Tests that to_spec_dict() includes all spec §6.3 results fields."""

    def test_exhaust_velocity_in_results(self) -> None:
        report = _make_report(exhaust_velocity_ms=109800.0)
        d = report.to_spec_dict()
        assert "exhaust_velocity_ms" in d["results"]
        assert d["results"]["exhaust_velocity_ms"] == 109800.0

    def test_mass_flow_rate_in_results(self) -> None:
        report = _make_report(mass_flow_rate_kgs=4.39e-5)
        d = report.to_spec_dict()
        assert "mass_flow_rate_kgs" in d["results"]
        assert d["results"]["mass_flow_rate_kgs"] == pytest.approx(4.39e-5)

    def test_all_required_results_keys_present(self) -> None:
        report = _make_report()
        d = report.to_spec_dict()
        required = [
            "thrust_N", "isp_s", "exhaust_velocity_ms", "mass_flow_rate_kgs",
            "detachment_efficiency", "plume_half_angle_deg", "beam_efficiency",
            "radial_loss_fraction", "convergence",
        ]
        for key in required:
            assert key in d["results"], f"Missing required key: {key}"

    def test_detachment_efficiency_subkeys(self) -> None:
        report = _make_report()
        d = report.to_spec_dict()
        det = d["results"]["detachment_efficiency"]
        for key in ("momentum_based", "particle_based", "energy_based"):
            assert key in det


class TestValidationFlags:
    """Tests for computed validation_flags (spec §6.3, §7.1)."""

    def test_structure_present(self) -> None:
        report = _make_report()
        d = report.to_spec_dict()
        assert "validation_flags" in d
        vf = d["validation_flags"]
        assert "steady_state_reached" in vf
        assert "particle_statistics_sufficient" in vf
        assert "energy_conservation_error" in vf

    def test_steady_state_none_without_data(self) -> None:
        report = _make_report()
        d = report.to_spec_dict()
        assert d["validation_flags"]["steady_state_reached"] is None

    def test_steady_state_true_when_converged(self) -> None:
        report = _make_report(thrust_relative_change_last_10pct=0.003)
        d = report.to_spec_dict()
        assert d["validation_flags"]["steady_state_reached"] is True

    def test_steady_state_false_when_not_converged(self) -> None:
        report = _make_report(thrust_relative_change_last_10pct=0.05)
        d = report.to_spec_dict()
        assert d["validation_flags"]["steady_state_reached"] is False

    def test_steady_state_boundary_at_1pct(self) -> None:
        report = _make_report(thrust_relative_change_last_10pct=0.01)
        d = report.to_spec_dict()
        # 0.01 is NOT < 0.01 so should be False
        assert d["validation_flags"]["steady_state_reached"] is False

    def test_particle_stats_sufficient_when_large(self) -> None:
        report = _make_report(particle_count_exit=8_420_000)
        d = report.to_spec_dict()
        assert d["validation_flags"]["particle_statistics_sufficient"] is True

    def test_particle_stats_insufficient_when_small(self) -> None:
        report = _make_report(particle_count_exit=50_000)
        d = report.to_spec_dict()
        assert d["validation_flags"]["particle_statistics_sufficient"] is False

    def test_particle_stats_none_without_data(self) -> None:
        report = _make_report()
        d = report.to_spec_dict()
        assert d["validation_flags"]["particle_statistics_sufficient"] is None


class TestAutoPlots:
    """Tests for generate_all_plots (spec §6.3)."""

    def test_import(self) -> None:
        from helicon.postprocess.plots import generate_all_plots
        assert callable(generate_all_plots)

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        from helicon.postprocess.plots import generate_all_plots
        result = generate_all_plots(tmp_path)
        assert isinstance(result, list)

    def test_missing_bfield_skips_topology(self, tmp_path: Path) -> None:
        from helicon.postprocess.plots import generate_all_plots
        # No bfield file → topology plot skipped gracefully
        result = generate_all_plots(tmp_path, bfield_file=tmp_path / "nonexistent.h5")
        assert all("bfield_topology" not in str(p) for p in result)

    def test_plots_dir_created(self, tmp_path: Path) -> None:
        from helicon.postprocess.plots import generate_all_plots
        generate_all_plots(tmp_path)
        assert (tmp_path / "plots").exists()

    def test_bfield_topology_saved(self, tmp_path: Path) -> None:
        """With a real BField HDF5, topology plot should be generated."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not available")

        from helicon.config.parser import CoilConfig, DomainConfig, NozzleConfig
        from helicon.fields.biot_savart import BField, Coil, Grid
        from helicon.postprocess.plots import generate_all_plots

        coils = [Coil(z=0.0, r=0.1, I=10000)]
        grid = Grid(z_min=0.0, z_max=1.0, r_max=0.5, nz=32, nr=16)

        from helicon.fields import compute_bfield
        bf = compute_bfield(coils, grid)
        bf_path = tmp_path / "applied_bfield.h5"
        bf.save(str(bf_path))

        saved = generate_all_plots(tmp_path, bfield_file=bf_path)
        topology_plots = [p for p in saved if "bfield_topology" in p.name]
        assert len(topology_plots) == 1
        assert topology_plots[0].exists()

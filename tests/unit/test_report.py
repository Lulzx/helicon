"""Tests for helicon.postprocess.report module."""

from __future__ import annotations

import json
from pathlib import Path

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

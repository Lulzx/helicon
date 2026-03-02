"""Tests for magnozzlex.postprocess.report module."""

from __future__ import annotations

import json
from pathlib import Path

from magnozzlex.postprocess.report import RunReport, generate_report, load_report, save_report


class TestRunReport:
    """Tests for RunReport dataclass."""

    def test_create_report(self) -> None:
        report = RunReport(
            magnozzlex_version="0.2.0",
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
            magnozzlex_version="0.2.0",
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
            magnozzlex_version="0.2.0",
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
            magnozzlex_version="0.2.0",
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
            magnozzlex_version="0.2.0",
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
        assert "magnozzlex_version" in data
        assert "thrust_N" in data["results"]


class TestGenerateReport:
    """Tests for generate_report (graceful failure with no data)."""

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Should return a report with all None metrics for empty dir."""
        report = generate_report(tmp_path)
        assert isinstance(report, RunReport)
        assert report.thrust_N is None
        assert report.detachment_momentum is None
        assert report.plume_half_angle_deg is None

    def test_version_populated(self, tmp_path: Path) -> None:
        report = generate_report(tmp_path)
        assert report.magnozzlex_version is not None
        assert len(report.magnozzlex_version) > 0

    def test_config_hash_forwarded(self, tmp_path: Path) -> None:
        report = generate_report(tmp_path, config_hash="abc123")
        assert report.config_hash == "abc123"

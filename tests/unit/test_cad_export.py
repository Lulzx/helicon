"""Tests for helicon.export.cad — STEP/IGES export (v2.0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from helicon.export.cad import coil_geometry_report, export_coils_iges, export_coils_step

# ---------------------------------------------------------------------------
# Fake config for testing
# ---------------------------------------------------------------------------


class FakeCoil:
    def __init__(self, z: float, r: float, I: float):
        self.z = z
        self.r = r
        self.I = I


class FakeNozzle:
    def __init__(self, coils):
        self.coils = coils
        self.type = "solenoid"


class FakeConfig:
    def __init__(self, coils):
        self.nozzle = FakeNozzle(coils)


def _make_config(n: int = 2) -> FakeConfig:
    coils = [FakeCoil(z=i * 0.1, r=0.1 + i * 0.02, I=10000.0 * (i + 1)) for i in range(n)]
    return FakeConfig(coils)


# ---------------------------------------------------------------------------
# STEP export
# ---------------------------------------------------------------------------


class TestExportCoilsStep:
    def test_creates_file(self):
        config = _make_config(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coils.step"
            result = export_coils_step(config, path)
            assert result.exists()

    def test_contains_iso_header(self):
        config = _make_config(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coils.step"
            export_coils_step(config, path)
            content = path.read_text()
        assert "ISO-10303-21" in content

    def test_contains_end_footer(self):
        config = _make_config(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coils.step"
            export_coils_step(config, path)
            content = path.read_text()
        assert "END-ISO-10303-21" in content

    def test_creates_parent_dirs(self):
        config = _make_config(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "dir" / "coils.step"
            export_coils_step(config, path)
            assert path.exists()

    def test_returns_absolute_path(self):
        config = _make_config(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coils.step"
            result = export_coils_step(config, path)
            assert result.is_absolute()


# ---------------------------------------------------------------------------
# IGES export
# ---------------------------------------------------------------------------


class TestExportCoilsIges:
    def test_creates_file(self):
        config = _make_config(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coils.igs"
            result = export_coils_iges(config, path)
            assert result.exists()

    def test_has_terminate_section(self):
        config = _make_config(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coils.igs"
            export_coils_iges(config, path)
            content = path.read_text()
        assert "T      1" in content

    def test_multiple_coils(self):
        config = _make_config(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coils.igs"
            export_coils_iges(config, path)
            content = path.read_text()
        # Each coil should appear as Coil00, Coil01, Coil02
        assert "Coil00" in content
        assert "Coil02" in content


# ---------------------------------------------------------------------------
# Geometry report
# ---------------------------------------------------------------------------


class TestCoilGeometryReport:
    def test_report_structure(self):
        config = _make_config(2)
        report = coil_geometry_report(config)
        assert "coils" in report
        assert "summary" in report
        assert report["summary"]["n_coils"] == 2

    def test_circumference_positive(self):
        config = _make_config(1)
        report = coil_geometry_report(config)
        assert report["coils"][0]["circumference_m"] > 0

    def test_z_span(self):
        config = _make_config(3)
        report = coil_geometry_report(config)
        # Coils at z=0, 0.1, 0.2 → span = 0.2
        assert report["summary"]["z_span_m"] == pytest.approx(0.2, rel=1e-5)

    def test_total_amp_turns(self):
        coils = [FakeCoil(z=0.0, r=0.1, I=10000.0), FakeCoil(z=0.1, r=0.1, I=20000.0)]
        config = FakeConfig(coils)
        report = coil_geometry_report(config)
        assert report["summary"]["total_amp_turns_A"] == pytest.approx(30000.0, rel=1e-5)

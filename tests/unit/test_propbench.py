"""Tests for helicon.postprocess.propbench module."""

from __future__ import annotations

import json
from pathlib import Path

from helicon.postprocess.propbench import (
    PropBenchResult,
    load_propbench,
    save_propbench,
    to_propbench,
)
from helicon.postprocess.report import RunReport


def _make_run_report(**overrides) -> RunReport:
    defaults = dict(
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
    defaults.update(overrides)
    return RunReport(**defaults)


class TestPropBenchResult:
    def test_defaults(self) -> None:
        result = PropBenchResult()
        assert result.propbench_version == "0.1"
        assert result.code_name == "Helicon"
        assert result.mass_ratio_reduced is False
        assert result.electron_model == "kinetic"
        assert result.species == []

    def test_custom_fields(self) -> None:
        result = PropBenchResult(
            code_version="0.2.0",
            thrust_N=1.5,
            isp_s=3000.0,
        )
        assert result.thrust_N == 1.5
        assert result.isp_s == 3000.0


class TestToPropBench:
    def test_from_run_report_all_none(self) -> None:
        rr = _make_run_report()
        pb = to_propbench(rr)
        assert pb.code_version == "0.2.0"
        assert pb.thrust_N is None
        assert pb.isp_s is None
        assert pb.code_name == "Helicon"
        assert pb.propbench_version == "0.1"

    def test_from_run_report_with_values(self) -> None:
        rr = _make_run_report(thrust_N=2.0, isp_s=4000.0, beam_efficiency=0.9)
        pb = to_propbench(rr)
        assert pb.thrust_N == 2.0
        assert pb.isp_s == 4000.0
        assert pb.beam_efficiency == 0.9

    def test_timestamp_is_set(self) -> None:
        rr = _make_run_report()
        pb = to_propbench(rr)
        assert len(pb.timestamp) > 0

    def test_mass_ratio_reduced_flag(self) -> None:
        rr = _make_run_report()
        pb = to_propbench(rr, mass_ratio_reduced=True)
        assert pb.mass_ratio_reduced is True

    def test_electron_model_override(self) -> None:
        rr = _make_run_report()
        pb = to_propbench(rr, electron_model="fluid")
        assert pb.electron_model == "fluid"


class TestSaveLoadPropBench:
    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        result = PropBenchResult(code_version="0.2.0", thrust_N=1.5)
        path = tmp_path / "propbench.json"
        save_propbench(result, path)
        data = json.loads(path.read_text())
        assert data["propbench_version"] == "0.1"
        assert data["code_name"] == "Helicon"
        assert data["thrust_N"] == 1.5

    def test_roundtrip(self, tmp_path: Path) -> None:
        original = PropBenchResult(
            code_version="0.2.0",
            thrust_N=2.5,
            isp_s=3500.0,
            species=["D+", "e-"],
            nozzle_type="solenoid",
            mass_ratio_reduced=True,
            electron_model="fluid",
        )
        path = tmp_path / "roundtrip.json"
        save_propbench(original, path)
        loaded = load_propbench(path)
        assert loaded.code_version == original.code_version
        assert loaded.thrust_N == original.thrust_N
        assert loaded.isp_s == original.isp_s
        assert loaded.species == original.species
        assert loaded.nozzle_type == original.nozzle_type
        assert loaded.mass_ratio_reduced is True
        assert loaded.electron_model == "fluid"

    def test_propbench_version_present(self, tmp_path: Path) -> None:
        result = PropBenchResult()
        path = tmp_path / "version.json"
        save_propbench(result, path)
        data = json.loads(path.read_text())
        assert "propbench_version" in data
        assert data["propbench_version"] == "0.1"

    def test_code_name_is_helicon(self, tmp_path: Path) -> None:
        result = PropBenchResult()
        path = tmp_path / "codename.json"
        save_propbench(result, path)
        data = json.loads(path.read_text())
        assert data["code_name"] == "Helicon"

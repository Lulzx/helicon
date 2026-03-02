"""Tests for helicon.postprocess.propbench module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from helicon.postprocess.propbench import (
    PropBenchResult,
    load_propbench,
    save_propbench,
    to_poliastro_csv,
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


class TestPoliastroCSV:
    """Tests for poliastro-compatible CSV export (spec §6.3 / v0.4)."""

    def test_writes_csv_file(self, tmp_path: Path) -> None:
        result = PropBenchResult(isp_s=11200.0, thrust_N=4.82)
        path = tmp_path / "poliastro.csv"
        to_poliastro_csv(result, path)
        assert path.exists()

    def test_header_row(self, tmp_path: Path) -> None:
        result = PropBenchResult(isp_s=11200.0, thrust_N=4.82)
        path = tmp_path / "poliastro.csv"
        to_poliastro_csv(result, path)
        with path.open() as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames or []) == {
                "isp_s",
                "thrust_N",
                "mass_flow_rate_kgs",
                "exhaust_velocity_ms",
            }

    def test_values_written(self, tmp_path: Path) -> None:
        result = PropBenchResult(
            isp_s=11200.0,
            thrust_N=4.82,
            mass_flow_rate_kgs=4.39e-5,
            exhaust_velocity_ms=109800.0,
        )
        path = tmp_path / "poliastro.csv"
        to_poliastro_csv(result, path)
        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["isp_s"]) == 11200.0
        assert float(rows[0]["thrust_N"]) == 4.82

    def test_multiple_results(self, tmp_path: Path) -> None:
        results = [
            PropBenchResult(isp_s=11200.0, thrust_N=4.82),
            PropBenchResult(isp_s=12000.0, thrust_N=5.0),
        ]
        path = tmp_path / "multi.csv"
        to_poliastro_csv(results, path)
        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2
        assert float(rows[1]["isp_s"]) == 12000.0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = PropBenchResult(isp_s=11200.0)
        path = tmp_path / "deep" / "nested" / "poliastro.csv"
        to_poliastro_csv(result, path)
        assert path.exists()

    def test_none_values_written_as_empty(self, tmp_path: Path) -> None:
        result = PropBenchResult()
        path = tmp_path / "nones.csv"
        to_poliastro_csv(result, path)
        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1

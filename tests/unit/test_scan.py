"""Tests for helicon.optimize.scan."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from helicon.config.parser import SimConfig
from helicon.optimize.scan import (
    ParameterRange,
    ScanPoint,
    _apply_params,
    _set_nested,
    generate_scan_points,
    run_scan,
)

# ---------------------------------------------------------------------------
# Minimal base config used across tests
# ---------------------------------------------------------------------------
_BASE_DATA = {
    "nozzle": {
        "type": "solenoid",
        "coils": [{"z": 0.0, "r": 0.1, "I": 1000.0}],
        "domain": {"z_min": -0.5, "z_max": 1.0, "r_max": 0.3},
        "resolution": {"nz": 16, "nr": 8},
    },
    "plasma": {
        "n0": 1e18,
        "T_i_eV": 1000.0,
        "T_e_eV": 500.0,
        "v_injection_ms": 50000.0,
    },
}


@pytest.fixture
def base_config() -> SimConfig:
    return SimConfig.model_validate(_BASE_DATA)


# ---------------------------------------------------------------------------
# ParameterRange
# ---------------------------------------------------------------------------
class TestParameterRange:
    def test_values_linspace(self):
        r = ParameterRange(path="coils.0.I", low=500.0, high=2000.0, n=4)
        v = r.values()
        assert len(v) == 4
        assert v[0] == pytest.approx(500.0)
        assert v[-1] == pytest.approx(2000.0)

    def test_values_single_point(self):
        r = ParameterRange(path="coils.0.I", low=1000.0, high=1000.0, n=1)
        assert r.values()[0] == pytest.approx(1000.0)

    def test_from_string(self):
        r = ParameterRange.from_string("coils.0.I:500:2000:4")
        assert r.path == "coils.0.I"
        assert r.low == pytest.approx(500.0)
        assert r.high == pytest.approx(2000.0)
        assert r.n == 4

    def test_from_string_float_bounds(self):
        r = ParameterRange.from_string("plasma.T_i_eV:500.5:1500.5:3")
        assert r.low == pytest.approx(500.5)

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            ParameterRange.from_string("bad_format")

    def test_from_string_too_few_parts(self):
        with pytest.raises(ValueError):
            ParameterRange.from_string("path:1:2")


# ---------------------------------------------------------------------------
# _set_nested
# ---------------------------------------------------------------------------
class TestSetNested:
    def test_simple_key(self):
        d = {"a": 1}
        _set_nested(d, "a", 42)
        assert d["a"] == 42

    def test_nested_dict(self):
        d = {"plasma": {"T_i_eV": 1000.0}}
        _set_nested(d, "plasma.T_i_eV", 2000.0)
        assert d["plasma"]["T_i_eV"] == 2000.0

    def test_full_path_list_index(self):
        d = {"nozzle": {"coils": [{"I": 100}]}}
        _set_nested(d, "nozzle.coils.0.I", 999)
        assert d["nozzle"]["coils"][0]["I"] == 999

    def test_shorthand_path_searches_nested(self):
        # "coils.0.I" should resolve even though coils is under nozzle
        d = {"nozzle": {"coils": [{"I": 100}]}}
        _set_nested(d, "coils.0.I", 999)
        assert d["nozzle"]["coils"][0]["I"] == 999

    def test_missing_path_raises(self):
        d = {"a": 1}
        with pytest.raises(KeyError):
            _set_nested(d, "nonexistent.path", 0)

    def test_returns_dict(self):
        d = {"x": 0}
        result = _set_nested(d, "x", 1)
        assert result is d


# ---------------------------------------------------------------------------
# _apply_params
# ---------------------------------------------------------------------------
def test_apply_params_does_not_mutate_base(base_config):
    params = {"coils.0.I": 9999.0}
    new_config = _apply_params(base_config, params)
    assert new_config.nozzle.coils[0].I == pytest.approx(9999.0)
    assert base_config.nozzle.coils[0].I == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# generate_scan_points
# ---------------------------------------------------------------------------
class TestGenerateScanPoints:
    def test_grid_single_axis(self, base_config):
        ranges = [ParameterRange("coils.0.I", 500.0, 1500.0, 3)]
        points = generate_scan_points(base_config, ranges)
        assert len(points) == 3
        currents = [p.params["coils.0.I"] for p in points]
        assert currents[0] == pytest.approx(500.0)
        assert currents[1] == pytest.approx(1000.0)
        assert currents[2] == pytest.approx(1500.0)

    def test_grid_two_axes_cartesian(self, base_config):
        ranges = [
            ParameterRange("coils.0.I", 500.0, 1500.0, 3),
            ParameterRange("plasma.T_i_eV", 500.0, 1500.0, 2),
        ]
        points = generate_scan_points(base_config, ranges)
        assert len(points) == 6  # 3 × 2

    def test_grid_empty_ranges(self, base_config):
        points = generate_scan_points(base_config, [])
        assert len(points) == 1  # one point: the base config

    def test_lhc_count(self, base_config):
        ranges = [
            ParameterRange("coils.0.I", 500.0, 1500.0, 4),
            ParameterRange("plasma.T_i_eV", 500.0, 1500.0, 3),
        ]
        points = generate_scan_points(base_config, ranges, method="lhc")
        assert len(points) == 12  # 4 × 3

    def test_lhc_bounds_respected(self, base_config):
        ranges = [ParameterRange("coils.0.I", 500.0, 1500.0, 8)]
        points = generate_scan_points(base_config, ranges, method="lhc")
        currents = [p.params["coils.0.I"] for p in points]
        assert all(500.0 <= c <= 1500.0 for c in currents)

    def test_lhc_reproducible(self, base_config):
        ranges = [ParameterRange("coils.0.I", 500.0, 1500.0, 5)]
        p1 = generate_scan_points(base_config, ranges, method="lhc", seed=7)
        p2 = generate_scan_points(base_config, ranges, method="lhc", seed=7)
        assert [pt.params["coils.0.I"] for pt in p1] == [
            pt.params["coils.0.I"] for pt in p2
        ]

    def test_config_modified_correctly(self, base_config):
        ranges = [ParameterRange("coils.0.I", 999.0, 999.0, 1)]
        points = generate_scan_points(base_config, ranges)
        assert points[0].config.nozzle.coils[0].I == pytest.approx(999.0)

    def test_unknown_method_raises(self, base_config):
        with pytest.raises(ValueError, match="Unknown scan method"):
            generate_scan_points(base_config, [], method="bad_method")

    def test_indices_sequential(self, base_config):
        ranges = [ParameterRange("coils.0.I", 500.0, 1500.0, 5)]
        points = generate_scan_points(base_config, ranges)
        assert [p.index for p in points] == list(range(5))

    def test_scan_point_type(self, base_config):
        ranges = [ParameterRange("coils.0.I", 500.0, 1500.0, 2)]
        points = generate_scan_points(base_config, ranges)
        assert all(isinstance(p, ScanPoint) for p in points)


# ---------------------------------------------------------------------------
# run_scan (dry run)
# ---------------------------------------------------------------------------
def test_run_scan_dry_run(base_config):
    ranges = [ParameterRange("coils.0.I", 800.0, 1200.0, 3)]
    with tempfile.TemporaryDirectory() as tmp:
        result = run_scan(base_config, ranges, output_base=tmp, dry_run=True)
    assert len(result.points) == 3
    assert len(result.metrics) == 3
    assert all(m["success"] for m in result.metrics)
    assert result.param_names == ["coils.0.I"]


def test_run_scan_creates_subdirs(base_config):
    ranges = [ParameterRange("coils.0.I", 800.0, 1200.0, 2)]
    with tempfile.TemporaryDirectory() as tmp:
        run_scan(base_config, ranges, output_base=tmp, dry_run=True)
        subdirs = sorted(Path(tmp).iterdir())
    assert len(subdirs) == 2
    assert subdirs[0].name == "point_0000"
    assert subdirs[1].name == "point_0001"

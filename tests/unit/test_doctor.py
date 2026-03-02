"""Tests for helicon.doctor environment checker."""

from __future__ import annotations

import json

from click.testing import CliRunner

from helicon.cli import main
from helicon.doctor import DepCheck, DoctorReport, check_environment


def test_check_environment_returns_doctor_report():
    report = check_environment()
    assert isinstance(report, DoctorReport)


def test_python_version_nonempty():
    report = check_environment()
    assert len(report.python_version) >= 5  # e.g. "3.12.0"


def test_python_ok_on_this_runtime():
    import sys

    report = check_environment()
    expected = (sys.version_info.major, sys.version_info.minor) >= (3, 11)
    assert report.python_ok == expected


def test_checks_nonempty():
    report = check_environment()
    assert len(report.checks) > 0


def test_all_checks_are_dep_check():
    report = check_environment()
    for c in report.checks:
        assert isinstance(c, DepCheck)


def test_core_deps_available():
    """numpy, pydantic, click must be available in the test environment."""
    report = check_environment()
    by_name = {c.name: c for c in report.checks}
    for pkg in ("numpy", "pydantic", "click"):
        assert by_name[pkg].available, f"{pkg} should be available"


def test_required_flags_set():
    report = check_environment()
    required = [c for c in report.checks if c.required]
    assert len(required) >= 5  # numpy, scipy, pydantic, click, h5py at minimum


def test_healthy_on_this_machine():
    report = check_environment()
    # We're in the dev environment; all required deps should be present
    assert report.healthy


def test_all_required_ok_implies_available():
    report = check_environment()
    if report.all_required_ok:
        for c in report.checks:
            if c.required:
                assert c.available


def test_summary_contains_python():
    report = check_environment()
    summary = report.summary()
    assert "Python" in summary or "python" in summary.lower()


def test_summary_contains_overall():
    report = check_environment()
    summary = report.summary()
    assert "Overall" in summary or "HEALTHY" in summary or "ISSUES" in summary


def test_to_dict_keys():
    report = check_environment()
    d = report.to_dict()
    assert "python_version" in d
    assert "python_ok" in d
    assert "healthy" in d
    assert "checks" in d
    assert "warpx_found" in d


def test_to_dict_json_serialisable():
    report = check_environment()
    d = report.to_dict()
    text = json.dumps(d)
    assert len(text) > 10


def test_dep_check_ok_status():
    c = DepCheck(name="numpy", available=True, version="2.1.0", required=True)
    assert "OK" in c.status_str()
    assert "numpy" in c.status_str()


def test_dep_check_missing_required():
    c = DepCheck(name="madeup", available=False, version=None, required=True, note="needed")
    s = c.status_str()
    assert "MISSING" in s


def test_dep_check_missing_optional():
    c = DepCheck(name="optpkg", available=False, version=None, required=False, note="optional")
    s = c.status_str()
    assert "OPTIONAL" in s


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_doctor_cmd_exit_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0, result.output


def test_doctor_cmd_shows_python():
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert "Python" in result.output or "python" in result.output.lower()


def test_doctor_cmd_json():
    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "python_version" in data
    assert "healthy" in data
    assert "checks" in data

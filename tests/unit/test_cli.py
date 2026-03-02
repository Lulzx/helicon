"""Tests for helicon.cli."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner

from helicon.cli import main


def _write_config(tmpdir: str) -> str:
    data = {
        "nozzle": {
            "type": "solenoid",
            "coils": [{"z": 0.0, "r": 0.1, "I": 10000}],
            "domain": {"z_min": -0.3, "z_max": 1.0, "r_max": 0.5},
        },
        "plasma": {
            "n0": 1e18,
            "T_i_eV": 100,
            "T_e_eV": 100,
            "v_injection_ms": 50000,
        },
    }
    path = Path(tmpdir) / "test.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.2.0" in result.output


def test_run_dry_run_with_config():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_config(tmpdir)
        output_dir = Path(tmpdir) / "out"
        result = runner.invoke(
            main, ["run", "--config", config_path, "--output", str(output_dir), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "Dry run complete" in result.output


def test_run_dry_run_with_preset():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "out"
        result = runner.invoke(
            main, ["run", "--preset", "sunbird", "--output", str(output_dir), "--dry-run"]
        )
        assert result.exit_code == 0, result.output


def test_run_validate_config():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_config(tmpdir)
        result = runner.invoke(main, ["run", "--config", config_path, "--validate-config"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()


def test_run_no_args():
    runner = CliRunner()
    result = runner.invoke(main, ["run"])
    assert result.exit_code != 0


def test_run_both_config_and_preset():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _write_config(tmpdir)
        result = runner.invoke(main, ["run", "--config", config_path, "--preset", "sunbird"])
        assert result.exit_code != 0

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
    assert "2.8.0" in result.output


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


# ---------------------------------------------------------------------------
# v2.1 commands
# ---------------------------------------------------------------------------


def test_array_cmd_defaults():
    runner = CliRunner()
    result = runner.invoke(main, ["array", "--n", "2"])
    assert result.exit_code == 0
    assert "thrust" in result.output.lower() or "Thrust" in result.output


def test_array_cmd_close():
    """Close thrusters should show a non-zero penalty."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["array", "--n", "2", "--sep", "0.05", "--angle", "30.0", "--ref-z", "1.0"],
    )
    assert result.exit_code == 0
    assert "%" in result.output  # interaction penalty displayed


def test_plugins_cmd_empty():
    runner = CliRunner()
    result = runner.invoke(main, ["plugins"])
    assert result.exit_code == 0


def test_valdb_stats_empty(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["valdb", "--db", str(tmp_path), "stats"])
    assert result.exit_code == 0
    assert "0" in result.output


def test_valdb_add_and_query(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "valdb",
            "--db",
            str(tmp_path),
            "add",
            "--case-id",
            "test_case",
            "--source",
            "unit test",
            "--contributor",
            "tester",
            "--type",
            "simulation",
        ],
    )
    assert result.exit_code == 0

    result2 = runner.invoke(main, ["valdb", "--db", str(tmp_path), "query"])
    assert result2.exit_code == 0
    assert "test_case" in result2.output


def test_valdb_export_json(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "out.json")
    runner.invoke(
        main,
        [
            "valdb",
            "--db",
            str(tmp_path),
            "add",
            "--case-id",
            "c1",
            "--source",
            "s",
            "--contributor",
            "me",
            "--type",
            "analytical",
        ],
    )
    result = runner.invoke(main, ["valdb", "--db", str(tmp_path), "export", "--output", out])
    assert result.exit_code == 0
    assert (tmp_path / "out.json").exists()


def test_perf_cmd():
    runner = CliRunner()
    result = runner.invoke(main, ["perf", "--no-bandwidth"])
    assert result.exit_code == 0
    assert "Chip" in result.output or "chip" in result.output.lower()


def test_perf_cmd_json():
    runner = CliRunner()
    result = runner.invoke(main, ["perf", "--no-bandwidth", "--json"])
    assert result.exit_code == 0
    data = __import__("json").loads(result.output)
    assert "chip_model" in data
    assert "openmp" in data

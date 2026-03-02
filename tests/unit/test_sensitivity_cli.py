"""Tests for helicon sensitivity CLI command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from helicon.cli import main


def test_sensitivity_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["sensitivity", "--n-samples", "8"])
    assert result.exit_code == 0, result.output


def test_sensitivity_output_contains_sobol():
    runner = CliRunner()
    result = runner.invoke(main, ["sensitivity", "--n-samples", "8"])
    output = result.output.lower()
    assert "sobol" in output or "s1" in output or "sensitivity" in output


def test_sensitivity_output_contains_params():
    runner = CliRunner()
    result = runner.invoke(main, ["sensitivity", "--n-samples", "8"])
    assert "coil_r_m" in result.output or "coil" in result.output.lower()


def test_sensitivity_json_output():
    runner = CliRunner()
    result = runner.invoke(main, ["sensitivity", "--n-samples", "8", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "S1" in data
    assert "ST" in data
    assert "param_names" in data


def test_sensitivity_json_s1_length():
    runner = CliRunner()
    result = runner.invoke(main, ["sensitivity", "--n-samples", "8", "--json"])
    data = json.loads(result.output)
    assert len(data["S1"]) == 3  # r, I, z_max
    assert len(data["ST"]) == 3


def test_sensitivity_preset_highpower():
    runner = CliRunner()
    result = runner.invoke(
        main, ["sensitivity", "--preset", "highpower", "--n-samples", "8"]
    )
    assert result.exit_code == 0, result.output


def test_sensitivity_json_to_file(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "sens.json")
    result = runner.invoke(
        main, ["sensitivity", "--n-samples", "8", "--json", "--output", out]
    )
    assert result.exit_code == 0, result.output
    with open(out) as fh:
        data = json.load(fh)
    assert "S1" in data


def test_sensitivity_text_to_file(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "sens.txt")
    result = runner.invoke(
        main, ["sensitivity", "--n-samples", "8", "--output", out]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "sens.txt").exists()

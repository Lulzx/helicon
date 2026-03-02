"""Tests for helicon.scaffold config generator and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from helicon.cli import main
from helicon.scaffold import AVAILABLE_PRESETS, scaffold_config

# ---------------------------------------------------------------------------
# scaffold_config
# ---------------------------------------------------------------------------


def test_scaffold_custom_returns_string():
    result = scaffold_config("custom")
    assert isinstance(result, str)
    assert len(result) > 10


def test_scaffold_sunbird_contains_nozzle():
    result = scaffold_config("sunbird")
    assert "nozzle" in result


def test_scaffold_highpower_contains_plasma():
    result = scaffold_config("highpower")
    assert "plasma" in result


def test_scaffold_all_presets_valid_yaml():
    for preset in AVAILABLE_PRESETS:
        text = scaffold_config(preset)
        parsed = yaml.safe_load(text)
        assert "nozzle" in parsed
        assert "plasma" in parsed


def test_scaffold_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        scaffold_config("does_not_exist")


def test_available_presets_nonempty():
    assert len(AVAILABLE_PRESETS) >= 3


# ---------------------------------------------------------------------------
# helicon init CLI
# ---------------------------------------------------------------------------


def test_init_cmd_creates_file(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "mysim.yaml")
    result = runner.invoke(main, ["init", "mysim", "--output", out])
    assert result.exit_code == 0, result.output
    assert Path(out).exists()


def test_init_cmd_default_filename(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init", "mytest"])
        assert result.exit_code == 0, result.output
        assert Path("mytest.yaml").exists()


def test_init_cmd_output_is_valid_yaml(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "cfg.yaml")
    runner.invoke(main, ["init", "test", "--preset", "sunbird", "--output", out])
    with open(out) as fh:
        data = yaml.safe_load(fh)
    assert "nozzle" in data


def test_init_cmd_preset_highpower(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "hp.yaml")
    result = runner.invoke(main, ["init", "test", "--preset", "highpower", "--output", out])
    assert result.exit_code == 0, result.output
    assert Path(out).exists()


def test_init_cmd_echoes_filename(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "echo.yaml")
    result = runner.invoke(main, ["init", "echo", "--output", out])
    assert "echo" in result.output.lower() or out in result.output


# ---------------------------------------------------------------------------
# helicon schema CLI
# ---------------------------------------------------------------------------


def test_schema_cmd_stdout():
    runner = CliRunner()
    result = runner.invoke(main, ["schema"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "properties" in data or "title" in data


def test_schema_cmd_to_file(tmp_path):
    runner = CliRunner()
    out = str(tmp_path / "schema.json")
    result = runner.invoke(main, ["schema", "--output", out])
    assert result.exit_code == 0, result.output
    assert Path(out).exists()
    with open(out) as fh:
        data = json.load(fh)
    assert isinstance(data, dict)


def test_schema_contains_nozzle_or_plasma():
    runner = CliRunner()
    result = runner.invoke(main, ["schema"])
    text = result.output.lower()
    assert "nozzle" in text or "plasma" in text or "properties" in text

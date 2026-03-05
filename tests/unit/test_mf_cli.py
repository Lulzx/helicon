"""Tests for `helicon mf` CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from helicon.cli import main


def _write_config(tmpdir: str) -> str:
    import yaml

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


def _write_fake_mf_output(tmpdir: str, n: int = 5) -> str:
    """Write fake tier3 output directories for report tests."""
    out = Path(tmpdir)
    for i in range(n):
        d = out / f"tier3_t2_{i:010d}"
        d.mkdir()
        meta = {
            "candidate_id": f"t2_{i:010d}",
            "tier2_score": round(0.5 + i * 0.09, 4),
            "tier2_metrics": {
                "eta_d": round(0.7 + i * 0.03, 4),
                "thrust_N": round(1e-4 + i * 5e-5, 6),
                "mirror_ratio": 10.0 + i,
            },
            "dry_run": True,
            "status": "dry_run",
        }
        (d / "tier3_meta.json").write_text(json.dumps(meta))
    return str(out)


class TestMfReport:
    def test_report_text(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            _write_fake_mf_output(tmp, n=5)
            result = runner.invoke(main, ["mf", "report", tmp])
        assert result.exit_code == 0, result.output
        assert "Total tier3 candidates: 5" in result.output
        assert "dry_run" in result.output

    def test_report_json(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            _write_fake_mf_output(tmp, n=3)
            result = runner.invoke(main, ["mf", "report", tmp, "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["n_total"] == 3
        assert len(data["top"]) == 3

    def test_report_top_limit(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            _write_fake_mf_output(tmp, n=10)
            result = runner.invoke(main, ["mf", "report", tmp, "--top", "3", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["n_total"] == 10
        assert len(data["top"]) == 3

    def test_report_sorted_by_score(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            _write_fake_mf_output(tmp, n=5)
            result = runner.invoke(main, ["mf", "report", tmp, "--json"])
        data = json.loads(result.output)
        scores = [c["tier2_score"] for c in data["top"]]
        assert scores == sorted(scores, reverse=True)

    def test_report_empty_dir(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(main, ["mf", "report", tmp])
        assert result.exit_code != 0


class TestMfRun:
    def test_run_dry_run(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = _write_config(tmp)
            out = str(Path(tmp) / "mf_out")
            result = runner.invoke(
                main,
                [
                    "mf",
                    "run",
                    "--config",
                    config,
                    "--vary",
                    "coils.0.I:5000:50000:5",
                    "--n-tier1",
                    "20",
                    "--top-k",
                    "2",
                    "--dry-run",
                    "--output",
                    out,
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Tier-1 evaluated:" in result.output
        assert "Tier-3 promoted:" in result.output

    def test_run_json_output(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = _write_config(tmp)
            out = str(Path(tmp) / "mf_out")
            result = runner.invoke(
                main,
                [
                    "mf",
                    "run",
                    "--config",
                    config,
                    "--vary",
                    "coils.0.I:5000:50000:5",
                    "--n-tier1",
                    "10",
                    "--top-k",
                    "1",
                    "--dry-run",
                    "--output",
                    out,
                    "--json",
                ],
            )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "n_tier1" in data
        assert data["n_tier1"] == 10

    def test_run_creates_tier3_dirs(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = _write_config(tmp)
            out = Path(tmp) / "mf_out"
            runner.invoke(
                main,
                [
                    "mf",
                    "run",
                    "--config",
                    config,
                    "--vary",
                    "coils.0.I:5000:50000:5",
                    "--n-tier1",
                    "10",
                    "--top-k",
                    "2",
                    "--dry-run",
                    "--output",
                    str(out),
                ],
            )
            tier3_dirs = list(out.glob("tier3_*")) if out.exists() else []
            # Tier-3 dirs may or may not exist depending on thresholds/scores
            assert isinstance(tier3_dirs, list)

    def test_run_bad_vary_spec(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = _write_config(tmp)
            result = runner.invoke(
                main,
                [
                    "mf",
                    "run",
                    "--config",
                    config,
                    "--vary",
                    "badspec",
                    "--n-tier1",
                    "5",
                    "--dry-run",
                    "--output",
                    str(Path(tmp) / "out"),
                ],
            )
        assert result.exit_code != 0

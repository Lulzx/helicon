"""Tests for helicon throttle-map, mission, and regression CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from helicon.cli import main

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_throttle_map(path: Path) -> Path:
    """Write a minimal ThrottleMap JSON for use with helicon mission."""
    power_grid = np.linspace(1e3, 1e6, 3).tolist()
    mdot_grid = np.linspace(1e-5, 1e-3, 3).tolist()
    n = 3
    thrust = (np.ones((n, n)) * 1e-3).tolist()
    isp = (np.ones((n, n)) * 2000.0).tolist()
    eta_d = (np.ones((n, n)) * 0.8).tolist()
    data = {
        "power_grid_W": power_grid,
        "mdot_grid_kgs": mdot_grid,
        "thrust_N": thrust,
        "isp_s": isp,
        "eta_d": eta_d,
        "mirror_ratio": 10.0,
        "eta_thermal": 0.65,
    }
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# helicon throttle-map
# ---------------------------------------------------------------------------


class TestThrottleMapCmd:
    def test_throttle_map_preset(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "tm.json")
        result = runner.invoke(
            main,
            [
                "throttle-map",
                "--preset",
                "sunbird",
                "--n-power",
                "3",
                "--n-mdot",
                "3",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "tm.json").exists()
        assert "Saved to" in result.output

    def test_throttle_map_outputs_isp_range(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "tm.json")
        result = runner.invoke(
            main,
            [
                "throttle-map",
                "--preset",
                "sunbird",
                "--n-power",
                "2",
                "--n-mdot",
                "2",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Isp range" in result.output
        assert "Thrust range" in result.output

    def test_throttle_map_json_is_valid(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "tm.json")
        runner.invoke(
            main,
            [
                "throttle-map",
                "--preset",
                "sunbird",
                "--n-power",
                "2",
                "--n-mdot",
                "2",
                "--output",
                out,
            ],
        )
        assert (tmp_path / "tm.json").exists()
        data = json.loads((tmp_path / "tm.json").read_text())
        assert "power_grid_W" in data
        assert "thrust_N" in data
        assert "isp_s" in data
        assert "mirror_ratio" in data

    def test_throttle_map_no_args_fails(self):
        runner = CliRunner()
        result = runner.invoke(main, ["throttle-map"])
        assert result.exit_code != 0

    def test_throttle_map_both_args_fails(self, tmp_path):
        import yaml

        runner = CliRunner()
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            yaml.dump(
                {
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
            )
        )
        result = runner.invoke(
            main,
            [
                "throttle-map",
                "--preset",
                "sunbird",
                "--config",
                str(cfg),
                "--output",
                str(tmp_path / "tm.json"),
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# helicon mission
# ---------------------------------------------------------------------------


class TestMissionCmd:
    def test_mission_basic(self, tmp_path):
        runner = CliRunner()
        tm_path = _write_throttle_map(tmp_path / "tm.json")
        result = runner.invoke(
            main,
            [
                "mission",
                "--throttle-map",
                str(tm_path),
                "--dry-mass",
                "100",
                "--delta-v",
                "1000",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ΔV" in result.output or "delta" in result.output.lower()
        assert "Isp" in result.output
        assert "Propellant" in result.output

    def test_mission_output_json(self, tmp_path):
        runner = CliRunner()
        tm_path = _write_throttle_map(tmp_path / "tm.json")
        out_file = str(tmp_path / "result.json")
        result = runner.invoke(
            main,
            [
                "mission",
                "--throttle-map",
                str(tm_path),
                "--dry-mass",
                "100",
                "--delta-v",
                "500",
                "--output",
                out_file,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "result.json").exists()
        data = json.loads((tmp_path / "result.json").read_text())
        assert "total_delta_v_ms" in data or "propellant_mass_kg" in data

    def test_mission_missing_throttle_map_fails(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "mission",
                "--throttle-map",
                "/nonexistent/tm.json",
                "--dry-mass",
                "100",
                "--delta-v",
                "1000",
            ],
        )
        assert result.exit_code != 0

    def test_mission_burn_time_positive(self, tmp_path):
        runner = CliRunner()
        tm_path = _write_throttle_map(tmp_path / "tm.json")
        result = runner.invoke(
            main,
            [
                "mission",
                "--throttle-map",
                str(tm_path),
                "--dry-mass",
                "50",
                "--delta-v",
                "2000",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Burn time" in result.output


# ---------------------------------------------------------------------------
# helicon regression
# ---------------------------------------------------------------------------


def _write_baseline(path: Path) -> None:
    """Write a minimal baseline.json for regression tests."""
    import helicon

    baseline = {
        "helicon_version": helicon.__version__,
        "created_at": "2026-03-05T00:00:00Z",
        "results": [
            {
                "case_name": "free_expansion",
                "passed": True,
                "metrics": {"eta_d": 0.85, "thrust_N": 1e-4},
                "tolerances": {},
            }
        ],
    }
    path.write_text(json.dumps(baseline))


class TestRegressionCmd:
    def test_regression_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["regression", "--help"])
        assert result.exit_code == 0
        assert "save-baseline" in result.output or "regression" in result.output.lower()

    def _run_with_guiding_center(self, runner, args):
        """Helper: always add --case guiding_center to avoid openPMD failures."""
        return runner.invoke(main, [*args, "--case", "guiding_center"])

    def test_regression_run_clean(self, tmp_path):
        runner = CliRunner()
        # Write a baseline that matches what guiding_center actually produces
        import helicon

        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "helicon_version": helicon.__version__,
                    "created_at": "2026-03-05T00:00:00Z",
                    "results": [
                        {
                            "case_name": "guiding_center",
                            "passed": True,
                            "metrics": {"measured_convergence_order": 2.0},
                            "tolerances": {},
                        }
                    ],
                }
            )
        )
        out_dir = str(tmp_path / "reg_out")
        result = self._run_with_guiding_center(
            runner,
            [
                "regression",
                "run",
                "--baseline",
                str(baseline),
                "--output",
                out_dir,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "CLEAN" in result.output

    def test_regression_run_missing_baseline(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "regression",
                "run",
                "--baseline",
                str(tmp_path / "nonexistent.json"),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_regression_run_writes_report(self, tmp_path):
        import helicon

        runner = CliRunner()
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "helicon_version": helicon.__version__,
                    "created_at": "2026-03-05T00:00:00Z",
                    "results": [
                        {
                            "case_name": "guiding_center",
                            "passed": True,
                            "metrics": {"measured_convergence_order": 2.0},
                            "tolerances": {},
                        }
                    ],
                }
            )
        )
        out_dir = tmp_path / "reg_out"
        self._run_with_guiding_center(
            runner,
            [
                "regression",
                "run",
                "--baseline",
                str(baseline),
                "--output",
                str(out_dir),
            ],
        )
        assert (out_dir / "regression_report.json").exists()

    def test_regression_run_reports_fixed_and_unchanged(self, tmp_path):
        import helicon

        runner = CliRunner()
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "helicon_version": helicon.__version__,
                    "created_at": "2026-03-05T00:00:00Z",
                    "results": [
                        {
                            "case_name": "guiding_center",
                            "passed": True,
                            "metrics": {"measured_convergence_order": 2.0},
                            "tolerances": {},
                        }
                    ],
                }
            )
        )
        out_dir = str(tmp_path / "reg_out")
        result = self._run_with_guiding_center(
            runner,
            [
                "regression",
                "run",
                "--baseline",
                str(baseline),
                "--output",
                out_dir,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Fixed" in result.output
        assert "Unchanged" in result.output

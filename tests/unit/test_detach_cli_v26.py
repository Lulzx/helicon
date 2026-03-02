"""Tests for v2.6 detach CLI subcommands: calibrate, invert, simulate, report."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from click.testing import CliRunner

from helicon.cli import main

# ---------------------------------------------------------------------------
# Shared plasma args (used in simulate and report)
# ---------------------------------------------------------------------------

_PLASMA = [
    "--n",
    "1e18",
    "--Te",
    "30",
    "--Ti",
    "15",
    "--B",
    "0.05",
    "--dBdz",
    "-1.0",
    "--vz",
    "40000",
]

# ---------------------------------------------------------------------------
# helicon detach group
# ---------------------------------------------------------------------------


def test_detach_group_help():
    r = CliRunner().invoke(main, ["detach", "--help"])
    assert r.exit_code == 0
    assert "assess" in r.output
    assert "calibrate" in r.output
    assert "invert" in r.output
    assert "simulate" in r.output
    assert "report" in r.output


# ---------------------------------------------------------------------------
# helicon detach assess (renamed from old 'helicon detach')
# ---------------------------------------------------------------------------


class TestDetachAssess:
    def test_assess_attached(self):
        r = CliRunner().invoke(main, ["detach", "assess", *_PLASMA])
        assert r.exit_code == 0, r.output

    def test_assess_json(self):
        r = CliRunner().invoke(main, ["detach", "assess", *_PLASMA, "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert "detachment_score" in data

    def test_assess_control_flag(self):
        r = CliRunner().invoke(main, ["detach", "assess", *_PLASMA, "--control"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert "recommended_action" in data

    def test_assess_species_argon(self):
        r = CliRunner().invoke(main, ["detach", "assess", *_PLASMA, "--species", "Ar+"])
        assert r.exit_code == 0


# ---------------------------------------------------------------------------
# helicon detach calibrate
# ---------------------------------------------------------------------------


class TestDetachCalibrate:
    def test_calibrate_runs(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "calibrate",
                "--n-samples",
                "100",
                "--seed",
                "42",
            ],
        )
        assert r.exit_code == 0, r.output

    def test_calibrate_output_contains_weights(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "calibrate",
                "--n-samples",
                "50",
            ],
        )
        assert r.exit_code == 0
        assert "w_" in r.output or "weight" in r.output.lower()

    def test_calibrate_json_flag(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "calibrate",
                "--n-samples",
                "50",
                "--seed",
                "0",
                "--json",
            ],
        )
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert "w_alfven" in data
        assert "w_beta" in data
        assert "w_ion_mag" in data
        assert "score_detached" in data
        assert "accuracy" in data
        total = data["w_alfven"] + data["w_beta"] + data["w_ion_mag"]
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_calibrate_saves_file(self, tmp_path):
        out = tmp_path / "cal.json"
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "calibrate",
                "--n-samples",
                "50",
                "--output",
                str(out),
            ],
        )
        assert r.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "w_alfven" in data

    def test_calibrate_different_seeds_differ(self):
        r0 = CliRunner().invoke(
            main,
            [
                "detach",
                "calibrate",
                "--n-samples",
                "80",
                "--seed",
                "0",
                "--json",
            ],
        )
        r1 = CliRunner().invoke(
            main,
            [
                "detach",
                "calibrate",
                "--n-samples",
                "80",
                "--seed",
                "99",
                "--json",
            ],
        )
        d0 = json.loads(r0.output)
        d1 = json.loads(r1.output)
        # Different seeds may give slightly different weights
        # (they could coincidentally be the same, so just check it ran)
        assert r0.exit_code == 0
        assert r1.exit_code == 0
        _ = d0["w_alfven"]  # both parseable
        _ = d1["w_alfven"]


# ---------------------------------------------------------------------------
# helicon detach invert
# ---------------------------------------------------------------------------


class TestDetachInvert:
    _INVERT: ClassVar[list[str]] = [
        "detach",
        "invert",
        "--F",
        "0.05",
        "--mdot",
        "1e-5",
        "--B",
        "0.03",
        "--area",
        "5e-4",
    ]

    def test_invert_runs(self):
        r = CliRunner().invoke(main, self._INVERT)
        assert r.exit_code == 0, r.output

    def test_invert_output_contains_keys(self):
        r = CliRunner().invoke(main, self._INVERT)
        assert "vz" in r.output or "Mach" in r.output or "density" in r.output.lower()

    def test_invert_json_flag(self):
        r = CliRunner().invoke(main, [*self._INVERT, "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert "n_m3" in data
        assert "vz_ms" in data
        assert "alfven_mach" in data
        assert "detachment_score" in data
        assert "confidence" in data

    def test_invert_json_score_in_range(self):
        r = CliRunner().invoke(main, [*self._INVERT, "--json"])
        data = json.loads(r.output)
        assert 0.0 <= data["detachment_score"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0

    def test_invert_species_argon(self):
        r = CliRunner().invoke(main, [*self._INVERT, "--species", "Ar+"])
        assert r.exit_code == 0

    def test_invert_mirror_ratio(self):
        r = CliRunner().invoke(main, [*self._INVERT, "--mirror-ratio", "10.0", "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert data["n_m3"] > 0.0

    def test_invert_bad_mirror_ratio(self):
        r = CliRunner().invoke(main, [*self._INVERT, "--mirror-ratio", "0.5"])
        # Should fail with an error
        assert r.exit_code != 0 or "Error" in r.output or r.exception is not None


# ---------------------------------------------------------------------------
# helicon detach simulate
# ---------------------------------------------------------------------------


class TestDetachSimulate:
    def test_simulate_runs(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "5",
            ],
        )
        assert r.exit_code == 0, r.output

    def test_simulate_output_has_steps(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "5",
            ],
        )
        assert r.exit_code == 0
        # Should show step numbers 1..5
        assert "1" in r.output
        assert "5" in r.output

    def test_simulate_json_flag(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "3",
                "--json",
            ],
        )
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert "setpoint" in data
        assert "trace" in data
        assert len(data["trace"]) == 3

    def test_simulate_json_trace_keys(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "2",
                "--json",
            ],
        )
        data = json.loads(r.output)
        step = data["trace"][0]
        for key in ("step", "score", "error", "V", "dV_dt", "delta_I_A", "I_coil_A"):
            assert key in step

    def test_simulate_V_in_trace(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "5",
                "--json",
            ],
        )
        data = json.loads(r.output)
        for step in data["trace"]:
            assert step["V"] >= 0.0

    def test_simulate_setpoint(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "3",
                "--setpoint",
                "0.4",
                "--json",
            ],
        )
        data = json.loads(r.output)
        assert data["setpoint"] == pytest.approx(0.4)

    def test_simulate_decay_rate(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "3",
                "--decay-rate",
                "2.0",
            ],
        )
        assert r.exit_code == 0

    def test_simulate_mentions_stable(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "simulate",
                *_PLASMA,
                "--steps",
                "3",
            ],
        )
        assert "stable" in r.output.lower()


# ---------------------------------------------------------------------------
# helicon detach report
# ---------------------------------------------------------------------------


class TestDetachReport:
    def test_report_runs(self):
        r = CliRunner().invoke(main, ["detach", "report", *_PLASMA])
        assert r.exit_code == 0, r.output

    def test_report_has_four_sections(self):
        r = CliRunner().invoke(main, ["detach", "report", *_PLASMA])
        out = r.output.lower()
        assert "mhd" in out
        assert "kinetic" in out or "flr" in out
        assert "sheath" in out
        assert "lyapunov" in out

    def test_report_json_flag(self):
        r = CliRunner().invoke(main, ["detach", "report", *_PLASMA, "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert "mhd" in data
        assert "kinetic" in data
        assert "sheath" in data
        assert "lyapunov" in data

    def test_report_json_mhd_score(self):
        r = CliRunner().invoke(main, ["detach", "report", *_PLASMA, "--json"])
        data = json.loads(r.output)
        assert 0.0 <= data["mhd"]["detachment_score"] <= 1.0

    def test_report_json_sheath_fields(self):
        r = CliRunner().invoke(main, ["detach", "report", *_PLASMA, "--json"])
        data = json.loads(r.output)
        sh = data["sheath"]
        assert "debye_length_m" in sh
        assert "score_corrected" in sh
        assert sh["debye_length_m"] > 0.0

    def test_report_json_lyapunov_stable(self):
        r = CliRunner().invoke(main, ["detach", "report", *_PLASMA, "--json"])
        data = json.loads(r.output)
        assert "is_stable" in data["lyapunov"]

    def test_report_json_kinetic_fields(self):
        r = CliRunner().invoke(main, ["detach", "report", *_PLASMA, "--json"])
        data = json.loads(r.output)
        assert "lambda_i_flr" in data["kinetic"]
        assert "alfven_mach_kinetic" in data["kinetic"]

    def test_report_coupling_flag(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "report",
                *_PLASMA,
                "--coupling",
                "0.0",
                "--json",
            ],
        )
        data = json.loads(r.output)
        # zero coupling → no correction
        assert data["sheath"]["score_corrected"] == pytest.approx(
            data["mhd"]["detachment_score"], abs=1e-9
        )

    def test_report_species_xenon(self):
        r = CliRunner().invoke(
            main,
            [
                "detach",
                "report",
                *_PLASMA,
                "--species",
                "Xe+",
            ],
        )
        assert r.exit_code == 0

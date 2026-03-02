"""Tests for helicon.runner.metal_runner — warpx-metal integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from helicon.runner.metal_runner import (
    _BUILD_SUBPATH,
    _EXE_2D,
    MetalRunResult,
    WarpXMetalDiag,
    WarpXMetalInfo,
    detect_warpx_metal,
    find_diag_dirs,
    find_warpx_metal_root,
    generate_metal_inputs,
    run_warpx_metal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_header(tmp_dir: Path, *, step: int = 4, n_comp: int = 3) -> Path:
    """Write a minimal AMReX-style Header file into tmp_dir/diag<step>/."""
    diag_dir = tmp_dir / f"diag{step}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    field_names = ["Ex", "Ey", "Bz"][:n_comp]
    lines = [
        "HyperCLaw-V1.1",
        str(n_comp),
        *field_names,
        "1",          # n_levels
        "1.234e-12",  # time
        "0",          # finest_level
        "-1.0e-05 -1.0e-05",  # prob_lo
        " 1.0e-05  1.0e-05",  # prob_hi
        "",
        "((0,0) (127,127) (0,0))",
    ]
    (diag_dir / "Header").write_text("\n".join(lines) + "\n")
    return diag_dir


# ---------------------------------------------------------------------------
# WarpXMetalInfo
# ---------------------------------------------------------------------------


class TestWarpXMetalInfo:
    def test_invalid_info_summary(self):
        info = WarpXMetalInfo(
            root=Path("."), exe_2d=None, exe_3d=None, acpp_bin=None, valid=False
        )
        s = info.summary()
        assert "not found" in s
        assert "valid:  False" in s

    def test_valid_info_summary(self, tmp_path):
        fake_exe = tmp_path / "warpx"
        fake_exe.touch()
        info = WarpXMetalInfo(
            root=tmp_path, exe_2d=fake_exe, exe_3d=None, acpp_bin=None, valid=True
        )
        s = info.summary()
        assert str(fake_exe) in s
        assert "valid:  True" in s


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestFindWarpXMetalRoot:
    def test_nonexistent_hint_returns_none(self, tmp_path):
        # A directory that definitely has no Metal exe
        result = find_warpx_metal_root(hint=tmp_path / "does_not_exist")
        # May return None or a real path from other search locations; just verify no crash
        assert result is None or isinstance(result, Path)

    def test_env_var_used(self, tmp_path, monkeypatch):
        # Create a fake executable
        bin_dir = tmp_path / _BUILD_SUBPATH
        bin_dir.mkdir(parents=True)
        exe = bin_dir / _EXE_2D
        exe.touch()
        exe.chmod(0o755)
        monkeypatch.setenv("WARPX_METAL_ROOT", str(tmp_path))
        result = find_warpx_metal_root()
        assert result == tmp_path.resolve()

    def test_env_var_cleared(self, monkeypatch):
        monkeypatch.delenv("WARPX_METAL_ROOT", raising=False)
        # Should not raise regardless of result
        result = find_warpx_metal_root()
        assert result is None or isinstance(result, Path)

    def test_hint_directory_used(self, tmp_path):
        bin_dir = tmp_path / _BUILD_SUBPATH
        bin_dir.mkdir(parents=True)
        exe = bin_dir / _EXE_2D
        exe.touch()
        exe.chmod(0o755)
        result = find_warpx_metal_root(hint=tmp_path)
        assert result == tmp_path.resolve()


class TestDetectWarpXMetal:
    def test_no_exe_returns_invalid(self, monkeypatch):
        # Patch find_warpx_metal_root to return None — simulates no build found
        monkeypatch.setattr(
            "helicon.runner.metal_runner.find_warpx_metal_root",
            lambda *a, **kw: None,
        )
        result = detect_warpx_metal()
        assert result.valid is False
        assert result.exe_2d is None
        assert result.exe_3d is None

    def test_with_fake_exe_returns_valid(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WARPX_METAL_ROOT", raising=False)
        bin_dir = tmp_path / _BUILD_SUBPATH
        bin_dir.mkdir(parents=True)
        exe = bin_dir / _EXE_2D
        exe.touch()
        exe.chmod(0o755)
        result = detect_warpx_metal(hint=tmp_path)
        assert result.valid is True
        assert result.exe_2d is not None
        assert result.exe_2d.name == _EXE_2D

    def test_acpp_bin_detected(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WARPX_METAL_ROOT", raising=False)
        bin_dir = tmp_path / _BUILD_SUBPATH
        bin_dir.mkdir(parents=True)
        exe = bin_dir / _EXE_2D
        exe.touch()
        exe.chmod(0o755)
        acpp_dir = tmp_path / "opt" / "adaptivecpp" / "bin"
        acpp_dir.mkdir(parents=True)
        (acpp_dir / "acpp").touch()
        result = detect_warpx_metal(hint=tmp_path)
        assert result.acpp_bin is not None
        assert result.acpp_bin.name == "acpp"


# ---------------------------------------------------------------------------
# WarpXMetalDiag
# ---------------------------------------------------------------------------


class TestWarpXMetalDiag:
    def test_from_dir_parses_header(self, tmp_path):
        diag_dir = _make_fake_header(tmp_path, step=4, n_comp=3)
        diag = WarpXMetalDiag.from_dir(diag_dir)
        assert diag.step == 4
        assert diag.time_s == pytest.approx(1.234e-12)
        assert diag.field_vars == ["Ex", "Ey", "Bz"]
        assert diag.n_cells == (128, 128)
        assert diag.domain_lo[0] == pytest.approx(-1e-5)
        assert diag.domain_hi[0] == pytest.approx(1e-5)

    def test_from_dir_missing_header_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            WarpXMetalDiag.from_dir(tmp_path / "no_header")

    def test_from_dir_step_parsed_from_name(self, tmp_path):
        diag_dir = _make_fake_header(tmp_path, step=1000000)
        diag = WarpXMetalDiag.from_dir(diag_dir)
        assert diag.step == 1000000

    def test_summary_no_crash(self, tmp_path):
        diag_dir = _make_fake_header(tmp_path, step=4)
        diag = WarpXMetalDiag.from_dir(diag_dir)
        s = diag.summary()
        assert "step=4" in s
        assert "Ex" in s

    def test_species_empty_when_no_warpx_header(self, tmp_path):
        diag_dir = _make_fake_header(tmp_path, step=2)
        diag = WarpXMetalDiag.from_dir(diag_dir)
        assert diag.species == []

    def test_species_parsed_from_warpx_header(self, tmp_path):
        diag_dir = _make_fake_header(tmp_path, step=2)
        warpx_hdr = diag_dir / "WarpXHeader"
        warpx_hdr.write_text("electrons 1\npositrons 2\n")
        diag = WarpXMetalDiag.from_dir(diag_dir)
        assert "electrons" in diag.species
        assert "positrons" in diag.species


# ---------------------------------------------------------------------------
# find_diag_dirs
# ---------------------------------------------------------------------------


class TestFindDiagDirs:
    def test_empty_directory(self, tmp_path):
        result = find_diag_dirs(tmp_path)
        assert result == []

    def test_finds_diag_dirs(self, tmp_path):
        _make_fake_header(tmp_path, step=4)
        _make_fake_header(tmp_path, step=8)
        result = find_diag_dirs(tmp_path)
        assert len(result) == 2
        assert result[0].step == 4
        assert result[1].step == 8

    def test_sorted_by_step(self, tmp_path):
        _make_fake_header(tmp_path, step=8)
        _make_fake_header(tmp_path, step=4)
        result = find_diag_dirs(tmp_path)
        steps = [d.step for d in result]
        assert steps == sorted(steps)

    def test_skips_dirs_without_header(self, tmp_path):
        # A directory named diag* but without a Header
        (tmp_path / "diag_no_header").mkdir()
        _make_fake_header(tmp_path, step=4)
        result = find_diag_dirs(tmp_path)
        assert len(result) == 1

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = find_diag_dirs(tmp_path / "nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# generate_metal_inputs
# ---------------------------------------------------------------------------


class TestGenerateMetalInputs:
    def test_returns_string(self):
        content = generate_metal_inputs()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_contains_required_sections(self):
        content = generate_metal_inputs()
        assert "max_step" in content
        assert "amr.n_cell" in content
        assert "geometry.dims = 2" in content
        assert "particles.species_names" in content
        assert "electrons" in content
        assert "positrons" in content
        assert "diagnostics.diags_names" in content

    def test_n_cell_param_respected(self):
        content = generate_metal_inputs(n_cell=64)
        assert "64 64" in content

    def test_max_step_param_respected(self):
        content = generate_metal_inputs(max_step=10)
        assert "max_step = 10" in content

    def test_extra_params_included(self):
        content = generate_metal_inputs(extra={"amrex.verbose": "2"})
        assert "amrex.verbose = 2" in content

    def test_ends_with_newline(self):
        content = generate_metal_inputs()
        assert content.endswith("\n")

    def test_periodic_boundaries(self):
        content = generate_metal_inputs()
        assert "is_periodic = 1 1" in content

    def test_single_precision_config(self):
        # CFL and sort settings match warpx-metal validated config
        content = generate_metal_inputs(cfl=1.0)
        assert "warpx.cfl = 1.0" in content


# ---------------------------------------------------------------------------
# run_warpx_metal — invalid metal_info path (no exe, no actual simulation)
# ---------------------------------------------------------------------------


class TestRunWarpXMetalNoExe:
    def test_returns_failure_when_no_exe(self, tmp_path):
        invalid_info = WarpXMetalInfo(
            root=tmp_path, exe_2d=None, exe_3d=None, acpp_bin=None, valid=False
        )
        result = run_warpx_metal(metal_info=invalid_info, output_dir=tmp_path / "out")
        assert result.success is False
        assert result.exit_code == -1
        assert result.error is not None
        assert "not found" in result.error.lower() or "warpx-metal" in result.error.lower()

    def test_result_dataclass_fields(self, tmp_path):
        invalid_info = WarpXMetalInfo(
            root=tmp_path, exe_2d=None, exe_3d=None, acpp_bin=None, valid=False
        )
        result = run_warpx_metal(metal_info=invalid_info, output_dir=tmp_path / "out")
        assert isinstance(result, MetalRunResult)
        assert isinstance(result.output_dir, Path)
        assert isinstance(result.log_path, Path)
        assert result.wall_time_s == pytest.approx(0.0)
        assert result.steps_completed == 0
        assert result.diags == []


# ---------------------------------------------------------------------------
# MetalRunResult
# ---------------------------------------------------------------------------


class TestMetalRunResult:
    def test_summary_ok(self, tmp_path):
        r = MetalRunResult(
            success=True,
            exit_code=0,
            output_dir=tmp_path,
            log_path=tmp_path / "warpx_metal.log",
            wall_time_s=5.3,
            steps_completed=4,
            diags=[],
        )
        s = r.summary()
        assert "OK" in s
        assert "5.30" in s
        assert "steps" in s.lower()

    def test_summary_failed(self, tmp_path):
        r = MetalRunResult(
            success=False,
            exit_code=1,
            output_dir=tmp_path,
            log_path=tmp_path / "warpx_metal.log",
            wall_time_s=0.1,
            steps_completed=0,
            diags=[],
            error="segfault",
        )
        s = r.summary()
        assert "FAILED" in s
        assert "segfault" in s


# ---------------------------------------------------------------------------
# Integration: doctor detects Metal build
# ---------------------------------------------------------------------------


def test_doctor_report_has_metal_fields():
    from helicon.doctor import check_environment

    report = check_environment()
    # Fields must exist (regardless of whether Metal build is present)
    assert hasattr(report, "warpx_metal_found")
    assert hasattr(report, "warpx_metal_path")
    assert isinstance(report.warpx_metal_found, bool)


def test_doctor_to_dict_has_metal_keys():
    from helicon.doctor import check_environment

    report = check_environment()
    d = report.to_dict()
    assert "warpx_metal_found" in d
    assert "warpx_metal_path" in d


def test_doctor_summary_has_metal_section():
    from helicon.doctor import check_environment

    report = check_environment()
    s = report.summary()
    assert "Metal" in s or "metal" in s.lower()

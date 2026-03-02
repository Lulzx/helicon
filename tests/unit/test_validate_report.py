"""Tests for helicon.validate.report module."""

from __future__ import annotations

from pathlib import Path

import pytest

from helicon.validate.report import generate_html_report

try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

needs_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB, reason="matplotlib not installed"
)


def _make_result(case_name: str = "test_case", passed: bool = True) -> dict:
    return {
        "case_name": case_name,
        "passed": passed,
        "metrics": {"thrust_N": 1.5, "isp_s": 3000.0},
        "tolerances": {"thrust_N": 0.1, "isp_s": 50.0},
        "description": f"Test case {case_name}",
    }


@needs_matplotlib
class TestPlotValidationComparison:
    def test_returns_path(self, tmp_path: Path) -> None:
        from helicon.validate.report import plot_validation_comparison

        result = _make_result()
        output = tmp_path / "plot.png"
        p = plot_validation_comparison(result, output)
        assert isinstance(p, Path)
        assert p.exists()

    def test_empty_metrics(self, tmp_path: Path) -> None:
        from helicon.validate.report import plot_validation_comparison

        result = {
            "case_name": "empty",
            "passed": True,
            "metrics": {},
            "tolerances": {},
            "description": "Empty metrics",
        }
        output = tmp_path / "empty.png"
        p = plot_validation_comparison(result, output)
        assert p.exists()


@needs_matplotlib
class TestSaveValidationPlots:
    def test_returns_list(self, tmp_path: Path) -> None:
        from helicon.validate.report import save_validation_plots

        results = [_make_result("case_a"), _make_result("case_b", passed=False)]
        paths = save_validation_plots(results, tmp_path / "plots")
        assert isinstance(paths, list)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        from helicon.validate.report import save_validation_plots

        results = [_make_result()]
        out_dir = tmp_path / "deep" / "nested"
        paths = save_validation_plots(results, out_dir)
        assert out_dir.exists()
        assert len(paths) == 1


class TestGenerateHtmlReport:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        results = [_make_result("case_1"), _make_result("case_2", passed=False)]
        p = generate_html_report(results, tmp_path)
        assert p.exists()
        assert p.name == "validation_report.html"

    def test_html_contains_pass_fail(self, tmp_path: Path) -> None:
        results = [_make_result("case_1"), _make_result("case_2", passed=False)]
        p = generate_html_report(results, tmp_path)
        html = p.read_text()
        assert "PASS" in html
        assert "FAIL" in html

    def test_html_contains_case_name(self, tmp_path: Path) -> None:
        results = [_make_result("my_special_case")]
        p = generate_html_report(results, tmp_path)
        html = p.read_text()
        assert "my_special_case" in html

    def test_html_contains_metrics(self, tmp_path: Path) -> None:
        results = [_make_result()]
        p = generate_html_report(results, tmp_path)
        html = p.read_text()
        assert "thrust_N" in html

"""Tests for helicon.validate.regression — annual validation report automation."""

from __future__ import annotations

import json

import pytest

from helicon.validate.regression import (
    CaseDiff,
    MetricDiff,
    RegressionReport,
    RegressionSuite,
    compare_to_baseline,
    generate_markdown_report,
    load_baseline,
    save_baseline,
    save_regression_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _r(case, passed, metrics):
    return {"case_name": case, "passed": passed, "metrics": metrics, "tolerances": {}}


BASELINE_RESULTS = [
    _r("free_expansion", True, {"eta_d": 0.85}),
    _r("guiding_center", True, {"drift_error": 0.01}),
    _r("merino_ahedo", False, {"eta_d": 0.60}),
]

CURRENT_RESULTS_NO_CHANGE = [
    _r("free_expansion", True, {"eta_d": 0.85}),
    _r("guiding_center", True, {"drift_error": 0.01}),
    _r("merino_ahedo", False, {"eta_d": 0.60}),
]

CURRENT_RESULTS_WITH_REGRESSION = [
    _r("free_expansion", False, {"eta_d": 0.60}),  # REGRESSION
    _r("guiding_center", True, {"drift_error": 0.01}),
    _r("merino_ahedo", True, {"eta_d": 0.72}),  # FIXED
]


# ---------------------------------------------------------------------------
# Baseline save / load
# ---------------------------------------------------------------------------


def test_save_and_load_baseline(tmp_path):
    p = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, p)
    data = load_baseline(p)
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["helicon_version"] is not None


def test_save_baseline_custom_version(tmp_path):
    p = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, p, helicon_version="1.2.3")
    data = load_baseline(p)
    assert data["helicon_version"] == "1.2.3"


def test_load_baseline_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Baseline not found"):
        load_baseline(tmp_path / "nonexistent.json")


def test_save_baseline_creates_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "baseline.json"
    save_baseline(BASELINE_RESULTS, p)
    assert p.exists()


# ---------------------------------------------------------------------------
# compare_to_baseline
# ---------------------------------------------------------------------------


def test_no_regressions(tmp_path):
    p = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, p)
    report = compare_to_baseline(CURRENT_RESULTS_NO_CHANGE, p)
    assert report.n_regressions == 0
    assert report.all_passed


def test_regression_detected(tmp_path):
    p = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, p)
    report = compare_to_baseline(CURRENT_RESULTS_WITH_REGRESSION, p)
    assert report.n_regressions == 1
    assert not report.all_passed


def test_fixed_detected(tmp_path):
    p = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, p)
    report = compare_to_baseline(CURRENT_RESULTS_WITH_REGRESSION, p)
    assert report.n_fixed == 1


def test_case_diff_regressed():
    case = CaseDiff(
        case_name="test",
        baseline_passed=True,
        current_passed=False,
        status_changed=True,
    )
    assert case.regressed
    assert not case.fixed


def test_case_diff_fixed():
    case = CaseDiff(
        case_name="test",
        baseline_passed=False,
        current_passed=True,
        status_changed=True,
    )
    assert case.fixed
    assert not case.regressed


def test_metric_diffs_populated(tmp_path):
    p = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, p)
    report = compare_to_baseline(CURRENT_RESULTS_WITH_REGRESSION, p)
    free_exp = next(c for c in report.cases if c.case_name == "free_expansion")
    # eta_d changed from 0.85 to 0.60
    eta_d_diff = next(d for d in free_exp.metric_diffs if d.metric_name == "eta_d")
    assert eta_d_diff.changed
    assert eta_d_diff.relative_change == pytest.approx((0.60 - 0.85) / 0.85, rel=1e-4)


def test_metric_diff_numeric_tolerance():
    """Very small numeric changes should not count as 'changed'."""
    base_results = [_r("case_a", True, {"val": 1.0})]
    curr_results = [_r("case_a", True, {"val": 1.0 + 1e-10})]
    baseline = {"results": base_results, "helicon_version": "x", "created_at": "now"}
    report = compare_to_baseline(curr_results, baseline, metric_tolerance=1e-6)
    case = report.cases[0]
    assert all(not d.changed for d in case.metric_diffs)


def test_new_case_not_in_baseline(tmp_path):
    """Case present in current but not baseline is still handled gracefully."""
    p = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, p)
    extended = [*CURRENT_RESULTS_NO_CHANGE, _r("new_case", True, {"x": 1.0})]
    report = compare_to_baseline(extended, p)
    # New case has no baseline → baseline_passed=False, current_passed=True → fixed
    assert report.n_fixed >= 1


def test_report_dict_baseline(tmp_path):
    """compare_to_baseline accepts a pre-loaded dict."""
    baseline = {"results": BASELINE_RESULTS, "helicon_version": "test", "created_at": "now"}
    report = compare_to_baseline(CURRENT_RESULTS_NO_CHANGE, baseline)
    assert report.baseline_path == "<in-memory>"


# ---------------------------------------------------------------------------
# generate_markdown_report
# ---------------------------------------------------------------------------


def test_markdown_no_regressions():
    report = RegressionReport(
        baseline_path="/tmp/b.json",
        run_timestamp="2026-01-01T00:00:00Z",
        helicon_version="2.1.0",
        cases=[
            CaseDiff("case_a", True, True, False),
            CaseDiff("case_b", True, True, False),
        ],
        n_regressions=0,
        n_fixed=0,
        n_unchanged=2,
    )
    md = generate_markdown_report(report)
    assert "No regressions detected" in md
    assert "2.1.0" in md


def test_markdown_with_regressions():
    report = RegressionReport(
        baseline_path="/tmp/b.json",
        run_timestamp="2026-01-01T00:00:00Z",
        helicon_version="2.1.0",
        cases=[
            CaseDiff("regressed_case", True, False, True),
        ],
        n_regressions=1,
        n_fixed=0,
        n_unchanged=0,
    )
    md = generate_markdown_report(report)
    assert "regression" in md.lower()
    assert "regressed_case" in md


def test_markdown_metric_changes():
    report = RegressionReport(
        baseline_path="/tmp/b.json",
        run_timestamp="2026-01-01T00:00:00Z",
        helicon_version="2.1.0",
        cases=[
            CaseDiff(
                "case_a",
                True,
                True,
                False,
                metric_diffs=[
                    MetricDiff("eta_d", 0.72, 0.68, changed=True, relative_change=-0.056)
                ],
            ),
        ],
        n_regressions=0,
        n_fixed=0,
        n_unchanged=1,
    )
    md = generate_markdown_report(report)
    assert "eta_d" in md
    assert "0.72" in md


# ---------------------------------------------------------------------------
# save_regression_report
# ---------------------------------------------------------------------------


def test_save_regression_report(tmp_path):
    report = RegressionReport(
        baseline_path="/tmp/b.json",
        run_timestamp="2026-01-01",
        helicon_version="2.1.0",
        cases=[CaseDiff("c1", True, True, False)],
        n_regressions=0,
        n_fixed=0,
        n_unchanged=1,
    )
    json_path, md_path = save_regression_report(report, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    data = json.loads(json_path.read_text())
    assert data["n_regressions"] == 0


# ---------------------------------------------------------------------------
# RegressionSuite
# ---------------------------------------------------------------------------


def test_regression_suite_no_regressions(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, baseline_path)
    suite = RegressionSuite(baseline_path, output_dir=tmp_path / "out")
    report = suite.run(current_results=CURRENT_RESULTS_NO_CHANGE)
    assert report.n_regressions == 0
    assert (tmp_path / "out" / "regression_report.json").exists()


def test_regression_suite_update_baseline(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    save_baseline(BASELINE_RESULTS, baseline_path)
    suite = RegressionSuite(baseline_path, output_dir=tmp_path / "out")

    new_results = [_r("free_expansion", True, {"eta_d": 0.90})]
    suite.update_baseline(new_results, helicon_version="2.1.0")
    data = load_baseline(baseline_path)
    assert data["results"][0]["metrics"]["eta_d"] == 0.90
    assert data["helicon_version"] == "2.1.0"

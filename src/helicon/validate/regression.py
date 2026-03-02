"""Automated regression and annual validation report generation.

Compares current validation results against a stored baseline to detect
regressions.  Produces both a JSON report and a Markdown summary.

Usage::

    from helicon.validate.regression import (
        RegressionSuite,
        save_baseline,
        load_baseline,
        compare_to_baseline,
        generate_markdown_report,
    )

    # Record a new baseline after a known-good run
    from helicon.validate.runner import run_validation
    report = run_validation(run_simulations=False)
    save_baseline(report.results, "results/baseline.json")

    # Later, compare a new run against it
    new_report = run_validation(run_simulations=False)
    comparison = compare_to_baseline(new_report.results, "results/baseline.json")
    md = generate_markdown_report(comparison)
    print(md)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricDiff:
    """Difference in a single metric between baseline and current."""

    metric_name: str
    baseline_value: Any
    current_value: Any
    changed: bool
    relative_change: float | None  # None if not numeric


@dataclass
class CaseDiff:
    """Regression diff for a single validation case."""

    case_name: str
    baseline_passed: bool
    current_passed: bool
    status_changed: bool  # pass→fail or fail→pass
    metric_diffs: list[MetricDiff] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        """True if the case went from PASS to FAIL."""
        return self.baseline_passed and not self.current_passed

    @property
    def fixed(self) -> bool:
        """True if the case went from FAIL to PASS."""
        return not self.baseline_passed and self.current_passed


@dataclass
class RegressionReport:
    """Complete regression comparison report."""

    baseline_path: str
    run_timestamp: str
    helicon_version: str
    cases: list[CaseDiff]
    n_regressions: int
    n_fixed: int
    n_unchanged: int

    @property
    def all_passed(self) -> bool:
        return self.n_regressions == 0


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------


def save_baseline(
    results: list[dict],
    path: str | Path,
    *,
    helicon_version: str | None = None,
) -> None:
    """Save current validation results as a baseline for future comparisons.

    Parameters
    ----------
    results : list[dict]
        Validation results (each dict has keys: case_name, passed, metrics, …).
    path : str | Path
        Destination JSON file.
    helicon_version : str, optional
        Version string to embed in the baseline.
    """
    if helicon_version is None:
        try:
            from helicon import __version__
            helicon_version = __version__
        except Exception:
            helicon_version = "unknown"

    baseline = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "helicon_version": helicon_version,
        "results": results,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")


def load_baseline(path: str | Path) -> dict:
    """Load a previously saved baseline.

    Parameters
    ----------
    path : str | Path
        Path to the baseline JSON file.

    Returns
    -------
    dict
        Baseline data with keys ``created_at``, ``helicon_version``, ``results``.

    Raises
    ------
    FileNotFoundError
        If the baseline file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _metric_diff(name: str, baseline_val: Any, current_val: Any) -> MetricDiff:
    """Compare two metric values."""
    changed = baseline_val != current_val
    rel_change = None
    if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
        if baseline_val != 0:
            rel_change = (current_val - baseline_val) / abs(baseline_val)
        else:
            rel_change = float("inf") if current_val != 0 else 0.0
    return MetricDiff(
        metric_name=name,
        baseline_value=baseline_val,
        current_value=current_val,
        changed=changed,
        relative_change=rel_change,
    )


def compare_to_baseline(
    current_results: list[dict],
    baseline: str | Path | dict,
    *,
    metric_tolerance: float = 1e-6,
) -> RegressionReport:
    """Compare current validation results against a stored baseline.

    Parameters
    ----------
    current_results : list[dict]
        Current validation results.
    baseline : str | Path | dict
        Either a path to a baseline file or a pre-loaded baseline dict.
    metric_tolerance : float
        Relative tolerance for numeric metric comparison (below this → not changed).

    Returns
    -------
    RegressionReport
    """
    if isinstance(baseline, (str, Path)):
        baseline_data = load_baseline(baseline)
        baseline_path = str(baseline)
    else:
        baseline_data = baseline
        baseline_path = "<in-memory>"

    baseline_results = {r["case_name"]: r for r in baseline_data.get("results", [])}

    try:
        from helicon import __version__
        helicon_version = __version__
    except Exception:
        helicon_version = "unknown"

    cases = []
    n_regressions = 0
    n_fixed = 0
    n_unchanged = 0

    for current in current_results:
        name = current["case_name"]
        baseline_case = baseline_results.get(name, {})

        baseline_passed = baseline_case.get("passed", False)
        current_passed = current.get("passed", False)
        status_changed = baseline_passed != current_passed

        # Metric diffs
        baseline_metrics = baseline_case.get("metrics", {})
        current_metrics = current.get("metrics", {})
        all_keys = set(baseline_metrics) | set(current_metrics)
        metric_diffs = []
        for k in sorted(all_keys):
            bv = baseline_metrics.get(k)
            cv = current_metrics.get(k)
            diff = _metric_diff(k, bv, cv)
            # Apply numeric tolerance
            within_tol = (
                diff.relative_change is not None
                and abs(diff.relative_change) <= metric_tolerance
            )
            if within_tol:
                diff.changed = False
            metric_diffs.append(diff)

        case_diff = CaseDiff(
            case_name=name,
            baseline_passed=baseline_passed,
            current_passed=current_passed,
            status_changed=status_changed,
            metric_diffs=metric_diffs,
        )
        cases.append(case_diff)

        if case_diff.regressed:
            n_regressions += 1
        elif case_diff.fixed:
            n_fixed += 1
        else:
            n_unchanged += 1

    return RegressionReport(
        baseline_path=baseline_path,
        run_timestamp=datetime.now(tz=UTC).isoformat(),
        helicon_version=helicon_version,
        cases=cases,
        n_regressions=n_regressions,
        n_fixed=n_fixed,
        n_unchanged=n_unchanged,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def generate_markdown_report(report: RegressionReport) -> str:
    """Generate a Markdown regression report.

    Parameters
    ----------
    report : RegressionReport

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# Helicon Annual Validation Report",
        "",
        f"**Run timestamp:** {report.run_timestamp}",
        f"**Helicon version:** {report.helicon_version}",
        f"**Baseline:** {report.baseline_path}",
        "",
        "## Summary",
        "",
        "| Category | Count |",
        "|---|---|",
        f"| Regressions (PASS→FAIL) | {report.n_regressions} |",
        f"| Fixed (FAIL→PASS) | {report.n_fixed} |",
        f"| Unchanged | {report.n_unchanged} |",
        "",
    ]

    if report.all_passed:
        no_reg = "> **No regressions detected.** All previously passing cases still pass."
        lines += [no_reg, ""]
    else:
        lines += [
            f"> **{report.n_regressions} regression(s) detected!**  "
            "Previously passing cases now fail.",
            "",
        ]

    lines += ["## Case Details", ""]

    for case in report.cases:
        b_status = "PASS" if case.baseline_passed else "FAIL"
        c_status = "PASS" if case.current_passed else "FAIL"

        if case.regressed:
            icon = "🔴"
        elif case.fixed:
            icon = "🟢"
        else:
            icon = "✅" if case.current_passed else "❌"

        lines.append(f"### {icon} {case.case_name}")
        lines.append("")
        lines.append(f"- Baseline: **{b_status}** → Current: **{c_status}**")

        changed_metrics = [d for d in case.metric_diffs if d.changed]
        if changed_metrics:
            lines.append("- Metric changes:")
            for d in changed_metrics:
                change_str = ""
                if d.relative_change is not None:
                    pct = d.relative_change * 100
                    change_str = f" ({pct:+.2f}%)"
                lines.append(
                    f"  - `{d.metric_name}`: "
                    f"`{d.baseline_value}` → `{d.current_value}`{change_str}"
                )
        else:
            lines.append("- No metric changes.")
        lines.append("")

    return "\n".join(lines)


def save_regression_report(
    report: RegressionReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save regression report as both JSON and Markdown.

    Parameters
    ----------
    report : RegressionReport
    output_dir : str | Path
        Directory to write output files.

    Returns
    -------
    tuple[Path, Path]
        Paths to ``(regression_report.json, regression_report.md)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_data = {
        "baseline_path": report.baseline_path,
        "run_timestamp": report.run_timestamp,
        "helicon_version": report.helicon_version,
        "n_regressions": report.n_regressions,
        "n_fixed": report.n_fixed,
        "n_unchanged": report.n_unchanged,
        "all_passed": report.all_passed,
        "cases": [
            {
                "case_name": c.case_name,
                "baseline_passed": c.baseline_passed,
                "current_passed": c.current_passed,
                "status_changed": c.status_changed,
                "regressed": c.regressed,
                "fixed": c.fixed,
                "metric_diffs": [
                    {
                        "metric_name": d.metric_name,
                        "baseline_value": d.baseline_value,
                        "current_value": d.current_value,
                        "changed": d.changed,
                        "relative_change": d.relative_change,
                    }
                    for d in c.metric_diffs
                ],
            }
            for c in report.cases
        ],
    }
    json_path = output_dir / "regression_report.json"
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")

    # Markdown
    md_path = output_dir / "regression_report.md"
    md_path.write_text(generate_markdown_report(report), encoding="utf-8")

    return json_path, md_path


# ---------------------------------------------------------------------------
# RegressionSuite convenience class
# ---------------------------------------------------------------------------


class RegressionSuite:
    """Convenience class for running the full regression workflow.

    Parameters
    ----------
    baseline_path : str | Path
        Path to the baseline JSON file.
    output_dir : str | Path
        Directory for output reports.
    """

    def __init__(
        self,
        baseline_path: str | Path,
        output_dir: str | Path = "results/regression",
    ) -> None:
        self.baseline_path = Path(baseline_path)
        self.output_dir = Path(output_dir)

    def run(
        self,
        current_results: list[dict] | None = None,
        *,
        run_simulations: bool = False,
        cases: list[str] | None = None,
    ) -> RegressionReport:
        """Run the regression suite.

        Parameters
        ----------
        current_results : list[dict], optional
            Pre-computed results.  If None, :func:`run_validation` is called.
        run_simulations : bool
            Whether to actually run WarpX for each case.
        cases : list[str], optional
            Subset of case names to check.

        Returns
        -------
        RegressionReport
        """
        if current_results is None:
            from helicon.validate.runner import run_validation

            val_report = run_validation(
                cases=cases,
                run_simulations=run_simulations,
            )
            current_results = val_report.results

        report = compare_to_baseline(current_results, self.baseline_path)
        save_regression_report(report, self.output_dir)
        return report

    def update_baseline(
        self,
        results: list[dict],
        *,
        helicon_version: str | None = None,
    ) -> None:
        """Overwrite the baseline with new results.

        Parameters
        ----------
        results : list[dict]
            New results to record as baseline.
        helicon_version : str, optional
        """
        save_baseline(results, self.baseline_path, helicon_version=helicon_version)

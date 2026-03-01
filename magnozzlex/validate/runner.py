"""Validation suite runner.

Runs all validation cases and produces a summary report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from magnozzlex.validate.cases.free_expansion import FreeExpansionCase
from magnozzlex.validate.cases.guiding_center import GuidingCenterCase
from magnozzlex.validate.cases.merino_ahedo import MerinoAhedoCase
from magnozzlex.validate.cases.mn1d_comparison import MN1DComparisonCase
from magnozzlex.validate.cases.resistive_dimov import ResistiveDimovCase
from magnozzlex.validate.cases.vasimr_plume import VASIMRPlumeCase

ALL_CASES = [
    FreeExpansionCase,
    GuidingCenterCase,
    MerinoAhedoCase,
    MN1DComparisonCase,
    ResistiveDimovCase,
    VASIMRPlumeCase,
]


@dataclass
class ValidationReport:
    """Summary of all validation results."""

    results: list[dict]
    all_passed: bool
    n_passed: int
    n_failed: int
    n_total: int

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = ["MagNozzleX Validation Report", "=" * 40]
        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"  [{status}] {r['case_name']}: {r['description']}")
        lines.append("-" * 40)
        lines.append(f"  {self.n_passed}/{self.n_total} passed")
        return "\n".join(lines)


def run_validation(
    *,
    cases: list[str] | None = None,
    output_base: str | Path = "results/validation",
    run_simulations: bool = True,
) -> ValidationReport:
    """Run the validation suite.

    Parameters
    ----------
    cases : list of str, optional
        Case names to run. None = run all.
    output_base : path
        Base directory for validation output.
    run_simulations : bool
        If True, actually run WarpX simulations. If False, only evaluate
        existing output (useful for re-analysis).
    """
    output_base = Path(output_base)

    selected = ALL_CASES
    if cases is not None:
        selected = [c for c in ALL_CASES if c.name in cases]
        if not selected:
            msg = f"No matching cases for {cases}. Available: {[c.name for c in ALL_CASES]}"
            raise ValueError(msg)

    results = []
    for case_cls in selected:
        case_dir = output_base / case_cls.name

        if run_simulations:
            config = case_cls.get_config()
            try:
                from magnozzlex.runner.launch import run_simulation

                run_simulation(config, output_dir=case_dir)
            except RuntimeError as exc:
                # WarpX not available — record failure
                results.append(
                    {
                        "case_name": case_cls.name,
                        "passed": False,
                        "metrics": {},
                        "tolerances": {},
                        "description": f"Simulation failed: {exc}",
                    }
                )
                continue

        # Evaluate results
        result = case_cls.evaluate(case_dir)
        results.append(
            {
                "case_name": result.case_name,
                "passed": result.passed,
                "metrics": result.metrics,
                "tolerances": result.tolerances,
                "description": result.description,
            }
        )

    n_passed = sum(1 for r in results if r["passed"])
    n_total = len(results)

    report = ValidationReport(
        results=results,
        all_passed=n_passed == n_total,
        n_passed=n_passed,
        n_failed=n_total - n_passed,
        n_total=n_total,
    )

    # Save report
    output_base.mkdir(parents=True, exist_ok=True)
    report_path = output_base / "validation_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))

    return report

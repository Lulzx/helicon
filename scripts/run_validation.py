#!/usr/bin/env python3
"""Run the Helicon validation suite."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Helicon validation suite")
    parser.add_argument("--cases", nargs="*", help="Case names to run (default: all)")
    parser.add_argument(
        "--output", default="results/validation", help="Output directory"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only evaluate existing output, don't run simulations",
    )
    parser.add_argument("--json", help="Save report JSON to this path")
    args = parser.parse_args()

    from helicon.validate.runner import run_validation

    report = run_validation(
        cases=args.cases, output_base=args.output, run_simulations=not args.no_run
    )
    print(report.summary())

    if args.json:
        import json

        Path(args.json).write_text(
            json.dumps([r for r in report.results], indent=2, default=str)
        )

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

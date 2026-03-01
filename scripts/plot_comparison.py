#!/usr/bin/env python3
"""Generate publication-quality comparison figures from validation results."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from validation results"
    )
    parser.add_argument("report_json", help="Path to validation_report.json")
    parser.add_argument(
        "--output",
        default="docs/validation_results",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png"
    )
    args = parser.parse_args()

    results = json.loads(Path(args.report_json).read_text())

    from magnozzlex.validate.report import save_validation_plots

    paths = save_validation_plots(results, args.output)
    for p in paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

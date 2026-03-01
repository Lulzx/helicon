"""Validation report generation.

Creates comparison plots and HTML summary reports from validation results.
"""

from __future__ import annotations

import json
from pathlib import Path


def plot_validation_comparison(result_dict: dict, output_path: str | Path) -> Path:
    """Create a bar chart comparing simulated vs reference metrics.

    Parameters
    ----------
    result_dict : dict
        Has keys: case_name, passed, metrics, tolerances, description.
    output_path : str or Path
        Where to save the PNG figure.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = "matplotlib is required for plotting: pip install matplotlib"
        raise ImportError(msg) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = result_dict.get("metrics", {})
    tolerances = result_dict.get("tolerances", {})
    case_name = result_dict.get("case_name", "unknown")
    passed = result_dict.get("passed", False)

    metric_names = list(metrics.keys())
    if not metric_names:
        # Create a minimal placeholder figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"{case_name}: no metrics", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{case_name} ({'PASS' if passed else 'FAIL'})")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    sim_values = []
    tol_values = []
    for name in metric_names:
        val = metrics[name]
        sim_values.append(val if isinstance(val, (int, float)) else 0.0)
        tol_values.append(tolerances.get(name, 0.0))

    fig, ax = plt.subplots(figsize=(max(6, len(metric_names) * 1.5), 4))
    x = range(len(metric_names))
    bars = ax.bar(x, sim_values, color="steelblue", alpha=0.8, label="Simulated")

    # Show tolerance as error bars if available
    if any(t > 0 for t in tol_values):
        ax.errorbar(x, sim_values, yerr=tol_values, fmt="none", ecolor="red",
                     capsize=4, label="Tolerance")

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    status = "PASS" if passed else "FAIL"
    ax.set_title(f"{case_name} ({status})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_validation_plots(
    validation_report_results: list[dict],
    output_dir: str | Path,
) -> list[Path]:
    """Save comparison plots for each validation case.

    Parameters
    ----------
    validation_report_results : list of dict
        Each dict has keys: case_name, passed, metrics, tolerances, description.
    output_dir : str or Path
        Directory for saved figures.

    Returns
    -------
    list of Path
        Paths to the saved PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for result in validation_report_results:
        case_name = result.get("case_name", "unknown")
        safe_name = case_name.replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"{safe_name}.png"
        p = plot_validation_comparison(result, output_path)
        paths.append(p)

    return paths


def generate_html_report(
    results: list[dict],
    output_dir: str | Path,
) -> Path:
    """Generate a self-contained HTML validation report.

    Parameters
    ----------
    results : list of dict
        Each dict has keys: case_name, passed, metrics, tolerances, description.
    output_dir : str or Path
        Directory where validation_report.html will be written.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "validation_report.html"

    n_passed = sum(1 for r in results if r.get("passed"))
    n_total = len(results)

    rows = []
    for r in results:
        case_name = r.get("case_name", "unknown")
        passed = r.get("passed", False)
        metrics = r.get("metrics", {})
        description = r.get("description", "")

        status_text = "PASS" if passed else "FAIL"
        status_color = "#28a745" if passed else "#dc3545"
        metrics_json = json.dumps(metrics, indent=2, default=str)

        rows.append(
            f"<tr>"
            f'<td>{case_name}</td>'
            f'<td style="color: {status_color}; font-weight: bold;">{status_text}</td>'
            f"<td><pre>{metrics_json}</pre></td>"
            f"<td>{description}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MagNozzleX Validation Report</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background-color: #f2f2f2; }}
pre {{ margin: 0; font-size: 0.85em; }}
h1 {{ color: #333; }}
.summary {{ font-size: 1.1em; margin-bottom: 1em; }}
</style>
</head>
<body>
<h1>MagNozzleX Validation Report</h1>
<p class="summary">{n_passed}/{n_total} cases passed.</p>
<table>
<thead>
<tr><th>Case Name</th><th>Status</th><th>Metrics</th><th>Description</th></tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>"""

    output_path.write_text(html)
    return output_path

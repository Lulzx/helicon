"""Helicon command-line interface.

Usage::

    helicon run --preset sunbird
    helicon run --config my_nozzle.yaml --output results/
    helicon postprocess --input warpx_output/ --metrics thrust
    helicon validate --all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

import helicon


@click.group()
@click.version_option(version=helicon.__version__, prog_name="helicon")
def main() -> None:
    """Helicon — Magnetic Nozzle Simulation Toolkit."""


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--preset", type=str, help="Built-in preset name (sunbird, dfd, ppr)")
@click.option("--output", "output_dir", type=click.Path(), help="Output directory")
@click.option("--dry-run", is_flag=True, help="Generate inputs without running WarpX")
@click.option("--validate-config", "validate_only", is_flag=True, help="Only validate config")
@click.option(
    "--gpu",
    type=click.Choice(["auto", "cuda", "cpu"]),
    default="auto",
    help="GPU backend selection (auto-detected if not specified)",
)
def run(
    config_path: str | None,
    preset: str | None,
    output_dir: str | None,
    dry_run: bool,
    validate_only: bool,
    gpu: str,
) -> None:
    """Run a magnetic nozzle simulation."""
    from helicon.config.parser import SimConfig
    from helicon.config.validators import validate_config

    if config_path and preset:
        click.echo("Error: specify --config or --preset, not both.", err=True)
        sys.exit(1)
    if not config_path and not preset:
        click.echo("Error: specify --config or --preset.", err=True)
        sys.exit(1)

    # Load config
    if preset:
        click.echo(f"Loading preset: {preset}")
        config = SimConfig.from_preset(preset)
    else:
        assert config_path is not None  # guaranteed by check above
        click.echo(f"Loading config: {config_path}")
        config = SimConfig.from_yaml(config_path)

    # Validate
    result = validate_config(config)
    if result.errors:
        click.echo("Configuration errors:", err=True)
        for e in result.errors:
            click.echo(f"  ERROR: {e}", err=True)
        sys.exit(1)
    if result.warnings:
        for w in result.warnings:
            click.echo(f"  WARNING: {w}", err=True)

    if validate_only:
        click.echo("Configuration is valid.")
        return

    # Run
    from helicon.runner.launch import run_simulation

    click.echo(f"Starting simulation... (gpu={gpu})")
    run_result = run_simulation(
        config,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    if dry_run:
        click.echo(f"Dry run complete. Input files written to: {run_result.output_dir}")
    elif run_result.success:
        click.echo(f"Simulation complete ({run_result.wall_time_seconds:.1f}s)")
        click.echo(f"Output: {run_result.output_dir}")
    else:
        click.echo("Simulation failed.", err=True)
        sys.exit(1)


@main.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option(
    "--metrics",
    default="thrust",
    help="Comma-separated metrics: thrust,detachment,plume,pulsed,report",
)
@click.option("--output", "output_file", type=click.Path(), help="Output JSON file")
def postprocess(input_dir: str, metrics: str, output_file: str | None) -> None:
    """Extract propulsion metrics from WarpX output.

    Available metrics: thrust, detachment, plume, pulsed, report
    """
    metric_list = [m.strip() for m in metrics.split(",")]

    results = {}

    if "thrust" in metric_list:
        from helicon.postprocess.thrust import compute_thrust

        click.echo("Computing thrust...")
        try:
            thrust = compute_thrust(input_dir)
            results["thrust"] = {
                "thrust_N": thrust.thrust_N,
                "mass_flow_rate_kgs": thrust.mass_flow_rate_kgs,
                "isp_s": thrust.isp_s,
                "exhaust_velocity_ms": thrust.exhaust_velocity_ms,
                "n_particles_counted": thrust.n_particles_counted,
            }
            click.echo(f"  Thrust:     {thrust.thrust_N:.4f} N")
            click.echo(f"  Isp:        {thrust.isp_s:.0f} s")
            click.echo(f"  v_exhaust:  {thrust.exhaust_velocity_ms:.0f} m/s")
        except FileNotFoundError as exc:
            click.echo(f"  Error: {exc}", err=True)

    if "detachment" in metric_list:
        from helicon.postprocess.detachment import compute_detachment

        click.echo("Computing detachment efficiency...")
        try:
            det = compute_detachment(input_dir)
            results["detachment"] = {
                "eta_momentum": det.eta_momentum,
                "eta_particle": det.eta_particle,
                "eta_energy": det.eta_energy,
            }
            click.echo(f"  η_det (momentum): {det.eta_momentum:.3f}")
            click.echo(f"  η_det (particle): {det.eta_particle:.3f}")
            click.echo(f"  η_det (energy):   {det.eta_energy:.3f}")
        except FileNotFoundError as exc:
            click.echo(f"  Error: {exc}", err=True)

    if "plume" in metric_list:
        from helicon.postprocess.plume import compute_plume_metrics

        click.echo("Computing plume metrics...")
        try:
            plume = compute_plume_metrics(input_dir)
            results["plume"] = {
                "half_angle_deg": plume.half_angle_deg,
                "beam_efficiency": plume.beam_efficiency,
                "radial_loss_fraction": plume.radial_loss_fraction,
                "thrust_coefficient": plume.thrust_coefficient,
            }
            click.echo(f"  Half-angle:     {plume.half_angle_deg:.1f}°")
            click.echo(f"  Beam efficiency: {plume.beam_efficiency:.3f}")
        except FileNotFoundError as exc:
            click.echo(f"  Error: {exc}", err=True)

    if "pulsed" in metric_list:
        from helicon.postprocess.pulsed import compute_pulsed_metrics

        click.echo("Computing pulsed metrics...")
        try:
            pulsed = compute_pulsed_metrics(input_dir)
            results["pulsed"] = {
                "impulse_Ns": pulsed.impulse_Ns,
                "energy_J": pulsed.energy_J,
                "isp_s": pulsed.isp_s,
            }
            click.echo(f"  Impulse bit: {pulsed.impulse_Ns:.4f} N·s")
        except FileNotFoundError as exc:
            click.echo(f"  Error: {exc}", err=True)

    if "report" in metric_list:
        from helicon.postprocess.report import generate_report, save_report

        click.echo("Generating full report...")
        try:
            report = generate_report(input_dir)
            report_path = Path(input_dir) / "report.json"
            save_report(report, report_path)
            results["report"] = {"saved_to": str(report_path)}
            click.echo(f"  Report saved to: {report_path}")
        except FileNotFoundError as exc:
            click.echo(f"  Error: {exc}", err=True)

    if output_file:
        Path(output_file).write_text(json.dumps(results, indent=2))
        click.echo(f"Results written to: {output_file}")
    else:
        click.echo(json.dumps(results, indent=2))


@main.command()
@click.option("--all", "run_all", is_flag=True, help="Run all validation cases")
@click.option("--case", multiple=True, help="Specific case name(s)")
@click.option("--output", "output_dir", default="results/validation", type=click.Path())
@click.option("--evaluate-only", is_flag=True, help="Only evaluate existing output")
def validate(
    run_all: bool,
    case: tuple[str, ...],
    output_dir: str,
    evaluate_only: bool,
) -> None:
    """Run the validation suite."""
    from helicon.validate.runner import run_validation

    cases = list(case) if case else None
    if not run_all and not cases:
        click.echo("Specify --all or --case <name>.", err=True)
        sys.exit(1)

    click.echo("Running validation suite...")
    report = run_validation(
        cases=cases,
        output_base=output_dir,
        run_simulations=not evaluate_only,
    )

    click.echo(report.summary())

    if not report.all_passed:
        sys.exit(1)


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--vary",
    "vary_specs",
    multiple=True,
    required=True,
    metavar="PATH:LOW:HIGH:N",
    help="Parameter range, e.g. coils.0.I:20000:80000:5",
)
@click.option("--output", "output_dir", default="scan_results", type=click.Path())
@click.option(
    "--method",
    type=click.Choice(["grid", "lhc"]),
    default="grid",
    show_default=True,
    help="Sampling method: full grid or Latin Hypercube",
)
@click.option("--dry-run", is_flag=True, help="Skip WarpX; generate configs only")
@click.option("--seed", default=0, type=int, show_default=True, help="RNG seed (LHC only)")
@click.option(
    "--prescreen", is_flag=True, help="Run Tier 1 analytical pre-screening before WarpX"
)
@click.option(
    "--min-mirror-ratio",
    default=1.5,
    show_default=True,
    type=float,
    help="Minimum mirror ratio for prescreening filter",
)
def scan(
    config_path: str,
    vary_specs: tuple[str, ...],
    output_dir: str,
    method: str,
    dry_run: bool,
    seed: int,
    prescreen: bool,
    min_mirror_ratio: float,
) -> None:
    """Run a parameter scan over coil/plasma parameters."""
    from helicon.config.parser import SimConfig
    from helicon.optimize.scan import ParameterRange, run_scan

    config = SimConfig.from_yaml(config_path)

    try:
        ranges = [ParameterRange.from_string(s) for s in vary_specs]
    except ValueError as exc:
        click.echo(f"Error parsing --vary: {exc}", err=True)
        sys.exit(1)

    n_points = 1
    for r in ranges:
        n_points *= r.n
    click.echo(f"Scan: {n_points} points, method={method}")
    for r in ranges:
        click.echo(f"  {r.path}: [{r.low}, {r.high}] n={r.n}")

    result = run_scan(
        config,
        ranges,
        output_base=output_dir,
        method=method,
        dry_run=dry_run,
        seed=seed,
        prescreening=prescreen,
        min_mirror_ratio=min_mirror_ratio,
    )

    n_ok = sum(1 for m in result.metrics if m.get("success"))
    click.echo(f"Done: {n_ok}/{n_points} points succeeded.")
    if result.n_screened > 0:
        click.echo(
            f"  Prescreened: {result.n_screened}/{n_points} "
            f"filtered by mirror ratio < {min_mirror_ratio}"
        )
    click.echo(f"Output: {output_dir}/")


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--vary", "vary_specs", multiple=True, required=True, metavar="PATH:LOW:HIGH:N")
@click.option("--objective", default="thrust_N", help="Objective to maximize")
@click.option("--n-iterations", default=20, show_default=True, type=int)
@click.option("--output", "output_dir", default="optimize_results", type=click.Path())
@click.option("--dry-run", is_flag=True)
def optimize(
    config_path: str,
    vary_specs: tuple[str, ...],
    objective: str,
    n_iterations: int,
    output_dir: str,
    dry_run: bool,
) -> None:
    """Run Bayesian optimization over coil/plasma parameters."""
    from helicon.config.parser import SimConfig
    from helicon.optimize.scan import ParameterRange, run_scan

    config = SimConfig.from_yaml(config_path)

    try:
        ranges = [ParameterRange.from_string(s) for s in vary_specs]
    except ValueError as exc:
        click.echo(f"Error parsing --vary: {exc}", err=True)
        sys.exit(1)

    # Override n per range to distribute n_iterations via LHC
    for r in ranges:
        r.n = max(1, n_iterations // len(ranges))

    click.echo(f"Optimize: {n_iterations} iterations, method=lhc, objective={objective}")
    for r in ranges:
        click.echo(f"  {r.path}: [{r.low}, {r.high}] n={r.n}")

    result = run_scan(
        config,
        ranges,
        output_base=output_dir,
        method="lhc",
        dry_run=dry_run,
        seed=0,
    )

    # Extract best point by objective
    best_metric = None
    best_value = None
    for m in result.metrics:
        val = m.get(objective)
        if val is not None and (best_value is None or val > best_value):
            best_value = val
            best_metric = m

    n_ok = sum(1 for m in result.metrics if m.get("success"))
    click.echo(f"Done: {n_ok}/{len(result.metrics)} points succeeded.")
    if best_metric is not None:
        click.echo(f"Best {objective}: {best_value}")
        click.echo(f"  Parameters: {best_metric}")
    else:
        click.echo(f"No points returned a value for objective '{objective}'.")
    click.echo(f"Output: {output_dir}/")


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--preset", type=str, help="Built-in preset")
@click.option(
    "--resolutions",
    default="128x64,256x128,512x256",
    help="Comma-separated nzxnr resolution pairs",
)
@click.option("--output", "output_dir", default="convergence", type=click.Path())
@click.option("--dry-run", is_flag=True)
def convergence(
    config_path: str,
    preset: str | None,
    resolutions: str,
    output_dir: str,
    dry_run: bool,
) -> None:
    """Run a grid convergence study."""
    from helicon.config.parser import SimConfig
    from helicon.runner.convergence import run_convergence_study

    config = SimConfig.from_yaml(config_path)

    # Parse resolutions string into list of (nz, nr) tuples
    res_list: list[tuple[int, int]] = []
    for token in resolutions.split(","):
        token = token.strip()
        parts = token.lower().split("x")
        if len(parts) != 2:
            click.echo(
                f"Error: could not parse resolution '{token}' (expected NZxNR format)",
                err=True,
            )
            sys.exit(1)
        try:
            nz, nr = int(parts[0]), int(parts[1])
        except ValueError:
            click.echo(f"Error: non-integer values in resolution '{token}'", err=True)
            sys.exit(1)
        res_list.append((nz, nr))

    click.echo(f"Convergence study: {len(res_list)} resolution levels")
    for nz, nr in res_list:
        click.echo(f"  nz={nz}, nr={nr}")

    result = run_convergence_study(
        config,
        res_list,
        output_base=output_dir,
        dry_run=dry_run,
    )

    click.echo(f"Levels run: {len(result.levels)}")
    for lv in result.levels:
        status = "ok" if lv.success else "FAILED"
        thrust_str = f", thrust={lv.thrust_N:.4f} N" if lv.thrust_N is not None else ""
        click.echo(f"  nz={lv.nz} nr={lv.nr} [{status}]{thrust_str}")

    if result.convergence_order is not None:
        click.echo(f"Convergence order: {result.convergence_order:.2f}")
    if result.extrapolated_thrust_N is not None:
        click.echo(f"Extrapolated thrust: {result.extrapolated_thrust_N:.4f} N")
    click.echo(f"Converged: {result.converged}")
    click.echo(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()

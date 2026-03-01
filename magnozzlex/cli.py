"""MagNozzleX command-line interface.

Usage::

    magnozzlex run --preset sunbird
    magnozzlex run --config my_nozzle.yaml --output results/
    magnozzlex postprocess --input warpx_output/ --metrics thrust
    magnozzlex validate --all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

import magnozzlex


@click.group()
@click.version_option(version=magnozzlex.__version__, prog_name="magnozzlex")
def main() -> None:
    """MagNozzleX — Magnetic Nozzle Simulation Toolkit."""


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--preset", type=str, help="Built-in preset name (sunbird, dfd, ppr)")
@click.option("--output", "output_dir", type=click.Path(), help="Output directory")
@click.option("--dry-run", is_flag=True, help="Generate inputs without running WarpX")
@click.option("--validate-config", "validate_only", is_flag=True, help="Only validate config")
def run(
    config_path: str | None,
    preset: str | None,
    output_dir: str | None,
    dry_run: bool,
    validate_only: bool,
) -> None:
    """Run a magnetic nozzle simulation."""
    from magnozzlex.config.parser import SimConfig
    from magnozzlex.config.validators import validate_config

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
    from magnozzlex.runner.launch import run_simulation

    click.echo("Starting simulation...")
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
@click.option("--metrics", default="thrust", help="Comma-separated metrics: thrust")
@click.option("--output", "output_file", type=click.Path(), help="Output JSON file")
def postprocess(input_dir: str, metrics: str, output_file: str | None) -> None:
    """Extract propulsion metrics from WarpX output."""
    metric_list = [m.strip() for m in metrics.split(",")]

    results = {}

    if "thrust" in metric_list:
        from magnozzlex.postprocess.thrust import compute_thrust

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
    from magnozzlex.validate.runner import run_validation

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


if __name__ == "__main__":
    main()

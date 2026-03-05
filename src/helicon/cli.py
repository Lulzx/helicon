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
        error_msg = run_result.metadata.get("error", "unknown error")
        backend = run_result.metadata.get("backend", "unknown")
        rc = run_result.metadata.get("warpx_returncode", "?")
        click.echo(
            f"Simulation failed (backend={backend}, returncode={rc}): {error_msg}",
            err=True,
        )
        log = run_result.output_dir / "warpx_metal.log"
        if not log.exists():
            log = run_result.output_dir / "warpx.log"
        if log.exists():
            click.echo(f"Log: {log}", err=True)
        sys.exit(1)


@main.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option(
    "--metrics",
    default="thrust",
    help="Comma-separated metrics: thrust,detachment,plume,pulsed,report",
)
@click.option("--output", "output_file", type=click.Path(), help="Output JSON file")
@click.option(
    "--plots",
    is_flag=True,
    default=False,
    help="Auto-generate matplotlib figures (field topology, thrust convergence, detachment map)",  # noqa: E501
)
@click.option(
    "--plot-format",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
    help="Output format for auto-generated figures",
)
def postprocess(
    input_dir: str,
    metrics: str,
    output_file: str | None,
    plots: bool,
    plot_format: str,
) -> None:
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

    if plots:
        from helicon.postprocess.plots import generate_all_plots

        click.echo("Generating plots...")
        saved = generate_all_plots(input_dir, fmt=plot_format)
        if saved:
            for p in saved:
                click.echo(f"  Plot saved: {p}")
        else:
            click.echo("  No plots generated (insufficient data or matplotlib unavailable)")

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
@click.option(
    "--cloud",
    type=click.Choice(["local"]),
    default=None,
    help="Submit scan to cloud HPC backend (local runs inline)",
)
@click.option(
    "--cloud-gpus",
    default=1,
    show_default=True,
    type=int,
    help="Number of GPUs to request (cloud backends)",
)
@click.option("--cloud-instance", default=None, help="Cloud instance type override")
def scan(
    config_path: str,
    vary_specs: tuple[str, ...],
    output_dir: str,
    method: str,
    dry_run: bool,
    seed: int,
    prescreen: bool,
    min_mirror_ratio: float,
    cloud: str | None,
    cloud_gpus: int,
    cloud_instance: str | None,
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

    if cloud is not None:
        from helicon.cloud.submit import submit_cloud_scan

        click.echo(f"Submitting to cloud backend: {cloud}")
        job = submit_cloud_scan(
            config_path,
            ranges,
            output_dir=output_dir,
            backend=cloud,
            method=method,
            dry_run=dry_run,
            seed=seed,
            n_gpus=cloud_gpus,
            instance_type=cloud_instance,
        )
        click.echo(f"Job ID: {job.job_id}  Status: {job.status}")
        click.echo(f"Manifest: {Path(output_dir) / 'cloud_job.json'}")
        return

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


@main.command()
@click.option(
    "--repeat",
    default=3,
    type=int,
    show_default=True,
    help="Number of timing repetitions per benchmark.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Write results as JSON to this path.",
)
def benchmark(repeat: int, output_path: str | None) -> None:
    """Run the Helicon MLX-vs-NumPy benchmark suite."""

    from helicon.benchmark import run_benchmarks

    click.echo("Running benchmarks…")
    suite = run_benchmarks()
    click.echo(suite.summary())

    if output_path:
        import json
        from pathlib import Path

        data = [
            {
                "name": r.name,
                "numpy_ms": r.numpy_ms,
                "mlx_ms": r.mlx_ms,
                "speedup": r.speedup,
            }
            for r in suite.results
        ]
        Path(output_path).write_text(json.dumps(data, indent=2))
        click.echo(f"Results written to: {output_path}")


@main.command("throttle-map")
@click.option("--preset", type=str, help="Built-in preset name")
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--power-min", default=1e3, show_default=True, type=float, help="Min power [W]")
@click.option("--power-max", default=1e7, show_default=True, type=float, help="Max power [W]")
@click.option(
    "--mdot-min", default=1e-6, show_default=True, type=float, help="Min mass flow [kg/s]"
)
@click.option(
    "--mdot-max", default=1e-3, show_default=True, type=float, help="Max mass flow [kg/s]"
)
@click.option("--n-power", default=20, show_default=True, type=int, help="Power grid points")
@click.option("--n-mdot", default=20, show_default=True, type=int, help="Mdot grid points")
@click.option(
    "--eta-thermal",
    default=0.65,
    show_default=True,
    type=float,
    help="Thermal coupling efficiency",
)
@click.option(
    "--output",
    "output_path",
    default="throttle_map.json",
    type=click.Path(),
    help="Output JSON path",
)
@click.option("--backend", default="auto", show_default=True, help="Biot-Savart backend")
def throttle_map(
    preset: str | None,
    config_path: str | None,
    power_min: float,
    power_max: float,
    mdot_min: float,
    mdot_max: float,
    n_power: int,
    n_mdot: int,
    eta_thermal: float,
    output_path: str,
    backend: str,
) -> None:
    """Generate a thrust/Isp throttle curve over (power, mdot) grid."""
    from helicon.config.parser import SimConfig
    from helicon.mission.throttle import generate_throttle_map

    if preset and config_path:
        click.echo("Error: specify --preset or --config, not both.", err=True)
        sys.exit(1)
    if not preset and not config_path:
        click.echo("Error: specify --preset or --config.", err=True)
        sys.exit(1)

    config = SimConfig.from_preset(preset) if preset else SimConfig.from_yaml(config_path)

    click.echo(
        f"Generating throttle map: {n_power}×{n_mdot} grid "
        f"P=[{power_min:.0f}, {power_max:.0f}] W  "
        f"ṁ=[{mdot_min:.2e}, {mdot_max:.2e}] kg/s"
    )

    tm = generate_throttle_map(
        config,
        power_range_W=(power_min, power_max),
        mdot_range_kgs=(mdot_min, mdot_max),
        n_power=n_power,
        n_mdot=n_mdot,
        eta_thermal=eta_thermal,
        backend=backend,
    )

    path = tm.save_json(output_path)
    click.echo(f"Mirror ratio:   {tm.mirror_ratio:.2f}")
    click.echo(f"Detachment η_d: {float(tm.eta_d.mean()):.3f}")
    isp_range = (float(tm.isp_s.min()), float(tm.isp_s.max()))
    thrust_range = (float(tm.thrust_N.min()), float(tm.thrust_N.max()))
    click.echo(f"Isp range:      {isp_range[0]:.0f} – {isp_range[1]:.0f} s")
    click.echo(f"Thrust range:   {thrust_range[0]:.4f} – {thrust_range[1]:.4f} N")
    click.echo(f"Saved to: {path}")


@main.command()
@click.option(
    "--throttle-map",
    "throttle_path",
    required=True,
    type=click.Path(exists=True),
    help="Throttle map JSON (from helicon throttle-map)",
)
@click.option("--dry-mass", "dry_mass_kg", required=True, type=float, help="Dry mass [kg]")
@click.option(
    "--delta-v",
    "delta_v_ms",
    required=True,
    type=float,
    help="ΔV budget [m/s]",
)
@click.option(
    "--power",
    "power_W",
    default=None,
    type=float,
    help="Operating power [W] (defaults to map centre)",
)
@click.option(
    "--mdot",
    "mdot_kgs",
    default=None,
    type=float,
    help="Mass flow rate [kg/s] (defaults to map centre)",
)
@click.option("--output", "output_file", type=click.Path(), help="Output JSON file")
def mission(
    throttle_path: str,
    dry_mass_kg: float,
    delta_v_ms: float,
    power_W: float | None,
    mdot_kgs: float | None,
    output_file: str | None,
) -> None:
    """Estimate propellant budget and burn time for a ΔV manoeuvre."""
    from helicon.mission.throttle import ThrottleMap
    from helicon.mission.trajectory import MissionLeg, analyze_mission

    tm = ThrottleMap.load_json(throttle_path)

    # Default to centre of throttle map
    if power_W is None:
        power_W = float(tm.power_grid_W[len(tm.power_grid_W) // 2])
    if mdot_kgs is None:
        mdot_kgs = float(tm.mdot_grid_kgs[len(tm.mdot_grid_kgs) // 2])

    legs = [MissionLeg("main burn", delta_v_ms, power_W, mdot_kgs)]
    result = analyze_mission(legs, tm, dry_mass_kg)

    click.echo(f"ΔV:              {result.total_delta_v_ms:.0f} m/s")
    click.echo(f"Isp:             {result.isp_s:.0f} s")
    click.echo(f"Propellant:      {result.propellant_mass_kg:.2f} kg")
    click.echo(f"Wet mass:        {result.wet_mass_kg:.2f} kg")
    click.echo(f"Payload fraction:{result.payload_fraction:.3f}")
    burn_h = result.burn_time_s / 3600
    click.echo(f"Burn time:       {result.burn_time_s:.0f} s ({burn_h:.2f} h)")

    if output_file:
        import dataclasses

        data = dataclasses.asdict(result)
        Path(output_file).write_text(json.dumps(data, indent=2))
        click.echo(f"Results written to: {output_file}")


@main.command("run-3d")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", type=click.Path(), default="results/sim3d")
@click.option("--particles", "n_particles", default=2000, type=int)
@click.option("--steps", "n_steps", default=5000, type=int)
@click.option("--backend", default="auto", type=click.Choice(["auto", "mlx", "numpy"]))
@click.option("--seed", default=0, type=int)
@click.option("--save-json", "save_json", is_flag=True, help="Save result as JSON")
def run_3d(
    config_path: str,
    output_dir: str,
    n_particles: int,
    n_steps: int,
    backend: str,
    seed: int,
    save_json: bool,
) -> None:
    """Run a 3D Boris-pusher test-particle simulation."""
    import json

    from helicon.config.parser import SimConfig
    from helicon.runner.sim3d import Sim3DConfig, run_3d_simulation

    config = SimConfig.from_yaml(config_path)
    sim_cfg = Sim3DConfig(
        n_particles=n_particles,
        n_steps=n_steps,
        backend=backend,
        seed=seed,
    )

    click.echo(
        f"Running 3D simulation: {n_particles} particles × {n_steps} steps [backend={backend}]"
    )
    result = run_3d_simulation(config, sim_cfg)
    click.echo(result.summary())

    if save_json:
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "sim3d_result.json"
        p.write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"Result saved to: {p}")


@main.command()
@click.option("--port", default=8501, type=int, help="Streamlit server port")
@click.option(
    "--no-browser",
    "no_browser",
    is_flag=True,
    help="Do not open a browser tab automatically",
)
def app(port: int, no_browser: bool) -> None:
    """Launch the interactive nozzle design explorer (Streamlit)."""
    from helicon.app.launcher import launch_app

    click.echo(f"Launching Helicon design app on http://localhost:{port} ...")
    launch_app(port=port, browser=not no_browser)


@main.command("surrogate-train")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    help="YAML config (used for domain bounds in sampling)",
)
@click.option(
    "--n-samples",
    "n_samples",
    default=500,
    type=int,
    help="Number of training samples",
)
@click.option("--epochs", default=300, type=int, help="Training epochs")
@click.option(
    "--output", "output_dir", required=True, type=click.Path(), help="Output directory"
)
@click.option("--seed", default=0, type=int, help="Random seed")
def surrogate_train(
    config_path: str | None,
    n_samples: int,
    epochs: int,
    output_dir: str,
    seed: int,
) -> None:
    """Train the MLX neural surrogate on generated analytical data."""
    from helicon.surrogate.training import generate_training_data, train_surrogate

    click.echo(f"Generating {n_samples} training samples (seed={seed})...")
    data = generate_training_data(n_samples=n_samples, seed=seed)
    click.echo(f"Training MLP surrogate for {epochs} epochs...")
    surrogate = train_surrogate(data, epochs=epochs, verbose=True, seed=seed)
    path = Path(output_dir)
    surrogate.save(path)
    click.echo(f"Surrogate saved to: {path.resolve()}")
    click.echo(surrogate.accuracy_envelope()["notes"])


@main.command("export-cad")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["step", "iges"]),
    default="step",
    help="CAD output format",
)
def export_cad(config_path: str, output_path: str, fmt: str) -> None:
    """Export coil geometry to STEP or IGES for mechanical CAD."""
    from helicon.config.parser import SimConfig
    from helicon.export.cad import export_coils_iges, export_coils_step

    config = SimConfig.from_yaml(config_path)
    out = Path(output_path)
    if fmt == "step":
        result = export_coils_step(config, out)
    else:
        result = export_coils_iges(config, out)
    click.echo(f"Exported {len(config.nozzle.coils)} coil(s) to: {result}")


# ---------------------------------------------------------------------------
# helicon array — multi-thruster array
# ---------------------------------------------------------------------------


_DATA_TYPE_CHOICE = click.Choice(["experimental", "simulation", "analytical"])


@main.command("array")
@click.option(
    "--n", "n_thrusters", type=int, default=2, show_default=True, help="Number of thrusters"
)
@click.option(
    "--sep",
    "separation_m",
    type=float,
    default=0.5,
    show_default=True,
    help="Centre-to-centre separation [m]",
)
@click.option(
    "--thrust", "thrust_n", type=float, multiple=True, help="Per-thruster thrust [N]"
)
@click.option("--isp", "isp_s", type=float, multiple=True, help="Per-thruster Isp [s]")
@click.option(
    "--angle", "half_angle_deg", type=float, multiple=True, help="Plume half-angle [deg]"
)
@click.option(
    "--ref-z",
    "reference_z_m",
    type=float,
    default=1.0,
    show_default=True,
    help="Evaluation plane [m]",
)
def array_cmd(
    n_thrusters: int,
    separation_m: float,
    thrust_n: tuple[float, ...],
    isp_s: tuple[float, ...],
    half_angle_deg: tuple[float, ...],
    reference_z_m: float,
) -> None:
    """Compute combined performance of a multi-thruster array."""
    from helicon.multithruster import ArrayConfig, ThrusterArray

    thrust = list(thrust_n) if thrust_n else [0.1] * n_thrusters
    isp = list(isp_s) if isp_s else [3000.0] * n_thrusters
    angles = list(half_angle_deg) if half_angle_deg else [15.0] * n_thrusters

    # Pad or truncate to n_thrusters
    thrust = (thrust * n_thrusters)[:n_thrusters]
    isp = (isp * n_thrusters)[:n_thrusters]
    angles = (angles * n_thrusters)[:n_thrusters]

    cfg = ArrayConfig(
        n_thrusters=n_thrusters,
        separation_m=separation_m,
        thrust_N=thrust,
        isp_s=isp,
        plume_half_angle_deg=angles,
        reference_z_m=reference_z_m,
    )
    result = ThrusterArray(cfg).compute()

    click.echo(f"Array: {n_thrusters} thrusters, separation={separation_m}m")
    click.echo(f"  Nominal thrust:      {result.nominal_thrust_N * 1e3:.2f} mN")
    click.echo(f"  Effective thrust:    {result.total_thrust_N * 1e3:.2f} mN")
    click.echo(f"  Interaction penalty: {result.interaction_penalty * 100:.1f}%")
    click.echo(f"  Effective Isp:       {result.effective_isp_s:.0f} s")
    click.echo(f"  Mass flow rate:      {result.total_mass_flow_kgs * 1e6:.3f} mg/s")
    for pr in result.pair_interactions:
        click.echo(
            f"  Pair ({pr.i},{pr.j}): sep={pr.separation_m:.2f}m  "
            f"overlap={pr.overlap_factor:.3f}  penalty={pr.thrust_penalty_fraction * 100:.1f}%"
        )


# ---------------------------------------------------------------------------
# helicon plugins — list registered plugins
# ---------------------------------------------------------------------------


@main.command("plugins")
@click.option("--namespace", "namespace", type=str, default=None, help="Filter to a namespace")
def plugins_cmd(namespace: str | None) -> None:
    """List registered plugins in the default registry."""
    from helicon.plugins import list_plugins

    listing = list_plugins(namespace)
    if not listing:
        click.echo("No plugins registered.")
        return
    for ns, names in sorted(listing.items()):
        if names:
            click.echo(f"  {ns}:")
            for name in sorted(names):
                click.echo(f"    - {name}")


# ---------------------------------------------------------------------------
# helicon valdb — validation database operations
# ---------------------------------------------------------------------------


@main.group("valdb")
@click.option(
    "--db", "db_path", type=click.Path(), default="~/.helicon/valdb", show_default=True
)
@click.pass_context
def valdb_group(ctx: click.Context, db_path: str) -> None:
    """Manage the collaborative validation database."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


@valdb_group.command("add")
@click.option("--case-id", required=True)
@click.option("--source", required=True)
@click.option("--contributor", required=True)
@click.option("--type", "data_type", required=True, type=_DATA_TYPE_CHOICE)
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--notes", default="")
@click.pass_context
def valdb_add(
    ctx: click.Context,
    case_id: str,
    source: str,
    contributor: str,
    data_type: str,
    tags: str,
    notes: str,
) -> None:
    """Add a record to the validation database."""
    from helicon.valdb import ValidationDatabase, ValidationRecord

    db = ValidationDatabase(ctx.obj["db_path"])
    record = ValidationRecord(
        case_id=case_id,
        source=source,
        contributor=contributor,
        data_type=data_type,  # type: ignore[arg-type]
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        notes=notes,
    )
    db.add(record)
    click.echo(f"Added record {record.record_id} to {db._path}")


@valdb_group.command("query")
@click.option("--case-id", default=None)
@click.option("--contributor", default=None)
@click.option("--type", "data_type", default=None, type=_DATA_TYPE_CHOICE)
@click.option("--tag", "tags", multiple=True)
@click.pass_context
def valdb_query(
    ctx: click.Context,
    case_id: str | None,
    contributor: str | None,
    data_type: str | None,
    tags: tuple[str, ...],
) -> None:
    """Query the validation database."""
    from helicon.valdb import ValidationDatabase

    db = ValidationDatabase(ctx.obj["db_path"])
    results = db.query(
        case_id=case_id,
        contributor=contributor,
        data_type=data_type,
        tags=list(tags) if tags else None,
    )
    click.echo(f"Found {len(results)} record(s):")
    for r in results:
        click.echo(
            f"  [{r.data_type:12s}] {r.case_id}  contributor={r.contributor}  tags={r.tags}"
        )


@valdb_group.command("export")
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option(
    "--format", "fmt", type=click.Choice(["json", "csv"]), default="json", show_default=True
)
@click.pass_context
def valdb_export(ctx: click.Context, output_path: str, fmt: str) -> None:
    """Export the validation database."""
    from helicon.valdb import ValidationDatabase

    db = ValidationDatabase(ctx.obj["db_path"])
    if fmt == "json":
        db.export_json(output_path)
    else:
        db.export_csv(output_path)
    click.echo(f"Exported {len(db)} record(s) to {output_path}")


@valdb_group.command("stats")
@click.pass_context
def valdb_stats(ctx: click.Context) -> None:
    """Show summary statistics for the validation database."""
    from helicon.valdb import ValidationDatabase

    db = ValidationDatabase(ctx.obj["db_path"])
    records = db.query()
    click.echo(f"Total records: {len(records)}")
    by_type: dict[str, int] = {}
    by_contributor: dict[str, int] = {}
    for r in records:
        by_type[r.data_type] = by_type.get(r.data_type, 0) + 1
        by_contributor[r.contributor] = by_contributor.get(r.contributor, 0) + 1
    click.echo("By type:")
    for t, n in sorted(by_type.items()):
        click.echo(f"  {t}: {n}")
    click.echo("By contributor:")
    for c, n in sorted(by_contributor.items()):
        click.echo(f"  {c}: {n}")


# ---------------------------------------------------------------------------
# helicon regression — annual validation regression suite
# ---------------------------------------------------------------------------


@main.group("regression")
def regression_group() -> None:
    """Manage the annual validation regression suite."""


@regression_group.command("save-baseline")
@click.option(
    "--output",
    "baseline_path",
    default="results/baseline.json",
    show_default=True,
    type=click.Path(),
)
def regression_save_baseline(baseline_path: str) -> None:
    """Save current validation results as the regression baseline."""
    from helicon.validate import run_validation, save_baseline

    click.echo("Running validation suite to capture baseline...")
    report = run_validation(run_simulations=False)
    save_baseline(report.results, baseline_path)
    click.echo(f"Baseline saved: {baseline_path}  ({report.n_total} cases)")


@regression_group.command("run")
@click.option(
    "--baseline",
    "baseline_path",
    default="results/baseline.json",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "output_dir",
    default="results/regression",
    show_default=True,
    type=click.Path(),
)
@click.option("--case", "cases", multiple=True, help="Limit to specific case names")
def regression_run(baseline_path: str, output_dir: str, cases: tuple[str, ...]) -> None:
    """Run regression comparison against the saved baseline."""
    from helicon.validate import RegressionSuite

    suite = RegressionSuite(baseline_path, output_dir=output_dir)
    report = suite.run(
        run_simulations=False,
        cases=list(cases) if cases else None,
    )
    status = "CLEAN" if report.all_passed else f"{report.n_regressions} REGRESSION(S)"
    click.echo(f"Regression: {status}")
    click.echo(f"  Fixed:     {report.n_fixed}")
    click.echo(f"  Unchanged: {report.n_unchanged}")
    click.echo(f"  Report:    {output_dir}/regression_report.md")
    if not report.all_passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# helicon perf — Apple Silicon performance profiler
# ---------------------------------------------------------------------------


@main.command("perf")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output machine-readable JSON instead of formatted text",
)
@click.option(
    "--no-bandwidth",
    "skip_bandwidth",
    is_flag=True,
    help="Skip memory bandwidth probe (faster startup)",
)
def perf_cmd(output_json: bool, skip_bandwidth: bool) -> None:
    """Profile Apple Silicon hardware and print WarpX/MLX tuning recommendations."""
    from helicon.perf import AppleSiliconProfiler

    profiler = AppleSiliconProfiler(measure_bandwidth=not skip_bandwidth)
    profile = profiler.profile()

    if output_json:
        click.echo(json.dumps(profile.to_dict(), indent=2))
    else:
        click.echo(profile.summary())
        click.echo(profile.recommendations())


# ---------------------------------------------------------------------------
# helicon doctor — environment health checker
# ---------------------------------------------------------------------------


@main.command("doctor")
@click.option("--json", "output_json", is_flag=True, help="Output machine-readable JSON")
def doctor_cmd(output_json: bool) -> None:
    """Check the Helicon environment: Python, dependencies, and WarpX binary."""
    from helicon.doctor import check_environment

    report = check_environment()
    if output_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(report.summary())
    if not report.healthy:
        sys.exit(1)


# ---------------------------------------------------------------------------
# helicon init — scaffold a new simulation config
# ---------------------------------------------------------------------------


@main.command("init")
@click.argument("name")
@click.option(
    "--preset",
    type=click.Choice(["custom", "sunbird", "highpower"]),
    default="custom",
    show_default=True,
    help="Starting template preset",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Output path (default: <name>.yaml)",
)
def init_cmd(name: str, preset: str, output_path: str | None) -> None:
    """Scaffold a new simulation config YAML for NAME."""
    from helicon.scaffold import scaffold_config

    yaml_text = scaffold_config(preset)
    dest = output_path or f"{name}.yaml"
    with open(dest, "w") as fh:
        fh.write(yaml_text)
    click.echo(f"Created {dest}  (preset: {preset})")
    click.echo(f"Edit the config, then run:  helicon run --config {dest}")


# ---------------------------------------------------------------------------
# helicon schema — export SimConfig JSON schema
# ---------------------------------------------------------------------------


@main.command("schema")
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write schema to file (default: print to stdout)",
)
@click.option("--indent", default=2, show_default=True, type=int, help="JSON indent level")
def schema_cmd(output_path: str | None, indent: int) -> None:
    """Export the JSON schema for Helicon simulation configs."""
    from helicon.config import SimConfig

    schema = SimConfig.model_json_schema()
    text = json.dumps(schema, indent=indent)
    if output_path:
        with open(output_path, "w") as fh:
            fh.write(text)
            fh.write("\n")
        click.echo(f"Schema written to {output_path}")
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# helicon sensitivity — Sobol variance-based sensitivity analysis
# ---------------------------------------------------------------------------


@main.command("sensitivity")
@click.option(
    "--preset",
    type=click.Choice(["sunbird", "highpower"]),
    default="sunbird",
    show_default=True,
    help="Baseline coil preset to perturb",
)
@click.option(
    "--n-samples",
    default=64,
    show_default=True,
    type=int,
    help="Saltelli base sample count N (total evals: N×(k+2))",
)
@click.option("--json", "output_json", is_flag=True, help="Output machine-readable JSON")
@click.option("--output", "output_path", default=None, type=click.Path())
def sensitivity_cmd(
    preset: str, n_samples: int, output_json: bool, output_path: str | None
) -> None:
    """Run Sobol sensitivity analysis on nozzle design parameters."""
    import numpy as np

    from helicon.fields.biot_savart import Coil
    from helicon.optimize.analytical import screen_geometry
    from helicon.optimize.sensitivity import SobolResult, compute_sobol

    # Baseline coil parameters by preset
    _PRESETS = {
        "sunbird": {"r": 0.075, "I": 25_000, "z_max": 1.20, "z_min": -0.10},
        "highpower": {"r": 0.15, "I": 80_000, "z_max": 2.00, "z_min": -0.25},
    }
    base = _PRESETS[preset]

    # Vary r (±30%), I (±40%), z_max (±25%) around baseline
    param_names = ["coil_r_m", "coil_I_kA", "z_max_m"]
    bounds = [
        [base["r"] * 0.70, base["r"] * 1.30],
        [base["I"] * 0.60, base["I"] * 1.40],
        [base["z_max"] * 0.75, base["z_max"] * 1.25],
    ]

    def _objective(X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        for i, row in enumerate(X):
            r_, I_, zm_ = float(row[0]), float(row[1]), float(row[2])
            coil = Coil(z=0.0, r=r_, I=I_)
            result = screen_geometry(
                [coil], z_min=base["z_min"], z_max=zm_, n_pts=80, backend="numpy"
            )
            out[i] = result.thrust_efficiency
        return out

    sobol: SobolResult = compute_sobol(
        _objective, n_samples=n_samples, bounds=bounds, param_names=param_names
    )

    if output_json:
        import json as _json

        data = {
            "preset": preset,
            "n_samples": n_samples,
            "param_names": sobol.param_names,
            "S1": sobol.S1.tolist(),
            "ST": sobol.ST.tolist(),
        }
        text = _json.dumps(data, indent=2)
        if output_path:
            with open(output_path, "w") as fh:
                fh.write(text + "\n")
            click.echo(f"Sensitivity results written to {output_path}")
        else:
            click.echo(text)
    else:
        click.echo(f"Preset: {preset}  (n_samples={n_samples})")
        click.echo(sobol.summary())
        if output_path:
            with open(output_path, "w") as fh:
                fh.write(sobol.summary() + "\n")


# ---------------------------------------------------------------------------
# helicon provenance — design audit trail
# ---------------------------------------------------------------------------


@main.group("provenance")
@click.option(
    "--db",
    "db_path",
    default="results/provenance.jsonl",
    show_default=True,
    type=click.Path(),
    help="Path to the provenance JSONL file",
)
@click.pass_context
def provenance_group(ctx: click.Context, db_path: str) -> None:
    """Browse and query the design provenance audit trail."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


@provenance_group.command("list")
@click.option("--tail", default=10, show_default=True, type=int, help="Show last N records")
@click.pass_context
def provenance_list(ctx: click.Context, tail: int) -> None:
    """List the most recent provenance records."""
    from helicon.provenance import ProvenanceDB

    db = ProvenanceDB(ctx.obj["db_path"])
    records = db.all_records()
    shown = records[-tail:] if len(records) > tail else records
    click.echo(f"Showing {len(shown)} of {len(records)} record(s):")
    for r in shown:
        click.echo(
            f"  [{r.fidelity_tier}] {r.record_id[:8]}…  {r.source:<24}  {r.timestamp[:19]}"
        )


@provenance_group.command("show")
@click.argument("record_id")
@click.pass_context
def provenance_show(ctx: click.Context, record_id: str) -> None:
    """Show full details of a provenance record by its ID (or prefix)."""
    from helicon.provenance import ProvenanceDB

    db = ProvenanceDB(ctx.obj["db_path"])
    # Support prefix matching
    match = None
    for r in db.all_records():
        if r.record_id.startswith(record_id):
            match = r
            break
    if match is None:
        click.echo(f"Record not found: {record_id}", err=True)
        sys.exit(1)
    click.echo(f"ID:      {match.record_id}")
    click.echo(f"Source:  {match.source}")
    click.echo(f"Tier:    {match.fidelity_tier}")
    click.echo(f"Time:    {match.timestamp}")
    click.echo(f"Parents: {match.parent_ids or 'none'}")
    click.echo(f"Notes:   {match.notes or '—'}")
    click.echo("Metrics:")
    for k, v in match.metrics.items():
        click.echo(f"  {k}: {v}")


@provenance_group.command("lineage")
@click.argument("record_id")
@click.pass_context
def provenance_lineage(ctx: click.Context, record_id: str) -> None:
    """Show the full ancestor chain for a record."""
    from helicon.provenance import ProvenanceDB

    db = ProvenanceDB(ctx.obj["db_path"])
    # Resolve prefix
    full_id = record_id
    for r in db.all_records():
        if r.record_id.startswith(record_id):
            full_id = r.record_id
            break
    chain = db.lineage(full_id)
    if not chain:
        click.echo(f"No lineage found for: {record_id}", err=True)
        sys.exit(1)
    click.echo(f"Lineage chain ({len(chain)} node(s)):")
    for i, r in enumerate(chain):
        prefix = "  " * i + ("└─" if i > 0 else "  ")
        click.echo(f"{prefix} [{r.fidelity_tier}] {r.source}  {r.record_id[:8]}…")


# ---------------------------------------------------------------------------
# helicon detach — real-time detachment analysis group
# ---------------------------------------------------------------------------

_SPECIES_CHOICES = click.Choice(["H+", "He+", "N+", "Ar+", "Kr+", "Xe+"])


@main.group("detach")
def detach_group() -> None:
    """Magnetic nozzle detachment analysis (v2.5 novel contributions)."""


# -- detach assess -----------------------------------------------------------


@detach_group.command("assess")
@click.option("--n", "n_m3", type=float, required=True, help="Plasma density [m^-3]")
@click.option("--Te", "Te_eV", type=float, required=True, help="Electron temperature [eV]")
@click.option("--Ti", "Ti_eV", type=float, required=True, help="Ion temperature [eV]")
@click.option("--B", "B_T", type=float, required=True, help="Magnetic field [T]")
@click.option(
    "--dBdz", "dBdz", type=float, default=-1.0, show_default=True, help="B gradient [T/m]"
)
@click.option("--vz", "vz_ms", type=float, required=True, help="Axial bulk velocity [m/s]")
@click.option("--species", type=_SPECIES_CHOICES, default="H+", show_default=True)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output")
@click.option("--control", is_flag=True, help="Output control recommendation dict")
def detach_assess_cmd(
    n_m3: float,
    Te_eV: float,
    Ti_eV: float,
    B_T: float,
    dBdz: float,
    vz_ms: float,
    species: str,
    output_json: bool,
    control: bool,
) -> None:
    """Assess detachment onset from local plasma parameters.

    Exit codes: 0 = attached, 1 = imminent, 2 = detached.
    """
    from helicon.detach import DetachmentOnsetModel, PlasmaState
    from helicon.detach.invariants import species_mass

    state = PlasmaState(
        n_m3=n_m3,
        Te_eV=Te_eV,
        Ti_eV=Ti_eV,
        B_T=B_T,
        dBdz_T_per_m=dBdz,
        vz_ms=vz_ms,
        mass_amu=species_mass(species),
    )
    model = DetachmentOnsetModel()

    if control:
        rec = model.control_recommendation(state)
        click.echo(json.dumps(rec, indent=2))
    elif output_json:
        ds = model.assess(state)
        click.echo(json.dumps(ds.to_dict(), indent=2))
    else:
        ds = model.assess(state)
        click.echo(ds.summary())
        if ds.is_detached:
            sys.exit(2)
        if ds.is_imminent:
            sys.exit(1)


# -- detach calibrate --------------------------------------------------------


@detach_group.command("calibrate")
@click.option(
    "--n-samples", default=500, show_default=True, help="Synthetic training samples."
)
@click.option("--seed", default=0, show_default=True, help="Random seed.")
@click.option(
    "--sharpness",
    default=10.0,
    show_default=True,
    help="Logistic decision boundary sharpness.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Save calibration result to JSON file.",
)
@click.option("--json", "output_json", is_flag=True, help="Print JSON summary.")
def detach_calibrate_cmd(
    n_samples: int,
    seed: int,
    sharpness: float,
    output_path: str | None,
    output_json: bool,
) -> None:
    """Calibrate detachment model weights via binary cross-entropy.

    Generates physics-motivated synthetic data (oracle labels from the
    Merino-Ahedo 2011 criterion + 5 % label noise), fits SLSQP-optimised
    weights subject to the simplex constraint w_A + w_β + w_Λ = 1.
    """
    from helicon.detach.calibration import DetachmentCalibrator

    if not output_json:
        click.echo(f"Generating {n_samples} synthetic training samples (seed={seed})…")
    records = DetachmentCalibrator.generate_synthetic_data(n_samples=n_samples, seed=seed)

    cal = DetachmentCalibrator(sharpness=sharpness)
    if not output_json:
        click.echo("Fitting weights (SLSQP + simplex constraint)…")
    result = cal.fit(records)

    if output_json:
        click.echo(
            json.dumps(
                result.to_model_kwargs()
                | {
                    "n_samples": result.n_samples,
                    "log_loss": result.log_loss,
                    "accuracy": result.accuracy,
                },
                indent=2,
            )
        )
    else:
        click.echo(result.summary())

    if output_path:
        import pathlib

        pathlib.Path(output_path).write_text(json.dumps(result.to_model_kwargs(), indent=2))
        click.echo(f"Saved to {output_path}")


# -- detach invert ------------------------------------------------------------


@detach_group.command("invert")
@click.option("--F", "F_N", type=float, required=True, help="Measured thrust [N]")
@click.option("--mdot", type=float, required=True, help="Mass flow rate [kg/s]")
@click.option("--B", "B_T", type=float, required=True, help="Throat B-field [T]")
@click.option("--area", type=float, required=True, help="Throat area [m^2]")
@click.option(
    "--mirror-ratio", default=5.0, show_default=True, help="Mirror ratio R_B = B_throat/B_exit"
)
@click.option("--species", type=_SPECIES_CHOICES, default="H+", show_default=True)
@click.option(
    "--Te",
    "Te_eV",
    type=float,
    default=50.0,
    show_default=True,
    help="Nominal electron temperature [eV]",
)
@click.option(
    "--dBdz",
    "dBdz",
    type=float,
    default=-2.0,
    show_default=True,
    help="Axial B gradient at throat [T/m]",
)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output")
def detach_invert_cmd(
    F_N: float,
    mdot: float,
    B_T: float,
    area: float,
    mirror_ratio: float,
    species: str,
    Te_eV: float,
    dBdz: float,
    output_json: bool,
) -> None:
    """Infer plasma state from thrust observables (no diagnostics needed).

    Closed-form reconstruction: M_A = [F/(ṁ η_T B)] √(μ₀ ṁ mᵢ/A).
    """
    from helicon.detach.invariants import species_mass
    from helicon.detach.inverse import ThrustInverter, ThrustObservation

    obs = ThrustObservation(
        F_thrust_N=F_N,
        m_dot_kg_s=mdot,
        B_throat_T=B_T,
        A_throat_m2=area,
        mass_amu=species_mass(species),
        Te_eV_nominal=Te_eV,
        dBdz_T_per_m=dBdz,
    )
    inv = ThrustInverter(mirror_ratio=mirror_ratio)
    state = inv.invert(obs)

    if output_json:
        click.echo(
            json.dumps(
                {
                    "n_m3": state.n_m3,
                    "vz_ms": state.vz_ms,
                    "Ti_eV_est": state.Ti_eV_est,
                    "alfven_mach": state.alfven_mach,
                    "detachment_score": state.detachment_score,
                    "confidence": state.confidence,
                    "residual": state.residual,
                },
                indent=2,
            )
        )
    else:
        click.echo(f"  density        n = {state.n_m3:.3e} m⁻³")
        click.echo(f"  exhaust vel    vz = {state.vz_ms:.3e} m/s")
        click.echo(f"  Alfvén Mach    M_A = {state.alfven_mach:.3f}")
        click.echo(f"  detach score   S = {state.detachment_score:.3f}")
        click.echo(f"  confidence     = {state.confidence:.2%}")


# -- detach simulate ----------------------------------------------------------


@detach_group.command("simulate")
@click.option("--n", "n_m3", type=float, required=True, help="Plasma density [m^-3]")
@click.option("--Te", "Te_eV", type=float, required=True, help="Electron temperature [eV]")
@click.option("--Ti", "Ti_eV", type=float, required=True, help="Ion temperature [eV]")
@click.option("--B", "B_T", type=float, required=True, help="Magnetic field [T]")
@click.option(
    "--dBdz", "dBdz", type=float, default=-1.0, show_default=True, help="B gradient [T/m]"
)
@click.option("--vz", "vz_ms", type=float, required=True, help="Axial bulk velocity [m/s]")
@click.option("--species", type=_SPECIES_CHOICES, default="H+", show_default=True)
@click.option(
    "--setpoint", default=0.35, show_default=True, help="Target detachment score S* ∈ (0,1)"
)
@click.option("--steps", default=20, show_default=True, help="Number of control timesteps")
@click.option("--dt", "dt_s", default=0.01, show_default=True, help="Timestep [s]")
@click.option(
    "--decay-rate", default=1.0, show_default=True, help="Lyapunov decay rate α [1/s]"
)
@click.option("--json", "output_json", is_flag=True, help="Output full JSON trace")
def detach_simulate_cmd(
    n_m3: float,
    Te_eV: float,
    Ti_eV: float,
    B_T: float,
    dBdz: float,
    vz_ms: float,
    species: str,
    setpoint: float,
    steps: int,
    dt_s: float,
    decay_rate: float,
    output_json: bool,
) -> None:
    """Simulate Lyapunov-stable closed-loop detachment score control.

    Runs the feedback controller for STEPS timesteps and shows convergence
    of the detachment score S toward the target setpoint S*.

    Stability proof: V = (S−S*)²/2, dV/dt = −2αV < 0 (exponential decay).
    """
    from helicon.detach import LyapunovController, PlasmaState
    from helicon.detach.invariants import species_mass

    state = PlasmaState(
        n_m3=n_m3,
        Te_eV=Te_eV,
        Ti_eV=Ti_eV,
        B_T=B_T,
        dBdz_T_per_m=dBdz,
        vz_ms=vz_ms,
        mass_amu=species_mass(species),
    )
    ctrl = LyapunovController(setpoint=setpoint, decay_rate=decay_rate)
    updates = ctrl.simulate(state, n_steps=steps, dt_s=dt_s)

    if output_json:
        trace = [
            {
                "step": i + 1,
                "score": u.score,
                "error": u.error,
                "V": u.lyapunov_V,
                "dV_dt": u.lyapunov_dV_dt,
                "delta_I_A": u.delta_I_coil_A,
                "I_coil_A": u.new_I_coil_A,
            }
            for i, u in enumerate(updates)
        ]
        click.echo(json.dumps({"setpoint": setpoint, "trace": trace}, indent=2))
    else:
        cert = ctrl.stability_certificate(state)
        click.echo(
            f"  Lyapunov controller  S* = {setpoint:.2f}  "
            f"α = {decay_rate:.2f} s⁻¹  τ = {cert['convergence_time_s']:.2f} s"
        )
        click.echo(f"  {'step':>4}  {'score':>7}  {'error':>8}  {'V':>10}  {'ΔI [A]':>10}")
        click.echo("  " + "-" * 46)
        for i, u in enumerate(updates):
            click.echo(
                f"  {i + 1:>4}  {u.score:>7.4f}  {u.error:>+8.4f}"
                f"  {u.lyapunov_V:>10.2e}  {u.delta_I_coil_A:>+10.1f}"
            )
        click.echo(f"\n  Stable: {cert['is_stable']}  grad_S_B = {cert['grad_S_B']:.3e} T⁻¹")


# -- detach report ------------------------------------------------------------


@detach_group.command("report")
@click.option("--n", "n_m3", type=float, required=True, help="Plasma density [m^-3]")
@click.option("--Te", "Te_eV", type=float, required=True, help="Electron temperature [eV]")
@click.option("--Ti", "Ti_eV", type=float, required=True, help="Ion temperature [eV]")
@click.option("--B", "B_T", type=float, required=True, help="Magnetic field [T]")
@click.option(
    "--dBdz", "dBdz", type=float, default=-1.0, show_default=True, help="B gradient [T/m]"
)
@click.option("--vz", "vz_ms", type=float, required=True, help="Axial bulk velocity [m/s]")
@click.option("--species", type=_SPECIES_CHOICES, default="H+", show_default=True)
@click.option(
    "--coupling", default=0.30, show_default=True, help="Sheath coupling factor ξ ∈ [0,1]"
)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output")
def detach_report_cmd(
    n_m3: float,
    Te_eV: float,
    Ti_eV: float,
    B_T: float,
    dBdz: float,
    vz_ms: float,
    species: str,
    coupling: float,
    output_json: bool,
) -> None:
    """Full combined diagnostic: MHD + FLR kinetic + sheath + Lyapunov cert.

    Reports four layers of analysis in one call:

    \b
    1. MHD assessment (M_A, β_e, Λᵢ, score)
    2. FLR kinetic correction (Northrop 2nd-order Λ_FLR, kinetic M_kAW)
    3. Sheath coupling correction (Debye length, ε_ES, corrected score)
    4. Lyapunov stability certificate (V, dV/dt, convergence time)
    """
    from helicon.detach import (
        DetachmentOnsetModel,
        LyapunovController,
        PlasmaState,
        apply_sheath_correction,
    )
    from helicon.detach.invariants import species_mass
    from helicon.detach.kinetic import alfven_mach_kinetic, ion_magnetization_flr

    mass = species_mass(species)
    state = PlasmaState(
        n_m3=n_m3,
        Te_eV=Te_eV,
        Ti_eV=Ti_eV,
        B_T=B_T,
        dBdz_T_per_m=dBdz,
        vz_ms=vz_ms,
        mass_amu=mass,
    )

    # 1. MHD
    model = DetachmentOnsetModel()
    ds = model.assess(state)

    # 2. Kinetic FLR
    lambda_flr = ion_magnetization_flr(Ti_eV, B_T, dBdz, mass)
    m_kaw = alfven_mach_kinetic(vz_ms, B_T, n_m3, mass, Ti_eV, dBdz)

    # 3. Sheath
    sheath = apply_sheath_correction(
        score_raw=ds.detachment_score,
        n_m3=n_m3,
        Te_eV=Te_eV,
        Ti_eV=Ti_eV,
        B_T=B_T,
        dBdz_T_per_m=dBdz,
        mass_amu=mass,
        coupling_factor=coupling,
    )

    # 4. Lyapunov
    ctrl = LyapunovController(model=model)
    cert = ctrl.stability_certificate(state)

    if output_json:
        click.echo(
            json.dumps(
                {
                    "mhd": ds.to_dict(),
                    "kinetic": {
                        "lambda_i_flr": lambda_flr if lambda_flr != float("inf") else "inf",
                        "alfven_mach_kinetic": m_kaw,
                    },
                    "sheath": {
                        "debye_length_m": sheath.debye_length_m,
                        "sheath_potential_V": sheath.sheath_potential_V,
                        "epsilon_ES": sheath.epsilon_ES,
                        "score_corrected": sheath.score_corrected,
                        "correction_fraction": sheath.correction_fraction,
                    },
                    "lyapunov": cert,
                },
                indent=2,
            )
        )
    else:
        click.echo("── MHD Assessment ─────────────────────────────────────────")
        click.echo(ds.summary())
        click.echo("\n── Kinetic FLR Corrections ─────────────────────────────────")
        lf = f"{lambda_flr:.4f}" if lambda_flr != float("inf") else "inf"
        click.echo(f"  Λᵢ_FLR (Northrop 2nd-order) = {lf}")
        click.echo(f"  M_kAW  (kinetic Alfvén)      = {m_kaw:.4f}")
        click.echo("\n── Sheath Coupling Correction ──────────────────────────────")
        click.echo(f"  Debye length  λ_D = {sheath.debye_length_m:.3e} m")
        click.echo(f"  Sheath potential Φ_s = {sheath.sheath_potential_V:.2f} V")
        click.echo(f"  ε_ES (electric/mirror) = {sheath.epsilon_ES:.4f}")
        click.echo(
            f"  Score: {sheath.score_raw:.4f} → {sheath.score_corrected:.4f} "
            f"(−{sheath.correction_fraction:.1%})"
        )
        click.echo("\n── Lyapunov Stability Certificate ──────────────────────────")
        click.echo(f"  V = {cert['V']:.4e}   dV/dt = {cert['dV_dt']:.4e}")
        click.echo(
            f"  Stable: {cert['is_stable']}   τ_conv = {cert['convergence_time_s']:.3f} s"
        )


# ---------------------------------------------------------------------------
# helicon mf — multi-fidelity pipeline
# ---------------------------------------------------------------------------


@main.group("mf")
def mf_group() -> None:
    """Multi-fidelity optimisation pipeline (Tier 1 → 2 → 3)."""


@mf_group.command("report")
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--top", default=10, show_default=True, help="Number of top candidates to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def mf_report(output_dir: str, top: int, as_json: bool) -> None:
    """Summarise tier3 candidates from a multi-fidelity scan output directory."""
    import json as _json

    candidates = []
    for meta_path in Path(output_dir).glob("tier3_*/tier3_meta.json"):
        try:
            data = _json.loads(meta_path.read_text())
            data["_dir"] = meta_path.parent.name
            candidates.append(data)
        except Exception:
            continue

    if not candidates:
        click.echo(f"No tier3_meta.json files found under {output_dir}", err=True)
        sys.exit(1)

    candidates.sort(key=lambda c: c.get("tier2_score", 0.0), reverse=True)
    shown = candidates[:top]

    if as_json:
        click.echo(_json.dumps({"n_total": len(candidates), "top": shown}, indent=2))
        return

    click.echo(f"Multi-fidelity report: {output_dir}")
    click.echo(f"Total tier3 candidates: {len(candidates)}")
    statuses = {}
    for c in candidates:
        s = c.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    for s, n in sorted(statuses.items()):
        click.echo(f"  {s}: {n}")
    click.echo(f"\nTop {len(shown)} by tier2_score:")
    click.echo(f"  {'Candidate':<28}  {'Score':>6}  {'η_d':>6}  {'Thrust/N':>10}  Status")
    click.echo("  " + "-" * 60)
    for c in shown:
        m = c.get("tier2_metrics", {})
        eta = m.get("eta_d", float("nan"))
        thrust = m.get("thrust_N", float("nan"))
        score = c.get("tier2_score", float("nan"))
        status = c.get("status", "?")
        cid = c.get("candidate_id", c.get("_dir", "?"))
        click.echo(f"  {cid:<28}  {score:6.4f}  {eta:6.4f}  {thrust:10.3e}  {status}")


@mf_group.command("run")
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option(
    "--vary",
    "vary_specs",
    multiple=True,
    metavar="PATH:LOW:HIGH:N",
    required=True,
    help="Parameter range, e.g. coils.0.I:20000:80000:5",
)
@click.option(
    "--output", "output_dir", default="mf_output", show_default=True, type=click.Path()
)
@click.option(
    "--n-tier1", default=500, show_default=True, help="Tier-1 analytical evaluations"
)
@click.option(
    "--tier2-threshold", default=0.5, show_default=True, help="Tier-1 → 2 score cutoff"
)
@click.option(
    "--tier3-threshold", default=0.6, show_default=True, help="Tier-2 → 3 score cutoff"
)
@click.option("--top-k", default=3, show_default=True, help="Candidates to promote to Tier 3")
@click.option(
    "--objective",
    default="thrust_N",
    show_default=True,
    type=click.Choice(["thrust_N", "eta_d"]),
    help="Metric to maximise",
)
@click.option("--dry-run", is_flag=True, help="Generate Tier-3 inputs but do not run WarpX")
@click.option("--seed", default=0, show_default=True, help="RNG seed for LHC sampling")
@click.option("--json", "as_json", is_flag=True, help="Output result summary as JSON")
def mf_run(
    config_path: str,
    vary_specs: tuple[str, ...],
    output_dir: str,
    n_tier1: int,
    tier2_threshold: float,
    tier3_threshold: float,
    top_k: int,
    objective: str,
    dry_run: bool,
    seed: int,
    as_json: bool,
) -> None:
    """Run the multi-fidelity pipeline: analytical → surrogate → WarpX PIC."""
    import json as _json

    import numpy as np

    from helicon.optimize.multifidelity import FidelityConfig, MultiFidelityPipeline

    # Parse vary specs into LHC candidates
    ranges = []
    for spec in vary_specs:
        parts = spec.split(":")
        if len(parts) != 4:
            click.echo(f"Bad --vary spec {spec!r}: expected PATH:LOW:HIGH:N", err=True)
            sys.exit(1)
        path, lo, hi, n = parts[0], float(parts[1]), float(parts[2]), int(parts[3])
        ranges.append((path, lo, hi, n))

    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n_tier1):
        c: dict = {}
        for path, lo, hi, _n in ranges:
            key = path.replace(".", "_").replace("coils_0_", "coil_")
            c[key] = float(rng.uniform(lo, hi))
        candidates.append(c)

    cfg = FidelityConfig(
        tier2_threshold=tier2_threshold,
        tier3_threshold=tier3_threshold,
        top_k_to_tier3=top_k,
        tier1_n_eval=n_tier1,
        dry_run_tier3=dry_run,
    )
    pipeline = MultiFidelityPipeline(fidelity_config=cfg, output_dir=output_dir)

    if not as_json:
        click.echo(
            f"Multi-fidelity run: {n_tier1} Tier-1 candidates, "
            f"objective={objective}, top_k={top_k}, dry_run={dry_run}"
        )
    result = pipeline.run(candidates, objective=objective)

    summary = result.to_dict()
    if as_json:
        click.echo(_json.dumps(summary, indent=2))
    else:
        click.echo(f"Tier-1 evaluated:  {summary['n_tier1']}")
        click.echo(f"Tier-2 evaluated:  {summary['n_tier2']}")
        click.echo(f"Tier-3 promoted:   {summary['n_tier3']}")
        click.echo(f"Best candidate:    {summary['best_candidate_id']}")
        click.echo(f"Wall time:         {summary['total_wall_time_s']:.1f} s")
        if summary["best_metrics"]:
            m = summary["best_metrics"]
            click.echo(f"Best η_d:          {m.get('eta_d', float('nan')):.4f}")
            click.echo(f"Best thrust:       {m.get('thrust_N', float('nan')):.3e} N")
        click.echo(f"Tier-3 outputs in: {output_dir}/")


@mf_group.command("promote")
@click.argument("candidate_dir", type=click.Path(exists=True))
@click.option(
    "--output", "output_dir", default=None, type=click.Path(), help="Override output dir"
)
@click.option("--json", "as_json", is_flag=True, help="Output result as JSON")
def mf_promote(candidate_dir: str, output_dir: str | None, as_json: bool) -> None:
    """Promote a single dry-run tier3 candidate to a full WarpX PIC run.

    CANDIDATE_DIR is a tier3_* subdirectory produced by `helicon mf run`.
    Reads tier3_meta.json, builds a SimConfig, and dispatches run_simulation().
    """
    import json as _json

    from helicon.optimize.multifidelity import _candidate_to_config
    from helicon.runner.launch import run_simulation

    meta_path = Path(candidate_dir) / "tier3_meta.json"
    if not meta_path.exists():
        click.echo(f"No tier3_meta.json found in {candidate_dir}", err=True)
        sys.exit(1)

    meta = _json.loads(meta_path.read_text())
    cid = meta.get("candidate_id", Path(candidate_dir).name)

    if meta.get("status") == "completed":
        click.echo(f"Candidate {cid} already completed. Re-running anyway.")

    out = Path(output_dir) if output_dir else Path(candidate_dir) / "pic_run"
    metrics = meta.get("tier2_metrics", {})

    if not as_json:
        click.echo(f"Promoting {cid} to full WarpX PIC → {out}")

    try:
        config = _candidate_to_config(metrics, output_dir=str(out))
        run_result = run_simulation(config, output_dir=out, dry_run=False)
        meta["status"] = "completed"
        meta["wall_time_s"] = run_result.wall_time_s
        meta["pic_output_dir"] = str(out)
        meta_path.write_text(_json.dumps(meta, indent=2))

        if as_json:
            click.echo(
                _json.dumps(
                    {
                        "candidate_id": cid,
                        "status": "completed",
                        "wall_time_s": run_result.wall_time_s,
                        "pic_output_dir": str(out),
                    },
                    indent=2,
                )
            )
        else:
            click.echo(f"Completed in {run_result.wall_time_s:.1f} s → {out}")
    except Exception as exc:
        meta["status"] = f"error: {exc}"
        meta_path.write_text(_json.dumps(meta, indent=2))
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

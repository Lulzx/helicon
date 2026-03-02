"""Helicon — GPU-Accelerated Magnetic Nozzle Simulation Toolkit."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

__version__ = "1.1.0"


# ---------------------------------------------------------------------------
# Config alias
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    if name == "Config":
        from helicon.config.parser import SimConfig

        return SimConfig
    if name == "fields":
        from helicon import fields as _fields_mod

        return _fields_mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Lazy Config alias at module level for direct attribute access
# ---------------------------------------------------------------------------


class _LazyConfig:
    """Proxy so ``mnx.Config`` resolves to ``SimConfig`` on first access."""

    def __repr__(self) -> str:
        from helicon.config.parser import SimConfig

        return repr(SimConfig)

    def __call__(self, *args, **kwargs):
        from helicon.config.parser import SimConfig

        return SimConfig(*args, **kwargs)

    def from_yaml(self, path):
        from helicon.config.parser import SimConfig

        return SimConfig.from_yaml(path)

    def from_preset(self, name):
        from helicon.config.parser import SimConfig

        return SimConfig.from_preset(name)

    def model_validate(self, data):
        from helicon.config.parser import SimConfig

        return SimConfig.model_validate(data)


Config = _LazyConfig()


# ---------------------------------------------------------------------------
# _FieldsNamespace
# ---------------------------------------------------------------------------


class _FieldsNamespace:
    """Namespace object exposing ``fields.compute(...)``."""

    def compute(self, nozzle_or_coils, grid=None, **kwargs):
        """Compute the magnetic field for a nozzle config or a list of coils.

        Parameters
        ----------
        nozzle_or_coils : NozzleConfig | list[Coil]
            Either a ``NozzleConfig`` (has ``.coils``, ``.domain``, ``.resolution``)
            or a list of ``Coil`` objects.
        grid : Grid, optional
            Required when *nozzle_or_coils* is a list of ``Coil`` objects.

        Returns
        -------
        BField
        """
        from helicon.fields.biot_savart import Coil, Grid, compute_bfield

        # Detect whether we received a NozzleConfig
        if (
            hasattr(nozzle_or_coils, "coils")
            and hasattr(nozzle_or_coils, "domain")
            and hasattr(nozzle_or_coils, "resolution")
        ):
            nozzle = nozzle_or_coils
            coils = [Coil(z=c.z, r=c.r, I=c.I) for c in nozzle.coils]
            grid = Grid(
                z_min=nozzle.domain.z_min,
                z_max=nozzle.domain.z_max,
                r_max=nozzle.domain.r_max,
                nz=nozzle.resolution.nz,
                nr=nozzle.resolution.nr,
            )
        else:
            # Assume list of Coil objects
            coils = nozzle_or_coils
            if grid is None:
                raise TypeError(
                    "grid is required when nozzle_or_coils is a list of Coil objects"
                )

        return compute_bfield(coils, grid, **kwargs)


fields = _FieldsNamespace()


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def run(config, output_dir=None, dry_run=False, **kwargs):
    """Run a Helicon simulation.

    Thin wrapper around :func:`helicon.runner.launch.run_simulation`.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration.
    output_dir : path-like, optional
        Override output directory (defaults to ``config.output_dir``).
    dry_run : bool
        Generate input files without launching WarpX.

    Returns
    -------
    RunResult
    """
    from helicon.runner.launch import run_simulation

    return run_simulation(config, output_dir=output_dir, dry_run=dry_run, **kwargs)


# ---------------------------------------------------------------------------
# Metrics / DetachmentMetrics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DetachmentMetrics:
    """Detachment efficiency metrics."""

    momentum: float | None
    particle: float | None
    energy: float | None


@dataclass
class Metrics:
    """Top-level postprocessing metrics."""

    thrust: float | None
    isp: float | None
    exhaust_velocity_ms: float | None
    mass_flow_rate_kgs: float | None
    plume_angle_deg: float | None
    beam_efficiency: float | None
    radial_loss_fraction: float | None
    detachment: DetachmentMetrics
    config_hash: str | None


# ---------------------------------------------------------------------------
# postprocess()
# ---------------------------------------------------------------------------


def postprocess(output_dir) -> Metrics:
    """Generate postprocessing metrics from a simulation output directory.

    Calls :func:`helicon.postprocess.report.generate_report` and wraps
    the result into a :class:`Metrics` object.

    Parameters
    ----------
    output_dir : path-like
        Directory containing WarpX/Helicon output files.

    Returns
    -------
    Metrics
    """
    import json
    from pathlib import Path

    from helicon.postprocess.report import generate_report

    # Read config_hash from run_metadata.json if present
    config_hash = None
    meta_path = Path(output_dir) / "run_metadata.json"
    with contextlib.suppress(Exception):
        meta = json.loads(meta_path.read_text())
        config_hash = meta.get("config_hash")

    report = generate_report(output_dir, config_hash=config_hash)

    return Metrics(
        thrust=report.thrust_N,
        isp=report.isp_s,
        exhaust_velocity_ms=report.exhaust_velocity_ms,
        mass_flow_rate_kgs=report.mass_flow_rate_kgs,
        plume_angle_deg=report.plume_half_angle_deg,
        beam_efficiency=report.beam_efficiency,
        radial_loss_fraction=report.radial_loss_fraction,
        detachment=DetachmentMetrics(
            momentum=report.detachment_momentum,
            particle=report.detachment_particle,
            energy=report.detachment_energy,
        ),
        config_hash=report.config_hash,
    )


# ---------------------------------------------------------------------------
# scan()
# ---------------------------------------------------------------------------


def scan(
    config,
    vary: dict,
    objectives=None,
    method: str = "lhc",
    dry_run: bool = False,
    output_base: str = "scan_results",
    seed: int = 0,
):
    """Run a parameter scan over the given config.

    Parameters
    ----------
    config : SimConfig
        Base configuration.
    vary : dict[str, tuple[float, float, int]]
        Mapping of dot-notation parameter path to ``(low, high, n)`` tuples.
    objectives : list[str], optional
        Objective metric names to track (stored on the returned result).
    method : ``"lhc"`` | ``"grid"``
        Sampling strategy.
    dry_run : bool
        Generate configs and B-fields without launching WarpX.
    output_base : str
        Root output directory for scan point subdirectories.
    seed : int
        RNG seed for LHC sampling.

    Returns
    -------
    ScanResult
        Scan result with ``objectives`` attribute set if supported.
    """
    from helicon.optimize.scan import ParameterRange, run_scan

    ranges = [
        ParameterRange(path=path, low=float(low), high=float(high), n=int(n))
        for path, (low, high, n) in vary.items()
    ]

    result = run_scan(
        config,
        ranges,
        output_base=output_base,
        method=method,
        dry_run=dry_run,
        seed=seed,
    )

    # Attach objectives to the result if possible
    try:
        object.__setattr__(result, "objectives", objectives)
    except (AttributeError, TypeError):
        with contextlib.suppress(AttributeError):
            result.objectives = objectives

    return result

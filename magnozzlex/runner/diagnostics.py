"""WarpX diagnostic scheduling configuration.

Provides preset diagnostic configurations for different run modes
(analysis vs scan) and custom scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

from magnozzlex.config.parser import DiagnosticsConfig


@dataclass
class DiagnosticSchedule:
    """Resolved diagnostic output schedule."""

    field_intervals: int
    particle_intervals: int | None
    fields_to_plot: list[str]
    species_to_dump: list[str]
    estimated_size_gb: float


def resolve_schedule(
    diag_config: DiagnosticsConfig,
    timesteps: int,
    nz: int,
    nr: int,
    n_species: int,
) -> DiagnosticSchedule:
    """Compute a diagnostic schedule and estimate output size.

    Parameters
    ----------
    diag_config : DiagnosticsConfig
        User diagnostic settings.
    timesteps : int
        Total simulation timesteps.
    nz, nr : int
        Grid dimensions.
    n_species : int
        Number of particle species.
    """
    fields = ["Br", "Bz", "Er", "Ez", "jr", "jz", "rho"]

    if diag_config.mode == "scan":
        # Scan mode: minimal output
        particle_intervals = None
        fields_to_plot = ["Bz", "rho"]
        n_field_dumps = timesteps // diag_config.field_dump_interval
        # ~4 bytes per float32, 2 fields
        size_gb = n_field_dumps * nz * nr * 2 * 4 / 1e9
    else:
        # Analysis mode: full output
        particle_intervals = diag_config.particle_dump_interval
        fields_to_plot = fields
        n_field_dumps = timesteps // diag_config.field_dump_interval
        n_part_dumps = timesteps // diag_config.particle_dump_interval
        # Fields: 7 fields * 4 bytes
        field_size = n_field_dumps * nz * nr * len(fields) * 4 / 1e9
        # Particles: rough estimate 10M particles * 7 floats * 4 bytes
        part_size = n_part_dumps * 10e6 * 7 * 4 / 1e9
        size_gb = field_size + part_size

    return DiagnosticSchedule(
        field_intervals=diag_config.field_dump_interval,
        particle_intervals=particle_intervals,
        fields_to_plot=fields_to_plot,
        species_to_dump=[f"species_{i}" for i in range(n_species)],
        estimated_size_gb=round(size_gb, 2),
    )

"""Magnetic field computation module."""

from __future__ import annotations

from helicon.fields.biot_savart import BField, Coil, Grid, compute_bfield
from helicon.fields.cache import FieldCache, compute_bfield_cached
from helicon.fields.field_lines import (
    FieldLine,
    FieldLineSet,
    FieldLineType,
    compute_flux_function,
    trace_field_line,
    trace_field_lines,
)
from helicon.fields.frc_topology import FRCTopologyResult, find_frc_topology
from helicon.fields.import_external import (
    load_comsol_bfield,
    load_csv_bfield,
    load_femm_bfield,
)


def compute(nozzle_or_coils, grid=None, **kwargs) -> BField:
    """Compute the magnetic field for a nozzle config or a list of coils.

    Convenience wrapper used by the top-level ``helicon.fields.compute()``
    API (spec §6.2).

    Parameters
    ----------
    nozzle_or_coils : NozzleConfig | list[Coil]
        Either a ``NozzleConfig`` (has ``.coils``, ``.domain``, ``.resolution``)
        or a list of :class:`Coil` objects.
    grid : Grid, optional
        Required when *nozzle_or_coils* is a list of ``Coil`` objects.

    Returns
    -------
    BField
    """
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
        coils = nozzle_or_coils
        if grid is None:
            raise TypeError("grid is required when nozzle_or_coils is a list of Coil objects")
    return compute_bfield(coils, grid, **kwargs)


__all__ = [
    "BField",
    "Coil",
    "FRCTopologyResult",
    "FieldCache",
    "FieldLine",
    "FieldLineSet",
    "FieldLineType",
    "Grid",
    "compute",
    "compute_bfield",
    "compute_bfield_cached",
    "compute_flux_function",
    "find_frc_topology",
    "load_comsol_bfield",
    "load_csv_bfield",
    "load_femm_bfield",
    "trace_field_line",
    "trace_field_lines",
]

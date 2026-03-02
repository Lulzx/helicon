"""CAD export: coil geometry → STEP / IGES (v2.0).

Writes minimal ASCII STEP (AP203) and IGES files representing each coil as
a torus centred on the nozzle axis.  The output can be imported into FreeCAD,
SolidWorks, Catia, or any STEP-compliant mechanical CAD tool.

Note: This is a geometry-only export.  No material properties, tolerances,
or winding details are encoded — those should be added in the CAD tool.

Usage::

    from helicon.config.parser import SimConfig
    from helicon.export.cad import export_coils_step

    config = SimConfig.from_yaml("my_nozzle.yaml")
    export_coils_step(config, "coils.step")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Minimal ASCII STEP (AP203) writer
# ---------------------------------------------------------------------------

_STEP_HEADER = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Helicon magnetic nozzle coil geometry'),'2;1');
FILE_NAME('{filename}','2026-01-01T00:00:00',('Helicon v2.0'),('Helicon'),
  'Helicon magnetic nozzle toolkit','','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
"""

_STEP_FOOTER = """\
ENDSEC;
END-ISO-10303-21;
"""


def _step_circle(
    entity_id: int,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    normal_x: float = 0.0,
    normal_y: float = 0.0,
    normal_z: float = 1.0,
) -> tuple[int, str]:
    """Write a STEP CIRCLE entity (wire-frame torus cross-section).

    Returns (next_entity_id, step_text).
    """
    eid = entity_id
    lines = []

    # Cartesian point for centre
    lines.append(f"#{eid} = CARTESIAN_POINT('',('{cx}','{cy}','{cz}'));")
    pt_id = eid
    eid += 1

    # Direction for axis
    lines.append(f"#{eid} = DIRECTION('',({normal_x},{normal_y},{normal_z}));")
    dir_id = eid
    eid += 1

    # Axis placement
    lines.append(f"#{eid} = AXIS2_PLACEMENT_3D('',#{pt_id},#{dir_id},$);")
    ax_id = eid
    eid += 1

    # Circle
    lines.append(f"#{eid} = CIRCLE('',#{ax_id},{radius});")
    eid += 1

    return eid, "\n".join(lines)


def export_coils_step(
    config: Any,
    path: str | Path,
) -> Path:
    """Export coil geometry as a minimal ASCII STEP (AP203) file.

    Each coil is represented as a circle (wire-frame torus cross-section)
    centred at (z_coil, 0, 0) with the given radius.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration containing ``nozzle.coils``.
    path : path-like
        Output ``.step`` file path.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coils = config.nozzle.coils

    lines = [_STEP_HEADER.format(filename=path.name)]

    eid = 1
    for i, coil in enumerate(coils):
        z = float(coil.z)
        r = float(coil.r)
        I = float(coil.I)
        # Add a comment-like label as description
        lines.append(f"/* Coil {i}: z={z:.4f} m, r={r:.4f} m, I={I:.1f} A */")
        eid, circ_text = _step_circle(eid, z, 0.0, 0.0, r)
        lines.append(circ_text)
        lines.append("")

    lines.append(_STEP_FOOTER)
    path.write_text("\n".join(lines))
    return path.resolve()


# ---------------------------------------------------------------------------
# Minimal IGES writer
# ---------------------------------------------------------------------------


def export_coils_iges(
    config: Any,
    path: str | Path,
) -> Path:
    """Export coil geometry as a minimal IGES file.

    Each coil is represented as a circular arc (entity type 100) in
    the XZ plane (y=0), which most CAD tools revolve to produce a torus.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration containing ``nozzle.coils``.
    path : path-like
        Output ``.igs`` file path.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coils = config.nozzle.coils

    global_lines = []
    directory_lines = []
    parameter_lines = []

    # Global section (IGES format requires fixed-width 80-char lines)
    g_line = (
        "1H,,1H;,7Hcoils  ,"
        "                                                                            "
        ",Helicon v2.0,,8,0.00000001,15,0.0001,6,6HHelicon,1.0,2,2HMM;             "
        "G      1"
    )
    global_lines.append(g_line)

    de_seq = 1
    pd_seq = 1

    for i, coil in enumerate(coils):
        z = float(coil.z)
        r = float(coil.r)
        # Entity type 100 = Circular Arc
        # Parameters: z_plane, x1, y1, x2, y2, x3, y3
        # Full circle: start = end = (z+r, 0), centre = (z, 0) in XZ plane
        param_str = f"100,{z * 1000:.6f},{(z + r) * 1000:.6f},0.0,{(z + r) * 1000:.6f},0.0;"
        param_line = f"{param_str:<72}     {de_seq}P {pd_seq:>6}".ljust(80)
        parameter_lines.append(param_line)

        de_line1 = (
            f"     100       {pd_seq}       0       1       0       0       0"
            f"        0D {de_seq:>6}"
        ).ljust(80)
        de_line2 = (
            f"     100       0       0       1       0"
            f"Coil{i:02d}                                 D {de_seq + 1:>6}"
        ).ljust(80)
        directory_lines.extend([de_line1, de_line2])

        de_seq += 2
        pd_seq += 1

    content = []
    # Start section
    content.append("Helicon magnetic nozzle coil geometry".ljust(72) + "S      1")
    # Global
    for line in global_lines:
        content.append(line)
    # Directory entry
    for line in directory_lines:
        content.append(line)
    # Parameter data
    for line in parameter_lines:
        content.append(line)
    # Terminate
    n_s = 1
    n_g = len(global_lines)
    n_d = len(directory_lines)
    n_p = len(parameter_lines)
    content.append(f"S{n_s:7d}G{n_g:7d}D{n_d:7d}P{n_p:7d}".ljust(72) + "T      1")

    path.write_text("\n".join(content) + "\n")
    return path.resolve()


def coil_geometry_report(config: Any) -> dict[str, Any]:
    """Return a human-readable geometry report for all coils.

    Parameters
    ----------
    config : SimConfig

    Returns
    -------
    dict
        ``"coils"`` list with per-coil geometry, plus ``"summary"`` stats.
    """
    coils = config.nozzle.coils
    coil_list = []
    total_amp_turns = 0.0
    for i, coil in enumerate(coils):
        z = float(coil.z)
        r = float(coil.r)
        I = float(coil.I)
        circumference = 2 * math.pi * r
        total_amp_turns += abs(I)
        coil_list.append(
            {
                "index": i,
                "z_m": z,
                "r_m": r,
                "I_A": I,
                "circumference_m": circumference,
            }
        )

    z_positions = [c["z_m"] for c in coil_list]
    return {
        "coils": coil_list,
        "summary": {
            "n_coils": len(coil_list),
            "z_span_m": max(z_positions) - min(z_positions) if len(z_positions) > 1 else 0.0,
            "total_amp_turns_A": total_amp_turns,
            "max_radius_m": max(c["r_m"] for c in coil_list),
        },
    }

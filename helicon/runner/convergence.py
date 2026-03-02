"""Grid convergence study automation.

Runs the same simulation configuration at multiple grid resolutions and
computes Richardson extrapolation to estimate the grid-converged value
and formal convergence order.

Typical usage::

    result = run_convergence_study(
        config,
        resolutions=[(128, 64), (256, 128), (512, 256)],
        output_base="convergence_study/",
        dry_run=True,
    )
    print(f"Convergence order: {result.convergence_order:.2f}")
    print(f"Extrapolated thrust: {result.extrapolated_thrust_N:.4f} N")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from helicon.config.parser import (
    ResolutionConfig,
    SimConfig,
)


@dataclass
class ConvergenceLevel:
    """Results at a single grid resolution."""

    nz: int
    nr: int
    h: float  # representative mesh spacing = 1/sqrt(nz*nr)
    output_dir: Path
    success: bool
    thrust_N: float | None


@dataclass
class ConvergenceResult:
    """Full grid convergence study result.

    Attributes
    ----------
    levels : list of ConvergenceLevel
        One entry per resolution, ordered coarse → fine.
    convergence_order : float or None
        Richardson-estimated formal convergence order p.
    extrapolated_thrust_N : float or None
        Richardson-extrapolated (grid-converged) thrust estimate.
    converged : bool
        True if the relative change between the two finest levels is
        below ``tol``.
    """

    levels: list[ConvergenceLevel]
    convergence_order: float | None
    extrapolated_thrust_N: float | None
    converged: bool


def _modified_config(base: SimConfig, nz: int, nr: int) -> SimConfig:
    """Return a copy of base with resolution overridden."""
    data = base.model_dump(mode="python")
    data["nozzle"]["resolution"] = {"nz": nz, "nr": nr}
    return SimConfig.model_validate(data)


def richardson_extrapolate(
    values: list[float],
    h_ratios: list[float],
) -> tuple[float, float]:
    """Richardson extrapolation from three values at doubling mesh spacings.

    Given values Q_1 (coarsest), Q_2, Q_3 (finest) at spacings
    h_1 = r*h_2, h_2 = r*h_3 (ratio r ≈ 2):

        p = log((Q_1 − Q_2) / (Q_2 − Q_3)) / log(r)
        Q_exact ≈ Q_3 + (Q_3 − Q_2) / (r^p − 1)

    Parameters
    ----------
    values : list of float
        Three scalar metric values at coarse → fine resolution.
    h_ratios : list of float
        Refinement ratios: ``[h_coarse/h_mid, h_mid/h_fine]``.

    Returns
    -------
    order : float
        Estimated convergence order p.
    extrapolated : float
        Richardson-extrapolated grid-converged estimate.
    """
    Q1, Q2, Q3 = values
    r1, r2 = h_ratios

    d21 = Q2 - Q1
    d32 = Q3 - Q2

    if abs(d32) < 1e-20 or abs(d21) < 1e-20:
        return float("nan"), Q3

    ratio = d21 / d32
    # Use mean refinement ratio
    r = math.sqrt(r1 * r2)
    if ratio <= 0 or r <= 1:
        return float("nan"), Q3

    p = math.log(abs(ratio)) / math.log(r)
    r_p = r**p
    if abs(r_p - 1.0) < 1e-10:
        return p, Q3

    extrapolated = Q3 + (Q3 - Q2) / (r_p - 1.0)
    return p, extrapolated


def run_convergence_study(
    base_config: SimConfig,
    resolutions: list[tuple[int, int]],
    *,
    output_base: str | Path = "convergence_study",
    dry_run: bool = False,
    tol: float = 0.05,
) -> ConvergenceResult:
    """Run a grid convergence study at multiple resolutions.

    Parameters
    ----------
    base_config : SimConfig
        Base simulation configuration; only resolution is varied.
    resolutions : list of (nz, nr) tuples
        Grid resolutions to run, ordered coarse → fine.
        Recommended: three levels with factor-2 refinement,
        e.g. ``[(128, 64), (256, 128), (512, 256)]``.
    output_base : path
        Root output directory; each level gets a subdirectory.
    dry_run : bool
        Generate inputs without launching WarpX.
    tol : float
        Relative convergence tolerance between the two finest levels.

    Returns
    -------
    ConvergenceResult
    """
    from helicon.runner.launch import run_simulation

    output_base = Path(output_base)
    levels: list[ConvergenceLevel] = []

    for nz, nr in resolutions:
        config_i = _modified_config(base_config, nz=nz, nr=nr)
        h_i = 1.0 / math.sqrt(nz * nr)
        level_dir = output_base / f"nz{nz}_nr{nr}"

        run_result = run_simulation(config_i, output_dir=level_dir, dry_run=dry_run)

        thrust_N: float | None = None
        if run_result.success and not dry_run:
            try:
                from helicon.postprocess.thrust import compute_thrust

                t = compute_thrust(level_dir)
                thrust_N = t.thrust_N
            except (FileNotFoundError, ValueError):
                pass

        levels.append(
            ConvergenceLevel(
                nz=nz,
                nr=nr,
                h=h_i,
                output_dir=level_dir,
                success=run_result.success,
                thrust_N=thrust_N,
            )
        )

    # Richardson extrapolation (requires ≥ 3 levels with valid thrust)
    thrust_values = [lv.thrust_N for lv in levels if lv.thrust_N is not None]
    conv_order: float | None = None
    extrap: float | None = None
    converged = False

    if len(thrust_values) >= 3:
        h_vals = [lv.h for lv in levels if lv.thrust_N is not None]
        h_ratios = [h_vals[i] / h_vals[i + 1] for i in range(len(h_vals) - 1)]
        p, extrap = richardson_extrapolate(thrust_values[-3:], h_ratios[-2:])
        conv_order = p if not math.isnan(p) else None

        # Check convergence between two finest levels
        q_fine = thrust_values[-1]
        q_mid = thrust_values[-2]
        if abs(q_fine) > 1e-20:
            converged = abs(q_fine - q_mid) / abs(q_fine) < tol

    elif len(thrust_values) == 2:
        q_fine = thrust_values[-1]
        q_mid = thrust_values[-2]
        if abs(q_fine) > 1e-20:
            converged = abs(q_fine - q_mid) / abs(q_fine) < tol

    return ConvergenceResult(
        levels=levels,
        convergence_order=conv_order,
        extrapolated_thrust_N=extrap,
        converged=converged,
    )

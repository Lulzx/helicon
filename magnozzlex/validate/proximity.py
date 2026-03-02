"""Validation proximity metric for MagNozzleX (§7.3).

Computes the ``validation_proximity`` field — the parameter-space distance
from a user's :class:`~magnozzlex.config.parser.SimConfig` to the nearest
validated case in the suite.

Each validated case owns a *reference parameter region* defined by a centre
value and a half-width for each physical parameter.  A normalised distance of
**< 1.0** in every dimension means the configuration sits inside that region;
a distance of exactly **1.0** places it at the boundary.

Parameters treated:

* ``n0``    — plasma number density [m⁻³], normalised on log₁₀ scale
* ``T_i_eV`` — ion temperature [eV], normalised linearly
* ``B_T``   — estimated on-axis throat magnetic field [T], normalised linearly
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from magnozzlex.config.parser import SimConfig

# Permeability of free space [T·m/A]
_MU_0: float = 4.0e-7 * math.pi

# ---------------------------------------------------------------------------
# Reference parameter regions
# Each entry: {"n0": (centre, half_width), "T_i_eV": (...), "B_T": (...)}
# Units: n0 [m⁻³], T_i_eV [eV], B_T [T]
# ---------------------------------------------------------------------------
VALIDATED_REGIONS: dict[str, dict[str, tuple[float, float]]] = {
    "free_expansion": {
        "n0":    (1e18, 5e17),
        "T_i_eV": (100.0, 50.0),
        "B_T":   (0.10, 0.05),
    },
    "guiding_center": {
        "n0":    (1e16, 5e15),
        "T_i_eV": (10.0, 5.0),
        "B_T":   (0.05, 0.02),
    },
    "merino_ahedo": {
        "n0":    (1e19, 5e18),
        "T_i_eV": (1000.0, 500.0),
        "B_T":   (1.0, 0.5),
    },
    "resistive_dimov": {
        "n0":    (1e21, 5e20),
        "T_i_eV": (10.0, 5.0),
        "B_T":   (0.05, 0.02),
    },
    "vasimr": {
        "n0":    (5e19, 2e19),
        "T_i_eV": (1000.0, 500.0),
        "B_T":   (2.0, 1.0),
    },
    "mn1d_comparison": {
        "n0":    (1e19, 5e18),
        "T_i_eV": (500.0, 200.0),
        "B_T":   (1.0, 0.5),
    },
}

# Parameters present in every validated region
_PARAM_KEYS: tuple[str, ...] = ("n0", "T_i_eV", "B_T")


@dataclass
class ProximityResult:
    """Distance from a :class:`~magnozzlex.config.parser.SimConfig` to the
    nearest validated case in parameter space.

    Attributes
    ----------
    nearest_case:
        Name of the closest validated case.
    distance:
        Normalised L2 distance to the nearest case.
        ``0.0`` = at the region centre; ``1.0`` = at the boundary edge
        (one half-width away in every dimension simultaneously).
    in_validated_region:
        ``True`` when the configuration lies strictly inside the validated
        region of the nearest case (i.e. every per-parameter normalised
        distance is < 1.0).
    parameter_distances:
        Per-parameter normalised distances for the nearest case.
        Keys are ``"n0"``, ``"T_i_eV"``, and ``"B_T"``.
    warning:
        Human-readable warning string when ``distance > 2.0``; ``None``
        otherwise.
    """

    nearest_case: str
    distance: float
    in_validated_region: bool
    parameter_distances: dict[str, float]
    warning: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_b_throat(config: SimConfig) -> float:
    """Estimate the on-axis magnetic field at the throat [T].

    Uses the on-axis field of a thin current loop at its centre:

    .. math::

        B = \\frac{\\mu_0 I}{2 r}

    where *r* is the coil radius and *I* the current.  When multiple coils
    are present the first coil (closest to the throat) is used as the
    representative coil.
    """
    coil = config.nozzle.coils[0]
    return _MU_0 * abs(coil.I) / (2.0 * coil.r)


def _normalised_distance_log(value: float, centre: float, half_width: float) -> float:
    """Normalised log₁₀ distance for a density-type parameter.

    Returns ``|log10(value) - log10(centre)| / log10(centre / (centre - half_width))``
    which gives 0 at the centre and 1 at the lower boundary.

    For robustness the half-width is interpreted symmetrically in log space:

    .. math::

        d = \\frac{|\\log_{10}(x) - \\log_{10}(c)|}{\\delta}

    where :math:`\\delta = \\log_{10}(c) - \\log_{10}(c - w)` and *w* is
    the half-width.  If ``centre - half_width <= 0`` we fall back to using
    ``log10(half_width)`` as the scale.
    """
    if value <= 0.0 or centre <= 0.0:
        return float("inf")
    log_centre = math.log10(centre)
    log_value = math.log10(value)
    lower = centre - half_width
    if lower > 0.0:
        delta = log_centre - math.log10(lower)
    else:
        # Fallback: use log10 of the half-width itself as scale
        delta = math.log10(half_width) if half_width > 0.0 else 1.0
    if delta <= 0.0:
        return 0.0
    return abs(log_value - log_centre) / delta


def _normalised_distance_linear(value: float, centre: float, half_width: float) -> float:
    """Normalised linear distance for a parameter.

    Returns ``|value - centre| / half_width``.
    """
    if half_width <= 0.0:
        return 0.0
    return abs(value - centre) / half_width


def _case_distance(
    n0: float,
    T_i_eV: float,
    B_T: float,
    region: dict[str, tuple[float, float]],
) -> tuple[float, dict[str, float]]:
    """Return (L2 distance, per-parameter distances) for one validated region."""
    d_n0 = _normalised_distance_log(n0, *region["n0"])
    d_T = _normalised_distance_linear(T_i_eV, *region["T_i_eV"])
    d_B = _normalised_distance_linear(B_T, *region["B_T"])

    per_param: dict[str, float] = {
        "n0": d_n0,
        "T_i_eV": d_T,
        "B_T": d_B,
    }
    l2 = math.sqrt(d_n0**2 + d_T**2 + d_B**2)
    return l2, per_param


def _build_warning(
    nearest_case: str,
    distance: float,
    per_param: dict[str, float],
) -> str:
    """Compose a warning string for configurations far from all validated cases."""
    far_params = sorted(
        ((k, v) for k, v in per_param.items() if v > 1.0),
        key=lambda kv: -kv[1],
    )
    param_hints: list[str] = []
    for param, dist in far_params:
        centre, half_width = VALIDATED_REGIONS[nearest_case][param]
        direction = "increase" if dist > 0 else "decrease"
        # Determine whether the user value is above or below the centre
        # We don't have the user value here, but we can describe the region.
        param_hints.append(
            f"{param} (currently {dist:.1f}x outside the validated half-width; "
            f"nearest validated centre is {centre:.3g})"
        )

    if param_hints:
        hints_str = "; ".join(param_hints)
        return (
            f"Config is far from all validated cases "
            f"(nearest: '{nearest_case}', distance={distance:.2f}). "
            f"Consider adjusting: {hints_str}."
        )
    return (
        f"Config is far from all validated cases "
        f"(nearest: '{nearest_case}', distance={distance:.2f})."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def config_proximity(config: SimConfig) -> ProximityResult:
    """Compute parameter-space distance from *config* to the nearest validated case.

    The distance metric uses:

    * **log₁₀ normalisation** for ``n0`` (density spans many orders of magnitude).
    * **Linear normalisation** for ``T_i_eV`` and the estimated throat field ``B_T``.

    A distance < 1.0 means the configuration lies inside the validated
    parameter region; a distance > 2.0 triggers a warning with suggestions.

    Parameters
    ----------
    config:
        The simulation configuration to evaluate.

    Returns
    -------
    ProximityResult
        Contains the nearest validated case name, overall L2 distance,
        whether the config is inside the validated region, per-parameter
        normalised distances, and an optional warning string.
    """
    n0 = config.plasma.n0
    T_i_eV = config.plasma.T_i_eV
    B_T = _estimate_b_throat(config)

    best_case: str = ""
    best_distance: float = float("inf")
    best_per_param: dict[str, float] = {}

    for case_name, region in VALIDATED_REGIONS.items():
        dist, per_param = _case_distance(n0, T_i_eV, B_T, region)
        if dist < best_distance:
            best_distance = dist
            best_case = case_name
            best_per_param = per_param

    # in_validated_region: every per-parameter distance < 1.0
    in_region = all(v < 1.0 for v in best_per_param.values())

    warning: str | None = None
    if best_distance > 2.0:
        warning = _build_warning(best_case, best_distance, best_per_param)

    return ProximityResult(
        nearest_case=best_case,
        distance=best_distance,
        in_validated_region=in_region,
        parameter_distances=best_per_param,
        warning=warning,
    )

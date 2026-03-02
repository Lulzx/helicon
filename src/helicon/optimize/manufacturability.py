"""Coil manufacturability constraints for optimised nozzle designs (v2.0).

Ensures that optimised coil geometries are physically buildable:
  - minimum bend radius (winding pack radius vs. wire/tape diameter)
  - winding pack cross-section feasibility
  - REBCO tape width discretisation (integer number of turns)
  - layer-wound vs. pancake topology recommendation

References
----------
- Vedrine et al. (2016) — HTS coil design guidelines.
- Shen et al. (2011) — REBCO tape conductor properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class REBCOTapeSpec:
    """REBCO (HTS) tape conductor specification.

    Attributes
    ----------
    tape_width_mm : float
        Tape width [mm].  Common values: 4, 6, 8, 10, 12 mm.
    tape_thickness_mm : float
        Total tape stack thickness including substrate [mm].
    critical_current_A : float
        Critical current at 20 K, self-field [A].
    min_bend_radius_mm : float
        Minimum bend radius without degradation [mm].
    """

    tape_width_mm: float = 6.0
    tape_thickness_mm: float = 0.1
    critical_current_A: float = 300.0
    min_bend_radius_mm: float = 15.0


@dataclass
class WindingPackSpec:
    """Winding pack geometry specification.

    Attributes
    ----------
    inner_radius_m : float
        Bore / inner radius of the winding pack [m].
    outer_radius_m : float
        Outer radius of the winding pack [m].
    axial_length_m : float
        Axial extent (coil height) [m].
    n_layers : int
        Number of radial layers.
    n_turns_per_layer : int
        Turns per radial layer.
    topology : str
        ``"pancake"`` or ``"layer_wound"``.
    """

    inner_radius_m: float = 0.08
    outer_radius_m: float = 0.12
    axial_length_m: float = 0.05
    n_layers: int = 10
    n_turns_per_layer: int = 20
    topology: str = "layer_wound"

    @property
    def total_turns(self) -> int:
        return self.n_layers * self.n_turns_per_layer

    @property
    def radial_build_m(self) -> float:
        return self.outer_radius_m - self.inner_radius_m


@dataclass
class ManufacturabilityResult:
    """Result of a coil manufacturability check.

    Attributes
    ----------
    feasible : bool
        True only if all individual checks pass.
    violations : list[str]
        Human-readable descriptions of any constraint violations.
    warnings : list[str]
        Non-blocking advisories.
    recommended_topology : str
        ``"pancake"`` or ``"layer_wound"``.
    n_turns_required : int
        Turns needed to carry the target current.
    n_turns_available : int
        Turns that fit in the specified winding pack.
    bend_radius_ratio : float
        Actual / minimum bend radius (must be >= 1 to be feasible).
    """

    feasible: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_topology: str = "layer_wound"
    n_turns_required: int = 0
    n_turns_available: int = 0
    bend_radius_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "violations": self.violations,
            "warnings": self.warnings,
            "recommended_topology": self.recommended_topology,
            "n_turns_required": self.n_turns_required,
            "n_turns_available": self.n_turns_available,
            "bend_radius_ratio": self.bend_radius_ratio,
        }


def check_manufacturability(
    coil_radius_m: float,
    coil_current_A: float,
    winding: WindingPackSpec,
    tape: REBCOTapeSpec | None = None,
) -> ManufacturabilityResult:
    """Check if a coil design is physically manufacturable.

    Parameters
    ----------
    coil_radius_m : float
        Mean coil radius (used as inner radius lower bound) [m].
    coil_current_A : float
        Total coil current [A].
    winding : WindingPackSpec
        Winding pack geometry.
    tape : REBCOTapeSpec, optional
        HTS tape specification.  Defaults to standard 6 mm REBCO.

    Returns
    -------
    ManufacturabilityResult
    """
    if tape is None:
        tape = REBCOTapeSpec()

    violations: list[str] = []
    warnings: list[str] = []

    # 1. Bend radius check
    actual_bend_mm = coil_radius_m * 1000.0
    bend_ratio = actual_bend_mm / tape.min_bend_radius_mm
    if bend_ratio < 1.0:
        violations.append(
            f"Bend radius {actual_bend_mm:.1f} mm < minimum {tape.min_bend_radius_mm:.1f} mm "
            f"(ratio={bend_ratio:.2f})"
        )

    # 2. Turns required to reach target current
    n_turns_req = max(1, int(coil_current_A / tape.critical_current_A) + 1)

    # 3. Turns available in winding pack (REBCO tape width discretisation)
    #    Axial direction: n_turns_per_layer limited by axial length / tape_width
    n_axial = max(1, int(winding.axial_length_m * 1000.0 / tape.tape_width_mm))
    #    Radial direction: n_layers limited by radial build / tape_thickness
    n_radial = max(1, int(winding.radial_build_m * 1000.0 / tape.tape_thickness_mm))
    n_turns_avail = n_axial * n_radial

    if n_turns_req > n_turns_avail:
        violations.append(
            f"Need {n_turns_req} turns but winding pack fits only {n_turns_avail} "
            f"({n_axial} axial × {n_radial} radial)"
        )
    elif n_turns_req > 0.9 * n_turns_avail:
        warnings.append(
            f"Winding pack utilisation {100*n_turns_req/n_turns_avail:.0f}% — "
            "consider a larger pack for quench margin"
        )

    # 4. Topology recommendation
    aspect = winding.axial_length_m / winding.radial_build_m
    recommended = "pancake" if aspect > 2.0 else "layer_wound"
    if winding.topology != recommended:
        warnings.append(
            f"Current topology '{winding.topology}' — '{recommended}' may be preferable "
            f"(aspect ratio = {aspect:.1f})"
        )

    # 5. Inner radius sanity check
    if winding.inner_radius_m > coil_radius_m:
        violations.append(
            f"Winding pack inner radius {winding.inner_radius_m*100:.1f} cm > "
            f"coil mean radius {coil_radius_m*100:.1f} cm"
        )

    return ManufacturabilityResult(
        feasible=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        recommended_topology=recommended,
        n_turns_required=n_turns_req,
        n_turns_available=n_turns_avail,
        bend_radius_ratio=bend_ratio,
    )


def suggest_winding_pack(
    coil_radius_m: float,
    coil_current_A: float,
    tape: REBCOTapeSpec | None = None,
) -> WindingPackSpec:
    """Auto-suggest a winding pack that satisfies the current requirement.

    Parameters
    ----------
    coil_radius_m : float
        Target mean coil radius [m].
    coil_current_A : float
        Total coil current [A].
    tape : REBCOTapeSpec, optional
        HTS tape spec.

    Returns
    -------
    WindingPackSpec
        A feasible winding pack for the requested coil.
    """
    if tape is None:
        tape = REBCOTapeSpec()

    n_turns = max(1, int(coil_current_A / tape.critical_current_A) + 1)

    # Square-ish winding pack with 10 % margin
    n_side = max(1, int((n_turns * 1.1) ** 0.5) + 1)
    radial_m = n_side * tape.tape_thickness_mm / 1000.0
    axial_m = n_side * tape.tape_width_mm / 1000.0

    inner_r = coil_radius_m - radial_m / 2.0
    if inner_r < tape.min_bend_radius_mm / 1000.0:
        inner_r = tape.min_bend_radius_mm / 1000.0

    topology = "pancake" if axial_m > 2 * radial_m else "layer_wound"

    return WindingPackSpec(
        inner_radius_m=inner_r,
        outer_radius_m=inner_r + radial_m,
        axial_length_m=axial_m,
        n_layers=n_side,
        n_turns_per_layer=n_side,
        topology=topology,
    )

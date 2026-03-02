"""Thermal-structural coil constraints for magnetic nozzle optimisation.

Imports coil thermal and structural limits from FEA exports and exposes
them as optimisation constraints that can be applied to coil geometries
produced by the scan / optimise workflow.

The constraint interface is:
- ``is_satisfied(coil) -> bool``
- ``violation(coil) -> float``  (0 if satisfied, positive if violated)

Constraints can be loaded from JSON or CSV FEA export formats::

    constraints = import_from_fea("coil_limits.json")
    ok, msgs = constraints.check_all(config.nozzle.coils)

References
----------
- Wilson et al. (2020) — Structural design of fusion propulsion coils.
- Montgomery (1980) — Solenoid magnet design.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CoilThermalConstraint:
    """Thermal operating limit for a coil.

    Attributes
    ----------
    max_current_A : float
        Maximum continuous current [A].
    max_current_density_Am2 : float
        Maximum current density [A/m²].
    max_temp_K : float
        Maximum coil temperature [K].
    coil_cross_section_m2 : float
        Conductor cross-sectional area [m²] (used for J = I/A).
    name : str
        Human-readable constraint label.
    """

    max_current_A: float
    max_current_density_Am2: float = 1e8  # 100 MA/m² (REBCO HTS typical)
    max_temp_K: float = 20.0  # HTS operating limit [K]
    coil_cross_section_m2: float = 1e-4  # 1 cm²
    name: str = "thermal"

    def current_density(self, coil: Any) -> float:
        """Current density for given coil [A/m²]."""
        return abs(coil.I) / self.coil_cross_section_m2

    def is_satisfied(self, coil: Any) -> bool:
        """True if coil current is within thermal limits."""
        return (
            abs(coil.I) <= self.max_current_A
            and self.current_density(coil) <= self.max_current_density_Am2
        )

    def violation(self, coil: Any) -> float:
        """Constraint violation magnitude (0 if satisfied, >0 if not).

        Returns the maximum fractional excess over any limit.
        """
        v_current = max(0.0, abs(coil.I) / self.max_current_A - 1.0)
        v_density = max(
            0.0,
            self.current_density(coil) / self.max_current_density_Am2 - 1.0,
        )
        return max(v_current, v_density)

    def to_dict(self) -> dict:
        return {
            "type": "thermal",
            "name": self.name,
            "max_current_A": self.max_current_A,
            "max_current_density_Am2": self.max_current_density_Am2,
            "max_temp_K": self.max_temp_K,
            "coil_cross_section_m2": self.coil_cross_section_m2,
        }


@dataclass
class CoilStructuralConstraint:
    """Structural (stress) limit for a coil.

    Hoop stress in a thin-walled solenoid:
        σ_hoop = μ₀ * J² * r² / 2

    Attributes
    ----------
    max_stress_Pa : float
        Maximum allowable hoop stress [Pa].
    safety_factor : float
        Design safety factor (stress_actual < max_stress / safety_factor).
    name : str
        Human-readable label.
    """

    max_stress_Pa: float = 500e6  # 500 MPa (REBCO structural limit)
    safety_factor: float = 2.0
    name: str = "structural"

    _MU0 = 4.0 * math.pi * 1e-7

    def hoop_stress(self, coil: Any, current_density_Am2: float) -> float:
        """Hoop stress in a thin-walled coil [Pa].

        σ = μ₀ * J² * r² / 2
        """
        return self._MU0 * current_density_Am2**2 * coil.r**2 / 2.0

    def is_satisfied(self, coil: Any, current_density_Am2: float) -> bool:
        """True if hoop stress is within structural limit."""
        sigma = self.hoop_stress(coil, current_density_Am2)
        return sigma <= self.max_stress_Pa / self.safety_factor

    def violation(self, coil: Any, current_density_Am2: float) -> float:
        """Fractional stress excess over limit (0 if satisfied)."""
        sigma = self.hoop_stress(coil, current_density_Am2)
        limit = self.max_stress_Pa / self.safety_factor
        return max(0.0, sigma / limit - 1.0)

    def to_dict(self) -> dict:
        return {
            "type": "structural",
            "name": self.name,
            "max_stress_Pa": self.max_stress_Pa,
            "safety_factor": self.safety_factor,
        }


@dataclass
class CoilConstraintSet:
    """Collection of thermal and structural constraints for all coils.

    Attributes
    ----------
    thermal : list of CoilThermalConstraint
        One per coil (or shared if single element).
    structural : list of CoilStructuralConstraint
        One per coil (or shared if single element).
    """

    thermal: list[CoilThermalConstraint] = field(default_factory=list)
    structural: list[CoilStructuralConstraint] = field(default_factory=list)

    def check_all(
        self, coils: list[Any]
    ) -> tuple[bool, list[str]]:
        """Check all constraints against a list of coils.

        Parameters
        ----------
        coils : list
            Coil objects with attributes ``I`` (current) and ``r`` (radius).

        Returns
        -------
        (satisfied, messages)
            ``satisfied`` is True if all constraints pass.
            ``messages`` lists any violations.
        """
        messages = []
        satisfied = True

        for i, coil in enumerate(coils):
            # Thermal constraints
            if self.thermal:
                tc = self.thermal[i] if i < len(self.thermal) else self.thermal[-1]
                if not tc.is_satisfied(coil):
                    v = tc.violation(coil)
                    messages.append(
                        f"Coil {i}: thermal violation {v:.2%} above limit "
                        f"(I={coil.I:.0f} A, max={tc.max_current_A:.0f} A)"
                    )
                    satisfied = False

            # Structural constraints (need current density from thermal)
            if self.structural:
                sc = self.structural[i] if i < len(self.structural) else self.structural[-1]
                tc_for_area = (
                    self.thermal[i] if self.thermal and i < len(self.thermal)
                    else self.thermal[-1] if self.thermal
                    else None
                )
                if tc_for_area is not None:
                    j = tc_for_area.current_density(coil)
                else:
                    j = abs(coil.I) / 1e-4  # default 1 cm²
                if not sc.is_satisfied(coil, j):
                    v = sc.violation(coil, j)
                    messages.append(
                        f"Coil {i}: structural violation {v:.2%} above limit "
                        f"(σ_hoop > {sc.max_stress_Pa/sc.safety_factor:.2e} Pa)"
                    )
                    satisfied = False

        return satisfied, messages

    def total_violation(self, coils: list[Any]) -> float:
        """Sum of all constraint violations (for use as penalty in optimisation)."""
        total = 0.0
        for i, coil in enumerate(coils):
            if self.thermal:
                tc = self.thermal[i] if i < len(self.thermal) else self.thermal[-1]
                total += tc.violation(coil)
            if self.structural:
                sc = self.structural[i] if i < len(self.structural) else self.structural[-1]
                tc_for_area = (
                    self.thermal[i] if self.thermal and i < len(self.thermal)
                    else self.thermal[-1] if self.thermal
                    else None
                )
                j = tc_for_area.current_density(coil) if tc_for_area else abs(coil.I) / 1e-4
                total += sc.violation(coil, j)
        return total

    def to_dict(self) -> dict:
        return {
            "thermal": [c.to_dict() for c in self.thermal],
            "structural": [c.to_dict() for c in self.structural],
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def import_from_fea(path: str | Path) -> CoilConstraintSet:
    """Load coil constraints from an FEA export file.

    Supports JSON and CSV formats.

    JSON format (single or per-coil)::

        {
          "thermal": [
            {"max_current_A": 50000, "max_current_density_Am2": 5e7,
             "max_temp_K": 20, "coil_cross_section_m2": 1e-3}
          ],
          "structural": [
            {"max_stress_Pa": 300e6, "safety_factor": 2.0}
          ]
        }

    CSV format (one coil per row)::

        coil_id,max_current_A,max_current_density_Am2,max_temp_K,max_stress_Pa,safety_factor
        0,50000,5e7,20,300e6,2.0
        1,80000,8e7,20,500e6,2.0

    Parameters
    ----------
    path : path-like
        Path to the FEA export file (.json or .csv).

    Returns
    -------
    CoilConstraintSet
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_json(path)
    elif path.suffix.lower() == ".csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported FEA export format: {path.suffix}. Use .json or .csv")


def _load_json(path: Path) -> CoilConstraintSet:
    data = json.loads(path.read_text())
    thermal = []
    for td in data.get("thermal", []):
        thermal.append(CoilThermalConstraint(
            max_current_A=float(td["max_current_A"]),
            max_current_density_Am2=float(td.get("max_current_density_Am2", 1e8)),
            max_temp_K=float(td.get("max_temp_K", 20.0)),
            coil_cross_section_m2=float(td.get("coil_cross_section_m2", 1e-4)),
            name=td.get("name", "thermal"),
        ))
    structural = []
    for sd in data.get("structural", []):
        structural.append(CoilStructuralConstraint(
            max_stress_Pa=float(sd["max_stress_Pa"]),
            safety_factor=float(sd.get("safety_factor", 2.0)),
            name=sd.get("name", "structural"),
        ))
    return CoilConstraintSet(thermal=thermal, structural=structural)


def _load_csv(path: Path) -> CoilConstraintSet:
    thermal = []
    structural = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            thermal.append(CoilThermalConstraint(
                max_current_A=float(row["max_current_A"]),
                max_current_density_Am2=float(row.get("max_current_density_Am2", 1e8)),
                max_temp_K=float(row.get("max_temp_K", 20.0)),
                coil_cross_section_m2=float(row.get("coil_cross_section_m2", 1e-4)),
            ))
            if "max_stress_Pa" in row:
                structural.append(CoilStructuralConstraint(
                    max_stress_Pa=float(row["max_stress_Pa"]),
                    safety_factor=float(row.get("safety_factor", 2.0)),
                ))
    return CoilConstraintSet(thermal=thermal, structural=structural)

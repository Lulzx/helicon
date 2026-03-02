"""ThrusterArray: model for 2-4 thruster arrays with plume interaction.

The array is assumed to be linear (thrusters arranged along a single axis
perpendicular to the thrust direction).  For 2D array layouts, use the
``positions_m`` override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from helicon.multithruster.interaction import (
    PlumeModel,
    compute_plume_interaction,
)


@dataclass
class ArrayConfig:
    """Configuration for a multi-thruster array.

    Parameters
    ----------
    n_thrusters : int
        Number of thrusters (2–4).
    separation_m : float
        Centre-to-centre separation between adjacent thrusters [m].
    thrust_N : list[float]
        Per-thruster nominal thrust [N].  Length must equal *n_thrusters*.
    isp_s : list[float]
        Per-thruster Isp [s].  Length must equal *n_thrusters*.
    plume_half_angle_deg : list[float]
        Per-thruster plume half-angle [deg].
    positions_m : list[tuple[float, float]] | None
        Optional explicit 2-D (y, z_perp) positions [m] for non-linear arrays.
        If None, a linear layout with *separation_m* pitch is used.
    reference_z_m : float
        Axial distance at which plume overlap is evaluated [m].
    """

    n_thrusters: int = 2
    separation_m: float = 0.5
    thrust_N: list[float] = field(default_factory=lambda: [0.1, 0.1])
    isp_s: list[float] = field(default_factory=lambda: [3000.0, 3000.0])
    plume_half_angle_deg: list[float] = field(default_factory=lambda: [15.0, 15.0])
    positions_m: list[tuple[float, float]] | None = None
    reference_z_m: float = 1.0

    def __post_init__(self) -> None:
        if not (2 <= self.n_thrusters <= 4):
            raise ValueError(f"n_thrusters must be 2–4, got {self.n_thrusters}")
        for attr in ("thrust_N", "isp_s", "plume_half_angle_deg"):
            if len(getattr(self, attr)) != self.n_thrusters:
                raise ValueError(
                    f"Length of '{attr}' must equal n_thrusters={self.n_thrusters}"
                )

    def get_positions(self) -> list[tuple[float, float]]:
        """Return (y, z_perp) positions for each thruster [m].

        If ``positions_m`` is provided, use it directly.
        Otherwise, generate a symmetric linear layout centred at origin.
        """
        if self.positions_m is not None:
            return list(self.positions_m)
        # Linear layout: evenly spaced, centred at 0
        n = self.n_thrusters
        offsets = np.linspace(
            -(n - 1) * self.separation_m / 2,
            (n - 1) * self.separation_m / 2,
            n,
        )
        return [(float(y), 0.0) for y in offsets]

    def get_plume_models(self) -> list[PlumeModel]:
        """Return a PlumeModel for each thruster."""
        return [
            PlumeModel(
                thrust_N=self.thrust_N[i],
                isp_s=self.isp_s[i],
                half_angle_deg=self.plume_half_angle_deg[i],
            )
            for i in range(self.n_thrusters)
        ]


@dataclass
class ThrusterPairInteraction:
    """Pairwise interaction result between thruster i and j."""

    i: int
    j: int
    separation_m: float
    overlap_factor: float
    thrust_penalty_fraction: float


@dataclass
class ArrayResult:
    """Result of a multi-thruster array analysis.

    Attributes
    ----------
    total_thrust_N : float
        Combined thrust after interaction penalties [N].
    nominal_thrust_N : float
        Sum of individual nominal thrusts (no penalties) [N].
    total_mass_flow_kgs : float
        Combined mass flow rate [kg/s].
    effective_isp_s : float
        System-level effective Isp [s].
    interaction_penalty : float
        Fractional thrust loss due to plume-plume interaction (0–1).
    pair_interactions : list[ThrusterPairInteraction]
        Detailed pairwise interaction data.
    per_thruster_thrust_N : list[float]
        Effective per-thruster thrust after interactions [N].
    """

    total_thrust_N: float
    nominal_thrust_N: float
    total_mass_flow_kgs: float
    effective_isp_s: float
    interaction_penalty: float
    pair_interactions: list[ThrusterPairInteraction]
    per_thruster_thrust_N: list[float]


class ThrusterArray:
    """Multi-thruster array analyser.

    Parameters
    ----------
    config : ArrayConfig
        Array configuration.
    """

    def __init__(self, config: ArrayConfig) -> None:
        self.config = config

    def compute(self) -> ArrayResult:
        """Compute combined performance metrics for the array.

        Returns
        -------
        ArrayResult
        """
        cfg = self.config
        plumes = cfg.get_plume_models()
        positions = cfg.get_positions()
        n = cfg.n_thrusters

        # Accumulate per-thruster penalty factors from all pairs
        penalty_accumulator = [0.0] * n  # sum of penalty fracs for each thruster
        pair_results = []

        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean separation
                dy = positions[j][0] - positions[i][0]
                dz = positions[j][1] - positions[i][1]
                sep = math.sqrt(dy**2 + dz**2)

                result = compute_plume_interaction(
                    plumes[i],
                    plumes[j],
                    separation_m=sep,
                    reference_z_m=cfg.reference_z_m,
                )
                pair_results.append(
                    ThrusterPairInteraction(
                        i=i,
                        j=j,
                        separation_m=sep,
                        overlap_factor=result.overlap_factor,
                        thrust_penalty_fraction=result.thrust_penalty_fraction,
                    )
                )
                # Each thruster in the pair suffers the penalty
                penalty_accumulator[i] += result.thrust_penalty_fraction
                penalty_accumulator[j] += result.thrust_penalty_fraction

        # Clamp penalties to [0, 1]
        effective_thrusts = [
            plumes[i].thrust_N * max(0.0, 1.0 - penalty_accumulator[i])
            for i in range(n)
        ]

        nominal = sum(p.thrust_N for p in plumes)
        total = sum(effective_thrusts)
        total_mdot = sum(p.mass_flow_rate_kgs for p in plumes)

        effective_isp = total / (total_mdot * 9.80665) if total_mdot > 0 else 0.0

        interaction_penalty = (nominal - total) / nominal if nominal > 0 else 0.0

        return ArrayResult(
            total_thrust_N=total,
            nominal_thrust_N=nominal,
            total_mass_flow_kgs=total_mdot,
            effective_isp_s=effective_isp,
            interaction_penalty=interaction_penalty,
            pair_interactions=pair_results,
            per_thruster_thrust_N=effective_thrusts,
        )

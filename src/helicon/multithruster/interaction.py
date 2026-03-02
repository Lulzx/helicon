"""Plume-plume interaction physics for multi-thruster arrays.

Models the interaction between adjacent thruster plumes using a
simplified geometric / fluid model.  Two effects are captured:

1. **Momentum flux overlap**: When two plumes intersect, momentum in the
   transverse direction partially cancels, reducing net axial thrust.
2. **Back-pressure effect**: Plume convergence raises downstream pressure,
   slightly reducing the effective exhaust velocity.

All models are intentionally lightweight (no WarpX call needed) so they can
run on Apple Silicon in milliseconds via NumPy / MLX.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PlumeModel:
    """Geometric description of a single thruster plume.

    Parameters
    ----------
    thrust_N : float
        Nominal thrust output [N].
    isp_s : float
        Specific impulse [s].
    half_angle_deg : float
        Plume half-angle (divergence) [deg].  Typical range 10–25°.
    exhaust_velocity_ms : float | None
        Exhaust velocity [m/s].  Derived from Isp if None.
    """

    thrust_N: float
    isp_s: float
    half_angle_deg: float
    exhaust_velocity_ms: float | None = None

    def __post_init__(self) -> None:
        if self.exhaust_velocity_ms is None:
            self.exhaust_velocity_ms = self.isp_s * 9.80665

    @property
    def mass_flow_rate_kgs(self) -> float:
        """Mass flow rate [kg/s] = F / (Isp * g0)."""
        return self.thrust_N / (self.isp_s * 9.80665)

    @property
    def half_angle_rad(self) -> float:
        return math.radians(self.half_angle_deg)

    def plume_radius_at(self, z: float) -> float:
        """Plume cone radius [m] at axial distance *z* from exit plane."""
        return z * math.tan(self.half_angle_rad)


def compute_overlap_factor(
    separation_m: float,
    half_angle_deg_a: float,
    half_angle_deg_b: float,
    reference_z_m: float = 1.0,
) -> float:
    """Compute the fractional plume overlap between two adjacent thrusters.

    Uses the intersection area of two overlapping circles at a reference
    axial distance.  Returns a value in [0, 1] where 0 = no overlap,
    1 = complete overlap.

    Parameters
    ----------
    separation_m : float
        Centre-to-centre separation between the two thrusters [m].
    half_angle_deg_a, half_angle_deg_b : float
        Plume half-angles of each thruster [deg].
    reference_z_m : float
        Axial distance at which to evaluate the overlap [m].

    Returns
    -------
    float
        Overlap factor ∈ [0, 1].
    """
    r_a = reference_z_m * math.tan(math.radians(half_angle_deg_a))
    r_b = reference_z_m * math.tan(math.radians(half_angle_deg_b))
    d = separation_m

    if d >= r_a + r_b:
        return 0.0  # No overlap

    if d <= abs(r_a - r_b):
        # One circle entirely inside the other
        smaller_area = math.pi * min(r_a, r_b) ** 2
        larger_area = math.pi * max(r_a, r_b) ** 2
        return smaller_area / larger_area

    # Partial overlap — lens-area formula
    d2 = d * d
    r_a2 = r_a * r_a
    r_b2 = r_b * r_b

    alpha = math.acos((d2 + r_a2 - r_b2) / (2 * d * r_a))
    beta = math.acos((d2 + r_b2 - r_a2) / (2 * d * r_b))

    intersection = r_a2 * (alpha - math.sin(alpha) * math.cos(alpha)) + r_b2 * (
        beta - math.sin(beta) * math.cos(beta)
    )

    total_area = math.pi * (r_a2 + r_b2)
    return intersection / total_area


@dataclass
class InteractionResult:
    """Result of a two-plume interaction analysis."""

    overlap_factor: float
    thrust_penalty_fraction: float  # fractional thrust loss due to interaction
    back_pressure_factor: float  # multiplicative factor on effective Ve
    combined_thrust_N: float
    combined_mass_flow_kgs: float
    combined_isp_s: float


def compute_plume_interaction(
    plume_a: PlumeModel,
    plume_b: PlumeModel,
    separation_m: float,
    reference_z_m: float = 1.0,
) -> InteractionResult:
    """Compute thrust and Isp degradation from plume-plume interaction.

    The penalty model uses a linear fit calibrated to 2D PIC plume-plume
    simulations reported in the literature (Hara et al. 2019 approximation):

        ΔF/F ≈ 0.6 × overlap_factor

    The back-pressure factor is:

        f_bp = 1 - 0.15 × overlap_factor

    Parameters
    ----------
    plume_a, plume_b : PlumeModel
        Individual plume models.
    separation_m : float
        Centre-to-centre separation [m].
    reference_z_m : float
        Evaluation plane [m].

    Returns
    -------
    InteractionResult
    """
    overlap = compute_overlap_factor(
        separation_m,
        plume_a.half_angle_deg,
        plume_b.half_angle_deg,
        reference_z_m=reference_z_m,
    )

    penalty_frac = 0.6 * overlap  # thrust loss fraction
    bp_factor = 1.0 - 0.15 * overlap

    thrust_a_eff = plume_a.thrust_N * (1.0 - penalty_frac)
    thrust_b_eff = plume_b.thrust_N * (1.0 - penalty_frac)
    combined_thrust = thrust_a_eff + thrust_b_eff

    mdot_a = plume_a.mass_flow_rate_kgs
    mdot_b = plume_b.mass_flow_rate_kgs
    combined_mdot = mdot_a + mdot_b

    # Effective combined Isp (thrust-weighted)
    combined_isp = combined_thrust / (combined_mdot * 9.80665) if combined_mdot > 0 else 0.0

    return InteractionResult(
        overlap_factor=overlap,
        thrust_penalty_fraction=penalty_frac,
        back_pressure_factor=bp_factor,
        combined_thrust_N=combined_thrust,
        combined_mass_flow_kgs=combined_mdot,
        combined_isp_s=combined_isp,
    )

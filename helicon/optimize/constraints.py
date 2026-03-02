"""Engineering constraints for coil optimization.

Implements the physical constraints that bound real coil designs:

* **Coil mass** — conductor volume × density. Mass budget drives launch cost.
* **Resistive power** — I²R dissipation. Drives thermal management requirements
  for non-superconducting coils.
* **Peak field at conductor** — REBCO superconductor limit ~20 T; copper limit
  determined by structural / quench margin.
* **Structural load** — magnetic pressure B²/(2μ₀) gives hoop stress on the coil.

These constraints enter gradient-based MLX optimization as a differentiable
penalty function and gradient-free Bayesian optimization as constraint
functions for botorch's constrained Expected Improvement.

Physics model
-------------
Each coil is characterized by its position (z, r) and current I [A-turns].
The conductor cross-section is sized to a maximum current density J:

    A_conductor = |I| / J_max           [m²]

From that, per-coil mass and resistive power follow:

    L_wire   = 2π r                     [m]  (single-turn mean path length)
    m_coil   = ρ_Cu · A_conductor · L   [kg]
    R_coil   = ρ_elec · L / A           [Ω]
    P_coil   = I² · R_coil              [W]

Peak magnetic field at the conductor location (thin-coil on-axis
approximation):

    B_peak = μ₀ |I| / (2r)             [T]

References
----------
Braginskii, S.I. (1965). Transport processes in a plasma. Rev. Plasma Phys. 1.
Weisend, J.G. (1998). Handbook of Accelerator Physics and Engineering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# Physical constants
_MU0 = 4.0e-7 * math.pi


@dataclass
class CoilConstraints:
    """Engineering constraint specification for coil optimization.

    Parameters
    ----------
    max_total_mass_kg : float or None
        Maximum total mass of all coils combined [kg]. None = unconstrained.
    max_total_power_W : float or None
        Maximum total resistive power dissipation [W]. None = unconstrained.
        For superconducting coils set this to None (zero resistive loss).
    max_B_conductor_T : float or None
        Maximum peak field at the conductor surface [T]. REBCO tapes have
        a practical limit of ~15-20 T; copper coils are limited by structural
        / Lorentz force considerations. None = unconstrained.
    current_density_Am2 : float
        Maximum current density for conductor sizing [A/m²].
        Typical values: 1-10 MA/m² for copper, 100-500 MA/m² for REBCO.
    conductor_resistivity_Ohm_m : float
        Electrical resistivity of conductor [Ω·m].
        Copper at 20 °C: 1.72e-8 Ω·m.
        Set to 0.0 for superconducting (zero resistive loss).
    conductor_density_kg_m3 : float
        Mass density of conductor [kg/m³]. Copper: 8960 kg/m³.
    """

    max_total_mass_kg: float | None = None
    max_total_power_W: float | None = None
    max_B_conductor_T: float | None = None
    current_density_Am2: float = 1.0e7          # 10 MA/m², typical for copper
    conductor_resistivity_Ohm_m: float = 1.72e-8  # copper at 20 °C
    conductor_density_kg_m3: float = 8960.0      # copper


@dataclass
class CoilConstraintResult:
    """Per-coil and aggregate constraint evaluation results.

    Attributes
    ----------
    coil_masses_kg : list of float
        Mass of each coil [kg].
    coil_powers_W : list of float
        Resistive power dissipation of each coil [W].
    coil_B_peak_T : list of float
        Peak field at the conductor of each coil [T].
    total_mass_kg : float
        Sum of all coil masses [kg].
    total_power_W : float
        Sum of all coil powers [W].
    violations : dict
        Constraint name → violation amount (positive = violated).
        Zero or negative means the constraint is satisfied.
    penalty : float
        Quadratic penalty value (for use in penalized objectives).
    satisfied : bool
        True if all active constraints are satisfied.
    """

    coil_masses_kg: list[float]
    coil_powers_W: list[float]
    coil_B_peak_T: list[float]
    total_mass_kg: float
    total_power_W: float
    violations: dict[str, float] = field(default_factory=dict)
    penalty: float = 0.0
    satisfied: bool = True


def evaluate_constraints(
    coil_params: np.ndarray,
    constraints: CoilConstraints,
    *,
    penalty_factor: float = 1.0e3,
) -> CoilConstraintResult:
    """Evaluate engineering constraints for a set of coil parameters.

    Parameters
    ----------
    coil_params : array, shape (N_coils, 3)
        Coil parameters with columns ``[z, r, I]``.
    constraints : CoilConstraints
        Active constraint specification.
    penalty_factor : float
        Scaling factor for quadratic penalty (used by optimizer).

    Returns
    -------
    CoilConstraintResult
    """
    params = np.atleast_2d(np.asarray(coil_params, dtype=float))
    z_arr = params[:, 0]
    r_arr = params[:, 1]
    I_arr = params[:, 2]

    J = constraints.current_density_Am2
    rho_elec = constraints.conductor_resistivity_Ohm_m
    rho_mass = constraints.conductor_density_kg_m3

    # Per-coil quantities
    abs_I = np.abs(I_arr)
    r_safe = np.maximum(r_arr, 1e-9)
    L_wire = 2.0 * math.pi * r_safe               # conductor path length per coil [m]
    A_cond = abs_I / max(J, 1e-30)                # conductor cross-section [m²]

    masses = rho_mass * A_cond * L_wire            # coil mass [kg]
    total_mass = float(np.sum(masses))

    R_coil = rho_elec * L_wire / np.maximum(A_cond, 1e-30)
    powers = abs_I**2 * R_coil                     # coil power [W]
    total_power = float(np.sum(powers))

    B_peak = _MU0 * abs_I / (2.0 * r_safe)        # on-axis field at coil [T]

    violations: dict[str, float] = {}
    penalty = 0.0

    if constraints.max_total_mass_kg is not None:
        v = total_mass - constraints.max_total_mass_kg
        violations["total_mass_kg"] = float(v)
        if v > 0:
            penalty += penalty_factor * v**2

    if constraints.max_total_power_W is not None:
        v = total_power - constraints.max_total_power_W
        violations["total_power_W"] = float(v)
        if v > 0:
            penalty += penalty_factor * v**2

    if constraints.max_B_conductor_T is not None:
        B_max = float(np.max(B_peak))
        v = B_max - constraints.max_B_conductor_T
        violations["max_B_conductor_T"] = float(v)
        if v > 0:
            penalty += penalty_factor * v**2

    satisfied = all(v <= 0 for v in violations.values())

    return CoilConstraintResult(
        coil_masses_kg=masses.tolist(),
        coil_powers_W=powers.tolist(),
        coil_B_peak_T=B_peak.tolist(),
        total_mass_kg=total_mass,
        total_power_W=total_power,
        violations=violations,
        penalty=penalty,
        satisfied=satisfied,
    )


def make_constrained_objective(
    objective_fn,
    constraints: CoilConstraints,
    *,
    penalty_factor: float = 1.0e3,
):
    """Wrap an MLX objective with a differentiable constraint penalty.

    The returned function has the same signature as ``objective_fn`` but
    adds a quadratic penalty for each violated constraint:

        f_constrained(x) = objective_fn(x) + Σ penalty_factor * max(g_i(x), 0)²

    All operations use MLX so the result is differentiable via ``mlx.core.grad``.

    Parameters
    ----------
    objective_fn : callable
        MLX-differentiable objective; takes ``coil_params`` (shape (N, 3))
        and returns a scalar ``mx.array``.
    constraints : CoilConstraints
        Active constraints.
    penalty_factor : float
        Quadratic penalty weight.

    Returns
    -------
    callable
        Penalized objective suitable for ``mlx.optimizers.Adam``.

    Examples
    --------
    ::

        bounds = constraints.CoilConstraints(max_total_mass_kg=50.0, max_B_conductor_T=15.0)
        penalized = make_constrained_objective(throat_ratio_objective, bounds)
        result = optimize_coils_mlx(init_params, penalized)
    """
    try:
        import mlx.core as mx
    except ImportError as exc:
        msg = "MLX is required for make_constrained_objective. Install with: pip install mlx"
        raise ImportError(msg) from exc

    J = constraints.current_density_Am2
    rho_elec = constraints.conductor_resistivity_Ohm_m
    rho_mass = constraints.conductor_density_kg_m3
    mu0 = float(_MU0)

    def constrained(coil_params: "mx.array") -> "mx.array":
        obj = objective_fn(coil_params)

        r = coil_params[:, 1]
        I = coil_params[:, 2]
        r_safe = mx.maximum(r, 1e-9)
        abs_I = mx.abs(I)

        L_wire = 2.0 * math.pi * r_safe
        A_cond = abs_I / max(J, 1e-30)
        masses = rho_mass * A_cond * L_wire
        total_mass = mx.sum(masses)

        R_coil = rho_elec * L_wire / mx.maximum(A_cond, 1e-30)
        powers = abs_I**2 * R_coil
        total_power = mx.sum(powers)

        B_peak = mu0 * abs_I / (2.0 * r_safe)

        pen = mx.array(0.0)

        if constraints.max_total_mass_kg is not None:
            v = mx.maximum(total_mass - constraints.max_total_mass_kg, 0.0)
            pen = pen + penalty_factor * v**2

        if constraints.max_total_power_W is not None:
            v = mx.maximum(total_power - constraints.max_total_power_W, 0.0)
            pen = pen + penalty_factor * v**2

        if constraints.max_B_conductor_T is not None:
            v = mx.maximum(mx.max(B_peak) - constraints.max_B_conductor_T, 0.0)
            pen = pen + penalty_factor * v**2

        return obj + pen

    return constrained

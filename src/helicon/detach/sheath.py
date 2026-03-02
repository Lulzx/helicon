"""Non-equilibrium sheath coupling correction to bulk detachment model.

Standard detachment models assume bulk plasma and boundary sheath are
independent.  In practice the expanding plasma drives a sheath at the
plume boundary whose ambipolar pre-sheath electric field re-couples to
the bulk through three mechanisms:

1. **Bohm current** — the sheath enforces u_s ≥ c_s at the sheath edge,
   imposing a parallel current boundary condition on the bulk.

2. **Ambipolar pre-sheath field** — the electric field
   E_ps ≈ T_e / (e L_ps) accelerates ions toward the sheath and
   competes with the magnetic mirror force in the demagnetization
   criterion.

3. **Debye shielding** — charge separation on scale λ_D modifies the
   effective perpendicular force on ions, reducing the E×B drift and
   altering the effective Larmor orbit.

When ε_ES = E_force / F_mirror > 1 the electric contribution dominates
and the magnetic criterion overestimates detachment probability.  The
correction reduces the raw detachment score::

    S_corr = S_raw / (1 + ξ ε_ES)^½

where ξ ∈ [0, 1] is a coupling factor calibrated against data.

References
----------
- Ahedo, E. & Merino, M. (2010). *Physics of Plasmas* 17, 073501.
- Merino, M. & Ahedo, E. (2011). *Physics of Plasmas* 18, 053504.
- Lieberman, M.A. & Lichtenberg, A.J. (2005). *Principles of Plasma
  Discharges*, 2nd ed.  Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from helicon.detach.invariants import EV_TO_J, M_PROTON, ion_magnetization

# Physical constants
_EPSILON_0: float = 8.8541878128e-12  # F/m
_M_ELECTRON: float = 9.1093837015e-31  # kg
_EL_CHARGE: float = EV_TO_J            # C (= 1 eV in J)


def debye_length(n_m3: float, Te_eV: float) -> float:
    """Electron Debye length λ_D = √(ε₀ T_e / (n e))  [m].

    The Debye length sets the spatial scale of electrostatic shielding
    and the minimum sheath thickness (typically 3–5 λ_D).

    Parameters
    ----------
    n_m3 : float
        Plasma density [m⁻³].
    Te_eV : float
        Electron temperature [eV].

    Returns
    -------
    float
        Debye length [m].
    """
    if n_m3 <= 0 or Te_eV <= 0:
        return math.inf
    return math.sqrt(_EPSILON_0 * Te_eV * EV_TO_J / (n_m3 * _EL_CHARGE ** 2))


def sheath_potential(Te_eV: float, mass_amu: float) -> float:
    """Floating sheath potential Φ_s = (T_e / 2e) · ln(m_i / (2π m_e))  [V].

    The potential drop across the sheath in the floating (zero net
    current) condition.  Depends on the ion-to-electron mass ratio:

    * H⁺:  Φ_s ≈ 2.84 T_e/e [V]
    * Ar⁺: Φ_s ≈ 4.68 T_e/e [V]

    Parameters
    ----------
    Te_eV : float
        Electron temperature [eV].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Floating sheath potential [V].
    """
    m_i = mass_amu * M_PROTON
    ratio = m_i / _M_ELECTRON
    if ratio <= 0:
        return 0.0
    return (Te_eV / 2.0) * math.log(ratio / (2.0 * math.pi))


def electric_to_mirror_ratio(
    Te_eV: float,
    Ti_eV: float,
    B_T: float,
    dBdz_T_per_m: float,
    mass_amu: float,
) -> float:
    """Dimensionless ratio ε_ES = pre-sheath electric force / mirror force.

    Derived from the pre-sheath electric field E_ps ≈ T_e / (e L_B)
    and the magnetic mirror force F_mirror ≈ μ |∂B/∂z| = μ B / L_B::

        ε_ES = e E_ps L_B / (μ · B/L_B)
             = T_e / T_i · 1/Λᵢ²

    where Λᵢ = r_Lᵢ / L_B is the ion demagnetization parameter.

    When ε_ES > 1: electric force exceeds mirror force.
    When ε_ES ≪ 1: magnetic detachment criterion valid as-is.

    Parameters
    ----------
    Te_eV, Ti_eV : float
        Electron and ion temperatures [eV].
    B_T : float
        Magnetic field [T].
    dBdz_T_per_m : float
        Axial B gradient [T/m].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Dimensionless coupling parameter ε_ES ≥ 0.
    """
    Lambda_i = ion_magnetization(Ti_eV, B_T, dBdz_T_per_m, mass_amu)
    if Lambda_i <= 0 or math.isinf(Lambda_i):
        return 0.0
    return (Te_eV / Ti_eV) / (Lambda_i ** 2)


@dataclass
class SheathCorrectedState:
    """Detachment assessment including sheath coupling corrections.

    Attributes
    ----------
    debye_length_m : float
        Electron Debye length λ_D [m].
    sheath_potential_V : float
        Floating sheath potential Φ_s [V].
    epsilon_ES : float
        Electric-to-mirror force ratio ε_ES.
    lambda_i_corrected : float
        Effective ion demagnetization after E-field correction.
    score_corrected : float
        Detachment score after sheath coupling correction.
    score_raw : float
        Original score from the MHD model.
    correction_fraction : float
        Fractional change (score_raw − score_corrected) / score_raw.
    """

    debye_length_m: float
    sheath_potential_V: float
    epsilon_ES: float
    lambda_i_corrected: float
    score_corrected: float
    score_raw: float
    correction_fraction: float


def apply_sheath_correction(
    score_raw: float,
    n_m3: float,
    Te_eV: float,
    Ti_eV: float,
    B_T: float,
    dBdz_T_per_m: float,
    mass_amu: float,
    coupling_factor: float = 0.30,
) -> SheathCorrectedState:
    """Apply non-equilibrium sheath coupling correction to detachment score.

    The pre-sheath electric field provides additional ion confinement
    even after magnetic detachment, reducing the effective score::

        S_corr = S_raw / √(1 + ξ · ε_ES)

    where:

    * ξ = coupling_factor ∈ [0, 1] (calibrate against data; 0.30 is
      a conservative estimate for open-field magnetic nozzles)
    * ε_ES = (T_e/T_i) / Λᵢ² (electric-to-mirror force ratio)
    * The √ exponent (½ power) comes from the force-balance dimension
      argument: the effective Larmor orbit modification scales as
      (1 + ξ ε_ES)^½ relative to the pure magnetic case.

    Additionally, the effective ion demagnetization is corrected::

        Λᵢ^eff = Λᵢ / √(1 + ξ ε_ES)

    since the electric field competes with and partially substitutes
    for the magnetic mirror force in the parallel dynamics.

    Parameters
    ----------
    score_raw : float
        Raw detachment score S ∈ [0, 1] from the MHD model.
    n_m3, Te_eV, Ti_eV, B_T, dBdz_T_per_m, mass_amu : float
        Local plasma state.
    coupling_factor : float
        ξ ∈ [0, 1].  0 = pure MHD (no correction).  Default 0.30.

    Returns
    -------
    SheathCorrectedState
    """
    lam_D = debye_length(n_m3, Te_eV)
    phi_s = sheath_potential(Te_eV, mass_amu)
    eps_ES = electric_to_mirror_ratio(Te_eV, Ti_eV, B_T, dBdz_T_per_m, mass_amu)

    # Coupling denominator √(1 + ξ ε_ES)
    if eps_ES < 0 or math.isinf(eps_ES):
        denom = 1.0
    else:
        denom = math.sqrt(1.0 + coupling_factor * eps_ES)

    score_corr = min(1.0, max(0.0, score_raw / denom))
    correction_frac = (score_raw - score_corr) / score_raw if score_raw > 1e-10 else 0.0

    # Corrected ion demagnetization
    lambda_i_raw = ion_magnetization(Ti_eV, B_T, dBdz_T_per_m, mass_amu)
    lambda_i_corr = (
        lambda_i_raw / denom if (math.isfinite(lambda_i_raw) and denom > 0) else lambda_i_raw
    )

    return SheathCorrectedState(
        debye_length_m=lam_D,
        sheath_potential_V=phi_s,
        epsilon_ES=eps_ES,
        lambda_i_corrected=lambda_i_corr,
        score_corrected=score_corr,
        score_raw=score_raw,
        correction_fraction=correction_frac,
    )

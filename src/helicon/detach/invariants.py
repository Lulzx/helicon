"""MHD invariants and dimensionless parameters for detachment analysis.

All functions are pure scalar arithmetic — no numpy, no scipy.
Suitable for direct port to C/C++ or MicroPython on embedded hardware.

Physical constants (SI):
    μ₀ = 1.256 637 061 × 10⁻⁶  H/m
    k_B = 1.380 649 × 10⁻²³   J/K
    eV  = 1.602 176 634 × 10⁻¹⁹ J
    mₚ  = 1.672 621 923 × 10⁻²⁷ kg
"""

from __future__ import annotations

import math

# Physical constants (SI)
MU0: float = 1.2566370614359173e-6  # permeability of free space [H/m]
EV_TO_J: float = 1.602176634e-19  # electron-volt to joule
M_PROTON: float = 1.67262192369e-27  # proton mass [kg]
KB: float = 1.380649e-23  # Boltzmann constant [J/K]

# Ion species mass table [amu]
SPECIES_MASS_AMU: dict[str, float] = {
    "H+": 1.00794,
    "He+": 4.00260,
    "N+": 14.0067,
    "Ar+": 39.9480,
    "Kr+": 83.7980,
    "Xe+": 131.293,
}


def species_mass(name: str) -> float:
    """Return ion mass in amu for a species name (e.g. 'Ar+').

    Raises ``ValueError`` for unknown species.
    """
    if name not in SPECIES_MASS_AMU:
        opts = ", ".join(SPECIES_MASS_AMU)
        raise ValueError(f"Unknown species {name!r}. Choose from: {opts}")
    return SPECIES_MASS_AMU[name]


def alfven_velocity(B_T: float, n_m3: float, mass_amu: float) -> float:
    """Alfvén velocity v_A = B / √(μ₀ n mᵢ)  [m/s].

    Parameters
    ----------
    B_T : float
        Magnetic field strength [T].
    n_m3 : float
        Plasma density [m⁻³].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Alfvén velocity [m/s].
    """
    m_i = mass_amu * M_PROTON
    denom = MU0 * n_m3 * m_i
    if denom <= 0:
        return math.inf
    return B_T / math.sqrt(denom)


def alfven_mach(vz_ms: float, B_T: float, n_m3: float, mass_amu: float) -> float:
    """Alfvén Mach number M_A = v_z / v_A.

    Detachment is strongly indicated when M_A > 1.

    Parameters
    ----------
    vz_ms : float
        Axial bulk velocity [m/s].
    B_T, n_m3, mass_amu : float
        As in :func:`alfven_velocity`.

    Returns
    -------
    float
        Dimensionless Alfvén Mach number.
    """
    va = alfven_velocity(B_T, n_m3, mass_amu)
    if va == 0:
        return math.inf
    return vz_ms / va


def electron_beta(n_m3: float, Te_eV: float, B_T: float) -> float:
    """Electron plasma β = n k_B T_e / (B²/2μ₀).

    When β_e ≳ 0.1 electrons can deform field lines; full detachment
    is likely once β_e > 1.

    Parameters
    ----------
    n_m3 : float
        Plasma density [m⁻³].
    Te_eV : float
        Electron temperature [eV].
    B_T : float
        Magnetic field [T].

    Returns
    -------
    float
        Dimensionless electron β.
    """
    p_e = n_m3 * Te_eV * EV_TO_J  # electron pressure [Pa]
    p_B = B_T**2 / (2.0 * MU0)  # magnetic pressure [Pa]
    if p_B <= 0:
        return math.inf
    return p_e / p_B


def ion_larmor_radius(Ti_eV: float, B_T: float, mass_amu: float) -> float:
    """Thermal ion Larmor radius r_Lᵢ = mᵢ v_th,⊥ / (q B)  [m].

    Uses the thermal speed v_th = √(2 T_i / mᵢ) as the representative
    perpendicular velocity.

    Parameters
    ----------
    Ti_eV : float
        Ion temperature [eV].
    B_T : float
        Magnetic field [T].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Thermal Larmor radius [m].
    """
    m_i = mass_amu * M_PROTON
    v_th = math.sqrt(2.0 * Ti_eV * EV_TO_J / m_i)
    q = EV_TO_J  # elementary charge [C]
    if B_T <= 0:
        return math.inf
    return m_i * v_th / (q * B_T)


def field_scale_length(B_T: float, dBdz_T_per_m: float) -> float:
    """Magnetic field scale length L_B = B / |dB/dz|  [m].

    Shorter L_B → steeper gradient → more likely to demagnetize.

    Parameters
    ----------
    B_T : float
        Local field magnitude [T].
    dBdz_T_per_m : float
        Axial B gradient [T/m].

    Returns
    -------
    float
        Field scale length [m]; ``inf`` if gradient is negligible.
    """
    grad = abs(dBdz_T_per_m)
    if grad < 1e-30:
        return math.inf
    return B_T / grad


def ion_magnetization(
    Ti_eV: float,
    B_T: float,
    dBdz_T_per_m: float,
    mass_amu: float,
) -> float:
    """Ion demagnetization parameter Λᵢ = r_Lᵢ / L_B.

    Ions demagnetize (stop following field lines) when Λᵢ > 1.

    Parameters
    ----------
    Ti_eV : float
        Ion temperature [eV].
    B_T : float
        Magnetic field [T].
    dBdz_T_per_m : float
        Axial B gradient [T/m].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Dimensionless demagnetization parameter.
    """
    r_Li = ion_larmor_radius(Ti_eV, B_T, mass_amu)
    L_B = field_scale_length(B_T, dBdz_T_per_m)
    if L_B == 0:
        return math.inf
    return r_Li / L_B


def bohm_velocity(Te_eV: float, mass_amu: float) -> float:
    """Bohm (ion acoustic) velocity c_s = √(k_B T_e / mᵢ)  [m/s].

    Ions must exceed c_s at the sheath edge (Bohm criterion).
    Equivalently this is the sound speed in the cold-ion limit.

    Parameters
    ----------
    Te_eV : float
        Electron temperature [eV].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Bohm velocity [m/s].
    """
    m_i = mass_amu * M_PROTON
    return math.sqrt(Te_eV * EV_TO_J / m_i)


def magnetic_mirror_force(
    mu_J_per_T: float,
    dBdz_T_per_m: float,
) -> float:
    """Parallel mirror force F_∥ = −μ (∂B/∂z)  [N].

    The magnetic moment μ = m v_⊥² / (2B) is an adiabatic invariant.
    Positive F_∥ (with dBdz < 0) accelerates ions downstream.

    Parameters
    ----------
    mu_J_per_T : float
        Magnetic moment μ = m v_⊥² / (2B)  [J/T].
    dBdz_T_per_m : float
        Axial field gradient [T/m].

    Returns
    -------
    float
        Mirror force [N]; negative = decelerating (toward higher B).
    """
    return -mu_J_per_T * dBdz_T_per_m


def magnetic_moment(
    Ti_eV: float,
    B_T: float,
    mass_amu: float,
) -> float:
    """First adiabatic invariant μ = mᵢ v_th,⊥² / (2B)  [J/T].

    Conserved as long as ions remain magnetized (Λᵢ ≪ 1).

    Parameters
    ----------
    Ti_eV, B_T, mass_amu : float
        Ion temperature [eV], field [T], mass [amu].

    Returns
    -------
    float
        Magnetic moment [J/T = A·m²].
    """
    m_i = mass_amu * M_PROTON
    v_th_sq = 2.0 * Ti_eV * EV_TO_J / m_i  # thermal speed² for one degree
    if B_T <= 0:
        return math.inf
    return m_i * v_th_sq / (2.0 * B_T)

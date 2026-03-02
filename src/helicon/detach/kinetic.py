"""Kinetic (finite-Larmor-radius) corrections beyond MHD.

Three independent corrections are implemented:

1. **Maxwellian-averaged Larmor radius** — the cold-beam estimate
   r_Li = m v_th/(qB) uses a single representative speed.  For an
   isotropic Maxwellian the mean perpendicular speed is √(π kT/2m),
   giving a factor √(π/4) ≈ 0.886 correction.

2. **Northrop FLR correction to demagnetization** — the next-order
   term in the guiding-centre Hamiltonian expansion (Northrop, 1963)
   modifies the adiabatic breakdown criterion::

       Λᵢ_FLR = Λᵢ √(1 + ¾ Λᵢ²)

   For Λᵢ = 0.5 this is a 9% correction; at Λᵢ = 1.0 it is 32%.
   This matters in the transition zone where the standard model
   under-predicts demagnetization.

3. **Kinetic Alfvén wave speed** — at sub-ion-Larmor scales the
   dispersive kinetic Alfvén wave travels faster than the ideal MHD
   Alfvén wave::

       v_kAW = v_A √(1 + k_⊥² r_Li²)

   Using k_⊥ ~ 1/L_B as the characteristic cross-field wavenumber.
   This raises the effective wave speed and reduces M_A, making
   detachment slightly less likely than the MHD estimate.

4. **Full Bohm velocity** with finite ion temperature::

       c_Bohm = √(kB(Te + Ti)/mi)

   The cold-ion approximation c_s = √(kB Te/mi) underestimates the
   sheath entrance speed by up to 41% when T_i ~ T_e.

References
----------
- Northrop, T.G. (1963). *The Adiabatic Motion of Charged Particles*.
  Interscience.
- Hasegawa, A. (1976). Particle acceleration by a plasma wave.
  *Physics of Fluids* 19, 1144.
"""

from __future__ import annotations

import math

from helicon.detach.invariants import (
    EV_TO_J,
    M_PROTON,
    alfven_velocity,
    field_scale_length,
    ion_larmor_radius,
)

# √(π/4) — ratio of Maxwellian mean perpendicular speed to √(2kT/m)
_MAXWELLIAN_FACTOR: float = math.sqrt(math.pi / 4.0)

# Physical constants needed for ion inertial length
_EPSILON_0: float = 8.8541878128e-12  # F/m
_C_LIGHT: float = 2.99792458e8        # m/s


def larmor_radius_maxwellian(Ti_eV: float, B_T: float, mass_amu: float) -> float:
    """Maxwellian-ensemble-averaged thermal Larmor radius [m].

    For an isotropic Maxwellian, the mean perpendicular speed is

        ⟨v_⊥⟩ = √(π k_B T_i / 2 m_i)

    giving a representative Larmor radius:

        ⟨r_⊥⟩ = √(π/4) · r_Li^thermal

    where r_Li^thermal = m_i √(2 T_i/m_i) / (q B) is the cold-beam estimate.

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
        Mean Larmor radius [m].
    """
    r_thermal = ion_larmor_radius(Ti_eV, B_T, mass_amu)
    return r_thermal * _MAXWELLIAN_FACTOR


def ion_magnetization_flr(
    Ti_eV: float,
    B_T: float,
    dBdz_T_per_m: float,
    mass_amu: float,
) -> float:
    """FLR-corrected demagnetization parameter Λᵢ_FLR.

    Includes the second-order correction from the guiding-centre
    Hamiltonian (Northrop, 1963)::

        Λᵢ_FLR = Λᵢ_0 · √(1 + ¾ Λᵢ_0²)

    where Λᵢ_0 = ⟨r_⊥⟩ / L_B is the Maxwellian-averaged parameter.

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
        FLR-corrected demagnetization parameter.
    """
    r_Li = larmor_radius_maxwellian(Ti_eV, B_T, mass_amu)
    L_B = field_scale_length(B_T, dBdz_T_per_m)
    if L_B == 0 or math.isinf(L_B):
        return math.inf if L_B == 0 else 0.0
    Lambda_0 = r_Li / L_B
    return Lambda_0 * math.sqrt(1.0 + 0.75 * Lambda_0 ** 2)


def alfven_mach_kinetic(
    vz_ms: float,
    B_T: float,
    n_m3: float,
    mass_amu: float,
    Ti_eV: float,
    dBdz_T_per_m: float,
) -> float:
    """Kinetic Alfvén Mach number at sub-ion-Larmor scales.

    The kinetic Alfvén wave phase speed at wavenumber k_⊥ is::

        v_kAW = v_A √(1 + k_⊥² r_Li²)

    Taking k_⊥ = 1/L_B as the characteristic cross-field scale::

        v_kAW = v_A √(1 + Λᵢ_0²)

    where Λᵢ_0 = ⟨r_⊥⟩ / L_B.  For steep gradients (Λᵢ ~ 0.5) this
    raises v_kAW by ~12% relative to the ideal MHD Alfvén speed, reducing
    M_kAW = v_z / v_kAW correspondingly.

    Parameters
    ----------
    vz_ms : float
        Axial bulk velocity [m/s].
    B_T : float
        Magnetic field [T].
    n_m3 : float
        Plasma density [m⁻³].
    mass_amu : float
        Ion mass [amu].
    Ti_eV : float
        Ion temperature [eV] (needed for Larmor radius).
    dBdz_T_per_m : float
        Axial B gradient [T/m] (sets characteristic k_⊥ = 1/L_B).

    Returns
    -------
    float
        Kinetic Alfvén Mach number M_kAW.
    """
    va = alfven_velocity(B_T, n_m3, mass_amu)
    if va == 0:
        return math.inf
    r_Li = larmor_radius_maxwellian(Ti_eV, B_T, mass_amu)
    L_B = field_scale_length(B_T, dBdz_T_per_m)
    Lambda_i = r_Li / L_B if (math.isfinite(L_B) and L_B > 0) else 0.0
    v_kAW = va * math.sqrt(1.0 + Lambda_i ** 2)
    return vz_ms / v_kAW


def bohm_velocity_full(Te_eV: float, Ti_eV: float, mass_amu: float) -> float:
    """Full ion acoustic (Bohm) velocity including finite ion temperature.

    The cold-ion Bohm condition c_s = √(k_B T_e / m_i) underestimates
    the sheath entrance speed when T_i is non-negligible.  The correct
    expression for a Maxwellian plasma is::

        c_Bohm = √(k_B (T_e + T_i) / m_i)

    For T_i = T_e this gives a 41% higher speed, which raises the Bohm
    Mach number and shifts the detachment operating point.

    Parameters
    ----------
    Te_eV : float
        Electron temperature [eV].
    Ti_eV : float
        Ion temperature [eV].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Full Bohm velocity [m/s].
    """
    m_i = mass_amu * M_PROTON
    return math.sqrt((Te_eV + Ti_eV) * EV_TO_J / m_i)


def ion_inertial_length(n_m3: float, mass_amu: float) -> float:
    """Ion inertial length d_i = c / ω_pi  [m].

    Below this scale the ideal MHD frozen-in condition breaks down and
    dispersive kinetic Alfvén waves govern the dynamics.

        d_i = c / √(n e² / (ε₀ m_i))

    Parameters
    ----------
    n_m3 : float
        Plasma density [m⁻³].
    mass_amu : float
        Ion mass [amu].

    Returns
    -------
    float
        Ion inertial length [m].
    """
    m_i = mass_amu * M_PROTON
    omega_pi_sq = n_m3 * EV_TO_J ** 2 / (_EPSILON_0 * m_i)
    if omega_pi_sq <= 0:
        return math.inf
    return _C_LIGHT / math.sqrt(omega_pi_sq)


def flr_correction_factor(Lambda_i: float) -> float:
    """Second-order Northrop FLR correction factor √(1 + ¾ Λᵢ²).

    Returns the multiplicative factor by which Λᵢ_FLR exceeds Λᵢ_0.
    Equals 1.0 at Λᵢ = 0 and grows as Λᵢ increases.

    Parameters
    ----------
    Lambda_i : float
        First-order demagnetization parameter Λᵢ_0.

    Returns
    -------
    float
        FLR correction factor ≥ 1.
    """
    if math.isinf(Lambda_i):
        return math.inf
    return math.sqrt(1.0 + 0.75 * Lambda_i ** 2)

"""Plasma-physics normalization utilities (spec §13.1).

Converts simulation output quantities to the normalized units commonly
used in the magnetic nozzle literature:

* Lengths → ion skin depth:       d_i = c / ω_pi
* Times   → ion cyclotron period: τ_ci = 2π / Ω_ci
* Velocities → ion sound speed:   c_s  = √(T_e / m_i)
* Magnetic field → throat value:  B̃  = B / B_throat

All input quantities use SI units unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants (SI)
_C: float = 2.997924e8  # speed of light [m/s]
_E: float = 1.602176634e-19  # elementary charge [C]
_EPS0: float = 8.854187817e-12  # vacuum permittivity [F/m]
_MU0: float = 4e-7 * math.pi  # vacuum permeability [T·m/A]
_ME: float = 9.1093837015e-31  # electron mass [kg]
_AMU: float = 1.66053906660e-27  # atomic mass unit [kg]


@dataclass
class PlasmaScales:
    """Characteristic plasma scales derived from local plasma parameters.

    Attributes
    ----------
    d_i_m:
        Ion skin depth [m]: ``c / ω_pi``
    tau_ci_s:
        Ion cyclotron period [s]: ``2π / Ω_ci``
    c_s_ms:
        Ion sound speed [m/s]: ``√(T_e / m_i)``
    omega_pi_rads:
        Ion plasma frequency [rad/s]
    omega_ci_rads:
        Ion cyclotron frequency [rad/s]
    """

    d_i_m: float
    tau_ci_s: float
    c_s_ms: float
    omega_pi_rads: float
    omega_ci_rads: float


def compute_plasma_scales(
    n0: float,
    T_e_eV: float,
    B_T: float,
    ion_mass_amu: float = 2.014,  # deuterium
    charge_number: int = 1,
) -> PlasmaScales:
    """Compute characteristic plasma length/time/velocity scales.

    Parameters
    ----------
    n0:
        Plasma number density [m⁻³].
    T_e_eV:
        Electron temperature [eV].
    B_T:
        Local magnetic field magnitude [T].
    ion_mass_amu:
        Ion mass in atomic mass units.  Default: 2.014 (deuterium).
    charge_number:
        Ion charge number (integer multiples of *e*).  Default: 1.

    Returns
    -------
    PlasmaScales
        All four characteristic scales plus intermediate frequencies.
    """
    m_i = ion_mass_amu * _AMU  # [kg]
    q_i = charge_number * _E  # [C]
    T_e_J = T_e_eV * _E  # [J]

    # Ion plasma frequency ω_pi = √(n q²/(ε₀ m_i))
    omega_pi = math.sqrt(n0 * q_i**2 / (_EPS0 * m_i))

    # Ion skin depth d_i = c / ω_pi
    d_i = _C / omega_pi

    # Ion cyclotron frequency Ω_ci = q B / m_i
    omega_ci = q_i * B_T / m_i

    # Ion cyclotron period τ_ci = 2π / Ω_ci
    tau_ci = 2.0 * math.pi / omega_ci if omega_ci > 0.0 else float("inf")

    # Ion sound speed c_s = √(T_e / m_i)
    c_s = math.sqrt(T_e_J / m_i)

    return PlasmaScales(
        d_i_m=d_i,
        tau_ci_s=tau_ci,
        c_s_ms=c_s,
        omega_pi_rads=omega_pi,
        omega_ci_rads=omega_ci,
    )


def normalize_length(length_m: float, scales: PlasmaScales) -> float:
    """Normalize a length to ion skin depths: ``x / d_i``."""
    return length_m / scales.d_i_m


def normalize_time(time_s: float, scales: PlasmaScales) -> float:
    """Normalize a time to ion cyclotron periods: ``t / τ_ci``."""
    return time_s / scales.tau_ci_s


def normalize_velocity(velocity_ms: float, scales: PlasmaScales) -> float:
    """Normalize a velocity to the ion sound speed: ``v / c_s``."""
    return velocity_ms / scales.c_s_ms


def normalize_bfield(B_T: float, B_throat_T: float) -> float:
    """Normalize a magnetic field to the throat value: ``B / B_throat``."""
    if B_throat_T == 0.0:
        raise ValueError("B_throat_T must be non-zero")
    return B_T / B_throat_T


def normalize_density(n: float, n0: float) -> float:
    """Normalize a density to the reference density: ``n / n0``."""
    if n0 == 0.0:
        raise ValueError("n0 must be non-zero")
    return n / n0


def normalize_pressure(P_Pa: float, n0: float, T_e_eV: float) -> float:
    """Normalize pressure to the reference electron thermal pressure.

    Returns ``P / (n0 T_e)`` where T_e is in joules.
    """
    T_e_J = T_e_eV * _E
    denom = n0 * T_e_J
    if denom == 0.0:
        raise ValueError("n0 and T_e_eV must be non-zero")
    return P_Pa / denom

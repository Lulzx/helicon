"""Collision cross-sections and Maxwellian-averaged rate coefficients.

Provides charge-exchange (CX), electron-impact ionization, and radiative
recombination data for propulsion-relevant species: D, H, He, Xe.

All cross-sections are functions of relative energy [eV] or temperature [eV].
Rate coefficients are Maxwellian-averaged: <σv> [m³/s].

References
----------
- Freeman & Jones (1974) — hydrogen CX cross-sections
- Janev et al. (1993) — "Elementary Processes in Hydrogen-Helium Plasmas"
- Haynes (2014) — Xe ionization / recombination
"""

from __future__ import annotations

import numpy as np

# Proton mass [kg]
_MP = 1.6726219236951e-27
# Electron mass [kg]
_ME = 9.1093837015e-31
# Elementary charge [C]
_QE = 1.6021766340e-19

# Species masses [kg]
SPECIES_MASS: dict[str, float] = {
    "H": _MP,
    "D": 2.0 * _MP,
    "T": 3.0 * _MP,
    "He": 4.0 * _MP,
    "Xe": 131.293 * _MP,
}


def cx_cross_section_m2(species: str, energy_eV: float | np.ndarray) -> np.ndarray:
    """Charge-exchange cross-section σ_cx(E) [m²].

    Uses the Freeman-Jones analytic fit for H-like species.
    For Xe, uses a constant value appropriate for ~100 eV ions.

    Parameters
    ----------
    species : str
        Ion species: 'H', 'D', 'T', 'He', 'Xe'
    energy_eV : float or ndarray
        Center-of-mass collision energy [eV]

    Returns
    -------
    ndarray
        Cross-section [m²]
    """
    E = np.asarray(energy_eV, dtype=float)
    E_safe = np.maximum(E, 0.1)  # avoid log(0)

    if species in ("H", "D", "T"):
        # Freeman-Jones fit: σ = A * exp(-B * ln²(E/E0)) [cm²] → [m²]
        # Valid ~0.1–10000 eV
        A = 35.2e-20  # m²
        B = 0.0406
        E0 = 4.04  # eV
        sigma = A * np.exp(-B * np.log(E_safe / E0) ** 2)
    elif species == "He":
        # Simplified Rapp-Francis-type fit [m²]
        A = 15.0e-20
        B = 0.05
        E0 = 10.0
        sigma = A * np.exp(-B * np.log(E_safe / E0) ** 2)
    elif species == "Xe":
        # Roughly constant ~50e-20 m² for Xe+ at thruster-relevant energies
        sigma = 50.0e-20 * np.ones_like(E_safe)
    else:
        msg = f"Unknown species for CX cross-section: {species!r}"
        raise ValueError(msg)

    return np.where(E > 0, sigma, 0.0)


def ionization_rate_m3s(species: str, T_eV: float | np.ndarray) -> np.ndarray:
    """Maxwellian-averaged electron-impact ionization rate coefficient <σv> [m³/s].

    Uses analytic fits from Janev et al. (1993) for H/D, polynomial fits for He/Xe.

    Parameters
    ----------
    species : str
        Neutral species: 'H', 'D', 'T', 'He', 'Xe'
    T_eV : float or ndarray
        Electron temperature [eV]

    Returns
    -------
    ndarray
        Rate coefficient [m³/s]
    """
    T = np.asarray(T_eV, dtype=float)
    T_safe = np.maximum(T, 0.1)

    if species in ("H", "D", "T"):
        # Janev et al. fit: <σv>_ion = exp(sum_i a_i * ln^i(T))  [m³/s]
        # 8-term Chebyshev fit from Janev 1993, valid 1–200 eV
        coeffs = np.array(
            [
                -3.271397e1,
                1.353656e1,
                -5.739329,
                1.563155,
                -2.877056e-1,
                3.482560e-2,
                -2.631976e-3,
                1.018614e-4,
                -1.831480e-6,
            ]
        )
        log_T = np.log(T_safe)
        log_rate = np.polyval(coeffs[::-1], log_T)
        rate = np.exp(log_rate) * 1e-6  # cm³/s → m³/s
    elif species == "He":
        # Simplified fit: rate peaks ~3×10⁻¹⁴ m³/s around 50 eV
        T0 = 15.8  # ionization potential [eV]
        rate = 3e-14 * np.sqrt(T_safe / T0) * np.exp(-T0 / T_safe)
    elif species == "Xe":
        # Xenon ionization potential 12.1 eV; fit from Lieberman & Lichtenberg
        T0 = 12.1
        rate = 1.8e-13 * np.sqrt(T_safe / T0) * np.exp(-T0 / T_safe)
    else:
        msg = f"Unknown species for ionization rate: {species!r}"
        raise ValueError(msg)

    return np.maximum(rate, 0.0)


def recombination_rate_m3s(species: str, T_eV: float | np.ndarray) -> np.ndarray:
    """Maxwellian-averaged radiative recombination rate coefficient [m³/s].

    Uses the Badnell (2006) / Pequignot (1991) fit for H-like ions.

    Parameters
    ----------
    species : str
        Ion species: 'H', 'D', 'T', 'He', 'Xe'
    T_eV : float or ndarray
        Electron temperature [eV]

    Returns
    -------
    ndarray
        Rate coefficient [m³/s]
    """
    T = np.asarray(T_eV, dtype=float)
    T_safe = np.maximum(T, 0.1)

    if species in ("H", "D", "T"):
        # Badnell fit: α = A * (√(T/T0) * (1 + √(T/T0))^(1-B) * (1 + √(T/T1))^(1+B))^-1
        A = 7.982e-11  # cm³/s
        B = 0.7480
        T0 = 3.148  # eV
        T1 = 7.036e5  # eV
        sqT = np.sqrt(T_safe)
        sqT0 = np.sqrt(T0)
        sqT1 = np.sqrt(T1)
        rate = A / (sqT / sqT0 * (1 + sqT / sqT0) ** (1 - B) * (1 + sqT / sqT1) ** (1 + B))
        rate *= 1e-6  # cm³/s → m³/s
    elif species == "He":
        A = 9.356e-10 * 1e-6  # m³/s
        T0 = 3.294  # eV
        T1 = 1.613e6
        sqT = np.sqrt(T_safe)
        rate = A / (sqT / np.sqrt(T0) * (1 + sqT / np.sqrt(T0)) * (1 + sqT / np.sqrt(T1)))
    elif species == "Xe":
        # Generic power-law falloff
        rate = 5e-13 * (T_safe / 10.0) ** (-0.7) * 1e-6  # rough estimate
    else:
        msg = f"Unknown species for recombination rate: {species!r}"
        raise ValueError(msg)

    return np.maximum(rate, 0.0)


def cx_rate_m3s(species: str, T_ion_eV: float | np.ndarray) -> np.ndarray:
    """Maxwellian-averaged charge-exchange rate coefficient <σv>_cx [m³/s].

    Approximated as σ_cx(3kT/2) * v_th_ion for a Maxwellian ion distribution.

    Parameters
    ----------
    species : str
        Ion species
    T_ion_eV : float or ndarray
        Ion temperature [eV]

    Returns
    -------
    ndarray
        Rate coefficient [m³/s]
    """
    T = np.asarray(T_ion_eV, dtype=float)
    T_safe = np.maximum(T, 0.1)

    mass = SPECIES_MASS.get(species, _MP)
    # Thermal velocity: v_th = sqrt(2kT/m)
    v_th = np.sqrt(2.0 * T_safe * _QE / mass)
    sigma = cx_cross_section_m2(species, T_safe)
    return sigma * v_th

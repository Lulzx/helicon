"""Spacecraft interaction model for magnetic nozzle thrusters.

Computes three interaction effects between the exhaust plume and spacecraft:

1. **Backflow fraction** — ions/electrons returning past the thruster plane,
   causing momentum loss and potential surface sputtering.

2. **Spacecraft charging** — floating potential from net electron/ion current
   to exposed surfaces.

3. **Magnetic torque** — net torque on the spacecraft from the nozzle magnetic
   field interacting with spacecraft magnetic moment.

All calculations use analytical approximations from particle exit statistics
or from the coil configuration directly.

References
----------
- Goebel & Katz (2008) — Fundamentals of Electric Propulsion, §11.
- Hofer et al. (2008) — Hall thruster backflow model.
- Birn & Priest (2007) — Reconnection of Magnetic Fields, §4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_QE = 1.602176634e-19  # C
_ME = 9.1093837015e-31  # kg
_K_B = 1.380649e-23  # J/K
_EV_TO_J = _QE


@dataclass
class BackflowResult:
    """Ion/electron backflow analysis.

    Attributes
    ----------
    backflow_fraction : float
        Fraction of propellant mass returning past thruster plane.
    backflow_thrust_loss_N : float
        Thrust reduction due to backflow [N].
    ion_return_current_A : float
        Ion backflow current to spacecraft [A].
    electron_return_current_A : float
        Electron backflow current estimate [A].
    """

    backflow_fraction: float
    backflow_thrust_loss_N: float
    ion_return_current_A: float
    electron_return_current_A: float


@dataclass
class ChargingResult:
    """Spacecraft surface charging from plume particles.

    Attributes
    ----------
    floating_potential_V : float
        Equilibrium surface potential [V].
    electron_current_A : float
        Electron current to spacecraft [A].
    ion_current_A : float
        Ion current to spacecraft [A].
    net_current_A : float
        Net charging current (electron - ion) [A].
    """

    floating_potential_V: float
    electron_current_A: float
    ion_current_A: float
    net_current_A: float


@dataclass
class MagneticTorqueResult:
    """Magnetic torque on spacecraft from nozzle field.

    Attributes
    ----------
    torque_Nm : float
        Magnitude of magnetic torque [N·m].
    dipole_moment_Am2 : float
        Equivalent magnetic dipole moment of thruster [A·m²].
    field_at_spacecraft_T : float
        Estimated nozzle field at spacecraft bus [T].
    """

    torque_Nm: float
    dipole_moment_Am2: float
    field_at_spacecraft_T: float


@dataclass
class SpacecraftInteractionResult:
    """Full spacecraft interaction analysis."""

    backflow: BackflowResult
    charging: ChargingResult
    magnetic_torque: MagneticTorqueResult


def compute_backflow(
    thrust_N: float,
    eta_d: float,
    mdot_kgs: float,
    *,
    v_exhaust_ms: float,
    ion_mass_kg: float = 3.3435837724e-27,  # deuterium
) -> BackflowResult:
    """Estimate ion backflow from detachment efficiency.

    A fraction (1 - η_d)² of ions fail to detach and return toward
    the spacecraft (empirical scaling from Merino & Ahedo 2012).

    Parameters
    ----------
    thrust_N : float
        Net thrust [N].
    eta_d : float
        Detachment efficiency (0–1).
    mdot_kgs : float
        Propellant mass flow rate [kg/s].
    v_exhaust_ms : float
        Exhaust velocity [m/s].
    ion_mass_kg : float
        Ion mass [kg].

    Returns
    -------
    BackflowResult
    """
    backflow_fraction = (1.0 - eta_d) ** 2
    backflow_thrust_loss = thrust_N * backflow_fraction
    mdot_back = mdot_kgs * backflow_fraction
    n_back_flux = mdot_back / ion_mass_kg  # [#/s]
    ion_return_current = n_back_flux * _QE
    # Electrons have higher thermal velocity but are electrostatically confined
    electron_return_current = (
        ion_return_current * math.sqrt(ion_mass_kg / _ME) * math.exp(-1.0)
    )  # approximate ambipolar correction

    return BackflowResult(
        backflow_fraction=backflow_fraction,
        backflow_thrust_loss_N=backflow_thrust_loss,
        ion_return_current_A=ion_return_current,
        electron_return_current_A=electron_return_current,
    )


def compute_spacecraft_charging(
    n_e_m3: float,
    T_e_eV: float,
    v_spacecraft_ms: float = 0.0,
    *,
    surface_area_m2: float = 10.0,
    ion_mass_kg: float = 3.3435837724e-27,
) -> ChargingResult:
    """Estimate spacecraft floating potential in the plume.

    Uses the OML (orbital motion limited) floating potential for a
    planar surface in a plasma with Maxwellian electrons.

    Parameters
    ----------
    n_e_m3 : float
        Plume electron density at spacecraft location [m⁻³].
    T_e_eV : float
        Electron temperature [eV].
    v_spacecraft_ms : float
        Spacecraft velocity relative to plasma [m/s].
    surface_area_m2 : float
        Total exposed surface area [m²].
    ion_mass_kg : float
        Ion mass [kg].

    Returns
    -------
    ChargingResult
    """
    T_e_J = T_e_eV * _EV_TO_J

    # Electron thermal velocity
    v_th_e = math.sqrt(2.0 * T_e_J / _ME)

    # Ion thermal velocity (assume T_i ≈ T_e for bulk plasma near spacecraft)
    v_th_i = math.sqrt(2.0 * T_e_J / ion_mass_kg)

    # Random thermal flux [A/m²]
    j_e = 0.25 * n_e_m3 * v_th_e * _QE
    j_i = 0.25 * n_e_m3 * v_th_i * _QE

    I_e = j_e * surface_area_m2
    I_i = j_i * surface_area_m2

    # Floating potential: φ_f = -(T_e/2e) * ln(m_i / (2π * m_e))
    # (standard OML result for Maxwellian plasma)
    phi_f = -(T_e_eV / 2.0) * math.log(ion_mass_kg / (2.0 * math.pi * _ME))
    # Clamp to physically reasonable range
    phi_f = max(-500.0, min(phi_f, 0.0))

    net_current = I_e - I_i

    return ChargingResult(
        floating_potential_V=phi_f,
        electron_current_A=I_e,
        ion_current_A=I_i,
        net_current_A=net_current,
    )


def compute_magnetic_torque(
    coils: list,
    *,
    spacecraft_magnetic_moment_Am2: float = 1.0,
    spacecraft_distance_m: float = 5.0,
) -> MagneticTorqueResult:
    """Estimate magnetic torque on spacecraft from the thruster coil field.

    Uses the dipole approximation: nozzle coils → equivalent dipole moment,
    then τ = m_spacecraft × B_nozzle(r_spacecraft).

    Parameters
    ----------
    coils : list of Coil
        Nozzle coils.
    spacecraft_magnetic_moment_Am2 : float
        Spacecraft residual magnetic moment [A·m²].
    spacecraft_distance_m : float
        Distance from coil centre to spacecraft centre-of-mass [m].

    Returns
    -------
    MagneticTorqueResult
    """
    _MU0 = 4.0 * math.pi * 1e-7

    # Dipole moment of all coils (sum, assuming all wound in same direction)
    m_coils = sum(math.pi * c.r**2 * c.I for c in coils)  # [A·m²]

    # On-axis dipole field at distance d: B = μ0 * m / (2π * d³)
    d = max(spacecraft_distance_m, 0.1)
    B_at_spacecraft = _MU0 * abs(m_coils) / (2.0 * math.pi * d**3)

    # Maximum torque: τ = m_spacecraft × B (worst case perpendicular)
    torque = spacecraft_magnetic_moment_Am2 * B_at_spacecraft

    return MagneticTorqueResult(
        torque_Nm=torque,
        dipole_moment_Am2=m_coils,
        field_at_spacecraft_T=B_at_spacecraft,
    )


def compute_spacecraft_interaction(
    config,
    *,
    thrust_N: float,
    eta_d: float,
    v_exhaust_ms: float,
    mdot_kgs: float,
    T_e_eV: float | None = None,
    plume_density_m3: float | None = None,
    spacecraft_distance_m: float = 5.0,
    surface_area_m2: float = 10.0,
    spacecraft_magnetic_moment_Am2: float = 1.0,
) -> SpacecraftInteractionResult:
    """Compute all spacecraft interaction effects.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration (coil geometry + plasma parameters).
    thrust_N : float
        Thruster thrust [N].
    eta_d : float
        Detachment efficiency.
    v_exhaust_ms : float
        Exhaust velocity [m/s].
    mdot_kgs : float
        Propellant mass flow rate [kg/s].
    T_e_eV : float, optional
        Electron temperature [eV] (defaults to config.plasma.T_e_eV).
    plume_density_m3 : float, optional
        Plume electron density at spacecraft [m⁻³] (defaults to
        n0 * 1e-3 as rough far-field estimate).
    spacecraft_distance_m : float
        Distance from thruster to spacecraft bus [m].
    surface_area_m2 : float
        Spacecraft surface area exposed to plume [m²].
    spacecraft_magnetic_moment_Am2 : float
        Spacecraft residual magnetic moment [A·m²].

    Returns
    -------
    SpacecraftInteractionResult
    """
    from helicon.fields.biot_savart import Coil

    if T_e_eV is None:
        T_e_eV = config.plasma.T_e_eV
    if plume_density_m3 is None:
        plume_density_m3 = config.plasma.n0 * 1e-3  # rough far-field estimate

    # Ion mass from species
    _ION_MASS = {
        "D+": 3.3435837724e-27,
        "H+": 1.67262192595e-27,
        "He3+": 5.0082353373e-27,
        "He4+": 6.6464764e-27,
        "He2+": 6.6464764e-27,
        "Ar+": 6.6335209e-26,
        "Xe+": 2.1801714e-25,
    }
    ion_species = [s for s in config.plasma.species if s != "e-"]
    ion_mass = _ION_MASS.get(ion_species[0] if ion_species else "D+", 3.34e-27)

    backflow = compute_backflow(
        thrust_N,
        eta_d,
        mdot_kgs,
        v_exhaust_ms=v_exhaust_ms,
        ion_mass_kg=ion_mass,
    )

    charging = compute_spacecraft_charging(
        plume_density_m3,
        T_e_eV,
        surface_area_m2=surface_area_m2,
        ion_mass_kg=ion_mass,
    )

    coils = [Coil(z=c.z, r=c.r, I=c.I) for c in config.nozzle.coils]
    torque = compute_magnetic_torque(
        coils,
        spacecraft_magnetic_moment_Am2=spacecraft_magnetic_moment_Am2,
        spacecraft_distance_m=spacecraft_distance_m,
    )

    return SpacecraftInteractionResult(
        backflow=backflow,
        charging=charging,
        magnetic_torque=torque,
    )

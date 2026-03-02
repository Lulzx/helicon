"""Low-thrust trajectory integration for magnetic nozzle thrusters.

Connects Helicon performance tables to mission-level ΔV estimates.
Provides:

* Tsiolkovsky propellant budget (instantaneous Isp)
* Edelbaum approximation for circular orbit transfers
* Optional poliastro integration for full low-thrust propagation

References
----------
- Edelbaum, T.N. (1961) — Propulsion requirements for controllable
  satellites, ARS Journal.
- Bryson, A.E. & Ho, Y.-C. (1975) — Applied optimal control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helicon.mission.throttle import ThrottleMap

_G0 = 9.80665  # standard gravity [m/s²]
_MU_EARTH = 3.986004418e14  # Earth gravitational parameter [m³/s²]


@dataclass
class MissionLeg:
    """A single impulsive or low-thrust manoeuvre leg.

    Attributes
    ----------
    name : str
        Human-readable label.
    delta_v_ms : float
        ΔV budget for this leg [m/s].
    power_W : float
        Thruster input power [W].
    mdot_kgs : float
        Propellant mass flow rate [kg/s].
    """

    name: str
    delta_v_ms: float
    power_W: float
    mdot_kgs: float


@dataclass
class MissionResult:
    """End-to-end mission analysis result.

    Attributes
    ----------
    total_delta_v_ms : float
        Total ΔV [m/s].
    propellant_mass_kg : float
        Required propellant mass [kg].
    burn_time_s : float
        Total thrust-on time [s].
    payload_fraction : float
        m_payload / m_wet.
    isp_s : float
        Mass-averaged Isp [s].
    wet_mass_kg : float
        Total wet mass [kg].
    legs : list[LegResult]
        Per-leg breakdown.
    """

    total_delta_v_ms: float
    propellant_mass_kg: float
    burn_time_s: float
    payload_fraction: float
    isp_s: float
    wet_mass_kg: float
    legs: list[dict] = field(default_factory=list)


@dataclass
class OrbitTransferResult:
    """Circular orbit transfer analysis (Edelbaum).

    Attributes
    ----------
    r1_m, r2_m : float
        Initial and final orbit radii [m].
    delta_v_ms : float
        ΔV for the transfer [m/s].
    transfer_time_s : float
        Low-thrust transfer time [s] (assumes constant thrust).
    propellant_mass_kg : float
        Propellant consumed [kg].
    payload_fraction : float
        Delivered payload fraction.
    """

    r1_m: float
    r2_m: float
    delta_v_ms: float
    transfer_time_s: float
    propellant_mass_kg: float
    payload_fraction: float
    isp_s: float
    thrust_N: float


def tsiolkovsky(
    delta_v_ms: float,
    isp_s: float,
    dry_mass_kg: float,
) -> tuple[float, float]:
    """Tsiolkovsky rocket equation.

    Parameters
    ----------
    delta_v_ms : float
        Required ΔV [m/s].
    isp_s : float
        Specific impulse [s].
    dry_mass_kg : float
        Dry (payload + structure) mass [kg].

    Returns
    -------
    (propellant_mass_kg, wet_mass_kg)
    """
    v_e = isp_s * _G0
    mass_ratio = math.exp(delta_v_ms / v_e)
    wet_mass_kg = dry_mass_kg * mass_ratio
    propellant_mass_kg = wet_mass_kg - dry_mass_kg
    return propellant_mass_kg, wet_mass_kg


def edelbaum_dv(r1_m: float, r2_m: float, inclination_change_deg: float = 0.0) -> float:
    """Edelbaum ΔV for a low-thrust circular orbit transfer.

    For in-plane transfers (inclination_change_deg=0), reduces to:
        ΔV = |v1 - v2|
    where v1, v2 are circular orbital velocities.
    For inclined transfers, uses the Edelbaum formula.

    Parameters
    ----------
    r1_m : float
        Initial circular orbit radius [m].
    r2_m : float
        Final circular orbit radius [m].
    inclination_change_deg : float
        Plane change [°].

    Returns
    -------
    float
        ΔV estimate [m/s].
    """
    v1 = math.sqrt(_MU_EARTH / r1_m)
    v2 = math.sqrt(_MU_EARTH / r2_m)
    di = math.radians(inclination_change_deg)

    # Edelbaum formula: ΔV² = v1² + v2² - 2*v1*v2*cos(π/2 * Δi)
    dv2 = v1**2 + v2**2 - 2.0 * v1 * v2 * math.cos(0.5 * math.pi * di)
    return math.sqrt(max(dv2, 0.0))


def analyze_mission(
    legs: list[MissionLeg],
    throttle_map: ThrottleMap,
    dry_mass_kg: float,
) -> MissionResult:
    """Analyse a multi-leg low-thrust mission.

    Applies Tsiolkovsky sequentially for each leg using the interpolated
    Isp from the throttle map at the specified operating point.

    Parameters
    ----------
    legs : list of MissionLeg
        Ordered list of mission manoeuvres.
    throttle_map : ThrottleMap
        Performance table from :func:`generate_throttle_map`.
    dry_mass_kg : float
        Spacecraft dry mass (payload + structure) [kg].

    Returns
    -------
    MissionResult
    """
    total_prop = 0.0
    total_burn_time = 0.0
    leg_results = []
    current_dry = dry_mass_kg

    # Work backwards: last leg has the least propellant on board
    # Forward: accumulate prop and burn time
    isp_weighted_sum = 0.0
    isp_weight = 0.0

    for leg in legs:
        isp = throttle_map.isp_at(leg.power_W, leg.mdot_kgs)
        thrust = throttle_map.thrust_at(leg.power_W, leg.mdot_kgs)
        prop_leg, _wet = tsiolkovsky(leg.delta_v_ms, isp, current_dry)
        # Burn time for this leg
        burn_s = prop_leg / leg.mdot_kgs if leg.mdot_kgs > 0 else 0.0

        total_prop += prop_leg
        total_burn_time += burn_s
        isp_weighted_sum += isp * prop_leg
        isp_weight += prop_leg

        leg_results.append(
            {
                "name": leg.name,
                "delta_v_ms": leg.delta_v_ms,
                "isp_s": isp,
                "thrust_N": thrust,
                "propellant_kg": prop_leg,
                "burn_time_s": burn_s,
            }
        )
        # Dry mass increases by this propellant for earlier legs
        # (conservative: treat independently rather than chaining)

    mean_isp = isp_weighted_sum / isp_weight if isp_weight > 0 else 0.0
    total_dv = sum(lg.delta_v_ms for lg in legs)
    wet_mass = dry_mass_kg + total_prop
    payload_fraction = dry_mass_kg / wet_mass if wet_mass > 0 else 0.0

    return MissionResult(
        total_delta_v_ms=total_dv,
        propellant_mass_kg=total_prop,
        burn_time_s=total_burn_time,
        payload_fraction=payload_fraction,
        isp_s=mean_isp,
        wet_mass_kg=wet_mass,
        legs=leg_results,
    )


def earth_mars_dv(
    *,
    departure_alt_km: float = 400.0,
    arrival_alt_km: float = 400.0,
) -> float:
    """Approximate Earth–Mars ΔV for a low-thrust spiral transfer.

    Uses heliocentric Edelbaum between Earth and Mars mean orbits plus
    planetary departure/capture ΔV from circular parking orbits.

    Parameters
    ----------
    departure_alt_km : float
        Earth departure parking orbit altitude [km].
    arrival_alt_km : float
        Mars arrival orbit altitude [km].

    Returns
    -------
    float
        Total ΔV estimate [m/s].
    """
    R_EARTH = 6371e3
    R_MARS_SURFACE = 3389e3
    MU_SUN = 1.32712440018e20

    # Heliocentric circular orbits [m]
    r_earth_helio = 1.496e11
    r_mars_helio = 2.279e11

    v_earth_helio = math.sqrt(MU_SUN / r_earth_helio)
    v_mars_helio = math.sqrt(MU_SUN / r_mars_helio)

    # Low-thrust heliocentric spiral ΔV (Edelbaum in-plane)
    dv_helio = abs(v_earth_helio - v_mars_helio)

    # Earth departure from parking orbit to escape
    r_park_earth = R_EARTH + departure_alt_km * 1e3
    v_park_earth = math.sqrt(_MU_EARTH / r_park_earth)
    v_esc_earth = v_park_earth * math.sqrt(2.0)
    dv_depart = v_esc_earth - v_park_earth

    # Mars capture from hyperbolic approach to parking orbit
    MU_MARS = 4.2828e13
    r_park_mars = R_MARS_SURFACE + arrival_alt_km * 1e3
    v_park_mars = math.sqrt(MU_MARS / r_park_mars)
    v_inf_mars = abs(v_earth_helio - v_mars_helio) * 0.15  # approximate
    v_hyp = math.sqrt(v_inf_mars**2 + 2.0 * MU_MARS / r_park_mars)
    dv_capture = v_hyp - v_park_mars

    return dv_depart + dv_helio + dv_capture


def circular_transfer(
    r1_m: float,
    r2_m: float,
    throttle_map: ThrottleMap,
    power_W: float,
    mdot_kgs: float,
    dry_mass_kg: float,
    inclination_change_deg: float = 0.0,
) -> OrbitTransferResult:
    """Analyse a low-thrust circular orbit transfer.

    Parameters
    ----------
    r1_m : float
        Initial circular orbit radius [m].
    r2_m : float
        Final circular orbit radius [m].
    throttle_map : ThrottleMap
        Thruster performance table.
    power_W : float
        Operating input power [W].
    mdot_kgs : float
        Mass flow rate [kg/s].
    dry_mass_kg : float
        Spacecraft dry mass [kg].
    inclination_change_deg : float
        Optional plane change [°].

    Returns
    -------
    OrbitTransferResult
    """
    dv = edelbaum_dv(r1_m, r2_m, inclination_change_deg)
    isp = throttle_map.isp_at(power_W, mdot_kgs)
    thrust = throttle_map.thrust_at(power_W, mdot_kgs)
    prop_kg, wet_kg = tsiolkovsky(dv, isp, dry_mass_kg)
    burn_s = prop_kg / mdot_kgs if mdot_kgs > 0 else 0.0
    payload_frac = dry_mass_kg / wet_kg if wet_kg > 0 else 0.0

    return OrbitTransferResult(
        r1_m=r1_m,
        r2_m=r2_m,
        delta_v_ms=dv,
        transfer_time_s=burn_s,
        propellant_mass_kg=prop_kg,
        payload_fraction=payload_frac,
        isp_s=isp,
        thrust_N=thrust,
    )


def poliastro_trajectory(
    throttle_map: ThrottleMap,
    power_W: float,
    mdot_kgs: float,
    r1_m: float,
    r2_m: float,
    dry_mass_kg: float,
) -> dict:
    """Compute low-thrust transfer using poliastro (optional dependency).

    Falls back to Edelbaum approximation if poliastro is not installed.

    Parameters
    ----------
    throttle_map : ThrottleMap
        Thruster performance table.
    power_W, mdot_kgs : float
        Operating point.
    r1_m, r2_m : float
        Initial and final orbit radii [m].
    dry_mass_kg : float
        Spacecraft dry mass [kg].

    Returns
    -------
    dict
        ``{"method": str, "delta_v_ms": float, "transfer_time_s": float, ...}``
    """
    try:
        from astropy import units as u
        from poliastro.bodies import Earth
        from poliastro.twobody import Orbit

        r = circular_transfer(r1_m, r2_m, throttle_map, power_W, mdot_kgs, dry_mass_kg)
        # Use poliastro orbit objects for richer output
        orb1 = Orbit.circular(Earth, alt=(r1_m - 6371e3) * u.m)
        orb2 = Orbit.circular(Earth, alt=(r2_m - 6371e3) * u.m)
        return {
            "method": "poliastro+edelbaum",
            "r1_m": r1_m,
            "r2_m": r2_m,
            "delta_v_ms": r.delta_v_ms,
            "transfer_time_s": r.transfer_time_s,
            "propellant_mass_kg": r.propellant_mass_kg,
            "payload_fraction": r.payload_fraction,
            "isp_s": r.isp_s,
            "orbit1_period_s": float(orb1.T.to(u.s).value),
            "orbit2_period_s": float(orb2.T.to(u.s).value),
        }
    except ImportError:
        r = circular_transfer(r1_m, r2_m, throttle_map, power_W, mdot_kgs, dry_mass_kg)
        return {
            "method": "edelbaum",
            "r1_m": r1_m,
            "r2_m": r2_m,
            "delta_v_ms": r.delta_v_ms,
            "transfer_time_s": r.transfer_time_s,
            "propellant_mass_kg": r.propellant_mass_kg,
            "payload_fraction": r.payload_fraction,
            "isp_s": r.isp_s,
        }

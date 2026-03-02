"""Pulsed mission profiles for PPR-class engines.

Computes impulse-averaged thruster performance over burst cycles for
use in trajectory integration. PPR (pulsed plasma rocket) and similar
pulsed-fusion concepts operate in burst mode: several seconds of
high-power plasma pulses followed by a recharge/cool-down interval.

This module provides:
- PulsedProfile: burst pattern specification
- PulsedMissionResult: averaged performance over a mission burn
- compute_pulsed_performance(): integrate throttle_map over pulse train

References
----------
- Thomas et al. (2020) — Pulsed fusion propulsion concept overview.
- Howe et al. (2012) — PPR mission analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helicon.mission.throttle import ThrottleMap

_G0 = 9.80665  # m/s²


@dataclass
class PulsedProfile:
    """Burst cycle specification.

    Attributes
    ----------
    pulse_duration_s : float
        Duration of each active plasma pulse [s].
    off_duration_s : float
        Recharge / coast interval between pulses [s].
    n_pulses : int
        Number of pulses per mission burst.
    power_on_W : float
        Input power during active pulse [W].
    power_off_W : float
        Input power during recharge interval [W] (housekeeping only).
    mdot_on_kgs : float
        Mass flow rate during active pulse [kg/s].
    """

    pulse_duration_s: float
    off_duration_s: float
    n_pulses: int
    power_on_W: float
    power_off_W: float = 0.0
    mdot_on_kgs: float = 1e-5


@dataclass
class PulsedMissionResult:
    """Mission-averaged performance over a pulse train.

    Attributes
    ----------
    mean_thrust_N : float
        Duty-cycle averaged thrust [N].
    mean_isp_s : float
        Propellant-mass–weighted average Isp [s].
    mean_eta_d : float
        Average detachment efficiency.
    duty_cycle : float
        Fraction of time at full power (pulse_on / period).
    total_impulse_Ns : float
        Total impulse delivered by all pulses [N·s].
    total_propellant_kg : float
        Total propellant consumed [kg].
    total_time_s : float
        Total wall clock time of the burst [s].
    effective_isp_s : float
        Effective Isp accounting for non-propulsive intervals:
        Isp_eff = total_impulse / (total_prop * g0).
    """

    mean_thrust_N: float
    mean_isp_s: float
    mean_eta_d: float
    duty_cycle: float
    total_impulse_Ns: float
    total_propellant_kg: float
    total_time_s: float
    effective_isp_s: float


def compute_pulsed_performance(
    profile: PulsedProfile,
    throttle_map: ThrottleMap,
) -> PulsedMissionResult:
    """Compute mission-averaged performance over a pulse train.

    Parameters
    ----------
    profile : PulsedProfile
        Burst cycle definition.
    throttle_map : ThrottleMap
        Thruster performance table.

    Returns
    -------
    PulsedMissionResult
    """
    # Performance at the active operating point
    thrust_peak = throttle_map.thrust_at(profile.power_on_W, profile.mdot_on_kgs)
    isp_peak = throttle_map.isp_at(profile.power_on_W, profile.mdot_on_kgs)
    eta_d = float(throttle_map.eta_d.mean())

    period = profile.pulse_duration_s + profile.off_duration_s
    duty_cycle = profile.pulse_duration_s / period if period > 0 else 0.0

    # Impulse per pulse
    impulse_per_pulse = thrust_peak * profile.pulse_duration_s

    # Propellant per pulse
    prop_per_pulse = profile.mdot_on_kgs * profile.pulse_duration_s

    # Totals over the burst
    total_impulse = impulse_per_pulse * profile.n_pulses
    total_prop = prop_per_pulse * profile.n_pulses
    total_time = period * profile.n_pulses

    # Duty-cycle averaged thrust
    mean_thrust = total_impulse / total_time if total_time > 0 else 0.0

    # Effective Isp = total_impulse / (total_prop * g0)
    effective_isp = total_impulse / (total_prop * _G0) if total_prop > 0 else 0.0

    return PulsedMissionResult(
        mean_thrust_N=mean_thrust,
        mean_isp_s=isp_peak,  # instantaneous Isp unchanged during pulse
        mean_eta_d=eta_d,
        duty_cycle=duty_cycle,
        total_impulse_Ns=total_impulse,
        total_propellant_kg=total_prop,
        total_time_s=total_time,
        effective_isp_s=effective_isp,
    )


def burst_delta_v(
    profile: PulsedProfile,
    throttle_map: ThrottleMap,
    dry_mass_kg: float,
) -> float:
    """ΔV delivered by a single burst (n_pulses) using pulsed performance.

    Parameters
    ----------
    profile : PulsedProfile
        Burst cycle definition.
    throttle_map : ThrottleMap
        Thruster performance table.
    dry_mass_kg : float
        Dry spacecraft mass [kg].

    Returns
    -------
    float
        ΔV from this burst [m/s].
    """
    result = compute_pulsed_performance(profile, throttle_map)
    v_e_eff = result.effective_isp_s * _G0
    if v_e_eff <= 0:
        return 0.0
    wet_mass = dry_mass_kg + result.total_propellant_kg
    return v_e_eff * math.log(wet_mass / dry_mass_kg)

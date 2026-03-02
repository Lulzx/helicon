"""Helicon mission integration package (v1.3).

Provides throttle curve generation, trajectory analysis, spacecraft
interaction modelling, and pulsed mission profiles.
"""

from __future__ import annotations

from helicon.mission.pulsed import (
    PulsedMissionResult,
    PulsedProfile,
    burst_delta_v,
    compute_pulsed_performance,
)
from helicon.mission.spacecraft import (
    BackflowResult,
    ChargingResult,
    MagneticTorqueResult,
    SpacecraftInteractionResult,
    compute_backflow,
    compute_magnetic_torque,
    compute_spacecraft_charging,
    compute_spacecraft_interaction,
)
from helicon.mission.throttle import (
    OperatingPoint,
    ThrottleMap,
    ThrottleResult,
    generate_throttle_map,
)
from helicon.mission.trajectory import (
    MissionLeg,
    MissionResult,
    OrbitTransferResult,
    analyze_mission,
    circular_transfer,
    earth_mars_dv,
    edelbaum_dv,
    poliastro_trajectory,
    tsiolkovsky,
)

__all__ = [
    "BackflowResult",
    "ChargingResult",
    "MagneticTorqueResult",
    "MissionLeg",
    "MissionResult",
    "OperatingPoint",
    "OrbitTransferResult",
    "PulsedMissionResult",
    "PulsedProfile",
    "SpacecraftInteractionResult",
    "ThrottleMap",
    "ThrottleResult",
    "analyze_mission",
    "burst_delta_v",
    "circular_transfer",
    "compute_backflow",
    "compute_magnetic_torque",
    "compute_pulsed_performance",
    "compute_spacecraft_charging",
    "compute_spacecraft_interaction",
    "earth_mars_dv",
    "edelbaum_dv",
    "generate_throttle_map",
    "poliastro_trajectory",
    "tsiolkovsky",
]

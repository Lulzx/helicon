"""Multi-thruster array simulation.

Models plume-plume interaction for spacecraft with 2-4 magnetic nozzle
thrusters operating simultaneously.

Usage::

    from helicon.multithruster import ThrusterArray, ArrayConfig

    config = ArrayConfig(
        n_thrusters=2,
        separation_m=0.5,
        thrust_N=[0.1, 0.1],
        isp_s=[3000.0, 3000.0],
        plume_half_angle_deg=[15.0, 15.0],
    )
    array = ThrusterArray(config)
    result = array.compute()
    print(result.total_thrust_N, result.interaction_penalty)
"""

from helicon.multithruster.array import ArrayConfig, ArrayResult, ThrusterArray
from helicon.multithruster.interaction import (
    PlumeModel,
    compute_overlap_factor,
    compute_plume_interaction,
)

__all__ = [
    "ArrayConfig",
    "ArrayResult",
    "PlumeModel",
    "ThrusterArray",
    "compute_overlap_factor",
    "compute_plume_interaction",
]

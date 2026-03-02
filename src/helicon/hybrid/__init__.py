"""Fluid-hybrid electron solver module (spec v1.2).

MLX-native CGL (Chew-Goldberger-Low) double-adiabatic electron fluid
coupled to WarpX kinetic ions, with LHDI anomalous transport closure.

This module implements the fluid-hybrid electron path, cutting wall time
~10× vs. full kinetic electrons while targeting <15% error on η_d for
the Merino-Ahedo case (spec v1.2 exit criterion).
"""

from helicon.hybrid.cgl_electron import CGLElectronFluid, CGLState
from helicon.hybrid.coupler import HybridCoupler, HybridState, IonMoments
from helicon.hybrid.lhdi import LHDIParams, LHDITransport

__all__ = [
    "CGLElectronFluid",
    "CGLState",
    "HybridCoupler",
    "HybridState",
    "IonMoments",
    "LHDIParams",
    "LHDITransport",
]

"""Neutral particle dynamics module (spec v1.2).

Self-consistent Monte Carlo neutral dynamics with:
- Charge exchange (CX)
- Electron-impact ionization
- Radiative recombination

MLX-accelerated on Apple Silicon; NumPy fallback on all platforms.
"""

from helicon.neutrals.cross_sections import (
    SPECIES_MASS,
    cx_cross_section_m2,
    cx_rate_m3s,
    ionization_rate_m3s,
    recombination_rate_m3s,
)
from helicon.neutrals.monte_carlo import (
    MCCCollider,
    MCCResult,
    NeutralDynamics,
    NeutralParticles,
)

__all__ = [
    "SPECIES_MASS",
    "MCCCollider",
    "MCCResult",
    "NeutralDynamics",
    "NeutralParticles",
    "cx_cross_section_m2",
    "cx_rate_m3s",
    "ionization_rate_m3s",
    "recombination_rate_m3s",
]

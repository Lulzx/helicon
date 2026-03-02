"""Real-time reduced model for magnetic nozzle detachment onset (v2.5).

Predicts detachment from local MHD invariants and sheath parameters.
Designed for embedded control loops — pure scalar arithmetic, no numpy.

The open question in magnetic nozzle propulsion is not *simulation* but
*control*: given real-time sensor data, when is plasma about to detach,
and can the coil current be adjusted to keep it in the optimal regime?

Three independent criteria determine detachment:

1. **Alfvénic**: plasma becomes super-Alfvénic when field weakens too fast.
   M_A = v_z / v_A > 1 → momentum coupling breaks down.

2. **Electron β**: thermal pressure overcomes magnetic pressure.
   β_e = n k T_e / (B²/2μ₀) > 0.1 → field lines can be pushed aside.

3. **Ion demagnetization**: ion Larmor radius exceeds field scale length.
   Λ_i = r_Li / L_B > 1 → ions stop following field lines.

References
----------
- Merino, M. & Ahedo, E. (2011). Plasma detachment in a propulsive
  magnetic nozzle. *Physics of Plasmas* 18, 053504.
- Ahedo, E. & Merino, M. (2010). Two-dimensional supersonic plasma
  acceleration. *Physics of Plasmas* 17, 073501.
- Little, J.M. & Choueiri, E.Y. (2013). Thrust and efficiency model.
  *Physics of Plasmas* 20, 103501.
"""

from __future__ import annotations

from helicon.detach.calibration import (
    CalibrationRecord,
    CalibrationResult,
    DetachmentCalibrator,
)
from helicon.detach.control import (
    ControlState,
    ControlUpdate,
    LyapunovController,
)
from helicon.detach.invariants import (
    alfven_mach,
    alfven_velocity,
    bohm_velocity,
    electron_beta,
    field_scale_length,
    ion_larmor_radius,
    ion_magnetization,
)
from helicon.detach.inverse import (
    InferredState,
    ThrustInverter,
    ThrustObservation,
)
from helicon.detach.kinetic import (
    alfven_mach_kinetic,
    bohm_velocity_full,
    flr_correction_factor,
    ion_inertial_length,
    ion_magnetization_flr,
    larmor_radius_maxwellian,
)
from helicon.detach.model import (
    DetachmentOnsetModel,
    DetachmentState,
    PlasmaState,
)
from helicon.detach.sheath import (
    SheathCorrectedState,
    apply_sheath_correction,
    debye_length,
    sheath_potential,
)

__all__ = [
    "CalibrationRecord",
    "CalibrationResult",
    "ControlState",
    "ControlUpdate",
    "DetachmentCalibrator",
    "DetachmentOnsetModel",
    "DetachmentState",
    "InferredState",
    "LyapunovController",
    "PlasmaState",
    "SheathCorrectedState",
    "ThrustInverter",
    "ThrustObservation",
    "alfven_mach",
    "alfven_mach_kinetic",
    "alfven_velocity",
    "apply_sheath_correction",
    "bohm_velocity",
    "bohm_velocity_full",
    "debye_length",
    "electron_beta",
    "field_scale_length",
    "flr_correction_factor",
    "ion_inertial_length",
    "ion_larmor_radius",
    "ion_magnetization",
    "ion_magnetization_flr",
    "larmor_radius_maxwellian",
    "sheath_potential",
]

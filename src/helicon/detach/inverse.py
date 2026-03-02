"""Inverse problem: infer plasma state from thrust measurements.

In flight, direct plasma diagnostics (Langmuir probes, Thomson
scattering) are unavailable.  The only observables are:

* F — thrust, measured by a load cell or strain gauge [N]
* ṁ — propellant mass flow rate [kg/s]
* B — coil current × calibration constant [T] (at nozzle throat)
* P — RF power (optional) [W]

This module inverts these four observables to reconstruct (n, v_z, M_A)
using three conservation laws:

1. **Mass flux**: ṁ = n m_i v_z A_throat   → n = ṁ/(m_i v_z A_throat)
2. **Momentum**:  F = ṁ v_ex η_T(R_B)      → v_ex = F/(ṁ η_T)
3. **Alfvénic**: M_A = v_ex / v_A(B, n)    (closed form)

Combining (1)–(3) gives the **key result** — Alfvén Mach number
directly from observables::

    M_A = [F/(ṁ η_T B)] · √(μ₀ ṁ m_i / A_throat)

No plasma diagnostics required.

Additionally, the **gradient test** uses d F/d B to encode regime:

* d F/d B > 0 : attached (more field → better confinement → thrust)
* d F/d B ≈ 0 : near the optimal Alfvénic boundary
* d F/d B < 0 : detached (field impedes free expansion)

References
----------
- Little, J.M. & Choueiri, E.Y. (2013). Thrust and efficiency model
  for electron-driven magnetic nozzles. *Physics of Plasmas* 20, 103501.
- Merino, M. & Ahedo, E. (2011). *Physics of Plasmas* 18, 053504.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from helicon.detach.invariants import (
    M_PROTON,
    alfven_velocity,
)


@dataclass
class ThrustObservation:
    """Direct measurement data available in flight hardware.

    Attributes
    ----------
    F_thrust_N : float
        Measured thrust [N].
    m_dot_kg_s : float
        Propellant mass flow rate [kg/s].
    B_throat_T : float
        Magnetic field at the nozzle throat [T].
    A_throat_m2 : float
        Throat cross-sectional area [m²].
    mass_amu : float
        Ion species mass [amu].
    Te_eV_nominal : float
        Nominal electron temperature [eV] from ground characterisation.
    dBdz_T_per_m : float
        Estimated axial B gradient at throat [T/m].
    """

    F_thrust_N: float
    m_dot_kg_s: float
    B_throat_T: float
    A_throat_m2: float
    mass_amu: float = 1.0
    Te_eV_nominal: float = 50.0
    dBdz_T_per_m: float = -2.0


@dataclass
class InferredState:
    """Plasma state inferred from thrust measurements.

    Attributes
    ----------
    n_m3 : float
        Inferred electron density [m⁻³].
    vz_ms : float
        Inferred exhaust (axial) velocity [m/s].
    Ti_eV_est : float
        Ion temperature estimate [eV] (assumed ≈ Te_nominal).
    alfven_mach : float
        Inferred Alfvén Mach number M_A (−1 if not computable).
    detachment_score : float
        Composite detachment score with inferred parameters.
    confidence : float
        Confidence ∈ [0, 1]; based on thrust residual and M_A validity.
    residual : float
        Fractional thrust residual |F_model − F_obs| / F_obs.
    """

    n_m3: float
    vz_ms: float
    Ti_eV_est: float
    alfven_mach: float
    detachment_score: float
    confidence: float
    residual: float

    def to_plasma_state(self, B_T: float, dBdz_T_per_m: float, mass_amu: float = 1.0):
        """Convert to :class:`~helicon.detach.model.PlasmaState`."""
        from helicon.detach.model import PlasmaState

        return PlasmaState(
            n_m3=self.n_m3,
            Te_eV=self.Ti_eV_est,
            Ti_eV=self.Ti_eV_est,
            B_T=B_T,
            dBdz_T_per_m=dBdz_T_per_m,
            vz_ms=self.vz_ms,
            mass_amu=mass_amu,
        )


class ThrustInverter:
    """Infer plasma state from thrust measurements.

    Uses conservation laws and the Alfvénic criterion to reconstruct
    the plasma state from the observables available in flight.

    Parameters
    ----------
    mirror_ratio : float
        Estimated mirror ratio R_B = B_throat / B_exit (default 5.0).
        Determines thrust efficiency η_T = 1 − 1/√R_B.
    """

    def __init__(self, mirror_ratio: float = 5.0) -> None:
        if mirror_ratio <= 1.0:
            raise ValueError(f"mirror_ratio must be > 1, got {mirror_ratio}")
        self.mirror_ratio = mirror_ratio
        self.eta_T = 1.0 - 1.0 / math.sqrt(mirror_ratio)

    def invert(self, obs: ThrustObservation) -> InferredState:
        """Infer plasma state from a :class:`ThrustObservation`.

        Derivation
        ----------
        Step 1 — exhaust velocity from momentum flux::

            v_ex = F / (ṁ · η_T)

        Step 2 — density from mass flux::

            n = ṁ / (m_i · v_ex · A_throat)

        Step 3 — closed-form Alfvén Mach number::

            M_A = v_ex / v_A = v_ex · √(μ₀ n m_i) / B
                = (F / (ṁ η_T B)) · √(μ₀ ṁ m_i / A_throat)

        This is the key result: M_A directly from observables (F, ṁ, B, A).
        """
        m_i = obs.mass_amu * M_PROTON

        # Step 1: exhaust velocity
        if obs.m_dot_kg_s <= 0 or self.eta_T <= 0:
            v_ex = 0.0
        else:
            v_ex = obs.F_thrust_N / (obs.m_dot_kg_s * self.eta_T)

        # Step 2: density
        if v_ex <= 0 or obs.A_throat_m2 <= 0:
            n = 0.0
        else:
            n = obs.m_dot_kg_s / (m_i * v_ex * obs.A_throat_m2)

        # Step 3: Alfvén Mach number (closed form)
        if n <= 0 or obs.B_throat_T <= 0:
            M_A = math.inf
        else:
            va = alfven_velocity(obs.B_throat_T, n, obs.mass_amu)
            M_A = v_ex / va if va > 0 else math.inf

        # Step 4: temperatures for composite score
        Te = obs.Te_eV_nominal
        Ti = Te  # assume T_i ≈ T_e (common approximation)

        # Step 5: composite score
        from helicon.detach.model import DetachmentOnsetModel, PlasmaState

        score = 0.0
        if n > 0 and v_ex > 0:
            state = PlasmaState(
                n_m3=n,
                Te_eV=Te,
                Ti_eV=Ti,
                B_T=obs.B_throat_T,
                dBdz_T_per_m=obs.dBdz_T_per_m,
                vz_ms=v_ex,
                mass_amu=obs.mass_amu,
            )
            score = DetachmentOnsetModel().assess(state).detachment_score

        # Consistency check
        F_model = obs.m_dot_kg_s * v_ex * self.eta_T
        residual = abs(F_model - obs.F_thrust_N) / max(abs(obs.F_thrust_N), 1e-12)
        confidence = max(0.0, 1.0 - residual) * (1.0 if math.isfinite(M_A) else 0.0)

        return InferredState(
            n_m3=n,
            vz_ms=v_ex,
            Ti_eV_est=Ti,
            alfven_mach=M_A if math.isfinite(M_A) else -1.0,
            detachment_score=score,
            confidence=confidence,
            residual=residual,
        )

    def gradient_test(
        self,
        obs_low: ThrustObservation,
        obs_high: ThrustObservation,
    ) -> dict:
        """Infer detachment regime from the ∂F/∂B gradient.

        The thrust-vs-field gradient encodes the detachment regime:

        * d F/d B > 0 : **attached** — increasing B improves confinement
        * d F/d B ≈ 0 : **optimal** — operating near the Alfvénic boundary
        * d F/d B < 0 : **detached** — field impedes free plasma expansion

        Parameters
        ----------
        obs_low, obs_high : ThrustObservation
            Two measurements at different B fields (same ṁ).

        Returns
        -------
        dict
            Keys: ``dF_dB`` [N/T], ``regime`` (str),
            ``optimal_B_T`` (estimate of peak-thrust field).
        """
        dB = obs_high.B_throat_T - obs_low.B_throat_T
        dF = obs_high.F_thrust_N - obs_low.F_thrust_N
        dF_dB = dF / dB if abs(dB) > 1e-12 else 0.0

        # Relative tolerance: 5% of thrust per unit B
        ref = abs(obs_low.F_thrust_N / max(abs(dB), 1e-12))
        tol = 0.05 * ref

        if dF_dB > tol:
            regime = "attached"
        elif dF_dB < -tol:
            regime = "detached"
        else:
            regime = "optimal"

        # Crude optimal-B estimate: linear interpolation to dF/dB = 0
        if regime == "optimal":
            optimal_B = 0.5 * (obs_low.B_throat_T + obs_high.B_throat_T)
        else:
            # Extrapolate to where slope crosses zero (gradient = max thrust)
            inv = self.invert(obs_low)
            shift = (0.5 - inv.detachment_score) * abs(dB)
            optimal_B = obs_low.B_throat_T + shift

        return {"dF_dB": dF_dB, "regime": regime, "optimal_B_T": optimal_B}

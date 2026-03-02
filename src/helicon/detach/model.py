"""Real-time reduced model for magnetic nozzle detachment onset.

The :class:`DetachmentOnsetModel` evaluates three independent criteria
and combines them into a scalar *detachment score* S ∈ [0, 1]:

  S = w_A · f(M_A) + w_β · f(β_e) + w_Λ · f(Λ_i)

where each f(·) is a piecewise-linear normalisation that saturates at 1.

Design principles
-----------------
* **No external dependencies** — only ``math`` from the standard library.
  The model can be transpiled to C in under 50 lines.
* **Interpretable** — each criterion maps to a named physical mechanism.
* **Tuneable** — weights and thresholds are constructor arguments so the
  model can be calibrated against WarpX data or experimental campaigns.
* **Control-oriented** — the ``onset_B_T`` output tells the controller
  exactly what field is needed to *prevent* detachment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from helicon.detach.invariants import (
    M_PROTON,
    MU0,
    alfven_mach,
    bohm_velocity,
    electron_beta,
    ion_magnetization,
)


@dataclass
class PlasmaState:
    """Local plasma state snapshot for detachment assessment.

    All fields are in SI units.  Instantiate one of these per sensor
    reading; pass it to :meth:`DetachmentOnsetModel.assess`.

    Attributes
    ----------
    n_m3 : float
        Electron/ion density [m⁻³].
    Te_eV : float
        Electron temperature [eV].
    Ti_eV : float
        Ion temperature [eV].
    B_T : float
        Magnetic field magnitude [T].
    dBdz_T_per_m : float
        Axial magnetic field gradient ∂B/∂z [T/m].
        Negative downstream of the throat (B decreasing).
    vz_ms : float
        Axial bulk plasma velocity [m/s].
    mass_amu : float
        Ion species mass [amu].  Default 1.0 (H⁺).
    """

    n_m3: float
    Te_eV: float
    Ti_eV: float
    B_T: float
    dBdz_T_per_m: float
    vz_ms: float
    mass_amu: float = 1.0

    def validate(self) -> None:
        """Raise :exc:`ValueError` for unphysical inputs."""
        if self.n_m3 <= 0:
            raise ValueError(f"n_m3 must be positive, got {self.n_m3}")
        if self.Te_eV <= 0:
            raise ValueError(f"Te_eV must be positive, got {self.Te_eV}")
        if self.Ti_eV <= 0:
            raise ValueError(f"Ti_eV must be positive, got {self.Ti_eV}")
        if self.B_T <= 0:
            raise ValueError(f"B_T must be positive, got {self.B_T}")
        if self.mass_amu <= 0:
            raise ValueError(f"mass_amu must be positive, got {self.mass_amu}")
        if self.vz_ms < 0:
            raise ValueError(f"vz_ms must be >= 0, got {self.vz_ms}")


@dataclass
class DetachmentState:
    """Output of a single :meth:`DetachmentOnsetModel.assess` call.

    Attributes
    ----------
    alfven_mach : float
        M_A = v_z / v_A.  Detachment strongly indicated when > 1.
    electron_beta : float
        β_e = n k T_e / (B²/2μ₀).  Significant above 0.1.
    ion_magnetization : float
        Λᵢ = r_Lᵢ / L_B.  Ions demagnetize when > 1.
    bohm_mach : float
        v_z / c_s.  Sheath Bohm criterion requires > 1 at sheath edge.
    detachment_score : float
        Composite score S ∈ [0, 1].
    is_detached : bool
        True when S ≥ score_detached threshold (default 0.70).
    is_imminent : bool
        True when S ≥ score_imminent threshold (default 0.40).
    onset_B_T : float
        B field [T] at which the Alfvénic criterion predicts detachment
        for the current density and velocity.  Reduce the nozzle field
        toward this value to approach the detachment boundary, or increase
        above it to maintain attachment.
    control_signal : float
        Normalised control output ∈ [0, 1] suitable for PWM or DAC output
        to a coil driver.  Defined as (1 − S), so 1.0 = fully attached,
        0.0 = fully detached.
    """

    alfven_mach: float
    electron_beta: float
    ion_magnetization: float
    bohm_mach: float
    detachment_score: float
    is_detached: bool
    is_imminent: bool
    onset_B_T: float
    control_signal: float

    def summary(self) -> str:
        """Return a human-readable status string."""
        if self.is_detached:
            status = "DETACHED "
        elif self.is_imminent:
            status = "IMMINENT "
        else:
            status = "ATTACHED "
        return (
            f"Detachment Status: {status} (score={self.detachment_score:.3f})\n"
            f"  Alfvén Mach:       {self.alfven_mach:.4f}  (threshold: 1.0)\n"
            f"  Electron β:        {self.electron_beta:.4f}  (threshold: 0.1)\n"
            f"  Ion magnetization: {self.ion_magnetization:.4f}  (threshold: 1.0)\n"
            f"  Bohm Mach:         {self.bohm_mach:.4f}\n"
            f"  Onset B:           {self.onset_B_T:.6f} T\n"
            f"  Control signal:    {self.control_signal:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "alfven_mach": self.alfven_mach,
            "electron_beta": self.electron_beta,
            "ion_magnetization": self.ion_magnetization,
            "bohm_mach": self.bohm_mach,
            "detachment_score": self.detachment_score,
            "is_detached": self.is_detached,
            "is_imminent": self.is_imminent,
            "onset_B_T": self.onset_B_T,
            "control_signal": self.control_signal,
        }


@dataclass
class ScanResult:
    """Result of a multi-position axial detachment scan.

    Attributes
    ----------
    z_m : list[float]
        Axial positions [m].
    states : list[DetachmentState]
        Per-position detachment assessment.
    onset_z_m : float or None
        First axial position where detachment is imminent, or None.
    detach_z_m : float or None
        First axial position where plasma is classified as detached.
    """

    z_m: list[float]
    states: list[DetachmentState]
    onset_z_m: float | None
    detach_z_m: float | None

    def score_profile(self) -> list[float]:
        """Return detachment scores at each position."""
        return [s.detachment_score for s in self.states]


class DetachmentOnsetModel:
    """Real-time reduced model for magnetic nozzle detachment onset.

    Combines three independent criteria into a scalar detachment score.
    Designed for control loops — all computation is scalar arithmetic;
    no imports beyond ``math``; evaluates in < 1 µs on any FPU-equipped MCU.

    Parameters
    ----------
    w_alfven : float
        Weight for Alfvénic criterion (0.45).
    w_beta : float
        Weight for electron β criterion (0.30).
    w_ion_mag : float
        Weight for ion demagnetization (0.25).
        *Weights must sum to 1.0.*
    beta_crit : float
        β_e value at which the β criterion contributes fully (0.15).
    score_detached : float
        Score threshold for DETACHED classification (0.70).
    score_imminent : float
        Score threshold for IMMINENT warning (0.40).

    Notes
    -----
    Default weights are motivated by the Merino-Ahedo (2011) analysis
    which identifies the Alfvénic transition as the primary mechanism,
    β-driven expansion as secondary, and ion demagnetization as tertiary.
    Recalibrate ``w_*`` against WarpX validation data for best accuracy.
    """

    def __init__(
        self,
        w_alfven: float = 0.45,
        w_beta: float = 0.30,
        w_ion_mag: float = 0.25,
        beta_crit: float = 0.15,
        score_detached: float = 0.70,
        score_imminent: float = 0.40,
    ) -> None:
        if abs(w_alfven + w_beta + w_ion_mag - 1.0) > 1e-9:
            raise ValueError(
                f"weights must sum to 1.0, got {w_alfven + w_beta + w_ion_mag:.6f}"
            )
        self.w_alfven = w_alfven
        self.w_beta = w_beta
        self.w_ion_mag = w_ion_mag
        self.beta_crit = beta_crit
        self.score_detached = score_detached
        self.score_imminent = score_imminent

    def assess(self, state: PlasmaState) -> DetachmentState:
        """Assess detachment state from a local plasma measurement.

        Parameters
        ----------
        state : PlasmaState
            Current local plasma conditions.

        Returns
        -------
        DetachmentState
            Detachment assessment with diagnostics and control signal.
        """
        state.validate()

        # --- Compute invariants ---
        M_A = alfven_mach(state.vz_ms, state.B_T, state.n_m3, state.mass_amu)
        beta_e = electron_beta(state.n_m3, state.Te_eV, state.B_T)
        lambda_i = ion_magnetization(
            state.Ti_eV, state.B_T, state.dBdz_T_per_m, state.mass_amu
        )
        c_s = bohm_velocity(state.Te_eV, state.mass_amu)
        bohm_mach = state.vz_ms / c_s if c_s > 0 else math.inf

        # --- Criterion contributions, each ∈ [0, 1] ---
        # Alfvénic: linearly rises from 0 (M_A=0) to 1 (M_A=2)
        f_alfven = min(1.0, max(0.0, M_A / 2.0))

        # β: rises from 0 to 1 over [0, 2·beta_crit]
        f_beta = min(1.0, max(0.0, beta_e / (2.0 * self.beta_crit)))

        # Ion demagnetization: saturates at Λᵢ = 2
        f_ion = min(1.0, max(0.0, lambda_i / 2.0))

        # Weighted score
        score = self.w_alfven * f_alfven + self.w_beta * f_beta + self.w_ion_mag * f_ion

        # --- Onset B prediction (Alfvénic criterion) ---
        # v_A = v_z at onset → B_onset = v_z √(μ₀ n mᵢ)
        m_i = state.mass_amu * M_PROTON
        onset_B = state.vz_ms * math.sqrt(MU0 * state.n_m3 * m_i) if state.vz_ms > 0 else 0.0

        return DetachmentState(
            alfven_mach=M_A,
            electron_beta=beta_e,
            ion_magnetization=lambda_i,
            bohm_mach=bohm_mach,
            detachment_score=score,
            is_detached=score >= self.score_detached,
            is_imminent=score >= self.score_imminent,
            onset_B_T=onset_B,
            control_signal=1.0 - score,
        )

    def scan_z(
        self,
        plasma_states: list[PlasmaState],
        z_positions: list[float],
    ) -> ScanResult:
        """Assess detachment across multiple axial positions.

        Parameters
        ----------
        plasma_states : list[PlasmaState]
            Plasma state at each axial position.
        z_positions : list[float]
            Corresponding axial positions [m].

        Returns
        -------
        ScanResult
            Per-position assessments, onset location, and detachment location.
        """
        if len(plasma_states) != len(z_positions):
            raise ValueError(
                f"plasma_states ({len(plasma_states)}) and "
                f"z_positions ({len(z_positions)}) must have equal length"
            )
        results = [self.assess(s) for s in plasma_states]
        onset_z: float | None = None
        detach_z: float | None = None
        for z, r in zip(z_positions, results):
            if onset_z is None and r.is_imminent:
                onset_z = z
            if detach_z is None and r.is_detached:
                detach_z = z
        return ScanResult(
            z_m=list(z_positions),
            states=results,
            onset_z_m=onset_z,
            detach_z_m=detach_z,
        )

    def control_recommendation(self, state: PlasmaState) -> dict:
        """Return a plain-dict control recommendation for embedded use.

        Suitable for serialising to JSON and sending to a coil controller.
        All values are either scalars or booleans — no numpy arrays.

        Returns
        -------
        dict with keys:
            ``score`` — detachment score ∈ [0, 1]
            ``control_signal`` — (1 − score); drive coil PWM to this
            ``is_detached``, ``is_imminent`` — boolean flags
            ``onset_B_T`` — target B for onset boundary
            ``recommended_action`` — human-readable string
        """
        ds = self.assess(state)
        if ds.is_detached:
            action = "INCREASE_B: raise coil current to restore attachment"
        elif ds.is_imminent:
            action = "MONITOR: approach detachment boundary — consider small B increase"
        else:
            action = "NOMINAL: plasma attached, optimize for thrust efficiency"
        return {
            "score": ds.detachment_score,
            "control_signal": ds.control_signal,
            "is_detached": ds.is_detached,
            "is_imminent": ds.is_imminent,
            "onset_B_T": ds.onset_B_T,
            "recommended_action": action,
        }

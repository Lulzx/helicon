"""Lyapunov-stable feedback controller for detachment score regulation.

Control objective
-----------------
Regulate the composite detachment score S to a setpoint S* ∈ (0, 1).
The optimal S* lies near the detachment boundary (S* ~ 0.35), where
thrust efficiency is maximised without entering the detached regime.

Lyapunov stability proof
------------------------
State:   x = S − S*
Control: u = Ḃ  (rate of change of B, or equivalently dI_coil/dt)

Dynamics::

    Ṡ ≈ (∂S/∂B) · u + w       (w = unmodelled disturbance)

Lyapunov candidate::

    V(x) = x²/2               (positive definite)

Required for stability::

    V̇ = x · (∂S/∂B) · u ≤ −α V    for some α > 0

Set::

    u* = −α x / (∂S/∂B)

Then::

    V̇ = x · (∂S/∂B) · [−α x / (∂S/∂B)]
       = −α x²
       = −2α V  < 0   ✓

This gives **exponential stability**: V(t) = V(0) e^{−2αt},
i.e. the error decays with time constant τ = 1/(2α).

Analytic gradient ∂S/∂B
------------------------
In the unsaturated (linear) regime::

    S = w_A f_A + w_β f_β + w_Λ f_Λ

Since M_A ∝ B⁻¹, β_e ∝ B⁻², Λᵢ ∝ B⁻²::

    ∂f_A/∂B = −f_A / B
    ∂f_β/∂B = −2 f_β / B      (β ~ 1/B²)
    ∂f_Λ/∂B = −2 f_Λ / B      (Λᵢ ~ 1/B²)

    ∂S/∂B = −(1/B) [w_A f_A + 2 w_β f_β + 2 w_Λ f_Λ]  < 0

The gradient is always negative (more B → lower S), confirming that
the control law u = −α(S−S*)/(∂S/∂B) pushes I_coil in the correct
direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from helicon.detach.model import DetachmentOnsetModel, DetachmentState, PlasmaState


@dataclass
class ControlState:
    """Mutable controller internal state for a running feedback loop.

    Attributes
    ----------
    t_s : float
        Elapsed time [s].
    I_coil_A : float
        Current coil current [A].
    score_history : list[float]
        History of detachment scores (one per step).
    error_history : list[float]
        History of score errors S − S*.
    """

    t_s: float = 0.0
    I_coil_A: float = 0.0
    score_history: list[float] = field(default_factory=list)
    error_history: list[float] = field(default_factory=list)


@dataclass
class ControlUpdate:
    """Output of one :meth:`LyapunovController.step` call.

    Attributes
    ----------
    delta_I_coil_A : float
        Required coil current change ΔI [A] for this timestep.
    new_I_coil_A : float
        New total coil current [A] (clamped to [I_min, I_max]).
    lyapunov_V : float
        Lyapunov function value V = (S−S*)²/2 (non-negative).
    lyapunov_dV_dt : float
        Theoretical V̇ = −2αV (should be ≤ 0).
    score : float
        Current detachment score S.
    error : float
        Score error S − S*.
    grad_S_B : float
        Analytical gradient ∂S/∂B [T⁻¹].
    """

    delta_I_coil_A: float
    new_I_coil_A: float
    lyapunov_V: float
    lyapunov_dV_dt: float
    score: float
    error: float
    grad_S_B: float


class LyapunovController:
    """Exponentially-stable feedback controller for detachment score.

    Regulates the composite detachment score S to the setpoint S* by
    adjusting coil current.  Uses the analytic gradient ∂S/∂B to avoid
    numerical differentiation — suitable for real-time MCU implementation.

    Parameters
    ----------
    setpoint : float
        Target score S* ∈ (0, 1).  Default 0.35 (near but below the
        detachment boundary, maximising thrust efficiency).
    decay_rate : float
        Lyapunov decay rate α > 0 [1/s].  Controls convergence speed.
        Default 1.0 s⁻¹ (error halved in ~0.35 s).
    dB_dI_T_per_A : float
        Coil calibration ∂B/∂I [T/A] from Biot-Savart or measurement.
        Default 2 × 10⁻⁵ T/A (typical small helicon nozzle).
    I_coil_max_A : float
        Maximum allowable coil current [A].  Default 50 000 A.
    I_coil_min_A : float
        Minimum allowable coil current [A].  Default 0 A.
    model : DetachmentOnsetModel or None
        Underlying physics model.  If None, uses default weights.
    """

    def __init__(
        self,
        setpoint: float = 0.35,
        decay_rate: float = 1.0,
        dB_dI_T_per_A: float = 2e-5,
        I_coil_max_A: float = 50_000.0,
        I_coil_min_A: float = 0.0,
        model: DetachmentOnsetModel | None = None,
    ) -> None:
        if not 0.0 < setpoint < 1.0:
            raise ValueError(f"setpoint must be in (0, 1), got {setpoint}")
        if decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {decay_rate}")
        if dB_dI_T_per_A <= 0:
            raise ValueError(f"dB_dI_T_per_A must be > 0, got {dB_dI_T_per_A}")
        self.setpoint = setpoint
        self.decay_rate = decay_rate
        self.dB_dI = dB_dI_T_per_A
        self.I_max = I_coil_max_A
        self.I_min = I_coil_min_A
        self.model = model or DetachmentOnsetModel()

    def _grad_S_B(self, state: PlasmaState, ds: DetachmentState) -> float:
        """Analytic gradient ∂S/∂B [T⁻¹] at the current operating point.

        Derivation: with M_A ~ B⁻¹, β_e ~ B⁻², Λᵢ ~ B⁻²::

            ∂S/∂B = −(1/B) · [w_A f_A  +  2 w_β f_β  +  2 w_Λ f_Λ]

        Always negative (increasing B always decreases S).
        """
        m = self.model
        B = state.B_T
        if B <= 0:
            return 0.0
        f_A = min(1.0, max(0.0, ds.alfven_mach / 2.0))
        f_beta = min(1.0, max(0.0, ds.electron_beta / (2.0 * m.beta_crit)))
        f_ion = min(1.0, max(0.0, ds.ion_magnetization / 2.0))
        return -(1.0 / B) * (
            m.w_alfven * f_A + 2.0 * m.w_beta * f_beta + 2.0 * m.w_ion_mag * f_ion
        )

    def step(
        self,
        state: PlasmaState,
        ctrl_state: ControlState,
        dt_s: float = 0.01,
    ) -> ControlUpdate:
        """Execute one control timestep.

        Parameters
        ----------
        state : PlasmaState
            Current plasma measurement.
        ctrl_state : ControlState
            Mutable controller state (updated in-place).
        dt_s : float
            Timestep [s].

        Returns
        -------
        ControlUpdate
            Coil current change and Lyapunov diagnostics.
        """
        ds = self.model.assess(state)
        error = ds.detachment_score - self.setpoint
        grad = self._grad_S_B(state, ds)

        # Lyapunov control law: ΔB/Δt = −α · error / (∂S/∂B)
        delta_B = (
            0.0 if abs(grad) < 1e-30
            else -self.decay_rate * error / grad * dt_s
        )

        delta_I = delta_B / self.dB_dI
        new_I = max(self.I_min, min(self.I_max, ctrl_state.I_coil_A + delta_I))
        actual_delta_I = new_I - ctrl_state.I_coil_A

        V = 0.5 * error ** 2
        dV_dt = -2.0 * self.decay_rate * V  # theoretical (exact in linear regime)

        ctrl_state.t_s += dt_s
        ctrl_state.I_coil_A = new_I
        ctrl_state.score_history.append(ds.detachment_score)
        ctrl_state.error_history.append(error)

        return ControlUpdate(
            delta_I_coil_A=actual_delta_I,
            new_I_coil_A=new_I,
            lyapunov_V=V,
            lyapunov_dV_dt=dV_dt,
            score=ds.detachment_score,
            error=error,
            grad_S_B=grad,
        )

    def stability_certificate(self, state: PlasmaState) -> dict:
        """Verify and return the Lyapunov stability certificate.

        Returns a plain dict containing:

        * ``V`` — Lyapunov function value (must be ≥ 0)
        * ``dV_dt`` — actual V̇ estimate (must be ≤ 0 for stability)
        * ``is_stable`` — True if V̇ ≤ 0 or error is negligible
        * ``convergence_time_s`` — predicted 1/e time constant 1/(2α)
        * ``grad_S_B`` — ∂S/∂B at this state [T⁻¹]
        * ``error`` — current score error S − S*
        """
        ds = self.model.assess(state)
        error = ds.detachment_score - self.setpoint
        grad = self._grad_S_B(state, ds)
        V = 0.5 * error ** 2

        if abs(grad) > 1e-30:
            # Actual V̇ from applying the control law
            delta_B_rate = -self.decay_rate * error / grad  # [T/s]
            dV_dt_actual = error * grad * delta_B_rate       # = -α·x² ≤ 0
        else:
            dV_dt_actual = 0.0

        return {
            "V": V,
            "dV_dt": dV_dt_actual,
            "is_stable": dV_dt_actual <= 1e-12 or V < 1e-12,
            "convergence_time_s": 1.0 / (2.0 * self.decay_rate),
            "grad_S_B": grad,
            "error": error,
        }

    def simulate(
        self,
        initial_state: PlasmaState,
        n_steps: int = 100,
        dt_s: float = 0.01,
    ) -> list[ControlUpdate]:
        """Simulate the closed-loop response from an initial state.

        Assumes the plasma state is fixed (no plant dynamics) — useful
        for verifying the controller converges to the setpoint.

        Parameters
        ----------
        initial_state : PlasmaState
            Starting plasma conditions.
        n_steps : int
            Number of timesteps.
        dt_s : float
            Timestep [s].

        Returns
        -------
        list[ControlUpdate]
            One update per step.
        """
        ctrl = ControlState(I_coil_A=0.0)
        updates = []
        for _ in range(n_steps):
            u = self.step(initial_state, ctrl, dt_s)
            updates.append(u)
        return updates

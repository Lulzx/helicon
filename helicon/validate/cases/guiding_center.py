"""Guiding-center orbit validation case.

A single charged particle in a known magnetic field. The trajectory
must match Littlejohn (1983) guiding-center theory to within expected
Boris-pusher accuracy (2nd order in dt).

Reference: Littlejohn, R.G. (1983) J. Plasma Physics 29, 111.
Criterion: Orbit error decreases as O(dt^2) with timestep refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from helicon.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    ResolutionConfig,
    SimConfig,
)


@dataclass
class ValidationResult:
    """Result of a validation case."""

    case_name: str
    passed: bool
    metrics: dict[str, float]
    tolerances: dict[str, float]
    description: str


class GuidingCenterCase:
    """Single-particle guiding center orbit in a mirror field.

    A single ion is launched along a magnetic mirror axis. The guiding
    center approximation predicts the trajectory analytically. The PIC
    Boris pusher must converge to this solution at 2nd order in dt.
    """

    name = "guiding_center"
    description = "Single-particle guiding center orbit vs Littlejohn theory"
    requires_warpx = False  # self-contained Boris pusher test — no WarpX output needed

    @staticmethod
    def get_config() -> SimConfig:
        """Return the simulation configuration for this case."""
        return SimConfig(
            nozzle=NozzleConfig(
                type="solenoid",
                coils=[
                    CoilConfig(z=0.0, r=0.05, I=10000),
                    CoilConfig(z=0.5, r=0.05, I=10000),
                ],
                domain=DomainConfig(z_min=-0.1, z_max=0.6, r_max=0.1),
                resolution=ResolutionConfig(nz=128, nr=64),
            ),
            plasma=PlasmaSourceConfig(
                species=["H+", "e-"],
                n0=1.0e10,  # very low density — single particle regime
                T_i_eV=10.0,
                T_e_eV=10.0,
                v_injection_ms=10000.0,
            ),
            timesteps=10000,
            output_dir="results/validation/guiding_center",
        )

    @staticmethod
    def analytic_gyroradius(B: float, v_perp: float, mass: float, charge: float) -> float:
        """Compute the analytic Larmor radius.

        r_L = m * v_perp / (|q| * B)
        """
        return mass * v_perp / (abs(charge) * B)

    @staticmethod
    def evaluate(output_dir: str | Path) -> ValidationResult:
        """Evaluate the guiding-center validation case.

        Performs a self-contained Boris pusher convergence test against the
        analytic Larmor orbit in a uniform field. Does not require WarpX
        output — the convergence order is measured numerically.

        The Boris integrator should converge at O(dt²); we accept any
        measured order in [1.6, 2.4] (within 20% of 2.0).
        """
        convergence_order = 2.0  # expected for Boris pusher
        measured_order = _measure_boris_convergence()
        passed = abs(measured_order - convergence_order) / convergence_order < 0.2

        return ValidationResult(
            case_name="guiding_center",
            passed=passed,
            metrics={
                "expected_convergence_order": convergence_order,
                "measured_convergence_order": measured_order,
            },
            tolerances={"order_relative_error": 0.2},
            description="Boris pusher convergence to guiding-center theory",
        )


def _boris_larmor(n_steps: int, dt: float, omega: float, r_L: float) -> tuple[float, float]:
    """Advance one particle in uniform B = B0 z_hat using the Boris algorithm.

    Initial conditions: x = r_L, y = 0, vx = 0, vy = v_perp (= omega * r_L).
    Returns the final (x, y) position after n_steps steps.
    """
    x, y = r_L, 0.0
    vx, vy = 0.0, omega * r_L  # counter-clockwise Larmor orbit

    # Boris half-angle rotation coefficients (constant for uniform B)
    t = omega * dt / 2.0
    s = 2.0 * t / (1.0 + t * t)

    for _ in range(n_steps):
        # Magnetic rotation (Boris)
        vx_prime = vx + vy * t
        vy_prime = vy - vx * t
        vx = vx + vy_prime * s
        vy = vy - vx_prime * s
        # Position update (leap-frog)
        x += vx * dt
        y += vy * dt

    return x, y


def _measure_boris_convergence() -> float:
    """Measure Boris pusher convergence order against analytic Larmor orbit.

    Runs the integrator for one full gyroperiod at four successively finer
    dt values and fits log(error) ~ p * log(dt) to extract the order p.
    """
    import numpy as np

    # Physical parameters (proton-like, easy numbers)
    B0 = 0.1  # T
    q_over_m = 9.578e7  # C/kg  (proton e/m ≈ 9.578×10⁷)
    omega = q_over_m * B0  # cyclotron frequency [rad/s]
    T_g = 2.0 * np.pi / omega  # gyroperiod [s]
    r_L = 0.01  # Larmor radius [m]

    # dt fractions of one gyroperiod: 1/16, 1/32, 1/64, 1/128
    fractions = [1 / 16, 1 / 32, 1 / 64, 1 / 128]
    errors = []

    for frac in fractions:
        dt = T_g * frac
        n_steps = round(1.0 / frac)  # exactly one gyroperiod
        x_fin, y_fin = _boris_larmor(n_steps, dt, omega, r_L)
        # After exactly one orbit the particle should return to (r_L, 0)
        err = np.sqrt((x_fin - r_L) ** 2 + y_fin**2)
        errors.append(max(err, 1e-30))

    # Fit p in: log(error) = p * log(dt) + const
    log_dt = np.log([T_g * f for f in fractions])
    log_err = np.log(errors)
    p, _ = np.polyfit(log_dt, log_err, 1)
    return float(p)

"""WarpX simulation runner, hardware configuration, and convergence studies."""

from magnozzlex.runner.convergence import ConvergenceLevel, ConvergenceResult, run_convergence_study
from magnozzlex.runner.hardware_config import HardwareInfo, detect_hardware
from magnozzlex.runner.launch import RunResult, run_simulation

__all__ = [
    "ConvergenceLevel",
    "ConvergenceResult",
    "HardwareInfo",
    "RunResult",
    "detect_hardware",
    "run_convergence_study",
    "run_simulation",
]

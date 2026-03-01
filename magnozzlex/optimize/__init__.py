"""Nozzle design optimization — parameter scans, Bayesian optimization,
Sobol sensitivity analysis, and MLX gradient-based coil optimization.
"""

from magnozzlex.optimize.objectives import OptimizationResult, optimize_coils_mlx
from magnozzlex.optimize.scan import (
    ParameterRange,
    ScanPoint,
    ScanResult,
    generate_scan_points,
    run_scan,
)
from magnozzlex.optimize.sensitivity import SobolResult, compute_sobol, saltelli_sample
from magnozzlex.optimize.surrogate import BayesianOptimizer, GPSurrogate

__all__ = [
    "BayesianOptimizer",
    "GPSurrogate",
    "OptimizationResult",
    "ParameterRange",
    "ScanPoint",
    "ScanResult",
    "SobolResult",
    "compute_sobol",
    "generate_scan_points",
    "optimize_coils_mlx",
    "run_scan",
    "saltelli_sample",
]

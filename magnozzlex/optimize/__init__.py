"""Nozzle design optimization — parameter scans, Bayesian optimization,
Sobol sensitivity analysis, MLX gradient-based coil optimization,
analytical pre-screening, and Pareto front computation.
"""

from magnozzlex.optimize.analytical import (
    NozzleScreeningResult,
    divergence_half_angle,
    mirror_ratio,
    screen_geometry,
    thrust_coefficient_paraxial,
    thrust_efficiency,
)
from magnozzlex.optimize.objectives import OptimizationResult, optimize_coils_mlx
from magnozzlex.optimize.pareto import ParetoResult, hypervolume_indicator, pareto_front
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
    "NozzleScreeningResult",
    "OptimizationResult",
    "ParameterRange",
    "ParetoResult",
    "ScanPoint",
    "ScanResult",
    "SobolResult",
    "compute_sobol",
    "divergence_half_angle",
    "generate_scan_points",
    "hypervolume_indicator",
    "mirror_ratio",
    "optimize_coils_mlx",
    "pareto_front",
    "run_scan",
    "saltelli_sample",
    "screen_geometry",
    "thrust_coefficient_paraxial",
    "thrust_efficiency",
]

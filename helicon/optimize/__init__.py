"""Nozzle design optimization — parameter scans, Bayesian optimization,
Sobol sensitivity analysis, MLX gradient-based coil optimization,
analytical pre-screening, Pareto front computation, and engineering
constraint evaluation.
"""

from helicon.optimize.analytical import (
    NozzleScreeningResult,
    divergence_half_angle,
    mirror_ratio,
    screen_geometry,
    thrust_coefficient_paraxial,
    thrust_efficiency,
)
from helicon.optimize.constraints import (
    CoilConstraintResult,
    CoilConstraints,
    evaluate_constraints,
    make_constrained_objective,
)
from helicon.optimize.gradient import (
    GradientOptimizer,
    GradientOptimizerConfig,
    GradientResult,
    optimize_mirror_ratio,
)
from helicon.optimize.objectives import OptimizationResult, optimize_coils_mlx
from helicon.optimize.pareto import ParetoResult, hypervolume_indicator, pareto_front
from helicon.optimize.scan import (
    ParameterRange,
    ScanPoint,
    ScanResult,
    generate_scan_points,
    run_scan,
)
from helicon.optimize.sensitivity import SobolResult, compute_sobol, saltelli_sample
from helicon.optimize.surrogate import BayesianOptimizer, GPSurrogate

__all__ = [
    "BayesianOptimizer",
    "CoilConstraintResult",
    "CoilConstraints",
    "GPSurrogate",
    "GradientOptimizer",
    "GradientOptimizerConfig",
    "GradientResult",
    "NozzleScreeningResult",
    "OptimizationResult",
    "ParameterRange",
    "ParetoResult",
    "ScanPoint",
    "ScanResult",
    "SobolResult",
    "compute_sobol",
    "divergence_half_angle",
    "evaluate_constraints",
    "generate_scan_points",
    "hypervolume_indicator",
    "make_constrained_objective",
    "mirror_ratio",
    "optimize_coils_mlx",
    "optimize_mirror_ratio",
    "pareto_front",
    "run_scan",
    "saltelli_sample",
    "screen_geometry",
    "thrust_coefficient_paraxial",
    "thrust_efficiency",
]

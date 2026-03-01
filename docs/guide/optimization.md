# Optimization

MagNozzleX provides analytical pre-screening, Bayesian optimization, and Pareto front analysis.

## Analytical Pre-Screening

Fast coil geometry evaluation without PIC simulation:

```python
from magnozzlex.optimize.analytical import screen_geometry
from magnozzlex.fields.biot_savart import Coil

coils = [Coil(z=0.0, r=0.12, I=50000.0)]
result = screen_geometry(coils, z_min=-0.3, z_max=2.0)
print(f"Mirror ratio:       R_B = {result.mirror_ratio:.2f}")
print(f"Thrust efficiency:  η_T = {result.thrust_efficiency:.3f}")
print(f"Divergence angle:   θ = {result.divergence_half_angle_deg:.1f}°")
```

The analytical model applies paraxial approximation theory (Breizman-Arefiev / Little-Choueiri):

$$\eta_T = 1 - \frac{1}{R_B}$$

where $R_B = B_{throat}/B_{exhaust}$ is the magnetic mirror ratio.

## Constrained Optimization

```python
from magnozzlex.optimize.constraints import CoilConstraints, evaluate_constraints

constraints = CoilConstraints(
    max_current_MA_turns=0.1,   # 100 kA-turns per coil
    min_separation_m=0.05,      # 5 cm between coils
    max_field_T=5.0,            # 5 T peak field
)

violations = evaluate_constraints(coils, constraints)
print(violations)  # empty list if all satisfied
```

## Bayesian Optimization

Requires `pip install "magnozzlex[botorch]"`:

```python
from magnozzlex.optimize.bayesian import BayesianOptimizer, SearchSpace

space = SearchSpace(
    parameters={
        "coils.0.I": (20000, 80000),
        "coils.0.r": (0.05, 0.20),
        "coils.0.z": (-0.10, 0.0),
    },
    objectives=["thrust_N", "detachment_momentum"],
)

optimizer = BayesianOptimizer(
    base_config=config,
    space=space,
    n_initial=10,
    n_iterations=40,
    output_base="bo_results/",
)
result = optimizer.run(dry_run=True)
print(f"Best thrust: {result.best_thrust_N:.4f} N")
```

## Pareto Front Analysis

For multi-objective optimization:

```python
from magnozzlex.optimize.pareto import ParetoFront

# objectives: list of (thrust_N, efficiency) tuples
objectives = [(r.thrust_N, r.detachment_momentum) for r in reports]
front = ParetoFront.compute(objectives, maximize=[True, True])

print(f"Pareto front has {len(front.points)} points")
print(f"Hypervolume: {front.hypervolume:.4f}")
for pt in front.points:
    print(f"  Thrust={pt[0]:.4f} N, η_det={pt[1]:.3f}")
```

## MLX Gradient-Based Optimization

For Apple Silicon, the MLX backend supports `mx.grad` for coil current optimization:

```python
import mlx.core as mx
from magnozzlex.fields.biot_savart import compute_bfield_mlx_differentiable, Grid

grid = Grid(z_min=-0.3, z_max=2.0, r_max=0.5, nz=64, nr=32)
r_flat = mx.array(grid.r_flat.astype("float32"))
z_flat = mx.array(grid.z_flat.astype("float32"))

def objective(coil_params):
    Br, Bz = compute_bfield_mlx_differentiable(coil_params, r_flat, z_flat)
    # Maximize B_z at throat (z=0, r=0)
    throat_idx = mx.argmin((r_flat**2 + z_flat**2))
    return -Bz[throat_idx]   # minimize negative = maximize

coil_params = mx.array([[0.0, 0.12, 50000.0]], dtype=mx.float32)
grad_fn = mx.grad(objective)
grads = grad_fn(coil_params)
print(f"∂obj/∂I = {grads[0, 2].item():.6e}")
```

## Grid Convergence

Verify that simulation results are resolution-independent:

```python
from magnozzlex.runner.convergence import run_convergence_study

result = run_convergence_study(
    config,
    resolutions=[(128, 64), (256, 128), (512, 256)],
    output_base="convergence/",
    dry_run=True,
)
print(f"Convergence order: {result.convergence_order:.2f}")
print(f"Extrapolated thrust: {result.extrapolated_thrust_N:.4f} N")
```

Richardson extrapolation gives the grid-converged thrust estimate.

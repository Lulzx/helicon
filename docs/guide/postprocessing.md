# Post-Processing

MagNozzleX post-processing extracts propulsion metrics from WarpX simulation output.

## Generating a Run Report

```python
from magnozzlex.postprocess.report import generate_report, save_report

report = generate_report("results/dfd")
print(f"Thrust: {report.thrust_N:.4f} N")
print(f"Isp: {report.isp_s:.1f} s")
print(f"η_T (momentum): {report.detachment_momentum:.3f}")
save_report(report, "results/dfd/report.json")
```

## Thrust Calculation

```python
from magnozzlex.postprocess.thrust import compute_thrust

thrust = compute_thrust("results/dfd")
print(f"Thrust: {thrust.thrust_N:.4f} N")
print(f"Isp: {thrust.isp_s:.1f} s")
print(f"T/P: {thrust.thrust_to_power_mN_kW:.2f} mN/kW")
```

## Detachment Metrics

```python
from magnozzlex.postprocess.detachment import compute_detachment

det = compute_detachment("results/dfd")
print(f"η_det (momentum): {det.eta_momentum:.3f}")
print(f"η_det (particle): {det.eta_particle:.3f}")
print(f"η_det (energy):   {det.eta_energy:.3f}")
```

Three efficiency definitions:

| Metric | Definition | Notes |
|--------|-----------|-------|
| `eta_momentum` | Downstream momentum flux / total ion momentum | ~0.9 well-detached |
| `eta_particle` | Escaped particles / injected particles | Particle counting |
| `eta_energy` | Axial kinetic energy / total kinetic energy | Direction quality |

## Plume Analysis

```python
from magnozzlex.postprocess.plume import compute_plume_metrics

plume = compute_plume_metrics("results/dfd")
print(f"Half-angle: {plume.half_angle_deg:.1f}°")
print(f"Beam efficiency: {plume.beam_efficiency:.3f}")
```

## Particle Moments

```python
from magnozzlex.postprocess.moments import compute_moments

mom = compute_moments("results/dfd", species="D+", step=10000)
print(f"n: {mom.density.max():.2e} m^-3")
print(f"T_i: {mom.temperature_eV.mean():.0f} eV")
```

## Pulsed Engine Metrics

For pulsed/FRC-type engines:

```python
from magnozzlex.postprocess.pulsed import compute_pulsed_metrics

pulsed = compute_pulsed_metrics("results/frc_pulsed")
print(f"Impulse bit: {pulsed.impulse_Ns:.4f} N·s")
print(f"Energy per pulse: {pulsed.energy_J:.1f} J")
print(f"Specific impulse: {pulsed.isp_s:.0f} s")
```

## PropBench Export

Export results to the PropBench interchange format:

```python
from magnozzlex.postprocess.report import generate_report
from magnozzlex.postprocess.propbench import to_propbench, save_propbench

report = generate_report("results/dfd")
pb = to_propbench(report, config=config)
save_propbench(pb, "results/dfd/propbench.json")
```

The PropBench JSON schema is interoperable with other propulsion simulation codes.

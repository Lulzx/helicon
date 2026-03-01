# Parameter Scans

MagNozzleX supports automated parameter scans for multi-dimensional design space exploration.

## CLI

```bash
magnozzlex scan --config my_nozzle.yaml \
    --vary "coils.0.I:20000:80000:5" \
    --vary "coils.0.r:0.05:0.20:5" \
    --output scan_results/ \
    --method lhc
```

Sampling methods: `grid` (full factorial), `lhc` (Latin hypercube), `random`.

## Python API

```python
from magnozzlex.optimize.scan import build_scan_configs, ScanConfig

scan = ScanConfig(
    base_config=config,
    parameters={
        "coils.0.I": (20000, 80000),
        "coils.0.r": (0.05, 0.20),
    },
    n_samples=25,
    method="lhc",
)
configs = build_scan_configs(scan)
print(f"Generated {len(configs)} configs")
```

## Running a Batch Scan

```python
from magnozzlex.runner.batch import run_local_batch, BatchConfig

batch = BatchConfig(n_workers=4)
result = run_local_batch(configs, output_base="scan_results/", batch_config=batch, dry_run=True)
print(f"{result.n_completed}/{result.n_total} completed")
```

## SLURM Cluster Submission

```python
from magnozzlex.runner.batch import submit_batch, BatchConfig

batch = BatchConfig(
    backend="slurm",
    slurm_partition="gpu",
    slurm_account="my_project",
    slurm_time="08:00:00",
    slurm_cpus_per_task=16,
)
submit_batch(configs, batch, output_base="scan_results/")
```

## PBS/Torque Submission

```python
batch = BatchConfig(
    backend="pbs",
    pbs_queue="default",
    pbs_walltime="08:00:00",
    pbs_nodes=1,
    pbs_ppn=16,
)
submit_batch(configs, batch, output_base="scan_results/")
```

## Collecting Results

After runs complete:

```python
from magnozzlex.postprocess.report import load_report
import pathlib

reports = []
for path in sorted(pathlib.Path("scan_results").glob("run_*/report.json")):
    reports.append(load_report(path))

thrusts = [r.thrust_N for r in reports]
efficiencies = [r.detachment_momentum for r in reports]
```

## Gaussian Process Surrogate

Fit a surrogate model to scan results for fast function evaluations:

```python
from magnozzlex.optimize.surrogate import GPSurrogate

X = [[c.nozzle.coils[0].I, c.nozzle.coils[0].r] for c in configs]
y = [r.thrust_N for r in reports]

surrogate = GPSurrogate()
surrogate.fit(X, y)
thrust_pred, std = surrogate.predict([[50000, 0.12]])
print(f"Predicted thrust: {thrust_pred[0]:.4f} ± {std[0]:.4f} N")
```

See [Optimization](optimization.md) for surrogate-guided search.

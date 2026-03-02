# Running Simulations

## Single Run

```bash
helicon run --preset dfd --output results/dfd
helicon run --config my_nozzle.yaml --output results/my_run
helicon run --config my_nozzle.yaml --dry-run  # generate inputs only
```

Via Python API:

```python
from helicon.runner.launch import run_simulation
from helicon.config.parser import SimConfig

config = SimConfig.from_preset("dfd")
result = run_simulation(config, output_dir="results/dfd", dry_run=True)
```

## Hardware Detection

Helicon auto-detects hardware and configures WarpX accordingly:

```python
from helicon.runner.hardware_config import detect_hardware

hw = detect_hardware()
print(hw.summary())
# Platform: Darwin (Apple Silicon M4 Pro)
# WarpX backend: cpu (OpenMP)
# MLX available: True
# OMP threads: 12
```

## Batch Runs

For parameter scans, submit multiple runs in parallel:

```python
from helicon.runner.batch import run_local_batch, BatchConfig

configs = [...]  # list of SimConfig from a parameter scan
result = run_local_batch(configs, output_base="scan_results/", n_workers=4, dry_run=True)
print(f"{result.n_completed}/{len(configs)} completed")
```

For SLURM clusters:

```python
from helicon.runner.batch import submit_batch, BatchConfig

batch = BatchConfig(
    backend="slurm",
    slurm_partition="gpu",
    slurm_account="my_project",
    slurm_time="08:00:00",
    slurm_cpus_per_task=16,
)
submit_batch(configs, batch, output_base="scan_results/")
```

## Checkpoint Restart

```python
from helicon.runner.checkpoints import find_latest_checkpoint, get_restart_flag

ckpt = find_latest_checkpoint("results/dfd")
if ckpt:
    print(f"Restart from step {ckpt.step}: {ckpt.path}")
    flag = get_restart_flag("results/dfd")  # "--restart results/dfd/chk01000"
```

## Grid Convergence

```python
from helicon.runner.convergence import run_convergence_study

result = run_convergence_study(
    config,
    resolutions=[(128, 64), (256, 128), (512, 256)],
    output_base="convergence/",
    dry_run=True,
)
print(f"Convergence order: {result.convergence_order:.2f}")
print(f"Extrapolated thrust: {result.extrapolated_thrust_N:.4f} N")
```

## Apple Silicon Metal GPU

When `warpx-metal` is detected, `helicon run` automatically adapts inputs and runs on the
Apple Silicon GPU via AdaptiveCpp → Metal:

```bash
helicon run --preset sunbird
# WarpX Metal:  42%|████▏     | 210/500 [03:22<04:39,  1.04step/s]
# Simulation complete (287.4s)
# Output: results/sunbird
```

The first run is slow (~2–10 s/step) while Metal shaders are JIT-compiled. Subsequent
runs reuse `~/.acpp/apps/global/jit-cache/` and reach ~1 ms/step.

Override the step cap:

```bash
HELICON_METAL_MAX_STEP=2000 helicon run --preset sunbird
```

### Direct Metal API

```python
from helicon.runner.metal_runner import detect_warpx_metal, run_warpx_metal

metal = detect_warpx_metal()
print(metal.summary())

result = run_warpx_metal(
    metal_info=metal,
    output_dir="metal_run/",
    n_cell=128,
    max_step=200,
    progress=True,          # tqdm bar
)
print(result.summary())
for diag in result.diags:
    fields = diag.read_fields()   # {"Ex": np.ndarray(nx, ny), ...}
    print(diag.summary())
```

### Build warpx-metal

```bash
git clone https://github.com/lulzx/warpx-metal ../warpx-metal
cd ../warpx-metal
./scripts/00-install-deps.sh
./scripts/01-build-adaptivecpp.sh
./scripts/05-build-warpx.sh
```

Helicon auto-detects the build at `../warpx-metal` or `~/work/warpx-metal`.
Override with `WARPX_METAL_ROOT`.

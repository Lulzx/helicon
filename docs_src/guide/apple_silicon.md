# Apple Silicon Performance Guide

Helicon runs on Apple Silicon (M-series) in three modes:

| Component | Hardware | Notes |
|-----------|----------|-------|
| WarpX PIC (CPU) | P-cores via OpenMP | No Metal backend in upstream WarpX |
| MLX field solver / postprocessing | Metal GPU | Biot-Savart, thrust reduction, surrogate |
| warpx-metal (PIC) | Metal GPU | Native SYCL/Metal build, single precision |

## Quick start: profile your machine

```bash
helicon perf
```

Output example (M4 Pro, 48 GB):

```
============================================================
Helicon Apple Silicon Performance Profile
============================================================
  Chip:           Apple M4 Pro
  P-cores:        12
  E-cores:        4
  GPU cores:      38
  Unified memory: 48 GB
  Mem bandwidth:  98.3 GB/s  (NumPy stream-copy estimate)
  MLX available:  True  (v0.31.0)
  Python:         3.12.11
  NumPy:          2.4.2
============================================================

WarpX OpenMP Tuning
----------------------------------------
  OMP_NUM_THREADS=12
  OMP_PLACES=cores
  OMP_PROC_BIND=close
  Rationale: Use only P-cores (12) — E-cores (4) are slower and create
  load imbalance in PIC loops. OMP_PLACES=cores pins threads to physical cores.

Shell snippet:
export OMP_NUM_THREADS=12
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

Get a JSON-serialisable report:

```bash
helicon perf --json > perf_report.json
```

---

## WarpX OpenMP tuning

### Use only P-cores

M-series chips have two core types. P-cores (performance) are 3–5× faster
per thread than E-cores (efficiency) for PIC inner loops. Mixing them causes
thread-barrier imbalance and degrades throughput.

```bash
# Recommended: pin WarpX to P-cores only
export OMP_NUM_THREADS=$(helicon perf --json | python3 -c \
  "import sys,json; print(json.load(sys.stdin)['openmp']['omp_num_threads'])")
export OMP_PLACES=cores
export OMP_PROC_BIND=close
helicon run --config my_nozzle.yaml --output results/
```

### Thread binding matters

`OMP_PLACES=cores` + `OMP_PROC_BIND=close` ensures each thread is pinned to a
physical core and neighbouring threads share L2 cache. This matters because
particle-in-cell loops are memory-bandwidth-bound — cache locality is critical.

### Memory bandwidth headroom

WarpX's particle push is bounded by memory bandwidth (~100 GB/s on M4 Pro).
At 1M particles with 7 floats each (28 bytes/particle), a single timestep
reads/writes ~56 MB. You can sustain ~1800 steps/second before memory saturation
on M4 Pro with 12 P-cores.

---

## MLX on Metal GPU

Helicon uses MLX for every non-PIC computation: B-field solving, thrust
reduction, analytical screening, surrogate inference, and Monte Carlo UQ.

### Enable mlx.core.compile

Compiled kernels fuse Metal dispatch calls and eliminate kernel launch overhead.
Helicon enables compilation automatically where it's beneficial. To check:

```python
import mlx.core as mx
from helicon.fields.biot_savart import _compute_mlx

# This call will compile on first run (~0.5 s), then reuse the compiled graph
Br, Bz, r, z = _compute_mlx(coils, grid, n_phi=64)
mx.eval(Br, Bz)
```

### Batch size tuning

The optimal `vmap` batch size for Biot-Savart depends on GPU core count:

| GPU cores | Recommended batch |
|-----------|------------------|
| ≥ 38 (M4 Pro/Max) | 2048 coils |
| ≥ 20 (M3 Pro) | 1024 coils |
| ≥ 10 (M2 Pro) | 512 coils |
| < 10 | 256 coils |

`helicon perf` reports the recommended batch size for your chip.

### Lazy evaluation

MLX uses lazy evaluation — operations are queued and executed only when
`mx.eval()` is called. For large scans:

```python
import mlx.core as mx

results = []
for i, candidate in enumerate(candidates):
    r = analytical_screen(candidate)
    results.append(r)
    if i % 1024 == 0:
        mx.eval(*results)  # flush every 1024 to prevent queue overflow
        results = []
mx.eval(*results)
```

---

## Unified memory management

Apple Silicon has a single unified memory pool shared between CPU (WarpX) and
GPU (MLX). Simultaneous WarpX + MLX pressure can cause memory stalls.

### Capacity estimates

`helicon perf` computes conservative capacity estimates based on unified memory:

- **Max particles**: `0.70 × memory_GB × 1e9 / 28 bytes` (7 floats/particle)
- **Recommended**: 60% of max (leaves headroom for diagnostics + Python heap)
- **Max grid** (nz×nr): estimated from field-array footprint (7 fields × float32)

Example for 48 GB M4 Pro:

| | Value |
|--|-------|
| Max particles | ~1200M |
| Recommended | ~720M |
| Max grid (nz×nr) | ~8192×4096 |

### Reduce diagnostic footprint

Full diagnostic mode (analysis) writes particle dumps every 5000 steps.
For parameter scans use `scan` mode:

```yaml
diagnostics:
  mode: scan           # no particle dumps; field dumps only
  field_dump_interval: 500
```

This cuts storage from ~5 GB to ~50 MB per run and reduces unified memory
pressure during postprocessing.

### Keep checkpoints off

```yaml
keep_checkpoints: false  # default; saves 50–80% disk space
```

Enable only for multi-day runs where restart capability is needed.

---

## AMR (Adaptive Mesh Refinement)

WarpX supports native AMR to resolve the narrow electron demagnetization layer
(detachment front) without paying the cost of uniform fine grid everywhere.

Enable in the YAML config:

```yaml
nozzle:
  resolution:
    nz: 512
    nr: 256
    amr_max_level: 1      # one level of factor-2 refinement
    amr_ref_ratio: 2      # refinement ratio
    amr_regrid_int: 10    # regrid every 10 steps
```

The WarpX input file will include:

```
amr.max_level = 1
amr.ref_ratio = 2
amr.regrid_int = 10
amr.blocking_factor = 8
amr.max_grid_size = 128
warpx.refine_plasma = 1   # refine where plasma density is highest
amr.n_error_buf = 2
```

**Expected benefit:** ~40–50% wall-time reduction vs. a uniformly doubled
grid for the same detachment-front resolution, as the fine region is typically
<20% of the domain area.

**Note:** AMR is not supported by warpx-metal (the SYCL/Metal build uses a
uniform grid). AMR applies only to CPU OpenMP and CUDA backends.

---

## warpx-metal: PIC on Metal GPU

For full PIC on the Metal GPU, build [warpx-metal](https://github.com/lulzx/warpx-metal):

```bash
git clone https://github.com/lulzx/warpx-metal ../warpx-metal
cd ../warpx-metal
./scripts/00-install-deps.sh
./scripts/01-build-adaptivecpp.sh
./scripts/05-build-warpx.sh
```

Helicon auto-detects the build at `../warpx-metal` or `~/work/warpx-metal`.

**First-run JIT:** AdaptiveCpp compiles LLVM IR → Metal shaders on first use
(~2–10 s/step for the first ~200 steps). Subsequent runs reuse the JIT cache
at `~/.acpp/apps/global/jit-cache/` and reach ~1 ms/step.

**Limitations:** single precision (FP32), FDTD only (no PSATD), 2D Cartesian
only (no RZ), periodic boundaries only (PML triggers a Metal JIT bug for
D⁺/e⁻ mass-ratio plasmas).

```bash
# Check Metal detection
helicon doctor

# Run with Metal backend (auto-selected when warpx-metal is found)
helicon run --preset sunbird --output results/
```

Override the step cap:

```bash
HELICON_METAL_MAX_STEP=2000 helicon run --preset sunbird --output results/
```

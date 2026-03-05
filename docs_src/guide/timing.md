# Simulation Timing Reference

Estimated wall times for each Helicon validation case on Apple Silicon chips
and NVIDIA GPUs.  All times assume default grid resolutions and `--dry-run=False`.

!!! note "Estimates"
    These estimates are derived from particle-push bandwidth models
    (28 bytes/particle for 7 float32 fields) and measured memory bandwidth
    on each chip.  Actual times depend on diagnostic I/O, field solver
    complexity, and system load.

---

## Apple Silicon (WarpX CPU + MLX postprocessing)

WarpX runs on P-cores via OpenMP.  MLX handles postprocessing on the Metal GPU.
Use `helicon perf` to get the OMP tuning for your specific chip.

| Validation Case | Grid | Particles | Steps | M2 Pro (12P) | M4 Pro (12P) | M4 Max (14P) |
|----------------|------|-----------|-------|--------------|--------------|--------------|
| Free expansion | 128×64 | 1M | 5k | ~8 min | ~5 min | ~4 min |
| Guiding center | analytic | — | 50k | <1 s | <1 s | <1 s |
| Merino-Ahedo | 512×256 | 10M | 20k | ~4 h | ~2.5 h | ~2 h |
| MN1D comparison (mid-β) | 512×256 | 10M | 20k | ~4 h | ~2.5 h | ~2 h |
| VASIMR plume | 256×128 | 5M | 15k | ~1.5 h | ~55 min | ~45 min |
| Resistive Dimov | 256×128 | 5M | 20k | ~2 h | ~75 min | ~60 min |

**warpx-metal backend** (Apple Silicon GPU via SYCL/AdaptiveCpp): ~3-5× faster
than CPU for the particle push, but first-run JIT compilation adds ~2-10 s/step
for the first ~200 steps.  Subsequent runs use the JIT cache.

---

## NVIDIA GPU (CUDA backend, Linux)

| Validation Case | Grid | Particles | Steps | RTX 3090 | A100 | H100 |
|----------------|------|-----------|-------|-----------|------|------|
| Free expansion | 128×64 | 1M | 5k | ~1 min | <30 s | <20 s |
| Merino-Ahedo | 512×256 | 10M | 20k | ~20 min | ~8 min | ~5 min |
| MN1D comparison | 512×256 | 10M | 20k | ~20 min | ~8 min | ~5 min |
| VASIMR plume | 256×128 | 5M | 15k | ~10 min | ~4 min | ~2.5 min |
| Resistive Dimov | 256×128 | 5M | 20k | ~12 min | ~5 min | ~3 min |

---

## Wall-time scaling rules

For quick estimation on your hardware:

```
T_wall ≈ (N_particles × N_steps × bytes_per_particle) / (η × bandwidth)
```

where:
- `bytes_per_particle` = 28 B (7 float32 fields: x, y, z, px, py, pz, w)
- `η` ≈ 0.40–0.65 (memory bandwidth utilization — PIC has ~50% efficiency due to indirect addressing)
- `bandwidth` = system memory bandwidth (GB/s; use `helicon perf` to measure)

**Example (M4 Pro, 10M particles, 20k steps):**

```
T = (10e6 × 20000 × 28) / (0.50 × 98.3e9)
  = 5.6e12 / 4.9e10
  ≈ 114 s per thread × 1 thread / 12 threads ≈ ~9.5 min (ideal)
```

The 9.5 min ideal scales to ~2.5 h when accounting for field solve (~10×
particle push cost for 2D-RZ at 512×256), diagnostic I/O, and Python overhead.

---

## Checking performance on your machine

```bash
# Profile hardware
helicon perf

# Time a scan-mode (no field dumps) validation case
time helicon run --preset free_expansion --output /tmp/test_run
```

Add `HELICON_METAL_MAX_STEP=200` to limit warpx-metal runs to JIT warm-up only.

---

## Unified memory considerations

On Apple Silicon, particle data and field arrays share unified memory with
the OS and all other processes.  At 10M particles:

- **Particle data**: 10M × 28 B = 280 MB
- **Field arrays** (512×256×7 float32): ~4 MB each diagnostic
- **WarpX working set**: ~2–4× particle size = ~0.6–1.2 GB
- **MLX postprocessing**: additional ~0.5–1 GB during computation

Total WarpX + MLX peak ≈ 1.5–2 GB for the standard 10M-particle case.
Safe on any M-series with ≥ 16 GB unified memory.

At 100M particles, peak approaches 15–20 GB — use a 36+ GB config.

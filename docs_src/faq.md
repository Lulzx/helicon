# Frequently Asked Questions

## General

**Q: Does Helicon include a PIC solver?**

No. Helicon wraps WarpX — the DOE Exascale Computing Project's particle-in-cell code.
You must install WarpX separately. Helicon handles configuration, field pre-computation,
parameter scanning, optimization, and post-processing.

**Q: Can I use Helicon without WarpX?**

Yes — for everything except running PIC simulations:
- Biot-Savart field computation works standalone
- Analytical pre-screening (`helicon.optimize.analytical`) requires no WarpX
- Post-processing can re-analyze existing WarpX output
- Use `dry_run=True` to generate input files without launching WarpX

**Q: Why does the default configuration use full physical mass ratio?**

Reduced mass ratio (`mass_ratio=100` instead of 3672 for deuterium) runs ~6x faster
but gives qualitatively wrong electron dynamics. All preset validation cases use the
physical ratio. Reduced ratio is only for qualitative parameter scans — always flagged
in output metadata as `mass_ratio_reduced=true`.

## Apple Silicon / MLX

**Q: Does WarpX run on the Apple Silicon GPU?**

No — WarpX uses CPU/OpenMP on macOS (no Metal backend). The Apple Silicon GPU is used
by Helicon's Python layer: Biot-Savart computation, optimization, and post-processing
via MLX.

**Q: How fast is Helicon on Apple Silicon vs NVIDIA?**

WarpX itself runs ~2-4x slower on Apple Silicon CPU vs an NVIDIA A100. For production
parameter scans (100+ runs), use a Linux/NVIDIA cluster. Apple Silicon is excellent
for development, quick validation, and optimization (which uses MLX GPU).

## Physics

**Q: Which detachment efficiency definition should I use?**

Report all three (`momentum`, `particle`, `energy`) when comparing with literature.
Specify which one you're comparing — different papers use different definitions and
they can differ by 10-20% for the same simulation.

**Q: What does the Hall parameter Ω_e τ_e = 1 threshold mean physically?**

When Ω_e τ_e ~ 1, electrons complete approximately one gyration per collision.
Below this (Ω_e τ_e < 1), electrons are effectively demagnetized — resistive detachment
can occur. The standard simulation target is Ω_e τ_e >> 1 for magnetized plasma.

**Q: What grid resolution do I need?**

- Quick tests: 128×64 (minutes on CPU)
- Standard runs: 512×256 (hours on CPU, ~1h on A100)
- Production: 1024×512 (10+ hours on A100)

Use `run_convergence_study()` to verify your results are grid-converged before
publishing.

## Troubleshooting

**Q: `ImportError: No module named 'mlx'`**

Install MLX: `pip install mlx` (macOS with Apple Silicon only).
On other platforms, Helicon automatically falls back to NumPy.

**Q: WarpX input is generated but simulation doesn't start**

Install WarpX Python bindings: see [WarpX installation guide](https://warpx.readthedocs.io/en/latest/install/).
Use `dry_run=True` to test without WarpX.

**Q: How do I reproduce a result from a previous run?**

Every run saves `run_metadata.json` containing the `config_hash`, `helicon_git_sha`,
`warpx_version`, `random_seed`, and all dependency versions. Replicate the environment
using the provided `Dockerfile` or `environment.yml`.

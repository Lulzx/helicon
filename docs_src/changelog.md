# Changelog

## v2.7.0 (2026-03)

### New Features

**Apple Silicon Metal GPU backend (warpx-metal)**
- `runner/metal_runner.py`: Native Metal GPU runner integrating [warpx-metal](https://github.com/lulzx/warpx-metal) — discovers the build, generates/adapts WarpX inputs, launches the SYCL/Metal 2D executable, and parses AMReX plotfile diagnostics
- `WarpXMetalInfo`, `WarpXMetalDiag`, `MetalRunResult` dataclasses with `.summary()` methods
- `find_diag_dirs()`: discovers and sorts all AMReX `diag*/` snapshots under an output directory
- `WarpXMetalDiag.read_fields()`: assembles multi-box AMReX FAB tiles into full-domain NumPy arrays (handles 512×256 domains split into 8 tiles via `Cell_H` + `Cell_D_00000`)
- `generate_metal_inputs()`: generates validated 2D pair-plasma input files for standalone Metal runs
- `run_warpx_metal()`: launches the `warpx.2d.NOMPI.SYCL.SP.PSP.EB` executable with live tqdm progress bar (streams stdout line-by-line, parses `STEP N ends`)

**`_adapt_inputs_for_metal()`** in `runner/launch.py` — transparent RZ→2D Cartesian adapter:
- `geometry.dims = RZ` → `geometry.dims = 2`, `coord_sys = 1` → `0`
- Removes openPMD format specifiers (Metal build has no openPMD)
- Maps RZ field names (`Br/Er/jr/Bt/Et/jt`) → Cartesian (`Bx/Ex/jx/By/Ey/jy`)
- Forces periodic boundaries (PML kernels trigger AdaptiveCpp Metal JIT `t_badref_` bug)
- Reduces `num_particles_per_cell` to 2×2; caps `max_step` at 500 (override: `HELICON_METAL_MAX_STEP`)
- Domain symmetrisation: `(z, r≥0)` → `(z, x∈[-r_max, r_max])`
- Injects required fields missing from pywarpx-style inputs (`algo.particle_shape`, etc.)

**Hardware & CLI improvements**
- `hardware_config.py`: `HardwareInfo` now reports `has_warpx_metal`, `warpx_metal_exe_2d`, `recommended_backend = "metal"` on Apple Silicon with build detected
- `doctor.py`: `DoctorReport` gains `warpx_metal_found` / `warpx_metal_path`; `helicon doctor` prints a Metal section
- `run_simulation()` auto-selects Metal backend before pywarpx; shows live tqdm bar
- CLI `helicon run` failure now prints backend, returncode, error string, and log path

### Bug Fixes
- `WarpXMetalDiag.species`: now scans particle subdirectories (not WarpXHeader text)
- `WarpXMetalDiag.n_cells`: AMReX box formula corrected (`hi − lo + 1` per dimension)
- `WarpXMetalDiag.read_fields()`: multi-box FAB assembly via `Cell_H` byte offsets — fixes `ValueError: buffer size must be a multiple of element size`
- `run_warpx_metal()`: resolves `output_dir` to absolute path (was passing relative path → SIGABRT)
- `perf/profiler.py`: replaced `psutil` with `/proc/meminfo` → `os.sysconf` fallback (Linux CI)

### CI / tooling
- Added `ruff format --check` to CI; formatted all source files

---


## v0.4.0 (2026-03)

### New Features

**Optimization & Constraints (v0.3 completion)**
- `optimize/constraints.py`: Engineering constraint evaluation for coil optimization — mass budget, resistive power, peak B at conductor, MLX-differentiable penalty wrapper
- `optimize/pareto.py`: Multi-objective Pareto front with hypervolume indicator
- `optimize/analytical.py`: Fast Tier-1 analytical pre-screening (Breizman-Arefiev paraxial model, Little-Choueiri thrust coefficient)
- `runner/convergence.py`: Grid convergence study automation with Richardson extrapolation
- `runner/batch.py`: Batch simulation submission (local multiprocessing, SLURM, PBS)
- `runner/checkpoints.py`: WarpX checkpoint management and restart utilities
- `fields/cache.py`: HDF5 field map cache (avoids recomputing Biot-Savart)
- `fields/frc_topology.py`: FRC separatrix identification and field line topology classification

**Plasma Models**
- `PlasmaSourceConfig.electron_model`: "kinetic" (default) or "fluid" hybrid electron model
- `PlasmaSourceConfig.mass_ratio`: Reduced mass ratio support with citation-guard warning
- WarpX generator: proper electron mass override, fluid-hybrid parameters

**Validation & Output**
- Validation case: VASIMR VX-200 plume (Olsen et al. 2015)
- Validation case: Resistive detachment threshold (Moses 1991 / Dimov 2005)
- `validate/report.py`: Comparison plots and HTML validation report
- `postprocess/propbench.py`: PropBench-compatible JSON output schema
- `postprocess/report.py`: `save_scan_csv()` for parameter scan export

**Community & Infrastructure**
- `CITATION.cff`: Machine-readable citation metadata
- `CONTRIBUTING.md`: Contributor guide
- `Dockerfile`: Reproducible container build
- `environment.yml`: Conda lock environment
- `mkdocs.yml`: Documentation site configuration

### Version bump
- Version: 0.2.0 → 0.4.0

---

## v0.3.0 (2026-02)

### New Features
- `optimize/scan.py`: Grid and Latin Hypercube parameter scans
- `optimize/surrogate.py`: GP surrogate with Bayesian optimizer (Expected Improvement)
- `optimize/sensitivity.py`: Sobol sensitivity analysis (Saltelli 2010)
- `optimize/objectives.py`: MLX-differentiable coil optimization objectives
- `fields/import_external.py`: FEMM `.ans` and CSV B-field import
- CLI: `helicon scan` command

---

## v0.2.0 (2026-01)

### New Features
- `postprocess/detachment.py`: Three-definition detachment efficiency
- `postprocess/plume.py`: Plume divergence angle and beam efficiency
- `postprocess/pulsed.py`: Pulsed-mode impulse bit diagnostics
- `postprocess/fieldline_classify.py`: Open/closed field line topology
- Validation cases: Merino-Ahedo (2016), MN1D comparison
- Preset configs: `sunbird.yaml`, `dfd.yaml`, `ppr.yaml`

---

## v0.1.0 (2025-12)

### Initial Release
- `fields/biot_savart.py`: Biot-Savart solver (MLX + NumPy backends, HDF5 save/load)
- `config/parser.py`: Pydantic YAML configuration schema
- `config/warpx_generator.py`: WarpX input file generation
- `runner/launch.py`: WarpX simulation execution
- `postprocess/thrust.py`: Momentum flux thrust computation
- Validation cases: free expansion, guiding center orbit
- CLI: `helicon run`, `helicon postprocess`, `helicon validate`

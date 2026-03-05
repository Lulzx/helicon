# Changelog

## v2.9.0 (2026-03)

### New Features

**COMSOL field map import** (spec §3.2 Module 2)
- `fields/import_external.py`: `load_comsol_bfield()` — parse COMSOL Multiphysics export
  files (%-comment header, space/tab-delimited data) with flexible column selection by
  index or partial name, and `length_scale` for unit conversion (mm→m etc.)
- Exported from `helicon.fields` public API

**ScanResult CSV / JSON export** (spec §6.3)
- `optimize/scan.py`: `ScanResult.to_csv(path)` — write tabulated parameter scan results
  to CSV (one row per scan point; parameter columns + metric columns)
- `optimize/scan.py`: `ScanResult.to_json_summary(path)` — write structured JSON summary
  with params + metrics + screened_out flag per point
- `helicon scan` CLI now automatically writes `scan_results.csv` and `scan_summary.json`
  to the output directory

**Documentation improvements**
- Added validation case docs for `guiding_center` and `mn1d_comparison` (previously
  missing from `docs_src/validation/`; all 6 spec-required cases now documented)
- Added API reference pages for `helicon.surrogate`, `helicon.detach`, `helicon.perf`
- Added Apple Silicon guide and Timing Reference to User Guide nav
- Added simulation wall-time estimates for all 6 validation cases on M2 Pro, M4 Pro,
  M4 Max, RTX 3090, A100, H100

### Bug Fixes
- `cli.py` `mf promote`: fixed `wall_time_s` → `wall_time_seconds` attribute name
  (`RunResult.wall_time_seconds` is the correct field name)

### Infrastructure
- Removed Lambda Labs and AWS cloud backend stubs; only `LocalBackend` remains
- Added `mf_output/`, `scan_results/`, `cloud_scan_results/` to `.gitignore`
- Updated `CITATION.cff` to v2.9.0
- Expanded CLI test coverage to all commands: `throttle-map`, `mission`, `regression`,
  `validate`, `run-3d`, `postprocess`, `optimize`, `convergence`, `surrogate-train`,
  `export-cad` — total CLI commands with test coverage: 22/22
- Added `tests/unit/test_public_api.py`: 14 tests for spec §6.2 top-level API
  (`helicon.Config`, `helicon.fields.compute()`, `helicon.run()`, `helicon.scan()`,
  `helicon.Metrics`, `helicon.DetachmentMetrics`)
- Added `tests/unit/test_mission_cli.py`: 14 tests for throttle-map + mission + regression CLI
- Added `tests/unit/test_diagnostics.py`: 9 tests for `DiagnosticSchedule` / `resolve_schedule`
- Total: 1258 tests (was 1198)

---

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

## v2.8.0 (2026-03)

### New Features

**`helicon mf` CLI command group**
- `helicon mf report PATH` — reads all `tier3_meta.json` files from a multi-fidelity scan output directory, ranks candidates by tier2 score, and prints a formatted table; `--json` for machine-readable output
- `helicon mf run` — CLI surface for `MultiFidelityPipeline`: `--vary`, `--n-tier1`, `--tier2-threshold`, `--tier3-threshold`, `--top-k`, `--objective`, `--dry-run` flags; reports tier promotion counts and best candidate metrics
- 9 tests added; total 1174

---

## v2.6.0 (2026-03)

### New Features

**Detachment CLI group**
- `helicon detach` command group with five subcommands: `assess`, `calibrate`, `invert`, `simulate`, `report`
- `helicon detach assess` — full multi-criterion detachment state evaluation with optional `--control` and `--json` flags
- `helicon detach invert` — invert thrust measurement to plasma state estimate
- `helicon detach calibrate` — calibrate detachment score weights from synthetic data
- `helicon detach simulate` — simulate Lyapunov-stable feedback controller over N steps
- `helicon detach report` — full multi-criterion report (MHD, kinetic, sheath, Lyapunov)

---

## v2.5.0 (2026-03)

### New Features

**Novel detachment physics contributions**
- Five novel detachment analysis models beyond the base MHD framework
- Kinetic FLR corrections (Northrop 2nd-order, kinetic Alfvén magnetisation)
- Sheath coupling correction: Debye length, sheath potential, electric-to-mirror ratio
- Lyapunov stability certificate: V, dV/dt, convergence time per controller step
- Detachment score BCE calibration from synthetic PIC scan data
- Thrust inversion: recover n₀, Tᵢ, η_d from measured force and mass flow

---

## v2.4.0 (2026-03)

### New Features

**Real-time detachment control model (`helicon/detach/`)**
- `detach/mhd.py`: MHD detachment model — mirror ratio, beta, field-line divergence criterion
- `detach/kinetic.py`: Kinetic FLR corrections to MHD detachment scores
- `detach/sheath.py`: Sheath coupling model with Debye-length correction to η_d
- `detach/controller.py`: Lyapunov-stable feedback controller — computes control action to drive plasma toward target detachment state; proves stability via V̇ < 0 certificate
- `detach/calibrate.py`: BCE weight calibration on synthetic scan database
- `detach/invert.py`: Thrust-measurement inversion to plasma state

---

## v2.3.0 (2026-03)

### New Features

**Sensitivity & Provenance CLI**
- `helicon sensitivity` — Sobol sensitivity analysis with first-order and total-effect indices; `--json` output
- `helicon provenance list/show/lineage` — browse the JSON-lines audit trail, inspect individual records, trace design lineage graphs
- `helicon perf` — Apple Silicon hardware profile with WarpX OpenMP and MLX tuning recommendations

---

## v2.2.0 (2026-03)

### Improvements

**Hardening & tooling**
- Pre-commit hook (`.git/hooks/pre-commit`): runs `ruff check` then `pytest` before every commit; blocks on failure
- CI lint + test failures resolved across all modules
- `ruff` format applied to entire codebase; stale `noqa` directives removed
- `helicon doctor --json` added; structured environment report for CI integration

---

## v2.1.0 (2026-03)

### New Features

**Ecosystem & Community modules**
- `helicon/plugins/`: `PluginRegistry` — register, get, call, and list plugins by namespace; entry-point auto-discovery via `importlib.metadata`
- `helicon/multithruster/`: `ThrusterArray`, `ArrayConfig` — model 2–4 nozzle arrays with plume-plume interaction and combined thrust/Isp
- `helicon/valdb/`: `ValidationDatabase` (JSON-lines) — add, query, delete, export `ValidationRecord` entries; `helicon valdb` CLI group
- `helicon/widgets/`: `FieldTopologyWidget`, `CoilEditorWidget` — ipywidgets UI with matplotlib rendering for Jupyter
- `helicon/validate/`: `RegressionSuite`, `save_baseline`, `compare_to_baseline` — automated annual regression against stored baselines; `helicon regression` CLI group
- `helicon array` CLI command — combined multi-thruster array performance
- `helicon plugins` CLI command — list registered plugins

---

## v2.0.0 (2026-03)

### New Features

**Engineering design tool**
- `helicon/surrogate/`: `NozzleSurrogate` MLX MLP — train on PIC scan database via `mlx.nn`, predict thrust/η_d/plume angle in microseconds on Metal GPU; Monte Carlo UQ via `surrogate.uq`
- `helicon/app/`: Streamlit interactive nozzle design explorer — real-time surrogate inference on Metal GPU, one-click dispatch to full WarpX PIC; `helicon app` launcher
- `helicon/provenance.py`: JSON-lines audit trail — every design decision logged with inputs, outputs, and git SHA; lineage graph reconstruction
- `helicon/export/`: STEP/IGES CAD export for optimised coil geometry via `cadquery`; `helicon export-cad` CLI command
- `optimize/manufacturability.py`: REBCO coil buildability constraints — winding pack geometry, minimum bend radius, tape width discretisation
- `optimize/multifidelity.py`: Three-tier multi-fidelity pipeline — Tier 1 analytical (seconds) → Tier 2 surrogate (µs) → Tier 3 WarpX PIC (hours); automatic candidate promotion
- `helicon surrogate-train` CLI command

---

## v1.3.0 (2026-03)

### New Features

**Mission integration**
- `helicon/mission/throttle.py`: Throttle curve generation — thrust and Isp as functions of power and mass flow rate; interpolation tables for mission planners; `helicon throttle-map` CLI command
- `helicon/mission/trajectory.py`: Propellant budget and burn-time estimation for ΔV targets; `helicon mission` CLI command
- `helicon/mission/spacecraft.py`: Spacecraft interaction model — backflow fraction, plume electron charging, magnetic torque from nozzle field on bus
- `helicon/mission/pulsed.py`: Pulsed mission profiles for PPR-class engines — impulse-averaged thrust/Isp/η_d over burst cycles
- `helicon/cloud/`: Cloud HPC backends — local, Lambda Labs, AWS p4d; `helicon scan --cloud` for remote WarpX submission

---

## v1.2.0 (2026-03)

### New Features

**Extended physics**
- `helicon/neutrals/`: Monte Carlo neutral dynamics — `cross_sections.py` (ionisation, charge exchange, recombination), `monte_carlo.py` (neutral particle push on Metal GPU via MLX); replaces static background neutral model
- `helicon/hybrid/`: CGL double-adiabatic electron fluid — `cgl_fluid.py` (Chew-Goldberger-Low electron pressure tensor), `lhdi.py` (lower-hybrid drift instability anomalous transport sub-grid model), `coupler.py` (WarpX ion PIC + MLX fluid electron coupling)
- `postprocess/species.py`: Multi-ion species detachment tracking — separate η_d per species (D⁺, He²⁺, H⁺, α); MLX-accelerated species-resolved moment computation

---

## v1.1.0 (2026-03)

### New Features

**MLX-native acceleration**
- `postprocess/thrust.py`: `_thrust_reduce_mlx` — thrust and mass-flow reduction on Metal GPU via `mlx.core`; auto-selected when MLX available
- `postprocess/plume.py`: `compute_electron_magnetization(..., backend="mlx")` — electron magnetisation grid computed on Metal GPU
- `fields/biot_savart.py`: `_compute_mlx` with `mlx.core.compile` for fused kernels; batch-evaluate B(r,z) on AMR-resolution grids via `vmap`
- `optimize/analytical.py`: Fully differentiable Breizman-Arefiev paraxial model and Little-Choueiri C_T in MLX — `mlx.core.grad` through entire Tier-1 analytical pipeline
- `benchmark.py`: `BenchmarkSuite` — NumPy vs MLX timing comparison for Biot-Savart, thrust reduction, electron magnetisation, analytical screening, and differentiable gradient; `helicon benchmark` CLI command
- `perf/profiler.py`: `AppleSiliconProfiler` — chip detection, P/E core count, GPU cores, memory bandwidth probe, WarpX OpenMP and MLX tuning recommendations, unified memory capacity estimates

---

## v1.0.0 (2026-03)

### Milestone: Validated Research Tool

- All six validation cases passing with documented error bounds
- Stable public API under semantic versioning: `helicon.Config`, `helicon.fields.compute()`, `helicon.run()`, `helicon.postprocess()`, `helicon.scan()`
- `helicon/_reproducibility.py`: full run metadata per spec §14 — config hash, git SHA, WarpX version, random seeds, environment snapshot
- `helicon doctor` reports backend availability across Metal, CUDA, OMP, and MLX

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

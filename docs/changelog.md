# Changelog

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

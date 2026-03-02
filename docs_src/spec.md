# Helicon — Technical Specification

**GPU-Accelerated Magnetic Nozzle Simulation & Detachment Analysis Toolkit for Fusion Propulsion**

Version: 0.1.0-draft
Author: lulzx
License: MIT
Date: March 2026

---

## 1. Problem Statement

Every direct-fusion-drive concept — Pulsar Sunbird (DDFD), Princeton DFD, Howe PPR, and others — terminates in a magnetic nozzle: a diverging magnetic field that converts thermal plasma energy into directed exhaust momentum. The nozzle is where fusion energy becomes thrust.

The central unsolved problem is **plasma detachment**. The magnetic field that confines and accelerates the plasma must eventually release it. If the plasma remains tied to field lines, the exhaust curves back toward the spacecraft and net thrust drops to zero. Current estimates of detachment efficiency in the literature range from 50% to 95% — a factor-of-two uncertainty that makes or breaks every mission architecture built on these engines.

**Why this uncertainty exists:** Ion detachment is straightforward (large Larmor radius → ions decouple from weakening field). Electron detachment is the hard problem. Electrons are light, magnetized, and drag ions back via ambipolar electric fields. The physics governing electron demagnetization involves:

- Electron inertia (finite Larmor radius effects at the detachment front)
- Pressure anisotropy (parallel vs. perpendicular temperature diverge downstream)
- Resistive and collisional effects (anomalous transport, wave-particle interactions)
- Cross-field transport (instabilities that break frozen-in flux)

No existing open-source tool provides a validated, GPU-accelerated workflow for studying these effects in propulsion-relevant geometries. Helicon fills this gap.

### 1.1 What Helicon Is

A simulation **toolkit** — not a solver from scratch. Helicon wraps and extends WarpX (the DOE Exascale Computing Project's high-performance, open-source particle-in-cell code) with:

- Curated input configurations for fusion-propulsion magnetic nozzle geometries
- Post-processing pipelines that extract propulsion-relevant quantities (thrust, Isp, detachment efficiency, plume divergence) from raw PIC output
- Parameter-scan and optimization infrastructure for nozzle design
- Validation cases against published analytical solutions and experimental data

### 1.2 What Helicon Is Not

- Not a new PIC/MHD solver (WarpX handles numerics; we don't rewrite them)
- Not a full engine simulator (upstream fusion plasma source is a boundary condition, not modeled)
- Not a mission planner (outputs feed into trajectory tools like poliastro; we don't couple them internally)
- Not ready for engineering design on day one (v0.x is a research tool; validated design capability is a v1.0+ goal)

---

## 2. Physics Scope

### 2.1 Core Physical Model

Helicon operates in **2D axisymmetric cylindrical coordinates (r, z)** using WarpX's existing RZ geometry support. Full 3D is available for asymmetric studies but is not the default due to computational cost.

**Particle species modeled:**

| Species | Treatment | Justification |
|---|---|---|
| Ions (D⁺, He²⁺, H⁺, α) | Fully kinetic PIC | Large Larmor radius → kinetic effects dominate detachment |
| Electrons | Kinetic PIC or fluid hybrid | Electron demagnetization is the key physics; kinetic treatment is default. Fluid option available for parameter scans where full kinetic cost is prohibitive |
| Neutrals | Monte Carlo background (optional) | Charge exchange in low-ionization edge regions |

**Fields:**

- Electromagnetic fields solved via Maxwell's equations (FDTD in WarpX)
- External applied magnetic field from coil geometry (pre-computed via Biot-Savart or imported from FEMM/Comsol)
- Self-consistent plasma-generated fields included

**Key physics captured:**

- Ion ballistic detachment (Larmor radius >> local B scale length)
- Electron inertia effects (retained in kinetic treatment; optional inertia correction in fluid hybrid)
- Ambipolar electric field formation and its role in coupling ion/electron detachment
- Pressure anisotropy development downstream of throat
- Resistive detachment via anomalous transport (parametrized collision operator, benchmarkable against theory)
- Current closure topology (how return currents flow and where they close)

**Physics explicitly excluded (out of scope for v0.x):**

- Fusion reactions in the nozzle region (energy source is upstream; plasma enters nozzle as a specified boundary condition)
- Radiative losses (relevant for very high density; negligible for most propulsion nozzle regimes)
- Plasma-wall interaction (nozzle is magnetic, no material walls in exhaust path)
- Relativistic effects (exhaust velocities ~100-500 km/s, well below relativistic regime)

### 2.2 Geometry

The nozzle is defined by its **applied magnetic field topology**, which is set by coil positions and currents. Helicon provides parameterized geometry generators for the three canonical propulsion nozzle types:

1. **Simple solenoid nozzle** — Single coil or coil pair producing a diverging field downstream. Baseline case; analytically tractable for validation.

2. **Converging-diverging (de Laval analog)** — Magnetic mirror geometry with a defined throat (B_max) and controlled divergence angle. Most propulsion concepts use this.

3. **Field-Reversed Configuration (FRC) exhaust** — Plasmoid ejection through an open field region. Relevant to DFD/Sunbird-class engines. Requires handling of closed + open field line topology.

Geometry is specified via a YAML configuration file:

```yaml
nozzle:
  type: converging_diverging  # or solenoid, frc_exhaust
  coils:
    - z: 0.0       # axial position [m]
      r: 0.15      # coil radius [m]
      I: 50000     # current [A-turns]
    - z: 0.3
      r: 0.25
      I: 30000
  domain:
    z_min: -0.5    # [m] upstream of throat
    z_max: 2.0     # [m] downstream
    r_max: 0.8     # [m]
  resolution:
    nz: 512
    nr: 256
```

The applied B-field is pre-computed on the grid from coil definitions using a Biot-Savart integrator (included) or imported from external field maps (FEMM export, CSV).

### 2.3 Boundary Conditions

**Upstream (plasma inlet):**
The fusion plasma source is represented as a particle injection boundary. Users specify:

- Ion/electron density and temperature (or distribution function)
- Bulk flow velocity (sonic or super-sonic at throat)
- Species mix (e.g., 50/50 D/He3 ash, or pure proton beam for p-B11 alphas)
- Temporal profile (steady-state or pulsed injection for PPR-type engines)

Defaults are provided for each engine class (Sunbird, DFD, PPR) based on published parameters.

**Downstream (exhaust exit):**
Open/absorbing boundary. Particles leaving the domain are counted for thrust calculation and removed. Fields use perfectly matched layer (PML) or silver-müller conditions (WarpX native).

**Radial (outer wall):**
Absorbing or reflecting, depending on whether spacecraft structure is modeled. Default is absorbing (particles lost radially are a loss mechanism).

**Axis (r = 0):**
Axisymmetric boundary condition (WarpX RZ native handling).

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────┐
│                   Helicon                     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Config   │  │  B-field  │  │   WarpX       │  │
│  │  Parser   ├──► Generator ├──►  Runner       │  │
│  │  (YAML)   │  │ (Biot-   │  │  (PIC/hybrid) │  │
│  └──────────┘  │  Savart)  │  └──────┬────────┘  │
│                └──────────┘         │            │
│                                     ▼            │
│  ┌──────────────────────────────────────────┐    │
│  │         Post-Processing Pipeline         │    │
│  │                                          │    │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │ Thrust  │ │Detachment│ │  Plume   │  │    │
│  │  │ Integr. │ │ Metrics  │ │ Diverge. │  │    │
│  │  └─────────┘ └──────────┘ └──────────┘  │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│  ┌──────────────────────────────────────────┐    │
│  │       Parameter Scan / Optimizer          │    │
│  │    (coil geometry, B strength, beta)      │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### 3.2 Module Breakdown

#### Module 1: `helicon.config`
**Purpose:** Parse and validate YAML input files. Generate WarpX input scripts.

- Reads nozzle geometry, plasma source, simulation parameters
- Validates physical consistency (e.g., plasma beta < 1 at throat for confinement, domain large enough for detachment)
- Emits WarpX-compatible input file (`warpx_input`) with all physics settings pre-configured
- Provides preset configurations for known engine types

**Key files:**
```
helicon/config/
├── parser.py          # YAML → internal config dataclass
├── presets/
│   ├── sunbird.yaml   # Pulsar Sunbird reference case
│   ├── dfd.yaml       # Princeton DFD reference case
│   ├── ppr.yaml       # Howe PPR reference case
│   └── validation/
│       ├── analytic_expansion.yaml   # free expansion (known solution)
│       └── littlejohn.yaml           # guiding-center benchmark
├── warpx_generator.py # config → WarpX input file
└── validators.py      # physics sanity checks
```

#### Module 2: `helicon.fields`
**Purpose:** Compute and manage the applied magnetic field.

- Biot-Savart integrator for arbitrary coil sets (vectorized NumPy/MLX)
- Field line tracer (RK4/RK45) for visualization and topology analysis
- Import from external tools (FEMM `.ans` files, CSV field maps, COMSOL export)
- Caching: computed B-fields are stored as HDF5 to avoid recomputation
- Critical invariant: separatrix identification for FRC geometries

**Key files:**
```
helicon/fields/
├── biot_savart.py     # coil → B(r,z) on grid (MLX primary, NumPy fallback)
├── field_lines.py     # tracing and topology classification
├── import_external.py # FEMM, COMSOL, CSV loaders
└── cache.py           # HDF5 field map storage
```

**MLX/Apple Silicon acceleration note:** The Biot-Savart module detects MLX availability at import and falls back to NumPy transparently. When MLX is available, computation runs on the Apple Silicon GPU via Metal. `mlx.core.grad` through the field computation enables gradient-based coil optimization in Module 5; `mlx.core.vmap` vectorizes the Biot-Savart integration over coil segments and grid points. Without MLX, the optimizer is limited to Bayesian/gradient-free methods. On Linux/HPC systems with NVIDIA GPUs, an optional JAX backend is available for CUDA acceleration.

#### Module 3: `helicon.runner`
**Purpose:** Configure, launch, and monitor WarpX simulations.

- Programmatic WarpX invocation via `pywarpx` Python bindings
- Hardware detection and configuration (Apple Silicon / CPU with MLX for Python-layer; CUDA/ROCm on Linux/HPC)
- Checkpoint management (restart from intermediate state)
- Diagnostic scheduling (particle snapshots, field dumps, reduced diagnostics at configurable intervals)
- Batch mode for parameter scans (submit multiple runs to SLURM/PBS or local multiprocessing)

This module does **not** contain any physics — it is pure orchestration. All physics lives in WarpX.

**Apple Silicon note:** On macOS with Apple Silicon, WarpX runs on CPU via OpenMP (no Metal backend exists for WarpX). MLX handles Python-layer GPU acceleration (Biot-Savart, postprocessing, optimization). On Linux with NVIDIA GPUs, WarpX uses CUDA natively.

**Key files:**
```
helicon/runner/
├── launch.py          # single run execution
├── hardware_config.py # hardware detection and WarpX backend selection
├── batch.py           # parameter scan submission (local / SLURM / PBS)
├── checkpoints.py     # restart management
└── diagnostics.py     # configure WarpX in-situ diagnostics
```

#### Module 4: `helicon.postprocess`
**Purpose:** Extract propulsion metrics from WarpX output. This is the core value-add of Helicon.

**Computed quantities:**

| Quantity | Definition | Method |
|---|---|---|
| **Thrust (F)** | Net axial momentum flux through downstream boundary | Integrate `Σ_s (n_s m_s v_z v_z + P_zz) · dA` over exit plane for all species s |
| **Specific impulse (Isp)** | F / (ṁ · g₀) | Thrust divided by mass flow rate at exit |
| **Exhaust velocity (v_ex)** | Effective directed velocity | F / ṁ |
| **Detachment efficiency (η_d)** | Fraction of plasma momentum not recaptured by field | 1 - (momentum on returning field lines / total injected momentum). Requires field-line classification of each particle at exit |
| **Plume divergence half-angle (θ)** | Angular spread of exhaust | cos⁻¹(F / ∫ n m v² dA) — momentum-weighted divergence |
| **Beam efficiency (η_b)** | Fraction of kinetic energy in axial direction | (½ṁv_z²) / (½ṁ|v|²) at exit plane |
| **Radial loss fraction** | Particles absorbed at r_max boundary | Count / total injected |
| **Electron magnetization parameter (Ω_e)** | Local electron gyrofrequency / collision frequency | Computed on grid; detachment occurs where Ω_e ~ 1 |
| **Thrust coefficient (C_T)** | F / (ṁ · c_s), where c_s is ion sound speed at throat | Standard nozzle performance metric in Ahedo/Merino literature; enables direct comparison to published results |
| **Pressure anisotropy (A)** | P_perp / P_parallel - 1 | From velocity-space moments of particle data |

**Key insight on detachment efficiency:** The standard metric in the literature (η_d) is ambiguous because different papers define it differently. Helicon computes three definitions and reports all:

1. **Momentum-based:** Net axial momentum at exit / injected axial momentum
2. **Particle-based:** Fraction of injected particles that exit through the downstream boundary (vs. radial loss or reflection)
3. **Energy-based:** Directed kinetic energy at exit / total injected energy

Users must specify which they're comparing against when citing literature values.

**Key files:**
```
helicon/postprocess/
├── thrust.py          # momentum flux integration
├── detachment.py      # three definitions of η_d
├── plume.py           # divergence angle and beam efficiency
├── moments.py         # density, velocity, pressure tensor from particles
├── fieldline_classify.py  # tag particles by field line topology
└── report.py          # summary JSON/CSV + auto-generated plots
```

#### Module 5: `helicon.optimize`
**Purpose:** Nozzle design optimization via parameter sweeps and gradient-free optimization.

- **Parameter scan:** Grid or Latin hypercube sampling over coil positions, currents, plasma beta, injection velocity. Each point is a WarpX run.
- **Bayesian optimization:** Gaussian process surrogate of η_d(coil_params) → suggest next evaluation point. Uses `botorch` or `scikit-optimize`.
- **Gradient-based optimization (MLX path):** When MLX is available, the Biot-Savart field computation is differentiable. For objectives that depend only on field geometry (e.g., throat ratio, divergence angle, field-line topology), gradient-based optimization via `mlx.core.grad` and `mlx.optimizers` is fast (seconds on Apple Silicon GPU). For objectives requiring WarpX evaluation (e.g., detachment efficiency), gradients are not available through the PIC solver — use Bayesian optimization or GP surrogate. botorch runs on PyTorch with MPS backend on Apple Silicon.
- **Fast pre-screening (v0.3+):** Two-tier optimization approach. Tier 1: analytical/semi-analytical model (paraxial approximation per Breizman & Arefiev 2008, thrust coefficient per Little & Choueiri 2013) evaluates thousands of geometries in seconds on Apple Silicon GPU via MLX (or NVIDIA via JAX/CUDA). Tier 2: full WarpX PIC on the top N candidates only. Tier 1 is implemented *after* Tier 2 is validated, so we know what the fast model is approximating and can quantify its error bounds.
- **Sensitivity analysis:** Sobol indices for which parameters most affect thrust and detachment. Critical for identifying what to measure experimentally.
- **Engineering constraints (v0.3+):** Real coil design is constrained by total coil mass budget (kg), resistive power dissipation or cryogenic power for superconducting coils, peak field at conductor (REBCO limit ~20 T), structural loads (magnetic pressure ~ B²/2μ₀), and thermal limits. These enter as inequality constraints in botorch's constrained optimization.
- **NOT gradient-based on WarpX internals** (PIC codes are stochastic; finite-difference gradients are noisy). Surrogate-based optimization is the correct approach for PIC-dependent objectives.

**Key files:**
```
helicon/optimize/
├── scan.py            # parameter sweep generation
├── surrogate.py       # GP surrogate model (botorch)
├── sensitivity.py     # Sobol analysis
└── objectives.py      # multi-objective (thrust vs Isp vs η_d)
```

#### Module 6: `helicon.validate`
**Purpose:** Automated validation against known solutions. This is what gives the tool credibility.

**Validation cases (v0.1):**

| Case | Reference | What it tests |
|---|---|---|
| Free expansion into vacuum | Analytic (adiabatic expansion of Maxwellian into diverging B) | Momentum conservation, basic thrust recovery |
| Guiding-center limit | Littlejohn (1983) guiding-center theory | Electron/ion orbits in slowly varying B |
| Collisionless detachment (high beta) | Merino & Ahedo (2016), PoP | Ion detachment onset, ambipolar potential |
| Resistive detachment | Dimov (2005); Moses (1991) semi-analytic | Effect of collisions on electron demagnetization |
| VASIMR plume benchmark | Olsen (2015) experimental + Chang-Díaz data | Only publicly available experimental nozzle data with sufficient detail |

Each case has a reference solution (analytic or digitized from publication) and an automated pass/fail criterion (e.g., thrust within 5% of reference, detachment front location within 10%).

**Key files:**
```
helicon/validate/
├── cases/
│   ├── free_expansion.py
│   ├── guiding_center.py
│   ├── merino_ahedo_2016.py
│   ├── resistive_dimov.py
│   └── vasimr_plume.py
├── reference_data/       # digitized curves, analytic solutions
├── runner.py             # run all validation cases
└── report.py             # pass/fail summary + comparison plots
```

---

## 4. Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| PIC solver | WarpX (≥ 24.08) | DOE-backed, high-performance (CPU/OpenMP on macOS, CUDA+ROCm on Linux), 2D-RZ support, open-source, actively maintained, already validated for plasma physics |
| Python bindings | pywarpx | Official WarpX Python interface for in-situ control |
| Language | Python 3.11+ | Ecosystem, accessibility. Performance-critical paths are in WarpX (C++/GPU) |
| Array computation | MLX (primary), NumPy (fallback) | MLX for Biot-Savart vectorization (`mlx.core.vmap`), autodiff through field computation (`mlx.core.grad`) for gradient-based coil optimization, and Metal GPU acceleration on Apple Silicon. NumPy fallback if MLX unavailable — optimizer falls back to gradient-free methods. Optional JAX backend for NVIDIA/CUDA users on Linux |
| Data format | openPMD (HDF5) | WarpX native output format; community standard for PIC data |
| Visualization | Matplotlib, ParaView (via openPMD-viewer) | Publication-quality 2D plots + 3D exploration |
| Optimization | botorch / MLX optimizers / scikit-optimize | Bayesian optimization via botorch (PyTorch MPS backend on Apple Silicon); gradient-based coil optimization via `mlx.optimizers`; handles noisy PIC objectives |
| Configuration | YAML + Pydantic | Human-readable config with strict validation |
| Testing | pytest + validation suite | Unit tests for postprocessing; physics validation as integration tests |
| CI/CD | GitHub Actions | Lint + unit tests on CPU; Apple Silicon tests via self-hosted macOS runner; GPU tests via self-hosted NVIDIA runner or manual |
| Documentation | MkDocs + mkdocstrings | API docs auto-generated; tutorials as Jupyter notebooks |

### 4.1 Hardware Requirements

**Primary Development Platform (Apple Silicon):**
- Apple M4 Pro or later, 24+ GB unified memory (M2 Pro/Max also supported)
- WarpX runs on CPU via OpenMP; MLX accelerates Python-layer computation (Biot-Savart, postprocessing, optimization) on the Apple GPU
- A single WarpX run for a 512×256 RZ grid with 10M particles completes in ~4-12 hours depending on timesteps and core count
- Unified memory eliminates CPU↔GPU transfer overhead for MLX operations

**High-Performance (NVIDIA GPU, Linux):**
- Single NVIDIA GPU (A100/H100 or consumer 3090/4090 with 24 GB VRAM)
- 64+ GB system RAM
- WarpX uses CUDA natively; ~2-8 hours per standard run
- Recommended for production parameter scans and large simulations

**Production (parameter scans, optimization):**
- Multi-GPU node or small cluster (NVIDIA)
- SLURM/PBS scheduler
- 10-100 runs per scan → ~1-4 days wall time on 4-8 GPUs
- Apple Silicon can orchestrate batch runs but WarpX is CPU-bound on macOS

**Minimum (any platform):**
- CPU with 8+ cores, 32 GB RAM
- WarpX CPU backend + NumPy (no MLX or CUDA required)
- Runs on any laptop. Slow, but functional.

---

## 5. Repository Structure

```
helicon/
├── helicon/                # main Python package
│   ├── __init__.py
│   ├── config/
│   ├── fields/
│   ├── runner/
│   ├── postprocess/
│   ├── optimize/
│   └── validate/
├── presets/                   # YAML configs for known engines
│   ├── sunbird.yaml
│   ├── dfd.yaml
│   ├── ppr.yaml
│   └── validation/
├── notebooks/                 # tutorial Jupyter notebooks
│   ├── 01_quickstart.ipynb
│   ├── 02_custom_nozzle.ipynb
│   ├── 03_parameter_scan.ipynb
│   └── 04_interpret_detachment.ipynb
├── tests/
│   ├── unit/                  # fast, no WarpX required
│   └── integration/           # requires WarpX, runs validation cases
├── docs/
│   ├── physics.md             # detailed physics documentation
│   ├── validation.md          # validation case descriptions + results
│   └── faq.md
├── scripts/
│   ├── run_validation.py      # one-command validation suite
│   └── plot_comparison.py     # generate publication figures
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── CITATION.cff
└── LICENSE                    # MIT
```

---

## 6. Interface & Usage

### 6.1 CLI

```bash
# Run a preset case
helicon run --preset sunbird

# Run on NVIDIA GPU (Linux)
helicon run --preset sunbird --gpu cuda

# Run a custom configuration
helicon run --config my_nozzle.yaml --output results/

# Run full validation suite
helicon validate --all

# Parameter scan
helicon scan --config my_nozzle.yaml \
    --vary coils.0.I:20000:80000:5 \
    --vary coils.1.z:0.2:0.6:5 \
    --output scan_results/

# Postprocess existing WarpX output
helicon postprocess --input warpx_output/ --metrics thrust,detachment,plume
```

### 6.2 Python API

```python
import helicon as mnx

# Load configuration
config = mnx.Config.from_yaml("my_nozzle.yaml")

# Pre-compute applied B-field
bfield = mnx.fields.compute(config.nozzle)
bfield.plot()           # quick visualization
bfield.save("bfield.h5")

# Run simulation
run = mnx.run(config)

# Extract propulsion metrics
metrics = mnx.postprocess(run.output_dir)
print(f"Thrust:      {metrics.thrust:.3f} N")
print(f"Isp:         {metrics.isp:.0f} s")
print(f"η_d (momentum): {metrics.detachment.momentum:.3f}")
print(f"η_d (particle): {metrics.detachment.particle:.3f}")
print(f"η_d (energy):   {metrics.detachment.energy:.3f}")
print(f"Plume half-angle: {metrics.plume_angle_deg:.1f}°")

# Parameter scan
scan = mnx.scan(
    config,
    vary={"coils.0.I": (20000, 80000, 10)},
    objectives=["thrust", "detachment.momentum"],
)
scan.plot_pareto()
```

### 6.3 Output Format

All results are written as:

- **openPMD/HDF5** — raw particle and field data (WarpX native)
- **JSON summary** — scalar propulsion metrics per run
- **CSV** — tabulated scan results for external analysis
- **Matplotlib figures** — auto-generated plots (field topology, detachment map, thrust convergence)

Example JSON summary:

```json
{
  "helicon_version": "0.1.0",
  "config_hash": "a3f8c1...",
  "timestamp": "2026-03-15T14:30:00Z",
  "nozzle_type": "converging_diverging",
  "plasma_source": {
    "species": ["D+", "e-"],
    "n0": 1e19,
    "T_i_eV": 5000,
    "T_e_eV": 2000,
    "v_injection_ms": 200000
  },
  "results": {
    "thrust_N": 4.82,
    "isp_s": 11200,
    "exhaust_velocity_ms": 109800,
    "mass_flow_rate_kgs": 4.39e-5,
    "detachment_efficiency": {
      "momentum_based": 0.87,
      "particle_based": 0.91,
      "energy_based": 0.83
    },
    "plume_half_angle_deg": 14.2,
    "beam_efficiency": 0.88,
    "radial_loss_fraction": 0.04,
    "convergence": {
      "thrust_relative_change_last_10pct": 0.003,
      "particle_count_exit": 8420000
    }
  },
  "validation_flags": {
    "steady_state_reached": true,
    "particle_statistics_sufficient": true,
    "energy_conservation_error": 0.002
  }
}
```

---

## 7. Validation Strategy

Validation is not optional. Helicon ships with automated validation, and every release must pass all cases before tagging.

### 7.1 Verification (does the code solve the equations correctly?)

- **Particle orbit convergence:** Single particle in known B-field; compare trajectory to analytic guiding-center solution. Error must decrease with timestep as expected (2nd order for Boris pusher).
- **Momentum conservation:** Zero-current plasma expanding into vacuum. Total momentum (particles + fields) must be conserved to < 0.1% over full simulation.
- **Energy conservation:** Same test. Total energy drift < 0.5% (PIC codes have finite energy error; this is a known limitation).
- **Grid convergence:** Run a reference case at 2× and 4× resolution. Thrust and detachment must converge (Richardson extrapolation within 5%).

### 7.2 Validation (does the code match reality?)

- **Merino & Ahedo (2016):** 2D fully kinetic simulations of collisionless magnetic nozzle. Published ion and electron velocity profiles at multiple axial stations. Our results must match within published uncertainty.
- **VASIMR plume data:** Ad Astra has published limited experimental plume measurements. Where data is available, compare thrust efficiency and divergence angle.
- **MN1D benchmark (Ahedo group):** 1D magnetic nozzle code with published results for multiple beta values. Our 2D code, averaged radially, must reproduce 1D results.

### 7.3 Validation Reporting

Every validation run generates a comparison plot (our result vs. reference) and a quantitative error metric. These plots are committed to the repository under `docs/validation_results/` and linked from the documentation.

**No result from Helicon should be published or cited without stating which validation cases the specific configuration has been tested against.** The tool enforces this by including a `validation_proximity` field in output — a measure of how similar the user's configuration is to the nearest validated case (in parameter space).

---

## 8. Roadmap

### v0.1 — Foundation (Months 1-3)

- [ ] Config parser + YAML schema + Pydantic models
- [ ] Biot-Savart field solver (MLX primary with NumPy fallback, tested against analytic dipole and single-coil exact solution)
- [ ] WarpX input generator for RZ geometry
- [ ] WarpX runner (single run, CPU/OpenMP on Apple Silicon, CUDA on Linux)
- [ ] Post-processing: thrust and mass flow rate from openPMD output
- [ ] Validation case: free expansion (momentum conservation)
- [ ] Validation case: single-particle guiding center orbit
- [ ] CLI: `helicon run` and `helicon postprocess`
- [ ] README, installation docs, one tutorial notebook
- [ ] CI: linting + unit tests
- [ ] Reproducibility infrastructure: every run logs config hash, code version (git SHA), WarpX version + commit hash, random seeds for PIC particle initialization, and environment info in output metadata

**Exit criterion:** Correctly compute thrust for a simple solenoid nozzle. Validation cases pass.

### v0.2 — Detachment Physics (Months 3-5)

- [ ] Detachment efficiency computation (all three definitions)
- [ ] Field line classifier (open/closed/separatrix topology)
- [ ] Particle tagging by field line at injection → track through simulation
- [ ] Plume divergence and beam efficiency metrics
- [ ] Thrust coefficient C_T computation (per Ahedo/Merino convention)
- [ ] Electron magnetization parameter mapped on grid
- [ ] Pressure anisotropy diagnostic
- [ ] Pulsed-mode postprocessing: impulse bit (N·s per pulse), per-pulse detachment efficiency, inter-pulse field relaxation diagnostics (required for PPR-class engines)
- [ ] Validation case: Merino & Ahedo (2016) collisionless nozzle
- [ ] Validation case: MN1D comparison at three beta values
- [ ] Preset configs: sunbird.yaml, dfd.yaml, ppr.yaml
- [ ] Second tutorial notebook (interpreting detachment results)

**Exit criterion:** Reproduce published detachment efficiency trends (η_d vs beta) from Merino & Ahedo within 10%.

### v0.3 — Optimization & Scans (Months 5-8)

- [ ] Parameter scan infrastructure (local + SLURM batch)
- [ ] Bayesian optimization with GP surrogate (botorch)
- [ ] Gradient-based coil optimization via MLX autodiff (`mlx.core.grad`, field-geometry objectives only)
- [ ] Optional JAX backend for NVIDIA/CUDA users on Linux
- [ ] Constrained optimization: coil mass, power dissipation, peak field at conductor, thermal limits as inequality constraints
- [ ] Sobol sensitivity analysis
- [ ] Multi-objective Pareto front visualization (thrust vs η_d vs coil mass)
- [ ] Fast pre-screening layer: analytical model (Breizman-Arefiev paraxial + Little-Choueiri C_T) for Tier 1 sweeps, validated against Tier 2 WarpX results with documented error bounds
- [ ] External B-field import (FEMM, CSV)
- [ ] FRC exhaust geometry support (closed + open field topology)
- [ ] Grid convergence study automation
- [ ] Reduced mass ratio mode: user-configurable m_i/m_e with guidance documentation (reduced ratio acceptable for qualitative trends and fast scans; full ratio required for quantitative detachment predictions; default full ratio for validation cases)
- [ ] Third tutorial notebook (parameter scan + optimization)

**Exit criterion:** Demonstrate that optimization finds a coil configuration with >10% higher η_d than the baseline for a reference case.

### v0.4 — Community & Integration (Months 8-12)

- [ ] Fluid-hybrid electron model option (faster parameter scans)
- [ ] Validation case: VASIMR plume data comparison
- [ ] Validation case: resistive detachment (Dimov)
- [ ] JSON output compatible with PropBench schema (if/when it exists)
- [ ] CSV output compatible with poliastro mission analysis input
- [ ] MkDocs documentation site with full API reference
- [ ] CITATION.cff and first preprint on arXiv (methodology + validation paper)
- [ ] Contributor guide and issue templates for community onboarding
- [ ] Community engagement: present at APS-DPP annual meeting (poster/talk), submit to IEPC, attend WarpX developer meetings, join PlasmaPy community calls, announce on Fusion Industry Association channels
- [ ] Native macOS/Apple Silicon install via Homebrew + pip
- [ ] Docker container and conda-lock environment for reproducible builds (Linux/NVIDIA)

**Exit criterion:** Peer review of methodology. At least one external group reproduces a validation case independently.

### v1.0 — Validated Research Tool (Month 12+)

- [ ] All validation cases passing with documented error bounds
- [ ] Published methodology paper (peer-reviewed)
- [ ] At least 3 external users/groups running the code
- [ ] Stable API; semantic versioning enforced

### v1.1 — MLX-Native Acceleration (Month 15+)

The core bottleneck on Apple Silicon is that WarpX runs CPU-only (no Metal backend). The strategy is to push every computation that *isn't* the PIC loop into MLX on the Apple GPU, and minimize how often a full WarpX run is needed.

- [ ] MLX-accelerated postprocessing: rewrite thrust integration, moment computation, and field-line classification as `mlx.core` operations — process openPMD particle data on Metal GPU instead of NumPy on CPU; target 5-10× speedup for postprocessing of 10M+ particle datasets on M4 Pro
- [ ] MLX Biot-Savart on full grids: extend current coil-level `vmap` to batch-evaluate B(r,z) on AMR-resolution grids (1024×512+) with `mlx.core.compile` for fused kernels; this is the inner loop of the Tier 1 analytical pre-screener
- [ ] Differentiable nozzle physics in MLX: implement the Breizman-Arefiev paraxial model and Little-Choueiri C_T model as end-to-end differentiable MLX graphs — enables `mlx.core.grad` through the entire Tier 1 analytical pipeline, not just field geometry
- [ ] In-situ MLX postprocessing via WarpX Python callbacks: compute thrust and η_d on the Metal GPU every N timesteps *during* WarpX execution, eliminating the need for full particle dumps to disk; reduces storage from ~5 GB to ~10 MB per run
- [ ] Adaptive mesh refinement (AMR) around detachment front using WarpX native AMR — resolve the narrow electron demagnetization layer without paying full grid cost; critical for making Apple Silicon CPU-bound WarpX runs practical
- [ ] Benchmark suite with published timing data on Apple M-series chips (M2 Pro, M4 Pro, M4 Max) for each validation case: wall time, Metal GPU utilization, unified memory high-water mark; gives users realistic expectations per chip tier

**Exit criterion:** Full postprocessing pipeline runs on Metal GPU via MLX. Tier 1 analytical screening evaluates 10,000 coil geometries in <60 seconds on M4 Pro. Standard validation case with AMR completes in <50% of uniform-grid wall time.

### v1.2 — Extended Physics (Month 18+)

- [ ] Self-consistent neutral dynamics: Monte Carlo neutrals with ionization, charge exchange, and recombination — replaces v0.1 static background model; critical for VASIMR-class engines with neutral propellant injection. Neutral particle push implemented in MLX for Metal GPU acceleration (neutrals don't need WarpX's electromagnetic solver)
- [ ] MLX-native fluid-hybrid electron solver: replace the v0.4 stub with a real CGL (Chew-Goldberger-Low) double-adiabatic electron fluid coupled to kinetic ions via WarpX — the fluid solve runs on Metal GPU via MLX while only ions are pushed by WarpX on CPU, cutting wall time ~10× vs. full kinetic electrons
- [ ] Multi-ion species detachment tracking: separate η_d computation for each ion species (D⁺, He²⁺, protons, α-particles) since heavier ions detach at different locations; MLX-accelerated species-resolved moment computation
- [ ] Anomalous transport models beyond parametrized collision frequency: lower-hybrid drift instability (LHDI) driven cross-field transport via sub-grid model calibrated to full kinetic runs; the sub-grid model itself runs in MLX as a closure on the fluid-hybrid path
- [ ] Full 3D simulations for non-axisymmetric nozzles (asymmetric coil placement, tilted exhaust); validate against 2D-RZ results for symmetric cases to quantify 3D overhead vs. accuracy trade-off. Note: 3D WarpX on Apple Silicon CPU is expensive — this is the use case that justifies optional cloud HPC offload (see v1.3)

**Exit criterion:** Fluid-hybrid electron path produces detachment results within 15% of full kinetic on Merino-Ahedo case at 10× lower wall time on M4 Pro. Multi-species detachment validated for D⁺/He²⁺ mix.

### v1.3 — Mission Integration (Month 21+)

- [ ] Throttle curve generation: map thrust and Isp as functions of input power and propellant flow rate across a grid of operating points; output as interpolation tables consumable by mission planners. Built on Tier 1 (MLX analytical) + selective Tier 2 (WarpX PIC) validation
- [ ] Native poliastro integration: Helicon performance tables feed directly into low-thrust trajectory optimization (e.g., Earth-Mars spiral with variable Isp)
- [ ] Spacecraft interaction model: backflow fraction (ions returning past thruster plane), spacecraft charging from exhaust plume electrons, magnetic torque from nozzle field on spacecraft bus — all computed in MLX postprocessing from particle exit data
- [ ] Pulsed mission profiles: for PPR-class engines, compute impulse-averaged performance over burst cycles (thrust/Isp/η_d averaged over pulse train) for trajectory integration
- [ ] Optional cloud HPC offload: submit large WarpX runs (3D, high-resolution parameter scans) to cloud GPU instances (Lambda, AWS p4d) directly from `helicon scan --cloud`; develop and postprocess locally on Apple Silicon, run PIC remotely only when needed
- [ ] Thermal-structural coil constraints: import coil thermal limits from external FEA exports as optimization constraints, closing the loop between magnetic design and structural feasibility

**Exit criterion:** Generate a complete thrust/Isp/η_d performance map for Sunbird configuration on a single M4 Pro Mac in <48 hours (Tier 1 sweep + Tier 2 PIC on top 10 candidates). Demonstrate end-to-end coil geometry → simulation → mission ΔV estimate.

### v2.0 — Engineering Design Tool (Month 24+)

- [ ] MLX-trained neural surrogate: train a small MLP on the PIC parameter scan database using `mlx.nn` — predicts thrust, η_d, and plume angle in microseconds on Metal GPU given coil geometry + plasma source parameters; document the surrogate's validated accuracy envelope and extrapolation boundaries
- [ ] Interactive local-first design app (Streamlit or Marimo notebook): real-time nozzle design exploration using the MLX surrogate on Metal GPU, with one-click dispatch to full WarpX PIC for selected candidate designs; runs entirely on a MacBook, no server required
- [ ] Uncertainty quantification: Monte Carlo propagation of input parameter uncertainties (coil tolerances, plasma source variability) through the MLX surrogate — 100,000 MC samples in seconds on Metal GPU to produce confidence intervals on performance predictions
- [ ] Coil manufacturability constraints: winding pack geometry, minimum bend radius, layer-wound vs. pancake topology, REBCO tape width discretization — ensures optimized designs are physically buildable
- [ ] Multi-fidelity optimization pipeline: Tier 1 analytical in MLX (seconds) → Tier 2 neural surrogate in MLX (microseconds, but trained) → Tier 3 full WarpX PIC (hours, local or cloud); automatic promotion of promising candidates up the fidelity ladder
- [ ] Provenance and audit trail: every design decision traceable from final coil geometry back through optimization history, PIC validation, and input assumptions
- [ ] Export to CAD: output optimized coil geometry as STEP/IGES files for mechanical integration

**Exit criterion:** An engineer on a MacBook runs the interactive app, explores 50 nozzle designs in an afternoon using the surrogate, dispatches 3 to full PIC, and gets a validated coil geometry recommendation with uncertainty bounds — all locally, no HPC cluster.

### v2.1 — Ecosystem & Community (Month 30+)

- [ ] Plugin architecture: third-party postprocessing modules and custom physics operators can be registered without modifying Helicon core
- [ ] WarpX upstream contributions: push for Metal/MPS backend in WarpX (or contribute OpenCL path); contribute nozzle-specific RZ boundary improvements and propulsion diagnostics back to WarpX mainline
- [ ] Jupyter widget for interactive field topology exploration: drag coils, see field lines and detachment surface update in real time via MLX surrogate on Metal GPU
- [ ] Multi-thruster array simulation: model plume-plume interaction for spacecraft with 2-4 nozzles (relevant for attitude control via differential thrust)
- [ ] Collaborative validation database: community-contributed experimental data with standardized metadata, enabling cross-validation beyond the original 5 cases
- [ ] Annual validation report: automated regression suite run on each WarpX release; published as living document on documentation site
- [ ] Apple Silicon performance guide: documented best practices for WarpX OpenMP tuning on M-series (thread pinning, memory bandwidth utilization, performance cores vs. efficiency cores), MLX compilation tips, and unified memory management for large particle counts

**Exit criterion:** At least 10 external users/groups. Plugin used by at least one group for a use case not anticipated in the original spec. Validation database has contributions from 3+ independent sources.

---

## 9. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| WarpX RZ mode has bugs or limitations for nozzle-relevant regimes | High | Low (mature code, but edge cases exist) | Engage WarpX developers early; file issues; contribute upstream fixes |
| Insufficient published experimental data for validation | High | Medium | Use multiple independent simulation codes (MN1D, published PIC results) as cross-validation; clearly label which cases have experimental vs. simulation-only validation |
| Full kinetic electron runs are too expensive for practical parameter scans | Medium | High | Fluid-hybrid electron option in v0.4 (MLX-native CGL solver in v1.2); GP surrogate reduces required evaluations by 10-50×; reduced mass ratio mode for qualitative scans; MLX-accelerated Tier 1 analytical pre-screening avoids unnecessary PIC runs; optional cloud HPC offload in v1.3 for cases that truly need full kinetic resolution |
| PIC noise corrupts detachment metrics | Medium | Medium | Large particle counts (>10M); time-averaging over multiple plasma transit times; statistical convergence checks in postprocessing |
| No community adoption | Medium | Medium | Start with validation paper establishing credibility; engage Ahedo group, Pulsar, Princeton DFD team directly; present at APS-DPP and IEPC |
| MLX maturity and API stability | Medium | Medium | MLX is under active development by Apple; pin to tested MLX releases; NumPy fallback for all MLX-dependent code; monitor MLX changelog for breaking changes |
| WarpX CPU-only performance on Apple Silicon | High | High | WarpX has no Metal backend; runs CPU/OpenMP only on macOS. Mitigated by: (a) pushing all non-PIC computation into MLX on Metal GPU (postprocessing, optimization, surrogate inference); (b) AMR to reduce grid cost (v1.1); (c) fluid-hybrid electrons in MLX to cut PIC wall time ~10× (v1.2); (d) optional cloud HPC offload for rare full-kinetic 3D runs (v1.3); (e) long-term: contribute Metal/OpenCL backend to WarpX upstream (v2.1) |
| Limited Apple Silicon adoption in HPC community | Medium | High | Apple Silicon is the primary development and single-user platform; cloud GPU instances (Lambda, AWS) available for production scans via `helicon scan --cloud` (v1.3). Most researchers develop locally — meeting them where they are (MacBooks) lowers adoption barrier |
| Unified memory constraints on Apple Silicon | Low | Medium | 24-96 GB unified memory is shared between CPU and GPU; large particle counts may require careful memory management. Mitigated by MLX lazy evaluation and WarpX's efficient memory use |
| WarpX version drift breaks backward compatibility | Medium | Medium | Pin to tested WarpX release in CI; run validation suite before accepting new WarpX versions; document minimum + maximum tested version |
| Storage costs for parameter scans | Low | High | A single RZ run with 10M particles produces ~0.5-10 GB depending on diagnostic frequency. A 100-run scan: 0.5-1 TB. Document storage estimates; provide diagnostic scheduling guidance that minimizes data volume while preserving postprocessing requirements; support reduced-diagnostic "scan mode" vs. full-diagnostic "analysis mode" |
| Fast pre-screening model gives false confidence | Medium | Medium | Tier 1 analytical model is only enabled after Tier 2 PIC validation; all Tier 1 outputs carry explicit error bounds derived from Tier 1 vs. Tier 2 comparison; documentation emphasizes that Tier 1 is for candidate selection, not quantitative prediction |

---

## 10. Success Criteria

Helicon succeeds if:

1. **It produces a number someone publishes.** A peer-reviewed paper uses Helicon results for a magnetic nozzle detachment study.

2. **It resolves a design question.** An engine team (Pulsar, RocketStar, Howe, or academic group) uses the optimization module to select a coil geometry and can cite the validated basis for that choice.

3. **It makes the community more honest.** By providing three definitions of detachment efficiency with clear documentation, it forces the field to stop comparing apples to oranges in conference papers.

The tool does not succeed merely by having GitHub stars or a pretty dashboard. It succeeds by being *trusted* — and trust comes from validation, not velocity.

---

## 11. Prior Art & Differentiation

| Existing Tool | What It Does | Gap Helicon Fills |
|---|---|---|
| WarpX | General-purpose PIC solver | No propulsion-specific postprocessing, presets, or optimization |
| EPOCH | UK PIC code | Less GPU-optimized; same postprocessing gap |
| MN1D (Ahedo group) | 1D magnetic nozzle fluid code | 1D only; not public; no kinetic electrons |
| BALOO (Merino) | 2D MN fluid/hybrid code | Not publicly available |
| PlasmaPy | General plasma analysis library | No PIC integration; no nozzle-specific tools |
| FUSE.jl | Tokamak integrated modeling | Wrong confinement geometry for propulsion |

Helicon is differentiated by being: (a) open-source, (b) accelerated via WarpX (CPU/CUDA) and MLX (Apple Silicon GPU), (c) propulsion-specific in its analysis pipeline, and (d) validation-first in its development philosophy.

---

## 12. Licensing & Contribution

**License:** MIT. Chosen for maximum adoption by both academic and commercial users (Pulsar, RocketStar, etc. need permissive licensing).

**Contribution model:** Fork + PR on GitHub. All PRs require:
- Unit tests for new postprocessing code
- Validation case update if physics scope changes
- Documentation for any user-facing API change

**Code of conduct:** Contributor Covenant v2.1.

**Citation:** All publications using Helicon must cite the methodology paper (once published) and the specific version used. CITATION.cff is provided in the repository root.

---

## 13. Units, Normalization & Mass Ratio

### 13.1 Unit Convention

Helicon uses **SI units throughout** (meters, kilograms, seconds, amperes, tesla, eV for temperatures). This matches WarpX's native unit system and avoids conversion errors.

For comparison with the magnetic nozzle literature (which frequently uses normalized units), the postprocessing module provides optional conversions to plasma-physics normalizations:

- Lengths normalized to ion skin depth: d_i = c / ω_pi
- Times normalized to ion cyclotron period: τ_ci = 2π / Ω_ci
- Velocities normalized to ion sound speed: c_s = √(T_e / m_i)
- Magnetic field normalized to throat value: B̃ = B / B_throat

These are computed and stored alongside SI values in output files but are never used internally.

### 13.2 Ion-to-Electron Mass Ratio

Full mass ratio PIC simulations (m_i/m_e = 3672 for deuterium, 7344 for helium-4) are expensive because electrons require ~√(m_i/m_e) ≈ 60× smaller timesteps than ions. This is the single largest driver of computational cost.

Helicon supports user-configurable mass ratio with the following guidance:

| Mass Ratio | Use Case | Cost | Detachment Accuracy |
|---|---|---|---|
| Full (m_i/m_e = 3672) | Quantitative detachment predictions, validation | 60× baseline | Required for publication-quality results |
| Reduced (m_i/m_e = 100-400) | Parameter scans, qualitative trends | 3-6× baseline | Captures ion detachment correctly; electron dynamics are qualitatively right but quantitatively shifted |
| Equal mass (m_i/m_e = 1) | Code verification only | 1× baseline | Physically meaningless for detachment; useful for testing momentum conservation |

**Default:** Full mass ratio for all preset configurations. Reduced ratio only via explicit user override with a logged warning in output metadata.

**Validation requirement:** Every validation case must pass at full mass ratio. Reduced-ratio results carry a metadata flag `mass_ratio_reduced: true` to prevent accidental citation of qualitative results as quantitative predictions.

---

## 14. Reproducibility

Every Helicon run must be fully reproducible from its output metadata alone. The following are logged automatically in the output JSON:

- `helicon_version`: package version string
- `helicon_git_sha`: git commit hash of the helicon installation
- `warpx_version`: WarpX version string
- `warpx_git_sha`: WarpX commit hash (if built from source)
- `config_hash`: SHA-256 of the input YAML file
- `config_contents`: full YAML config embedded in output
- `random_seed`: PIC particle initialization seed (set deterministically from config hash if not specified by user)
- `python_version`, `mlx_version`, `torch_version`, `numpy_version`: dependency versions
- `apple_silicon_chip`, `metal_version`, `mlx_device`: Apple Silicon hardware info (if applicable)
- `cuda_version`, `gpu_model`: NVIDIA hardware info (if applicable)
- `hostname`, `timestamp`: execution environment
- `wall_time_seconds`: total runtime

**Reproducibility contract:** Given the same `config_hash` and `helicon_git_sha` on the same hardware, the output must be bit-for-bit identical (deterministic PIC initialization, deterministic reduction order). On different hardware (e.g., different GPU), results may differ at floating-point level but must agree within statistical noise bounds documented for each metric.

From v0.4, a `Dockerfile` and `conda-lock.yml` are provided so that any result can be reproduced in an identical environment.

---

## 15. Data Management

### 15.1 Storage Estimates

| Run Type | Grid | Particles | Timesteps | Field Dumps | Particle Dumps | Total Size |
|---|---|---|---|---|---|---|
| Quick test | 128×64 | 1M | 5k | every 500 | none | ~50 MB |
| Standard | 512×256 | 10M | 50k | every 500 | every 5000 | ~5 GB |
| High-res | 1024×512 | 100M | 100k | every 500 | every 5000 | ~80 GB |
| Parameter scan (100 runs, standard) | — | — | — | — | — | ~500 GB |

### 15.2 Diagnostic Modes

To manage storage, Helicon provides two diagnostic presets:

**Scan mode** (default for parameter scans): Only reduced diagnostics (scalar thrust, Isp, η_d computed in-situ by WarpX callbacks). No full field or particle dumps. ~10 MB per run.

**Analysis mode** (default for single runs): Full field dumps at configurable intervals + particle snapshots for postprocessing. ~5 GB per run at standard resolution.

Users can override either preset via the `diagnostics` section of the YAML config.

### 15.3 Data Retention

Intermediate WarpX checkpoint files (for restart capability) are deleted after successful run completion unless the user specifies `keep_checkpoints: true`. This typically saves 50-80% of disk usage.

---

## 16. References

Core magnetic nozzle physics:

1. Arefiev, A.V. & Breizman, B.N. (2005). "Magnetohydrodynamic scenario of plasma detachment in a magnetic nozzle." *Physics of Plasmas*, 12, 043504.
2. Breizman, B.N. & Arefiev, A.V. (2008). "Paraxial model of a magnetic nozzle." *Physics of Plasmas*, 15, 057103.
3. Little, J.M. & Choueiri, E.Y. (2013). "Thrust and efficiency model for electron-driven magnetic nozzles." *Physics of Plasmas*, 20, 103501.
4. Merino, M. & Ahedo, E. (2016). "Fully magnetohydrodynamic plasma flow in a magnetic nozzle." *Physics of Plasmas*, 23, 023506.
5. Ahedo, E. & Merino, M. (2010). "Two-dimensional supersonic plasma acceleration in a magnetic nozzle." *Physics of Plasmas*, 17, 073501.

Detachment mechanisms:

6. Moses, R.W., Gerwin, R.A., & Schoenberg, K.F. (1991). "Resistive plasma detachment in nozzle based coaxial thrusters." *AIP Conference Proceedings*, 246.
7. Dimov, G.I. (2005). "The ambipolar trap." *Physics-Uspekhi*, 48(11), 1129.
8. Hooper, E.B. (1993). "Plasma detachment from a magnetic nozzle." *Journal of Propulsion and Power*, 9(5), 757-763.

WarpX:

9. Vay, J.-L., et al. (2018). "Warp-X: A new exascale computing platform for beam-plasma simulations." *Nuclear Instruments and Methods A*, 909, 476-479.
10. Fedeli, L., et al. (2022). "Pushing the frontier in the design of laser-based electron accelerators with groundbreaking mesh-refined particle-in-cell simulations on exascale-class supercomputers." *Proc. SC22*.

Experimental references:

11. Olsen, C.S., et al. (2015). "Investigation of plasma detachment from a magnetic nozzle in the plume of the VX-200 magnetoplasma thruster." *IEEE Transactions on Plasma Science*, 43(1), 252-268.
12. Chang-Díaz, F.R., et al. (2004). "The physics and engineering of the VASIMR engine." *AIAA 2004-0149*.

Array computation:

13. Apple MLX Team. "MLX: An array framework for Apple silicon." https://github.com/ml-explore/mlx

Optimization methods:

14. Balandat, M., et al. (2020). "BoTorch: A framework for efficient Monte-Carlo Bayesian optimization." *Advances in Neural Information Processing Systems*, 33.
15. Sobol, I.M. (2001). "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates." *Mathematics and Computers in Simulation*, 55(1-3), 271-280.

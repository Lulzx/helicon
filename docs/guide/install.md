# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Basic Install

```bash
pip install magnozzlex
```

## With GPU Acceleration (Apple Silicon)

```bash
pip install "magnozzlex[mlx]"
```

## With Optimization

```bash
pip install "magnozzlex[optimize]"   # scikit-learn for GP surrogate
pip install "magnozzlex[botorch]"    # botorch for Bayesian optimization
```

## Development Install

```bash
git clone https://github.com/magnozzlex/magnozzlex
cd magnozzlex
uv sync --all-extras
uv run pytest  # verify install
```

## WarpX (for PIC simulations)

WarpX must be installed separately. See the [WarpX installation guide](https://warpx.readthedocs.io/en/latest/install/).

On macOS (Apple Silicon):
```bash
# WarpX runs CPU/OpenMP on macOS — no GPU PIC
conda install -c conda-forge warpx
# or build from source with OpenMP
```

On Linux (NVIDIA GPU):
```bash
conda install -c conda-forge warpx=*=mpi_openmpi_cuda*
# or use the provided Dockerfile
```

Without WarpX, all MagNozzleX features work except actually running PIC simulations
(`dry_run=True` generates input files without launching WarpX).

## Docker

```bash
docker build -t magnozzlex .
docker run --rm -v $(pwd)/results:/app/results magnozzlex \
    run --preset dfd --output results/dfd --dry-run
```

## Conda Environment

```bash
conda env create -f environment.yml
conda activate magnozzlex
pip install -e .
```

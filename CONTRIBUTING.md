# Contributing to Helicon

Thank you for your interest in contributing to Helicon.

## Development Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/helicon/helicon
cd helicon
uv sync --all-extras
uv run pytest
```

On Apple Silicon (MLX backend):
```bash
uv sync --extra mlx
uv run pytest tests/unit/test_biot_savart.py -k mlx -v
```

## Running Tests

```bash
# All tests
uv run pytest

# Specific module
uv run pytest tests/unit/test_biot_savart.py -v

# Skip slow tests
uv run pytest -m "not slow"
```

## Code Style

Helicon uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

The CI pipeline enforces `ruff check` and `ruff format --check`. Fix issues before submitting a PR.

## Project Structure

```
helicon/
├── fields/          # Biot-Savart field solver (NumPy + MLX backends)
├── config/          # Pydantic simulation config parser
├── runner/          # WarpX launcher, hardware detection, convergence studies
├── postprocess/     # Thrust, detachment, plume metrics, report generation
├── optimize/        # Parameter scans, Bayesian opt, Sobol sensitivity, Pareto front
├── validate/        # Validation cases (VASIMR, Merino-Ahedo, Dimov, etc.)
└── cli.py           # Click CLI entry point
tests/
└── unit/            # pytest unit tests (no WarpX required)
docs/
└── spec.md          # Architecture and milestone specification
```

## Adding a Validation Case

1. Create `helicon/validate/cases/your_case.py` with a class following the pattern in `free_expansion.py`.
2. Implement `get_config() -> SimConfig`, `evaluate(output_dir) -> ValidationResult`.
3. Register in `helicon/validate/cases/__init__.py` and `helicon/validate/runner.py`.
4. Add tests in `tests/unit/test_your_case.py`.

## Adding an Optimization Objective

MLX gradient-based objectives live in `helicon/optimize/objectives.py`. They must:
- Accept `coil_params: mx.array` (shape `(N, 3)`, columns `[z, r, I]`) and grid arrays.
- Return a scalar `mx.array` (no `mx.eval` call — preserve the graph for `mx.grad`).

## Pull Request Guidelines

- Keep PRs focused on a single concern.
- All new code must have corresponding unit tests.
- Ensure `uv run pytest` passes before opening the PR.
- Reference relevant papers in docstrings using author-year format.

## Reporting Issues

Please open a GitHub issue with:
- Helicon version (`helicon --version`)
- Python version and platform
- Minimal reproducible example
- Full traceback

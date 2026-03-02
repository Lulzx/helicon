# Validation Overview

Helicon maintains a suite of physics validation tests against published experimental and numerical benchmarks.

## Running All Validation Cases

```bash
helicon validate --all
```

Or from Python:

```python
from helicon.validate.runner import run_all_validations

results = run_all_validations()
for name, result in results.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {name}: {result.summary}")
```

## Validation Cases

| Case | Reference | Observable | Tolerance |
|------|-----------|-----------|-----------|
| [Free Expansion](free_expansion.md) | Andersen et al. (1969) | Velocity distribution | < 5% RMS |
| [Merino-Ahedo](merino_ahedo.md) | Merino & Ahedo (2011) | Detachment efficiency vs R_B | < 10% |
| [VASIMR VX-200](vasimr.md) | Olsen et al. (2015) | Thrust, Isp | < 15% |
| [Resistive Detachment](dimov.md) | Dimov & Taskaev (2003) | Hall parameter threshold | within range |

## Validation Report

```python
from helicon.validate.report import generate_html_report

generate_html_report(results, output_dir="docs/validation_results/")
```

HTML reports with field plots are saved to `docs/validation_results/`.

## Adding Custom Validation Cases

Subclass `ValidationCase`:

```python
from helicon.validate.base import ValidationCase, ValidationResult

class MyBenchmark(ValidationCase):
    name = "my_benchmark"
    reference = "Author et al. (2020)"

    def get_config(self):
        return SimConfig.from_yaml("benchmarks/my_case.yaml")

    def run(self, output_dir, dry_run=False):
        ...
        return ValidationResult(passed=True, summary="Error < 5%")
```

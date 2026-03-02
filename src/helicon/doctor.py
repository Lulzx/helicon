"""Environment health checker for Helicon.

Inspects the Python runtime and installed packages to report which
Helicon features are available, and highlights any missing optional
dependencies or version mismatches.

CLI::

    helicon doctor
    helicon doctor --json
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass, field


@dataclass
class DepCheck:
    """Result of checking a single dependency.

    Attributes
    ----------
    name : str
        Package name (e.g. ``"numpy"``).
    available : bool
        Whether the package can be imported.
    version : str | None
        Installed version string, or ``None`` if unavailable.
    required : bool
        Whether this is a mandatory core dependency.
    note : str
        Short human-readable note (e.g. ``"enables MLX GPU acceleration"``).
    """

    name: str
    available: bool
    version: str | None
    required: bool
    note: str = ""

    def status_str(self) -> str:
        if self.available:
            return f"OK      {self.name} {self.version or ''}"
        label = "MISSING" if self.required else "OPTIONAL"
        return f"{label} {self.name}  — {self.note}"


@dataclass
class DoctorReport:
    """Full environment health report.

    Attributes
    ----------
    python_version : str
        Running Python version (``major.minor.micro``).
    python_ok : bool
        Whether the Python version meets the ``>=3.11`` requirement.
    checks : list[DepCheck]
        Per-package results.
    warpx_found : bool
        Whether the ``warpx`` binary was found in ``PATH``.
    warpx_path : str | None
        Full path to warpx binary if found.
    warpx_metal_found : bool
        Whether the warpx-metal Apple Silicon build was found.
    warpx_metal_path : str | None
        Root directory of the warpx-metal build if found.
    """

    python_version: str
    python_ok: bool
    checks: list[DepCheck] = field(default_factory=list)
    warpx_found: bool = False
    warpx_path: str | None = None
    warpx_metal_found: bool = False
    warpx_metal_path: str | None = None

    @property
    def all_required_ok(self) -> bool:
        """True if all mandatory deps are available."""
        return all(c.available for c in self.checks if c.required)

    @property
    def healthy(self) -> bool:
        return self.python_ok and self.all_required_ok

    def summary(self) -> str:
        lines = [
            "=" * 58,
            "Helicon Environment Doctor",
            "=" * 58,
            f"  Python:   {self.python_version}  "
            f"{'OK' if self.python_ok else 'REQUIRES >=3.11'}",
            "",
            "Dependencies:",
        ]
        for c in self.checks:
            lines.append(f"  {c.status_str()}")
        lines += [
            "",
            "WarpX binary:",
            "  FOUND  " + (self.warpx_path or "")
            if self.warpx_found
            else "  NOT FOUND — install WarpX separately",
            "",
            "WarpX Metal (Apple Silicon GPU):",
            "  FOUND  " + (self.warpx_metal_path or "")
            if self.warpx_metal_found
            else "  NOT FOUND — build warpx-metal for Apple GPU acceleration",
            "=" * 58,
            "Overall: " + ("HEALTHY" if self.healthy else "ISSUES FOUND"),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "python_ok": self.python_ok,
            "healthy": self.healthy,
            "warpx_found": self.warpx_found,
            "warpx_path": self.warpx_path,
            "warpx_metal_found": self.warpx_metal_found,
            "warpx_metal_path": self.warpx_metal_path,
            "checks": [
                {
                    "name": c.name,
                    "available": c.available,
                    "version": c.version,
                    "required": c.required,
                    "note": c.note,
                }
                for c in self.checks
            ],
        }


def _pkg_version(name: str) -> str | None:
    """Return installed version of *name*, or None."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_environment() -> DoctorReport:
    """Probe the runtime environment and return a :class:`DoctorReport`."""
    vi = sys.version_info
    py_ver = f"{vi.major}.{vi.minor}.{vi.micro}"
    py_ok = (vi.major, vi.minor) >= (3, 11)

    _DEPS: list[tuple[str, bool, str]] = [
        # (import_name, required, note)
        ("numpy", True, "numerical arrays"),
        ("scipy", True, "field integration and ODEs"),
        ("pydantic", True, "config validation"),
        ("yaml", True, "YAML config parsing"),
        ("click", True, "CLI framework"),
        ("h5py", True, "WarpX HDF5 output reading"),
        ("mlx", False, "Apple Silicon GPU acceleration"),
        ("sklearn", False, "Bayesian optimization / GP surrogate"),
        ("botorch", False, "advanced BoTorch acquisition functions"),
        ("streamlit", False, "interactive design explorer (helicon app)"),
        ("ipywidgets", False, "Jupyter interactive widgets"),
        ("matplotlib", False, "plotting and field-line visualisation"),
        ("tqdm", False, "progress bars for parameter scans"),
    ]

    # Map import name → package name for version lookup
    _PKG_NAME: dict[str, str] = {
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "mlx": "mlx",
    }

    checks: list[DepCheck] = []
    for imp, required, note in _DEPS:
        avail = _available(imp)
        pkg = _PKG_NAME.get(imp, imp)
        ver = _pkg_version(pkg) if avail else None
        checks.append(
            DepCheck(name=pkg, available=avail, version=ver, required=required, note=note)
        )

    warpx_path = shutil.which("warpx") or shutil.which("warpx.RZ") or shutil.which("warpx.3d")

    # warpx-metal detection (Apple Silicon native GPU build)
    warpx_metal_found = False
    warpx_metal_path: str | None = None
    try:
        from helicon.runner.metal_runner import detect_warpx_metal

        metal = detect_warpx_metal()
        if metal.valid:
            warpx_metal_found = True
            warpx_metal_path = str(metal.root)
    except Exception:
        pass

    return DoctorReport(
        python_version=py_ver,
        python_ok=py_ok,
        checks=checks,
        warpx_found=warpx_path is not None,
        warpx_path=warpx_path,
        warpx_metal_found=warpx_metal_found,
        warpx_metal_path=warpx_metal_path,
    )

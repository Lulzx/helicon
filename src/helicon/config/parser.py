"""YAML configuration parser with Pydantic models.

Defines the full configuration schema for Helicon simulations and
provides loading from YAML files or preset names.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class CoilConfig(BaseModel):
    """A single circular current loop."""

    z: float = Field(description="Axial position [m]")
    r: float = Field(gt=0, description="Coil radius [m]")
    I: float = Field(description="Current [A-turns]")


class DomainConfig(BaseModel):
    """Simulation domain extent."""

    z_min: float = Field(description="Upstream axial boundary [m]")
    z_max: float = Field(description="Downstream axial boundary [m]")
    r_max: float = Field(gt=0, description="Radial extent [m]")

    @model_validator(mode="after")
    def _check_z_order(self) -> DomainConfig:
        if self.z_max <= self.z_min:
            msg = f"z_max ({self.z_max}) must be greater than z_min ({self.z_min})"
            raise ValueError(msg)
        return self


class ResolutionConfig(BaseModel):
    """Grid resolution."""

    nz: int = Field(default=512, gt=0, description="Axial grid points")
    nr: int = Field(default=256, gt=0, description="Radial grid points")
    geometry: Literal["2d_rz", "3d"] = Field(
        default="2d_rz",
        description=(
            "Simulation geometry. '2d_rz' is the default axisymmetric cylindrical "
            "geometry (2D). '3d' enables full 3D Cartesian for non-axisymmetric "
            "nozzles (asymmetric coil placement, tilted exhaust). Note: 3D WarpX "
            "on Apple Silicon CPU is expensive — use cloud HPC offload (v1.3) for "
            "production 3D runs. Default: '2d_rz'."
        ),
    )
    np_phi: int = Field(
        default=1,
        gt=0,
        description=(
            "Azimuthal grid points (used only when geometry='3d'). "
            "For 2D-RZ this field is ignored. "
            "Recommended: 32–128 for production 3D runs."
        ),
    )


class NozzleConfig(BaseModel):
    """Nozzle geometry definition."""

    type: Literal["solenoid", "converging_diverging", "frc_exhaust"] = "solenoid"
    coils: list[CoilConfig] = Field(min_length=1)
    domain: DomainConfig
    resolution: ResolutionConfig = ResolutionConfig()


class NeutralsConfig(BaseModel):
    """Optional Monte Carlo background neutral gas (spec §2.1).

    When present, WarpX is configured to include neutral–ion charge-exchange
    collisions via Monte Carlo Collision (MCC) physics.
    """

    species: str = Field(
        default="D",
        description="Neutral species name (e.g. 'D', 'H', 'Xe').",
    )
    n_neutral_m3: Annotated[
        float,
        Field(gt=0, description="Background neutral number density [m⁻³]"),
    ]
    T_neutral_eV: float = Field(
        default=0.025,
        gt=0,
        description="Neutral temperature (≈ room temperature ≈ 0.025 eV).",
    )
    cx_cross_section_m2: float = Field(
        default=5e-19,
        gt=0,
        description="Charge-exchange cross section [m²]. Default: 5×10⁻¹⁹ m² (D–D+).",
    )
    ionization_cross_section_m2: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Electron-impact ionization cross section [m²]. None = disabled.",
        ),
    ] = None


class PlasmaSourceConfig(BaseModel):
    """Upstream plasma injection boundary condition."""

    species: list[str] = Field(default=["D+", "e-"])
    n0: Annotated[float, Field(gt=0, description="Number density [m^-3]")]
    T_i_eV: Annotated[float, Field(gt=0, description="Ion temperature [eV]")]
    T_e_eV: Annotated[float, Field(gt=0, description="Electron temperature [eV]")]
    v_injection_ms: Annotated[float, Field(gt=0, description="Injection velocity [m/s]")]
    mass_ratio: float | None = Field(
        default=None,
        description="Ion-to-electron mass ratio. None = physical ratio.",
    )
    electron_model: Literal["kinetic", "fluid"] = Field(
        default="kinetic",
        description=(
            "Electron treatment: 'kinetic' (default, full PIC) or 'fluid' "
            "(hybrid, faster parameter scans, less accurate electron detachment)."
        ),
    )
    neutrals: NeutralsConfig | None = Field(
        default=None,
        description=(
            "Optional background neutral gas for Monte Carlo charge-exchange "
            "and ionization collisions (spec §2.1). None = no neutrals."
        ),
    )


class DiagnosticsConfig(BaseModel):
    """Diagnostic output configuration."""

    mode: Literal["analysis", "scan"] = "analysis"
    field_dump_interval: int = Field(default=500, gt=0)
    particle_dump_interval: int = Field(default=5000, gt=0)


class SimConfig(BaseModel):
    """Top-level simulation configuration."""

    nozzle: NozzleConfig
    plasma: PlasmaSourceConfig
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    timesteps: int = Field(default=50000, gt=0)
    dt_multiplier: float = Field(default=0.95, gt=0, le=1.0)
    keep_checkpoints: bool = False
    random_seed: int | None = None
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_preset(cls, name: str) -> SimConfig:
        """Load a built-in preset configuration by name.

        Available presets: ``sunbird``, ``dfd``, ``ppr``.
        """
        preset_dir = importlib.resources.files("helicon.config.presets")
        preset_file = preset_dir / f"{name}.yaml"
        if not preset_file.is_file():
            available = [
                p.name.removesuffix(".yaml")
                for p in preset_dir.iterdir()
                if p.name.endswith(".yaml")
            ]
            msg = f"Unknown preset {name!r}. Available: {available}"
            raise ValueError(msg)
        text = preset_file.read_text()
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write this configuration to a YAML file."""
        path = Path(path)
        data = self.model_dump(mode="python")
        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

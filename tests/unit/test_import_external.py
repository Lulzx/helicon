"""Tests for magnozzlex.fields.import_external."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from magnozzlex.fields.biot_savart import BField
from magnozzlex.fields.import_external import load_csv_bfield, load_femm_bfield


# ---------------------------------------------------------------------------
# Helpers to write sample files
# ---------------------------------------------------------------------------
def _write_csv(path: Path, nr: int = 4, nz: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Write a synthetic CSV field map and return (Br, Bz) ground truth."""
    r_vals = np.linspace(0.0, 0.3, nr)
    z_vals = np.linspace(-0.5, 0.5, nz)
    Z, R = np.meshgrid(z_vals, r_vals)  # shapes (nr, nz)

    Br_true = 0.01 * R
    Bz_true = 0.05 * (1.0 - R / 0.3)

    rows = []
    for i, r in enumerate(r_vals):
        for j, z in enumerate(z_vals):
            rows.append(f"{r},{z},{Br_true[i, j]},{Bz_true[i, j]}")

    with path.open("w") as f:
        f.write("r,z,Br,Bz\n")
        for row in rows:
            f.write(row + "\n")

    return Br_true, Bz_true


def _write_femm(path: Path, nr: int = 3, nz: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Write a synthetic FEMM .ans-style file (coordinates in mm)."""
    r_mm = np.linspace(0.0, 100.0, nr)  # 0–100 mm
    z_mm = np.linspace(-200.0, 200.0, nz)
    Z, R = np.meshgrid(z_mm, r_mm)

    Br_true = 0.01 * R / 100.0  # in Tesla
    Bz_true = 0.05 * np.ones_like(R)

    with path.open("w") as f:
        for i, r in enumerate(r_mm):
            for j, z in enumerate(z_mm):
                Bmag = np.sqrt(Br_true[i, j] ** 2 + Bz_true[i, j] ** 2)
                f.write(f"{r}\t{z}\t{Br_true[i, j]:.6e}\t{Bz_true[i, j]:.6e}\t{Bmag:.6e}\n")

    return Br_true, Bz_true


# ---------------------------------------------------------------------------
# load_csv_bfield
# ---------------------------------------------------------------------------
class TestLoadCSVBField:
    def test_returns_bfield(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.csv"
            _write_csv(path)
            bf = load_csv_bfield(path)
        assert isinstance(bf, BField)

    def test_correct_shapes(self):
        nr, nz = 4, 5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.csv"
            _write_csv(path, nr=nr, nz=nz)
            bf = load_csv_bfield(path)
        assert bf.Br.shape == (nr, nz)
        assert bf.Bz.shape == (nr, nz)
        assert len(bf.r) == nr
        assert len(bf.z) == nz

    def test_values_match_ground_truth(self):
        nr, nz = 4, 5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.csv"
            Br_true, Bz_true = _write_csv(path, nr=nr, nz=nz)
            bf = load_csv_bfield(path)
        np.testing.assert_allclose(bf.Br, Br_true, atol=1e-12)
        np.testing.assert_allclose(bf.Bz, Bz_true, atol=1e-12)

    def test_backend_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.csv"
            _write_csv(path)
            bf = load_csv_bfield(path)
        assert bf.backend == "csv"

    def test_empty_coils(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.csv"
            _write_csv(path)
            bf = load_csv_bfield(path)
        assert bf.coils == []

    def test_non_meshgrid_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            with path.open("w") as f:
                f.write("r,z,Br,Bz\n0.0,0.0,0.0,0.0\n0.1,0.0,0.0,0.0\n0.2,0.1,0.0,0.0\n")
            with pytest.raises(ValueError, match="meshgrid"):
                load_csv_bfield(path)

    def test_custom_column_names(self):
        nr, nz = 3, 4
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.csv"
            r_vals = np.linspace(0.0, 0.2, nr)
            z_vals = np.linspace(0.0, 0.4, nz)
            with path.open("w") as f:
                f.write("radius,axial,Br_T,Bz_T\n")
                for r in r_vals:
                    for z in z_vals:
                        f.write(f"{r},{z},{0.01*r},{0.05}\n")
            bf = load_csv_bfield(path, r_col="radius", z_col="axial", Br_col="Br_T", Bz_col="Bz_T")
        assert bf.Br.shape == (nr, nz)


# ---------------------------------------------------------------------------
# load_femm_bfield
# ---------------------------------------------------------------------------
class TestLoadFEMMBField:
    def test_returns_bfield(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.ans"
            _write_femm(path)
            bf = load_femm_bfield(path)
        assert isinstance(bf, BField)

    def test_correct_shapes(self):
        nr, nz = 3, 4
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.ans"
            _write_femm(path, nr=nr, nz=nz)
            bf = load_femm_bfield(path)
        assert bf.Br.shape == (nr, nz)
        assert bf.Bz.shape == (nr, nz)

    def test_coordinate_conversion_mm_to_m(self):
        nr, nz = 3, 4
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.ans"
            _write_femm(path, nr=nr, nz=nz)  # 0–100 mm, -200–200 mm
            bf = load_femm_bfield(path)
        # After mm→m conversion: r in [0, 0.1], z in [-0.2, 0.2]
        assert bf.r.max() == pytest.approx(0.1, abs=1e-9)
        assert bf.z.min() == pytest.approx(-0.2, abs=1e-9)

    def test_backend_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.ans"
            _write_femm(path)
            bf = load_femm_bfield(path)
        assert bf.backend == "femm"

    def test_empty_coils(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.ans"
            _write_femm(path)
            bf = load_femm_bfield(path)
        assert bf.coils == []

    def test_skips_comment_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.ans"
            with path.open("w") as f:
                f.write("[Solution Data]\n")
                f.write("Block 1\n")
                f.write("0.0\t0.0\t0.05\t0.01\t0.051\n")
                f.write("50.0\t0.0\t0.03\t0.01\t0.032\n")
                f.write("0.0\t100.0\t0.04\t0.01\t0.041\n")
                f.write("50.0\t100.0\t0.02\t0.01\t0.022\n")
            bf = load_femm_bfield(path)
        assert bf.Br.shape == (2, 2)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.ans"
            path.write_text("[No data here]\n")
            with pytest.raises(ValueError, match="No valid field data"):
                load_femm_bfield(path)

    def test_custom_length_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.ans"
            with path.open("w") as f:
                f.write("1.0\t0.0\t0.05\t0.01\t0.051\n")
                f.write("2.0\t0.0\t0.03\t0.01\t0.032\n")
                f.write("1.0\t1.0\t0.04\t0.01\t0.041\n")
                f.write("2.0\t1.0\t0.02\t0.01\t0.022\n")
            bf = load_femm_bfield(path, length_scale=1.0)
        assert bf.r[0] == pytest.approx(1.0)

"""External magnetic field map loaders.

Supports importing B-field maps from:

- **CSV files** — any delimiter, header row with column names
- **FEMM .ans exports** — tab-delimited text from FEMM axisymmetric analyses
- **COMSOL exports** — space/tab delimited text with ``%`` comment header

The returned :class:`~helicon.fields.biot_savart.BField` uses an empty
coil list (``coils=[]``) and a ``backend`` tag of ``"csv"``, ``"femm"``,
or ``"comsol"`` to indicate the source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from helicon.fields.biot_savart import BField


def load_csv_bfield(
    path: str | Path,
    *,
    r_col: str = "r",
    z_col: str = "z",
    Br_col: str = "Br",
    Bz_col: str = "Bz",
    delimiter: str = ",",
) -> BField:
    """Load a B-field map from a CSV file.

    The CSV must have a header row and at minimum four columns for
    ``r``, ``z``, ``Br``, ``Bz``.  Data must lie on a regular meshgrid —
    all combinations of the unique ``r`` and ``z`` values must be present.

    Parameters
    ----------
    path : str or Path
    r_col, z_col, Br_col, Bz_col : str
        Column names in the header row.
    delimiter : str
        Field delimiter (default ``","``).

    Returns
    -------
    BField
        Arrays shaped ``(nr, nz)``.
    """
    path = Path(path)
    data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float)

    r_flat = data[r_col]
    z_flat = data[z_col]
    Br_flat = data[Br_col]
    Bz_flat = data[Bz_col]

    r_unique = np.unique(r_flat)
    z_unique = np.unique(z_flat)
    nr, nz = len(r_unique), len(z_unique)

    if nr * nz != len(r_flat):
        raise ValueError(
            f"CSV has {len(r_flat)} rows but {nr} unique r values and "
            f"{nz} unique z values — does not form a {nr}×{nz} meshgrid."
        )

    Br = Br_flat.reshape(nr, nz)
    Bz = Bz_flat.reshape(nr, nz)

    return BField(Br=Br, Bz=Bz, r=r_unique, z=z_unique, coils=[], backend="csv")


def load_femm_bfield(
    path: str | Path,
    *,
    length_scale: float = 1e-3,
) -> BField:
    """Load a B-field map exported from FEMM in text (.ans) format.

    FEMM axisymmetric analyses can export field data as a tab-delimited
    table.  The expected columns are ``r  z  Br  Bz  |B|`` (coordinates
    in mm by default).

    Parameters
    ----------
    path : str or Path
        Path to the FEMM ``.ans`` export file.
    length_scale : float
        Multiply coordinates by this factor to convert to SI metres.
        Default 1e-3 converts FEMM's default millimetres → metres.

    Returns
    -------
    BField
    """
    path = Path(path)
    rows: list[tuple[float, float, float, float]] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("[") or line.startswith("Block"):
                continue
            try:
                parts = line.split()
                if len(parts) >= 4:
                    r, z, Br, Bz = (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                    rows.append((r, z, Br, Bz))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid field data found in {path}")

    arr = np.array(rows)
    r_flat = arr[:, 0] * length_scale
    z_flat = arr[:, 1] * length_scale
    Br_flat = arr[:, 2]
    Bz_flat = arr[:, 3]

    r_unique = np.unique(r_flat)
    z_unique = np.unique(z_flat)
    nr, nz = len(r_unique), len(z_unique)

    Br = np.zeros((nr, nz))
    Bz = np.zeros((nr, nz))

    for ri, zi, br, bz in zip(r_flat, z_flat, Br_flat, Bz_flat):
        i = int(np.searchsorted(r_unique, ri))
        j = int(np.searchsorted(z_unique, zi))
        Br[i, j] = br
        Bz[i, j] = bz

    return BField(Br=Br, Bz=Bz, r=r_unique, z=z_unique, coils=[], backend="femm")


def load_comsol_bfield(
    path: str | Path,
    *,
    length_scale: float = 1.0,
    r_col: int | str | None = None,
    z_col: int | str | None = None,
    Br_col: int | str | None = None,
    Bz_col: int | str | None = None,
) -> BField:
    """Load a B-field map exported from COMSOL Multiphysics.

    COMSOL exports are space/tab-delimited text files where lines beginning
    with ``%`` are comments or header metadata.  The first non-comment line
    is a column-name header (also starting with ``%``), followed by numeric
    data rows.

    Expected column order (or override via ``*_col`` arguments):
    ``r  z  Br  Bz``  (SI units, metres and tesla).

    Example COMSOL export header::

        % COMSOL 6.1.0.357 export
        % Data: Magnetic field
        % r (m)    z (m)    mf.Br (T)    mf.Bz (T)
        0.00  -0.30  0.0021  0.1234
        ...

    Parameters
    ----------
    path : str or Path
        Path to the COMSOL export file.
    length_scale : float
        Multiply spatial coordinates by this factor.  Default 1.0 assumes
        COMSOL output is already in metres.  Use 1e-3 if exported in mm.
    r_col, z_col, Br_col, Bz_col : int or str, optional
        Column index (0-based) or partial column name to select.  When
        ``None`` (default), columns are assumed to be in order
        ``[r, z, Br, Bz]`` from the first four data columns.

    Returns
    -------
    BField
    """
    path = Path(path)

    col_names: list[str] = []
    rows: list[list[float]] = []

    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                # Last comment line that looks like a column header
                candidate = stripped.lstrip("% ").strip()
                if any(c.isalpha() for c in candidate):
                    # Looks like column names — parse words
                    col_names = candidate.split()
                continue
            try:
                vals = [float(v) for v in stripped.split()]
                if len(vals) >= 4:
                    rows.append(vals)
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid numeric data found in COMSOL export: {path}")

    arr = np.array(rows)

    def _resolve_col(spec: int | str | None, default: int) -> int:
        if spec is None:
            return default
        if isinstance(spec, int):
            return spec
        # Partial name match against col_names
        matches = [i for i, n in enumerate(col_names) if spec.lower() in n.lower()]
        if not matches:
            raise ValueError(f"Column {spec!r} not found in COMSOL header: {col_names}")
        return matches[0]

    ri = _resolve_col(r_col, 0)
    zi = _resolve_col(z_col, 1)
    bri = _resolve_col(Br_col, 2)
    bzi = _resolve_col(Bz_col, 3)

    r_flat = arr[:, ri] * length_scale
    z_flat = arr[:, zi] * length_scale
    Br_flat = arr[:, bri]
    Bz_flat = arr[:, bzi]

    r_unique = np.unique(r_flat)
    z_unique = np.unique(z_flat)
    nr, nz = len(r_unique), len(z_unique)

    Br = np.zeros((nr, nz))
    Bz = np.zeros((nr, nz))

    for ri_v, zi_v, br, bz in zip(r_flat, z_flat, Br_flat, Bz_flat):
        i = int(np.searchsorted(r_unique, ri_v))
        j = int(np.searchsorted(z_unique, zi_v))
        Br[i, j] = br
        Bz[i, j] = bz

    return BField(Br=Br, Bz=Bz, r=r_unique, z=z_unique, coils=[], backend="comsol")

"""Collaborative validation database for Helicon.

Provides infrastructure for community-contributed experimental and simulation
data with standardised metadata, enabling cross-validation beyond the built-in
validation cases.

Usage::

    from helicon.valdb import ValidationDatabase, ValidationRecord

    db = ValidationDatabase("~/.helicon/valdb")

    record = ValidationRecord(
        case_id="merino_ahedo_2016_table2",
        source="Merino & Ahedo (2016) — Plasma Sources Sci. Technol.",
        contributor="lulzx",
        data_type="simulation",
        parameters={"beta": 0.1, "B0_T": 0.05},
        metrics={"eta_d_momentum": 0.72, "eta_d_energy": 0.68},
        tags=["merino_ahedo", "collisionless"],
    )
    db.add(record)
    results = db.query(tags=["merino_ahedo"])
"""

from helicon.valdb.database import ValidationDatabase
from helicon.valdb.schema import ValidationRecord

__all__ = ["ValidationDatabase", "ValidationRecord"]

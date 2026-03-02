"""Standardised metadata schema for validation database records."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ValidationRecord(BaseModel):
    """A single validation data record.

    Fields
    ------
    record_id : str
        Unique identifier (auto-generated UUID4 if not provided).
    case_id : str
        Logical case name, e.g. ``"merino_ahedo_2016_table2"``.
    source : str
        Human-readable citation or description of data origin.
    contributor : str
        Name / identifier of the contributor.
    data_type : "experimental" | "simulation" | "analytical"
        Nature of the data.
    parameters : dict
        Input conditions (nozzle geometry, plasma parameters, etc.).
    metrics : dict
        Measured / computed quantities (η_d, thrust, Isp, …).
    tags : list[str]
        Free-form tags for filtering (e.g. ``["merino_ahedo", "collisionless"]``).
    helicon_version : str | None
        Helicon version used to generate the data (if applicable).
    warpx_version : str | None
        WarpX version used (if applicable).
    notes : str
        Free-text notes.
    created_at : datetime
        Record creation timestamp (UTC, auto-set).
    """

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    source: str
    contributor: str
    data_type: Literal["experimental", "simulation", "analytical"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    helicon_version: str | None = None
    warpx_version: str | None = None
    notes: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )

    model_config = {"frozen": False}

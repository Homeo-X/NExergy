"""Shared typed enumerations for Exergy Kernel v0."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum with readable values on formatting."""

    def __str__(self) -> str:
        return str(self.value)


class BoundaryType(StrEnum):
    ASSET = "asset"
    SUBSYSTEM = "subsystem"
    SITE = "site"
    FLEET = "fleet"
    REGION = "region"


class MeasurementMethod(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    SIMULATED = "simulated"
    FORECASTED = "forecasted"
    COMMANDED = "commanded"
    AUDITED = "audited"


class Carrier(StrEnum):
    ELECTRIC = "electric"
    THERMAL = "thermal"
    COOLING = "cooling"
    CHEMICAL = "chemical"
    MECHANICAL = "mechanical"
    PRESSURE = "pressure"
    HYDRAULIC = "hydraulic"
    RADIATIVE = "radiative"
    VIRTUAL = "virtual"


class QualityGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

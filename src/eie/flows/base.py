"""Base schemas shared by typed physical flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite

from eie.core.enums import Carrier, MeasurementMethod, QualityGrade
from eie.core.errors import BoundaryError, DomainError, MissingReferenceError
from eie.core.units import POWER, Dimension, dimension_for_unit, require_dimension, require_known_unit


def require_finite(value: float, field: str) -> None:
    if not isfinite(value):
        raise DomainError(f"{field} must be finite")


def require_non_negative(value: float, field: str) -> None:
    require_finite(value, field)
    if value < 0:
        raise DomainError(f"{field} must be >= 0")


def require_positive(value: float, field: str) -> None:
    require_finite(value, field)
    if value <= 0:
        raise DomainError(f"{field} must be > 0")


def require_probability(value: float, field: str) -> None:
    require_finite(value, field)
    if not 0.0 <= value <= 1.0:
        raise DomainError(f"{field} must be in [0, 1]")


def require_optional_probability(value: float | None, field: str) -> None:
    if value is not None:
        require_probability(value, field)


def require_binding(boundary_id: str | None, reference_state_id: str | None) -> None:
    if not boundary_id:
        raise BoundaryError("boundary_id is required")
    if not reference_state_id:
        raise MissingReferenceError("reference_state_id is required")


@dataclass(frozen=True)
class Metadata:
    """Audit metadata attached to measurements, estimates, commands, and ledger facts."""

    timestamp: datetime
    source: str
    method: MeasurementMethod | str
    unit: str | None = None
    confidence: float = 1.0
    uncertainty: float | None = None
    boundary_id: str | None = None
    reference_state_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise DomainError("metadata.timestamp must be a datetime")
        if not self.source:
            raise DomainError("metadata.source is required")
        if not isinstance(self.method, MeasurementMethod):
            object.__setattr__(self, "method", MeasurementMethod(self.method))
        require_known_unit(self.unit)
        require_probability(self.confidence, "metadata.confidence")
        if self.uncertainty is not None:
            require_non_negative(self.uncertainty, "metadata.uncertainty")

    def require_binding(self) -> None:
        require_binding(self.boundary_id, self.reference_state_id)

    def require_dimension(self, expected: Dimension, *, field: str = "metadata.unit") -> None:
        require_dimension(self.unit, expected, field=field)

    @property
    def dimension(self) -> Dimension | None:
        return dimension_for_unit(self.unit)


@dataclass(frozen=True)
class ExergyFlow:
    """A typed, boundary-bound, reference-state-bound exergy flow."""

    flow_id: str
    carrier: Carrier | str
    source_node_id: str
    target_node_id: str
    energy_rate_w: float
    exergy_rate_w: float
    quality_factor: float
    quality_grade: QualityGrade | str
    boundary_id: str
    reference_state_id: str
    metadata: Metadata
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.flow_id:
            raise DomainError("flow_id is required")
        if not isinstance(self.carrier, Carrier):
            object.__setattr__(self, "carrier", Carrier(self.carrier))
        if not self.source_node_id or not self.target_node_id:
            raise DomainError("source_node_id and target_node_id are required")
        require_non_negative(self.energy_rate_w, "energy_rate_w")
        require_non_negative(self.exergy_rate_w, "exergy_rate_w")
        require_finite(self.quality_factor, "quality_factor")
        if self.quality_factor < 0:
            raise DomainError("quality_factor must be >= 0 for a declared flow")
        if not isinstance(self.quality_grade, QualityGrade):
            object.__setattr__(self, "quality_grade", QualityGrade(self.quality_grade))
        require_binding(self.boundary_id, self.reference_state_id)
        if self.metadata.boundary_id is not None and self.metadata.boundary_id != self.boundary_id:
            raise BoundaryError("metadata.boundary_id does not match flow.boundary_id")
        if (
            self.metadata.reference_state_id is not None
            and self.metadata.reference_state_id != self.reference_state_id
        ):
            raise MissingReferenceError("metadata.reference_state_id does not match flow.reference_state_id")
        self.metadata.require_dimension(POWER)

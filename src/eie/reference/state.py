"""Dynamic reference state for exergy accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite

from eie.core.constants import STANDARD_ATMOSPHERE_PA, STANDARD_AMBIENT_TEMPERATURE_K
from eie.core.errors import DomainError
from eie.core.tolerance import DEFAULT_TOLERANCE, Tolerance


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_datetime(value: datetime, field: str) -> None:
    if not isinstance(value, datetime):
        raise DomainError(f"{field} must be a datetime")


def _require_positive_finite(value: float, field: str) -> None:
    if not isfinite(value) or value <= 0:
        raise DomainError(f"{field} must be finite and > 0")


def _require_probability(value: float, field: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise DomainError(f"{field} must be finite and in [0, 1]")


@dataclass(frozen=True)
class ReferenceState:
    """Reference environment required by every exergy value."""

    reference_state_id: str
    timestamp: datetime
    ambient_temperature_k: float = STANDARD_AMBIENT_TEMPERATURE_K
    ambient_pressure_pa: float = STANDARD_ATMOSPHERE_PA
    relative_humidity: float | None = None
    sky_temperature_k: float | None = None
    nominal_grid_voltage_v: float | None = None
    nominal_grid_frequency_hz: float | None = None
    marginal_carbon_kg_per_kwh: float | None = None
    marginal_price_per_kwh: float | None = None
    confidence: float = 1.0
    valid_until: datetime | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.reference_state_id:
            raise DomainError("reference_state_id is required")
        _require_datetime(self.timestamp, "timestamp")
        _require_positive_finite(self.ambient_temperature_k, "ambient_temperature_k")
        _require_positive_finite(self.ambient_pressure_pa, "ambient_pressure_pa")
        _require_probability(self.confidence, "confidence")
        if self.relative_humidity is not None:
            _require_probability(self.relative_humidity, "relative_humidity")
        if self.sky_temperature_k is not None:
            _require_positive_finite(self.sky_temperature_k, "sky_temperature_k")
        if self.nominal_grid_voltage_v is not None:
            _require_positive_finite(self.nominal_grid_voltage_v, "nominal_grid_voltage_v")
        if self.nominal_grid_frequency_hz is not None:
            _require_positive_finite(self.nominal_grid_frequency_hz, "nominal_grid_frequency_hz")
        if self.marginal_carbon_kg_per_kwh is not None and self.marginal_carbon_kg_per_kwh < 0:
            raise DomainError("marginal_carbon_kg_per_kwh must be >= 0")
        if self.valid_until is not None:
            _require_datetime(self.valid_until, "valid_until")
            if self.valid_until < self.timestamp:
                raise DomainError("valid_until must be at or after timestamp")

    def is_stale(self, at: datetime | None = None, tolerance: Tolerance = DEFAULT_TOLERANCE) -> bool:
        """Return True when the reference state should not be trusted."""

        at = utc_now() if at is None else at
        _require_datetime(at, "at")
        if self.valid_until is not None and at > self.valid_until:
            return True
        return at - self.timestamp > timedelta(seconds=tolerance.stale_reference_seconds)

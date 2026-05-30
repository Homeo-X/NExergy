"""Audited offset-temperature conversion (Phase 4).

Celsius and Fahrenheit are interval (offset) scales — their zero points are
arbitrary, so ratios of their values are physically meaningless.  Exergy
equations require absolute temperatures in Kelvin.

Every °C → K or °F → K conversion that crosses the offset boundary must go
through AuditedTemperatureConverter so the provenance is preserved in the
immutable ConversionRecord log.

Temperature *differences* (ΔT) are ratio-safe: ΔK = Δ°C, ΔK = Δ°F × 5/9.
delta_to_kelvin() handles these without creating an audit record, because
differences carry no offset ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite

from eie.core.errors import DomainError, OffsetScaleError, UnitError

_CELSIUS_OFFSET_K: float = 273.15
_FAHRENHEIT_SCALE: float = 5.0 / 9.0
_FAHRENHEIT_OFFSET_K: float = 459.67 * _FAHRENHEIT_SCALE  # ≈ 255.372 K

_KELVIN = "K"
_CELSIUS_UNITS = frozenset({"°C", "degC"})
_FAHRENHEIT_UNITS = frozenset({"°F", "degF"})
_ABSOLUTE_TEMP_UNITS = frozenset({_KELVIN}) | _CELSIUS_UNITS | _FAHRENHEIT_UNITS
_DELTA_UNITS = frozenset({"ΔK", "Δ°C", "Δ°F"})


def _to_k(value: float, unit: str) -> tuple[float, str]:
    """Return (value_k, conversion_label) without domain validation."""
    if unit == _KELVIN:
        return value, "K → K (identity)"
    if unit in _CELSIUS_UNITS:
        return value + _CELSIUS_OFFSET_K, "°C → K"
    if unit in _FAHRENHEIT_UNITS:
        return (value + 459.67) * _FAHRENHEIT_SCALE, "°F → K"
    raise UnitError(f"unsupported temperature unit: {unit!r}")


def _from_k(value_k: float, unit: str) -> tuple[float, str]:
    """Return (result, conversion_label) from Kelvin without domain validation."""
    if unit == _KELVIN:
        return value_k, "K → K (identity)"
    if unit in _CELSIUS_UNITS:
        return value_k - _CELSIUS_OFFSET_K, "K → °C"
    if unit in _FAHRENHEIT_UNITS:
        return value_k / _FAHRENHEIT_SCALE - 459.67, "K → °F"
    raise UnitError(f"unsupported target temperature unit: {unit!r}")


@dataclass(frozen=True)
class ConversionRecord:
    """Immutable provenance record for one offset-temperature conversion.

    Appended to AuditedTemperatureConverter._records on every to_kelvin()
    or from_kelvin() call.  Never mutated or deleted after creation.
    """

    source_value: float
    source_unit: str
    result_value: float
    result_unit: str
    converted_at: datetime
    conversion_label: str


@dataclass
class AuditedTemperatureConverter:
    """Converts between offset and ratio temperature scales with a full audit trail.

    Every to_kelvin() and from_kelvin() call appends an immutable
    ConversionRecord.  delta_to_kelvin() is audit-free because temperature
    differences carry no offset ambiguity.

    The converter is intentionally stateful — accumulating records lets callers
    review the full conversion history for a sensor session or control loop.
    """

    _records: list[ConversionRecord] = field(default_factory=list, repr=False)

    def to_kelvin(self, value: float, source_unit: str, *, at: datetime | None = None) -> float:
        """Convert a temperature to Kelvin, recording provenance.

        Parameters
        ----------
        value : float
            Temperature in source_unit.
        source_unit : str
            One of: "K", "°C", "degC", "°F", "degF".
        at : datetime | None
            Conversion timestamp; defaults to now (UTC).

        Returns
        -------
        float
            Temperature in Kelvin (> 0).

        Raises
        ------
        UnitError
            If value is non-finite or source_unit is unrecognised.
        DomainError
            If the resulting Kelvin value is ≤ 0 (at or below absolute zero).
        """
        if not isfinite(value):
            raise UnitError(f"temperature value must be finite, got {value!r}")
        result_k, label = _to_k(value, source_unit)
        if result_k <= 0.0:
            raise DomainError(
                f"temperature {value} {source_unit} converts to {result_k:.6f} K, "
                f"which is at or below absolute zero; exergy equations require T > 0 K"
            )
        self._records.append(ConversionRecord(
            source_value=value,
            source_unit=source_unit,
            result_value=result_k,
            result_unit=_KELVIN,
            converted_at=at or datetime.now(timezone.utc),
            conversion_label=label,
        ))
        return result_k

    def from_kelvin(self, value_k: float, target_unit: str, *, at: datetime | None = None) -> float:
        """Convert a Kelvin temperature to another scale, recording provenance.

        Parameters
        ----------
        value_k : float
            Temperature in Kelvin; must be > 0.
        target_unit : str
            One of: "K", "°C", "degC", "°F", "degF".
        at : datetime | None
            Conversion timestamp; defaults to now (UTC).

        Raises
        ------
        UnitError
            If value_k is non-finite or target_unit is unrecognised.
        DomainError
            If value_k ≤ 0.
        """
        if not isfinite(value_k):
            raise UnitError(f"temperature value must be finite, got {value_k!r}")
        if value_k <= 0.0:
            raise DomainError(
                f"value_k {value_k} K is at or below absolute zero"
            )
        result, label = _from_k(value_k, target_unit)
        self._records.append(ConversionRecord(
            source_value=value_k,
            source_unit=_KELVIN,
            result_value=result,
            result_unit=target_unit,
            converted_at=at or datetime.now(timezone.utc),
            conversion_label=label,
        ))
        return result

    def delta_to_kelvin(self, delta: float, source_unit: str) -> float:
        """Convert a temperature *difference* to Kelvin — no audit record created.

        ΔK = Δ°C (exactly); ΔK = Δ°F × 5/9.  No offset is applied, so no
        provenance record is needed.

        Parameters
        ----------
        delta : float
            Temperature difference in source_unit.
        source_unit : str
            One of: "K", "ΔK", "°C", "degC", "Δ°C", "°F", "degF", "Δ°F".

        Raises
        ------
        UnitError
            If delta is non-finite or source_unit is unrecognised.
        """
        if not isfinite(delta):
            raise UnitError(f"temperature delta must be finite, got {delta!r}")
        if source_unit in {_KELVIN, "ΔK", "°C", "degC", "Δ°C"}:
            return delta
        if source_unit in {"°F", "degF", "Δ°F"}:
            return delta * _FAHRENHEIT_SCALE
        raise UnitError(f"unsupported temperature-difference unit: {source_unit!r}")

    @property
    def records(self) -> tuple[ConversionRecord, ...]:
        """Immutable snapshot of the full conversion audit trail."""
        return tuple(self._records)

    def records_since(self, cutoff: datetime) -> list[ConversionRecord]:
        """All conversion records at or after cutoff (inclusive)."""
        return [r for r in self._records if r.converted_at >= cutoff]

    @property
    def conversion_count(self) -> int:
        """Total number of conversions recorded."""
        return len(self._records)


def to_kelvin(value: float, source_unit: str) -> float:
    """Stateless °C / °F → K conversion without an audit trail.

    For production use, prefer AuditedTemperatureConverter.to_kelvin() so
    every conversion is recorded with provenance.  This function is provided
    for equations and tests that need a one-off conversion.

    Raises
    ------
    UnitError
        If value is non-finite or source_unit is unrecognised.
    DomainError
        If the result is ≤ 0 K.
    """
    if not isfinite(value):
        raise UnitError(f"temperature value must be finite, got {value!r}")
    result_k, _ = _to_k(value, source_unit)
    if result_k <= 0.0:
        raise DomainError(
            f"temperature {value} {source_unit} → {result_k:.6f} K is at or below "
            f"absolute zero; exergy equations require T > 0 K"
        )
    return result_k


def require_ratio_temperature(value_k: float, label: str = "temperature") -> None:
    """Assert that value_k is a valid absolute Kelvin temperature (> 0).

    Call this at exergy equation entry points to catch accidental °C or °F
    values that slipped past the unit system without conversion.

    Raises
    ------
    UnitError
        If value_k is non-finite.
    DomainError
        If value_k ≤ 0, with a hint that °C / °F may have been passed.
    """
    if not isfinite(value_k):
        raise UnitError(f"{label} must be finite, got {value_k!r}")
    if value_k <= 0.0:
        raise DomainError(
            f"{label} {value_k} K is at or below absolute zero; "
            f"exergy equations require Kelvin — did you pass °C or °F without converting?"
        )

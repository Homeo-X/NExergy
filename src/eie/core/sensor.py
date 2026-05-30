"""Sensor ingestion pipeline (Phase 4).

Physical sensors report raw measurements — often in °C, °F, kW, bar.
The engine's exergy equations require Kelvin and SI units.  This module
provides the audited boundary between hardware and the physics layer.

Every temperature crossing from an offset scale (°C, °F) to Kelvin is
recorded in the AuditedTemperatureConverter log so the provenance of each
sensor value is traceable.  Non-temperature scalars are dimension-checked
and scale-converted without a separate audit record.

Usage example
-------------
    pipeline = SensorIngestionPipeline()
    t_k = pipeline.ingest_temperature(SensorReading("T-001", 22.5, "°C", ts))
    p_w = pipeline.ingest_scalar(SensorReading("P-001", 3.2, "kW", ts), POWER)
    dt_k = pipeline.ingest_temperature_delta(SensorReading("DT-001", 8.0, "°C", ts))
    log = pipeline.conversion_log   # immutable tuple of ConversionRecord
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite

from eie.core.errors import OffsetScaleError, UnitError
from eie.core.temperature import AuditedTemperatureConverter, ConversionRecord
from eie.core.units import (
    TEMPERATURE,
    Dimension,
    convert_value,
    dimension_for_unit,
    is_offset_scale_unit,
    require_known_unit,
)
from eie.flows.base import require_probability


@dataclass(frozen=True)
class SensorReading:
    """A single raw measurement from a physical sensor.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for the sensor that produced this reading.
    value : float
        Raw measured value in `unit`.
    unit : str
        Unit symbol (must be registered in UNIT_REGISTRY; may be °C or °F).
    measured_at : datetime
        UTC timestamp of the measurement.
    confidence : float
        Measurement confidence in [0, 1]; default 1.0.

    Raises
    ------
    ValueError
        If sensor_id is empty.
    UnitError
        If value is non-finite or unit is unknown.
    """

    sensor_id: str
    value: float
    unit: str
    measured_at: datetime
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise ValueError("sensor_id is required")
        if not isfinite(self.value):
            raise UnitError(f"sensor value must be finite, got {self.value!r}")
        require_known_unit(self.unit)
        require_probability(self.confidence, "confidence")


@dataclass
class SensorIngestionPipeline:
    """Audited boundary between hardware sensor readings and the physics engine.

    Converts raw SensorReadings to validated SI/Kelvin values.  Every
    absolute temperature conversion (°C → K, °F → K) is recorded as an
    immutable ConversionRecord.  Scalar (non-temperature) conversions are
    dimension-checked and scale-converted without an audit entry.

    Parameters
    ----------
    converter : AuditedTemperatureConverter
        Shared converter instance; injected for testing or composed across
        multiple pipelines in the same session.
    """

    converter: AuditedTemperatureConverter = field(
        default_factory=AuditedTemperatureConverter
    )

    def ingest_temperature(
        self,
        reading: SensorReading,
        *,
        at: datetime | None = None,
    ) -> float:
        """Convert a temperature reading to Kelvin, recording provenance.

        Accepts any temperature unit (K, °C, degC, °F, degF).  The conversion
        is delegated to the AuditedTemperatureConverter so every offset
        crossing is logged.

        Parameters
        ----------
        reading : SensorReading
            Raw temperature measurement.  Unit must be a temperature unit.
        at : datetime | None
            Conversion timestamp; defaults to now (UTC) if not supplied.

        Returns
        -------
        float
            Temperature in Kelvin (> 0).

        Raises
        ------
        UnitError
            If the reading's unit is not a temperature unit.
        DomainError
            If the converted Kelvin value is ≤ 0.
        """
        dim = dimension_for_unit(reading.unit)
        if dim != TEMPERATURE:
            raise UnitError(
                f"sensor {reading.sensor_id!r}: expected a temperature unit, "
                f"got {reading.unit!r} (dimension {dim})"
            )
        return self.converter.to_kelvin(
            reading.value, reading.unit, at=at or reading.measured_at
        )

    def ingest_scalar(
        self,
        reading: SensorReading,
        expected_dimension: Dimension,
    ) -> float:
        """Validate dimension and convert a non-temperature reading to SI.

        Parameters
        ----------
        reading : SensorReading
            Raw scalar measurement.
        expected_dimension : Dimension
            Required SI dimension (e.g. POWER, ENERGY, PRESSURE).

        Returns
        -------
        float
            Value in the base SI unit for expected_dimension.

        Raises
        ------
        UnitError
            If the reading's unit is unknown, wrong dimension, or is an
            offset-scale temperature (use ingest_temperature instead).
        """
        if is_offset_scale_unit(reading.unit):
            raise OffsetScaleError(
                f"sensor {reading.sensor_id!r}: unit {reading.unit!r} is an "
                f"offset-scale temperature — use ingest_temperature() instead"
            )
        dim = dimension_for_unit(reading.unit)
        if dim != expected_dimension:
            raise UnitError(
                f"sensor {reading.sensor_id!r}: expected dimension "
                f"{expected_dimension.label()}, got {reading.unit!r} "
                f"(dimension {dim.label() if dim else 'unknown'})"
            )
        base_unit = _si_base_unit(expected_dimension)
        if base_unit is None or reading.unit == base_unit:
            return reading.value
        return convert_value(reading.value, reading.unit, base_unit)

    def ingest_temperature_delta(self, reading: SensorReading) -> float:
        """Convert a temperature *difference* to Kelvin — no audit record.

        ΔK = Δ°C exactly; ΔK = Δ°F × 5/9.  No offset is applied, so no
        provenance record is needed.

        Parameters
        ----------
        reading : SensorReading
            Raw temperature-difference measurement.  Unit should be a
            temperature or delta-temperature unit.

        Returns
        -------
        float
            Temperature difference in Kelvin.

        Raises
        ------
        UnitError
            If the reading's unit is not recognised as a temperature unit.
        """
        dim = dimension_for_unit(reading.unit)
        if dim != TEMPERATURE:
            raise UnitError(
                f"sensor {reading.sensor_id!r}: expected a temperature unit "
                f"for delta conversion, got {reading.unit!r}"
            )
        return self.converter.delta_to_kelvin(reading.value, reading.unit)

    @property
    def conversion_log(self) -> tuple[ConversionRecord, ...]:
        """Immutable snapshot of all temperature conversion records."""
        return self.converter.records


def _si_base_unit(dimension: Dimension) -> str | None:
    """Return the SI base unit symbol for common dimensions, or None."""
    from eie.core.units import (
        ENERGY,
        MASS,
        MASS_FLOW,
        POWER,
        PRESSURE,
        TEMPERATURE,
        TIME,
    )
    _MAP = {
        POWER: "W",
        ENERGY: "J",
        PRESSURE: "Pa",
        TEMPERATURE: "K",
        MASS: "kg",
        MASS_FLOW: "kg/s",
        TIME: "s",
    }
    return _MAP.get(dimension)

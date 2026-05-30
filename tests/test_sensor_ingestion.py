"""Tests for Phase 4: sensor ingestion pipeline.

Covers:
- SensorReading construction and validation
- SensorIngestionPipeline.ingest_temperature (°C, °F, K, errors)
- SensorIngestionPipeline.ingest_scalar (dimension check, scale conversion)
- SensorIngestionPipeline.ingest_temperature_delta (no audit record)
- conversion_log audit trail
- Integration: require_ratio_temperature wired into equation boundaries
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from eie.core.errors import DomainError, OffsetScaleError, UnitError
from eie.core.sensor import SensorIngestionPipeline, SensorReading
from eie.core.units import ENERGY, POWER, PRESSURE, TEMPERATURE
from eie.exergy.cooling import cooling_service_exergy_rate
from eie.exergy.heat import heat_exergy_rate
from eie.flows.storage import ThermalLayer


@pytest.fixture
def ts() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def pipeline() -> SensorIngestionPipeline:
    return SensorIngestionPipeline()


# ── SensorReading construction ────────────────────────────────────────────────

def test_sensor_reading_valid(ts: datetime) -> None:
    r = SensorReading("T-001", 22.5, "°C", ts)
    assert r.sensor_id == "T-001"
    assert r.value == pytest.approx(22.5)
    assert r.unit == "°C"
    assert r.confidence == pytest.approx(1.0)


def test_sensor_reading_custom_confidence(ts: datetime) -> None:
    r = SensorReading("P-001", 1500.0, "W", ts, confidence=0.95)
    assert r.confidence == pytest.approx(0.95)


def test_sensor_reading_is_frozen(ts: datetime) -> None:
    r = SensorReading("T-001", 22.5, "°C", ts)
    with pytest.raises((AttributeError, TypeError)):
        r.value = 99.0  # type: ignore[misc]


def test_sensor_reading_requires_sensor_id(ts: datetime) -> None:
    with pytest.raises(ValueError, match="sensor_id"):
        SensorReading("", 22.5, "°C", ts)


def test_sensor_reading_requires_known_unit(ts: datetime) -> None:
    with pytest.raises(UnitError):
        SensorReading("T-001", 22.5, "BTU", ts)


def test_sensor_reading_rejects_nan(ts: datetime) -> None:
    with pytest.raises(UnitError):
        SensorReading("T-001", float("nan"), "°C", ts)


def test_sensor_reading_rejects_inf(ts: datetime) -> None:
    with pytest.raises(UnitError):
        SensorReading("T-001", float("inf"), "°C", ts)


def test_sensor_reading_rejects_invalid_confidence(ts: datetime) -> None:
    with pytest.raises((ValueError, DomainError)):
        SensorReading("T-001", 22.5, "°C", ts, confidence=1.5)


# ── ingest_temperature ────────────────────────────────────────────────────────

def test_ingest_celsius_to_kelvin(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("T-001", 25.0, "°C", ts)
    assert pipeline.ingest_temperature(r) == pytest.approx(298.15)


def test_ingest_fahrenheit_to_kelvin(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("T-001", 77.0, "°F", ts)
    assert pipeline.ingest_temperature(r) == pytest.approx(298.15)


def test_ingest_kelvin_passthrough(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("T-001", 300.0, "K", ts)
    assert pipeline.ingest_temperature(r) == pytest.approx(300.0)


def test_ingest_degC_ascii_alias(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("T-001", 0.0, "degC", ts)
    assert pipeline.ingest_temperature(r) == pytest.approx(273.15)


def test_ingest_temperature_wrong_unit_raises(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("P-001", 1500.0, "W", ts)
    with pytest.raises(UnitError, match="temperature unit"):
        pipeline.ingest_temperature(r)


def test_ingest_temperature_below_absolute_zero_raises(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    r = SensorReading("T-001", -300.0, "°C", ts)
    with pytest.raises(DomainError, match="absolute zero"):
        pipeline.ingest_temperature(r)


def test_ingest_temperature_records_audit_entry(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    r = SensorReading("T-001", 22.5, "°C", ts)
    pipeline.ingest_temperature(r)
    assert pipeline.converter.conversion_count == 1
    log = pipeline.conversion_log
    assert log[0].source_unit == "°C"
    assert log[0].result_unit == "K"


def test_ingest_temperature_uses_reading_timestamp(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    r = SensorReading("T-001", 20.0, "°C", ts)
    pipeline.ingest_temperature(r)
    assert pipeline.conversion_log[0].converted_at == ts


def test_ingest_temperature_at_override(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    r = SensorReading("T-001", 20.0, "°C", ts)
    override = datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc)
    pipeline.ingest_temperature(r, at=override)
    assert pipeline.conversion_log[0].converted_at == override


# ── ingest_scalar ─────────────────────────────────────────────────────────────

def test_ingest_scalar_kilowatt_to_watt(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("P-001", 3.2, "kW", ts)
    assert pipeline.ingest_scalar(r, POWER) == pytest.approx(3200.0)


def test_ingest_scalar_watt_passthrough(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("P-001", 1500.0, "W", ts)
    assert pipeline.ingest_scalar(r, POWER) == pytest.approx(1500.0)


def test_ingest_scalar_kwh_to_joule(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("E-001", 1.0, "kWh", ts)
    assert pipeline.ingest_scalar(r, ENERGY) == pytest.approx(3_600_000.0)


def test_ingest_scalar_bar_to_pascal(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("P-002", 2.0, "bar", ts)
    assert pipeline.ingest_scalar(r, PRESSURE) == pytest.approx(200_000.0)


def test_ingest_scalar_wrong_dimension_raises(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("T-001", 300.0, "K", ts)
    with pytest.raises(UnitError, match="dimension"):
        pipeline.ingest_scalar(r, POWER)


def test_ingest_scalar_offset_temperature_raises(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    r = SensorReading("T-001", 25.0, "°C", ts)
    with pytest.raises(OffsetScaleError):
        pipeline.ingest_scalar(r, TEMPERATURE)


def test_ingest_scalar_does_not_add_audit_record(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    r = SensorReading("P-001", 3.2, "kW", ts)
    pipeline.ingest_scalar(r, POWER)
    assert pipeline.converter.conversion_count == 0


# ── ingest_temperature_delta ──────────────────────────────────────────────────

def test_ingest_delta_celsius_to_kelvin(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("DT-001", 5.0, "°C", ts)
    assert pipeline.ingest_temperature_delta(r) == pytest.approx(5.0)


def test_ingest_delta_fahrenheit_to_kelvin(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("DT-001", 9.0, "°F", ts)
    assert pipeline.ingest_temperature_delta(r) == pytest.approx(5.0)


def test_ingest_delta_kelvin_passthrough(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("DT-001", 5.0, "ΔK", ts)
    assert pipeline.ingest_temperature_delta(r) == pytest.approx(5.0)


def test_ingest_delta_does_not_add_audit_record(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    r = SensorReading("DT-001", 5.0, "°C", ts)
    pipeline.ingest_temperature_delta(r)
    assert pipeline.converter.conversion_count == 0


def test_ingest_delta_wrong_unit_raises(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    r = SensorReading("P-001", 100.0, "W", ts)
    with pytest.raises(UnitError):
        pipeline.ingest_temperature_delta(r)


# ── conversion_log ────────────────────────────────────────────────────────────

def test_conversion_log_is_tuple(pipeline: SensorIngestionPipeline, ts: datetime) -> None:
    assert isinstance(pipeline.conversion_log, tuple)


def test_conversion_log_grows_with_temperature_ingestions(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    pipeline.ingest_temperature(SensorReading("T-001", 20.0, "°C", ts))
    pipeline.ingest_temperature(SensorReading("T-002", 68.0, "°F", ts))
    assert len(pipeline.conversion_log) == 2


def test_conversion_log_unaffected_by_scalar_ingestions(
    pipeline: SensorIngestionPipeline, ts: datetime
) -> None:
    pipeline.ingest_scalar(SensorReading("P-001", 1.5, "kW", ts), POWER)
    pipeline.ingest_temperature_delta(SensorReading("DT-001", 3.0, "°C", ts))
    assert len(pipeline.conversion_log) == 0


# ── require_ratio_temperature wired into equation boundaries ──────────────────

def test_heat_exergy_rejects_negative_temperature_with_kelvin_hint() -> None:
    # -248.15 is what you'd pass if you wrote celsius_value - 273.15 by mistake
    with pytest.raises(DomainError, match="Kelvin"):
        heat_exergy_rate(1000.0, -248.15, 300.0)


def test_cooling_exergy_rejects_sub_zero_cold_temperature() -> None:
    with pytest.raises(DomainError, match="absolute zero"):
        cooling_service_exergy_rate(500.0, -5.0, 300.0)


def test_thermal_layer_rejects_celsius_temperature() -> None:
    with pytest.raises(DomainError, match="Kelvin"):
        ThermalLayer(temperature_k=60.0 - 273.15, energy_j=1000.0)


def test_thermal_layer_rejects_zero_kelvin() -> None:
    with pytest.raises(DomainError):
        ThermalLayer(temperature_k=0.0, energy_j=1000.0)

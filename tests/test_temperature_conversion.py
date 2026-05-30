"""Tests for Phase 4: audited offset-temperature conversion.

Covers:
- is_offset_scale_unit() classification
- convert_value() refusal of offset-scale units (OffsetScaleError)
- Delta-unit scale-only conversion (ΔK = Δ°C, ΔK = Δ°F × 5/9)
- to_kelvin() free function: °C, °F, K, domain errors
- AuditedTemperatureConverter: to_kelvin, from_kelvin, delta_to_kelvin
- Audit trail: records, records_since, conversion_count, immutability
- require_ratio_temperature() guard
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from eie.core.errors import DomainError, OffsetScaleError, UnitError
from eie.core.temperature import (
    AuditedTemperatureConverter,
    ConversionRecord,
    require_ratio_temperature,
    to_kelvin,
)
from eie.core.units import (
    TEMPERATURE,
    convert_value,
    dimension_for_unit,
    is_offset_scale_unit,
)


# ── is_offset_scale_unit ────────────────────────────────────────────────────────────────────────────

def test_celsius_is_offset_scale() -> None:
    assert is_offset_scale_unit("°C") is True
    assert is_offset_scale_unit("degC") is True


def test_fahrenheit_is_offset_scale() -> None:
    assert is_offset_scale_unit("°F") is True
    assert is_offset_scale_unit("degF") is True


def test_kelvin_is_not_offset_scale() -> None:
    assert is_offset_scale_unit("K") is False


def test_delta_units_are_not_offset_scale() -> None:
    assert is_offset_scale_unit("ΔK") is False
    assert is_offset_scale_unit("Δ°C") is False
    assert is_offset_scale_unit("Δ°F") is False


def test_unknown_unit_returns_false() -> None:
    assert is_offset_scale_unit("BTU") is False


# ── convert_value refuses offset-scale temperatures ─────────────────────────────────────────────────

def test_convert_value_rejects_celsius_source() -> None:
    with pytest.raises(OffsetScaleError, match="offset-scale"):
        convert_value(25.0, "°C", "K")


def test_convert_value_rejects_celsius_target() -> None:
    with pytest.raises(OffsetScaleError, match="offset-scale"):
        convert_value(298.15, "K", "°C")


def test_convert_value_rejects_fahrenheit_source() -> None:
    with pytest.raises(OffsetScaleError):
        convert_value(77.0, "°F", "K")


def test_convert_value_rejects_degC_ascii() -> None:
    with pytest.raises(OffsetScaleError):
        convert_value(25.0, "degC", "K")


# ── delta units work with scale-only convert_value ────────────────────────────────────────────────

def test_delta_kelvin_to_delta_celsius_is_identity() -> None:
    assert convert_value(5.0, "ΔK", "Δ°C") == pytest.approx(5.0)
    assert convert_value(5.0, "Δ°C", "ΔK") == pytest.approx(5.0)


def test_delta_fahrenheit_to_delta_kelvin() -> None:
    assert convert_value(9.0, "Δ°F", "ΔK") == pytest.approx(5.0)
    assert convert_value(5.0, "ΔK", "Δ°F") == pytest.approx(9.0)


def test_delta_units_have_temperature_dimension() -> None:
    assert dimension_for_unit("ΔK") == TEMPERATURE
    assert dimension_for_unit("Δ°C") == TEMPERATURE
    assert dimension_for_unit("Δ°F") == TEMPERATURE


# ── to_kelvin free function ───────────────────────────────────────────────────────────────────────────────────

def test_zero_celsius_to_kelvin() -> None:
    assert to_kelvin(0.0, "°C") == pytest.approx(273.15)


def test_hundred_celsius_to_kelvin() -> None:
    assert to_kelvin(100.0, "°C") == pytest.approx(373.15)


def test_negative_celsius_to_kelvin() -> None:
    assert to_kelvin(-40.0, "°C") == pytest.approx(233.15)


def test_below_absolute_zero_celsius_raises() -> None:
    with pytest.raises(DomainError, match="absolute zero"):
        to_kelvin(-274.0, "°C")


def test_freezing_point_fahrenheit() -> None:
    assert to_kelvin(32.0, "°F") == pytest.approx(273.15)


def test_boiling_point_fahrenheit() -> None:
    assert to_kelvin(212.0, "°F") == pytest.approx(373.15)


def test_below_absolute_zero_fahrenheit_raises() -> None:
    with pytest.raises(DomainError, match="absolute zero"):
        to_kelvin(-500.0, "°F")


def test_kelvin_passthrough() -> None:
    assert to_kelvin(298.15, "K") == pytest.approx(298.15)


def test_degC_ascii_alias() -> None:
    assert to_kelvin(25.0, "degC") == pytest.approx(298.15)


def test_degF_ascii_alias() -> None:
    assert to_kelvin(77.0, "degF") == pytest.approx(298.15)


def test_to_kelvin_rejects_unknown_unit() -> None:
    with pytest.raises(UnitError):
        to_kelvin(25.0, "BTU")


def test_to_kelvin_rejects_nan() -> None:
    with pytest.raises(UnitError):
        to_kelvin(float("nan"), "°C")


def test_to_kelvin_rejects_inf() -> None:
    with pytest.raises(UnitError):
        to_kelvin(float("inf"), "°C")


# ── AuditedTemperatureConverter.to_kelvin ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def conv() -> AuditedTemperatureConverter:
    return AuditedTemperatureConverter()


@pytest.fixture
def ts() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def test_converter_celsius_to_kelvin(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    assert conv.to_kelvin(25.0, "°C", at=ts) == pytest.approx(298.15)


def test_converter_fahrenheit_to_kelvin(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    assert conv.to_kelvin(77.0, "°F", at=ts) == pytest.approx(298.15)


def test_converter_kelvin_passthrough(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    assert conv.to_kelvin(300.0, "K", at=ts) == pytest.approx(300.0)


def test_converter_below_absolute_zero_raises(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    with pytest.raises(DomainError, match="absolute zero"):
        conv.to_kelvin(-274.0, "°C", at=ts)


def test_converter_nan_raises(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    with pytest.raises(UnitError):
        conv.to_kelvin(float("nan"), "°C", at=ts)


# ── AuditedTemperatureConverter.from_kelvin ──────────────────────────────────────────────────────────────────────────

def test_converter_kelvin_to_celsius(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    assert conv.from_kelvin(273.15, "°C", at=ts) == pytest.approx(0.0)


def test_converter_kelvin_to_fahrenheit(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    assert conv.from_kelvin(373.15, "°F", at=ts) == pytest.approx(212.0)


def test_converter_kelvin_to_kelvin(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    assert conv.from_kelvin(300.0, "K", at=ts) == pytest.approx(300.0)


def test_converter_from_kelvin_below_zero_raises(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    with pytest.raises(DomainError):
        conv.from_kelvin(-1.0, "°C", at=ts)


def test_converter_from_kelvin_unknown_unit_raises(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    with pytest.raises(UnitError):
        conv.from_kelvin(300.0, "BTU", at=ts)


# ── Audit trail ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_audit_trail_grows_on_each_call(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    conv.to_kelvin(25.0, "°C", at=ts)
    conv.to_kelvin(77.0, "°F", at=ts)
    assert conv.conversion_count == 2


def test_audit_trail_records_correct_fields(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    conv.to_kelvin(0.0, "°C", at=ts)
    r = conv.records[0]
    assert r.source_value == pytest.approx(0.0)
    assert r.source_unit == "°C"
    assert r.result_value == pytest.approx(273.15)
    assert r.result_unit == "K"
    assert r.converted_at == ts
    assert "°C" in r.conversion_label


def test_conversion_record_is_frozen(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    conv.to_kelvin(20.0, "°C", at=ts)
    r = conv.records[0]
    assert isinstance(r, ConversionRecord)
    with pytest.raises((AttributeError, TypeError)):
        r.source_value = 99.0  # type: ignore[misc]


def test_records_property_returns_tuple(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    conv.to_kelvin(0.0, "°C", at=ts)
    assert isinstance(conv.records, tuple)


def test_records_since_filters_by_time(conv: AuditedTemperatureConverter) -> None:
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)
    conv.to_kelvin(0.0, "°C", at=t1)
    conv.to_kelvin(20.0, "°C", at=t2)
    conv.to_kelvin(40.0, "°C", at=t3)
    cutoff = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    since = conv.records_since(cutoff)
    assert len(since) == 2
    assert since[0].source_value == pytest.approx(20.0)
    assert since[1].source_value == pytest.approx(40.0)


def test_from_kelvin_also_creates_record(conv: AuditedTemperatureConverter, ts: datetime) -> None:
    conv.from_kelvin(298.15, "°C", at=ts)
    assert conv.conversion_count == 1
    r = conv.records[0]
    assert r.source_unit == "K"
    assert r.result_unit == "°C"


def test_records_since_cutoff_inclusive(conv: AuditedTemperatureConverter) -> None:
    ts = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    conv.to_kelvin(20.0, "°C", at=ts)
    assert len(conv.records_since(ts)) == 1


# ── delta_to_kelvin ──────────────────────────────────────────────────────────────────────────────────────────

def test_delta_celsius_equals_delta_kelvin(conv: AuditedTemperatureConverter) -> None:
    assert conv.delta_to_kelvin(5.0, "°C") == pytest.approx(5.0)
    assert conv.delta_to_kelvin(5.0, "ΔK") == pytest.approx(5.0)
    assert conv.delta_to_kelvin(5.0, "Δ°C") == pytest.approx(5.0)


def test_delta_fahrenheit_scale(conv: AuditedTemperatureConverter) -> None:
    assert conv.delta_to_kelvin(9.0, "°F") == pytest.approx(5.0)
    assert conv.delta_to_kelvin(9.0, "Δ°F") == pytest.approx(5.0)


def test_delta_does_not_create_audit_record(conv: AuditedTemperatureConverter) -> None:
    conv.delta_to_kelvin(10.0, "°C")
    conv.delta_to_kelvin(18.0, "°F")
    assert conv.conversion_count == 0


def test_delta_unknown_unit_raises(conv: AuditedTemperatureConverter) -> None:
    with pytest.raises(UnitError):
        conv.delta_to_kelvin(5.0, "BTU")


def test_delta_nan_raises(conv: AuditedTemperatureConverter) -> None:
    with pytest.raises(UnitError):
        conv.delta_to_kelvin(float("nan"), "°C")


# ── require_ratio_temperature ─────────────────────────────────────────────────────────────────────────────────

def test_valid_kelvin_passes() -> None:
    require_ratio_temperature(298.15)  # no exception


def test_high_kelvin_passes() -> None:
    require_ratio_temperature(1000.0)


def test_zero_kelvin_raises() -> None:
    with pytest.raises(DomainError, match="absolute zero"):
        require_ratio_temperature(0.0)


def test_negative_kelvin_raises() -> None:
    with pytest.raises(DomainError):
        require_ratio_temperature(-1.0)


def test_nan_raises_unit_error() -> None:
    with pytest.raises(UnitError):
        require_ratio_temperature(float("nan"))


def test_custom_label_appears_in_error() -> None:
    with pytest.raises(DomainError, match="ambient_temperature_k"):
        require_ratio_temperature(0.0, "ambient_temperature_k")


def test_celsius_value_without_conversion_raises() -> None:
    with pytest.raises(DomainError):
        require_ratio_temperature(25.0 - 273.15)  # 25°C passed as raw float ≈ -248.15

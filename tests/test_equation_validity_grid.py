from __future__ import annotations

import math

import pytest

from eie.core.enums import Carrier, QualityGrade
from eie.core.errors import DomainError
from eie.exergy.cooling import cooling_service_exergy_rate
from eie.exergy.electrical import electrical_service_exergy_rate
from eie.exergy.heat import (
    ColdThermalDomainError,
    finite_stream_heat_exergy_rate,
    heat_exergy_rate,
)
from eie.exergy.quality import classify_quality, validate_quality_factor
from eie.exergy.storage import thermal_storage_exergy
from eie.flows.storage import ThermalLayer


def test_hot_heat_exergy_is_monotonic_with_source_temperature():
    t0 = 300.0
    heat_rate = 1_000.0
    temperatures = [301.0, 320.0, 350.0, 400.0, 600.0, 1_000.0]
    values = [heat_exergy_rate(heat_rate, temperature, t0) for temperature in temperatures]

    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert all(0.0 < value < heat_rate for value in values)


def test_hot_heat_exergy_is_linear_in_heat_rate_for_fixed_temperature():
    t0 = 300.0
    source_temperature = 450.0

    base = heat_exergy_rate(500.0, source_temperature, t0)
    doubled = heat_exergy_rate(1_000.0, source_temperature, t0)

    assert doubled == pytest.approx(2.0 * base)


def test_hot_heat_exergy_never_returns_negative_for_zero_or_positive_hot_heat():
    for heat_rate in [0.0, 1.0, 100.0, 10_000.0]:
        assert heat_exergy_rate(heat_rate, 350.0, 300.0) >= 0.0


@pytest.mark.parametrize(
    ("mass_flow", "cp", "tin", "tout", "t0"),
    [
        (0.1, 4_180.0, 340.0, 310.0, 300.0),
        (1.0, 1_005.0, 500.0, 400.0, 298.15),
        (2.5, 2_000.0, 900.0, 600.0, 300.0),
    ],
)
def test_finite_stream_exergy_is_nonnegative_and_below_sensible_heat(mass_flow, cp, tin, tout, t0):
    exergy = finite_stream_heat_exergy_rate(mass_flow, cp, tin, tout, t0)
    sensible_heat = mass_flow * cp * (tin - tout)

    assert exergy >= 0.0
    assert exergy < sensible_heat


def test_finite_stream_zero_flow_or_zero_cp_returns_zero():
    assert finite_stream_heat_exergy_rate(0.0, 4_180.0, 350.0, 320.0, 300.0) == 0.0
    assert finite_stream_heat_exergy_rate(1.0, 0.0, 350.0, 320.0, 300.0) == 0.0


def test_finite_stream_rejects_heating_stream_sign_confusion():
    with pytest.raises(DomainError):
        finite_stream_heat_exergy_rate(1.0, 4_180.0, 320.0, 350.0, 300.0)


def test_finite_stream_rejects_cold_crossing_before_log_can_hide_it():
    with pytest.raises(ColdThermalDomainError):
        finite_stream_heat_exergy_rate(1.0, 4_180.0, 330.0, 290.0, 300.0)


def test_cooling_exergy_increases_as_cold_temperature_decreases():
    t0 = 300.0
    cooling_rate = 1_000.0
    warm_to_cold = [295.0, 280.0, 260.0, 240.0]
    values = [cooling_service_exergy_rate(cooling_rate, tc, t0) for tc in warm_to_cold]

    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert all(value > 0.0 for value in values)


def test_cooling_zero_load_returns_zero_across_valid_temperatures():
    for cold_temperature in [250.0, 280.0, 300.0, 320.0]:
        assert cooling_service_exergy_rate(0.0, cold_temperature, 300.0) == 0.0


def test_electrical_service_derating_is_never_above_real_power_over_grid():
    real_power = 1_000.0
    factors = [0.0, 0.25, 0.5, 0.75, 1.0]
    for voltage_quality in factors:
        for frequency_quality in factors:
            for harmonic_quality in factors:
                for availability in factors:
                    result = electrical_service_exergy_rate(
                        real_power,
                        voltage_quality=voltage_quality,
                        frequency_quality=frequency_quality,
                        harmonic_quality=harmonic_quality,
                        availability=availability,
                    )
                    assert 0.0 <= result <= real_power


def test_thermal_storage_can_report_hot_only_or_hot_plus_cold_without_averaging():
    layers = [
        ThermalLayer(temperature_k=340.0, energy_j=1_000.0),
        ThermalLayer(temperature_k=260.0, energy_j=1_000.0),
    ]
    hot_only = thermal_storage_exergy(layers, 300.0, include_cold_exergy=False)
    hot_plus_cold = thermal_storage_exergy(layers, 300.0, include_cold_exergy=True)

    assert hot_only == pytest.approx(heat_exergy_rate(1_000.0, 340.0, 300.0))
    assert hot_plus_cold > hot_only


def test_thermal_storage_exergy_is_order_independent_for_layers():
    layers = [
        ThermalLayer(temperature_k=360.0, energy_j=500.0),
        ThermalLayer(temperature_k=330.0, energy_j=800.0),
        ThermalLayer(temperature_k=305.0, energy_j=300.0),
    ]
    assert thermal_storage_exergy(layers, 300.0) == pytest.approx(
        thermal_storage_exergy(list(reversed(layers)), 300.0)
    )


def test_quality_factor_is_carrier_specific_not_universal_x_over_e():
    validate_quality_factor(1.2, Carrier.THERMAL)
    assert classify_quality(1.2, Carrier.THERMAL) == QualityGrade.A

    with pytest.raises(DomainError):
        validate_quality_factor(1.2, Carrier.ELECTRIC)


def test_quality_classifier_boundaries_are_stable():
    assert classify_quality(0.95, Carrier.ELECTRIC) == QualityGrade.A
    assert classify_quality(0.80, Carrier.ELECTRIC) == QualityGrade.B
    assert classify_quality(0.55, Carrier.ELECTRIC) == QualityGrade.C
    assert classify_quality(0.25, Carrier.ELECTRIC) == QualityGrade.D
    assert classify_quality(0.249, Carrier.ELECTRIC) == QualityGrade.E


def test_quality_factor_rejects_nonfinite_and_negative_values():
    for bad in [math.nan, math.inf, -0.01]:
        with pytest.raises(DomainError):
            validate_quality_factor(bad, Carrier.THERMAL)

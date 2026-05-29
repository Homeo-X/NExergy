from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eie.exergy.cooling import cooling_service_exergy_rate
from eie.exergy.electrical import electrical_service_exergy_rate
from eie.exergy.heat import finite_stream_heat_exergy_rate, heat_exergy_rate
from eie.exergy.kernel import energy_balance_residual, exergy_balance_residual
from eie.exergy.storage import battery_stored_exergy, thermal_storage_exergy
from eie.flows.storage import ThermalLayer


finite_positive = st.floats(
    min_value=1.0e-6,
    max_value=1.0e9,
    allow_nan=False,
    allow_infinity=False,
)
probability = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@settings(deadline=None, max_examples=150)
@given(
    heat_rate=finite_positive,
    reference_temperature=st.floats(min_value=250.0, max_value=330.0, allow_nan=False, allow_infinity=False),
    temperature_ratio=st.floats(min_value=1.000001, max_value=8.0, allow_nan=False, allow_infinity=False),
)
def test_hot_heat_exergy_property_bounds(heat_rate, reference_temperature, temperature_ratio):
    source_temperature = reference_temperature * temperature_ratio
    exergy = heat_exergy_rate(heat_rate, source_temperature, reference_temperature)

    assert 0.0 <= exergy < heat_rate


@settings(deadline=None, max_examples=150)
@given(
    cooling_rate=finite_positive,
    reference_temperature=st.floats(min_value=250.0, max_value=330.0, allow_nan=False, allow_infinity=False),
    cold_fraction=st.floats(min_value=0.2, max_value=0.999999, allow_nan=False, allow_infinity=False),
)
def test_cooling_exergy_property_is_positive_below_reference(
    cooling_rate,
    reference_temperature,
    cold_fraction,
):
    cold_temperature = reference_temperature * cold_fraction
    exergy = cooling_service_exergy_rate(cooling_rate, cold_temperature, reference_temperature)

    assert exergy > 0.0
    assert exergy == pytest.approx(cooling_rate * (reference_temperature / cold_temperature - 1.0))


@settings(deadline=None, max_examples=150)
@given(
    real_power=finite_positive,
    voltage_quality=probability,
    frequency_quality=probability,
    harmonic_quality=probability,
    availability=probability,
)
def test_electrical_service_derating_property_never_increases_power(
    real_power,
    voltage_quality,
    frequency_quality,
    harmonic_quality,
    availability,
):
    service = electrical_service_exergy_rate(
        real_power,
        voltage_quality=voltage_quality,
        frequency_quality=frequency_quality,
        harmonic_quality=harmonic_quality,
        availability=availability,
    )

    assert 0.0 <= service <= real_power


@settings(deadline=None, max_examples=150)
@given(
    mass_flow=st.floats(min_value=1.0e-6, max_value=100.0, allow_nan=False, allow_infinity=False),
    cp=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    reference_temperature=st.floats(min_value=250.0, max_value=330.0, allow_nan=False, allow_infinity=False),
    hot_lift=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    cooling_drop=st.floats(min_value=1.0e-6, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_finite_stream_exergy_property_bounded_by_sensible_heat(
    mass_flow,
    cp,
    reference_temperature,
    hot_lift,
    cooling_drop,
):
    t_out = reference_temperature + hot_lift
    t_in = t_out + cooling_drop
    exergy = finite_stream_heat_exergy_rate(mass_flow, cp, t_in, t_out, reference_temperature)
    sensible = mass_flow * cp * (t_in - t_out)

    assert 0.0 <= exergy < sensible
    assert math.isfinite(exergy)


@settings(deadline=None, max_examples=100)
@given(
    stored_energy=finite_positive,
    availability=probability,
)
def test_battery_stored_exergy_property_bounds(stored_energy, availability):
    exergy = battery_stored_exergy(stored_energy, availability)

    assert 0.0 <= exergy <= stored_energy


@settings(deadline=None, max_examples=100)
@given(
    layer_one_energy=st.floats(min_value=0.0, max_value=1.0e8, allow_nan=False, allow_infinity=False),
    layer_two_energy=st.floats(min_value=0.0, max_value=1.0e8, allow_nan=False, allow_infinity=False),
    layer_one_temperature=st.floats(min_value=301.0, max_value=700.0, allow_nan=False, allow_infinity=False),
    layer_two_temperature=st.floats(min_value=301.0, max_value=700.0, allow_nan=False, allow_infinity=False),
)
def test_thermal_storage_property_order_independent(
    layer_one_energy,
    layer_two_energy,
    layer_one_temperature,
    layer_two_temperature,
):
    layers = [
        ThermalLayer(temperature_k=layer_one_temperature, energy_j=layer_one_energy),
        ThermalLayer(temperature_k=layer_two_temperature, energy_j=layer_two_energy),
    ]

    assert thermal_storage_exergy(layers, 300.0) == pytest.approx(
        thermal_storage_exergy(list(reversed(layers)), 300.0)
    )


@settings(deadline=None, max_examples=100)
@given(
    energy_in=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    energy_out=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    stored_delta=st.floats(min_value=-1.0e9, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    rejected=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    known_losses=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
)
def test_energy_balance_residual_property_matches_definition(
    energy_in,
    energy_out,
    stored_delta,
    rejected,
    known_losses,
):
    assert energy_balance_residual(energy_in, energy_out, stored_delta, rejected, known_losses) == pytest.approx(
        energy_in - energy_out - stored_delta - rejected - known_losses
    )


@settings(deadline=None, max_examples=100)
@given(
    exergy_in=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    useful=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    stored_delta=st.floats(min_value=-1.0e9, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    recovered=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    rejected=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
    destroyed=st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
)
def test_exergy_balance_residual_property_matches_definition(
    exergy_in,
    useful,
    stored_delta,
    recovered,
    rejected,
    destroyed,
):
    assert exergy_balance_residual(
        exergy_in,
        useful,
        stored_delta,
        recovered,
        rejected,
        destroyed,
    ) == pytest.approx(exergy_in - useful - stored_delta - recovered - rejected - destroyed)

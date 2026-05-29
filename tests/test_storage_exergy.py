from __future__ import annotations

import pytest

from eie.exergy.heat import heat_exergy_rate
from eie.exergy.storage import battery_stored_exergy, thermal_storage_exergy
from eie.flows.storage import BatteryState, ThermalLayer


def test_battery_stored_exergy_uses_availability_factor():
    assert battery_stored_exergy(1_000.0, availability_factor=0.8) == pytest.approx(800.0)


def test_battery_state_represents_reserve_energy(boundary, reference, energy_metadata):
    state = BatteryState(
        storage_id="battery",
        stored_energy_j=10_000.0,
        soc=0.5,
        soh=0.9,
        reserve_energy_j=2_000.0,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
        metadata=energy_metadata,
    )
    assert state.reserve_energy_j == 2_000.0


def test_thermal_storage_integrates_stratified_layers_separately():
    layers = [
        ThermalLayer(temperature_k=350.0, energy_j=1_000.0),
        ThermalLayer(temperature_k=310.0, energy_j=1_000.0),
    ]
    result = thermal_storage_exergy(layers, 300.0)
    separate = heat_exergy_rate(1_000.0, 350.0, 300.0) + heat_exergy_rate(1_000.0, 310.0, 300.0)
    average_temperature_shortcut = heat_exergy_rate(2_000.0, 330.0, 300.0)
    assert result == pytest.approx(separate)
    assert result != pytest.approx(average_temperature_shortcut)

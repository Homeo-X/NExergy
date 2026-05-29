from __future__ import annotations

import pytest

from eie.core.errors import UnitError
from eie.core.units import (
    CARBON_INTENSITY,
    DIMENSIONLESS,
    ENERGY,
    MASS_FLOW,
    POWER,
    PRICE_PER_ENERGY,
    SPECIFIC_HEAT,
    Dimension,
    Quantity,
    compatible_units,
    convert_value,
    dimension_for_unit,
    quantity,
    require_dimension,
    require_same_dimension,
    scale_to_si,
    unit_definition,
)
from eie.flows.storage import BatteryState


def test_registered_units_have_expected_dimensions():
    assert dimension_for_unit("W") == POWER
    assert dimension_for_unit("kW") == POWER
    assert dimension_for_unit("J") == ENERGY
    assert dimension_for_unit("kWh") == ENERGY
    assert dimension_for_unit("kg/s") == MASS_FLOW
    assert dimension_for_unit("J/(kg*K)") == SPECIFIC_HEAT
    assert dimension_for_unit("kg/kWh") == CARBON_INTENSITY
    assert dimension_for_unit("currency/kWh") == PRICE_PER_ENERGY
    assert dimension_for_unit("dimensionless") == DIMENSIONLESS


def test_dimension_algebra_is_explicit_and_stable():
    assert POWER * Dimension(time=1) == ENERGY
    assert ENERGY / Dimension(time=1) == POWER
    assert (Dimension(length=1) ** 2).label() == "L^2"
    assert DIMENSIONLESS.is_dimensionless


def test_scale_only_conversion_between_compatible_units():
    assert convert_value(1.0, "kW", "W") == pytest.approx(1_000.0)
    assert convert_value(3_600_000.0, "J", "kWh") == pytest.approx(1.0)
    assert convert_value(2.0, "h", "s") == pytest.approx(7_200.0)
    assert scale_to_si("bar") == pytest.approx(100_000.0)


def test_conversion_rejects_incompatible_units():
    with pytest.raises(UnitError):
        convert_value(1.0, "W", "J")
    with pytest.raises(UnitError):
        require_same_dimension("Pa", "V")


def test_unknown_units_fail_before_dimension_checks():
    with pytest.raises(UnitError):
        unit_definition("BTU/hr")
    with pytest.raises(UnitError):
        require_dimension("BTU/hr", POWER)


def test_quantity_preserves_dimension_and_converts():
    q = quantity(5.0, "kW")
    converted = q.to("W")

    assert isinstance(q, Quantity)
    assert q.dimension == POWER
    assert converted.value == pytest.approx(5_000.0)
    assert converted.unit == "W"


def test_quantity_rejects_nonfinite_values():
    with pytest.raises(UnitError):
        quantity(float("nan"), "W")


def test_compatible_units_uses_dimensions_not_symbol_strings():
    assert compatible_units("kW", "W")
    assert compatible_units("kWh", "J")
    assert not compatible_units("kW", "kWh")


def test_storage_schema_rejects_power_metadata_for_energy_state(boundary, reference, metadata):
    with pytest.raises(UnitError):
        BatteryState(
            storage_id="bad-unit-battery",
            stored_energy_j=1_000.0,
            soc=0.5,
            soh=0.9,
            reserve_energy_j=100.0,
            boundary_id=boundary.boundary_id,
            reference_state_id=reference.reference_state_id,
            metadata=metadata,
        )

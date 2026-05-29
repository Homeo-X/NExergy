from __future__ import annotations

import pytest

from eie.core.errors import DomainError
from eie.exergy.electrical import electrical_exergy_rate, electrical_service_exergy_rate


def test_real_power_maps_to_electrical_exergy():
    assert electrical_exergy_rate(1_234.0) == pytest.approx(1_234.0)


def test_electrical_service_derating_reduces_service_value():
    result = electrical_service_exergy_rate(
        1_000.0,
        voltage_quality=0.95,
        frequency_quality=0.90,
        harmonic_quality=0.80,
        availability=0.50,
    )
    assert result == pytest.approx(342.0)


def test_electrical_derating_factors_must_be_probability():
    with pytest.raises(DomainError):
        electrical_service_exergy_rate(1_000.0, voltage_quality=1.2)

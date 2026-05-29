from __future__ import annotations

import math

import pytest

from eie.core.errors import DomainError
from eie.exergy.heat import ColdThermalDomainError, finite_stream_heat_exergy_rate, heat_exergy_rate


def test_heat_exergy_zero_at_reference_temperature():
    assert heat_exergy_rate(1_000.0, 300.0, 300.0) == pytest.approx(0.0)


def test_heat_exergy_positive_above_reference_temperature():
    assert heat_exergy_rate(1_000.0, 600.0, 300.0) == pytest.approx(500.0)


def test_heat_exergy_rejects_invalid_temperature():
    with pytest.raises(DomainError):
        heat_exergy_rate(1_000.0, -1.0, 300.0)


def test_hot_heat_function_rejects_below_ambient_source():
    with pytest.raises(ColdThermalDomainError):
        heat_exergy_rate(1_000.0, 280.0, 300.0)


def test_finite_stream_heat_exergy_matches_constant_cp_formula():
    result = finite_stream_heat_exergy_rate(2.0, 4_000.0, 500.0, 400.0, 300.0)
    expected = 2.0 * 4_000.0 * ((500.0 - 400.0) - 300.0 * math.log(500.0 / 400.0))
    assert result == pytest.approx(expected)


def test_finite_stream_heat_exergy_rejects_crossing_below_reference():
    with pytest.raises(ColdThermalDomainError):
        finite_stream_heat_exergy_rate(1.0, 4_000.0, 310.0, 290.0, 300.0)

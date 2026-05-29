from __future__ import annotations

import pytest

from eie.exergy.cooling import cooling_service_exergy_rate


def test_cooling_exergy_positive_below_ambient():
    assert cooling_service_exergy_rate(1_000.0, 250.0, 300.0) == pytest.approx(200.0)


def test_cooling_exergy_zero_when_no_refrigeration_needed():
    assert cooling_service_exergy_rate(1_000.0, 300.0, 300.0) == 0.0
    assert cooling_service_exergy_rate(1_000.0, 310.0, 300.0) == 0.0

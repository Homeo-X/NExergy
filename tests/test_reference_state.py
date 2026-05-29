from __future__ import annotations

from datetime import timedelta

import pytest

from eie.core.errors import DomainError
from eie.guards.reference_guard import ReferenceGuard
from eie.reference.state import ReferenceState


def test_reference_state_requires_positive_temperature(now):
    with pytest.raises(DomainError):
        ReferenceState(
            reference_state_id="bad",
            timestamp=now,
            ambient_temperature_k=0.0,
            ambient_pressure_pa=101_325.0,
            confidence=1.0,
        )


def test_reference_state_confidence_must_be_probability(now):
    with pytest.raises(DomainError):
        ReferenceState(
            reference_state_id="bad",
            timestamp=now,
            ambient_temperature_k=300.0,
            ambient_pressure_pa=101_325.0,
            confidence=1.1,
        )


def test_reference_guard_detects_stale_state(now):
    state = ReferenceState(
        reference_state_id="stale",
        timestamp=now - timedelta(hours=2),
        ambient_temperature_k=300.0,
        ambient_pressure_pa=101_325.0,
        confidence=0.9,
    )
    result = ReferenceGuard().check(state, at=now)
    assert not result.passed
    assert "stale" in result.reason


def test_reference_guard_accepts_fresh_state(reference, now):
    result = ReferenceGuard().check(reference, at=now)
    assert result.passed

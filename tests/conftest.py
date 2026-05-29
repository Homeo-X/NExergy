from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eie.boundary.boundary import Boundary
from eie.core.enums import BoundaryType, MeasurementMethod
from eie.exergy.kernel import ExergyKernelV0
from eie.flows.base import Metadata
from eie.reference.state import ReferenceState


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def reference(now: datetime) -> ReferenceState:
    return ReferenceState(
        reference_state_id="ref-test",
        timestamp=now,
        ambient_temperature_k=300.0,
        ambient_pressure_pa=101_325.0,
        confidence=0.99,
        valid_until=now + timedelta(minutes=30),
    )


@pytest.fixture
def boundary(reference: ReferenceState, now: datetime) -> Boundary:
    return Boundary(
        boundary_id="boundary-test",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=["asset-a", "asset-b"],
        excluded_entity_ids=[],
        reference_state_id=reference.reference_state_id,
        accounting_period_start=now,
        accounting_period_end=now + timedelta(hours=1),
    )


@pytest.fixture
def metadata(reference: ReferenceState, boundary: Boundary, now: datetime) -> Metadata:
    return Metadata(
        timestamp=now,
        source="pytest",
        method=MeasurementMethod.SIMULATED,
        unit="W",
        confidence=1.0,
        uncertainty=0.0,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
    )


@pytest.fixture
def energy_metadata(reference: ReferenceState, boundary: Boundary, now: datetime) -> Metadata:
    return Metadata(
        timestamp=now,
        source="pytest",
        method=MeasurementMethod.SIMULATED,
        unit="J",
        confidence=1.0,
        uncertainty=0.0,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
    )


@pytest.fixture
def mass_flow_metadata(reference: ReferenceState, boundary: Boundary, now: datetime) -> Metadata:
    return Metadata(
        timestamp=now,
        source="pytest",
        method=MeasurementMethod.SIMULATED,
        unit="kg/s",
        confidence=1.0,
        uncertainty=0.0,
        boundary_id=boundary.boundary_id,
        reference_state_id=reference.reference_state_id,
    )


@pytest.fixture
def kernel(reference: ReferenceState, boundary: Boundary) -> ExergyKernelV0:
    return ExergyKernelV0(reference_state=reference, boundary=boundary)

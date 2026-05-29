from __future__ import annotations

import pytest

from eie.boundary.boundary import Boundary
from eie.core.enums import BoundaryType
from eie.core.errors import BoundaryError, MissingReferenceError
from eie.exergy.kernel import ExergyKernelV0
from eie.flows.electrical import ElectricalFlow


def test_boundary_requires_reference_state_id(now):
    with pytest.raises(BoundaryError):
        Boundary(
            boundary_id="bad",
            boundary_type=BoundaryType.SITE,
            reference_state_id="",
            accounting_period_start=now,
            accounting_period_end=now,
        )


def test_kernel_rejects_boundary_reference_mismatch(reference, boundary):
    wrong_boundary = Boundary(
        boundary_id=boundary.boundary_id,
        boundary_type=BoundaryType.SITE,
        included_entity_ids=[],
        excluded_entity_ids=[],
        reference_state_id="other-reference",
    )
    with pytest.raises(BoundaryError):
        ExergyKernelV0(reference_state=reference, boundary=wrong_boundary)


def test_kernel_rejects_flow_reference_mismatch(kernel, boundary, metadata):
    flow = ElectricalFlow(
        flow_id="bad-flow",
        real_power_w=100.0,
        voltage_v=None,
        frequency_hz=None,
        power_factor=None,
        thd=None,
        availability_factor=None,
        boundary_id=boundary.boundary_id,
        reference_state_id="other-reference",
        metadata=metadata,
    )
    with pytest.raises(MissingReferenceError):
        kernel.electrical_flow(flow)

"""Boundary guard."""

from __future__ import annotations

from eie.boundary.boundary import Boundary
from eie.core.enums import Severity
from eie.guards.physics_guard import GuardResult


class BoundaryGuard:
    """Require explicit, reference-bound accounting boundary."""

    guard_name = "BoundaryGuard"

    def check(
        self,
        boundary: Boundary | None,
        *,
        expected_reference_state_id: str | None = None,
    ) -> GuardResult:
        if boundary is None:
            return GuardResult(self.guard_name, False, Severity.CRITICAL, "missing boundary", {})
        if not boundary.boundary_id:
            return GuardResult(self.guard_name, False, Severity.CRITICAL, "boundary_id is empty", {})
        if not boundary.reference_state_id:
            return GuardResult(
                self.guard_name,
                False,
                Severity.CRITICAL,
                "boundary is not bound to a reference state",
                {"boundary_id": boundary.boundary_id},
            )
        if expected_reference_state_id is not None and boundary.reference_state_id != expected_reference_state_id:
            return GuardResult(
                self.guard_name,
                False,
                Severity.ERROR,
                "boundary reference does not match expected reference",
                {
                    "boundary_id": boundary.boundary_id,
                    "boundary_reference_state_id": boundary.reference_state_id,
                    "expected_reference_state_id": expected_reference_state_id,
                },
            )
        return GuardResult(
            self.guard_name,
            True,
            Severity.INFO,
            "boundary is explicit and reference-bound",
            {"boundary_id": boundary.boundary_id, "reference_state_id": boundary.reference_state_id},
        )

    def check_binding(
        self,
        *,
        object_boundary_id: str | None,
        object_reference_state_id: str | None,
        boundary: Boundary,
    ) -> GuardResult:
        if object_boundary_id != boundary.boundary_id:
            return GuardResult(
                self.guard_name,
                False,
                Severity.ERROR,
                "object boundary does not match accounting boundary",
                {"object_boundary_id": object_boundary_id, "boundary_id": boundary.boundary_id},
            )
        if object_reference_state_id != boundary.reference_state_id:
            return GuardResult(
                self.guard_name,
                False,
                Severity.ERROR,
                "object reference does not match boundary reference",
                {
                    "object_reference_state_id": object_reference_state_id,
                    "boundary_reference_state_id": boundary.reference_state_id,
                },
            )
        return GuardResult(
            self.guard_name,
            True,
            Severity.INFO,
            "object binding matches boundary",
            {"boundary_id": boundary.boundary_id, "reference_state_id": boundary.reference_state_id},
        )

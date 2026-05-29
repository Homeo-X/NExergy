"""Reference-state guard."""

from __future__ import annotations

from datetime import datetime

from eie.core.enums import Severity
from eie.core.tolerance import DEFAULT_TOLERANCE, Tolerance
from eie.guards.physics_guard import GuardResult
from eie.reference.state import ReferenceState


class ReferenceGuard:
    """Require present, confident, and fresh reference state."""

    guard_name = "ReferenceGuard"

    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        tolerance: Tolerance = DEFAULT_TOLERANCE,
    ) -> None:
        self.min_confidence = min_confidence
        self.tolerance = tolerance

    def check(self, reference_state: ReferenceState | None, *, at: datetime | None = None) -> GuardResult:
        if reference_state is None:
            return GuardResult(
                self.guard_name,
                False,
                Severity.CRITICAL,
                "missing reference state",
                {},
            )
        if reference_state.confidence < self.min_confidence:
            return GuardResult(
                self.guard_name,
                False,
                Severity.ERROR,
                "reference confidence below minimum",
                {
                    "reference_state_id": reference_state.reference_state_id,
                    "confidence": reference_state.confidence,
                    "min_confidence": self.min_confidence,
                },
            )
        if reference_state.is_stale(at=at, tolerance=self.tolerance):
            return GuardResult(
                self.guard_name,
                False,
                Severity.ERROR,
                "reference state is stale",
                {"reference_state_id": reference_state.reference_state_id},
            )
        return GuardResult(
            self.guard_name,
            True,
            Severity.INFO,
            "reference state is valid for v0 accounting",
            {"reference_state_id": reference_state.reference_state_id},
        )

"""Reference-state validation helpers."""

from __future__ import annotations

from datetime import datetime

from eie.core.errors import MissingReferenceError, StaleReferenceError
from eie.core.tolerance import DEFAULT_TOLERANCE, Tolerance
from eie.reference.state import ReferenceState


def require_reference_state(reference_state: ReferenceState | None) -> ReferenceState:
    if reference_state is None:
        raise MissingReferenceError("reference_state is required for exergy accounting")
    return reference_state


def require_fresh_reference_state(
    reference_state: ReferenceState | None,
    *,
    at: datetime | None = None,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> ReferenceState:
    reference_state = require_reference_state(reference_state)
    if reference_state.is_stale(at=at, tolerance=tolerance):
        raise StaleReferenceError(f"reference state {reference_state.reference_state_id!r} is stale")
    return reference_state


def require_reference_state_id(reference_state_id: str | None) -> str:
    if not reference_state_id:
        raise MissingReferenceError("reference_state_id is required")
    return reference_state_id

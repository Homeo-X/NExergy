"""Boundary registry and binding checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from eie.boundary.boundary import Boundary
from eie.core.errors import BoundaryError


@dataclass
class BoundaryManager:
    """In-memory registry for v0 boundary definitions."""

    _boundaries: dict[str, Boundary] = field(default_factory=dict)

    def add(self, boundary: Boundary) -> None:
        if boundary.boundary_id in self._boundaries:
            raise BoundaryError(f"boundary already exists: {boundary.boundary_id!r}")
        self._boundaries[boundary.boundary_id] = boundary

    def get(self, boundary_id: str) -> Boundary:
        try:
            return self._boundaries[boundary_id]
        except KeyError as exc:
            raise BoundaryError(f"unknown boundary: {boundary_id!r}") from exc

    def has(self, boundary_id: str) -> bool:
        return boundary_id in self._boundaries

    def require_reference_match(self, boundary_id: str, reference_state_id: str) -> Boundary:
        boundary = self.get(boundary_id)
        if boundary.reference_state_id != reference_state_id:
            raise BoundaryError(
                f"boundary {boundary_id!r} is bound to reference {boundary.reference_state_id!r}, "
                f"not {reference_state_id!r}"
            )
        return boundary

    def all(self) -> list[Boundary]:
        return list(self._boundaries.values())

"""Explicit accounting boundary model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from eie.core.enums import BoundaryType
from eie.core.errors import BoundaryError


@dataclass(frozen=True)
class Boundary:
    """System boundary for energy, exergy, entropy, and ledger accounting."""

    boundary_id: str
    boundary_type: BoundaryType | str
    included_entity_ids: list[str] = field(default_factory=list)
    excluded_entity_ids: list[str] = field(default_factory=list)
    reference_state_id: str = ""
    accounting_period_start: datetime | None = None
    accounting_period_end: datetime | None = None

    def __post_init__(self) -> None:
        if not self.boundary_id:
            raise BoundaryError("boundary_id is required")
        if not isinstance(self.boundary_type, BoundaryType):
            object.__setattr__(self, "boundary_type", BoundaryType(self.boundary_type))
        if not self.reference_state_id:
            raise BoundaryError("boundary.reference_state_id is required")
        overlap = set(self.included_entity_ids).intersection(self.excluded_entity_ids)
        if overlap:
            raise BoundaryError(f"entities cannot be both included and excluded: {sorted(overlap)}")
        if (
            self.accounting_period_start is not None
            and self.accounting_period_end is not None
            and self.accounting_period_end < self.accounting_period_start
        ):
            raise BoundaryError("accounting_period_end must be at or after accounting_period_start")

    def contains(self, entity_id: str) -> bool:
        if entity_id in self.excluded_entity_ids:
            return False
        return not self.included_entity_ids or entity_id in self.included_entity_ids

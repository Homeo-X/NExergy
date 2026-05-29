"""Energy, exergy, entropy, and loss ledger records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isfinite

from eie.core.errors import BoundaryError, DomainError, MissingReferenceError
from eie.flows.base import require_probability


def _finite_number(value: float, field: str) -> None:
    if not isfinite(value):
        raise DomainError(f"{field} must be finite")


@dataclass(frozen=True)
class LedgerEntry:
    """Append-only accounting fact for a boundary and reference state."""

    ledger_id: str
    timestamp: datetime
    boundary_id: str
    reference_state_id: str
    energy_in_j: float
    energy_out_j: float
    energy_stored_delta_j: float
    energy_rejected_j: float
    energy_residual_j: float
    exergy_in_j: float
    useful_exergy_j: float
    stored_exergy_delta_j: float
    recovered_exergy_j: float
    rejected_exergy_j: float
    destroyed_exergy_j: float
    exergy_residual_j: float
    entropy_generated_j_per_k: float
    flags: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.ledger_id:
            raise DomainError("ledger_id is required")
        if not isinstance(self.timestamp, datetime):
            raise DomainError("timestamp must be a datetime")
        if not self.boundary_id:
            raise BoundaryError("ledger boundary_id is required")
        if not self.reference_state_id:
            raise MissingReferenceError("ledger reference_state_id is required")
        for field_name in (
            "energy_in_j",
            "energy_out_j",
            "energy_stored_delta_j",
            "energy_rejected_j",
            "energy_residual_j",
            "exergy_in_j",
            "useful_exergy_j",
            "stored_exergy_delta_j",
            "recovered_exergy_j",
            "rejected_exergy_j",
            "destroyed_exergy_j",
            "exergy_residual_j",
            "entropy_generated_j_per_k",
        ):
            _finite_number(getattr(self, field_name), field_name)
        require_probability(self.confidence, "confidence")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AuditLedger:
    """Append-only in-memory ledger for v0.

    Corrections are new entries. Existing entries are never mutated or removed.
    """

    entries: list[LedgerEntry] = field(default_factory=list)

    def append(self, entry: LedgerEntry) -> None:
        if any(existing.ledger_id == entry.ledger_id for existing in self.entries):
            raise DomainError(f"duplicate ledger_id: {entry.ledger_id!r}")
        self.entries.append(entry)

    def correction(self, original_ledger_id: str, correction_entry: LedgerEntry) -> None:
        if not any(existing.ledger_id == original_ledger_id for existing in self.entries):
            raise DomainError(f"cannot correct unknown ledger_id: {original_ledger_id!r}")
        flags = list(correction_entry.flags)
        marker = f"correction_of:{original_ledger_id}"
        if marker not in flags:
            flags.append(marker)
        corrected = LedgerEntry(
            ledger_id=correction_entry.ledger_id,
            timestamp=correction_entry.timestamp,
            boundary_id=correction_entry.boundary_id,
            reference_state_id=correction_entry.reference_state_id,
            energy_in_j=correction_entry.energy_in_j,
            energy_out_j=correction_entry.energy_out_j,
            energy_stored_delta_j=correction_entry.energy_stored_delta_j,
            energy_rejected_j=correction_entry.energy_rejected_j,
            energy_residual_j=correction_entry.energy_residual_j,
            exergy_in_j=correction_entry.exergy_in_j,
            useful_exergy_j=correction_entry.useful_exergy_j,
            stored_exergy_delta_j=correction_entry.stored_exergy_delta_j,
            recovered_exergy_j=correction_entry.recovered_exergy_j,
            rejected_exergy_j=correction_entry.rejected_exergy_j,
            destroyed_exergy_j=correction_entry.destroyed_exergy_j,
            exergy_residual_j=correction_entry.exergy_residual_j,
            entropy_generated_j_per_k=correction_entry.entropy_generated_j_per_k,
            flags=flags,
            confidence=correction_entry.confidence,
        )
        self.append(corrected)

    def by_boundary(self, boundary_id: str) -> list[LedgerEntry]:
        return [entry for entry in self.entries if entry.boundary_id == boundary_id]

    def as_dicts(self) -> list[dict[str, object]]:
        return [entry.as_dict() for entry in self.entries]

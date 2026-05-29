"""Physics guard for residuals, entropy signs, and destroyed exergy."""

from __future__ import annotations

from dataclasses import dataclass, field

from eie.core.enums import Severity
from eie.core.tolerance import DEFAULT_TOLERANCE, Tolerance
from eie.ledger.audit import audit_ledger_entry
from eie.ledger.entries import LedgerEntry


@dataclass(frozen=True)
class GuardResult:
    guard_name: str
    passed: bool
    severity: Severity
    reason: str
    evidence: dict[str, object] = field(default_factory=dict)


class PhysicsGuard:
    """Validate first-law, second-law, and exergy accounting invariants."""

    guard_name = "PhysicsGuard"

    def __init__(self, tolerance: Tolerance = DEFAULT_TOLERANCE) -> None:
        self.tolerance = tolerance

    def check_ledger_entry(self, entry: LedgerEntry) -> GuardResult:
        flags = audit_ledger_entry(entry, tolerance=self.tolerance)
        critical_flags = {
            "negative_destroyed_exergy",
            "negative_entropy_generation",
            "energy_residual_mismatch",
            "exergy_residual_mismatch",
        }
        hard_failures = sorted(set(flags).intersection(critical_flags))
        if hard_failures:
            return GuardResult(
                guard_name=self.guard_name,
                passed=False,
                severity=Severity.CRITICAL,
                reason="ledger violates physics accounting invariants",
                evidence={"ledger_id": entry.ledger_id, "flags": flags, "hard_failures": hard_failures},
            )
        balance_flags = [flag for flag in flags if flag.endswith("_high")]
        if balance_flags:
            return GuardResult(
                guard_name=self.guard_name,
                passed=False,
                severity=Severity.ERROR,
                reason="ledger residual exceeds tolerance",
                evidence={"ledger_id": entry.ledger_id, "flags": flags},
            )
        return GuardResult(
            guard_name=self.guard_name,
            passed=True,
            severity=Severity.INFO,
            reason="ledger physics checks passed",
            evidence={"ledger_id": entry.ledger_id, "flags": flags},
        )

    def check_ledger(self, entries: list[LedgerEntry]) -> list[GuardResult]:
        return [self.check_ledger_entry(entry) for entry in entries]

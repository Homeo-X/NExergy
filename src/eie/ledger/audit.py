"""Ledger audit helpers."""

from __future__ import annotations

from eie.core.tolerance import DEFAULT_TOLERANCE, Tolerance
from eie.exergy.kernel import energy_balance_residual, exergy_balance_residual
from eie.ledger.entries import LedgerEntry


def audit_ledger_entry(
    entry: LedgerEntry,
    *,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> list[str]:
    """Return audit flags for ledger residuals and sign-sensitive values."""

    flags = list(entry.flags)
    energy_scale = max(abs(entry.energy_in_j), abs(entry.energy_out_j), abs(entry.energy_rejected_j), 1.0)
    exergy_scale = max(abs(entry.exergy_in_j), abs(entry.useful_exergy_j), abs(entry.destroyed_exergy_j), 1.0)
    recomputed_energy = energy_balance_residual(
        entry.energy_in_j,
        entry.energy_out_j,
        entry.energy_stored_delta_j,
        entry.energy_rejected_j,
    )
    recomputed_exergy = exergy_balance_residual(
        entry.exergy_in_j,
        entry.useful_exergy_j,
        entry.stored_exergy_delta_j,
        entry.recovered_exergy_j,
        entry.rejected_exergy_j,
        entry.destroyed_exergy_j,
    )
    if abs(recomputed_energy - entry.energy_residual_j) > tolerance.residual_limit(energy_scale):
        flags.append("energy_residual_mismatch")
    if abs(entry.energy_residual_j) > tolerance.residual_limit(energy_scale):
        flags.append("energy_balance_residual_high")
    if abs(recomputed_exergy - entry.exergy_residual_j) > tolerance.residual_limit(exergy_scale):
        flags.append("exergy_residual_mismatch")
    if abs(entry.exergy_residual_j) > tolerance.residual_limit(exergy_scale):
        flags.append("exergy_balance_residual_high")
    if entry.destroyed_exergy_j < -tolerance.absolute_j:
        flags.append("negative_destroyed_exergy")
    if entry.entropy_generated_j_per_k < -tolerance.entropy_j_per_k:
        flags.append("negative_entropy_generation")
    return flags

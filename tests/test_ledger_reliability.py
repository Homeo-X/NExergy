from __future__ import annotations

import pytest

from eie.exergy.kernel import energy_balance_residual, exergy_balance_residual
from eie.ledger.audit import audit_ledger_entry
from eie.ledger.entries import AuditLedger, LedgerEntry


def _entry(now, boundary, ledger_id="entry", **overrides):
    values = {
        "ledger_id": ledger_id,
        "timestamp": now,
        "boundary_id": boundary.boundary_id,
        "reference_state_id": boundary.reference_state_id,
        "energy_in_j": 100.0,
        "energy_out_j": 60.0,
        "energy_stored_delta_j": 10.0,
        "energy_rejected_j": 30.0,
        "energy_residual_j": energy_balance_residual(100.0, 60.0, 10.0, 30.0),
        "exergy_in_j": 80.0,
        "useful_exergy_j": 40.0,
        "stored_exergy_delta_j": 10.0,
        "recovered_exergy_j": 5.0,
        "rejected_exergy_j": 10.0,
        "destroyed_exergy_j": 15.0,
        "exergy_residual_j": exergy_balance_residual(80.0, 40.0, 10.0, 5.0, 10.0, 15.0),
        "entropy_generated_j_per_k": 0.05,
        "flags": [],
        "confidence": 1.0,
    }
    values.update(overrides)
    return LedgerEntry(**values)


def test_ledger_correction_appends_new_entry_and_marks_original(now, boundary):
    ledger = AuditLedger()
    original = _entry(now, boundary, ledger_id="original")
    correction = _entry(now, boundary, ledger_id="correction", energy_in_j=110.0)

    ledger.append(original)
    ledger.correction(original.ledger_id, correction)

    assert [entry.ledger_id for entry in ledger.entries] == ["original", "correction"]
    assert ledger.entries[0].flags == []
    assert "correction_of:original" in ledger.entries[1].flags


def test_ledger_rejects_correction_for_unknown_original(now, boundary):
    ledger = AuditLedger()
    with pytest.raises(Exception):
        ledger.correction("missing", _entry(now, boundary, ledger_id="correction"))


def test_ledger_by_boundary_filters_without_mutating_entries(now, boundary):
    ledger = AuditLedger()
    entry = _entry(now, boundary, ledger_id="entry")
    ledger.append(entry)

    filtered = ledger.by_boundary(boundary.boundary_id)

    assert filtered == [entry]
    assert ledger.entries == [entry]


def test_audit_flags_residual_mismatch_and_residual_high_separately(now, boundary):
    mismatch = _entry(now, boundary, energy_residual_j=123.0)
    mismatch_flags = audit_ledger_entry(mismatch)
    assert "energy_residual_mismatch" in mismatch_flags
    assert "energy_balance_residual_high" in mismatch_flags

    high_residual = _entry(
        now,
        boundary,
        energy_in_j=101.0,
        energy_residual_j=energy_balance_residual(101.0, 60.0, 10.0, 30.0),
    )
    high_flags = audit_ledger_entry(high_residual)
    assert "energy_residual_mismatch" not in high_flags
    assert "energy_balance_residual_high" in high_flags

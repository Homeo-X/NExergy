from __future__ import annotations

import pytest

from eie.exergy.kernel import energy_balance_residual, exergy_balance_residual
from eie.ledger.audit import audit_ledger_entry
from eie.ledger.entries import AuditLedger, LedgerEntry


def make_entry(now, boundary, **overrides):
    values = {
        "ledger_id": "entry-1",
        "timestamp": now,
        "boundary_id": boundary.boundary_id,
        "reference_state_id": boundary.reference_state_id,
        "energy_in_j": 100.0,
        "energy_out_j": 60.0,
        "energy_stored_delta_j": 10.0,
        "energy_rejected_j": 20.0,
        "energy_residual_j": energy_balance_residual(100.0, 60.0, 10.0, 20.0),
        "exergy_in_j": 80.0,
        "useful_exergy_j": 40.0,
        "stored_exergy_delta_j": 10.0,
        "recovered_exergy_j": 5.0,
        "rejected_exergy_j": 10.0,
        "destroyed_exergy_j": 15.0,
        "exergy_residual_j": exergy_balance_residual(80.0, 40.0, 10.0, 5.0, 10.0, 15.0),
        "entropy_generated_j_per_k": 0.05,
        "flags": ["test"],
        "confidence": 0.9,
    }
    values.update(overrides)
    return LedgerEntry(**values)


def test_energy_balance_residual_computed_correctly():
    assert energy_balance_residual(100.0, 60.0, 10.0, 20.0, known_losses_j=5.0) == pytest.approx(5.0)


def test_exergy_balance_residual_computed_correctly():
    assert exergy_balance_residual(80.0, 40.0, 10.0, 5.0, 10.0, 15.0) == pytest.approx(0.0)


def test_ledger_flags_are_stored(now, boundary):
    entry = make_entry(now, boundary)
    assert entry.flags == ["test"]


def test_audit_ledger_entry_detects_negative_destroyed_exergy(now, boundary):
    entry = make_entry(now, boundary, destroyed_exergy_j=-1.0)
    flags = audit_ledger_entry(entry)
    assert "negative_destroyed_exergy" in flags


def test_audit_ledger_is_append_only_and_rejects_duplicate_ids(now, boundary):
    ledger = AuditLedger()
    entry = make_entry(now, boundary)
    ledger.append(entry)
    with pytest.raises(Exception):
        ledger.append(entry)

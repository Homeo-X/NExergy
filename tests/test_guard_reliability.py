from __future__ import annotations

from datetime import timedelta

from eie.boundary.boundary import Boundary
from eie.core.enums import BoundaryType, Carrier, Severity
from eie.exergy.kernel import energy_balance_residual, exergy_balance_residual
from eie.guards.boundary_guard import BoundaryGuard
from eie.guards.false_exergy_gain import FalseExergyGainGuard, detect_false_exergy_gain
from eie.guards.physics_guard import PhysicsGuard
from eie.guards.reference_guard import ReferenceGuard
from eie.ledger.entries import LedgerEntry
from eie.reference.state import ReferenceState


def _ledger_entry(now, boundary, **overrides):
    values = {
        "ledger_id": "guard-ledger-entry",
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


def _codes(findings):
    return {finding.code for finding in findings}


def test_reference_guard_fails_closed_for_missing_or_low_confidence(now):
    missing = ReferenceGuard().check(None, at=now)
    assert not missing.passed
    assert missing.severity == Severity.CRITICAL

    low_confidence = ReferenceState(
        reference_state_id="low-confidence",
        timestamp=now,
        ambient_temperature_k=300.0,
        ambient_pressure_pa=101_325.0,
        confidence=0.2,
    )
    result = ReferenceGuard(min_confidence=0.5).check(low_confidence, at=now)
    assert not result.passed
    assert result.severity == Severity.ERROR


def test_boundary_guard_fails_closed_for_missing_and_binding_mismatch(boundary):
    missing = BoundaryGuard().check(None)
    assert not missing.passed
    assert missing.severity == Severity.CRITICAL

    boundary_mismatch = BoundaryGuard().check_binding(
        object_boundary_id="other-boundary",
        object_reference_state_id=boundary.reference_state_id,
        boundary=boundary,
    )
    assert not boundary_mismatch.passed

    reference_mismatch = BoundaryGuard().check_binding(
        object_boundary_id=boundary.boundary_id,
        object_reference_state_id="other-reference",
        boundary=boundary,
    )
    assert not reference_mismatch.passed


def test_physics_guard_distinguishes_high_residual_from_residual_mismatch(now, boundary):
    high_residual = _ledger_entry(
        now,
        boundary,
        energy_in_j=101.0,
        energy_residual_j=energy_balance_residual(101.0, 60.0, 10.0, 30.0),
    )
    high_result = PhysicsGuard().check_ledger_entry(high_residual)
    assert not high_result.passed
    assert high_result.severity == Severity.ERROR

    mismatch = _ledger_entry(now, boundary, energy_residual_j=999.0)
    mismatch_result = PhysicsGuard().check_ledger_entry(mismatch)
    assert not mismatch_result.passed
    assert mismatch_result.severity == Severity.CRITICAL


def test_physics_guard_fails_on_negative_entropy_generation(now, boundary):
    entry = _ledger_entry(now, boundary, entropy_generated_j_per_k=-0.1)
    result = PhysicsGuard().check_ledger_entry(entry)
    assert not result.passed
    assert result.severity == Severity.CRITICAL


def test_false_gain_detector_catches_zero_input_stale_reference_and_boundary_mismatch(now, reference):
    stale_reference = ReferenceState(
        reference_state_id="stale-reference",
        timestamp=now - timedelta(hours=2),
        ambient_temperature_k=300.0,
        ambient_pressure_pa=101_325.0,
        confidence=0.9,
    )
    mismatched_boundary = Boundary(
        boundary_id="mismatched-boundary",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=[],
        excluded_entity_ids=[],
        reference_state_id="different-reference",
    )
    findings = detect_false_exergy_gain(
        exergy_in_j=0.0,
        useful_exergy_j=10.0,
        reference_state=stale_reference,
        boundary=mismatched_boundary,
        reference_state_id=stale_reference.reference_state_id,
        at=now,
    )
    assert {
        "exergy_from_zero_input",
        "stale_reference_state",
        "boundary_reference_mismatch",
    }.issubset(_codes(findings))


def test_false_gain_detector_catches_impossible_heat_upgrade(reference, boundary, now):
    findings = detect_false_exergy_gain(
        exergy_in_j=100.0,
        useful_exergy_j=50.0,
        reference_state=reference,
        boundary=boundary,
        at=now,
        carrier=Carrier.THERMAL,
        quality_factor=0.2,
        high_grade_exergy_out_j=500.0,
        low_grade_exergy_in_j=100.0,
        work_input_j=50.0,
    )
    assert "impossible_heat_upgrade_without_work" in _codes(findings)


def test_false_gain_guard_returns_structured_evidence(reference, boundary, now):
    result = FalseExergyGainGuard().check(
        exergy_in_j=100.0,
        useful_exergy_j=150.0,
        destroyed_exergy_j=-1.0,
        entropy_generated_j_per_k=-0.1,
        reference_state=reference,
        boundary=boundary,
        at=now,
    )
    assert not result.passed
    assert result.evidence["findings"]
    finding_codes = {finding["code"] for finding in result.evidence["findings"]}
    assert {"exergy_efficiency_above_one", "negative_destroyed_exergy", "negative_entropy_generation"}.issubset(
        finding_codes
    )

from __future__ import annotations

from eie.core.enums import Carrier
from eie.guards.false_exergy_gain import FalseExergyGainGuard, detect_false_exergy_gain


def codes(findings):
    return {finding.code for finding in findings}


def test_detector_catches_efficiency_above_one(reference, boundary, now):
    findings = detect_false_exergy_gain(
        exergy_in_j=100.0,
        useful_exergy_j=120.0,
        reference_state=reference,
        boundary=boundary,
        at=now,
    )
    assert "exergy_efficiency_above_one" in codes(findings)


def test_detector_catches_missing_reference_and_boundary():
    findings = detect_false_exergy_gain(exergy_in_j=100.0, useful_exergy_j=50.0)
    assert "missing_reference_state" in codes(findings)
    assert "missing_boundary" in codes(findings)


def test_detector_catches_negative_destroyed_exergy(reference, boundary, now):
    findings = detect_false_exergy_gain(
        exergy_in_j=100.0,
        useful_exergy_j=50.0,
        destroyed_exergy_j=-1.0,
        reference_state=reference,
        boundary=boundary,
        at=now,
    )
    assert "negative_destroyed_exergy" in codes(findings)


def test_detector_catches_invalid_carrier_quality(reference, boundary, now):
    findings = detect_false_exergy_gain(
        exergy_in_j=100.0,
        useful_exergy_j=50.0,
        reference_state=reference,
        boundary=boundary,
        at=now,
        carrier=Carrier.ELECTRIC,
        quality_factor=1.2,
    )
    assert "invalid_quality_factor_for_carrier" in codes(findings)


def test_guard_fails_closed_on_false_gain(reference, boundary, now):
    result = FalseExergyGainGuard().check(
        exergy_in_j=100.0,
        useful_exergy_j=150.0,
        reference_state=reference,
        boundary=boundary,
        at=now,
    )
    assert not result.passed

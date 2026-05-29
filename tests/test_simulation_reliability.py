from __future__ import annotations

import pytest

from eie.exergy.kernel import energy_balance_residual, exergy_balance_residual
from eie.ledger.audit import audit_ledger_entry
from eie.simulation.simple_site import run_simple_site


def test_simulation_ledger_entries_are_boundary_and_reference_bound():
    report = run_simple_site()
    for entry in report.ledger.entries:
        assert entry.boundary_id == report.boundary.boundary_id
        assert entry.reference_state_id == report.reference_state.reference_state_id


def test_simulation_ledger_entries_recompute_residuals_exactly():
    report = run_simple_site()
    for entry in report.ledger.entries:
        assert entry.energy_residual_j == pytest.approx(
            energy_balance_residual(
                entry.energy_in_j,
                entry.energy_out_j,
                entry.energy_stored_delta_j,
                entry.energy_rejected_j,
            )
        )
        assert entry.exergy_residual_j == pytest.approx(
            exergy_balance_residual(
                entry.exergy_in_j,
                entry.useful_exergy_j,
                entry.stored_exergy_delta_j,
                entry.recovered_exergy_j,
                entry.rejected_exergy_j,
                entry.destroyed_exergy_j,
            )
        )


def test_simulation_normal_ledger_has_no_blocking_audit_flags():
    report = run_simple_site()
    blocking_flags = {
        "energy_residual_mismatch",
        "energy_balance_residual_high",
        "exergy_residual_mismatch",
        "exergy_balance_residual_high",
        "negative_destroyed_exergy",
        "negative_entropy_generation",
    }
    for entry in report.ledger.entries:
        assert not blocking_flags.intersection(audit_ledger_entry(entry))


def test_heat_pump_entry_accounts_for_destroyed_exergy_and_entropy_consistently():
    report = run_simple_site()
    hp_entry = next(entry for entry in report.ledger.entries if entry.ledger_id == "ledger-heat-pump-hour-1")

    assert hp_entry.destroyed_exergy_j == pytest.approx(hp_entry.exergy_in_j - hp_entry.useful_exergy_j)
    assert hp_entry.entropy_generated_j_per_k == pytest.approx(
        hp_entry.destroyed_exergy_j / report.reference_state.ambient_temperature_k
    )
    assert hp_entry.destroyed_exergy_j > 0


def test_simulation_impossible_case_reports_expected_pathology_codes():
    report = run_simple_site()
    finding_codes = set()
    for result in report.impossible_case_guard_results:
        for finding in result.evidence.get("findings", []):
            finding_codes.add(finding["code"])

    assert {
        "exergy_efficiency_above_one",
        "negative_destroyed_exergy",
        "negative_entropy_generation",
        "impossible_heat_upgrade_without_work",
    }.issubset(finding_codes)


def test_simple_report_text_is_stable_and_actionable():
    text = run_simple_site().to_text()

    assert "normal_guard_status: PASS" in text
    assert "impossible_case_detected: PASS" in text
    assert "not be wasted on low-grade heat" in text

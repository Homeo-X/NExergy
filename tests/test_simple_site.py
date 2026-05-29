from __future__ import annotations

from eie.simulation.simple_site import run_simple_site


def test_simple_site_runs_and_produces_ledger():
    report = run_simple_site()
    assert len(report.ledger.entries) == 3
    assert report.pv_exergy_j > 0
    assert report.thermal_storage_exergy_j > 0


def test_simple_site_guard_stack_passes_normal_case():
    report = run_simple_site()
    assert all(result.passed for result in report.guard_results)


def test_simple_site_guard_stack_fails_intentionally_impossible_case():
    report = run_simple_site()
    assert any(not result.passed for result in report.impossible_case_guard_results)


def test_simple_site_demonstrates_quality_preservation():
    report = run_simple_site()
    assert report.heat_pump_useful_thermal_exergy_j > report.resistance_heat_exergy_for_same_input_j

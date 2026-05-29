"""Tests for ShadowOptimizer, DispatchSpace, SiteSnapshot, and guard non-bypass."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eie.boundary.boundary import Boundary
from eie.core.enums import BoundaryType, MeasurementMethod
from eie.core.errors import BoundaryError, DomainError, MissingReferenceError
from eie.flows.base import Metadata
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.optimization.decision import (
    DispatchDecision,
    DispatchSpace,
    DispatchVariable,
    FeasibilityNote,
)
from eie.optimization.evaluator import evaluate_objectives, simulate_dispatch
from eie.optimization.objective import STANDARD_OBJECTIVES, ObjectiveValue
from eie.optimization.optimizer import SIMPLE_SITE_DISPATCH_SPACE, OptimizationResult, ShadowOptimizer
from eie.optimization.snapshot import (
    BatteryConstraints,
    GridConstraints,
    SiteSnapshot,
    ThermalStorageConstraints,
)
from eie.reference.state import ReferenceState


@pytest.fixture
def opt_now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def opt_reference(opt_now: datetime) -> ReferenceState:
    return ReferenceState(
        reference_state_id="ref-opt-test",
        timestamp=opt_now,
        ambient_temperature_k=298.15,
        ambient_pressure_pa=101_325.0,
        confidence=0.99,
        marginal_carbon_kg_per_kwh=0.42,
        marginal_price_per_kwh=0.18,
        valid_until=opt_now + timedelta(minutes=60),
    )


@pytest.fixture
def opt_boundary(opt_reference: ReferenceState, opt_now: datetime) -> Boundary:
    return Boundary(
        boundary_id="opt-boundary-test",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=["pv", "battery", "heat-pump", "thermal-store", "building"],
        excluded_entity_ids=[],
        reference_state_id=opt_reference.reference_state_id,
        accounting_period_start=opt_now,
        accounting_period_end=opt_now + timedelta(hours=1),
    )


@pytest.fixture
def opt_energy_metadata(opt_reference: ReferenceState, opt_boundary: Boundary, opt_now: datetime) -> Metadata:
    return Metadata(
        timestamp=opt_now,
        source="pytest-optimizer",
        method=MeasurementMethod.SIMULATED,
        unit="J",
        confidence=1.0,
        uncertainty=0.0,
        boundary_id=opt_boundary.boundary_id,
        reference_state_id=opt_reference.reference_state_id,
    )


@pytest.fixture
def site_snapshot(opt_reference, opt_boundary, opt_energy_metadata, opt_now) -> SiteSnapshot:
    battery_state = BatteryState(
        storage_id="opt-battery",
        stored_energy_j=3_600_000.0 * 5,  # 5 kWh
        soc=0.60,
        soh=0.95,
        reserve_energy_j=3_600_000.0 * 0.5,  # 0.5 kWh reserve
        boundary_id=opt_boundary.boundary_id,
        reference_state_id=opt_reference.reference_state_id,
        metadata=opt_energy_metadata,
    )
    thermal_storage_state = ThermalStorageState(
        storage_id="opt-thermal-store",
        layers=[
            ThermalLayer(temperature_k=333.15, energy_j=2_000_000.0),
            ThermalLayer(temperature_k=318.15, energy_j=3_000_000.0),
        ],
        boundary_id=opt_boundary.boundary_id,
        reference_state_id=opt_reference.reference_state_id,
        metadata=opt_energy_metadata,
    )
    return SiteSnapshot(
        snapshot_id="snapshot-test-001",
        timestamp=opt_now,
        reference_state=opt_reference,
        boundary=opt_boundary,
        horizon_duration_s=3600.0,
        available_pv_power_w=5_000.0,
        battery_state=battery_state,
        battery_constraints=BatteryConstraints(
            max_charge_power_w=3_000.0,
            max_discharge_power_w=3_000.0,
            min_soc=0.10,
            max_soc=0.95,
            round_trip_efficiency=0.92,
            capacity_j=3_600_000.0 * 10.0,  # 10 kWh
        ),
        thermal_storage_state=thermal_storage_state,
        thermal_storage_constraints=ThermalStorageConstraints(
            max_charge_power_w=2_000.0,
            max_discharge_power_w=2_000.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            standby_loss_w=50.0,
        ),
        building_electric_load_w=1_200.0,
        heat_pump_rated_cop=3.5,
        heat_pump_max_power_w=2_000.0,
        delivered_heat_temperature_k=318.15,
        heat_demand_w=3_000.0,
        grid_constraints=GridConstraints(
            import_limit_w=10_000.0,
            export_limit_w=5_000.0,
            import_tariff_per_kwh=0.28,
            export_tariff_per_kwh=0.05,
            carbon_intensity_kg_per_kwh=0.42,
        ),
    )


# ── SiteSnapshot validation ──────────────────────────────────────────────────

def test_site_snapshot_rejects_mismatched_boundary_reference(opt_reference, opt_boundary, opt_energy_metadata, opt_now):
    bad_boundary = Boundary(
        boundary_id="other-boundary",
        boundary_type=BoundaryType.SITE,
        included_entity_ids=[],
        excluded_entity_ids=[],
        reference_state_id="DIFFERENT-REF",
        accounting_period_start=opt_now,
        accounting_period_end=opt_now + timedelta(hours=1),
    )
    battery_state = BatteryState(
        storage_id="bat",
        stored_energy_j=1000.0,
        soc=0.5,
        soh=0.9,
        reserve_energy_j=100.0,
        boundary_id=opt_boundary.boundary_id,
        reference_state_id=opt_reference.reference_state_id,
        metadata=opt_energy_metadata,
    )
    ts = ThermalStorageState(
        storage_id="ts",
        layers=[ThermalLayer(temperature_k=330.0, energy_j=1000.0)],
        boundary_id=opt_boundary.boundary_id,
        reference_state_id=opt_reference.reference_state_id,
        metadata=opt_energy_metadata,
    )
    with pytest.raises(BoundaryError):
        SiteSnapshot(
            snapshot_id="bad",
            timestamp=opt_now,
            reference_state=opt_reference,
            boundary=bad_boundary,  # reference mismatch
            horizon_duration_s=3600.0,
            available_pv_power_w=1000.0,
            battery_state=battery_state,
            battery_constraints=BatteryConstraints(
                max_charge_power_w=1000.0, max_discharge_power_w=1000.0,
                min_soc=0.1, max_soc=0.9, round_trip_efficiency=0.9, capacity_j=10000.0,
            ),
            thermal_storage_state=ts,
            thermal_storage_constraints=ThermalStorageConstraints(
                max_charge_power_w=1000.0, max_discharge_power_w=1000.0,
                charge_efficiency=0.9, discharge_efficiency=0.9, standby_loss_w=0.0,
            ),
            building_electric_load_w=500.0,
            heat_pump_rated_cop=3.0,
            heat_pump_max_power_w=1000.0,
            delivered_heat_temperature_k=318.0,
            heat_demand_w=1500.0,
            grid_constraints=GridConstraints(
                import_limit_w=5000.0, export_limit_w=2000.0,
                import_tariff_per_kwh=0.25, export_tariff_per_kwh=0.05,
                carbon_intensity_kg_per_kwh=0.4,
            ),
        )


# ── simulate_dispatch ────────────────────────────────────────────────────────

def test_simulate_dispatch_returns_outcome(site_snapshot, opt_now):
    point = {"heat_pump_fraction": 0.5, "battery_charge_fraction": 0.3, "thermal_storage_fraction": 0.5}
    outcome = simulate_dispatch(site_snapshot, point, timestamp=opt_now, decision_id="test-d1")
    assert outcome.pv_energy_j > 0
    assert outcome.heat_pump_energy_in_j >= 0
    assert outcome.destroyed_exergy_j >= 0


def test_simulate_dispatch_pv_energy_equals_power_times_duration(site_snapshot, opt_now):
    point = {"heat_pump_fraction": 0.0, "battery_charge_fraction": 0.0, "thermal_storage_fraction": 0.0}
    outcome = simulate_dispatch(site_snapshot, point, timestamp=opt_now, decision_id="test-d2")
    expected_pv_j = site_snapshot.available_pv_power_w * site_snapshot.horizon_duration_s
    assert outcome.pv_energy_j == pytest.approx(expected_pv_j, rel=1e-9)


def test_simulate_dispatch_guard_results_are_present(site_snapshot, opt_now):
    point = {"heat_pump_fraction": 0.5, "battery_charge_fraction": 0.3, "thermal_storage_fraction": 0.0}
    outcome = simulate_dispatch(site_snapshot, point, timestamp=opt_now, decision_id="test-d3")
    assert len(outcome.guard_results) >= 4  # Reference, Boundary, Physics, FalseGain


def test_simulate_dispatch_guards_pass_for_valid_snapshot(site_snapshot, opt_now):
    point = {"heat_pump_fraction": 0.3, "battery_charge_fraction": 0.2, "thermal_storage_fraction": 0.1}
    outcome = simulate_dispatch(site_snapshot, point, timestamp=opt_now, decision_id="test-d4")
    assert all(r.passed for r in outcome.guard_results), (
        [r.reason for r in outcome.guard_results if not r.passed]
    )


# ── evaluate_objectives ──────────────────────────────────────────────────────

def test_evaluate_objectives_returns_score_when_guards_pass(site_snapshot, opt_now):
    point = {"heat_pump_fraction": 0.3, "battery_charge_fraction": 0.2, "thermal_storage_fraction": 0.1}
    outcome = simulate_dispatch(site_snapshot, point, timestamp=opt_now, decision_id="test-e1")
    score = evaluate_objectives(outcome, STANDARD_OBJECTIVES)
    assert score is not None
    for obj in STANDARD_OBJECTIVES:
        assert obj.name in score.values


# ── DispatchDecision validation ──────────────────────────────────────────────

def test_dispatch_decision_is_advisory_flag_must_be_true(opt_now, opt_boundary, opt_reference):
    with pytest.raises(DomainError, match="is_advisory must be True"):
        DispatchDecision(
            decision_id="bad",
            timestamp=opt_now,
            horizon_start=opt_now,
            horizon_end=opt_now + timedelta(hours=1),
            boundary_id=opt_boundary.boundary_id,
            reference_state_id=opt_reference.reference_state_id,
            variable_values={},
            is_advisory=False,  # must be True in v0
            guard_verified=True,
            guard_results=[],
        )


def test_dispatch_decision_horizon_end_before_start_raises(opt_now, opt_boundary, opt_reference):
    with pytest.raises(DomainError, match="at or after"):
        DispatchDecision(
            decision_id="bad-horizon",
            timestamp=opt_now,
            horizon_start=opt_now + timedelta(hours=1),
            horizon_end=opt_now,   # before start
            boundary_id=opt_boundary.boundary_id,
            reference_state_id=opt_reference.reference_state_id,
            variable_values={},
            is_advisory=True,
            guard_verified=True,
            guard_results=[],
        )


# ── ShadowOptimizer ──────────────────────────────────────────────────────────

def test_shadow_optimizer_returns_result(site_snapshot, opt_now):
    optimizer = ShadowOptimizer(grid_points_per_variable=5, refine_steps=0)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    assert isinstance(result, OptimizationResult)
    assert result.guard_bypass_count == 0  # INVARIANT: never bypass guards


def test_shadow_optimizer_guard_bypass_count_always_zero(site_snapshot, opt_now):
    optimizer = ShadowOptimizer(grid_points_per_variable=5, refine_steps=0)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    assert result.guard_bypass_count == 0


def test_shadow_optimizer_all_feasible_decisions_are_guard_verified(site_snapshot, opt_now):
    optimizer = ShadowOptimizer(grid_points_per_variable=5, refine_steps=0)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    for decision in result.feasible_decisions:
        assert decision.guard_verified
        assert decision.is_advisory


def test_shadow_optimizer_recommended_decision_is_advisory(site_snapshot, opt_now):
    optimizer = ShadowOptimizer(grid_points_per_variable=5, refine_steps=0)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    if result.recommended_decision is not None:
        assert result.recommended_decision.is_advisory


def test_shadow_optimizer_produces_pareto_front(site_snapshot, opt_now):
    optimizer = ShadowOptimizer(grid_points_per_variable=5, refine_steps=0)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    assert len(result.pareto_front) >= 1
    # All Pareto decisions must be feasible
    feasible_ids = {d.decision_id for d in result.feasible_decisions}
    for pd in result.pareto_front:
        assert pd.decision_id in feasible_ids or "refined" in pd.decision_id


def test_shadow_optimizer_evaluated_count_matches_grid(site_snapshot, opt_now):
    n = 5
    optimizer = ShadowOptimizer(grid_points_per_variable=n, refine_steps=0)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    expected = n ** len(SIMPLE_SITE_DISPATCH_SPACE.variables)
    assert result.evaluated_count == expected


def test_shadow_optimizer_with_refinement_runs(site_snapshot, opt_now):
    optimizer = ShadowOptimizer(grid_points_per_variable=5, refine_steps=3)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    assert result.guard_bypass_count == 0
    assert result.recommended_decision is not None


def test_shadow_optimizer_notes_include_advisory_warning(site_snapshot, opt_now):
    optimizer = ShadowOptimizer(grid_points_per_variable=5, refine_steps=0)
    result = optimizer.optimise(site_snapshot, at=opt_now)
    advisory_notes = [n for n in result.notes if "advisory" in n.lower()]
    assert len(advisory_notes) >= 1


# ── DispatchSpace ────────────────────────────────────────────────────────────

def test_dispatch_space_requires_at_least_one_variable():
    with pytest.raises(DomainError):
        DispatchSpace(variables=[])


def test_dispatch_space_rejects_duplicate_names():
    var = DispatchVariable(name="x", lower_bound=0.0, upper_bound=1.0, unit="1", description="x")
    with pytest.raises(DomainError):
        DispatchSpace(variables=[var, var])


def test_dispatch_space_grid_points_correct_count():
    space = SIMPLE_SITE_DISPATCH_SPACE
    points = space.grid_points(5)
    # 5^3 = 125
    assert len(points) == 125


def test_dispatch_space_grid_all_points_are_feasible():
    space = SIMPLE_SITE_DISPATCH_SPACE
    points = space.grid_points(5)
    for point in points:
        assert space.is_feasible(point)


def test_dispatch_variable_discrete_points_include_bounds():
    var = DispatchVariable(name="x", lower_bound=0.0, upper_bound=1.0, unit="1", description="x")
    pts = var.discrete_points(5)
    assert pts[0] == pytest.approx(0.0)
    assert pts[-1] == pytest.approx(1.0)
    assert len(pts) == 5


def test_feasibility_note_rejects_invalid_severity():
    with pytest.raises(DomainError):
        FeasibilityNote(code="x", message="y", severity="catastrophic")


# ── DispatchVariable validation ───────────────────────────────────────────────

def test_dispatch_variable_rejects_empty_name():
    with pytest.raises(DomainError):
        DispatchVariable(name="", lower_bound=0.0, upper_bound=1.0, unit="1", description="x")


def test_dispatch_variable_rejects_infinite_bounds():
    with pytest.raises(DomainError):
        DispatchVariable(name="x", lower_bound=float("inf"), upper_bound=1.0, unit="1", description="x")


def test_dispatch_variable_rejects_lower_gt_upper():
    with pytest.raises(DomainError):
        DispatchVariable(name="x", lower_bound=1.0, upper_bound=0.0, unit="1", description="x")


def test_dispatch_variable_rejects_empty_unit():
    with pytest.raises(DomainError):
        DispatchVariable(name="x", lower_bound=0.0, upper_bound=1.0, unit="", description="x")


def test_dispatch_variable_discrete_points_single():
    var = DispatchVariable(name="x", lower_bound=0.5, upper_bound=0.5, unit="1", description="x")
    pts = var.discrete_points(1)
    assert len(pts) == 1
    assert pts[0] == pytest.approx(0.5)


# ── DispatchDecision properties ───────────────────────────────────────────────

def test_dispatch_decision_properties(opt_now, opt_boundary, opt_reference):
    from eie.guards.physics_guard import GuardResult
    from eie.core.enums import Severity
    note = FeasibilityNote(code="test", message="test note", severity="infeasible")
    gr = GuardResult("TestGuard", False, Severity.ERROR, "test failure")
    dd = DispatchDecision(
        decision_id="test-props",
        timestamp=opt_now,
        horizon_start=opt_now,
        horizon_end=opt_now,
        boundary_id=opt_boundary.boundary_id,
        reference_state_id=opt_reference.reference_state_id,
        variable_values={"heat_pump_fraction": 0.5},
        is_advisory=True,
        guard_verified=False,
        guard_results=[gr],
        feasibility_notes=[note],
    )
    assert dd.is_infeasible
    assert dd.has_guard_failure
    assert not dd.is_usable
    assert dd.variable("heat_pump_fraction") == pytest.approx(0.5)
    with pytest.raises(DomainError):
        dd.variable("nonexistent")


# ── DispatchSpace variable lookup ─────────────────────────────────────────────

def test_dispatch_space_variable_lookup():
    space = SIMPLE_SITE_DISPATCH_SPACE
    var = space.variable("heat_pump_fraction")
    assert var.name == "heat_pump_fraction"
    with pytest.raises(DomainError):
        space.variable("nonexistent_var")


def test_dispatch_space_is_feasible_and_clamp():
    space = SIMPLE_SITE_DISPATCH_SPACE
    good = {"heat_pump_fraction": 0.5, "battery_charge_fraction": 0.5, "thermal_storage_fraction": 0.5}
    bad = {"heat_pump_fraction": 2.0, "battery_charge_fraction": 0.5, "thermal_storage_fraction": 0.5}
    assert space.is_feasible(good)
    assert not space.is_feasible(bad)
    clamped = space.clamp(bad)
    assert clamped["heat_pump_fraction"] == pytest.approx(1.0)


def test_dispatch_space_grid_points_rejects_zero():
    with pytest.raises(DomainError):
        SIMPLE_SITE_DISPATCH_SPACE.grid_points(0)


# ── ShadowOptimizer constructor validation ───────────────────────────────────

def test_shadow_optimizer_rejects_low_grid_points():
    with pytest.raises(DomainError):
        ShadowOptimizer(grid_points_per_variable=1)


def test_shadow_optimizer_rejects_negative_refine_steps():
    with pytest.raises(DomainError):
        ShadowOptimizer(refine_steps=-1)


def test_optimization_result_rejects_nonzero_bypass():
    from eie.optimization.optimizer import OptimizationResult
    from datetime import timedelta
    with pytest.raises(DomainError):
        OptimizationResult(
            result_id="bad",
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            snapshot_id="snap",
            feasible_decisions=[],
            pareto_front=[],
            recommended_decision=None,
            objective_scores={},
            evaluated_count=0,
            infeasible_count=0,
            guard_bypass_count=1,  # must be 0
            objectives=[],
            notes=[],
        )
